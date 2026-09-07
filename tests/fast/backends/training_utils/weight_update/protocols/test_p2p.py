from types import ModuleType, SimpleNamespace
from typing import Any


def _server_args(rl_quant_profile: str | None = None) -> Any:
    return SimpleNamespace(rl_quant_profile=rl_quant_profile)


class _FakeReplica:
    def __init__(self, name: str) -> None:
        self.name = name
        self.loaded: list[list[tuple[str, Any]]] = []

    def named_parameters(self) -> list[tuple[str, Any]]:
        return [(f"{self.name}.weight", object())]

    def load_weights(self, tensors: list[tuple[str, Any]]) -> None:
        self.loaded.append(list(tensors))


def _manager(p2p_protocol: ModuleType, replicas: list[_FakeReplica], monkeypatch) -> Any:
    created: list[bool] = []

    def _create(parallelism_config, model_path, server_args, shared_params_dict, first_rollout_engine_rank=False):
        created.append(first_rollout_engine_rank)
        return replicas[len(created) - 1]

    monkeypatch.setattr(p2p_protocol, "_create_cpu_replica", _create)
    monkeypatch.setattr(
        p2p_protocol, "ParameterMapper", type("_Mapper", (), {"from_model": staticmethod(lambda m: m)})
    )
    manager = p2p_protocol._CPUReplicasManager(model_path="/model")
    return manager, created


class TestCPUReplicasManager:
    """Bookkeeping of the CPU replicas the p2p sender stages its weights through."""

    def test_a_fresh_manager_holds_no_replica_and_no_shared_state(self, p2p_protocol: ModuleType) -> None:
        """The shared buffers only exist once a replica has published them, so they must start empty."""
        manager = p2p_protocol._CPUReplicasManager(model_path="/model")

        assert manager.replicas == []
        assert manager.shared_params_dict == {}
        assert manager.shared_param_mapper is None

    def test_the_first_replica_publishes_the_shared_buffers_and_the_mapper(
        self, p2p_protocol: ModuleType, monkeypatch
    ) -> None:
        """Every later rank writes into these buffers, so the first replica has to hand them over."""
        replica = _FakeReplica("first")
        manager, created = _manager(p2p_protocol, [replica], monkeypatch)

        manager.create_replica(parallelism_config=object(), server_args=object())

        assert created == [True]
        assert list(manager.shared_params_dict) == ["first.weight"]
        assert manager.shared_param_mapper is replica

    def test_only_the_first_replica_allocates_its_own_buffers(self, p2p_protocol: ModuleType, monkeypatch) -> None:
        """A second allocation would give one rank buffers nobody else writes into."""
        first, second = _FakeReplica("first"), _FakeReplica("second")
        manager, created = _manager(p2p_protocol, [first, second], monkeypatch)

        manager.create_replica(parallelism_config=object(), server_args=object())
        manager.create_replica(parallelism_config=object(), server_args=object())

        assert created == [True, False]
        assert list(manager.shared_params_dict) == ["first.weight"]
        assert manager.shared_param_mapper is first

    def test_every_created_replica_is_kept(self, p2p_protocol: ModuleType, monkeypatch) -> None:
        """A replica the manager forgets is a shard layout nothing can stage into any more."""
        first, second = _FakeReplica("first"), _FakeReplica("second")
        manager, _created = _manager(p2p_protocol, [first, second], monkeypatch)

        manager.create_replica(parallelism_config=object(), server_args=object())
        manager.create_replica(parallelism_config=object(), server_args=object())

        assert manager.replicas == [first, second]


class TestShardLayoutKey:
    """What makes two rollout engine ranks able to share one CPU replica."""

    def test_ranks_differing_only_in_placement_share_one_shard_layout(self, p2p_protocol: ModuleType) -> None:
        """The process a rank runs in says nothing about which slice of the weights it holds."""
        first = p2p_protocol._shard_layout_key({"tp_rank": 0, "global_rank": 3, "local_rank": 3}, _server_args())
        second = p2p_protocol._shard_layout_key({"tp_rank": 0, "global_rank": 9, "local_rank": 1}, _server_args())

        assert first == second

    def test_a_different_shard_index_is_a_different_shard_layout(self, p2p_protocol: ModuleType) -> None:
        """Two tp ranks hold different slices, so one replica cannot serve both."""
        first = p2p_protocol._shard_layout_key({"tp_rank": 0, "global_rank": 0}, _server_args())
        second = p2p_protocol._shard_layout_key({"tp_rank": 1, "global_rank": 0}, _server_args())

        assert first != second

    def test_a_different_quantization_profile_is_a_different_shard_layout(self, p2p_protocol: ModuleType) -> None:
        """The quantization profile decides the dtype of every buffer the replica allocates."""
        first = p2p_protocol._shard_layout_key({"tp_rank": 0}, _server_args(rl_quant_profile=None))
        second = p2p_protocol._shard_layout_key({"tp_rank": 0}, _server_args(rl_quant_profile="fp8"))

        assert first != second

    def test_the_key_does_not_depend_on_the_order_of_the_parallelism_fields(self, p2p_protocol: ModuleType) -> None:
        """The remote answers a mapping, whose iteration order must not split one layout into two."""
        first = p2p_protocol._shard_layout_key({"tp_rank": 1, "ep_rank": 2}, _server_args())
        second = p2p_protocol._shard_layout_key({"ep_rank": 2, "tp_rank": 1}, _server_args())

        assert first == second


class TestGetOrCreateReplica:
    """One replica per shard layout, reused across reconnects."""

    def test_one_shard_layout_is_built_only_once(self, p2p_protocol: ModuleType, monkeypatch) -> None:
        """Rebuilding a replica the sender already holds wastes host memory and re-registers its buffers."""
        first, second = _FakeReplica("first"), _FakeReplica("second")
        manager, created = _manager(p2p_protocol, [first, second], monkeypatch)
        monkeypatch.setattr(
            p2p_protocol, "RankParallelismConfig", type("_Cfg", (), {"from_dict": staticmethod(lambda d: d)})
        )

        replica = manager.get_or_create_replica(parallelism_info={"tp_rank": 0}, server_args=_server_args())
        again = manager.get_or_create_replica(parallelism_info={"tp_rank": 0}, server_args=_server_args())

        assert (replica, again) == (first, first)
        assert created == [True]

    def test_another_shard_layout_gets_its_own_replica(self, p2p_protocol: ModuleType, monkeypatch) -> None:
        """Two ranks sharded differently cannot read the same replica without sending each other's shard."""
        first, second = _FakeReplica("first"), _FakeReplica("second")
        manager, created = _manager(p2p_protocol, [first, second], monkeypatch)
        monkeypatch.setattr(
            p2p_protocol, "RankParallelismConfig", type("_Cfg", (), {"from_dict": staticmethod(lambda d: d)})
        )

        manager.get_or_create_replica(parallelism_info={"tp_rank": 0}, server_args=_server_args())
        manager.get_or_create_replica(parallelism_info={"tp_rank": 1}, server_args=_server_args())

        assert manager.replicas == [first, second]
        assert created == [True, False]
