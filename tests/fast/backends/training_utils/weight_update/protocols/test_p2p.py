from types import ModuleType
from typing import Any


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
