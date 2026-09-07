import importlib
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType

import pytest

_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils"


@dataclass
class _FakeServerArgs:
    model_path: str | None = None


_EXTERNAL_SDK_ATTRIBUTES = {
    "mooncake.engine": {"TransferEngine": object},
    "sglang.srt.server_args": {"ServerArgs": _FakeServerArgs},
}


class _FakeRolloutEngine:
    def __init__(self, engine_index: int):
        self._engine_index = engine_index
        self.calls: list[tuple[str, dict]] = []

    async def get_remote_instance_transfer_engine_info(self, rank: int):
        self.calls.append(("get_remote_instance_transfer_engine_info", {"rank": rank}))
        return f"session-{self._engine_index}-{rank}", {f"weight-{rank}": (0x1000 + rank, 4, 2)}

    async def get_parallelism_info(self, rank: int):
        self.calls.append(("get_parallelism_info", {"rank": rank}))
        return {"tp_rank": rank}

    async def get_server_info(self):
        self.calls.append(("get_server_info", {}))
        return {"model_path": f"/model/{self._engine_index}"}


class _JsonRolloutEngine(_FakeRolloutEngine):
    async def get_remote_instance_transfer_engine_info(self, rank: int):
        session_id, weights_info = await super().get_remote_instance_transfer_engine_info(rank)
        return session_id, {name: list(location) for name, location in weights_info.items()}


@contextmanager
def _stubbed_missing_external_sdks():
    missing = object()
    saved_modules: dict[str, object] = {}
    saved_attributes: list[tuple[ModuleType, str, object]] = []
    for module_name, attributes in _EXTERNAL_SDK_ATTRIBUTES.items():
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            saved_modules[name] = sys.modules.get(name, missing)
            if depth == len(parts) or name not in sys.modules:
                module = ModuleType(name)
                if depth < len(parts):
                    module.__path__ = []
                sys.modules[name] = module
            if depth > 1:
                parent = sys.modules[".".join(parts[: depth - 1])]
                attribute = parts[depth - 1]
                saved_attributes.append((parent, attribute, getattr(parent, attribute, missing)))
                setattr(parent, attribute, sys.modules[name])
        for attribute, value in attributes.items():
            setattr(sys.modules[module_name], attribute, value)

    try:
        yield
    finally:
        for parent, attribute, value in reversed(saved_attributes):
            if value is missing:
                delattr(parent, attribute)
            else:
                setattr(parent, attribute, value)
        for name, module in reversed(saved_modules.items()):
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def p2p_transfer_utils():
    package_name, attribute = _MODULE.rsplit(".", 1)
    package = importlib.import_module(package_name)
    missing = object()
    saved_module = sys.modules.get(_MODULE, missing)
    saved_attribute = getattr(package, attribute, missing)

    with _stubbed_missing_external_sdks():
        sys.modules.pop(_MODULE, None)
        if hasattr(package, attribute):
            delattr(package, attribute)
        try:
            yield importlib.import_module(_MODULE)
        finally:
            sys.modules.pop(_MODULE, None)
            if saved_module is not missing:
                sys.modules[_MODULE] = saved_module
            if saved_attribute is missing:
                if hasattr(package, attribute):
                    delattr(package, attribute)
            else:
                setattr(package, attribute, saved_attribute)


def _query(module, engine: _FakeRolloutEngine, engine_ranks: list[int]):
    return module.query_remote_weight_infos(engine, engine_ranks)


class TestQueryRemoteWeightInfos:
    """Remote-info discovery over one rollout engine's HTTP API."""

    def test_repeated_ranks_are_queried_once_each(self, p2p_transfer_utils):
        """The same engine rank appears once per source shard, and re-querying it wastes round trips."""
        engine = _FakeRolloutEngine(0)

        _query(p2p_transfer_utils, engine, [0, 1, 0])

        assert Counter(name for name, _kwargs in engine.calls) == Counter(
            {
                "get_remote_instance_transfer_engine_info": 2,
                "get_parallelism_info": 2,
                "get_server_info": 2,
            }
        )
        assert sorted(kwargs["rank"] for name, kwargs in engine.calls if name == "get_parallelism_info") == [0, 1]

    def test_every_rank_reports_its_own_session_parallelism_and_server_args(self, p2p_transfer_utils):
        """One cell serves several shards, and pairing a rank with another rank's session writes the wrong shard."""
        engine = _FakeRolloutEngine(3)

        targets = _query(p2p_transfer_utils, engine, [0, 1])

        assert {rank: target.session_id for rank, target in targets.items()} == {0: "session-3-0", 1: "session-3-1"}
        assert {rank: target.parallelism_info for rank, target in targets.items()} == {
            0: {"tp_rank": 0},
            1: {"tp_rank": 1},
        }
        assert all(isinstance(target.server_args, p2p_transfer_utils.ServerArgs) for target in targets.values())
        assert {target.server_args.model_path for target in targets.values()} == {"/model/3"}

    def test_weight_locations_are_decoded_from_the_wire_into_named_fields(self, p2p_transfer_utils):
        """The engines answer over HTTP, so JSON lists must become RemoteWeightLocation before any caller indexes them."""
        targets = _query(p2p_transfer_utils, _JsonRolloutEngine(0), [0])

        location = targets[0].weights_info["weight-0"]
        assert isinstance(location, p2p_transfer_utils.RemoteWeightLocation)
        assert (location.address, location.numel, location.element_size) == (0x1000, 4, 2)
