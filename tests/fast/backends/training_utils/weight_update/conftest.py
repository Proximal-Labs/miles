import importlib
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import ModuleType

import pytest

from miles.backends.training_utils.weight_update.protocol import UpdatableEngine

_P2P_TRANSFER_UTILS_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils"


@contextmanager
def stubbed_missing_external_sdks(module_attributes: dict[str, dict[str, object]]) -> Iterator[None]:
    created_modules: list[str] = []
    created_attributes: list[tuple[ModuleType, str]] = []

    for module_name, attributes in module_attributes.items():
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            if name in sys.modules:
                continue
            try:
                importlib.import_module(name)
                continue
            except ImportError:
                pass
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            created_modules.append(name)
            if depth > 1:
                parent = sys.modules[".".join(parts[: depth - 1])]
                setattr(parent, parts[depth - 1], module)
                created_attributes.append((parent, parts[depth - 1]))
        for attribute, value in attributes.items():
            module = sys.modules[module_name]
            if not hasattr(module, attribute):
                setattr(module, attribute, value)
                created_attributes.append((module, attribute))

    try:
        yield
    finally:
        for parent, attribute in reversed(created_attributes):
            delattr(parent, attribute)
        for name in reversed(created_modules):
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def p2p_transfer_utils() -> ModuleType:
    with stubbed_missing_external_sdks(
        {
            "mooncake.engine": {"TransferEngine": object},
            "sglang.srt.server_args": {"ServerArgs": object},
        }
    ):
        return importlib.import_module(_P2P_TRANSFER_UTILS_MODULE)


def make_updatable_engines(
    api_clients: Sequence[object],
    *,
    gpu_counts: Sequence[int] | None = None,
    gpu_offsets: Sequence[int] | None = None,
) -> list[UpdatableEngine]:
    counts = gpu_counts if gpu_counts is not None else [1] * len(api_clients)
    offsets = gpu_offsets if gpu_offsets is not None else list(range(len(api_clients)))
    return [
        UpdatableEngine(
            cell_id=f"cell-{index}",
            api_client=api_client,
            gpu_count=count,
            gpu_offset=offset,
            workers_hash=f"hash-{index}",
        )
        for index, (api_client, count, offset) in enumerate(zip(api_clients, counts, offsets, strict=True))
    ]
