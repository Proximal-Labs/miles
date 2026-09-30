"""Native LoRA checkpoint completion and per-rank random state.

The final marker is written by rank zero after every rank's save barrier. It is
not a serving export and does not depend on the Proximal plugin.
"""

import json
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

MARKER = "native_checkpoint.json"


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        # Megatron is optional on CPU; these are its named model-parallel streams.
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        state["cuda"] = torch.cuda.get_rng_state()
        state["tracker"] = get_cuda_rng_tracker().get_states()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        torch.cuda.set_rng_state(state["cuda"])
        get_cuda_rng_tracker().set_states(state["tracker"])


def layout(args: Namespace) -> dict[str, int]:
    return {
        name: int(getattr(args, name, 1) or 1)
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "virtual_pipeline_model_parallel_size",
        )
    }


def complete(
    path: Path, args: Namespace, *, iteration: int | None, optimizer: bool, scheduler: bool, rng: bool
) -> None:
    value = {
        "schema_version": 1,
        "iteration": iteration,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "layout": layout(args),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "rng": rng,
    }
    temporary = path / f".{MARKER}.tmp"
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path / MARKER)
