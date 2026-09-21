"""Checkpoint directories: written collectively, complete at their final path."""

# TODO: isolate checkpoint IO failures in Tinker; they still terminate the trainer cell.

import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
) -> None:
    """Write collectively, then atomically point ``path`` at the completed version.

    All ranks must call. Writers must finish their collectives before raising local IO errors.
    Readers may still hold an older version, so retain it.
    """
    final_dir = Path(path)
    tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}"
    version_dir = None

    def make_tmp_dir():
        if _rank() == 0:
            if not overwrite and final_dir.exists():
                raise FileExistsError(f"checkpoint {final_dir} already exists")
            if final_dir.exists() and not final_dir.is_symlink():
                raise NotImplementedError(
                    f"cannot overwrite a legacy checkpoint directory {final_dir}; save under a new name"
                )
            # a crashed attempt may leave shards or an unpublished version link
            if tmp_dir.is_symlink():
                tmp_dir.unlink()
            elif tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)

    def publish_dir():
        nonlocal version_dir
        if _rank() != 0:
            return
        if metadata is not None:
            (tmp_dir / "META.json").write_text(json.dumps(metadata, indent=2))
        version_dir = final_dir.parent / f"_version_{final_dir.name}_{uuid4().hex}"
        os.replace(tmp_dir, version_dir)
        tmp_dir.symlink_to(version_dir.name, target_is_directory=True)
        os.replace(tmp_dir, final_dir)

    def cleanup_dir():
        if _rank() != 0:
            return
        if tmp_dir.is_symlink():
            tmp_dir.unlink()
        elif tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        if version_dir is not None and version_dir.exists() and final_dir.resolve() != version_dir.resolve():
            shutil.rmtree(version_dir)

    for phase in (make_tmp_dir, lambda: write_shards(tmp_dir), publish_dir):
        error = _run_checkpoint_phase(phase)
        if error is not None:
            cleanup_error = _run_checkpoint_phase(cleanup_dir)
            if cleanup_error is not None:
                logger.error(f"Failed to clean up checkpoint {final_dir}: {cleanup_error}")
            raise error


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _run_checkpoint_phase(phase: Callable[[], None]) -> Exception | None:
    """Agree on errors after all ranks return; callbacks must finish their collectives before raising."""
    error = None
    try:
        phase()
    except Exception as exc:
        error = exc
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, f"{type(error).__name__}: {error}" if error else None, group=get_gloo_group())
        failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None]
        if failures:
            return RuntimeError("Checkpoint write failed: " + "; ".join(failures))
    return error
