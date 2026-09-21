"""Collective checkpoint writes with coordinated failure cleanup."""

# TODO: isolate checkpoint IO failures in Tinker; they still terminate the trainer cell.

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

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
    """Replace a checkpoint directory collectively; callers must exclude concurrent readers.

    All ranks must call. Writers must finish their collectives before raising local IO errors.
    """
    checkpoint_dir = prepare_checkpoint_dir(path, overwrite=overwrite)

    def write_metadata():
        if _rank() == 0 and metadata is not None:
            (checkpoint_dir / "META.json").write_text(json.dumps(metadata, indent=2))

    for phase in (lambda: write_shards(checkpoint_dir), write_metadata):
        error = _run_checkpoint_phase(phase)
        if error is not None:
            cleanup_error = _run_checkpoint_phase(lambda: remove_checkpoint_dir(checkpoint_dir))
            if cleanup_error is not None:
                logger.error(f"Failed to clean up checkpoint {checkpoint_dir}: {cleanup_error}")
            raise error


def prepare_checkpoint_dir(path: str | Path, *, overwrite: bool = True) -> Path:
    """Prepare on rank 0 and agree on errors before writers enter weight collectives."""
    checkpoint_dir = Path(path)

    def prepare_dir():
        if _rank() == 0:
            if not overwrite and (checkpoint_dir.exists() or checkpoint_dir.is_symlink()):
                raise FileExistsError(f"checkpoint {checkpoint_dir} already exists")
            remove_checkpoint_dir(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True)

    error = _run_checkpoint_phase(prepare_dir)
    if error is not None:
        raise error
    return checkpoint_dir


def remove_checkpoint_dir(path: str | Path) -> None:
    """Remove on rank 0; callers must ensure readers and writers have stopped."""
    if _rank() != 0:
        return
    checkpoint_dir = Path(path)
    if checkpoint_dir.is_symlink():
        version_dir = checkpoint_dir.resolve()
        checkpoint_dir.unlink()
        if version_dir.exists():
            shutil.rmtree(version_dir)
    elif checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


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
