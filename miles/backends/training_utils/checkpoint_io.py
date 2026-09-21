"""Snapshot and native checkpoint directory writes."""

# TODO: isolate checkpoint IO failures in Tinker; they still terminate the trainer cell.

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


def write_snapshot_dir(
    path: str | Path,
    write_weights: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
    completion_marker: str | None = None,
) -> None:
    """Write collectively with rank 0 as the sole file writer.

    Finish all weight collectives before raising local write errors.
    """
    checkpoint_dir = prepare_checkpoint_dir(path, overwrite=overwrite)
    writers_stopped = False
    try:
        try:
            write_weights(checkpoint_dir)
        finally:
            if dist.is_initialized():
                dist.barrier(group=get_gloo_group())
            writers_stopped = True
        if _rank() == 0:
            if metadata is not None:
                (checkpoint_dir / "META.json").write_text(json.dumps(metadata, indent=2))
            if completion_marker is not None:
                (checkpoint_dir / completion_marker).touch()
    except Exception:
        if writers_stopped:
            try:
                remove_checkpoint_dir(checkpoint_dir)
            except OSError:
                logger.exception(f"Failed to clean up snapshot {checkpoint_dir}")
        raise


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
) -> None:
    """Write native checkpoint shards collectively; callers must exclude concurrent readers.

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
    """Prepare on rank 0 and broadcast failures before entering weight collectives."""
    checkpoint_dir = Path(path)
    error = [None]
    if _rank() == 0:
        try:
            if not overwrite and checkpoint_dir.exists():
                raise FileExistsError(f"checkpoint {checkpoint_dir} already exists")
            remove_checkpoint_dir(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True)
        except Exception as exc:
            error[0] = exc
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0, group=get_gloo_group())
    if error[0] is not None:
        raise error[0]
    return checkpoint_dir


def remove_checkpoint_dir(path: str | Path) -> None:
    """Remove on rank 0; callers must ensure readers and writers have stopped."""
    if _rank() != 0:
        return
    checkpoint_dir = Path(path)
    if checkpoint_dir.exists():
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
