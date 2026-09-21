"""Checkpoint directories: written collectively, complete at their final path."""

# TODO: isolate checkpoint IO failures; they currently terminate the trainer cell.

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
) -> None:
    """Write collectively, then atomically point ``path`` at the completed version.

    All ranks must call. Any rank's failure raises on every rank and discards the
    staged version. Readers may still hold an older version, so retain it.
    """
    final_dir = Path(path)
    tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}"
    version_dir = final_dir.parent / f"_version_{final_dir.name}_{uuid4().hex}"

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
        if _rank() != 0:
            return
        if metadata is not None:
            (tmp_dir / "META.json").write_text(json.dumps(metadata, indent=2))
        os.replace(tmp_dir, version_dir)
        tmp_dir.symlink_to(version_dir.name, target_is_directory=True)
        os.replace(tmp_dir, final_dir)

    def discard():
        # publish_dir points final_dir at version_dir last, so on failure version_dir is unpublished
        if _rank() != 0:
            return
        if tmp_dir.is_symlink():
            tmp_dir.unlink()
        elif tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        if version_dir.exists():
            shutil.rmtree(version_dir)

    _run_phase(make_tmp_dir, discard)
    _run_phase(lambda: write_shards(tmp_dir), discard)
    _run_phase(publish_dir, discard)


def _run_phase(step: Callable[[], None], discard: Callable[[], None]) -> None:
    """Run one collective step; a failure on any rank raises on every rank once all ranks have stopped."""
    error = None
    try:
        step()
    except Exception as e:
        error = e
    message = None if error is None else f"{type(error).__name__}: {error}"
    if dist.is_initialized():
        group = get_gloo_group()
        messages: list[str | None] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(messages, message, group=group)
    else:
        messages = [message]
    if not any(messages):
        return
    discard()
    if error is not None:
        raise error
    failed_rank = next(rank for rank, m in enumerate(messages) if m is not None)
    raise RuntimeError(f"checkpoint write failed on rank {failed_rank}: {messages[failed_rank]}")


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0
