"""Checkpoint filesystem errors propagate to the trainer cell."""

import multiprocessing
import os
from pathlib import Path

import pytest

from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("directory creation failed")])
def test_directory_errors_propagate(error, tmp_path, monkeypatch):
    def make_dir(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "mkdir", make_dir)
    with pytest.raises(type(error), match=str(error)) as caught:
        write_checkpoint_dir(tmp_path / "checkpoint", lambda _: None)
    assert caught.value is error


@pytest.mark.parametrize("crash_before_publish", [True, False])
def test_crashed_overwrite_keeps_a_complete_checkpoint(tmp_path, crash_before_publish):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("old"))
    old_version = checkpoint.resolve()

    def overwrite_and_crash():
        replace = os.replace

        def crash_at_publish(source, destination):
            if Path(destination) == checkpoint and crash_before_publish:
                os._exit(73)
            replace(source, destination)
            if Path(source) == checkpoint or Path(destination) == checkpoint:
                os._exit(73)

        os.replace = crash_at_publish
        write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("new"))

    child = multiprocessing.get_context("fork").Process(target=overwrite_and_crash)
    child.start()
    child.join(timeout=10)
    assert child.exitcode == 73
    assert (checkpoint / "value").read_text() == ("old" if crash_before_publish else "new")
    assert (old_version / "value").read_text() == "old"

    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("retry"))
    assert (checkpoint / "value").read_text() == "retry"


def test_failed_write_discards_staging_and_keeps_previous_version(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("old"))

    def write_and_fail(directory):
        (directory / "value").write_text("new")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_checkpoint_dir(checkpoint, write_and_fail)
    assert (checkpoint / "value").read_text() == "old"
    assert {p.name for p in tmp_path.iterdir()} == {"checkpoint", checkpoint.resolve().name}


def test_failed_publish_discards_unpublished_version(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    replace = os.replace

    def fail_at_publish(source, destination):
        if Path(destination) == checkpoint:
            raise OSError("publish failed")
        replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_at_publish)
    with pytest.raises(OSError, match="publish failed"):
        write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("new"))
    assert list(tmp_path.iterdir()) == []


def _write_as_rank(rank, tmp_path):
    from datetime import timedelta

    import torch.distributed as dist

    from miles.utils import distributed_utils

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    distributed_utils.init_gloo_group()

    def write_shards(directory):
        (directory / f"rank{rank}").write_text("x")
        if rank == 0:
            raise OSError("disk full")

    try:
        write_checkpoint_dir(tmp_path / "checkpoint", write_shards)
        outcome = "ok"
    except Exception as e:
        outcome = f"{type(e).__name__}: {e}"
    (tmp_path / f"rank{rank}.outcome").write_text(outcome)
    dist.destroy_process_group()


def test_one_rank_failing_raises_on_every_rank(tmp_path):
    ctx = multiprocessing.get_context("fork")
    workers = [ctx.Process(target=_write_as_rank, args=(rank, tmp_path)) for rank in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    assert (tmp_path / "rank0.outcome").read_text() == "OSError: disk full"
    assert (
        tmp_path / "rank1.outcome"
    ).read_text() == "RuntimeError: checkpoint write failed on rank 0: OSError: disk full"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rank0.outcome", "rank1.outcome", "store"]
