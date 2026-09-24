"""Per-step snapshots of a training node's state onto a persistent directory (a Modal Volume).

The trainer keeps working state on local disk: Miles checkpoints under ``checkpoints``,
the rollout store's Postgres, and the store's payload files under ``artifacts``. After
each saved step this copies a consistent snapshot to ``snapshot_root``; on restart,
``restore`` puts the latest one back so Miles resumes from it.

Consistency: a step is snapshotted only once both its LoRA checkpoint and the task
source's cursor file exist (Miles writes the cursor after the weights). The store is
dumped after that, so it is never behind the checkpoint; being ahead is expected and
safe (resume abandons newer policy versions and never trains their groups). The dump
is taken before payload files are copied, so every dumped row has its file.
"""

import os
import shutil
import subprocess
from pathlib import Path

LATEST = "LATEST"
ADAPTER_FILES = ("adapter_model.bin", "adapter_config.json")


def iter_dir(checkpoints: Path, step: int) -> Path:
    return checkpoints / f"iter_{step:07d}"


def complete_steps(checkpoints: Path) -> list[int]:
    """Steps whose LoRA checkpoint and task-source cursor are both written."""
    steps = []
    for cursor in (checkpoints / "rollout").glob("proximal_*.json"):
        step = int(cursor.stem.removeprefix("proximal_"))
        adapter = iter_dir(checkpoints, step) / "adapter"
        if all((adapter / name).is_file() for name in ADAPTER_FILES) and any(
            adapter.glob("adapter_megatron_rank*.pt")
        ):
            steps.append(step)
    return sorted(steps)


def latest_snapshot(snapshot_root: Path) -> int | None:
    path = snapshot_root / LATEST
    return int(path.read_text()) if path.is_file() else None


def _copy_new(source: Path, target: Path) -> None:
    """Copy files missing from target; store payloads and adapters are write-once."""
    for path in source.rglob("*"):
        if path.is_file():
            destination = target / path.relative_to(source)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def step_dir(snapshot_root: Path, step: int) -> Path:
    return snapshot_root / "steps" / f"{step:07d}"


def take(step: int, *, checkpoints: Path, artifacts: Path, dsn: str, snapshot_root: Path, pg_bin: Path) -> None:
    """Snapshot one completed step as a self-contained directory, then prune.

    ``steps/<step>`` holds the store dump, the LoRA checkpoint and the task cursor; it is
    assembled under a temporary name and renamed into place before ``LATEST`` points at
    it. The two newest step directories are kept, so a reader that has just read
    ``LATEST`` still finds its files while the next snapshot is written. Payload files
    are write-once and shared across steps.
    """
    final = step_dir(snapshot_root, step)
    staging = final.with_name(f".{final.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    subprocess.run([str(pg_bin / "pg_dump"), "--format=custom", f"--file={staging / 'store.dump'}", dsn], check=True)
    _copy_new(artifacts, snapshot_root / "artifacts")  # After the dump: every dumped row has its file.
    shutil.copytree(iter_dir(checkpoints, step), staging / "checkpoint")
    shutil.copy2(checkpoints / "rollout" / f"proximal_{step}.json", staging / "cursor.json")
    shutil.rmtree(final, ignore_errors=True)
    os.replace(staging, final)
    _write_atomic(snapshot_root / LATEST, str(step))
    kept = sorted(int(path.name) for path in (snapshot_root / "steps").iterdir() if path.name.isdigit())[-2:]
    for path in (snapshot_root / "steps").iterdir():
        if path.name.isdigit() and int(path.name) not in kept:
            shutil.rmtree(path, ignore_errors=True)
    for old in checkpoints.glob("iter_*"):
        if int(old.name.removeprefix("iter_")) < step:
            shutil.rmtree(old, ignore_errors=True)


def reset_local_state(state: Path) -> None:
    """Start an attempt from empty local state.

    Modal may retry a failed input in the same container, where the previous attempt's
    Postgres data and checkpoints still exist; restoring into them fails.
    """
    state.mkdir(parents=True, exist_ok=True)
    for child in state.iterdir():  # Contents, not the directory: it may be a mount point.
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def restore(*, snapshot_root: Path, checkpoints: Path, artifacts: Path, dsn: str, pg_bin: Path) -> int | None:
    """Put the latest snapshot back on empty local disk and into the empty store; returns its step."""
    step = latest_snapshot(snapshot_root)
    if step is None:
        return None
    source = step_dir(snapshot_root, step)
    shutil.copytree(source / "checkpoint", iter_dir(checkpoints, step))
    (checkpoints / "rollout").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "cursor.json", checkpoints / "rollout" / f"proximal_{step}.json")
    _copy_new(snapshot_root / "artifacts", artifacts)
    subprocess.run(
        [str(pg_bin / "pg_restore"), "--no-owner", "--exit-on-error", f"--dbname={dsn}", str(source / "store.dump")],
        check=True,
    )
    return step
