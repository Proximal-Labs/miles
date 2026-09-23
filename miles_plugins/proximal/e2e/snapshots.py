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


def take(step: int, *, checkpoints: Path, artifacts: Path, dsn: str, snapshot_root: Path, pg_bin: Path) -> None:
    """Snapshot one completed step, then drop older steps from both sides."""
    snap_checkpoints = snapshot_root / "checkpoints"
    snap_checkpoints.mkdir(parents=True, exist_ok=True)
    dump = snapshot_root / f"store-{step:07d}.dump"
    subprocess.run([str(pg_bin / "pg_dump"), "--format=custom", f"--file={dump}", dsn], check=True)
    _copy_new(artifacts, snapshot_root / "artifacts")
    shutil.copytree(iter_dir(checkpoints, step), iter_dir(snap_checkpoints, step), dirs_exist_ok=True)
    cursor = checkpoints / "rollout" / f"proximal_{step}.json"
    (snap_checkpoints / "rollout").mkdir(exist_ok=True)
    shutil.copy2(cursor, snap_checkpoints / "rollout" / cursor.name)
    _write_atomic(snapshot_root / LATEST, str(step))
    for old in snapshot_root.glob("store-*.dump"):
        if old != dump:
            old.unlink()
    for root in (snap_checkpoints, checkpoints):
        for old in root.glob("iter_*"):
            if int(old.name.removeprefix("iter_")) < step:
                shutil.rmtree(old, ignore_errors=True)


def restore(*, snapshot_root: Path, checkpoints: Path, artifacts: Path, dsn: str, pg_bin: Path) -> int | None:
    """Put the latest snapshot back on local disk and into the empty store; returns its step."""
    step = latest_snapshot(snapshot_root)
    if step is None:
        return None
    snap_checkpoints = snapshot_root / "checkpoints"
    shutil.copytree(iter_dir(snap_checkpoints, step), iter_dir(checkpoints, step), dirs_exist_ok=True)
    (checkpoints / "rollout").mkdir(parents=True, exist_ok=True)
    cursor = f"proximal_{step}.json"
    shutil.copy2(snap_checkpoints / "rollout" / cursor, checkpoints / "rollout" / cursor)
    _copy_new(snapshot_root / "artifacts", artifacts)
    subprocess.run(
        [
            str(pg_bin / "pg_restore"),
            "--no-owner",
            "--exit-on-error",
            f"--dbname={dsn}",
            str(snapshot_root / f"store-{step:07d}.dump"),
        ],
        check=True,
    )
    return step
