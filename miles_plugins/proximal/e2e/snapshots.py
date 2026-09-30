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

``restore`` copies back only the payloads the resumed run can still train; the rest of
``artifacts`` (consumed or stale groups, accepted-attempt evidence, publication
staging) stays in the snapshot alone. That keeps the invariant: ``take`` only adds
files missing from the snapshot and never removes one, so a row left out of a lean
restore still has its file there when a later dump names it again.
"""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import psycopg

from miles_plugins.proximal.data_source import Cursor
from miles_plugins.proximal.storage import write_immutable
from miles_plugins.proximal.store import GroupRow

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
    """Copy files missing from target; store payloads and adapters are write-once.

    Dot-directories are other writers' scratch space (the publisher stages each export
    in ``.export-*`` and deletes it), so a file there can vanish between listing and
    copying; neither is state to snapshot.
    """
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        destination = target / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, destination)
            except FileNotFoundError:
                destination.unlink(missing_ok=True)


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
    # After the dump: every dumped row has its file (a row a lean restore left out already had).
    _copy_new(artifacts, snapshot_root / "artifacts")
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


# Stored groups a resumed run could still select, whatever their policy's state: a
# version abandoned by the resume comes back if identical weights are published again
# (RolloutStore.commit_policy), so only the lag window and the ledger rule a group out.
_TRAINABLE = """
SELECT group_id, policy_version, payload_path, payload_sha256
FROM proximal_rollout_groups
WHERE policy_version >= %s AND NOT (group_id = ANY(%s))
ORDER BY group_id
"""


def _trainable_groups(
    dsn: str, *, oldest_version: int, consumed: list[str], run_id: str | None = None
) -> list[GroupRow]:
    with psycopg.connect(dsn) as connection:
        query = (
            _TRAINABLE.replace("ORDER BY group_id", "AND training_run_id = %s ORDER BY group_id")
            if run_id
            else _TRAINABLE
        )
        parameters = (oldest_version, consumed, run_id) if run_id else (oldest_version, consumed)
        rows = connection.execute(query, parameters).fetchall()
    return [GroupRow(str(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in rows]


def _restore_payload(row: GroupRow, *, snapshot_artifacts: Path, artifacts: Path) -> None:
    """Copy one group's payload back to the path its row names, verified against its digest."""
    target = Path(row.payload_path)
    source = snapshot_artifacts / target.relative_to(artifacts)
    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The snapshot lacks the payload of trainable group {row.group_id}: {source}"
        ) from None
    if hashlib.sha256(payload).hexdigest() != row.payload_sha256:
        raise ValueError(f"The snapshot's payload of trainable group {row.group_id} fails its checksum: {source}")
    write_immutable(target, payload)


def restore(
    *, snapshot_root: Path, checkpoints: Path, artifacts: Path, dsn: str, pg_bin: Path, max_policy_lag: int
) -> int | None:
    """Put the latest snapshot back on empty local disk and into the empty store; returns its step.

    Of ``artifacts``, only the payloads of groups the resumed run can still train are
    copied, each checked against its row, so a resume fails here rather than on a
    missing payload mid-step. A group is trainable unless the restored ledger has
    consumed it or it is older than the lag window: a trainer resumed from step N
    starts at weight version N + 1 (the updater's initial version is Miles's
    ``start_rollout_id``) and its version only grows, so no batch query after the
    restore selects below N + 1 - ``max_policy_lag``. Training reads nothing else
    under ``artifacts``: accepted-attempt evidence and publication staging are write-only.
    """
    step = latest_snapshot(snapshot_root)
    if step is None:
        return None
    source = step_dir(snapshot_root, step)
    shutil.copytree(source / "checkpoint", iter_dir(checkpoints, step))
    (checkpoints / "rollout").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "cursor.json", checkpoints / "rollout" / f"proximal_{step}.json")
    subprocess.run(
        [str(pg_bin / "pg_restore"), "--no-owner", "--exit-on-error", f"--dbname={dsn}", str(source / "store.dump")],
        check=True,
    )
    cursor = Cursor.model_validate_json((source / "cursor.json").read_bytes())
    groups = _trainable_groups(
        dsn, oldest_version=step + 1 - max_policy_lag, consumed=[entry.group_id for entry in cursor.consumed]
    )
    for row in groups:
        _restore_payload(row, snapshot_artifacts=snapshot_root / "artifacts", artifacts=artifacts)
    return step
