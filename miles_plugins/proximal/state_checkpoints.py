"""Verified recovery bundles for the existing Modal run snapshot path.

Only native Miles save completion plus its matching task cursor certifies a
training boundary. Serving exports are deliberately not part of that proof.
"""

import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, assert_never

from pydantic import Field, model_validator

from miles_plugins.proximal.contracts import Contract, SafeId, canonical_bytes
from miles_plugins.proximal.snapshot import Digest
from miles_plugins.proximal.state_artifacts import StateFile, copy_verified, describe, verify
from miles_plugins.proximal.storage import write_atomic
from miles_plugins.proximal.training import AutoResume, FreshRun, LatestResume, ResumeSelection, SelectedResume

logger = logging.getLogger(__name__)


class RecoveryContext(Contract):
    run_id: SafeId
    contract_sha256: Digest
    world_size: Annotated[int, Field(gt=0)]
    max_policy_lag: Annotated[int, Field(ge=0)]
    model_args: tuple[str, ...]
    train_args: tuple[str, ...]
    image: str
    code_sha256: Digest


class NativeCompletion(Contract):
    schema_version: Literal[1]
    iteration: int
    world_size: Annotated[int, Field(gt=0)]
    layout: dict[str, int]
    optimizer: bool
    scheduler: bool
    rng: bool


class CheckpointManifest(Contract):
    schema_version: Literal[1] = 1
    step: Annotated[int, Field(ge=0)]
    context: RecoveryContext
    launch_id: SafeId
    parent: str | None  # Legacy parents are integer strings.
    previous: str | None  # Last published bundle, independent of a rewind parent.
    native: NativeCompletion
    files: tuple[StateFile, ...]

    @model_validator(mode="after")
    def _complete(self) -> "CheckpointManifest":
        if self.native.iteration != self.step or self.native.world_size != self.context.world_size:
            raise ValueError("Native checkpoint step/layout differs from recovery manifest")
        if not (self.native.optimizer and self.native.scheduler and self.native.rng):
            raise ValueError("Full recovery requires optimizer, scheduler and RNG state")
        names = [f.path for f in self.files]
        expected = {"store.dump", "cursor.json", "launch.json", "checkpoint/adapter/native_checkpoint.json"}
        for rank in range(self.native.world_size):
            expected.update(
                {f"checkpoint/adapter/{name}{rank}.pt" for name in ("adapter_megatron_rank", "training_state_rank")}
            )
        if len(names) != len(set(names)) or not expected.issubset(names):
            raise ValueError("Recovery manifest is missing required checkpoint files or contains duplicates")
        if any(
            not (name in {"store.dump", "cursor.json", "launch.json"} or name.startswith("checkpoint/adapter/"))
            for name in names
        ):
            raise ValueError("Unexpected file in checkpoint manifest")
        return self


def iter_dir(checkpoints: Path, step: int) -> Path:
    return checkpoints / f"iter_{step:07d}"


def native_completion(checkpoints: Path, step: int) -> NativeCompletion:
    adapter = iter_dir(checkpoints, step) / "adapter"
    native = NativeCompletion.model_validate_json((adapter / "native_checkpoint.json").read_bytes())
    if native.iteration != step or not (native.optimizer and native.scheduler and native.rng):
        raise ValueError("Checkpoint is not a complete optimizer/scheduler/RNG boundary")
    for rank in range(native.world_size):
        for prefix in ("adapter_megatron_rank", "training_state_rank"):
            if not (adapter / f"{prefix}{rank}.pt").is_file():
                raise ValueError(f"Checkpoint lacks {prefix}{rank}.pt")
    return native


def complete_steps(checkpoints: Path) -> list[int]:
    from miles_plugins.proximal.data_source import Cursor  # Trainer-only Miles dependencies.

    result = []
    for cursor in (checkpoints / "rollout").glob("proximal_*.json"):
        step = int(cursor.stem.removeprefix("proximal_"))
        try:
            native_completion(checkpoints, step)
            Cursor.model_validate_json(cursor.read_bytes())
        except (ValueError, FileNotFoundError):
            continue  # An interrupted save must never advance LATEST.
        result.append(step)
    return sorted(result)


def read_manifest(root: Path, checkpoint_id: str) -> CheckpointManifest:
    from miles_plugins.proximal.data_source import Cursor

    if not re.fullmatch(r"[0-9]{7,}-[0-9a-f]{64}", checkpoint_id):
        raise ValueError("Invalid recovery checkpoint identity")
    folder = root / "checkpoints" / checkpoint_id
    data = (folder / "manifest.json").read_bytes()
    manifest = CheckpointManifest.model_validate_json(data)
    if checkpoint_id != f"{manifest.step:07d}-{hashlib.sha256(data).hexdigest()}":
        raise ValueError("Recovery manifest fails its identity checksum")
    for file in manifest.files:
        verify(folder / file.path, file)
    if (
        NativeCompletion.model_validate_json((folder / "checkpoint/adapter/native_checkpoint.json").read_bytes())
        != manifest.native
    ):
        raise ValueError("Native completion differs from recovery manifest")
    Cursor.model_validate_json((folder / "cursor.json").read_bytes())
    return manifest


def _flags(args: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    key = ""
    for token in args:
        if token.startswith("--"):
            key, _, value = token.partition("=")
            result[key] = (value,) if "=" in token else ()
        else:
            result[key] = (*result.get(key, ()), token)
    return result


def check_context(old: RecoveryContext, new: RecoveryContext, selection: ResumeSelection) -> None:
    if isinstance(selection, FreshRun):
        raise ValueError("A fresh run cannot resume a checkpoint")
    for name in ("run_id", "contract_sha256", "world_size", "max_policy_lag", "model_args"):
        if getattr(old, name) != getattr(new, name):
            raise ValueError(f"Resume changes incompatible {name}")
    changes = [name for name in ("code_sha256", "image") if getattr(old, name) != getattr(new, name)]
    before, after = _flags(old.train_args), _flags(new.train_args)
    changes += [flag for flag in before.keys() | after.keys() if before.get(flag) != after.get(flag)]
    # The native optimizer format is not a resharding or optimizer-conversion format.
    forbidden = {
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--context-parallel-size",
        "--expert-model-parallel-size",
        "--expert-tensor-parallel-size",
        "--optimizer",
        "--actor-num-nodes",
        "--actor-num-gpus-per-node",
        "--no-load-optim",
        "--no-load-rng",
    }
    forbidden.update(flag for flag in changes if "parallel" in flag)
    rejected = set(changes) - set(selection.allow_changes) | set(changes) & forbidden
    if rejected:
        raise ValueError(f"Resume changes require compatible, explicit allow_changes: {sorted(rejected)}")


def resolve(root: Path, selection: ResumeSelection) -> str | None:
    pointer = root / "LATEST"
    latest = pointer.read_text().strip() if pointer.exists() else None
    if isinstance(selection, FreshRun):
        if latest is not None or (root / "run.json").exists():
            raise ValueError("Fresh launch requires an unused run ID")
        return None
    if isinstance(selection, SelectedResume):
        return selection.checkpoint
    if isinstance(selection, LatestResume):
        if latest is None:
            raise FileNotFoundError("Requested resume has no committed checkpoint")
        return latest
    if isinstance(selection, AutoResume):
        return latest
    assert_never(selection)


def take(
    step: int,
    *,
    checkpoints: Path,
    dsn: str,
    snapshot_root: Path,
    pg_bin: Path,
    context: RecoveryContext,
    launch_id: str,
    parent: str | None,
    commit: Callable[[], None],
) -> str:
    native = native_completion(checkpoints, step)
    # Stage locally, not on the Volume: no half-built directory can be a checkpoint.
    with tempfile.TemporaryDirectory(prefix="recovery-", dir=checkpoints) as temporary:
        staging = Path(temporary)
        subprocess.run(
            [str(pg_bin / "pg_dump"), "--format=custom", f"--file={staging / 'store.dump'}", dsn], check=True
        )
        shutil.copy2(snapshot_root / "launches" / launch_id / "config.json", staging / "launch.json")
        shutil.copytree(iter_dir(checkpoints, step) / "adapter", staging / "checkpoint/adapter")
        shutil.copy2(checkpoints / "rollout" / f"proximal_{step}.json", staging / "cursor.json")
        files = tuple(
            describe(path, relative=path.relative_to(staging).as_posix())
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        )
        manifest = CheckpointManifest(
            step=step,
            context=context,
            launch_id=launch_id,
            parent=parent,
            previous=(snapshot_root / "LATEST").read_text().strip() if (snapshot_root / "LATEST").exists() else None,
            native=native,
            files=files,
        )
        data = canonical_bytes(manifest)
        checkpoint_id = f"{step:07d}-{hashlib.sha256(data).hexdigest()}"
        target = snapshot_root / "checkpoints" / checkpoint_id
        for file in files:
            copy_verified(staging / file.path, target / file.path, file)
        commit()
        write_atomic(target / "manifest.json", data)
        commit()
        read_manifest(snapshot_root, checkpoint_id)
        write_atomic(snapshot_root / "LATEST", checkpoint_id.encode())
        commit()
    return checkpoint_id


def prune(root: Path, *, latest: str, commit: Callable[[], None]) -> None:
    """Keep latest and the previously published bundle plus explicit pins; never delete legacy/artifacts.

    Use publication order, not step number: rewinds can publish smaller steps.
    A pin is a file pins/<checkpoint-id>. Deletion is restricted to verified bundles.
    """
    manifest = read_manifest(root, latest)
    keep = {latest, manifest.previous}
    pins = root / "pins"
    if pins.exists():
        keep.update(path.name for path in pins.iterdir())
    for path in (root / "checkpoints").iterdir():
        if path.name in keep or not (path / "manifest.json").is_file():
            continue
        read_manifest(root, path.name)
        shutil.rmtree(path)
    commit()


def reconcile_groups(*, snapshot_root: Path, artifacts: Path, dsn: str, context: RecoveryContext) -> None:
    # Optional DB/codec dependencies are needed only on the training node.
    import psycopg

    from miles_plugins.proximal.state_artifacts import SCHEMA as OUTBOX_SCHEMA
    from miles_plugins.proximal.store import GroupIndex

    root = snapshot_root / "artifacts" / context.run_id / "groups"
    with psycopg.connect(dsn) as connection:
        connection.execute(OUTBOX_SCHEMA)
        # Requests in an old dump refer to that container's disk, not this attempt.
        connection.execute("DELETE FROM proximal_artifact_outbox WHERE training_run_id = %s", (context.run_id,))
        for path in sorted(root.glob("*.json")):
            index = GroupIndex.model_validate_json(path.read_bytes())
            group, policy = index.header, index.header.policy
            if path.stem != group.group_id or policy.run_id != context.run_id:
                raise ValueError("Durable group index belongs to another run/group")
            if group.contract_sha256 != context.contract_sha256:
                raise ValueError("Durable group index has a different training contract")
            if not group.identities or len({i.group_index for i in group.identities}) != 1:
                raise ValueError("Durable group index has inconsistent membership")
            # Bind the index's lineage/membership to the payload header without
            # loading historical tensor bodies. Eligible bodies are hashed below.
            expected_header = group.model_dump_json().encode()
            with path.with_suffix(".bin").open("rb") as stream:
                size = int.from_bytes(stream.read(8), "big")
                if size != len(expected_header) or stream.read(size) != expected_header:
                    raise ValueError(f"Durable index differs from payload header {group.group_id}")
            # Never revive a discarded version just because its bytes survived.
            connection.execute(
                "INSERT INTO proximal_policies (training_run_id, version, snapshot_sha256, policy, abandoned)"
                " VALUES (%s, %s, %s, %s, true) ON CONFLICT DO NOTHING",
                (context.run_id, policy.version, policy.snapshot.sha256, policy.model_dump_json()),
            )
            connection.execute(
                "INSERT INTO proximal_rollout_groups (training_run_id, group_id, policy_version, policy_sha256,"
                " group_index, contract_sha256, payload_path, payload_sha256, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    context.run_id,
                    group.group_id,
                    policy.version,
                    policy.snapshot.sha256,
                    group.identities[0].group_index,
                    group.contract_sha256,
                    str(artifacts / context.run_id / "groups" / f"{group.group_id}.bin"),
                    index.payload_sha256,
                    index.created_at,
                ),
            )
            row = connection.execute(
                "SELECT payload_sha256, created_at, policy_sha256, policy_version, contract_sha256, group_index"
                " FROM proximal_rollout_groups WHERE training_run_id = %s AND group_id = %s",
                (context.run_id, group.group_id),
            ).fetchone()
            if row != (
                index.payload_sha256,
                index.created_at,
                policy.snapshot.sha256,
                policy.version,
                group.contract_sha256,
                group.identities[0].group_index,
            ):
                raise ValueError(f"Conflicting durable group index {group.group_id}")
        # Paths in a pg_dump belong to its old container. Derive paths from typed IDs.
        connection.execute(
            "UPDATE proximal_rollout_groups SET payload_path = %s || group_id || '.bin' WHERE training_run_id = %s",
            (str(artifacts / context.run_id / "groups") + "/", context.run_id),
        )


def restore(
    *,
    snapshot_root: Path,
    checkpoints: Path,
    artifacts: Path,
    dsn: str,
    pg_bin: Path,
    context: RecoveryContext,
    selection: ResumeSelection,
) -> tuple[int | None, str | None]:
    from miles_plugins.proximal.data_source import Cursor
    from miles_plugins.proximal.e2e import snapshots

    checkpoint_id = resolve(snapshot_root, selection)
    if checkpoint_id is None:
        # A crash before the first checkpoint may still have durable complete groups.
        # They start abandoned; initial weight publication alone can make them live.
        import psycopg

        from miles_plugins.proximal.store import SCHEMA

        with psycopg.connect(dsn) as connection:
            connection.execute(SCHEMA)
        reconcile_groups(snapshot_root=snapshot_root, artifacts=artifacts, dsn=dsn, context=context)
        for row in snapshots._trainable_groups(dsn, oldest_version=0, consumed=[], run_id=context.run_id):
            snapshots._restore_payload(row, snapshot_artifacts=snapshot_root / "artifacts", artifacts=artifacts)
        return None, None
    if checkpoint_id.isdigit():
        logger.warning("Restoring legacy snapshot: no verified manifest or saved RNG evidence")
        step = snapshots.restore(
            snapshot_root=snapshot_root,
            checkpoints=checkpoints,
            artifacts=artifacts,
            dsn=dsn,
            pg_bin=pg_bin,
            max_policy_lag=context.max_policy_lag,
        )
        return step, checkpoint_id
    manifest = read_manifest(snapshot_root, checkpoint_id)
    check_context(manifest.context, context, selection)
    source = snapshot_root / "checkpoints" / checkpoint_id
    shutil.copytree(source / "checkpoint", iter_dir(checkpoints, manifest.step))
    cursor_path = checkpoints / "rollout" / f"proximal_{manifest.step}.json"
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "cursor.json", cursor_path)
    subprocess.run(
        [str(pg_bin / "pg_restore"), "--no-owner", "--exit-on-error", f"--dbname={dsn}", str(source / "store.dump")],
        check=True,
    )
    reconcile_groups(snapshot_root=snapshot_root, artifacts=artifacts, dsn=dsn, context=context)
    cursor = Cursor.model_validate_json(cursor_path.read_bytes())
    groups = snapshots._trainable_groups(
        dsn,
        oldest_version=manifest.step + 1 - context.max_policy_lag,
        consumed=[entry.group_id for entry in cursor.consumed],
        run_id=context.run_id,
    )
    for row in groups:
        snapshots._restore_payload(row, snapshot_artifacts=snapshot_root / "artifacts", artifacts=artifacts)
        index = snapshot_root / "artifacts" / context.run_id / "groups" / f"{row.group_id}.json"
        if index.exists():
            write_atomic(Path(row.payload_path).with_suffix(".json"), index.read_bytes())
    return manifest.step, checkpoint_id
