"""A resumed training node gets back only the payloads it can still train (e2e.snapshots).

Run 013's resume copied the whole artifacts tree from the state Volume: 5.1 GB after
four steps at ~5 MB/s, about 16 minutes, and growing every step.
"""

import asyncio
import hashlib
import shutil
import subprocess
import uuid
from pathlib import Path

import psycopg
import pytest
from tests.integration.proximal_async.test_buffer import entry, make_buffer, versioned

from miles_plugins.proximal.buffer import accepted
from miles_plugins.proximal.data_source import ConsumedGroup, ConsumptionLedger, Cursor
from miles_plugins.proximal.e2e import snapshots
from miles_plugins.proximal.e2e.local_postgres import server_binaries
from miles_plugins.proximal.store import open_store

# The fixture's max_policy_lag is 1. A trainer resumed from step 3 starts at version 4
# (step 3's weights were published as version 5), so its batch queries select version 3
# and later; version 6 was published after the checkpoint.
STEP = 3
GROUPS = {"stale": 2, "consumed": 3, "backlog": 3, "fresh": 4, "current": 5, "ahead": 6}
TRAINABLE = {"backlog", "fresh", "current", "ahead"}
EVIDENCE = ("accepted/backlog-0/accepted.json", "accepted/backlog-0/samples.safetensors")
PUBLICATION = f"publication/snapshots/{'e' * 64}/manifest.json"


@pytest.fixture
def pg_bin():
    return server_binaries()


@pytest.fixture
def empty_database(postgres_server, monkeypatch):
    """Makes a new empty database the run's store, as in a fresh container."""

    def create() -> str:
        name = f"t_{uuid.uuid4().hex}"
        with psycopg.connect(postgres_server, autocommit=True) as admin:
            admin.execute(f"CREATE DATABASE {name}")
        dsn = postgres_server.replace("dbname=postgres", f"dbname={name}")
        monkeypatch.setenv("STORE_TEST_DSN", dsn)
        return dsn

    return create


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def payloads(*names: str) -> set[str]:
    return {f"test-run/groups/{name}.bin" for name in names}


def save_checkpoint(checkpoints: Path, step: int, consumed: dict[str, int]) -> None:
    """What Miles leaves for a completed step: the LoRA checkpoint, then the task cursor."""
    adapter = snapshots.iter_dir(checkpoints, step) / "adapter"
    for name in (*snapshots.ADAPTER_FILES, "adapter_megatron_rank0.pt"):
        write(adapter / name, name)
    cursor = Cursor(
        dataset_sha256="0" * 64,
        next_group=len(GROUPS),
        pending_tasks=(),
        consumed=tuple(ConsumedGroup(group_id=g, policy_version=v) for g, v in sorted(consumed.items())),
    )
    write(checkpoints / "rollout" / f"proximal_{step}.json", cursor.model_dump_json())


async def lose_container(config, checkpoints: Path, store) -> None:
    await store.close()
    shutil.rmtree(config.artifact_directory)
    shutil.rmtree(checkpoints)


def restore(config, checkpoints: Path, snapshot_root: Path, dsn: str, pg_bin: Path) -> int | None:
    return snapshots.restore(
        snapshot_root=snapshot_root,
        checkpoints=checkpoints,
        artifacts=config.artifact_directory,
        dsn=dsn,
        pg_bin=pg_bin,
        max_policy_lag=config.research.max_policy_lag,
    )


def dumped_rows(pg_bin: Path, dump: Path, dsn: str) -> list[tuple[str, str, str]]:
    subprocess.run(
        [str(pg_bin / "pg_restore"), "--no-owner", "--exit-on-error", f"--dbname={dsn}", str(dump)], check=True
    )
    with psycopg.connect(dsn) as connection:
        rows = connection.execute("SELECT group_id, payload_path, payload_sha256 FROM proximal_rollout_groups")
        return [(str(g), str(p), str(s)) for g, p, s in rows.fetchall()]


@pytest.fixture
async def snapshot_of_step_3(config, tmp_path, attempt, policy, store, store_dsn, pg_bin):
    """Step 3 snapshotted with groups on both sides of the lag window, evidence and a publication;
    then the container's local state is gone."""
    for version in range(1, 7):
        await store.commit_policy(versioned(policy, version))
    for index, (name, version) in enumerate(GROUPS.items()):
        group = entry(attempt, versioned(policy, version), group=name, group_index=index)
        await store.add_group(name, versioned(policy, version), group.group)
    for relative in (*EVIDENCE, PUBLICATION):
        write(config.artifact_directory / "test-run" / relative, relative)
    checkpoints, snapshot_root = tmp_path / "checkpoints", tmp_path / "snapshot"
    save_checkpoint(checkpoints, STEP, {"consumed": 3})
    snapshots.take(
        STEP,
        checkpoints=checkpoints,
        artifacts=config.artifact_directory,
        dsn=store_dsn,
        snapshot_root=snapshot_root,
        pg_bin=pg_bin,
    )
    await lose_container(config, checkpoints, store)
    return checkpoints, snapshot_root


async def test_restore_copies_only_payloads_the_run_can_still_train(
    config, tmp_path, policy, snapshot_of_step_3, empty_database, pg_bin
):
    checkpoints, snapshot_root = snapshot_of_step_3
    assert restore(config, checkpoints, snapshot_root, empty_database(), pg_bin) == STEP

    # Not the consumed group, the one below the window, the evidence or the publication.
    assert files(config.artifact_directory) == payloads(*TRAINABLE)
    assert (checkpoints / "rollout" / f"proximal_{STEP}.json").is_file()
    adapter = snapshots.iter_dir(checkpoints, STEP) / "adapter"
    assert all((adapter / name).is_file() for name in snapshots.ADAPTER_FILES)

    # Resume: rewind to the resumed version and query from it (the lowest version the
    # trainer can hold), then republish versions 5 and 6. Identical weights revive an
    # abandoned version, so a deterministic rerun trains even the group sampled after
    # the checkpoint.
    store = await open_store(config)
    try:
        await store.rewind(keep_through=STEP + 1)
        ledger = ConsumptionLedger()
        cursor = Cursor.model_validate_json((checkpoints / "rollout" / f"proximal_{STEP}.json").read_bytes())
        ledger.restore(cursor.consumed)
        buffer = make_buffer(config, tmp_path, store, ledger)
        trained = []
        for version, count in ((4, 2), (5, 1), (6, 1)):
            if version > STEP + 1:
                await store.commit_policy(versioned(policy, version))
            trained += [await buffer.get(current_version=version) for _ in range(count)]
        # Each get loaded its payload and checked it against its row and its evidence.
        assert [accepted(group.group[0]).attempt.group_id for group in trained] == [
            "backlog",
            "fresh",
            "current",
            "ahead",
        ]
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(buffer.get(current_version=6), 0.2)
    finally:
        await store.close()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
async def test_restore_fails_loudly_when_the_snapshot_lacks_a_trainable_payload(
    config, snapshot_of_step_3, empty_database, pg_bin, damage
):
    checkpoints, snapshot_root = snapshot_of_step_3
    kept = snapshot_root / "artifacts" / "test-run" / "groups" / "fresh.bin"
    if damage == "missing":
        kept.unlink()
    else:
        kept.write_bytes(b"not the stored group")
    with pytest.raises((FileNotFoundError, ValueError), match="trainable group fresh"):
        restore(config, checkpoints, snapshot_root, empty_database(), pg_bin)


async def test_a_snapshot_after_a_lean_restore_is_complete(
    config, attempt, policy, snapshot_of_step_3, empty_database, pg_bin
):
    checkpoints, snapshot_root = snapshot_of_step_3
    volume = snapshot_root / "artifacts"
    before = {name: (volume / name).read_bytes() for name in files(volume)}
    dsn = empty_database()
    restore(config, checkpoints, snapshot_root, dsn, pg_bin)

    # Step 4 consumes two groups and a new one lands.
    store = await open_store(config)
    new = versioned(policy, 5)
    await store.add_group("new", new, entry(attempt, new, group="new", group_index=9).group)
    save_checkpoint(checkpoints, STEP + 1, {"backlog": 3, "fresh": 4})
    snapshots.take(
        STEP + 1,
        checkpoints=checkpoints,
        artifacts=config.artifact_directory,
        dsn=dsn,
        snapshot_root=snapshot_root,
        pg_bin=pg_bin,
    )
    assert snapshots.latest_snapshot(snapshot_root) == STEP + 1

    # Every row of the new dump has its payload in the snapshot, including the rows whose
    # payloads never came back to local disk; nothing already there was removed or rewritten.
    rows = dumped_rows(pg_bin, snapshots.step_dir(snapshot_root, STEP + 1) / "store.dump", empty_database())
    assert {group for group, _, _ in rows} == {*GROUPS, "new"}
    for group, path, sha256 in rows:
        kept = volume / Path(path).relative_to(config.artifact_directory)
        assert hashlib.sha256(kept.read_bytes()).hexdigest() == sha256, group
    assert files(volume) == {*before, *payloads("new")}
    assert all((volume / name).read_bytes() == data for name, data in before.items())

    # That snapshot restores leanly in turn: version 4 and later, minus step 4's ledger.
    await lose_container(config, checkpoints, store)
    assert restore(config, checkpoints, snapshot_root, empty_database(), pg_bin) == STEP + 1
    assert files(config.artifact_directory) == payloads("current", "ahead", "new")
