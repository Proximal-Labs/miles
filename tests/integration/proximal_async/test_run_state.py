"""Crash/restart witnesses using real files, Postgres, codecs and publication code.

Only the Volume commit RPC is substituted. These do not claim Modal's real mount
semantics or GPU numerical equivalence.
"""

import asyncio
import hashlib
import shutil

import psycopg
import pytest
from tests.integration.proximal_async.test_buffer import entry, versioned

from miles_plugins.proximal import state_artifacts
from miles_plugins.proximal import state_checkpoints as checkpoints
from miles_plugins.proximal.contracts import RunStateArtifacts, digest, training_contract
from miles_plugins.proximal.data_source import ConsumedGroup, Cursor
from miles_plugins.proximal.state_checkpoints import NativeCompletion, RecoveryContext
from miles_plugins.proximal.state_writer import StateWriter
from miles_plugins.proximal.store import open_store
from miles_plugins.proximal.training import AutoResume, LatestResume


@pytest.fixture
def context(config, tmp_path):
    result = RecoveryContext(
        run_id=config.run_id,
        contract_sha256=digest(training_contract(config)),
        world_size=2,
        max_policy_lag=config.research.max_policy_lag,
        model_args=("--tensor-model-parallel-size", "2"),
        train_args=("--lr", "0.00004"),
        image="training@sha256:abc",
        code_sha256="f" * 64,
    )
    # The real composition root records each launch before its writer starts.
    for root in ("volume", "mount"):
        for launch in ("one", "launch-1"):
            path = tmp_path / root / "launches" / launch / "config.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result.model_dump_json())
    return result


def native_save(root, *, step=3, consumed=()):
    adapter = checkpoints.iter_dir(root, step) / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    for rank in range(2):
        (adapter / f"adapter_megatron_rank{rank}.pt").write_bytes(f"weights-{step}-{rank}".encode())
        (adapter / f"training_state_rank{rank}.pt").write_bytes(f"optimizer-{step}-{rank}".encode())
    record = NativeCompletion(
        schema_version=1, iteration=step, world_size=2, layout={"tp": 2}, optimizer=True, scheduler=True, rng=True
    )
    (adapter / "native_checkpoint.json").write_text(record.model_dump_json())
    (root / "rollout").mkdir(exist_ok=True)
    (root / "rollout" / f"proximal_{step}.json").write_text(
        Cursor(
            dataset_sha256="0" * 64,
            next_group=10,
            pending_tasks=(),
            consumed=tuple(ConsumedGroup(group_id=g, policy_version=v) for g, v in consumed),
        ).model_dump_json()
    )


async def run_state_store(config):
    return await open_store(config.model_copy(update={"artifact_storage": RunStateArtifacts(kind="run_state")}))


async def settle(task, *, config, store_dsn, root, commit=lambda: None):
    for _ in range(200):
        await asyncio.to_thread(
            state_artifacts.publish_pending,
            dsn=store_dsn,
            run_id=config.run_id,
            artifacts=config.artifact_directory,
            snapshot_root=root,
            commit=commit,
        )
        if task.done():
            return await task
        await asyncio.sleep(0.01)
    raise TimeoutError("outbox did not settle")


async def test_groups_after_snapshot_are_reconciled_without_reviving_policy(
    config,
    context,
    tmp_path,
    store_dsn,
    policy,
    attempt,
    empty_database,
    pg_bin,
):
    store = await run_state_store(config)
    root, working = tmp_path / "volume", tmp_path / "checkpoints"
    await store.commit_policy(policy)
    await settle(
        asyncio.create_task(store.add_group("consumed", policy, entry(attempt, policy, group="consumed").group)),
        config=config,
        store_dsn=store_dsn,
        root=root,
    )
    native_save(working, step=0, consumed=(("consumed", 1),))
    first = checkpoints.take(
        0,
        checkpoints=working,
        dsn=store_dsn,
        snapshot_root=root,
        pg_bin=pg_bin,
        context=context,
        launch_id="launch-1",
        parent=None,
        commit=lambda: None,
    )
    for group, p in (("after", policy), ("future", versioned(policy, 2))):
        if p.version == 2:
            await store.commit_policy(p)
        await settle(
            asyncio.create_task(store.add_group(group, p, entry(attempt, p, group=group).group)),
            config=config,
            store_dsn=store_dsn,
            root=root,
        )
    with psycopg.connect(store_dsn) as db:
        times = db.execute("SELECT group_id, created_at FROM proximal_rollout_groups ORDER BY group_id").fetchall()
    await store.close()
    shutil.rmtree(config.artifact_directory)
    shutil.rmtree(working)
    dsn = empty_database()
    step, restored = checkpoints.restore(
        snapshot_root=root,
        checkpoints=working,
        artifacts=config.artifact_directory,
        dsn=dsn,
        pg_bin=pg_bin,
        context=context,
        selection=LatestResume(),
    )
    assert (step, restored) == (0, first)
    store = await open_store(config)
    try:
        assert await store.policy(2) is None  # Missing policy metadata is imported abandoned.
        rows = await store.select(min_version=1, max_version=2, exclude=["consumed"], limit=10)
        assert [row.group_id for row in rows] == ["after"]
        assert not (config.artifact_directory / config.run_id / "groups/consumed.bin").exists()
        with psycopg.connect(dsn) as db:
            assert (
                db.execute("SELECT group_id, created_at FROM proximal_rollout_groups ORDER BY group_id").fetchall()
                == times
            )
        checkpoints.reconcile_groups(snapshot_root=root, artifacts=config.artifact_directory, dsn=dsn, context=context)
        assert await store.count(min_version=1, max_version=2, exclude=["consumed"]) == 1
        # Existing publisher alone may revive identical future weights.
        await store.commit_policy(versioned(policy, 2))
        assert await store.count(min_version=1, max_version=2, exclude=["consumed"]) == 2
    finally:
        await store.close()


async def test_lost_commit_ack_does_not_index_or_repeat_paid_work(config, tmp_path, store_dsn, attempt, policy):
    store = await run_state_store(config)
    task = asyncio.create_task(store.add_group("g", policy, entry(attempt, policy).group))
    root = tmp_path / "volume"
    commits = []

    def lost_ack():
        commits.append(1)
        assert (root / "artifacts" / config.run_id / "groups/g.bin").exists()
        assert not (root / "artifacts" / config.run_id / "groups/g.json").exists()
        raise OSError("commit succeeded, acknowledgement lost")

    try:
        for _ in range(100):
            await asyncio.sleep(0.01)
            try:
                count = state_artifacts.publish_pending(
                    dsn=store_dsn,
                    run_id=config.run_id,
                    artifacts=config.artifact_directory,
                    snapshot_root=root,
                    commit=lost_ack,
                )
            except OSError:
                break
            assert count == 0
        assert commits and not task.done()
        with psycopg.connect(store_dsn) as db:
            assert db.execute("SELECT count(*) FROM proximal_rollout_groups").fetchone()[0] == 0
        await settle(task, config=config, store_dsn=store_dsn, root=root)
        with psycopg.connect(store_dsn) as db:
            assert db.execute("SELECT count(*) FROM proximal_rollout_groups").fetchone()[0] == 1
        # Idempotent retry keeps the immutable ordering record too.
        await store.add_group("g", policy, entry(attempt, policy).group)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()


@pytest.mark.parametrize("fail_at", [1, 2, 3, None])
async def test_checkpoint_commit_order_and_failure(config, context, tmp_path, store, store_dsn, pg_bin, fail_at):
    root, working, durable = tmp_path / "mount", tmp_path / "local", tmp_path / "durable"
    native_save(working)
    calls = 0

    def commit():
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("lost mount before commit")
        shutil.rmtree(durable, ignore_errors=True)
        shutil.copytree(root, durable)
        if calls == 1:
            assert not list(durable.rglob("manifest.json"))
        if calls == 2:
            assert list(durable.rglob("manifest.json")) and not (durable / "LATEST").exists()

    kwargs = dict(
        checkpoints=working,
        dsn=store_dsn,
        snapshot_root=root,
        pg_bin=pg_bin,
        context=context,
        launch_id="one",
        parent=None,
        commit=commit,
    )
    if fail_at:
        with pytest.raises(OSError):
            checkpoints.take(3, **kwargs)
        assert checkpoints.resolve(durable, AutoResume()) is None
    else:
        identity = checkpoints.take(3, **kwargs)
        assert checkpoints.resolve(durable, LatestResume()) == identity
        assert checkpoints.read_manifest(durable, identity).step == 3


async def test_retraining_same_step_keeps_both_immutable_identities_and_pins(
    context,
    tmp_path,
    store,
    store_dsn,
    pg_bin,
):
    root, working = tmp_path / "volume", tmp_path / "local"
    kwargs = dict(
        checkpoints=working,
        dsn=store_dsn,
        snapshot_root=root,
        pg_bin=pg_bin,
        context=context,
        launch_id="one",
        commit=lambda: None,
    )
    native_save(working)
    first = checkpoints.take(3, parent=None, **kwargs)
    old = (root / "checkpoints" / first / "checkpoint/adapter/adapter_megatron_rank0.pt").read_bytes()
    native_save(working, consumed=(("different", 1),))
    second = checkpoints.take(3, parent=first, **kwargs)
    assert first != second
    assert checkpoints.read_manifest(root, first).step == 3
    assert (root / "checkpoints" / first / "checkpoint/adapter/adapter_megatron_rank0.pt").read_bytes() == old
    (root / "pins").mkdir()
    (root / "pins" / first).touch()
    native_save(working, step=4)
    third = checkpoints.take(4, parent=second, **kwargs)
    checkpoints.prune(root, latest=third, commit=lambda: None)
    assert {p.name for p in (root / "checkpoints").iterdir()} == {first, second, third}
    (root / "pins" / first).unlink()
    checkpoints.prune(root, latest=third, commit=lambda: None)
    assert {p.name for p in (root / "checkpoints").iterdir()} == {second, third}


@pytest.mark.parametrize("missing", ["adapter_megatron_rank1.pt", "training_state_rank1.pt", "native_checkpoint.json"])
def test_partial_native_save_never_counts_as_complete(tmp_path, missing):
    native_save(tmp_path)
    assert checkpoints.complete_steps(tmp_path) == [3]
    (checkpoints.iter_dir(tmp_path, 3) / "adapter" / missing).unlink()
    assert checkpoints.complete_steps(tmp_path) == []


async def test_corrupt_checkpoint_is_rejected_before_database_restore(context, tmp_path, store, store_dsn, pg_bin):
    root, working = tmp_path / "volume", tmp_path / "local"
    native_save(working)
    identity = checkpoints.take(
        3,
        checkpoints=working,
        dsn=store_dsn,
        snapshot_root=root,
        pg_bin=pg_bin,
        context=context,
        launch_id="one",
        parent=None,
        commit=lambda: None,
    )
    target = root / "checkpoints" / identity / "checkpoint/adapter/training_state_rank1.pt"
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.read_manifest(root, identity)


def test_explicit_resume_never_silently_starts_fresh(tmp_path, context):
    with pytest.raises(FileNotFoundError):
        checkpoints.resolve(tmp_path, LatestResume())
    changed = context.model_copy(update={"train_args": ("--lr", "0.00002")})
    with pytest.raises(ValueError, match="--lr"):
        checkpoints.check_context(context, changed, AutoResume())
    checkpoints.check_context(context, changed, AutoResume(allow_changes=("--lr",)))
    with pytest.raises(ValueError, match="world_size"):
        checkpoints.check_context(context, context.model_copy(update={"world_size": 4}), AutoResume())


async def test_final_flush_publishes_last_checkpoint_without_waiting_for_timer(
    config,
    context,
    tmp_path,
    store,
    store_dsn,
    pg_bin,
):
    root, working = tmp_path / "volume", tmp_path / "local"
    native_save(working)
    writer = StateWriter(
        dsn=store_dsn,
        pg_bin=pg_bin,
        checkpoints=working,
        artifacts=config.artifact_directory,
        snapshot_root=root,
        context=context,
        launch_id="one",
        parent=None,
        taken=None,
        commit=lambda: None,
    )
    writer.close()
    assert writer.taken == 3
    assert checkpoints.read_manifest(root, (root / "LATEST").read_text()).step == 3


def test_existing_partial_artifact_is_never_silently_skipped(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"complete")
    target.write_bytes(b"partial")
    file = state_artifacts.describe(source, relative="group.bin")
    with pytest.raises(ValueError, match="checksum"):
        state_artifacts.copy_verified(source, target, file)


async def test_completed_group_before_first_checkpoint_survives_restart(
    config,
    context,
    tmp_path,
    store_dsn,
    attempt,
    policy,
    empty_database,
    pg_bin,
):
    store = await run_state_store(config)
    root = tmp_path / "volume"
    await settle(
        asyncio.create_task(store.add_group("first", policy, entry(attempt, policy, group="first").group)),
        config=config,
        store_dsn=store_dsn,
        root=root,
    )
    await store.close()
    shutil.rmtree(config.artifact_directory)
    dsn = empty_database()
    assert checkpoints.restore(
        snapshot_root=root,
        checkpoints=tmp_path / "checkpoints",
        artifacts=config.artifact_directory,
        dsn=dsn,
        pg_bin=pg_bin,
        context=context,
        selection=AutoResume(),
    ) == (None, None)
    store = await open_store(config)
    try:
        assert await store.count(min_version=1, max_version=1, exclude=[]) == 0
        await store.commit_policy(policy)
        assert await store.count(min_version=1, max_version=1, exclude=[]) == 1
    finally:
        await store.close()


@pytest.mark.parametrize("failure", ["none", "cancel", "local_write"])
async def test_capture_release_waits_for_durable_handoff(
    config,
    tmp_path,
    store_dsn,
    attempt,
    policy,
    failure,
    monkeypatch,
):
    from tests.integration.proximal_async.test_buffer import sample_for

    from miles.rollout.session.samples.codec import encode_samples
    from miles.utils.types import Sample
    from miles_plugins.proximal.contracts import SessionHandle
    from miles_plugins.proximal.rollout import execute_attempt, wait_for_releases

    sample, evidence = sample_for(attempt)
    sample.index, sample.group_index = 0, 0
    payload = encode_samples([sample], {})
    receipt = evidence.capture.model_copy(update={"payload_sha256": hashlib.sha256(payload).hexdigest()})
    handle = SessionHandle(
        session_id="a" * 32, rollout_id="r", base_url="http://localhost:1/v1", request_sha256=digest(attempt)
    )
    released, calls = [], []

    class Capture:
        async def create(self, _):
            return handle

        async def collect(self, *_):
            return receipt, payload

        async def release(self, _):
            released.append(True)

    class Platform:
        async def execute(self, *_):
            calls.append(True)
            return evidence.grade

        async def cancel(self, _):
            raise AssertionError("Do not cancel completed paid work on storage failure")

    if failure == "local_write":
        from miles_plugins.proximal import rollout

        def fail_write(*_):
            raise OSError("local disk unavailable")

        monkeypatch.setattr(rollout, "write_immutable", fail_write)
    store = await run_state_store(config)
    task = asyncio.create_task(
        execute_attempt(
            attempt,
            Sample(index=0, group_index=0),
            capture=Capture(),
            platform=Platform(),
            artifact_root=config.artifact_directory / config.run_id / "accepted",
            store=store,
        )
    )
    try:
        if failure == "local_write":
            with pytest.raises(OSError, match="local disk"):
                await task
            await wait_for_releases()
            assert calls == [True] and not released
            return
        for _ in range(100):
            if (
                config.artifact_directory / config.run_id / "accepted" / attempt.attempt_id / "accepted.json"
            ).exists():
                break
            await asyncio.sleep(0.01)
        assert calls == [True] and not released and not task.done()
        if failure == "cancel":
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert not released
        else:
            await settle(task, config=config, store_dsn=store_dsn, root=tmp_path / "volume")
            await wait_for_releases()
            assert released == [True] and calls == [True]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()


async def test_run_namespaces_do_not_share_groups_or_delete_each_other(
    config,
    context,
    tmp_path,
    store_dsn,
    attempt,
    policy,
    pg_bin,
):
    first = await run_state_store(config)
    other_config = config.model_copy(update={"run_id": "other-run"})
    second = await run_state_store(other_config)
    second_policy = policy.model_copy(update={"run_id": "other-run"})
    try:
        for store, cfg, p in ((first, config, policy), (second, other_config, second_policy)):
            await settle(
                asyncio.create_task(
                    store.add_group(
                        "same-id",
                        p,
                        entry(
                            attempt.model_copy(update={"run_id": cfg.run_id, "policy": p}), p, group="same-id"
                        ).group,
                    )
                ),
                config=cfg,
                store_dsn=store_dsn,
                root=tmp_path / cfg.run_id,
            )
        assert (tmp_path / "other-run/artifacts/other-run/groups/same-id.bin").exists()
        assert not (tmp_path / "test-run/artifacts/other-run").exists()
    finally:
        await first.close()
        await second.close()


def test_resume_cannot_authorize_a_different_parallel_layout(context):
    changed = context.model_copy(update={"train_args": (*context.train_args, "--expert-model-parallel-size", "2")})
    with pytest.raises(ValueError, match="parallel"):
        checkpoints.check_context(context, changed, AutoResume(allow_changes=("--expert-model-parallel-size",)))


async def test_checkpoint_retention_uses_publication_order_after_rewind(context, tmp_path, store, store_dsn, pg_bin):
    root, working = tmp_path / "volume", tmp_path / "local"
    kwargs = dict(
        checkpoints=working,
        dsn=store_dsn,
        snapshot_root=root,
        pg_bin=pg_bin,
        context=context,
        launch_id="one",
        commit=lambda: None,
    )
    native_save(working, step=3)
    first = checkpoints.take(3, parent=None, **kwargs)
    native_save(working, step=20)
    second = checkpoints.take(20, parent=first, **kwargs)
    native_save(working, step=4)
    # Rewind to step 3; the two most recent publications are step 20 and the new 4.
    third = checkpoints.take(4, parent=first, **kwargs)
    checkpoints.prune(root, latest=third, commit=lambda: None)
    assert {path.name for path in (root / "checkpoints").iterdir()} == {second, third}
    assert checkpoints.read_manifest(root, third).parent == first


async def test_reconciliation_rejects_index_that_changes_payload_lineage(
    config,
    context,
    tmp_path,
    store_dsn,
    attempt,
    policy,
):
    import json

    store = await run_state_store(config)
    root = tmp_path / "volume"
    try:
        await settle(
            asyncio.create_task(store.add_group("g", policy, entry(attempt, policy).group)),
            config=config,
            store_dsn=store_dsn,
            root=root,
        )
        index = root / "artifacts" / config.run_id / "groups/g.json"
        value = json.loads(index.read_bytes())
        value["header"]["policy"]["snapshot"]["sha256"] = "9" * 64
        index.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="payload header"):
            checkpoints.reconcile_groups(
                snapshot_root=root, artifacts=config.artifact_directory, dsn=store_dsn, context=context
            )
    finally:
        await store.close()


async def test_release_drain_removes_already_completed_tasks():
    from miles_plugins.proximal.rollout import _releases, wait_for_releases

    task = asyncio.create_task(asyncio.sleep(0))
    await task
    _releases.add(task)  # Completion callback did not get another loop turn.
    await wait_for_releases()
    assert task not in _releases


def test_resume_cannot_expand_a_pruned_consumption_window(context):
    changed = context.model_copy(update={"max_policy_lag": context.max_policy_lag + 1})
    with pytest.raises(ValueError, match="max_policy_lag"):
        checkpoints.check_context(context, changed, AutoResume())
