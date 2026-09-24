import asyncio
import os
from argparse import Namespace
from pathlib import Path

import pytest

from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall
from miles_plugins.proximal.buffer import PlatformDataBuffer, validate_sample
from miles_plugins.proximal.contracts import AcceptedAttempt, CaptureReceipt, Grade, digest
from miles_plugins.proximal.data_source import ConsumptionLedger
from miles_plugins.proximal.snapshot import SnapshotReference
from miles_plugins.proximal.store import PayloadSync, PolicyConflict, open_store


def sample_for(attempt):
    proof = AcceptedAttempt(
        attempt=attempt,
        capture=CaptureReceipt(
            session_id="a" * 32,
            request_sha256=digest(attempt),
            policy=attempt.policy,
            payload_sha256="e" * 64,
            num_calls=1,
            num_tokens=3,
        ),
        grade=Grade(
            run_id=attempt.attempt_id,
            rollout_id="rollout-1",
            request_sha256=digest(attempt),
            reward=0.0,
            status="success",
            artifacts_url=None,
        ),
    )
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        response="ok",
        reward=0.0,
        status=Sample.Status.COMPLETED,
        loss_mask=[1, 1],
        rollout_log_probs=[-0.2, -0.4],
        weight_versions=[
            WeightVersionsPerCall(
                spans=[WeightVersionSpan(version=str(attempt.policy.version), abs_start=1, abs_end=3)]
            )
        ],
        metadata={"proximal_accepted": proof.model_dump_json()},
    )
    return sample, proof


def versioned(policy, version, sha="d"):
    return policy.model_copy(update={"version": version, "snapshot": SnapshotReference(sha256=sha * 64)})


def entry(attempt, policy, group="g", group_index=0):
    samples = []
    for i in range(2):
        sample, _ = sample_for(
            attempt.model_copy(
                update={
                    "sample_index": i,
                    "attempt_id": f"{group}-{i}",
                    "group_id": group,
                    "policy": policy,
                }
            )
        )
        sample.index, sample.group_index = group_index * 2 + i, group_index
        samples.append(sample)
    return DataBufferInput(prompt_group=samples, group=samples)


def make_buffer(config, tmp_path, store, ledger, unused=None):
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    buffer = PlatformDataBuffer(
        DataBufferConstructorInput(
            args=Namespace(proximal_config=str(path)),
            unused_handler_fn=(unused if unused is not None else []).append,
        )
    )
    buffer.attach(store=store, ledger=ledger)
    return buffer


async def test_batch_query_selects_fresh_groups_oldest_first(config, tmp_path, attempt, policy, store):
    for version in (1, 2, 3):
        await store.commit_policy(versioned(policy, version))
    buffer = make_buffer(config, tmp_path, store, ConsumptionLedger())
    await buffer.put(entry(attempt, versioned(policy, 1), group="stale"))
    await buffer.put(entry(attempt, versioned(policy, 2), group="older"))
    await buffer.put(entry(attempt, versioned(policy, 3), group="newer"))
    # max_policy_lag is 1: at version 3 only versions 2..3 are eligible.
    first = await buffer.get(current_version=3)
    second = await buffer.get(current_version=3)
    groups = [
        AcceptedAttempt.model_validate_json(x.group[0].metadata["proximal_accepted"]).attempt.group_id
        for x in (first, second)
    ]
    assert groups == ["older", "newer"]
    assert first.group[0].tokens == [1, 2, 3] and first.group[0].rollout_log_probs == [-0.2, -0.4]
    # The learning signal survives storage (a scored zero, not an absent reward).
    assert [sample.reward for sample in first.group] == [0.0, 0.0]
    assert first.group[0].index == 0 and first.group[0].group_index == 0
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(buffer.get(current_version=3), 0.2)
    metrics = buffer.get_metrics()
    assert metrics["rollout/platform/persisted_groups"] == 3
    assert metrics["rollout/platform/consumed_groups"] == 2


async def test_consumption_follows_the_checkpoint_across_restart(config, tmp_path, attempt, policy, store):
    await store.commit_policy(policy)
    ledger = ConsumptionLedger()
    buffer = make_buffer(config, tmp_path, store, ledger)
    await buffer.put(entry(attempt, policy, group="a"))
    await buffer.put(entry(attempt, policy, group="b"))
    await buffer.get(current_version=1)
    saved = ledger.snapshot()
    assert [c.group_id for c in saved] == ["a"]

    # A new process with a new connection sees the same store.
    reopened = await open_store(config)
    try:
        restored = ConsumptionLedger()
        restored.restore(saved)
        resumed = make_buffer(config, tmp_path, reopened, restored)
        got = await resumed.get(current_version=1)
        assert got.group[0].metadata["proximal_accepted"].count('"group_id":"b"') == 1
        # Resuming an earlier checkpoint forgets later consumption: "a" is selectable again.
        earlier = make_buffer(config, tmp_path, reopened, ConsumptionLedger())
        again = await earlier.get(current_version=1)
        assert again.group[0].metadata["proximal_accepted"].count('"group_id":"a"') == 1
    finally:
        await reopened.close()


async def test_abandoned_weights_are_never_selected(config, tmp_path, attempt, policy, store):
    await store.commit_policy(versioned(policy, 1))
    lost = versioned(policy, 2, sha="a")
    await store.commit_policy(lost)
    buffer = make_buffer(config, tmp_path, store, ConsumptionLedger())
    await buffer.put(entry(attempt, lost, group="from-lost-weights"))
    # Resume from the checkpoint behind version 1; version 2 is republished with new weights.
    assert await store.rewind(keep_through=1) == 1
    replacement = versioned(policy, 2, sha="b")
    await store.commit_policy(replacement)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(buffer.get(current_version=2), 0.2)
    await buffer.put(entry(attempt, replacement, group="from-replacement"))
    got = await buffer.get(current_version=2)
    assert '"group_id":"from-replacement"' in got.group[0].metadata["proximal_accepted"]


async def test_policy_versions_are_immutable_and_sequential(policy, store):
    await store.commit_policy(policy)
    await store.commit_policy(policy)  # Identical weights: idempotent.
    with pytest.raises(PolicyConflict):
        await store.commit_policy(versioned(policy, 1, sha="f"))
    with pytest.raises(PolicyConflict):
        await store.commit_policy(versioned(policy, 3))
    assert await store.current_policy() == policy


async def test_backpressure_and_failure_budget(config, tmp_path, attempt, policy, store):
    await store.commit_policy(policy)
    unused = []
    buffer = make_buffer(config, tmp_path, store, ConsumptionLedger(), unused)
    await buffer.put(entry(attempt, policy, group="one"))
    await buffer.put(entry(attempt, policy, group="two"))
    blocked = asyncio.create_task(buffer.put(entry(attempt, policy, group="three")))
    await asyncio.sleep(0.1)
    # completed_group_capacity is 2: the third put stored its group, then waits.
    assert not blocked.done()
    await buffer.get(current_version=1)
    await asyncio.wait_for(blocked, 2)
    aborted = DataBufferInput(prompt_group=[], group=[Sample(status=Sample.Status.ABORTED) for _ in range(2)])
    await buffer.put(aborted)
    assert len(unused) == 1
    with pytest.raises(RuntimeError, match="failure budget"):
        await buffer.put(aborted)


@pytest.mark.parametrize("bad", ["logprobs", "mask", "policy", "grade"])
def test_missing_training_evidence_fails_closed(attempt, bad):
    sample, proof = sample_for(attempt)
    if bad == "logprobs":
        sample.rollout_log_probs = [float("nan"), -1]
    elif bad == "mask":
        sample.loss_mask = [0, 0]
    elif bad == "policy":
        sample.weight_versions = []
    else:
        sample.reward = 1.0
    with pytest.raises(ValueError):
        validate_sample(sample, proof)


async def test_group_cannot_mix_versions_or_duplicate_members(config, tmp_path, attempt, policy, store):
    buffer = make_buffer(config, tmp_path, store, ConsumptionLedger())
    mixed = entry(attempt, policy)
    mixed.group[1] = entry(attempt, versioned(policy, 2)).group[1]
    with pytest.raises(ValueError, match="mixes"):
        await buffer.put(mixed)
    repeated = entry(attempt, policy)
    repeated.group[1] = repeated.group[0]
    with pytest.raises(ValueError, match="Repeated"):
        await buffer.put(repeated)
    with pytest.raises(ValueError, match="committed"):
        await buffer.get()


async def test_a_different_training_contract_never_consumes_the_group(config, tmp_path, attempt, policy, store):
    await store.commit_policy(policy)
    await make_buffer(config, tmp_path, store, ConsumptionLedger()).put(entry(attempt, policy, group="g"))
    other = config.model_copy(update={"harness": config.harness.model_copy(update={"revision": "e" * 40})})
    reader = await open_store(other)
    try:
        (tmp_path / "other").mkdir()
        buffer = make_buffer(other, tmp_path / "other", reader, ConsumptionLedger())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(buffer.get(current_version=1), 0.2)
    finally:
        await reader.close()


async def test_payload_is_committed_before_indexing_and_reloaded_on_miss(config, attempt, policy):
    import psycopg

    events = []
    hidden = config.artifact_directory / "hidden.bin"

    def commit():
        with psycopg.connect(os.environ[config.store_dsn_env]) as check:
            rows = check.execute("SELECT count(*) FROM proximal_rollout_groups").fetchone()
        events.append(("commit", rows[0]))

    def reload():
        events.append(("reload",))
        hidden.rename(target)  # This container's view catches up with the writer's commit.

    from miles_plugins.proximal.contracts import digest, training_contract
    from miles_plugins.proximal.store import RolloutStore

    store = await RolloutStore.open(
        os.environ[config.store_dsn_env],
        run_id=config.run_id,
        contract_sha256=digest(training_contract(config)),
        root=config.artifact_directory,
        sync=PayloadSync(commit=commit, reload=reload),
    )
    try:
        await store.commit_policy(policy)
        await store.add_group("g", policy, entry(attempt, policy, group="g").group)
        assert events == [("commit", 0)]  # The payload is committed while no row exists yet.
        [row] = await store.select(min_version=1, max_version=1, exclude=[], limit=1)
        target = Path(row.payload_path)
        target.rename(hidden)  # A reader whose mount view predates the commit.
        _, samples = await store.load(row)
        assert events[-1] == ("reload",) and len(samples) == 2
    finally:
        await store.close()


def test_every_group_member_is_checked_against_the_contract(config, attempt, policy):
    from miles_plugins.proximal.buffer import validate_group

    group = entry(attempt, policy)
    assert validate_group(config, group.group) == policy
    # A second member recorded under a different harness revision.
    other = attempt.model_copy(
        update={
            "sample_index": 1,
            "attempt_id": "g-1",
            "group_id": "g",
            "policy": policy,
            "harness": attempt.harness.model_copy(update={"revision": "e" * 40}),
        }
    )
    stray, _ = sample_for(other)
    stray.index, stray.group_index = 1, 0
    group.group[1] = stray
    with pytest.raises(ValueError, match="mixes|different training contract"):
        validate_group(config, group.group)
