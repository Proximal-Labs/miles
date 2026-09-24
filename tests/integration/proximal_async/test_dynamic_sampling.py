"""Real Postgres and Miles producer regressions for durable dynamic sampling."""

import argparse
import asyncio
import sys
from argparse import Namespace

import pytest
from tests.integration.proximal_async.test_buffer import entry, make_buffer

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.function_registry import function_registry
from miles_plugins.proximal.contracts import AcceptedAttempt
from miles_plugins.proximal.data_source import ConsumptionLedger, PlatformTaskSource
from miles_plugins.proximal.options import validate_args
from miles_plugins.proximal.rollout import PlatformRolloutFn
from miles_plugins.proximal.runtime import training_argv
from miles_plugins.proximal.store import open_store

NONZERO_STD = "miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter"


async def rows(store, filter_path):
    return await store.select(min_version=1, max_version=1, exclude=[], limit=100, filter_path=filter_path)


async def count(store, filter_path):
    return await store.count(min_version=1, max_version=1, exclude=[], filter_path=filter_path)


async def test_zero_variance_is_retained_but_does_not_fill_capacity_or_train(config, tmp_path, attempt, policy, store):
    await store.commit_policy(policy)
    unused = []
    ledger = ConsumptionLedger()
    buffer = make_buffer(config, tmp_path, store, ledger, unused, filter_path=NONZERO_STD)
    # More rejected groups than the capacity and execution failure budget combined.
    for i in range(6):
        reward = float(i % 2)
        await asyncio.wait_for(buffer.put(entry(attempt, policy, group=f"drop-{i}", rewards=(reward, reward))), 2)
    assert await count(store, NONZERO_STD) == 0
    assert await count(store, None) == 6
    assert unused == [] and ledger.snapshot() == ()
    for row in await rows(store, None):
        _, samples = await store.load(row)
        assert len(samples) == 2 and samples[0].reward == samples[1].reward

    await buffer.put(entry(attempt, policy, group="mixed", rewards=(0.0, 1.0)))
    got = await asyncio.wait_for(buffer.get(current_version=1), 2)
    assert [sample.reward for sample in got.group] == [0.0, 1.0]
    assert [group.group_id for group in ledger.snapshot()] == ["mixed"]
    metrics = buffer.get_metrics()
    assert metrics["rollout/platform/persisted_groups"] == 7
    assert metrics["rollout/platform/consumed_groups"] == 1
    assert metrics["rollout/platform/consecutive_failed_groups"] == 0
    assert metrics["rollout/platform/dynamic_filtered_groups"] == 6
    assert metrics["rollout/dynamic_filter/drop_zero_std_0.0"] == 3
    assert metrics["rollout/dynamic_filter/drop_zero_std_1.0"] == 3
    assert buffer.get_metrics()["rollout/platform/dynamic_filtered_groups"] == 0
    assert not any("drop_" in name for name in buffer.get_metrics())


async def test_rejections_survive_connection_restart_and_checkpoint_rollback(config, tmp_path, attempt, policy, store):
    await store.commit_policy(policy)
    buffer = make_buffer(config, tmp_path, store, ConsumptionLedger(), filter_path=NONZERO_STD)
    await buffer.put(entry(attempt, policy, group="rejected"))
    await buffer.put(entry(attempt, policy, group="kept", rewards=(0.0, 1.0)))
    await buffer.get(current_version=1)

    reopened = await open_store(config)
    try:
        # Roll back all consumption, as if resuming a checkpoint before that step.
        ledger = ConsumptionLedger()
        resumed = make_buffer(config, tmp_path, reopened, ledger, filter_path=NONZERO_STD)
        got = await asyncio.wait_for(resumed.get(current_version=1), 2)
        assert (
            AcceptedAttempt.model_validate_json(got.group[0].metadata["proximal_accepted"]).attempt.group_id == "kept"
        )
        assert [row.group_id for row in await rows(reopened, NONZERO_STD)] == ["kept"]
        assert len(await rows(reopened, None)) == 2
        assert resumed.get_metrics()["rollout/platform/dynamic_filtered_groups"] == 0
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(resumed.get(current_version=1), 0.1)
    finally:
        await reopened.close()


async def test_unclassified_backlog_is_filtered_and_releases_backpressure(config, tmp_path, attempt, policy, store):
    await store.commit_policy(policy)
    # Old schema rows, or a put interrupted after the payload/index was written.
    for i in range(3):
        await store.add_group(f"old-{i}", policy, entry(attempt, policy, group=f"old-{i}").group)
    buffer = make_buffer(config, tmp_path, store, ConsumptionLedger(), filter_path=NONZERO_STD)
    blocked = asyncio.create_task(buffer.put(entry(attempt, policy, group="kept", rewards=(0.0, 1.0))))
    try:
        await asyncio.sleep(0.1)
        assert not blocked.done()
        got = await asyncio.wait_for(buffer.get(current_version=1), 2)
        assert [sample.reward for sample in got.group] == [0.0, 1.0]
        await asyncio.wait_for(blocked, 2)
        assert await count(store, NONZERO_STD) == 1  # Consumption is tracked separately.
        assert buffer.get_metrics()["rollout/platform/dynamic_filtered_groups"] == 3
    finally:
        blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)


async def test_legacy_bool_and_filter_scoped_rejections(config, tmp_path, attempt, policy, store):
    await store.commit_policy(policy)
    with function_registry.temporary("test.reject", lambda args, samples: False):
        rejected = make_buffer(config, tmp_path, store, ConsumptionLedger(), filter_path="test.reject")
        await rejected.put(entry(attempt, policy))
    assert await count(store, "test.reject") == 0
    assert rejected.get_metrics()["rollout/platform/dynamic_filtered_groups"] == 1
    with function_registry.temporary("test.keep", lambda args, samples: True):
        kept = make_buffer(config, tmp_path, store, ConsumptionLedger(), filter_path="test.keep")
        assert len((await asyncio.wait_for(kept.get(current_version=1), 2)).group) == 2


async def test_filter_errors_fail_closed_after_preserving_paid_group(config, tmp_path, attempt, policy, store):
    def broken(args, samples):
        raise ValueError("broken research filter")

    await store.commit_policy(policy)
    ledger = ConsumptionLedger()
    with function_registry.temporary("test.broken", broken):
        buffer = make_buffer(config, tmp_path, store, ledger, filter_path="test.broken")
        with pytest.raises(ValueError, match="broken research filter"):
            await buffer.put(entry(attempt, policy))
        assert await count(store, None) == 1
        with pytest.raises(ValueError, match="broken research filter"):
            await buffer.get(current_version=1)
    assert ledger.snapshot() == ()


async def test_consumer_racing_put_records_one_rejection(config, tmp_path, attempt, policy, store, monkeypatch):
    await store.commit_policy(policy)
    buffer = make_buffer(config, tmp_path, store, ConsumptionLedger(), filter_path=NONZERO_STD)
    inserted, rejected, release_put = asyncio.Event(), asyncio.Event(), asyncio.Event()
    add_group, reject_group = store.add_group, store.reject_group

    async def paused_add(*args, **kwargs):
        await add_group(*args, **kwargs)
        inserted.set()
        await release_put.wait()

    async def observed_reject(*args, **kwargs):
        first = await reject_group(*args, **kwargs)
        rejected.set()
        return first

    monkeypatch.setattr(store, "add_group", paused_add)
    monkeypatch.setattr(store, "reject_group", observed_reject)
    put = asyncio.create_task(buffer.put(entry(attempt, policy)))
    get = None
    try:
        await asyncio.wait_for(inserted.wait(), 2)
        get = asyncio.create_task(buffer.get(current_version=1))
        await asyncio.wait_for(rejected.wait(), 2)
        release_put.set()
        await asyncio.wait_for(put, 2)
        assert buffer.get_metrics()["rollout/platform/dynamic_filtered_groups"] == 1
        assert await count(store, NONZERO_STD) == 0
    finally:
        tasks = [put] + ([] if get is None else [get])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_filter_flag_passes_actual_parser_and_rejects_invalid_hooks(config, tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    argv = training_argv(str(path)) + [
        "--proximal-yes-rollouts",
        "--proximal-yes-publish",
        "--rollout-batch-size",
        "1",
        "--dynamic-sampling-filter-path",
        NONZERO_STD,
    ]
    monkeypatch.setattr(sys, "argv", ["train_async.py", *argv])
    args = get_miles_extra_args_provider()(argparse.ArgumentParser()).parse_args(argv)
    validate_args(args)
    args.reward_key = "reward"
    with pytest.raises(ValueError, match="scalar"):
        validate_args(args)
    args.reward_key = None

    async def async_filter(args, samples):
        return True

    for name, hook in (("test.async", async_filter), ("test.not_callable", 7)):
        with function_registry.temporary(name, hook):
            args.dynamic_sampling_filter_path = name
            with pytest.raises(ValueError, match="synchronous|callable"):
                validate_args(args)


async def test_real_async_worker_replenishes_filtered_groups_into_full_batch(config, tmp_path, attempt, policy, store):
    await store.commit_policy(policy)
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    args = Namespace(
        proximal_config=str(path),
        rollout_submission_granularity="group",
        n_samples_per_prompt=2,
        async_unused_samples_handler="retry",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path=NONZERO_STD,
        reward_key=None,
        rollout_batch_size=2,
        rollout_global_dataset=True,
        async_max_concurrent_samples=4,
        custom_async_data_buffer_path="miles_plugins.proximal.buffer.PlatformDataBuffer",
        save=None,
        load=None,
        proximal_yes_rollouts=True,
        proximal_yes_publish=True,
    )
    generated = 0
    gate = asyncio.Event()

    class Producer(PlatformRolloutFn):
        async def _generate_group(self, prompt_group):
            nonlocal generated
            index = generated
            generated += 1
            if index >= 5:
                await gate.wait()
            rewards = (0.0, 0.0) if index < 3 else (0.0, 1.0)
            return entry(attempt, policy, group=f"group-{index}", group_index=index, rewards=rewards)

    source = PlatformTaskSource(args)
    producer = Producer(RolloutFnConstructorInput(args=args, data_source=source))
    try:
        result = await asyncio.wait_for(producer(RolloutFnTrainInput(rollout_id=0, weight_version=1)), 5)
        assert len(result.samples) == 2
        assert all([sample.reward for sample in group] == [0.0, 1.0] for group in result.samples)
        assert {group.group_id for group in source.consumed.snapshot()} == {"group-3", "group-4"}
        assert source.get_buffer_length() == 0  # No immediate prompt retry for research filtering.
        assert result.metrics["rollout/platform/dynamic_filtered_groups"] == 3
        assert await count(store, None) == 5
        assert await count(store, NONZERO_STD) == 2
    finally:
        await producer.close()
