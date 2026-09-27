import argparse
import asyncio
import sys
from argparse import Namespace
from dataclasses import replace

import pytest
from tests.integration.proximal_async.test_buffer import entry

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.utils.arguments import get_miles_extra_args_provider, resolve_rollout_function_paths
from miles.utils.types import Sample
from miles_plugins.proximal.clients import IneligibleAttempt
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.options import ROLLOUT, validate_args
from miles_plugins.proximal.rollout import PlatformRolloutFn
from miles_plugins.proximal.runtime import training_argv


def test_runtime_arguments_use_actual_miles_parser(config, tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    argv = training_argv(str(path)) + [
        "--proximal-yes-rollouts",
        "--proximal-yes-publish",
        "--rollout-batch-size",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", ["train_async.py", *argv])
    parser = get_miles_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    validate_args(args)
    assert resolve_rollout_function_paths(args)[0] == ROLLOUT
    args.use_rollout_logprobs = False
    with pytest.raises(ValueError, match="use-rollout-logprobs"):
        validate_args(args)
    args.use_rollout_logprobs = True
    args.target_modules = "linear_qkv"
    with pytest.raises(ValueError, match="target-modules"):
        validate_args(args)


def test_truncated_importance_sampling_selects_miles_tis(config, tmp_path, monkeypatch):
    from miles_plugins.proximal.contracts import TruncatedImportanceSampling, read_run_config

    correction = TruncatedImportanceSampling(kind="truncated_importance_sampling", clip=2.0, clip_low=0.0)
    tis = config.model_copy(
        update={"research": config.research.model_copy(update={"behavior_correction": correction})}
    )
    path = tmp_path / "run.json"
    path.write_text(tis.model_dump_json())
    assert read_run_config(path).research.behavior_correction == correction
    argv = training_argv(str(path)) + [
        "--proximal-yes-rollouts",
        "--proximal-yes-publish",
        "--rollout-batch-size",
        "1",
    ]
    assert "--use-tis" in argv and "--use-rollout-logprobs" not in argv
    monkeypatch.setattr(sys, "argv", ["train_async.py", *argv])
    args = get_miles_extra_args_provider()(argparse.ArgumentParser()).parse_args(argv)
    validate_args(args)
    assert args.use_tis and not args.use_rollout_logprobs and (args.tis_clip, args.tis_clip_low) == (2.0, 0.0)
    args.use_rollout_logprobs = True
    with pytest.raises(ValueError, match="use-rollout-logprobs"):
        validate_args(args)


def test_truncated_importance_sampling_bounds_are_explicit_and_valid():
    from pydantic import ValidationError

    from miles_plugins.proximal.contracts import TruncatedImportanceSampling

    for clip, clip_low in ((1.0, 0.0), (2.0, 1.0), (2.0, -0.1)):
        with pytest.raises(ValidationError):
            TruncatedImportanceSampling(kind="truncated_importance_sampling", clip=clip, clip_low=clip_low)
    with pytest.raises(ValidationError):
        TruncatedImportanceSampling.model_validate({"kind": "truncated_importance_sampling", "clip": 2.0})


async def test_existing_async_worker_overlaps_consumption_and_cancels_children(
    config, tmp_path, attempt, policy, store
):
    await store.commit_policy(policy)
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    args = Namespace(
        proximal_config=str(path),
        rollout_submission_granularity="group",
        n_samples_per_prompt=2,
        async_unused_samples_handler="drop",
        rollout_sample_filter_path=None,
        rollout_batch_size=1,
        rollout_global_dataset=True,
        async_max_concurrent_samples=4,
        custom_async_data_buffer_path="miles_plugins.proximal.buffer.PlatformDataBuffer",
        save=None,
        load=None,
        proximal_yes_rollouts=True,
        proximal_yes_publish=True,
    )
    started = 0
    cancelled = asyncio.Event()
    gate = asyncio.Event()

    class Producer(PlatformRolloutFn):
        async def _preflight(self):  # No capture here; test_preflight covers the canary.
            return None

        async def _generate_group(self, prompt_group):
            nonlocal started
            started += 1
            if started == 1:
                return entry(attempt, policy, group="first")
            try:
                await gate.wait()
            finally:
                cancelled.set()
            return entry(attempt, policy, group=f"late-{started}")

    source = PlatformTaskSource(args)
    producer = Producer(RolloutFnConstructorInput(args=args, data_source=source))
    result = await asyncio.wait_for(producer(RolloutFnTrainInput(rollout_id=0, weight_version=1)), 2)
    assert len(result.samples) == 1 and started >= 2
    assert [c.group_id for c in source.consumed.snapshot()] == ["first"]
    assert producer._state is None  # Platform producer needs no local inference state/tokenizer.
    await producer.close()
    assert cancelled.is_set()
    assert producer._worker is None


def _producer_args(path, granularity="sample"):
    return Namespace(
        proximal_config=str(path),
        rollout_submission_granularity=granularity,
        n_samples_per_prompt=2,
        async_unused_samples_handler="drop",
        rollout_sample_filter_path=None,
        rollout_batch_size=1,
        rollout_global_dataset=True,
        async_max_concurrent_samples=4,
        custom_async_data_buffer_path="miles_plugins.proximal.buffer.PlatformDataBuffer",
        save=None,
        load=None,
        proximal_yes_rollouts=True,
        proximal_yes_publish=True,
    )


async def _run_group(config, tmp_path, policy, store, monkeypatch, execute):
    """One platform group through PlatformRolloutFn with ``execute`` standing in for each attempt;
    returns the group task and the list the scheduler's per-sample callback appends to."""
    from miles_plugins.proximal import rollout

    await store.commit_policy(policy)
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    args = _producer_args(path)
    monkeypatch.setattr(rollout, "execute_attempt", execute)
    source = PlatformTaskSource(args)
    producer = PlatformRolloutFn(RolloutFnConstructorInput(args=args, data_source=source))
    producer._store = store
    freed: list[int] = []
    producer._scheduler.sample_done_callback = lambda: freed.append(1)
    [group] = source.get_samples(1)
    return producer, asyncio.create_task(producer._generate_group(group)), freed


async def test_each_finished_rollout_frees_its_submission_slot(config, tmp_path, policy, store, monkeypatch):
    """Miles's sample backfill: the next group can start before this group's slowest rollout ends."""
    slow = asyncio.Event()
    calls = 0

    async def execute(attempt, sample, **_):
        nonlocal calls
        calls += 1
        if calls == 2:
            await slow.wait()
        return replace(sample, reward=1.0)

    producer, running, freed = await _run_group(config, tmp_path, policy, store, monkeypatch, execute)

    async def first_freed():
        while not freed:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(first_freed(), 2)
    assert len(freed) == 1 and not running.done()
    slow.set()
    result = await running
    assert len(freed) == 2 and [sample.reward for sample in result.group] == [1.0, 1.0]
    await producer.close()


async def test_a_failed_group_frees_every_slot(config, tmp_path, policy, store, monkeypatch):
    calls = 0

    async def execute(attempt, sample, **_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IneligibleAttempt("platform rollout is ineligible")
        await asyncio.Event().wait()  # Cancelled when its group-mate fails.

    producer, running, freed = await _run_group(config, tmp_path, policy, store, monkeypatch, execute)
    result = await asyncio.wait_for(running, 2)
    assert len(freed) == 2 and {sample.status for sample in result.group} == {Sample.Status.ABORTED}
    await producer.close()


def test_rollback_rewrites_step_state_and_refuses_partial_checkpoints(config, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    checkpoints = tmp_path / "checkpoints"
    args = Namespace(proximal_config=str(path), save=str(checkpoints), load=str(checkpoints))
    source = PlatformTaskSource(args)
    source.consumed.add("a", 1)
    source.save(0)
    source.consumed.add("b", 1)
    source.save(1)
    # Resume from step 0, train a different batch, then save step 1 again.
    resumed = PlatformTaskSource(args)
    resumed.load(0)
    assert [c.group_id for c in resumed.consumed.snapshot()] == ["a"]
    resumed.consumed.add("c", 1)
    resumed.save(1)
    again = PlatformTaskSource(args)
    again.load(1)
    assert [c.group_id for c in again.consumed.snapshot()] == ["a", "c"]
    # Weights saved for step 2 but no task/consumption state: refuse, never guess.
    with pytest.raises(FileNotFoundError, match="previous complete checkpoint"):
        PlatformTaskSource(args).load(2)


def test_rollback_invalidates_later_state_before_new_weights_exist(config, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    checkpoints = tmp_path / "checkpoints"
    args = Namespace(proximal_config=str(path), save=str(checkpoints), load=str(checkpoints))
    source = PlatformTaskSource(args)
    source.consumed.add("old-0", 1)
    source.save(0)
    source.consumed.add("old-1", 1)
    source.save(1)
    # Restore step 0. Training then re-saves step 1's weights and crashes before
    # re-saving its state: the old timeline's step-1 ledger must not pair with them.
    PlatformTaskSource(args).load(0)
    with pytest.raises(FileNotFoundError, match="previous complete checkpoint"):
        PlatformTaskSource(args).load(1)
    # A fresh start (Miles passes -1) invalidates every saved step in the save directory.
    source.save(3)
    PlatformTaskSource(args).load(-1)
    assert not list((checkpoints / "rollout").glob("proximal_*.json"))


def test_a_failed_restore_deletes_no_checkpoint_state(config, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    checkpoints = tmp_path / "checkpoints"
    args = Namespace(proximal_config=str(path), save=str(checkpoints), load=str(checkpoints))
    source = PlatformTaskSource(args)
    source.save(0)
    source.consumed.add("kept", 1)
    source.save(1)
    # Restoring step 0 under a different dataset fails validation...
    other = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"project_id": config.dataset.project_id + 1})}
    )
    other_path = tmp_path / "other.json"
    other_path.write_text(other.model_dump_json())
    other_args = Namespace(proximal_config=str(other_path), save=str(checkpoints), load=str(checkpoints))
    with pytest.raises(ValueError, match="differs from this run"):
        PlatformTaskSource(other_args).load(0)
    # ...and step 1's ledger is still there.
    again = PlatformTaskSource(args)
    again.load(1)
    assert [c.group_id for c in again.consumed.snapshot()] == ["kept"]
