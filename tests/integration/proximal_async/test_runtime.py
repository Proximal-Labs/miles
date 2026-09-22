import argparse
import asyncio
import sys
from argparse import Namespace

import pytest
from tests.integration.proximal_async.test_buffer import entry

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.utils.arguments import get_miles_extra_args_provider, resolve_rollout_function_paths
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
