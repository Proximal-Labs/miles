"""A group member whose rollout fails to execute is relaunched in its group as soon as it fails.

2026-10-03: Modal drained serving replicas during a DeepSWE eval and 137 attempts lost their
capture sessions. Each failed group waited 1-2 hours for its other attempts to finish, then
re-ran all of them. Now the failed member alone is relaunched at once: same group, sample
index, task and policy, under a new attempt identity. The siblings keep running.
"""

import asyncio
from collections import Counter

import httpx
import pytest
from tests.integration.proximal_async.test_buffer import sample_for, versioned
from tests.integration.proximal_async.test_runtime import _producer_args

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.utils.types import Sample
from miles_plugins.proximal import rollout
from miles_plugins.proximal.buffer import accepted, validate_group
from miles_plugins.proximal.clients import IneligibleAttempt, LaunchFailed
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.rollout import PlatformRolloutFn, member_relaunches


def _with(config, **research):
    return config.model_copy(update={"research": config.research.model_copy(update=research)})


def _graded(attempt, sample, reward=1.0):
    """What execute_attempt returns for a graded attempt: the sample with its acceptance evidence."""
    result, _ = sample_for(attempt, reward=reward)
    result.index, result.group_index = sample.index, sample.group_index
    return result


async def _producer(config, tmp_path, policy, store, monkeypatch, execute):
    """A platform producer bound to ``store`` with ``execute`` standing in for each attempt.
    Returns it, a group from its task source, and the list each freed sample slot appends to."""
    await store.commit_policy(policy)
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    monkeypatch.setattr(rollout, "execute_attempt", execute)
    args = _producer_args(path)
    source = PlatformTaskSource(args)
    producer = PlatformRolloutFn(RolloutFnConstructorInput(args=args, data_source=source))
    producer._store = store
    freed: list[int] = []
    producer._scheduler.sample_done_callback = lambda: freed.append(1)
    [group] = source.get_samples(1)
    return producer, group, freed


async def _until(condition):
    while not condition():
        await asyncio.sleep(0.01)


def test_only_retry_relaunches_members_and_never_more_than_a_group_retry(config):
    assert member_relaunches(config.research) == config.research.group_size
    assert member_relaunches(_with(config, unused_groups="drop").research) == 0


async def test_a_failed_member_is_relaunched_before_its_siblings_finish(config, tmp_path, policy, store, monkeypatch):
    sibling = asyncio.Event()
    launches = []

    async def execute(attempt, sample, **_):
        launches.append(attempt)
        if attempt.sample_index == 0 and len([a for a in launches if a.sample_index == 0]) == 1:
            raise IneligibleAttempt("Platform rollout is ineligible: ROLLOUT_CONTAINER_STATUS_ERROR")
        if attempt.sample_index == 1:
            await sibling.wait()
        return _graded(attempt, sample)

    producer, group, freed = await _producer(config, tmp_path, policy, store, monkeypatch, execute)
    running = asyncio.create_task(producer._generate_group(group))
    try:
        # The replacement runs and finishes while the sibling's first attempt is still running.
        await asyncio.wait_for(_until(lambda: freed), 2)
        assert not running.done() and not sibling.is_set()
        failed, replacement = [attempt for attempt in launches if attempt.sample_index == 0]
        [other] = [attempt for attempt in launches if attempt.sample_index == 1]
        assert replacement.attempt_id not in {failed.attempt_id, other.attempt_id}
        assert replacement.model_copy(update={"attempt_id": failed.attempt_id}) == failed
        sibling.set()
        result = await asyncio.wait_for(running, 2)
    finally:
        await producer.close()
    # A complete group: every member on the group's id, task and policy, each sample once.
    assert validate_group(config, result.group) == policy
    assert [accepted(sample).attempt.attempt_id for sample in result.group] == [
        replacement.attempt_id,
        other.attempt_id,
    ]
    assert [sample.index for sample in result.group] == [sample.index for sample in group]
    assert len(launches) == 3 and len(freed) == 2  # A relaunch keeps its member's submission slot.


@pytest.mark.parametrize(
    "failures, completes",
    [((1, 1), True), ((2, 0), True), ((3, 0), False), ((2, 1), False)],
)
async def test_members_share_the_groups_relaunch_budget(
    config, tmp_path, policy, store, monkeypatch, failures, completes
):
    launched: Counter[int] = Counter()

    async def execute(attempt, sample, **_):
        launched[attempt.sample_index] += 1
        if launched[attempt.sample_index] <= failures[attempt.sample_index]:
            raise IneligibleAttempt("Platform rollout is ineligible: ROLLOUT_CONTAINER_STATUS_ERROR")
        return _graded(attempt, sample)

    producer, group, freed = await _producer(config, tmp_path, policy, store, monkeypatch, execute)
    try:
        result = await asyncio.wait_for(producer._generate_group(group), 2)
    finally:
        await producer.close()
    relaunched = launched.total() - config.research.group_size
    if completes:
        assert relaunched == sum(failures)
        assert validate_group(config, result.group) == policy
    else:
        # Never past the budget; once a member gives up, the others stop relaunching too.
        assert relaunched <= member_relaunches(config.research)
        assert {sample.status for sample in result.group} == {Sample.Status.ABORTED}
    assert len(freed) == 2


async def test_an_exhausted_group_fails_only_after_its_siblings_finish(config, tmp_path, policy, store, monkeypatch):
    sibling = asyncio.Event()
    launched: Counter[int] = Counter()
    finished = []

    async def execute(attempt, sample, **_):
        launched[attempt.sample_index] += 1
        if attempt.sample_index == 0:
            raise httpx.ConnectError("capture session lost")
        await sibling.wait()
        finished.append(attempt.attempt_id)
        return _graded(attempt, sample)

    producer, group, _ = await _producer(config, tmp_path, policy, store, monkeypatch, execute)
    running = asyncio.create_task(producer._generate_group(group))
    try:
        await asyncio.wait_for(_until(lambda: launched[0] == 1 + config.research.group_size), 2)
        await asyncio.sleep(0.05)
        # Budget spent: no further relaunch, and the sibling is neither cancelled nor re-run.
        assert launched == {0: 1 + config.research.group_size, 1: 1} and not running.done()
        sibling.set()
        result = await asyncio.wait_for(running, 2)
    finally:
        await producer.close()
    assert len(finished) == 1 and launched[1] == 1
    assert {sample.status for sample in result.group} == {Sample.Status.ABORTED}


async def test_a_launch_that_outlasts_its_launch_retries_costs_one_relaunch(
    config, tmp_path, policy, store, monkeypatch
):
    """LaunchFailed is first retried with backoff (test_launch_retry); only when those retries
    run out does the member fail, and it is relaunched like any other failed member."""
    launched: Counter[int] = Counter()

    async def execute(attempt, sample, **_):
        launched[attempt.sample_index] += 1
        if attempt.sample_index == 0 and launched[0] <= config.launch_retry.attempts:
            raise LaunchFailed("Platform could not launch the rollout: Launch failed: admission queue")
        return _graded(attempt, sample)

    producer, group, _ = await _producer(config, tmp_path, policy, store, monkeypatch, execute)
    try:
        result = await asyncio.wait_for(producer._generate_group(group), 2)
    finally:
        await producer.close()
    assert launched == {0: config.launch_retry.attempts + 1, 1: 1}
    assert validate_group(config, result.group) == policy


async def test_a_graded_zero_is_trained_on_not_relaunched(config, tmp_path, policy, store, monkeypatch):
    launches = []

    async def execute(attempt, sample, **_):
        launches.append(attempt.attempt_id)
        return _graded(attempt, sample, reward=0.0)

    producer, group, _ = await _producer(config, tmp_path, policy, store, monkeypatch, execute)
    try:
        result = await asyncio.wait_for(producer._generate_group(group), 2)
    finally:
        await producer.close()
    assert len(launches) == config.research.group_size
    assert validate_group(config, result.group) == policy
    assert [sample.reward for sample in result.group] == [0.0, 0.0]


async def test_a_failure_that_is_not_an_execution_failure_stops_relaunches(
    config, tmp_path, policy, store, monkeypatch
):
    """A broken contract still stops the run, and the group's other member is not relaunched."""
    failing = asyncio.Event()
    launched: Counter[int] = Counter()

    async def execute(attempt, sample, **_):
        launched[attempt.sample_index] += 1
        if attempt.sample_index == 0:
            failing.set()
            raise ValueError("Sealed capture provenance differs from the attempt")
        await failing.wait()
        await asyncio.sleep(0.01)
        raise IneligibleAttempt("Platform rollout is ineligible: ROLLOUT_CONTAINER_STATUS_ERROR")

    producer, group, _ = await _producer(config, tmp_path, policy, store, monkeypatch, execute)
    try:
        with pytest.raises(ValueError, match="provenance"):
            await asyncio.wait_for(producer._generate_group(group), 2)
    finally:
        await producer.close()
    assert launched == {0: 1, 1: 1}


@pytest.mark.parametrize("published, relaunches", [(1, True), (2, False)])
async def test_a_member_is_relaunched_only_while_its_groups_policy_is_selectable(
    config, tmp_path, policy, store, monkeypatch, published, relaunches
):
    """Online training: the replacement is sampled from the group's own policy, and only while
    the batch query could still select the group (max_policy_lag 1 here)."""
    launches = []

    async def execute(attempt, sample, **_):
        launches.append(attempt)
        if len(launches) == 1:
            # The trainer publishes while this rollout runs.
            for version in range(2, 2 + published):
                await store.commit_policy(versioned(policy, version, sha="ef"[version - 2]))
            raise IneligibleAttempt("Platform rollout is ineligible: ROLLOUT_CONTAINER_STATUS_ERROR")
        return _graded(attempt, sample)

    producer, group, _ = await _producer(config, tmp_path, policy, store, monkeypatch, execute)
    try:
        result = await asyncio.wait_for(producer._generate_group(group), 2)
    finally:
        await producer.close()
    if relaunches:
        assert len(launches) == 3 and {attempt.policy for attempt in launches} == {policy}
        assert validate_group(config, result.group) == policy
    else:
        assert len(launches) == 2
        assert {sample.status for sample in result.group} == {Sample.Status.ABORTED}


async def _collect(config, tmp_path, policy, store, monkeypatch, execute, *, groups):
    """Finite collection as collect_batch runs it: the real producer, buffer and store."""
    await store.commit_policy(policy)
    path = tmp_path / "collect.json"
    path.write_text(config.model_dump_json())
    monkeypatch.setattr(rollout, "execute_attempt", execute)
    args = _producer_args(path)
    args.async_unused_samples_handler = config.research.unused_groups

    class Collector(PlatformRolloutFn):
        async def _preflight(self):  # test_preflight covers the canary.
            pass

    source = PlatformTaskSource(args, num_groups=groups)
    collector = Collector(RolloutFnConstructorInput(args=args, data_source=source))
    drained = []
    try:
        for step in range(groups):
            output = await asyncio.wait_for(collector(RolloutFnTrainInput(rollout_id=step, weight_version=1)), 5)
            drained.extend(output.samples)
            metrics = output.metrics
    finally:
        await collector.close()
    return source, drained, metrics


async def test_collection_relaunches_a_member_without_spending_task_budget(
    config, tmp_path, policy, store, monkeypatch
):
    tasks = tuple(config.dataset.tasks[0].model_copy(update={"environment_id": i}) for i in (7, 8))
    config = config.model_copy(update={"dataset": config.dataset.model_copy(update={"tasks": tasks})})
    launches = []

    async def execute(attempt, sample, **_):
        launches.append(attempt)
        if len(launches) == 1:
            raise IneligibleAttempt("Platform rollout is ineligible: ROLLOUT_CONTAINER_STATUS_ERROR")
        return _graded(attempt, sample)

    source, drained, metrics = await _collect(config, tmp_path, policy, store, monkeypatch, execute, groups=2)
    # One relaunch, no extra group: both tasks once, sample identities as admitted.
    assert len(launches) == 2 * config.research.group_size + 1
    assert source.next_group == 2 and not source.has_samples and source.get_buffer_length() == 0
    assert sorted(accepted(group[0]).attempt.task.environment_id for group in drained) == [7, 8]
    assert sorted(sample.index for group in drained for sample in group) == [0, 1, 2, 3]
    assert all(validate_group(config, group) == policy for group in drained)
    attempt_ids = [accepted(sample).attempt.attempt_id for group in drained for sample in group]
    assert len(set(attempt_ids)) == len(attempt_ids) and launches[0].attempt_id not in attempt_ids
    assert metrics["rollout/platform/consecutive_failed_groups"] == 0


async def test_collection_retries_a_group_whose_relaunches_ran_out(config, tmp_path, policy, store, monkeypatch):
    launched: Counter[int] = Counter()

    async def execute(attempt, sample, **_):
        launched[attempt.sample_index] += 1
        if attempt.sample_index == 0:  # Only the first group's first member, every time.
            raise IneligibleAttempt("Platform rollout is ineligible: ROLLOUT_CONTAINER_STATUS_ERROR")
        return _graded(attempt, sample)

    source, drained, _ = await _collect(config, tmp_path, policy, store, monkeypatch, execute, groups=1)
    # The failed group falls back to the group retry: the same task, a new group, no new task.
    assert launched == {0: 1 + config.research.group_size, 1: 1, 2: 1, 3: 1}
    [group] = drained
    assert [sample.group_index for sample in group] == [1, 1]
    assert validate_group(config, group) == policy
    assert source.next_group == 2 and not source.has_samples
