"""A rollout the platform could not start is relaunched with backoff; any other failure is left to
its group, which relaunches the member or fails (test_attempt_relaunch)."""

import random
from dataclasses import replace

import pytest

from miles.utils.types import Sample
from miles_plugins.proximal import rollout
from miles_plugins.proximal.clients import IneligibleAttempt, LaunchFailed
from miles_plugins.proximal.contracts import LaunchRetry


@pytest.fixture
def launches(monkeypatch):
    """Stand in for execute_attempt: fail each attempt with the scripted outcome, then succeed."""
    calls: list[str] = []
    outcomes: list[Exception | None] = []

    async def execute(attempt, sample, **_):
        calls.append(attempt.attempt_id)
        outcome = outcomes.pop(0) if outcomes else None
        if outcome is not None:
            raise outcome
        return replace(sample, reward=1.0)

    monkeypatch.setattr(rollout, "execute_attempt", execute)
    return calls, outcomes


def _retry(attempts):
    return LaunchRetry(attempts=attempts, backoff_seconds=0.001, max_backoff_seconds=0.002, stagger_seconds=0.001)


async def _run(attempt, retry):
    return await rollout.execute_with_launch_retry(
        attempt, Sample(index=0), capture=None, platform=None, artifact_root=None, retry=retry, rng=random.Random(0)
    )


async def test_a_launch_failure_is_relaunched_under_a_new_attempt_identity(attempt, launches):
    calls, outcomes = launches
    outcomes += [LaunchFailed("admission queue"), LaunchFailed("admission queue")]
    result = await _run(attempt, _retry(3))
    assert result.reward == 1.0
    assert len(calls) == 3 and calls[0] == attempt.attempt_id and len(set(calls)) == 3


async def test_launch_retries_are_bounded(attempt, launches):
    calls, outcomes = launches
    outcomes += [LaunchFailed("admission queue")] * 3
    with pytest.raises(LaunchFailed):
        await _run(attempt, _retry(3))
    assert len(calls) == 3


async def test_a_rollout_that_ran_and_failed_is_never_retried(attempt, launches):
    calls, outcomes = launches
    outcomes += [IneligibleAttempt("Platform rollout is ineligible: ROLLOUT_CONTAINER_STATUS_ERROR")]
    with pytest.raises(IneligibleAttempt):
        await _run(attempt, _retry(3))
    assert calls == [attempt.attempt_id]


def test_backoff_bounds_are_validated():
    with pytest.raises(ValueError):
        LaunchRetry(attempts=3, backoff_seconds=10, max_backoff_seconds=5, stagger_seconds=0)
