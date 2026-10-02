"""A rollout the platform could not start is relaunched; any other failure still fails its group."""

import random
from dataclasses import replace

import httpx
import pytest

from miles.utils.types import Sample
from miles_plugins.proximal import rollout
from miles_plugins.proximal.clients import IneligibleAttempt, LaunchFailed
from miles_plugins.proximal.contracts import LaunchRetry, ReplicaLoad, platform_rollout_id


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


def _retry(attempts, replica_choices=1):
    return LaunchRetry(
        attempts=attempts,
        backoff_seconds=0.001,
        max_backoff_seconds=0.002,
        stagger_seconds=0.001,
        replica_choices=replica_choices,
    )


async def _run(attempt, retry, capture=None):
    return await rollout.execute_with_launch_retry(
        attempt, Sample(index=0), capture=capture, platform=None, artifact_root=None, retry=retry, rng=random.Random(0)
    )


class Replicas:
    """Each probed identity routes to a replica holding the next scripted load."""

    def __init__(self, loads):
        self.loads = list(loads)
        self.probed: dict[str, object] = {}

    async def load(self, affinity):
        outcome = self.probed[affinity] = self.loads.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ReplicaLoad(sessions=outcome)

    def identity_holding(self, sessions):
        (affinity,) = [a for a, load in self.probed.items() if load == sessions]
        return affinity


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


async def test_a_launch_takes_the_identity_whose_replica_holds_the_fewest_rollouts(attempt, launches):
    calls, _ = launches
    replicas = Replicas([20, 7, 12])
    await _run(attempt, _retry(3, replica_choices=3), capture=replicas)
    assert platform_rollout_id(attempt.attempt_id) in replicas.probed
    assert [platform_rollout_id(call) for call in calls] == [replicas.identity_holding(7)]


async def test_the_given_identity_wins_a_tie(attempt, launches):
    calls, _ = launches
    await _run(attempt, _retry(3, replica_choices=2), capture=Replicas([5, 5]))
    assert calls == [attempt.attempt_id]


async def test_a_relaunch_is_placed_again(attempt, launches):
    calls, outcomes = launches
    outcomes += [LaunchFailed("admission queue")]
    replicas = Replicas([3, 9, 8, 2])
    await _run(attempt, _retry(3, replica_choices=2), capture=replicas)
    assert calls[0] == attempt.attempt_id and platform_rollout_id(calls[1]) == replicas.identity_holding(2)


async def test_a_replica_that_does_not_answer_is_not_chosen(attempt, launches):
    calls, _ = launches
    down = httpx.ConnectError("replica is restarting")
    replicas = Replicas([down, 4])
    await _run(attempt, _retry(3, replica_choices=2), capture=replicas)
    await _run(attempt, _retry(3, replica_choices=2), capture=Replicas([down, down]))
    assert platform_rollout_id(calls[0]) == replicas.identity_holding(4) and calls[1] == attempt.attempt_id


def test_backoff_bounds_are_validated():
    with pytest.raises(ValueError):
        LaunchRetry(attempts=3, backoff_seconds=10, max_backoff_seconds=5, stagger_seconds=0)
