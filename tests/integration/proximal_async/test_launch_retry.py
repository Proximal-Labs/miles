"""A rollout the platform could not start, or a full replica refused, is relaunched; any other failure still fails its group."""

import random
from dataclasses import replace

import pytest

from miles.utils.types import Sample
from miles_plugins.proximal import rollout
from miles_plugins.proximal.clients import IneligibleAttempt, LaunchFailed, ReplicaFull
from miles_plugins.proximal.contracts import FailedAttempt, LaunchRetry


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


async def test_refusals_by_full_replicas_do_not_spend_launch_retries(attempt, launches, monkeypatch):
    monkeypatch.setattr(rollout, "REPLICA_FULL_BACKOFF_SECONDS", 0.001)
    calls, outcomes = launches
    outcomes += [ReplicaFull("full")] * 5 + [LaunchFailed("admission queue"), ReplicaFull("full")]
    result = await _run(attempt, _retry(2))
    # Seven retired identities and the eighth ran; one launch retry of two was used.
    assert result.reward == 1.0 and len(calls) == 8 and len(set(calls)) == 8


async def test_refusals_by_full_replicas_are_bounded(attempt, launches, monkeypatch):
    monkeypatch.setattr(rollout, "REPLICA_FULL_BACKOFF_SECONDS", 0.001)
    monkeypatch.setattr(rollout, "REPLICA_FULL_ATTEMPTS", 4)
    calls, outcomes = launches
    outcomes += [ReplicaFull("full")] * 4
    with pytest.raises(ReplicaFull):
        await _run(attempt, _retry(3))
    assert len(calls) == 4


async def test_a_refused_attempt_records_that_nothing_ran(attempt, tmp_path):
    class Full:
        async def create(self, attempt):
            raise ReplicaFull("full")

    class NoPlatform:
        async def cancel(self, attempt):
            raise AssertionError("A refused session precedes the platform run")

    with pytest.raises(ReplicaFull):
        await rollout.execute_attempt(
            attempt, Sample(index=0), capture=Full(), platform=NoPlatform(), artifact_root=tmp_path
        )
    failure = FailedAttempt.model_validate_json((tmp_path / attempt.attempt_id / "failed.json").read_bytes())
    assert (failure.error_type, failure.capture, failure.grade) == ("ReplicaFull", None, None)


def test_backoff_bounds_are_validated():
    with pytest.raises(ValueError):
        LaunchRetry(attempts=3, backoff_seconds=10, max_backoff_seconds=5, stagger_seconds=0)
