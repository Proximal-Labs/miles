import asyncio
from argparse import Namespace

import pytest

from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall
from miles_plugins.proximal.buffer import PlatformDataBuffer, validate_sample
from miles_plugins.proximal.contracts import AcceptedAttempt, CaptureReceipt, Grade, digest


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


def entry(attempt, version=1):
    policy = attempt.policy.model_copy(update={"version": version})
    samples = [
        sample_for(attempt.model_copy(update={"sample_index": i, "attempt_id": f"a-{i}-{version}", "policy": policy}))[
            0
        ]
        for i in range(2)
    ]
    return DataBufferInput(prompt_group=samples, group=samples)


def make_buffer(config, tmp_path, unused):
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    return PlatformDataBuffer(
        DataBufferConstructorInput(args=Namespace(proximal_config=str(path)), unused_handler_fn=unused.append)
    )


async def test_staleness_is_checked_at_consumption(config, tmp_path, attempt):
    unused = []
    buffer = make_buffer(config, tmp_path, unused)
    await buffer.put(entry(attempt, version=1))
    fresh = entry(attempt, version=3)
    await buffer.put(fresh)
    assert await buffer.get(current_version=3) is fresh
    assert len(unused) == 1
    assert buffer.get_metrics()["rollout/platform/stale_groups"] == 1


async def test_backpressure_and_failure_budget(config, tmp_path, attempt):
    buffer = make_buffer(config, tmp_path, [])
    await buffer.put(entry(attempt))
    await buffer.put(entry(attempt))
    blocked = asyncio.create_task(buffer.put(entry(attempt)))
    await asyncio.sleep(0)
    assert not blocked.done()
    await buffer.get(current_version=1)
    await asyncio.wait_for(blocked, 1)
    aborted = DataBufferInput(prompt_group=[], group=[Sample(status=Sample.Status.ABORTED) for _ in range(2)])
    await buffer.put(aborted)
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


async def test_group_cannot_mix_versions_or_duplicate_members(config, tmp_path, attempt):
    buffer = make_buffer(config, tmp_path, [])
    mixed = entry(attempt)
    mixed.group[1] = entry(attempt, version=2).group[1]
    with pytest.raises(ValueError, match="mixes"):
        await buffer.put(mixed)
    repeated = entry(attempt)
    repeated.group[1] = repeated.group[0]
    with pytest.raises(ValueError, match="Repeated"):
        await buffer.put(repeated)
    with pytest.raises(ValueError, match="committed"):
        await buffer.get()
