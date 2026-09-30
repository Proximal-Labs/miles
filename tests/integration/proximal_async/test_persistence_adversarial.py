"""Fault injection at persistence boundaries, with real files/codecs/Postgres."""

import asyncio
import hashlib
import random
import shutil

import pytest
from tests.integration.proximal_async.test_buffer import entry, sample_for
from tests.integration.proximal_async.test_offline_batch import freeze, populate
from tests.integration.proximal_async.test_run_state import run_state_store

from miles.rollout.session.samples.codec import encode_samples
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall
from miles_plugins.proximal import offline_batch
from miles_plugins.proximal.buffer import accepted, validate_group
from miles_plugins.proximal.clients import LaunchFailed
from miles_plugins.proximal.contracts import FailedAttempt, SessionHandle, digest
from miles_plugins.proximal.rollout import execute_attempt, wait_for_releases
from miles_plugins.proximal.state_writer import StateWriter
from miles_plugins.proximal.store import open_store


@pytest.mark.parametrize("execution", ["failed", "graded", "not_started"])
@pytest.mark.parametrize("fetch_error", [TimeoutError, ValueError])
async def test_capture_release_requires_saved_payload_or_proof_nothing_ran(
    config, attempt, store_dsn, tmp_path, execution, fetch_error
):
    """A timeout or corrupt download does not prove the replica has no useful data."""
    _, evidence = sample_for(attempt)
    handle = SessionHandle(
        session_id="a" * 32, rollout_id="r", base_url="http://localhost:1/v1", request_sha256=digest(attempt)
    )
    released = []

    class Capture:
        async def create(self, _):
            return handle

        async def collect(self, *_):
            raise fetch_error("capture could not be retrieved")

        async def release(self, _):
            released.append(True)

    class Platform:
        async def execute(self, *_):
            if execution == "graded":
                return evidence.grade
            if execution == "not_started":
                raise LaunchFailed("platform never started the rollout")
            raise RuntimeError("verifier failed")

        async def cancel(self, _):
            pass

    store = await run_state_store(config)
    mount, durable = tmp_path / "mount", tmp_path / "durable"
    writer = StateWriter(
        dsn=store_dsn,
        run_id=config.run_id,
        artifacts=config.artifact_directory,
        snapshot_root=mount,
        commit=lambda: shutil.copytree(mount, durable, dirs_exist_ok=True),
    )
    writer.start()
    try:
        with pytest.raises(fetch_error if execution == "graded" else RuntimeError):
            await asyncio.wait_for(
                execute_attempt(
                    attempt,
                    Sample(index=0, group_index=0),
                    capture=Capture(),
                    platform=Platform(),
                    artifact_root=config.artifact_directory / config.run_id / "accepted",
                    store=store,
                ),
                5,
            )
        await wait_for_releases()
        directory = durable / "artifacts" / config.run_id / "accepted" / attempt.attempt_id
        outcome = FailedAttempt.model_validate_json((directory / "failed.json").read_bytes())
        assert outcome.capture is None
        assert outcome.grade == (evidence.grade if execution == "graded" else None)
        assert not (directory / "accepted.json").exists()
        if execution == "not_started":
            assert released == [True], "A proven launch failure must not occupy replica capacity"
        else:
            assert not released, "Archiving a failed read must not delete the unread replica capture"
    finally:
        await asyncio.to_thread(writer.close)
        await wait_for_releases()
        await store.close()


async def test_failed_capture_disk_error_retains_replica_payload(config, attempt, monkeypatch):
    from miles_plugins.proximal import rollout

    sample, evidence = sample_for(attempt)
    payload = encode_samples([sample], {})
    receipt = evidence.capture.model_copy(update={"payload_sha256": hashlib.sha256(payload).hexdigest()})
    handle = SessionHandle(
        session_id="a" * 32, rollout_id="r", base_url="http://localhost:1/v1", request_sha256=digest(attempt)
    )
    released = []

    class Capture:
        async def create(self, _):
            return handle

        async def collect(self, *_):
            return receipt, payload

        async def release(self, _):
            released.append(True)

    class Platform:
        async def execute(self, *_):
            raise RuntimeError("verifier failed")

        async def cancel(self, _):
            pass

    write = rollout.write_immutable

    def fail_partial_write(path, data):
        if path.name == "partial.safetensors":
            raise OSError("disk full")
        write(path, data)

    monkeypatch.setattr(rollout, "write_immutable", fail_partial_write)
    try:
        with pytest.raises(OSError, match="disk full"):
            await execute_attempt(
                attempt, Sample(), capture=Capture(), platform=Platform(), artifact_root=config.artifact_directory
            )
    finally:
        await wait_for_releases()
    assert not released, "A local disk error must not discard a recoverable replica capture"
    assert not (config.artifact_directory / attempt.attempt_id / "failed.json").exists()


@pytest.mark.parametrize("stage", ["cancel", "collect", "publish"])
async def test_repeated_cancellation_preserves_failed_outcome(
    config, attempt, store_dsn, tmp_path, monkeypatch, stage
):
    """Group teardown may cancel again while the first cancellation is unwinding."""
    sample, evidence = sample_for(attempt)
    payload = encode_samples([sample], {})
    receipt = evidence.capture.model_copy(update={"payload_sha256": hashlib.sha256(payload).hexdigest()})
    handle = SessionHandle(
        session_id="a" * 32, rollout_id="r", base_url="http://localhost:1/v1", request_sha256=digest(attempt)
    )
    running, paused, proceed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    released = []

    async def pause(name):
        if stage == name:
            paused.set()
            await proceed.wait()

    class Capture:
        async def create(self, _):
            return handle

        async def collect(self, *_):
            await pause("collect")
            return receipt, payload

        async def release(self, _):
            released.append(True)

    class Platform:
        async def execute(self, *_):
            running.set()
            await asyncio.Event().wait()

        async def cancel(self, _):
            await pause("cancel")

    store = await run_state_store(config)
    publish = store.publish_artifact

    async def blocked_publish(payload, completion, *, record_id):
        if record_id.startswith("failed-"):
            await pause("publish")
        await publish(payload, completion, record_id=record_id)

    monkeypatch.setattr(store, "publish_artifact", blocked_publish)
    mount, durable = tmp_path / "mount", tmp_path / "durable"
    writer = StateWriter(
        dsn=store_dsn,
        run_id=config.run_id,
        artifacts=config.artifact_directory,
        snapshot_root=mount,
        commit=lambda: shutil.copytree(mount, durable, dirs_exist_ok=True),
    )
    writer.start()
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
        await asyncio.wait_for(running.wait(), 5)
        task.cancel()
        await asyncio.wait_for(paused.wait(), 5)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "Repeated cancellation abandoned the terminal archive"
        assert not released
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        await wait_for_releases()
        assert released == [True]
        directory = durable / "artifacts" / config.run_id / "accepted" / attempt.attempt_id
        outcome = FailedAttempt.model_validate_json((directory / "failed.json").read_bytes())
        assert outcome.status == "cancelled" and outcome.capture == receipt
        assert (directory / "partial.safetensors").read_bytes() == payload
        assert not (directory / "accepted.json").exists()
    finally:
        proceed.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(writer.close)
        await wait_for_releases()
        await store.close()


@pytest.mark.parametrize("fail_at", [1, 2])
@pytest.mark.parametrize("ack_lost", [False, True])
async def test_batch_publication_can_restart_after_each_failed_commit(
    config, policy, attempt, tmp_path, fail_at, ack_lost
):
    await populate(config, policy, attempt, 2)
    bundle, mount, durable = (tmp_path / name for name in ("batch", "mount", "durable"))
    expected = freeze(config, policy, config.artifact_directory / config.run_id, bundle)
    count = 0

    def commit():
        nonlocal count
        count += 1
        if count == fail_at and not ack_lost:
            raise OSError("commit failed before storage")
        shutil.copytree(mount, durable, dirs_exist_ok=True)
        if count == fail_at:
            raise OSError("storage succeeded but acknowledgement was lost")

    with pytest.raises(OSError):
        offline_batch.publish_batch(bundle, mount, commit=commit)
    # A failed call may have committed readiness only if both file sets survived.
    if (durable / "batch.json").exists():
        assert offline_batch.validate_batch(durable) == expected
    shutil.rmtree(mount)
    if durable.exists():
        shutil.copytree(durable, mount)
    offline_batch.publish_batch(bundle, mount, commit=lambda: shutil.copytree(mount, durable, dirs_exist_ok=True))
    shutil.rmtree(bundle)
    shutil.rmtree(mount)
    shutil.rmtree(config.artifact_directory)
    assert offline_batch.validate_batch(durable) == expected


async def test_payload_corruption_between_validation_and_copy_cannot_publish_readiness(
    config, policy, attempt, tmp_path, monkeypatch
):
    await populate(config, policy, attempt, 2)
    bundle, mount = tmp_path / "batch", tmp_path / "mount"
    freeze(config, policy, config.artifact_directory / config.run_id, bundle)
    validate = offline_batch.validate_batch

    def corrupt_after_validation(path):
        result = validate(path)
        payload = path / "groups/g0.bin"
        data = payload.read_bytes()
        payload.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        return result

    monkeypatch.setattr(offline_batch, "validate_batch", corrupt_after_validation)
    with pytest.raises(ValueError, match="checksum|changed"):
        offline_batch.publish_batch(bundle, mount, commit=lambda: None)
    assert not (mount / "batch.json").exists()


async def test_manifest_replacement_after_validation_cannot_change_published_batch(
    config, policy, attempt, tmp_path, monkeypatch
):
    await populate(config, policy, attempt, 2)
    bundle, mount = tmp_path / "batch", tmp_path / "mount"
    expected = freeze(config, policy, config.artifact_directory / config.run_id, bundle)
    validate = offline_batch.validate_batch

    def change_manifest(path):
        result = validate(path)
        (path / "batch.json").write_text('{"num_samples": 999999}')
        return result

    monkeypatch.setattr(offline_batch, "validate_batch", change_manifest)
    assert offline_batch.publish_batch(bundle, mount, commit=lambda: None) == expected
    assert validate(mount) == expected


async def test_multiturn_masks_and_logprobs_survive_storage_exactly(config, policy, attempt, tmp_path):
    rng = random.Random(91)
    originals = []
    store = await open_store(config)
    try:
        for group_index in range(16):
            samples = entry(attempt, policy, group=f"g{group_index}", group_index=group_index).group
            for sample in samples:
                # Different prompt/assistant/tool spans; include tiny and large negative logprobs.
                prompt = rng.randint(1, 256)
                lengths = [rng.randint(1, 32) for _ in range(5)]
                mask = [value for i, length in enumerate(lengths) for value in [int(i % 2 == 0)] * length]
                sample.tokens = [rng.randrange(150000) for _ in range(prompt + len(mask))]
                sample.loss_mask = mask
                sample.response_length = len(mask)
                sample.rollout_log_probs = [rng.choice([-1e-100, -173.25, -rng.random()]) if m else 0.0 for m in mask]
                cursor = prompt
                spans = []
                for i, length in enumerate(lengths):
                    if i % 2 == 0:
                        spans.append(WeightVersionSpan(version="1", abs_start=cursor, abs_end=cursor + length))
                    cursor += length
                sample.weight_versions = [WeightVersionsPerCall(spans=[span]) for span in spans]
                proof = accepted(sample)
                proof = proof.model_copy(
                    update={
                        "capture": proof.capture.model_copy(
                            update={"num_tokens": len(sample.tokens), "num_calls": len(spans)}
                        )
                    }
                )
                sample.metadata["proximal_accepted"] = proof.model_dump_json()
            validate_group(config, samples)
            await store.add_group(f"g{group_index}", policy, samples)
            originals.extend(samples)
    finally:
        await store.close()
    bundle = tmp_path / "batch"
    batch = freeze(config, policy, config.artifact_directory / config.run_id, bundle, count=16)
    shutil.rmtree(config.artifact_directory)
    restored = [sample for index in batch.groups for sample in offline_batch.load_group(bundle, index, config)]
    for original, actual in zip(originals, restored, strict=True):
        assert (actual.tokens, actual.response_length, actual.loss_mask, actual.reward) == (
            original.tokens,
            original.response_length,
            original.loss_mask,
            original.reward,
        )
        assert [value.hex() for value in actual.rollout_log_probs] == [
            value.hex() for value in original.rollout_log_probs
        ]
        assert actual.all_weight_version_spans == original.all_weight_version_spans
        assert accepted(actual) == accepted(original)
