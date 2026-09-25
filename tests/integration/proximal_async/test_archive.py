import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from tests.integration.proximal_async.test_buffer import sample_for

from miles.rollout.session.samples.codec import encode_samples
from miles_plugins.proximal.archive import publish_pending
from miles_plugins.proximal.clients import PlatformClient
from miles_plugins.proximal.contracts import platform_rollout_id
from miles_plugins.proximal.storage import write_immutable
from miles_plugins.proximal.store import PayloadSync, open_store


def archive_fixture(attempt, directory: Path):
    sample, evidence = sample_for(attempt)
    payload = encode_samples([sample], {})
    evidence = evidence.model_copy(
        update={
            "capture": evidence.capture.model_copy(update={"payload_sha256": hashlib.sha256(payload).hexdigest()}),
            "grade": evidence.grade.model_copy(update={"rollout_id": platform_rollout_id(attempt.attempt_id)}),
        }
    )
    path = directory / "samples.safetensors"
    write_immutable(path, payload)
    return evidence, payload, path


async def make_due(store):
    await store.connection.execute("UPDATE proximal_capture_archives SET next_attempt_at = clock_timestamp()")


async def test_archive_survives_upload_and_commit_failure_then_trainer_restart(config, authorization, attempt, store):
    evidence, payload, path = archive_fixture(attempt, config.artifact_directory)
    await store.enqueue_capture(evidence, path)
    assert await store.pending_capture_count() == 1
    await store.enqueue_capture(evidence, path)  # Exact retry creates one row.
    uploaded = None
    receipt = None
    put_failures = 1
    complete_failures = 1
    observed = []

    async def api(request):
        nonlocal uploaded, receipt, put_failures, complete_failures
        observed.append(request.url.path.rsplit("/", 1)[-1])
        if request.method == "PUT":
            assert request.headers.get("x-api-key") is None
            assert request.headers.get("authorization") is None
            assert request.headers["if-none-match"] == "*"
            assert request.content == payload
            if put_failures:
                put_failures -= 1
                return httpx.Response(503)
            if uploaded is not None:
                return httpx.Response(412)
            uploaded = request.content
            return httpx.Response(200)
        body = json.loads(request.content)
        assert body["rolloutId"] == platform_rollout_id(attempt.attempt_id)
        assert request.headers["x-api-key"] == "platform-secret"
        receipt = body["receipt"]
        attachment = {"state": "pending", "receipt": receipt, "receiptSha256": "f" * 64}
        if request.url.path.endswith("PrepareMilesCaptureUpload"):
            return httpx.Response(
                200,
                json={
                    "attachment": attachment,
                    "uploadUrl": "https://s3.test/object",
                    "uploadHeaders": {"if-none-match": "*"},
                },
            )
        assert request.url.path.endswith("CompleteMilesCaptureUpload")
        if complete_failures:
            complete_failures -= 1
            return httpx.Response(500)
        assert uploaded == payload
        return httpx.Response(200, json={"attachment": {**attachment, "state": "ready"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as http:
        platform = PlatformClient(authorization, http)
        assert await publish_pending(store, platform) == 1
        await make_due(store)
        assert len(await store.pending_captures()) == 1
        assert await publish_pending(store, platform) == 1
        await make_due(store)
        # Another process can reopen the real Postgres outbox with only the retained file.
        replacement = await open_store(config)
        try:
            assert await publish_pending(replacement, PlatformClient(authorization, http)) == 1
            assert await replacement.pending_captures() == []
            assert await replacement.pending_capture_count() == 0
        finally:
            await replacement.close()
    assert observed == [
        "PrepareMilesCaptureUpload",
        "object",
        "PrepareMilesCaptureUpload",
        "object",
        "CompleteMilesCaptureUpload",
        "PrepareMilesCaptureUpload",
        "object",
        "CompleteMilesCaptureUpload",
    ]
    assert path.read_bytes() == payload
    assert evidence.grade.reward == 0.0
    row = await (
        await store.connection.execute("SELECT published, attempts FROM proximal_capture_archives")
    ).fetchone()
    assert row == (True, 2)


async def test_lost_completion_ack_can_resume_from_ready_without_reupload(config, authorization, attempt, store):
    evidence, payload, path = archive_fixture(attempt, config.artifact_directory)
    await store.enqueue_capture(evidence, path)
    methods = []

    def api(request):
        methods.append(request.url.path.rsplit("/", 1)[-1])
        body = json.loads(request.content)
        return httpx.Response(
            200, json={"attachment": {"state": "ready", "receipt": body["receipt"], "receiptSha256": "f" * 64}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as http:
        await publish_pending(store, PlatformClient(authorization, http))
    assert methods == ["PrepareMilesCaptureUpload"]
    assert await store.pending_captures() == []
    assert path.read_bytes() == payload


async def test_mount_commit_precedes_index_and_conflicts_never_replace_evidence(config, attempt, store):
    evidence, _, path = archive_fixture(attempt, config.artifact_directory)

    def broken_commit():
        raise OSError("mount commit failed")

    original = store.sync
    store.sync = PayloadSync(commit=broken_commit, reload=lambda: None)
    with pytest.raises(OSError):
        await store.enqueue_capture(evidence, path)
    assert await store.pending_captures() == []
    store.sync = original
    await store.enqueue_capture(evidence, path)
    conflict = evidence.model_copy(
        update={"capture": evidence.capture.model_copy(update={"payload_sha256": "0" * 64})}
    )
    with pytest.raises(ValueError, match="identity conflict"):
        await store.enqueue_capture(conflict, path)
    assert await store.pending_captures() == [(evidence, path)]


async def test_corrupt_local_bytes_stay_pending_and_never_reach_platform(config, authorization, attempt, store):
    evidence, _, path = archive_fixture(attempt, config.artifact_directory)
    await store.enqueue_capture(evidence, path)
    path.write_bytes(b"corrupt")

    def api(request):
        pytest.fail("Corrupt bytes must fail locally")

    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as http:
        await publish_pending(store, PlatformClient(authorization, http))
    row = await (
        await store.connection.execute("SELECT published, last_error_type FROM proximal_capture_archives")
    ).fetchone()
    assert row == (False, "ValueError")


async def test_shutdown_during_upload_keeps_outbox_for_resume(config, authorization, attempt, store):
    evidence, _, path = archive_fixture(attempt, config.artifact_directory)
    await store.enqueue_capture(evidence, path)
    started = asyncio.Event()
    never = asyncio.Event()

    async def api(request):
        if request.method == "PUT":
            started.set()
            await never.wait()
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "attachment": {"state": "pending", "receipt": body["receipt"], "receiptSha256": "f" * 64},
                "uploadUrl": "https://s3.test/object",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as http:
        worker = asyncio.create_task(publish_pending(store, PlatformClient(authorization, http)))
        await asyncio.wait_for(started.wait(), 2)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
    assert await store.pending_captures() == [(evidence, path)]
