"""Capture in the serving replicas: sticky routing by rollout, and the replica's front process."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.clients import CaptureClient
from miles_plugins.proximal.contracts import (
    AFFINITY_HEADER,
    Attempt,
    Policy,
    RunConfig,
    affinity_headers,
    digest,
    platform_rollout_id,
)
from miles_plugins.proximal.serve_replica import front
from miles_plugins.proximal.snapshot import SnapshotReference

STAGE_A = Path(__file__).resolve().parents[3] / "examples" / "proximal" / "e2e" / "run.stage-a.json"
POOL = "https://pool.modal.direct"


def test_affinity_is_the_sha256_of_the_platform_rollout_id():
    # The platform's rollout_capture client derives the same value from the rollout ID
    # it already has, so both sides land on one replica without sharing any state.
    rollout = platform_rollout_id("0123456789abcdef0123456789abcdef")
    assert rollout == "0123456789abcdef0123456789abcdef-rollout-0"
    assert affinity_headers(rollout) == {AFFINITY_HEADER: hashlib.sha256(rollout.encode()).hexdigest()}
    assert AFFINITY_HEADER == "Modal-Session-Id"


@pytest.fixture
def run(monkeypatch) -> RunConfig:
    raw = json.loads(STAGE_A.read_text())
    raw["inference_url"] = raw["capture"]["url"] = POOL
    config = RunConfig.model_validate_json(json.dumps(raw))
    monkeypatch.setenv(config.capture.api_key_env, "admin-secret")
    return config


def _attempt(config: RunConfig) -> Attempt:
    return Attempt(
        attempt_id="0123456789abcdef0123456789abcdef",
        run_id=config.run_id,
        group_id="group-1",
        sample_index=0,
        dataset_sha256=digest(config.dataset),
        task=config.dataset.tasks[0],
        harness=config.harness,
        policy=Policy(
            run_id=config.run_id, version=1, snapshot=SnapshotReference(sha256="a" * 64), base_model=config.base_model
        ),
        sampling=config.research.sampling,
    )


async def test_the_trainer_pins_every_session_call_to_the_rollouts_replica(run):
    attempt = _attempt(run)
    rollout = platform_rollout_id(attempt.attempt_id)
    seen: list[tuple[str, str, str | None]] = []

    def pool(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get(AFFINITY_HEADER)))
        if request.url.path == "/sessions":
            return httpx.Response(
                200,
                json={
                    "session_id": "s" * 32,
                    "rollout_id": rollout,
                    "base_url": f"{POOL}/rollouts/{rollout}/v1",
                    "request_sha256": digest(attempt),
                },
            )
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(pool)) as http:
        client = CaptureClient(authorize_run(run, yes_rollouts=True, yes_publish=True), http)
        handle = await client.create(attempt)
        await client.release(handle)
    expected = affinity_headers(rollout)[AFFINITY_HEADER]
    assert seen == [("POST", "/sessions", expected), ("DELETE", f"/sessions/{'s' * 32}", expected)]


async def test_front_sends_capture_routes_to_capture_and_the_rest_to_the_gateway():
    def app(name: str):
        async def asgi(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": name.encode()})

        return asgi

    routes = {
        "/sessions": "capture",
        "/sessions/abc/seal": "capture",
        "/rollouts/r-rollout-0/v1/chat/completions": "capture",
        "/capture/contract": "capture",
        "/v1/chat/completions": "gateway",
        "/policies/prepare": "gateway",
        "/health": "gateway",
    }
    transport = httpx.ASGITransport(app=front(app("gateway"), app("capture")))
    async with httpx.AsyncClient(transport=transport, base_url="http://replica") as http:
        for path, owner in routes.items():
            assert (await http.post(path)).text == owner, path
