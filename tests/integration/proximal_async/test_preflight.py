"""The preflight checks against a real capture (Miles TITO on CPU); only the engine is scripted."""

import os

import httpx
import pytest
from tests.integration.proximal_async.test_capture import scripted_engine

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.capture_server import CaptureServer, EngineEndpoint, capture_tokenizer
from miles_plugins.proximal.preflight import PreflightFailed, canary, check_serving


@pytest.fixture
def tokenizer(config):
    return capture_tokenizer(os.environ["PROXIMAL_TEST_TOKENIZER"], config.tito_model)


@pytest.fixture
def replica(config, authorization, policy, tokenizer, tmp_path):
    """A capture composed like a replica's, deployed from ``config``."""
    requests: list[dict[str, object]] = []
    gateway = httpx.AsyncClient(transport=httpx.MockTransport(scripted_engine(config, policy, tokenizer, requests)))

    async def admits(candidate):
        return candidate.snapshot == policy.snapshot

    server = CaptureServer(
        authorization,
        tokenizer=tokenizer,
        engine=EngineEndpoint(client=gateway, url="http://replica-gateway", headers={"Authorization": "Bearer g"}),
        policy_known=admits,
        root=tmp_path / "capture",
    )
    return server, requests


async def test_serving_check_accepts_a_new_run_and_refuses_a_different_protocol(config, replica):
    server, _ = replica
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
        sampling = config.research.sampling
        next_run = config.model_copy(
            update={
                "run_id": "next-run",
                "research": config.research.model_copy(
                    update={"sampling": sampling.model_copy(update={"max_tokens": sampling.max_tokens // 2})}
                ),
            }
        )
        assert await check_serving(authorize_run(next_run, yes_rollouts=True, yes_publish=True), http, probes=3) == 1

        other_effort = config.model_copy(
            update={"model_protocol": config.model_protocol.model_copy(update={"reasoning_effort": "low"})}
        )
        with pytest.raises(PreflightFailed, match="model_protocol.*stop the app first"):
            await check_serving(authorize_run(other_effort, yes_rollouts=True, yes_publish=True), http, probes=3)


async def test_canary_seals_a_call_and_gets_the_context_limit_error(config, authorization, policy, replica):
    server, requests = replica
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
        report = await canary(authorization, http, policy)
    assert report["sealed_tokens"] > 0
    # The over-long turn never reached the engine, and both sessions were released.
    assert len(requests) == 1 and not server.sessions


async def test_canary_fails_when_an_over_long_turn_is_not_a_context_limit_error(
    authorization, policy, replica, monkeypatch
):
    """Run 004's failure: the agent got an error it could not recognize, and failed the rollout."""
    from fastapi.responses import Response

    from miles_plugins.proximal import capture_server

    server, _ = replica
    monkeypatch.setattr(
        capture_server,
        "context_limit_error",
        lambda limit, detail: Response('{"detail": "Sequence token budget exhausted"}', status_code=422),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
        with pytest.raises(PreflightFailed, match="context-limit error \\(422\\)"):
            await canary(authorization, http, policy)
    assert not server.sessions
