import json

import httpx
import pytest
from pydantic import ValidationError

from miles_plugins.proximal import clients
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.clients import CaptureClient, IneligibleAttempt, LaunchFailed, PlatformClient
from miles_plugins.proximal.contracts import RunConfig, SessionHandle, digest


def session(config, attempt):
    return SessionHandle(
        session_id="a" * 32,
        rollout_id=f"{attempt.attempt_id}-rollout-0",
        base_url=f"{config.capture.url}/rollouts/{attempt.attempt_id}-rollout-0/v1",
        request_sha256=digest(attempt),
    )


@pytest.mark.parametrize(
    "mutation", ["missing_grade", "wrong_source", "wrong_image", "execution_error", "wrong_harness"]
)
async def test_completed_transport_is_not_enough_for_training(config, authorization, attempt, mutation):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, mutation))) as client:
        with pytest.raises(IneligibleAttempt):
            await PlatformClient(authorization, client).execute(attempt, session(config, attempt))


def handler(config, attempt, mutation="", calls=None):
    polls = []

    def handle(request):
        assert request.headers["x-api-key"] == "platform-secret"
        method = request.url.path.rsplit("/", 1)[-1]
        if calls is not None:
            calls.append(method)
        if method == "ListProjectEnvironments":
            if mutation == "not_member":
                return httpx.Response(200, json={"memberships": [{"environmentId": 99}]})
            body = {"memberships": [{"environmentId": 7}]}
        elif method == "CreateEnvironmentRun":
            submitted = json.loads(request.content)
            # Only existing run API fields: routing is the registry endpoint, no credential.
            assert "trainingBinding" not in submitted and submitted["runId"] == attempt.attempt_id
            [agent] = submitted["config"]["agents"]
            assert agent == {
                "agentType": config.harness.agent_type,
                "agentModel": config.platform_route.model,
                "endpointName": config.platform_route.endpoint_name,
                "agentTimeoutSec": config.harness.timeout_seconds,
                "reasoningEffort": "AGENT_REASONING_EFFORT_HIGH",
            }
            assert submitted["autoTriggerAnalysis"] is False
            # The run config's rollout sandbox: Kata + Cloud Hypervisor on Kubernetes.
            assert submitted["deploymentConfig"] == {"nexusExact": {"runtime": "SANDBOX_RUNTIME_KATA_CLH"}}
            assert submitted["config"]["harborOptions"] == {
                "maxTurns": config.harness.max_turns,
                "maxSessionTokens": attempt.sampling.max_sequence_tokens,
                "p2pEnforce": config.harness.p2p_enforce,
            }
            body = {"runId": attempt.attempt_id, "instancesStarted": 1}
        elif method == "GetEnvironmentRunContainers" and (
            mutation == "status_down" or (mutation == "status_outage" and len(polls) < 3)
        ):
            polls.append(method)
            return httpx.Response(520, text="<!DOCTYPE html><title>Origin error</title>")
        elif method == "GetEnvironmentRunContainers" and mutation == "launching" and not polls:
            # Proto JSON omits the agent while the container is still launching.
            polls.append(method)
            body = {
                "runId": attempt.attempt_id,
                "containers": [{"id": "rollout-1", "status": "ROLLOUT_CONTAINER_STATUS_LAUNCHING"}],
            }
        elif method == "GetEnvironmentRunContainers" and mutation == "malformed":
            body = {"runId": attempt.attempt_id, "containers": [{"id": "rollout-1", "status": 3}]}
        elif method == "GetEnvironmentRunContainers" and mutation in ("launch_failed", "container_error"):
            error = (
                "Launch failed: [resource_exhausted] container-lease admission queue wait exceeded 75000ms"
                if mutation == "launch_failed"
                else "sandbox exited with code 137"
            )
            body = {
                "runId": attempt.attempt_id,
                "containers": [
                    {"id": "rollout-1", "status": "ROLLOUT_CONTAINER_STATUS_ERROR", "agentType": "x", "error": error}
                ],
            }
        elif method == "GetEnvironmentRunContainers":
            body = {
                "runId": attempt.attempt_id,
                "containers": [
                    {
                        "id": "rollout-1",
                        "status": "ROLLOUT_CONTAINER_STATUS_SUCCESS",
                        "agentType": "other" if mutation == "wrong_harness" else config.harness.agent_type,
                        "rewardScored": mutation != "missing_grade",
                        "error": "pipeline failed" if mutation == "execution_error" else None,
                    }
                ],
            }
        elif method == "GetRunSummary":
            body = {
                "runId": attempt.attempt_id,
                "imageId": 9 if mutation == "wrong_image" else 8,
                "sourceCommitSha": "e" * 40 if mutation == "wrong_source" else attempt.task.source_commit_sha,
            }
        elif method == "GetHarborRolloutDetail":
            body = {"isHarbor": True, "status": "error" if mutation == "execution_error" else "success"}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json=body)

    return handle


async def test_a_rollout_that_never_launched_is_told_apart_from_one_that_failed(config, authorization, attempt):
    """Only the platform's launch failure is retryable (rollout.execute_with_launch_retry)."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, "launch_failed"))) as client:
        with pytest.raises(LaunchFailed, match="admission queue"):
            await PlatformClient(authorization, client).execute(attempt, session(config, attempt))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, "container_error"))) as client:
        with pytest.raises(IneligibleAttempt) as failed:
            await PlatformClient(authorization, client).execute(attempt, session(config, attempt))
        assert not isinstance(failed.value, LaunchFailed)


async def test_a_launching_container_without_an_agent_is_still_polled(config, authorization, attempt):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, "launching"))) as client:
        grade = await PlatformClient(authorization, client).execute(attempt, session(config, attempt))
    assert grade.status == "success"


async def test_a_status_read_rides_out_a_platform_outage(config, authorization, attempt, monkeypatch):
    """Run 013 dropped a whole group on one 520 from a status poll; the rollout was still running."""
    monkeypatch.setattr(clients, "REQUEST_ATTEMPTS", 1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, "status_outage"))) as client:
        grade = await PlatformClient(authorization, client).execute(attempt, session(config, attempt))
    assert grade.status == "success"


async def test_a_lasting_outage_still_fails_the_attempt(config, authorization, attempt, monkeypatch):
    monkeypatch.setattr(clients, "REQUEST_ATTEMPTS", 1)
    monkeypatch.setattr(clients, "STATUS_OUTAGE_SECONDS", 0.05)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, "status_down"))) as client:
        with pytest.raises(httpx.HTTPStatusError, match="520"):
            await PlatformClient(authorization, client).execute(attempt, session(config, attempt))


@pytest.mark.parametrize(("status", "retried"), [(520, True), (500, True), (503, True), (404, False), (409, False)])
async def test_only_transient_statuses_are_retried(status, retried, monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(clients.asyncio, "sleep", no_sleep)
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={}) if len(calls) > 1 else httpx.Response(status, text="failed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        if retried:
            response = await clients.request(client, "POST", "https://platform.test/x", headers={})
            assert response.status_code == 200 and len(calls) == 2
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await clients.request(client, "POST", "https://platform.test/x", headers={})
            assert len(calls) == 1


async def test_a_load_probe_is_one_try(authorization):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503, text="busy")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await CaptureClient(authorization, client).load("any")
    assert len(calls) == 1


async def test_a_malformed_platform_reply_fails_the_attempt_not_the_run(config, authorization, attempt):
    """Only IneligibleAttempt is contained to its group; any other error stops the rollout worker."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, "malformed"))) as client:
        with pytest.raises(IneligibleAttempt, match="Containers reply broke the wire contract at containers.0.status"):
            await PlatformClient(authorization, client).execute(attempt, session(config, attempt))


@pytest.mark.parametrize(
    ("sandbox", "deployment"),
    [
        ("ecs-fargate", None),
        ("gvisor", {"nexusExact": {"runtime": "SANDBOX_RUNTIME_GVISOR"}}),
        ("kata-clh", {"nexusExact": {"runtime": "SANDBOX_RUNTIME_KATA_CLH"}}),
        ("kata-qemu", {"nexusExact": {"runtime": "SANDBOX_RUNTIME_KATA_QEMU"}}),
    ],
)
def test_the_rollout_sandbox_is_the_run_configs(config, attempt, sandbox, deployment):
    """ECS on Fargate is the platform's default placement; Kubernetes sandboxes name their runtime."""
    chosen = config.model_copy(update={"rollout_sandbox": sandbox})
    request = PlatformClient(authorize_run(chosen, yes_rollouts=True, yes_publish=True), None).run_request(attempt)
    assert request.get("deploymentConfig") == deployment


async def test_zero_reward_and_ordering(config, authorization, attempt):
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, calls=calls))) as client:
        grade = await PlatformClient(authorization, client).execute(attempt, session(config, attempt))
    assert grade.reward == 0
    assert calls.index("ListProjectEnvironments") < calls.index("CreateEnvironmentRun")


async def test_a_task_that_left_the_project_never_launches(config, authorization, attempt):
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler(config, attempt, "not_member", calls))
    ) as client:
        with pytest.raises(ValueError, match="no longer in the declared platform project"):
            await PlatformClient(authorization, client).execute(attempt, session(config, attempt))
    assert calls == ["ListProjectEnvironments"]


def test_semantic_choices_are_explicit(config):
    data = config.model_dump(mode="json")
    del data["research"]["max_policy_lag"]
    with pytest.raises(ValidationError):
        RunConfig.model_validate_json(json.dumps(data))
    with pytest.raises(PermissionError):
        authorize_run(config, yes_rollouts=True, yes_publish=False)
    data = config.model_dump(mode="json")
    data["research"]["sampling"]["temperature"] = 0.7
    with pytest.raises(ValidationError):
        RunConfig.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("field", ["base_url", "rollout_id"])
async def test_capture_rejects_credential_redirect(config, authorization, attempt, field):
    rollout = f"{attempt.attempt_id}-rollout-0"
    binding = {
        "session_id": "a" * 32,
        "rollout_id": rollout,
        "base_url": f"{config.capture.url}/rollouts/{rollout}/v1",
        "request_sha256": digest(attempt),
    }
    # A redirected route, or a session pinned to another rollout's replica.
    binding[field] = "https://untrusted.example" if field == "base_url" else "other-rollout-0"

    def handle(request):
        return httpx.Response(200, json=binding)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="mismatched"):
            await CaptureClient(authorization, client).create(attempt)
