import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.clients import CaptureClient, IneligibleAttempt, PlatformClient
from miles_plugins.proximal.contracts import RunConfig, SessionHandle, digest


def session(config, attempt):
    return SessionHandle(
        session_id="a" * 32,
        base_url=f"{config.capture.url}/sessions/" + "a" * 32,
        api_key=SecretStr("session-only-secret"),
        request_sha256=digest(attempt),
    )


@pytest.mark.parametrize(
    "mutation", ["missing_grade", "wrong_source", "wrong_digest", "execution_error", "wrong_harness"]
)
async def test_completed_transport_is_not_enough_for_training(config, authorization, attempt, mutation):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, mutation))) as client:
        with pytest.raises(IneligibleAttempt):
            await PlatformClient(authorization, client).execute(attempt, session(config, attempt))


def handler(config, attempt, mutation="", calls=None):
    def handle(request):
        assert request.headers["x-api-key"] == "platform-secret"
        method = request.url.path.rsplit("/", 1)[-1]
        if calls is not None:
            calls.append(method)
        if method == "GetTrainingCapabilities":
            if mutation == "old_server":
                return httpx.Response(404)
            body = {
                "protocolVersion": 1,
                "harnessRevision": config.harness.revision,
                "supportedAgentTypes": [config.harness.agent_type],
                "pinnedSource": True,
                "linearTito": True,
            }
        elif method == "ListProjectEnvironments":
            body = {"memberships": [{"environmentId": 7}]}
        elif method == "CreateEnvironmentRun":
            submitted = json.loads(request.content)
            assert submitted["trainingBinding"]["sessionApiKey"] == "session-only-secret"
            assert submitted["trainingBinding"]["requestSha256"] == digest(attempt)
            assert submitted["autoTriggerAnalysis"] is False
            body = {"runId": attempt.attempt_id, "instancesStarted": 1, "trainingRequestSha256": digest(attempt)}
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
                "imageId": 8,
                "sourceCommitSha": "e" * 40 if mutation == "wrong_source" else attempt.task.source_commit_sha,
                "trainingRequestSha256": "f" * 64 if mutation == "wrong_digest" else digest(attempt),
            }
        elif method == "GetHarborRolloutDetail":
            body = {"isHarbor": True, "status": "error" if mutation == "execution_error" else "success"}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json=body)

    return handle


async def test_zero_reward_and_ordering(config, authorization, attempt):
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(config, attempt, calls=calls))) as client:
        grade = await PlatformClient(authorization, client).execute(attempt, session(config, attempt))
    assert grade.reward == 0
    assert calls.index("GetTrainingCapabilities") < calls.index("CreateEnvironmentRun")
    assert calls.index("ListProjectEnvironments") < calls.index("CreateEnvironmentRun")


async def test_old_platform_cannot_silently_launch_default_model(config, authorization, attempt):
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler(config, attempt, "old_server", calls))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await PlatformClient(authorization, client).execute(attempt, session(config, attempt))
    assert calls == ["GetTrainingCapabilities"]


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


async def test_capture_rejects_credential_redirect(config, authorization, attempt):
    def handle(request):
        return httpx.Response(
            200,
            json={
                "session_id": "a" * 32,
                "base_url": "https://untrusted.example",
                "api_key": "credential",
                "request_sha256": digest(attempt),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="mismatched"):
            await CaptureClient(authorization, client).create(attempt)
