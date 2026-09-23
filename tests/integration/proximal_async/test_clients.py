import json

import httpx
import pytest
from pydantic import ValidationError

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.clients import CaptureClient, IneligibleAttempt, PlatformClient
from miles_plugins.proximal.contracts import RunConfig, SessionHandle, digest


def session(config, attempt):
    return SessionHandle(
        session_id="a" * 32,
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
            assert submitted["config"]["harborOptions"] == {
                "maxTurns": config.harness.max_turns,
                "maxSessionTokens": attempt.sampling.max_sequence_tokens,
                "p2pEnforce": config.harness.p2p_enforce,
            }
            body = {"runId": attempt.attempt_id, "instancesStarted": 1}
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


async def test_capture_rejects_credential_redirect(config, authorization, attempt):
    def handle(request):
        return httpx.Response(
            200,
            json={
                "session_id": "a" * 32,
                "base_url": "https://untrusted.example",
                "request_sha256": digest(attempt),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="mismatched"):
            await CaptureClient(authorization, client).create(attempt)
