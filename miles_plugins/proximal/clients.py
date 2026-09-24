"""HTTP adapters for existing Connect run RPCs and the Miles capture service.

The training-binding RPC extensions are specified in platform-contract.md.
No container, replica or Volume lifecycle operations exist in this client.
"""

import asyncio
import hashlib
import time

import httpx
from pydantic import BaseModel, ConfigDict, FiniteFloat
from pydantic.alias_generators import to_camel

from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization, secret_env
from miles_plugins.proximal.contracts import (
    Attempt,
    CaptureReceipt,
    Grade,
    Policy,
    PolicyEvidence,
    SessionHandle,
    digest,
    pinned_dataset,
)


class Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, alias_generator=to_camel, populate_by_name=True)


class Membership(Wire):
    environment_id: int


class Memberships(Wire):
    memberships: list[Membership]


class CreatedRun(Wire):
    run_id: str
    instances_started: int = 0  # Proto JSON omits zero: an idempotent replay starts none.


class Container(Wire):
    id: str
    status: str
    agent_type: str
    reward_scored: bool = False
    reward: FiniteFloat | None = None
    error: str | None = None


class Containers(Wire):
    run_id: str
    containers: list[Container] = []


class Summary(Wire):
    run_id: str
    image_id: int
    source_commit_sha: str


class IneligibleAttempt(RuntimeError):
    """An execution failure, never a fabricated zero-reward training example."""


async def request(
    client: httpx.AsyncClient, method: str, url: str, *, headers: dict[str, str], body: object = None
) -> httpx.Response:
    """Bounded retries for idempotent calls; identities are stable across retries."""
    for attempt in range(3):
        try:
            response = await client.request(method, url, headers=headers, json=body, follow_redirects=False)
            if response.status_code not in (429, 502, 503, 504) or attempt == 2:
                response.raise_for_status()
                return response
        except httpx.TransportError:
            if attempt == 2:
                raise
        await asyncio.sleep(0.2 * (2**attempt))
    raise AssertionError("unreachable")


class CaptureClient:
    def __init__(self, authorization: AuthorizedRun, client: httpx.AsyncClient):
        self.config = require_authorization(authorization)
        self.client = client
        self.headers = {"Authorization": f"Bearer {secret_env(self.config.capture.api_key_env)}"}
        self.url = self.config.capture.url

    async def create(self, attempt: Attempt) -> SessionHandle:
        response = await request(
            self.client, "POST", f"{self.url}/sessions", headers=self.headers, body=attempt.model_dump(mode="json")
        )
        handle = SessionHandle.model_validate_json(response.content)
        expected = f"{self.url}/rollouts/{attempt.attempt_id}-rollout-0/v1"
        if handle.request_sha256 != digest(attempt) or handle.base_url != expected:
            raise ValueError("Session service returned a mismatched binding")
        return handle

    def _session(self, handle: SessionHandle) -> str:
        return f"{self.url}/sessions/{handle.session_id}"

    async def collect(self, handle: SessionHandle, attempt: Attempt) -> tuple[CaptureReceipt, bytes]:
        response = await request(self.client, "POST", f"{self._session(handle)}/seal", headers=self.headers)
        receipt = CaptureReceipt.model_validate_json(response.content)
        if (
            receipt.session_id != handle.session_id
            or receipt.request_sha256 != digest(attempt)
            or receipt.policy != attempt.policy
        ):
            raise ValueError("Sealed capture provenance differs from the attempt")
        payload = (await request(self.client, "GET", f"{self._session(handle)}/samples", headers=self.headers)).content
        if hashlib.sha256(payload).hexdigest() != receipt.payload_sha256:
            raise ValueError("Sealed capture payload checksum mismatch")
        return receipt, payload

    async def release(self, handle: SessionHandle) -> None:
        await request(self.client, "DELETE", self._session(handle), headers=self.headers)


class ServingPoolClient:
    """The Miles-owned serving pool's control surface: warm and verify a version.

    Every replica independently loads and verifies the version each request names,
    so this does not assume broadcast or sticky routing. It proves one replica can
    serve the published weights before the version becomes selectable.
    """

    def __init__(self, authorization: AuthorizedRun, client: httpx.AsyncClient):
        self.config = require_authorization(authorization)
        self.client = client
        self.headers = {name: secret_env(env) for name, env in self.config.inference_header_env.items()}

    async def prepare(self, policy: Policy) -> None:
        response = await request(
            self.client,
            "POST",
            f"{self.config.inference_url}/policies/prepare",
            headers=self.headers,
            body={"snapshot": policy.snapshot.model_dump(), "base_model": policy.base_model.model_dump()},
        )
        evidence = PolicyEvidence.model_validate_json(response.content)
        if (
            evidence.snapshot != policy.snapshot
            or evidence.base_model != policy.base_model
            or evidence.request_model != f"{policy.base_model.name}:miles-{policy.snapshot.sha256}"
        ):
            raise ValueError("Replica did not verify the published policy")


class PlatformClient:
    def __init__(self, authorization: AuthorizedRun, client: httpx.AsyncClient):
        self.config = require_authorization(authorization)
        self.client = client
        self.headers = {"x-api-key": secret_env(self.config.platform.api_key_env), "Connect-Protocol-Version": "1"}
        self.url = f"{self.config.platform.url}/proximal.v1.EnvironmentRunService"
        self._membership_checked = False

    async def _rpc(self, name: str, body: object) -> httpx.Response:
        return await request(self.client, "POST", f"{self.url}/{name}", headers=self.headers, body=body)

    async def preflight(self) -> None:
        """Free check before the first paid run: the pinned tasks are still project members."""
        if self._membership_checked:
            return
        response = await request(
            self.client,
            "POST",
            f"{self.config.platform.url}/proximal.v1.ProjectService/ListProjectEnvironments",
            headers=self.headers,
            body={"projectId": self.config.dataset.project_id},
        )
        members = Memberships.model_validate_json(response.content)
        present = {member.environment_id for member in members.memberships}
        if any(task.environment_id not in present for task in self.config.dataset.tasks):
            raise ValueError("Pinned task is no longer in the declared platform project")
        self._membership_checked = True

    def run_request(self, attempt: Attempt) -> dict[str, object]:
        """A CreateEnvironmentRun body using only existing run API fields.

        The run ID is the attempt ID: retries are idempotent, and it is the key the
        registry puts in the capture rollout route. Routing to capture is the
        platform route's endpoint name; no credential or session URL is sent.
        """
        route = self.config.platform_route
        return {
            "runId": attempt.attempt_id,
            "environmentId": attempt.task.environment_id,
            "imageId": attempt.task.image_id,
            "sourceCommitSha": attempt.task.source_commit_sha,
            "instances": 1,
            "ensureRolloutLaunchWorkflows": True,
            "autoTriggerAnalysis": False,
            "autoTriggerPostQa": False,
            "config": {
                "agents": [
                    {
                        "agentType": attempt.harness.agent_type,
                        "agentModel": route.model,
                        "endpointName": route.endpoint_name,
                        "agentTimeoutSec": attempt.harness.timeout_seconds,
                        "reasoningEffort": f"AGENT_REASONING_EFFORT_{self.config.model_protocol.reasoning_effort.upper()}",
                    }
                ],
                "harborOptions": {
                    "maxTurns": attempt.harness.max_turns,
                    "maxSessionTokens": attempt.sampling.max_sequence_tokens,
                    "p2pEnforce": attempt.harness.p2p_enforce,
                },
            },
        }

    async def execute(self, attempt: Attempt, session: SessionHandle) -> Grade:
        if (
            attempt.run_id != self.config.run_id
            or attempt.harness != self.config.harness
            or attempt.task not in pinned_dataset(self.config.dataset).tasks
            or attempt.sampling != self.config.research.sampling
            or attempt.dataset_sha256 != pinned_dataset(self.config.dataset).sha256
            or attempt.policy.base_model != self.config.base_model
            or session.request_sha256 != digest(attempt)
        ):
            raise ValueError("Attempt is outside the authorized run")
        await self.preflight()
        response = await self._rpc("CreateEnvironmentRun", self.run_request(attempt))
        created = CreatedRun.model_validate_json(response.content)
        if created.run_id != attempt.attempt_id or created.instances_started not in (0, 1):
            raise IneligibleAttempt("Platform did not acknowledge the exact single-rollout request")
        deadline = time.monotonic() + attempt.harness.timeout_seconds + self.config.request_timeout_seconds
        while time.monotonic() < deadline:
            reply = Containers.model_validate_json(
                (await self._rpc("GetEnvironmentRunContainers", {"runId": created.run_id})).content
            )
            if reply.run_id != created.run_id or len(reply.containers) > 1:
                raise IneligibleAttempt("Platform returned a different run or multiple rollouts")
            if reply.containers:
                container = reply.containers[0]
                if container.status in {"ROLLOUT_CONTAINER_STATUS_SUCCESS", "ROLLOUT_CONTAINER_STATUS_COMPLETED"}:
                    return await self._grade(attempt, container)
                if container.status not in {
                    "ROLLOUT_CONTAINER_STATUS_PENDING",
                    "ROLLOUT_CONTAINER_STATUS_RUNNING",
                    "ROLLOUT_CONTAINER_STATUS_LAUNCHING",
                }:
                    raise IneligibleAttempt(f"Platform rollout is ineligible: {container.status}")
            await asyncio.sleep(self.config.poll_interval_seconds)
        raise IneligibleAttempt("Platform rollout exceeded its declared deadline")

    async def _grade(self, attempt: Attempt, container: Container) -> Grade:
        summary = Summary.model_validate_json(
            (await self._rpc("GetRunSummary", {"runId": attempt.attempt_id})).content
        )
        if (summary.run_id, summary.image_id, summary.source_commit_sha) != (
            attempt.attempt_id,
            attempt.task.image_id,
            attempt.task.source_commit_sha,
        ):
            raise IneligibleAttempt("Completed run provenance differs from the submitted task")
        if container.agent_type != attempt.harness.agent_type or not container.reward_scored:
            raise IneligibleAttempt("Missing valid verifier grade or wrong harness")
        if container.error:
            raise IneligibleAttempt("Completed rollout also reports an execution error")
        # Proto JSON omits zero; rewardScored distinguishes it from an absent grade.
        return Grade(
            run_id=attempt.attempt_id,
            rollout_id=container.id,
            request_sha256=digest(attempt),
            reward=container.reward if container.reward is not None else 0.0,
            status="success" if container.status.endswith("_SUCCESS") else "completed",
            artifacts_url=None,
        )

    async def cancel(self, attempt: Attempt) -> None:
        # Logical run cancellation; resource decisions stay inside the platform.
        await self._rpc("StopEnvironmentRun", {"runId": attempt.attempt_id})
