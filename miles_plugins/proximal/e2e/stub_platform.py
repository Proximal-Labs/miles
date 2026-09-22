"""A local stand-in for the Proximal platform's run API, for the Stage A harness.

It answers exactly the Connect calls Miles's PlatformClient makes. Creating a run
starts a scripted multi-turn agent that talks to the run's capture session like a
harness would: it offers a bash tool, answers tool calls with a tool result, and
otherwise sends a follow-up user turn. The verifier grade is deterministic per run
ID so groups carry mixed rewards. There are no sandboxes and nothing to tear down.

    python -m miles_plugins.proximal.e2e.stub_platform --config run.json --port 9010
"""

import argparse
import asyncio
import hashlib
import hmac
import os
from dataclasses import dataclass, field
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from miles_plugins.proximal.contracts import RunConfig, read_run_config

RewardRule = Literal["mixed", "zero", "one"]

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", alias_generator=to_camel, populate_by_name=True)


class Binding(_Wire):
    protocol_version: int
    request_sha256: str
    session_base_url: str
    session_api_key: str
    model: str
    harness_revision: str


class HarborOptions(_Wire):
    max_turns: int
    max_session_tokens: int


class Agent(_Wire):
    agent_type: str


class RunOptions(_Wire):
    agents: list[Agent]
    harbor_options: HarborOptions


class CreateRun(_Wire):
    run_id: str
    environment_id: int
    image_id: int
    source_commit_sha: str
    instances: int
    config: RunOptions
    training_binding: Binding


@dataclass
class RunState:
    request: CreateRun
    status: str = "ROLLOUT_CONTAINER_STATUS_RUNNING"
    reward: float | None = None
    error: str | None = None
    turns: int = 0
    task: asyncio.Task[None] | None = field(default=None, repr=False)


def grade(run_id: str, rule: RewardRule) -> float:
    if rule == "zero":
        return 0.0
    if rule == "one":
        return 1.0
    return float(int(hashlib.sha256(run_id.encode()).hexdigest(), 16) % 2)


class StubPlatform:
    def __init__(self, run: RunConfig, *, api_key: str, turns: int, reward: RewardRule, client: httpx.AsyncClient):
        self.run, self.api_key, self.turns, self.reward, self.client = run, api_key, turns, reward, client
        self.runs: dict[str, RunState] = {}
        self.app = FastAPI()
        self._routes()

    def _authorize(self, request: Request) -> None:
        if not hmac.compare_digest(request.headers.get("x-api-key", ""), self.api_key):
            raise HTTPException(401, "Invalid platform API key")

    async def _agent(self, state: RunState) -> None:
        binding = state.request.training_binding
        url = binding.session_base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {binding.session_api_key}"}
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "You are a software engineering agent. Use the bash tool when useful."},
            {
                "role": "user",
                "content": f"Implement the feature for environment {state.request.environment_id} "
                f"at commit {state.request.source_commit_sha[:12]}. Reply DONE when finished.",
            },
        ]
        try:
            limit = min(self.turns, state.request.config.harbor_options.max_turns)
            for turn in range(limit):
                reply = await self.client.post(
                    url,
                    headers=headers,
                    json={"model": binding.model, "messages": messages, "tools": [BASH_TOOL], "stream": False},
                )
                reply.raise_for_status()
                assistant = reply.json()["choices"][0]["message"]
                messages.append(assistant)  # Verbatim, as a harness replays history.
                state.turns = turn + 1
                calls = assistant.get("tool_calls") or []
                if calls:
                    for call in calls:
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "exit code 0\n"})
                elif "DONE" in (assistant.get("content") or ""):
                    break
                else:
                    messages.append({"role": "user", "content": "Continue, then reply DONE."})
            state.reward = grade(state.request.run_id, self.reward)
            state.status = (
                "ROLLOUT_CONTAINER_STATUS_SUCCESS" if state.reward >= 1 else "ROLLOUT_CONTAINER_STATUS_COMPLETED"
            )
        except Exception as exc:  # A harness failure is an execution error, never a zero grade.
            state.status, state.error = "ROLLOUT_CONTAINER_STATUS_ERROR", f"{type(exc).__name__}: {exc}"[:500]

    def _create(self, body: CreateRun) -> dict[str, object]:
        tasks = {task.environment_id: task for task in self.run.dataset.tasks}
        task = tasks.get(body.environment_id)
        if task is None or (task.image_id, task.source_commit_sha) != (body.image_id, body.source_commit_sha):
            raise HTTPException(400, "Unknown environment/image/source for this project")
        if body.instances != 1 or body.training_binding.harness_revision != self.run.harness.revision:
            raise HTTPException(400, "Stub supports one instance of the certified harness revision")
        if (existing := self.runs.get(body.run_id)) is not None:
            if existing.request != body:
                raise HTTPException(409, "Run ID reused with different inputs")
            started = 0
        else:
            state = RunState(body)
            state.task = asyncio.create_task(self._agent(state))
            self.runs[body.run_id] = state
            started = 1
        return {
            "runId": body.run_id,
            "instancesStarted": started,
            "trainingRequestSha256": body.training_binding.request_sha256,
        }

    def _state(self, run_id: str) -> RunState:
        if (state := self.runs.get(run_id)) is None:
            raise HTTPException(404, "Unknown run")
        return state

    def _routes(self) -> None:
        service = "/proximal.v1.EnvironmentRunService"

        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.post(f"{service}/GetTrainingCapabilities")
        async def capabilities(request: Request) -> dict[str, object]:
            self._authorize(request)
            return {
                "protocolVersion": 1,
                "harnessRevision": self.run.harness.revision,
                "supportedAgentTypes": [self.run.harness.agent_type],
                "pinnedSource": True,
                "linearTito": True,
            }

        @self.app.post("/proximal.v1.ProjectService/ListProjectEnvironments")
        async def memberships(request: Request) -> dict[str, object]:
            self._authorize(request)
            return {"memberships": [{"environmentId": task.environment_id} for task in self.run.dataset.tasks]}

        @self.app.post(f"{service}/CreateEnvironmentRun")
        async def create(request: Request) -> dict[str, object]:
            self._authorize(request)
            return self._create(CreateRun.model_validate(await request.json()))

        @self.app.post(f"{service}/GetEnvironmentRunContainers")
        async def containers(request: Request) -> dict[str, object]:
            self._authorize(request)
            state = self._state((await request.json())["runId"])
            container: dict[str, object] = {
                "id": f"{state.request.run_id}-0",
                "status": state.status,
                "agentType": state.request.config.agents[0].agent_type,
            }
            if state.reward is not None:
                container |= {"rewardScored": True, "reward": state.reward}
            if state.error is not None:
                container["error"] = state.error
            return {"runId": state.request.run_id, "containers": [container]}

        @self.app.post(f"{service}/GetRunSummary")
        async def summary(request: Request) -> dict[str, object]:
            self._authorize(request)
            state = self._state((await request.json())["runId"])
            return {
                "runId": state.request.run_id,
                "imageId": state.request.image_id,
                "sourceCommitSha": state.request.source_commit_sha,
                "trainingRequestSha256": state.request.training_binding.request_sha256,
            }

        @self.app.post(f"{service}/StopEnvironmentRun")
        async def stop(request: Request) -> dict[str, object]:
            self._authorize(request)
            state = self.runs.get((await request.json())["runId"])
            if state is None:
                return {}  # Idempotent: cancelled before the run was created.
            if state.task is not None and not state.task.done():
                state.task.cancel()
                state.status = "ROLLOUT_CONTAINER_STATUS_STOPPED"
            return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--reward", choices=["mixed", "zero", "one"], default="mixed")
    args = parser.parse_args()
    run = read_run_config(args.config)
    api_key = os.environ[run.platform.api_key_env]

    import uvicorn

    async def serve() -> None:
        async with httpx.AsyncClient(timeout=run.request_timeout_seconds) as client:
            stub = StubPlatform(run, api_key=api_key, turns=args.turns, reward=args.reward, client=client)
            await uvicorn.Server(uvicorn.Config(stub.app, host=args.host, port=args.port, access_log=False)).serve()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
