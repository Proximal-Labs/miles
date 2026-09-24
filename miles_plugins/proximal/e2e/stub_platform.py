"""A local stand-in for the Proximal platform's run API, for the Stage A harness.

It answers the Connect calls Miles's PlatformClient makes, and acts as the endpoint
registry: a run whose agent names the configured endpoint gets the base URL
``<capture url>/rollouts/<run id>-rollout-0/v1`` (the platform's rollout ID) and the
registry's static credential.

Each run plays agent-px's mini-swe traffic, as read from proximal-mono ``593de5e46063``
(agent-px/provider/openai-chat-completions/src/internals.ts,
backend/.../rollout-solver/rolloutSolverMiniSweAgent.ts, agents/interfaces/src/mini-swe/pxd.ts):
streamed requests with ``stream_options.include_usage``, ``prompt_cache_key``,
``max_completion_tokens`` and ``reasoning_effort``; one ``strict`` bash tool with a
closed schema; assistant turns rebuilt from parsed blocks (``content: null`` when
empty, ``reasoning_content``, compact re-serialized tool arguments, ``"{}"`` when
unparseable); pretty-printed JSON tool results; the format-error reminder after a
reply without a tool call (at most twice); and the submission command ends the run.
It is a pinned copy, not agent-px: Stage B against the real platform is the check.
The grade is deterministic per run ID so groups carry mixed rewards.

    python -m miles_plugins.proximal.e2e.stub_platform --config run.json --port 9010
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from miles_plugins.proximal.contracts import RunConfig, affinity_headers, platform_rollout_id, read_run_config

RewardRule = Literal["mixed", "zero", "one"]

SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer."
SUBMISSION_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FORMAT_REMINDERS = 2
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the repository workspace.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute in the repository workspace.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}
NO_TOOL_CALL = (
    "Tool call error:\n\n<error>\nNo tool calls found in the response.\n</error>\n\n"
    "Here is general guidance on how to submit correct toolcalls:\n\n"
    "Every response needs to use the 'bash' tool at least once to execute commands.\n\n"
    "Call the bash tool with your command as the argument:\n"
    "- Tool: bash\n"
    '- Arguments: {"command": "your_command_here"}\n\n'
    f"If you want to end the task, please issue the following command: `{SUBMISSION_COMMAND}`\n"
    "without any other command."
)
TRUNCATED_REASONING = "[reasoning was cut off at the output-token limit before any tool call or answer was produced]"


class AssembledTurn(TypedDict):
    content: str
    reasoning: str
    calls: list[dict[str, str]]


def assemble_stream(body: str) -> tuple[AssembledTurn, str, dict[str, object] | None]:
    """agent-px's stream assembly: concatenate deltas, tool calls keyed by index."""
    content, reasoning = "", ""
    finish: str | None = None
    usage: dict[str, object] | None = None
    calls: dict[int, dict[str, str]] = {}
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: ") :])
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                entry = calls.setdefault(call["index"], {"id": "", "name": "", "arguments": ""})
                entry["id"] = call.get("id") or entry["id"]
                function = call.get("function") or {}
                entry["name"] = function.get("name") or entry["name"]
                entry["arguments"] += function.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    if finish is None:
        raise RuntimeError("Stream ended without a finish_reason")
    parsed: AssembledTurn = {"content": content, "reasoning": reasoning, "calls": [calls[i] for i in sorted(calls)]}
    return parsed, finish, usage


def replay_assistant(parsed: AssembledTurn, finish: str | None) -> dict[str, object]:
    """agent-px's rebuilt assistant history message, not the provider's raw message."""
    content, reasoning = parsed["content"], parsed["reasoning"]
    calls = [call for call in parsed["calls"] if call["id"] and call["name"]]
    if finish == "length" and reasoning and not content and not calls:
        return {"role": "assistant", "content": TRUNCATED_REASONING}
    message: dict[str, object] = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        rebuilt = []
        for call in calls:
            try:
                arguments = json.dumps(json.loads(call["arguments"]), separators=(",", ":"), ensure_ascii=False)
            except json.JSONDecodeError:
                arguments = "{}"
            rebuilt.append(
                {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": arguments}}
            )
        message["tool_calls"] = rebuilt
    return message


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", alias_generator=to_camel, populate_by_name=True)


class HarborOptions(_Wire):
    max_turns: int
    max_session_tokens: int


class Agent(_Wire):
    agent_type: str
    agent_model: str
    endpoint_name: str | None = None  # Unset: the model's default registry endpoint.
    reasoning_effort: str


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
    def __init__(
        self, run: RunConfig, *, api_key: str, capture_key: str, reward: RewardRule, client: httpx.AsyncClient
    ):
        self.run, self.api_key, self.capture_key, self.reward, self.client = run, api_key, capture_key, reward, client
        self.runs: dict[str, RunState] = {}
        self.app = FastAPI()
        self._routes()

    def _authorize(self, request: Request) -> None:
        if not hmac.compare_digest(request.headers.get("x-api-key", ""), self.api_key):
            raise HTTPException(401, "Invalid platform API key")

    async def _agent(self, state: RunState) -> None:
        run_id = state.request.run_id
        url = f"{self.run.capture.url}/rollouts/{platform_rollout_id(run_id)}/v1/chat/completions"
        # Like the platform's rollout_capture client: sticky to the replica holding the session.
        headers = {"Authorization": f"Bearer {self.capture_key}"} | affinity_headers(platform_rollout_id(run_id))
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Implement the feature for environment {state.request.environment_id} "
                f"at commit {state.request.source_commit_sha[:12]}.",
            },
        ]
        reminders = 0
        try:
            for turn in range(state.request.config.harbor_options.max_turns):
                reply = await self.client.post(
                    url,
                    headers=headers,
                    json={
                        "model": self.run.base_model.name,  # The registry entry's wire model.
                        "messages": messages,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "prompt_cache_key": f"mini-swe:{run_id}",
                        "tools": [BASH_TOOL],
                        "max_completion_tokens": self.run.research.sampling.max_tokens,
                        "reasoning_effort": state.request.config.agents[0].reasoning_effort,
                    },
                )
                reply.raise_for_status()
                parsed, finish, _ = assemble_stream(reply.text)
                assistant = replay_assistant(parsed, finish)
                messages.append(assistant)
                state.turns = turn + 1
                calls = assistant.get("tool_calls") or []
                if not calls:
                    if reminders == FORMAT_REMINDERS:
                        raise RuntimeError("No tool call after the format reminders")
                    reminders += 1
                    messages.append({"role": "user", "content": NO_TOOL_CALL})
                    continue
                submitted = False
                for call in calls:  # type: ignore[attr-defined]
                    command = json.loads(call["function"]["arguments"]).get("command", "")
                    submitted = submitted or command.strip() == SUBMISSION_COMMAND
                    result = {"returncode": 0, "output": "" if submitted else "README.md\nsrc\n"}
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, indent=2)}
                    )
                if submitted:
                    break
            state.reward = grade(run_id, self.reward)
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
        [agent] = body.config.agents
        route = self.run.platform_route
        if body.instances != 1 or (agent.agent_model, agent.endpoint_name) != (route.model, route.endpoint_name):
            raise HTTPException(400, "Stub registry serves one instance on the configured endpoint")
        # Like the platform, the reasoning effort reaches agent-px as its lowercase name.
        agent.reasoning_effort = agent.reasoning_effort.removeprefix("AGENT_REASONING_EFFORT_").lower()
        if (existing := self.runs.get(body.run_id)) is not None:
            if existing.request != body:
                raise HTTPException(409, "Run ID reused with different inputs")
            started = 0
        else:
            state = RunState(body)
            state.task = asyncio.create_task(self._agent(state))
            self.runs[body.run_id] = state
            started = 1
        return {"runId": body.run_id, "instancesStarted": started}

    def _state(self, run_id: str) -> RunState:
        if (state := self.runs.get(run_id)) is None:
            raise HTTPException(404, "Unknown run")
        return state

    def _routes(self) -> None:
        service = "/proximal.v1.EnvironmentRunService"

        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

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
    parser.add_argument("--reward", choices=["mixed", "zero", "one"], default="mixed")
    args = parser.parse_args()
    run = read_run_config(args.config)
    api_key = os.environ[run.platform.api_key_env]
    capture_key = os.environ[run.capture.platform_key_env]

    import uvicorn

    async def serve() -> None:
        async with httpx.AsyncClient(timeout=run.request_timeout_seconds) as client:
            stub = StubPlatform(run, api_key=api_key, capture_key=capture_key, reward=args.reward, client=client)
            await uvicorn.Server(uvicorn.Config(stub.app, host=args.host, port=args.port, access_log=False)).serve()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
