"""A gsm8k task platform, for proving training on the real topology without sandboxes.

It answers the same run API as the Proximal platform (the stub platform's routes), so
Miles runs unchanged: one platform run per attempt, graded, consumed from the store.
Each run is one gsm8k problem, chosen by ``environment_id`` (row index + 1). Its agent
makes one Chat Completions call through the capture service's per-rollout route, as
agent-px would, and grades the reply with Miles's ``math`` reward (the boxed answer).

Everything else is real: capture records exact tokens and logprobs, replicas serve the
published LoRA version, the trainer trains on what capture recorded.

    python -m miles_plugins.proximal.e2e.math_platform serve --config run.json --data train.parquet --port 9010
    python -m miles_plugins.proximal.e2e.math_platform tasks --data train.parquet --limit 2000
"""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles_plugins.proximal.contracts import RunConfig, read_run_config
from miles_plugins.proximal.e2e.stub_platform import RunState, StubPlatform, assemble_stream

IMAGE_ID = 1


@dataclass(frozen=True)
class Problem:
    messages: list[dict[str, str]]
    label: str


def load_problems(path: Path) -> list[Problem]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["messages", "label"]).to_pylist()
    return [Problem(messages=[dict(m) for m in row["messages"]], label=str(row["label"])) for row in rows]


def data_sha(path: Path) -> str:
    """Pins the dataset file where a platform task pins its source commit (40 hex)."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


class MathPlatform(StubPlatform):
    def __init__(self, run: RunConfig, *, problems: list[Problem], sha: str, **kwargs: object):
        super().__init__(run, reward="zero", **kwargs)  # type: ignore[arg-type]
        self.problems, self.sha = problems, sha
        for task in run.dataset.tasks:
            if task.source_commit_sha != sha or not 1 <= task.environment_id <= len(problems):
                raise ValueError(f"Task {task.environment_id} does not pin this dataset file")

    async def _agent(self, state: RunState) -> None:
        request = state.request
        problem = self.problems[request.environment_id - 1]
        url = f"{self.run.capture.url}/rollouts/{request.run_id}-rollout-0/v1/chat/completions"
        try:
            reply = await self.client.post(
                url,
                headers={"Authorization": f"Bearer {self.capture_key}"},
                json={
                    "model": self.run.base_model.name,
                    "messages": problem.messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_completion_tokens": self.run.research.sampling.max_tokens,
                    "reasoning_effort": request.config.agents[0].reasoning_effort,
                },
            )
            reply.raise_for_status()
            parsed, _finish, _usage = assemble_stream(reply.text)
            state.turns = 1
            state.reward = 1.0 if grade_answer_verl(parsed["content"], problem.label) else 0.0
            state.status = (
                "ROLLOUT_CONTAINER_STATUS_SUCCESS" if state.reward >= 1 else "ROLLOUT_CONTAINER_STATUS_COMPLETED"
            )
        except Exception as exc:  # An execution failure, never a zero grade.
            state.status, state.error = "ROLLOUT_CONTAINER_STATUS_ERROR", f"{type(exc).__name__}: {exc}"[:500]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve", "tasks", "prepare"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--template", type=Path, help="prepare: run config with a placeholder task list")
    parser.add_argument("--inference-url", help="prepare: the deployed serving pool URL")
    parser.add_argument("--out", type=Path, help="prepare: where to write the run config")
    args = parser.parse_args()
    problems = load_problems(args.data)
    sha = data_sha(args.data)
    count = len(problems) if args.limit is None else min(args.limit, len(problems))
    tasks = [{"environment_id": i + 1, "image_id": IMAGE_ID, "source_commit_sha": sha} for i in range(count)]
    if args.command == "tasks":
        print(json.dumps(tasks))
        return
    if args.command == "prepare":
        if args.template is None or args.inference_url is None or args.out is None:
            parser.error("prepare needs --template, --inference-url and --out")
        config = json.loads(args.template.read_text())
        config["dataset"]["tasks"] = tasks
        config["inference_url"] = args.inference_url
        args.out.write_text(json.dumps(config, indent=2) + "\n")
        run = read_run_config(str(args.out))  # Validate the result against the contract.
        print(f"Wrote {args.out}: {len(run.dataset.tasks)} tasks pinned to {sha}")
        return
    if args.config is None:
        parser.error("serve needs --config")
    run = read_run_config(args.config)
    api_key = os.environ[run.platform.api_key_env]
    capture_key = os.environ[run.capture.platform_key_env]

    import uvicorn

    async def serve() -> None:
        async with httpx.AsyncClient(timeout=run.request_timeout_seconds) as client:
            platform = MathPlatform(
                run, problems=problems, sha=sha, api_key=api_key, capture_key=capture_key, client=client
            )
            config = uvicorn.Config(platform.app, host=args.host, port=args.port, access_log=False)
            await uvicorn.Server(config).serve()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
