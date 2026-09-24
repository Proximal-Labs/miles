"""Entrypoints for the CPU capture service and Miles's existing async trainer.

The replica gateway is an ASGI component embedded in the platform-owned Modal
container. This module never provisions a trainer, inference replica, or sandbox.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from miles_plugins.proximal.authorization import AuthorizedRun, authorize_run
from miles_plugins.proximal.contracts import RunConfig, read_run_config
from miles_plugins.proximal.options import BUFFER, ROLLOUT, SOURCE, TRANSFER

if TYPE_CHECKING:
    import httpx

    from miles_plugins.proximal.store import RolloutStore


def training_argv(path: str) -> list[str]:
    config = read_run_config(path)
    values = {
        "proximal-config": path,
        "rollout-function-path": ROLLOUT,
        "data-source-path": SOURCE,
        "custom-async-data-buffer-path": BUFFER,
        "custom-weight-transfer-protocol-path": TRANSFER,
        "rollout-num-gpus": 0,
        "update-weights-interval": 1,
        "n-samples-per-prompt": config.research.group_size,
        "max-weight-staleness": config.research.max_policy_lag,
        "async-unused-samples-handler": config.research.unused_groups,
        "async-max-concurrent-samples": config.max_in_flight_samples,
        "rollout-submission-granularity": "group",
        "rollout-temperature": config.research.sampling.temperature,
        "rollout-top-p": config.research.sampling.top_p,
        "rollout-top-k": config.research.sampling.top_k,
        "rollout-max-response-len": config.research.sampling.max_tokens,
        "rollout-max-context-len": config.research.sampling.max_sequence_tokens,
        "hf-checkpoint": str(config.tokenizer_path),
        "lora-rank": config.research.lora.rank,
        "lora-alpha": config.research.lora.alpha,
        "lora-dropout": 0,
        "target-modules": ",".join(config.research.lora.target_modules),
        "train-backend": "megatron",
        "megatron-to-hf-mode": "bridge",
    }
    return ["--fully-async", "--rollout-external", "--use-rollout-logprobs"] + [
        item for name, value in values.items() for item in (f"--{name}", str(value))
    ]


async def serve_capture(config: RunConfig, authorization: AuthorizedRun, host: str, port: int) -> None:
    # Runtime-only dependencies: local snapshot/publication CLI stays lightweight.
    import httpx
    import uvicorn
    from transformers import AutoTokenizer

    from miles_plugins.proximal.capture_server import CaptureServer
    from miles_plugins.proximal.store import open_store

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    store = await open_store(config)
    try:
        async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
            service = CaptureServer(authorization, tokenizer=tokenizer, client=client, store=store)
            await uvicorn.Server(
                uvicorn.Config(service.app, host=host, port=port, workers=1, access_log=False)
            ).serve()
    finally:
        await store.close()


async def run_control(args: argparse.Namespace, authorization: AuthorizedRun) -> None:
    import httpx

    from miles_plugins.proximal.store import open_store

    config = authorization.config
    if args.command == "rollout" and (args.task_index is None or not 0 <= args.task_index < len(config.dataset.tasks)):
        raise ValueError("rollout requires a valid --task-index into the pinned dataset")
    if args.command == "commit-policy" and args.policy_file is None:
        raise ValueError("commit-policy requires --policy-file")
    store = await open_store(config)
    async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as http:
        try:
            await _run_control(args, authorization, http, store)
        finally:
            await store.close()


async def _run_control(
    args: argparse.Namespace, authorization: AuthorizedRun, http: "httpx.AsyncClient", store: "RolloutStore"
) -> None:
    import uuid

    from miles.utils.types import Sample
    from miles_plugins.proximal.clients import CaptureClient, PlatformClient, ServingPoolClient
    from miles_plugins.proximal.contracts import Attempt, Policy, digest
    from miles_plugins.proximal.rollout import execute_attempt

    config = authorization.config
    capture = CaptureClient(authorization, http)
    if args.command == "commit-policy":
        published = Policy.model_validate_json(args.policy_file.read_bytes())
        if published.run_id != config.run_id or published.base_model != config.base_model:
            raise ValueError("Policy file belongs to another run/base model")
        await ServingPoolClient(authorization, http).prepare(published)
        await store.commit_policy(published)
        print(published.model_dump_json())
        return
    policy = await store.current_policy()
    if policy is None:
        raise ValueError("No committed policy; run commit-policy first")
    attempt = Attempt(
        attempt_id=uuid.uuid4().hex,
        run_id=config.run_id,
        group_id=uuid.uuid4().hex,
        sample_index=0,
        dataset_sha256=digest(config.dataset),
        task=config.dataset.tasks[args.task_index],
        harness=config.harness,
        policy=policy,
        sampling=config.research.sampling,
    )
    result = await execute_attempt(
        attempt,
        Sample(index=0, group_index=0),
        capture=capture,
        platform=PlatformClient(authorization, http),
        artifact_root=config.artifact_directory / config.run_id / "accepted",
    )
    print(
        json.dumps(
            {
                "attempt_id": attempt.attempt_id,
                "policy_version": policy.version,
                "reward": result.reward,
                "tokens": len(result.tokens),
                "assistant_tokens": result.effective_response_length,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "train-args", "capture", "train", "rollout", "commit-policy"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--yes-rollouts", action="store_true")
    parser.add_argument("--yes-publish", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--policy-file", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args, remaining = parser.parse_known_args()
    config = read_run_config(args.config)
    if args.command == "validate":
        from miles_plugins.proximal.capture_server import check_tito_protocol

        check_tito_protocol(config)
        print(f"Valid run {config.run_id}: {len(config.dataset.tasks)} pinned tasks; no remote work performed")
        return
    if args.command == "train-args":
        print(json.dumps(training_argv(args.config), indent=2))
        return
    authorization = authorize_run(config, yes_rollouts=args.yes_rollouts, yes_publish=args.yes_publish)
    if args.command in {"rollout", "commit-policy"}:
        if remaining:
            parser.error("Unknown smoke/publication arguments")
        asyncio.run(run_control(args, authorization))
        return
    if args.command == "capture":
        if remaining:
            parser.error("Unknown capture arguments")
        asyncio.run(serve_capture(config, authorization, args.host, args.port))
        return
    import sys

    from train_async import train

    from miles.utils.arguments import parse_args
    from miles.utils.tracking_utils.tracking import finish_tracking

    extra = remaining[1:] if remaining[:1] == ["--"] else remaining
    sys.argv = [
        "train_async.py",
        *training_argv(args.config),
        "--proximal-yes-rollouts",
        "--proximal-yes-publish",
        *extra,
    ]
    train_args = parse_args()  # type: ignore[no-untyped-call]  # Existing Miles CLI boundary.
    try:
        asyncio.run(train(train_args))  # type: ignore[no-untyped-call]  # Existing Miles driver.
    finally:
        finish_tracking()  # type: ignore[no-untyped-call]  # Existing Miles tracking boundary.


if __name__ == "__main__":
    main()
