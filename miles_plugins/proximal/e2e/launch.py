"""Run the whole Stage A harness in one container: store, capture, stub platform, trainer.

The run config only allows plain HTTP on loopback, so every local process shares one
network namespace. Ports come from the config's platform/capture/inference URLs.

Offline (no paid resources): a loopback ``inference_url`` starts the fake serving pool
and publication stays local.

    python -m miles_plugins.proximal.e2e.launch --config run.local.json \\
        --adapters /work/.stage-a/adapters --workdir /tmp/stage-a --publish local \\
        --yes-rollouts --yes-publish

Against the Modal serving pool: an HTTPS ``inference_url`` and ``--publish modal``
upload each version to the adapter Volume. Nothing here deploys, creates or deletes a
Modal resource.
"""

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

from miles_plugins.proximal.contracts import RunConfig, read_run_config
from miles_plugins.proximal.e2e import fake_trainer
from miles_plugins.proximal.e2e.local_postgres import local_postgres


def _loopback_port(url: str) -> int | None:
    parts = urlsplit(url)
    if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        return None
    return parts.port or (443 if parts.scheme == "https" else 80)


def _wait_healthy(url: str, process: subprocess.Popen[bytes], timeout_seconds: float = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{' '.join(map(str, process.args))} exited with {process.returncode}")  # type: ignore[arg-type]
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{url} did not become healthy")


@contextlib.contextmanager
def _services(config_path: Path, run: RunConfig, logs: Path) -> Iterator[None]:
    commands: list[tuple[str, list[str], str]] = []
    capture_port = _loopback_port(run.capture.url)
    platform_port = _loopback_port(run.platform.url)  # None: a real platform, no stub.
    if capture_port is None:
        raise ValueError(
            "The launcher runs capture locally; its URL must be loopback (expose it to a platform separately)"
        )
    python = [sys.executable, "-m"]
    commands.append(
        (
            "capture",
            [
                *python,
                "miles_plugins.proximal.runtime",
                "capture",
                "--config",
                str(config_path),
                "--yes-rollouts",
                "--yes-publish",
                "--port",
                str(capture_port),
            ],
            f"{run.capture.url}/health",
        )
    )
    commands.append(
        (
            "stub-platform",
            [
                *python,
                "miles_plugins.proximal.e2e.stub_platform",
                "--config",
                str(config_path),
                "--port",
                str(platform_port),
            ],
            f"{run.platform.url}/health",
        )
    )
    if (pool_port := _loopback_port(run.inference_url)) is not None:
        commands.append(
            (
                "fake-pool",
                [
                    *python,
                    "miles_plugins.proximal.e2e.fake_pool",
                    "--config",
                    str(config_path),
                    "--port",
                    str(pool_port),
                ],
                f"{run.inference_url}/health",
            )
        )
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for name, command, health in commands:
            with (logs / f"{name}.log").open("ab") as log:
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            _wait_healthy(health, processes[-1])
            print(f"[stage-a] {name} ready ({health})", flush=True)
        yield
    finally:
        for process in reversed(processes):
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True, help="Checkpoints, logs and the report")
    parser.add_argument("--publish", choices=["local", "modal"], required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--publish-every", type=int, default=2)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--yes-rollouts", action="store_true")
    parser.add_argument("--yes-publish", action="store_true")
    args = parser.parse_args()
    run = read_run_config(args.config)
    if args.publish == "local" and _loopback_port(run.inference_url) is None:
        raise ValueError("--publish local needs the loopback fake pool; the real pool reads the adapter Volume")

    with contextlib.ExitStack() as stack:
        if not os.environ.get(run.store_dsn_env):
            # Kept under the workdir: a resumed launch must see the same store.
            os.environ[run.store_dsn_env] = stack.enter_context(local_postgres(args.workdir / "postgres"))
            print(f"[stage-a] local Postgres for the rollout store at {args.workdir / 'postgres'}", flush=True)
        stack.enter_context(_services(args.config, run, args.workdir / "logs"))
        trainer_args = argparse.Namespace(
            config=args.config,
            adapters=args.adapters,
            checkpoints=args.workdir / "checkpoints",
            steps=args.steps,
            batch_size=args.batch_size,
            publish_every=args.publish_every,
            publish=args.publish,
            resume_step=args.resume_step,
            yes_rollouts=args.yes_rollouts,
            yes_publish=args.yes_publish,
        )
        report = asyncio.run(fake_trainer.run(trainer_args))
    path = args.workdir / ("report.json" if args.resume_step is None else f"report-resume-{args.resume_step}.json")
    path.write_text(json.dumps(report, indent=2))
    versions = sorted({group["policy_version"] for entry in report for group in entry["groups"]})
    print(f"[stage-a] {len(report)} steps trained on policy versions {versions}; report at {path}", flush=True)


if __name__ == "__main__":
    main()
