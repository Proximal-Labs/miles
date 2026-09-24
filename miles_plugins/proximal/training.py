"""The training node's deployment: hardware, state, secrets, and which platform it trains against.

``modal_training`` reads a ``TrainingDeployment`` next to the run and serving configs.
The platform is explicit:

- ``gsm8k``: the stand-in platform (``e2e.math_platform``) runs on loopback in the
  training container and grades gsm8k problems. Capture is loopback-only.
- ``real``: runs are created on the Proximal platform named by the run config. Its
  rollout workers reach capture through an HTTPS tunnel opened by the training node.
  Each start registers that tunnel as the platform model's default ``rollout_capture``
  endpoint (an operator runs the printed ``switch-endpoint`` command); the node creates
  no runs until the registry points at its tunnel.

This module has no Modal dependency, so the registry checks are testable offline.
"""

import json
import urllib.request
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from miles_plugins.proximal.contracts import Contract, Positive, RunConfig
from miles_plugins.proximal.modal_volume import VolumeDestination
from miles_plugins.proximal.snapshot import Nonempty

REGISTRY_KEY = "modal.inference.endpoints"


class Gsm8kPlatform(Contract):
    kind: Literal["gsm8k"]
    # Relative to the serving config's base mount (staged by e2e.stage_gsm8k).
    data: Nonempty


class RealPlatform(Contract):
    kind: Literal["real"]
    # How long a start waits for the operator to register its tunnel.
    registration_timeout_seconds: Positive


class TrainingDeployment(Contract):
    app_name: Nonempty
    gpu: Nonempty  # Modal GPU spec, e.g. "H200:8".
    num_gpus: Positive  # GPUs Ray may schedule; must match the spec's count.
    cpu: Positive
    # Modal retries after a crash; each resumes from the latest snapshot. A real-platform
    # retry opens a new tunnel and waits, holding its GPUs, for a new registration.
    max_retries: Annotated[int, Field(ge=0)]
    # Model args script under scripts/models (without ``.py``), e.g. "qwen3.8-27B".
    model_args: Nonempty
    # Miles training arguments file, relative to the repository root.
    train_args: Nonempty
    state_volume: VolumeDestination
    secrets: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    platform: Annotated[Gsm8kPlatform | RealPlatform, Field(discriminator="kind")]


def read_training_deployment(path: str | Path) -> TrainingDeployment:
    return TrainingDeployment.model_validate_json(Path(path).read_text())


def check_deployment(run: RunConfig, deployment: TrainingDeployment) -> None:
    """Refuse combinations the node cannot run correctly."""
    count = int(deployment.gpu.rpartition(":")[2]) if ":" in deployment.gpu else 1
    if count != deployment.num_gpus:
        raise ValueError(f"gpu {deployment.gpu!r} provides {count} GPUs, but num_gpus is {deployment.num_gpus}")
    loopback_platform = run.platform.url.startswith(("http://127.0.0.1", "http://localhost"))
    if isinstance(deployment.platform, Gsm8kPlatform) and not loopback_platform:
        raise ValueError("The gsm8k platform runs on loopback; point platform.url at it")
    if isinstance(deployment.platform, RealPlatform):
        if loopback_platform:
            raise ValueError("A real platform run needs the platform's URL, not loopback")
        if run.platform_route.endpoint_name is not None:
            raise ValueError(
                "A real platform run registers a new endpoint each start and routes by the model's default "
                "endpoint; leave platform_route.endpoint_name unset"
            )


def train_args_gpus(text: str) -> int:
    """GPUs the Miles training arguments ask for: nodes x GPUs per node."""
    tokens = [token for line in text.splitlines() if not line.lstrip().startswith("#") for token in line.split()]
    values = {}
    for flag in ("--actor-num-nodes", "--actor-num-gpus-per-node"):
        if flag not in tokens:
            raise ValueError(f"Training arguments are missing {flag}")
        values[flag] = int(tokens[tokens.index(flag) + 1])
    return values["--actor-num-nodes"] * values["--actor-num-gpus-per-node"]


def check_train_args(deployment: TrainingDeployment, text: str) -> None:
    """The training arguments must ask for exactly the GPUs the deployment provides."""
    gpus = train_args_gpus(text)
    if gpus != deployment.num_gpus:
        raise ValueError(
            f"{deployment.train_args} asks for {gpus} GPUs, but the deployment provides {deployment.num_gpus}"
        )


def registered_capture_url(registry_json: str, model: str, endpoint_name: str | None) -> str | None:
    """The base URL the platform routes ``model`` to, if it is a rollout-capture endpoint."""
    config = json.loads(registry_json)
    entry = config.get("models", {}).get(model)
    if not isinstance(entry, dict):
        return None
    name = endpoint_name or entry.get("defaultEndpoint")
    endpoint = entry.get("endpoints", {}).get(name)
    if not isinstance(endpoint, dict) or endpoint.get("kind") != "rollout_capture":
        return None
    url = endpoint.get("baseURL")
    return url if isinstance(url, str) else None


def fetch_registry(platform_url: str, api_key: str, timeout_seconds: float = 30) -> str:
    """The live endpoint registry, read through the platform's LiveConfig API."""
    request = urllib.request.Request(
        f"{platform_url.rstrip('/')}/proximal.v1.LiveConfigService/GetLiveConfig",
        data=json.dumps({"key": REGISTRY_KEY}).encode(),
        headers={"content-type": "application/json", "Connect-Protocol-Version": "1", "x-api-key": api_key},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # Read-only.
        body = json.load(response)
    value = body.get("config", {}).get("valueJson")
    if not isinstance(value, str):
        raise ValueError(f"LiveConfig {REGISTRY_KEY} is missing")
    return value


def registration_command(run: RunConfig, endpoint: str, tunnel_url: str) -> str:
    """The proximal-mono command that makes this node's tunnel the model's default endpoint."""
    return (
        "pnpm tsx packages/backend/scripts/modal/switch-endpoint.ts "
        f"--model {run.platform_route.model} --register {endpoint} --set-default {endpoint} "
        f"--kind rollout_capture --base-url {tunnel_url} --wire-model {run.base_model.name} "
        f"--api-key-env {run.capture.platform_key_env} --apply"
    )


def prepare_run_config(template: Path, tasks_file: Path, inference_url: str, out: Path) -> RunConfig:
    """Write a run config from a template, a task file and the deployed pool URL, then validate it."""
    from miles_plugins.proximal.capture_server import check_tito_protocol
    from miles_plugins.proximal.contracts import read_run_config

    config = json.loads(template.read_text())
    tasks = json.loads(tasks_file.read_text())
    if tasks["project_id"] != config["dataset"]["project_id"]:
        raise ValueError(
            f"{tasks_file} is for project {tasks['project_id']}, the template for {config['dataset']['project_id']}"
        )
    config["dataset"]["tasks"] = tasks["tasks"]
    config["inference_url"] = inference_url
    out.write_text(json.dumps(config, indent=2) + "\n")
    run = read_run_config(out)
    check_tito_protocol(run)
    return run


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check"])
    parser.add_argument("--template", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--inference-url")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path, help="check: run config")
    parser.add_argument("--training", type=Path, help="check: training deployment")
    args = parser.parse_args()
    if args.command == "prepare":
        if None in (args.template, args.tasks, args.inference_url, args.out):
            parser.error("prepare needs --template, --tasks, --inference-url and --out")
        run = prepare_run_config(args.template, args.tasks, args.inference_url, args.out)
        print(f"Wrote {args.out}: {len(run.dataset.tasks)} tasks in project {run.dataset.project_id}")
        return
    from miles_plugins.proximal.contracts import read_run_config

    if args.config is None or args.training is None:
        parser.error("check needs --config and --training")
    deployment = read_training_deployment(args.training)
    check_deployment(read_run_config(args.config), deployment)
    check_train_args(deployment, Path(deployment.train_args).read_text())
    print("Run config and training deployment agree")


if __name__ == "__main__":
    main()
