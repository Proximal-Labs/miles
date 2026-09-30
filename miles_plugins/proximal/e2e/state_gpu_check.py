"""Bounded live verification of Volume handoff, native resume, and serving exports.

This is a diagnostic experiment, not a training launcher: the identical historical
batch is deliberately replayed to isolate persistence from rollout scheduling.
Existing model and rollout Volumes are read-only. All outputs belong to a unique
test namespace on a separate Volume. GPU functions have no retries and time out.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import modal

from miles_plugins.proximal.authorization import AuthorizedRun, authorize_run, require_authorization
from miles_plugins.proximal.modal_sources import add_fork_sources

REPO = Path(__file__).resolve().parents[3]
IMAGE = "radixark/miles@sha256:e4fcc8c4d40ddd2fe3cbd46cfc414287d879b55cf9915250f6e5474f7c7462b9"
ROOT = Path("/verification")
BASE = modal.Volume.from_name("miles-gsm8k-base", environment_name="main", create_if_missing=False)
SOURCE = modal.Volume.from_name("miles-gsm8k-state", environment_name="main", create_if_missing=False)
OUTPUT = modal.Volume.from_name("miles-state-verification", environment_name="main", create_if_missing=False)
ENV = {
    "PYTHONPATH": "/root/Megatron-LM:/fork",
    "PYTHONUNBUFFERED": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "VERIFICATION_IMAGE": IMAGE,
}
if modal.is_local():
    code = hashlib.sha256()
    for folder in ("miles", "miles_plugins", "scripts/models"):
        for path in sorted((REPO / folder).rglob("*.py")):
            code.update(str(path.relative_to(REPO)).encode() + b"\0" + path.read_bytes())
    code.update((REPO / "train.py").read_bytes())
    ENV["VERIFICATION_CODE_SHA256"] = code.hexdigest()
image = (
    add_fork_sources(
        modal.Image.from_registry(IMAGE)
        .entrypoint([])
        .apt_install("postgresql")
        .pip_install("psycopg[binary]")
        .env(ENV)
    )
    .add_local_file(REPO / "train.py", "/fork/train.py")
    .add_local_file(REPO / "examples/proximal/gsm8k/train_args.txt", "/fork/train_args.txt")
    .add_local_dir(REPO / "scripts/models", "/fork/scripts/models")
)
app = modal.App("miles-state-gpu-verification")


@app.function(
    image=image,
    volumes={"/source": SOURCE.read_only(), "/verification": OUTPUT},
    cpu=8,
    memory=32768,
    timeout=900,
    retries=0,
    max_containers=1,
)
def prepare(test_id: str, config_json: str, samples: int, resume: bool) -> dict[str, object]:
    from miles_plugins.proximal.e2e.state_gpu_worker import prepare_batch

    result = prepare_batch(ROOT / test_id, config_json=config_json, samples=samples, resume=resume)
    OUTPUT.commit()
    return result


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": BASE.read_only(), "/verification": OUTPUT},
    cpu=16,
    memory=131072,
    timeout=1800,
    retries=0,
    max_containers=1,
)
def reference(test_id: str, authorization: AuthorizedRun) -> dict[str, object]:
    from miles_plugins.proximal.e2e.state_gpu_worker import train_reference
    from miles_plugins.proximal.offline_batch import read_batch

    OUTPUT.reload()
    if read_batch(ROOT / test_id / "batch").source != require_authorization(authorization):
        raise ValueError("Verification authorization names a different source run")
    try:
        return train_reference(ROOT / test_id, commit=OUTPUT.commit)
    finally:
        OUTPUT.commit()


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": BASE.read_only(), "/verification": OUTPUT},
    cpu=16,
    memory=131072,
    timeout=1800,
    retries=0,
    max_containers=1,
)
def resumed(test_id: str, authorization: AuthorizedRun) -> dict[str, object]:
    from miles_plugins.proximal.e2e.state_gpu_worker import train_resumed
    from miles_plugins.proximal.offline_batch import read_batch

    OUTPUT.reload()
    if read_batch(ROOT / test_id / "batch").source != require_authorization(authorization):
        raise ValueError("Verification authorization names a different source run")
    try:
        return train_resumed(ROOT / test_id)
    finally:
        OUTPUT.commit()


@app.function(image=image, volumes={"/verification": OUTPUT}, cpu=8, memory=32768, timeout=600, retries=0)
def compare(test_id: str) -> dict[str, object]:
    from miles_plugins.proximal.e2e.state_gpu_compare import compare_checkpoints

    OUTPUT.reload()
    try:
        return compare_checkpoints(ROOT / test_id)
    finally:
        OUTPUT.commit()


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": BASE.read_only(), "/verification": OUTPUT},
    cpu=8,
    memory=65536,
    timeout=900,
    retries=0,
    max_containers=1,
)
def serve(test_id: str, authorization: AuthorizedRun) -> dict[str, object]:
    from miles_plugins.proximal.e2e.state_gpu_compare import serving_check
    from miles_plugins.proximal.offline_batch import read_batch

    OUTPUT.reload()
    if read_batch(ROOT / test_id / "batch").source != require_authorization(authorization):
        raise ValueError("Verification authorization names a different source run")
    try:
        return serving_check(ROOT / test_id)
    finally:
        OUTPUT.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "reference", "resumed", "compare", "serve"])
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--yes-gpu-test", action="store_true", required=True)
    parser.add_argument("--resume-preparation", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 to match the pinned image's typed authorization serialization")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.test_id):
        parser.error("Invalid test ID")
    if args.samples <= 0 or args.samples % 8:
        parser.error("Select a positive multiple of the recorded group size, 8")
    config_json = args.config.read_text()
    from miles_plugins.proximal.contracts import read_run_config

    config = read_run_config(args.config)
    if config.base_model.name != "Qwen/Qwen3-0.6B" or config.research.group_size != 8:
        parser.error("This bounded verification recipe supports Qwen3-0.6B, group size 8")
    authorization = authorize_run(config, yes_rollouts=args.yes_gpu_test, yes_publish=args.yes_gpu_test)
    os.environ.setdefault("MODAL_ENVIRONMENT", "main")
    with modal.enable_output(), app.run():
        if args.phase == "prepare":
            result = prepare.remote(args.test_id, config_json, args.samples, args.resume_preparation)
        elif args.phase == "reference":
            result = reference.remote(args.test_id, authorization)
        elif args.phase == "resumed":
            result = resumed.remote(args.test_id, authorization)
        elif args.phase == "compare":
            result = compare.remote(args.test_id)
        else:
            result = serve.remote(args.test_id, authorization)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
