"""The training node for the gsm8k topology test. PAID: one GPU container, runs until stopped.

One Modal container runs everything on the training side:

- the rollout store (a local Postgres under ``/state``);
- the capture service and the gsm8k platform, both on loopback;
- the Miles trainer: Megatron LoRA on one actor GPU, fully async, publishing each
  version to the adapter Volume.

The serving pool (``serving_app``) must already be deployed, with its URL as the run
config's ``inference_url``. The replicas load each published version from the Volume.

    PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=serving.json \\
        modal run --detach --env main -m miles_plugins.proximal.e2e.training_app

``--detach`` keeps it running after the local client exits; stop it with
``modal app stop miles-gsm8k-training --env main``.

Crash recovery: Miles checkpoints every step (``--save-interval 1``; a LoRA checkpoint
is the adapter and its optimizer state, not the base). After each saved step a thread
copies a consistent snapshot (see ``snapshots``) to the ``miles-gsm8k-state`` Volume. On
start, the latest snapshot is restored before any service runs and Miles resumes from
it; Modal retries the function after a crash.
"""

import os
import secrets
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

from miles_plugins.proximal.serving_app import DEPLOYMENT, RUN, RUN_JSON, base_volume

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT = Path("/snapshot")
state_volume = modal.Volume.from_name(
    "miles-gsm8k-state", environment_name=RUN.volume.environment_name, create_if_missing=False
)
TRAIN_ARGS = REPO / "examples/proximal/gsm8k/train_args.txt"
FORK = Path("/fork")  # This fork's files that are not Python packages.
CONFIG = Path("/config/run.json")
STATE = Path("/state")
DATA = Path(DEPLOYMENT.base_mount) / "gsm8k" / "train.parquet"
GPU = os.environ.get("PROXIMAL_TRAINING_GPU", "H100")
# The recipe's Ray runtime environment, set before `ray start` so workers inherit it.
MEGATRON_ENV = {
    "PYTHONPATH": f"/root/Megatron-LM:{FORK}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONUNBUFFERED": "1",
}

image = (
    modal.Image.from_registry(DEPLOYMENT.image)
    .entrypoint([])
    .apt_install("postgresql")
    .pip_install("psycopg[binary]")
    .env({"PROXIMAL_RUN_CONFIG_JSON": RUN_JSON, **MEGATRON_ENV})
    .add_local_file(REPO / "train_async.py", str(FORK / "train_async.py"))
    .add_local_file(REPO / "scripts/models/qwen3-0.6B.py", str(FORK / "scripts/models/qwen3-0.6B.py"))
    .add_local_file(TRAIN_ARGS, str(FORK / "train_args.txt"))
    .add_local_python_source("miles", "miles_plugins")
)

app = modal.App("miles-gsm8k-training")


def _wait_healthy(url: str, process: subprocess.Popen[bytes], timeout_seconds: float = 300) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{process.args} exited with {process.returncode} before {url} was healthy")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"{url} did not become healthy")


def training_command(resume_step: int | None) -> list[str]:
    from miles.utils.external_utils.model_args_utils import load_model_args
    from miles_plugins.proximal.e2e.snapshots import iter_dir

    lines = (FORK / "train_args.txt").read_text().splitlines()
    args = [token for line in lines if line.strip() and not line.lstrip().startswith("#") for token in shlex.split(line)]
    model_args = shlex.split(load_model_args("qwen3-0.6B", model_script_dir=FORK / "scripts/models"))
    if resume_step is not None:
        # LoRA resume: the base from the HF checkpoint, the adapter (with optimizer and
        # step) from the restored checkpoint; the task source restores its cursor.
        adapter = iter_dir(STATE / "checkpoints", resume_step) / "adapter"
        args += ["--load", str(RUN.tokenizer_path), "--lora-adapter-path", str(adapter)]
    return [
        sys.executable,
        "-m",
        "miles_plugins.proximal.runtime",
        "train",
        "--config",
        str(CONFIG),
        "--yes-rollouts",
        "--yes-publish",
        "--",
        *args,
        *model_args,
    ]


def _snapshot_loop(dsn: str, pg_bin: Path, stop: threading.Event, taken: int | None) -> None:
    from miles_plugins.proximal.e2e import snapshots

    while not stop.wait(15):
        steps = [step for step in snapshots.complete_steps(STATE / "checkpoints") if taken is None or step > taken]
        if not steps:
            continue
        step = steps[-1]
        snapshots.take(
            step,
            checkpoints=STATE / "checkpoints",
            artifacts=Path(RUN.artifact_directory),
            dsn=dsn,
            snapshot_root=SNAPSHOT,
            pg_bin=pg_bin,
        )
        state_volume.commit()
        taken = step
        print(f"[training] snapshot of step {step} committed", flush=True)


def _run_trainer(command: list[str]) -> int:
    """Run Miles, failing loudly if a resume silently falls back to a fresh adapter."""
    process = subprocess.Popen(command, cwd=FORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if "Training will start with freshly initialized adapter weights" in line:
            process.kill()
            raise RuntimeError("Resume did not load the LoRA adapter; refusing to train from scratch")
    return process.wait()


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(DEPLOYMENT.base_mount): base_volume, str(SNAPSHOT): state_volume},
    retries=modal.Retries(max_retries=3, initial_delay=30.0),
    secrets=[
        modal.Secret.from_name(DEPLOYMENT.gateway_secret, environment_name=RUN.volume.environment_name),
        modal.Secret.from_name("miles-gsm8k-proxy", environment_name=RUN.volume.environment_name),
    ],
    timeout=24 * 3600,
)
def train() -> int:
    from miles_plugins.proximal.e2e import snapshots
    from miles_plugins.proximal.e2e.local_postgres import local_postgres

    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(RUN_JSON)
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} is missing; stage it with stage_gsm8k first")
    # Loopback-only credentials between processes in this container.
    os.environ[RUN.platform.api_key_env] = secrets.token_hex(16)
    os.environ[RUN.capture.api_key_env] = secrets.token_hex(16)
    os.environ[RUN.capture.platform_key_env] = secrets.token_hex(16)
    os.environ["MILES_GATEWAY_AUTHORIZATION"] = f"Bearer {os.environ[DEPLOYMENT.gateway_key_env]}"
    logs = STATE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    pg_bin = sorted(Path("/usr/lib/postgresql").glob("*/bin"))[-1]
    stop = threading.Event()
    with local_postgres(STATE / "postgres") as dsn:
        os.environ[RUN.store_dsn_env] = dsn
        # Before any service connects: the store must be restored into an empty database.
        resume_step = snapshots.restore(
            snapshot_root=SNAPSHOT,
            checkpoints=STATE / "checkpoints",
            artifacts=Path(RUN.artifact_directory),
            dsn=dsn,
            pg_bin=pg_bin,
        )
        print(f"[training] {'resuming from step ' + str(resume_step) if resume_step is not None else 'fresh start'}")
        snapshotter = threading.Thread(target=_snapshot_loop, args=(dsn, pg_bin, stop, resume_step), daemon=True)
        try:
            services = [
                (
                    "capture",
                    [sys.executable, "-m", "miles_plugins.proximal.runtime", "capture", "--config", str(CONFIG),
                     "--yes-rollouts", "--yes-publish", "--port", "9011"],
                    f"{RUN.capture.url}/health",
                ),
                (
                    "gsm8k-platform",
                    [sys.executable, "-m", "miles_plugins.proximal.e2e.math_platform", "serve", "--config",
                     str(CONFIG), "--data", str(DATA), "--port", "9010"],
                    f"{RUN.platform.url}/health",
                ),
            ]
            for name, command, health in services:
                log = (logs / f"{name}.log").open("ab")
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
                _wait_healthy(health, processes[-1])
                print(f"[training] {name} ready", flush=True)
            subprocess.run(
                ["ray", "start", "--head", "--node-ip-address", "127.0.0.1", "--num-gpus", "1",
                 "--disable-usage-stats"],
                check=True,
            )
            os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"
            command = training_command(resume_step)
            print("[training] " + shlex.join(command), flush=True)
            snapshotter.start()
            code = _run_trainer(command)
            if code != 0:  # Raise so Modal retries from the latest snapshot.
                raise RuntimeError(f"Trainer exited with {code}")
            return code
        finally:
            stop.set()
            subprocess.run(["ray", "stop", "--force"], check=False)
            for process in reversed(processes):
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()


@app.local_entrypoint()
def main() -> None:
    code = train.remote()
    print(f"Trainer exited with {code}")
