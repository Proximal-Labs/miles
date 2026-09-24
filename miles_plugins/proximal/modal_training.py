"""The training node on Modal. PAID: GPU containers, runs until stopped.

One Modal container runs the training side of a Proximal run:

- the rollout store (a local Postgres under ``/state``), snapshotted every step;
- the capture service;
- for a ``gsm8k`` deployment, the stand-in platform on loopback; for a ``real`` one, an
  HTTPS tunnel to capture that the platform's rollout workers call (see ``training``);
- the Miles trainer, fully async, publishing each LoRA version to the adapter Volume.

The serving pool (``serving_app``) must already be deployed, with its URL as the run
config's ``inference_url``.

    PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=serving.json \\
    PROXIMAL_TRAINING_CONFIG=training.json \\
        modal run --detach --env main -m miles_plugins.proximal.modal_training

``--detach`` keeps it running after the local client exits; stop it with
``modal app stop <app_name> --env main``.

Crash recovery: Miles checkpoints every step (a LoRA checkpoint is the adapter and its
optimizer state). After each saved step a thread copies a self-contained snapshot (see
``e2e.snapshots``) to the deployment's state Volume. On start, the latest snapshot is
restored before any service runs and Miles resumes from it; Modal retries the function
after a crash, up to the deployment's ``max_retries``. A real-platform retry opens a new tunnel and waits for its registration.
"""

import contextlib
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

from miles_plugins.proximal.modal_sources import add_fork_sources
from miles_plugins.proximal.serving_app import DEPLOYMENT, RUN, RUN_JSON, base_volume, with_configs
from miles_plugins.proximal.training import (
    Gsm8kPlatform,
    RealPlatform,
    check_deployment,
    check_train_args,
    fetch_registry,
    read_training_deployment,
    registered_capture_url,
    registration_command,
)

_TRAINING_PATH = "PROXIMAL_TRAINING_CONFIG"
_CONTAINER_TRAINING_CONFIG = "/proximal-config/training.json"
REPO = Path(__file__).resolve().parents[2]
TRAINING = read_training_deployment(os.environ[_TRAINING_PATH])
check_deployment(RUN, TRAINING)
if modal.is_local():
    check_train_args(TRAINING, (REPO / TRAINING.train_args).read_text())

SNAPSHOT_MOUNT = Path("/snapshot")
state_volume = modal.Volume.from_name(
    TRAINING.state_volume.volume_name,
    environment_name=TRAINING.state_volume.environment_name,
    create_if_missing=False,
)
# One snapshot namespace per run: a new run must never resume another run's state.
SNAPSHOT = SNAPSHOT_MOUNT / RUN.run_id
FORK = Path("/fork")  # This fork's files that are not Python packages.
CONFIG = Path("/config/run.json")
STATE = Path("/state")
CAPTURE_PORT = 9011
PLATFORM_PORT = 9010
# The recipe's Ray runtime environment, set before `ray start` so workers inherit it.
MEGATRON_ENV = {
    "PYTHONPATH": f"/root/Megatron-LM:{FORK}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONUNBUFFERED": "1",
}

image = add_fork_sources(
    with_configs(
        modal.Image.from_registry(DEPLOYMENT.image)
        .entrypoint([])
        .apt_install("postgresql")
        .pip_install("psycopg[binary]")
        .env({**MEGATRON_ENV, _TRAINING_PATH: _CONTAINER_TRAINING_CONFIG})
    )
    .add_local_file(os.environ[_TRAINING_PATH], _CONTAINER_TRAINING_CONFIG)
    .add_local_file(REPO / "train_async.py", str(FORK / "train_async.py"))
    .add_local_dir(REPO / "scripts/models", str(FORK / "scripts/models"))
    .add_local_file(REPO / TRAINING.train_args, str(FORK / "train_args.txt"))
)

app = modal.App(TRAINING.app_name)


def _wait_healthy(url: str, process: subprocess.Popen[bytes], timeout_seconds: float = 300) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{process.args!r} exited with {process.returncode} before {url} was healthy")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"{url} did not become healthy")


def _wait_for_registration(tunnel_url: str, platform: RealPlatform) -> None:
    """Create no runs until the platform routes this model to this node's tunnel."""
    endpoint = f"capture-{int(time.time())}"
    command = registration_command(RUN, endpoint, tunnel_url)
    api_key = os.environ[RUN.platform.api_key_env]
    deadline = time.monotonic() + platform.registration_timeout_seconds
    announced = 0.0
    while time.monotonic() < deadline:
        try:
            registry = fetch_registry(RUN.platform.url, api_key)
            if registered_capture_url(registry, RUN.platform_route.model, None) == tunnel_url:
                print(f"[training] {RUN.platform_route.model} routes to {tunnel_url}", flush=True)
                return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"[training] registry read failed ({exc}); retrying", flush=True)
        if time.monotonic() - announced > 60:
            print(
                f"[training] capture tunnel: {tunnel_url}\n[training] register it from proximal-mono:\n  {command}",
                flush=True,
            )
            announced = time.monotonic()
        time.sleep(15)
    raise TimeoutError(f"{RUN.platform_route.model} was not routed to {tunnel_url} in time")


def training_command(resume_step: int | None) -> list[str]:
    from miles.utils.external_utils.model_args_utils import load_model_args
    from miles_plugins.proximal.e2e.snapshots import iter_dir

    lines = (FORK / "train_args.txt").read_text().splitlines()
    args = [
        token for line in lines if line.strip() and not line.lstrip().startswith("#") for token in shlex.split(line)
    ]
    model_args = shlex.split(load_model_args(TRAINING.model_args, model_script_dir=FORK / "scripts/models"))
    args += ["--wandb-run-id", RUN.run_id]
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


def _set_keys() -> None:
    # The capture admin key is only ever used inside this container.
    os.environ[RUN.capture.api_key_env] = secrets.token_hex(16)
    if isinstance(TRAINING.platform, Gsm8kPlatform):
        # Loopback-only credentials between this container's processes.
        os.environ[RUN.platform.api_key_env] = secrets.token_hex(16)
        os.environ[RUN.capture.platform_key_env] = secrets.token_hex(16)
    else:
        # The platform key, and the key its rollout workers send to capture, come from
        # the deployment's secrets: they must match the platform's.
        for name in (RUN.platform.api_key_env, RUN.capture.platform_key_env):
            if not os.environ.get(name):
                raise RuntimeError(f"{name} is not set; add the secret that provides it to the training deployment")
    os.environ["MILES_GATEWAY_AUTHORIZATION"] = f"Bearer {os.environ[DEPLOYMENT.gateway_key_env]}"


def _service_commands() -> list[tuple[str, list[str], str]]:
    real = isinstance(TRAINING.platform, RealPlatform)
    services = [
        (
            "capture",
            [
                sys.executable,
                "-m",
                "miles_plugins.proximal.runtime",
                "capture",
                "--config",
                str(CONFIG),
                "--yes-rollouts",
                "--yes-publish",
                # A real platform reaches capture through the tunnel.
                "--host",
                "0.0.0.0" if real else "127.0.0.1",
                "--port",
                str(CAPTURE_PORT),
            ],
            f"{RUN.capture.url}/health",
        )
    ]
    if isinstance(TRAINING.platform, Gsm8kPlatform):
        data = Path(DEPLOYMENT.base_mount) / TRAINING.platform.data
        if not data.exists():
            raise FileNotFoundError(f"{data} is missing; stage it with e2e.stage_gsm8k first")
        services.append(
            (
                "gsm8k-platform",
                [
                    sys.executable,
                    "-m",
                    "miles_plugins.proximal.e2e.math_platform",
                    "serve",
                    "--config",
                    str(CONFIG),
                    "--data",
                    str(data),
                    "--port",
                    str(PLATFORM_PORT),
                ],
                f"{RUN.platform.url}/health",
            )
        )
    return services


@app.function(
    image=image,
    gpu=TRAINING.gpu,
    # Capture, the rollout executor, Ray and Postgres share this container's CPUs; a GPU
    # function otherwise gets about one core and they starve.
    cpu=float(TRAINING.cpu),
    volumes={str(DEPLOYMENT.base_mount): base_volume, str(SNAPSHOT_MOUNT): state_volume},
    retries=modal.Retries(max_retries=TRAINING.max_retries, initial_delay=30.0) if TRAINING.max_retries else None,
    secrets=[modal.Secret.from_name(name, environment_name=RUN.volume.environment_name) for name in TRAINING.secrets],
    timeout=24 * 3600,
)
def train() -> int:
    from miles_plugins.proximal.e2e import snapshots
    from miles_plugins.proximal.e2e.local_postgres import local_postgres

    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(RUN_JSON)
    _set_keys()
    # A retry may land in the container of the failed attempt: clear its Ray and state.
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    snapshots.reset_local_state(STATE)
    logs = STATE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    pg_bin = sorted(Path("/usr/lib/postgresql").glob("*/bin"))[-1]
    stop = threading.Event()
    with local_postgres(STATE / "postgres") as dsn, contextlib.ExitStack() as tunnels:
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
        snapshotter = threading.Thread(target=_snapshot_loop, args=(dsn, pg_bin, stop, resume_step))
        try:
            for name, command, health in _service_commands():
                log = (logs / f"{name}.log").open("ab")
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
                _wait_healthy(health, processes[-1])
                print(f"[training] {name} ready", flush=True)
            if isinstance(TRAINING.platform, RealPlatform):
                tunnel = tunnels.enter_context(modal.forward(CAPTURE_PORT))
                _wait_for_registration(tunnel.url.rstrip("/"), TRAINING.platform)
            subprocess.run(
                [
                    "ray",
                    "start",
                    "--head",
                    "--node-ip-address",
                    "127.0.0.1",
                    "--num-gpus",
                    str(TRAINING.num_gpus),
                    "--disable-usage-stats",
                ],
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
            # Stop everything that can still write a checkpoint, then the snapshot thread,
            # before re-raising: a retry must never overlap this attempt's writes.
            subprocess.run(["ray", "stop", "--force"], check=False)
            stop.set()
            if snapshotter.is_alive():
                snapshotter.join()
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
