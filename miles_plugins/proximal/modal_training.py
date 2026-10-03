"""The training node on Modal. PAID: GPU containers, runs until stopped.

One Modal container runs the training side of a Proximal run:

- the rollout store (a local Postgres under ``/state``), snapshotted every step;
- for a ``gsm8k`` deployment, the stand-in platform on loopback;
- the Miles trainer, fully async, publishing each LoRA version to the adapter Volume.

Capture runs in the serving replicas, not here (see ``serve_replica``): the serving pool
(``serving_app``) must already be deployed, with its URL as the run config's
``inference_url`` and ``capture.url``. The pool outlives the node unless the deployment
sets ``stop_pool_on_exit``. The trainer opens, seals and fetches each
rollout's session there. A ``real`` deployment creates no platform runs until the
platform's registry routes the model to the pool (a one-time registration per pool).

    PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=serving.json \\
    PROXIMAL_TRAINING_CONFIG=training.json \\
        modal run --detach --env main -m miles_plugins.proximal.modal_training

``--detach`` keeps it running after the local client exits; stop it with
``modal app stop <app_name> --env main``.

Crash recovery: Miles checkpoints every step (a LoRA checkpoint is the adapter and its
optimizer state). After each saved step a thread copies a self-contained snapshot (see
``e2e.snapshots``) to the deployment's state Volume. On start, the latest snapshot is
restored before any service runs and Miles resumes from it; Modal retries the function
after a crash, up to the deployment's ``max_retries``.

Detached collection: ``--collect-rollouts N --rollouts-persist-to-volume`` runs
the same producer and artifact publisher on CPU, without starting a trainer GPU.
Choose ``--fresh`` or ``--policy-file`` and explicitly authorize rollout/publication.
Completed results are saved incrementally; the selected batch is finalized on the
state Volume for a later independent training step. See docs/proximal/offline-batches.md.
"""

import asyncio
import hashlib
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

import modal
import modal.experimental
from pydantic import TypeAdapter

from miles_plugins.proximal.contracts import (
    RunConfig,
    RunStateArtifacts,
    SafeId,
    canonical_bytes,
    digest,
    training_contract,
)
from miles_plugins.proximal.modal_sources import add_fork_sources
from miles_plugins.proximal.serving_app import DEPLOYMENT, RUN, base_volume, with_configs
from miles_plugins.proximal.state_checkpoints import RecoveryContext
from miles_plugins.proximal.storage import write_atomic
from miles_plugins.proximal.training import (
    Gsm8kPlatform,
    RealPlatform,
    check_deployment,
    check_train_args,
    fetch_registry,
    pool_endpoint_name,
    read_training_deployment,
    registration_command,
    routes_to_pool,
    stops_pool,
)

_TRAINING_PATH = "PROXIMAL_TRAINING_CONFIG"
_CONTAINER_TRAINING_CONFIG = "/proximal-config/training.json"
REPO = Path(__file__).resolve().parents[2]
TRAINING = read_training_deployment(os.environ[_TRAINING_PATH])
check_deployment(RUN, TRAINING)
if TRAINING.state_volume == DEPLOYMENT.base_volume:
    raise ValueError("Training state must not share the serving base-weight Volume")
if modal.is_local():
    check_train_args(TRAINING, (REPO / TRAINING.train_args).read_text())

SNAPSHOT_MOUNT = Path("/snapshot")
state_volume = modal.Volume.from_name(
    TRAINING.state_volume.volume_name,
    environment_name=TRAINING.state_volume.environment_name,
    create_if_missing=False,
)
# Kernel compilation and tuning shared across runs (training.KernelCache).
kernel_volume = (
    modal.Volume.from_name(
        TRAINING.kernel_cache.volume.volume_name,
        environment_name=TRAINING.kernel_cache.volume.environment_name,
        create_if_missing=False,
    )
    if TRAINING.kernel_cache is not None
    else None
)
kernel_mounts: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = (
    {str(TRAINING.kernel_cache.mount): kernel_volume}
    if TRAINING.kernel_cache is not None and kernel_volume is not None
    else {}
)
# One snapshot namespace per run: a new run must never resume another run's state.
SNAPSHOT = SNAPSHOT_MOUNT / RUN.run_id
FORK = Path("/fork")  # This fork's files that are not Python packages.
CONFIG = Path("/config/run.json")
STATE = Path("/state")
PLATFORM_PORT = 9010
# The recipe's Ray runtime environment, set before `ray start` so workers inherit it.
MEGATRON_ENV = {
    "PYTHONPATH": f"/root/Megatron-LM:{FORK}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONUNBUFFERED": "1",
    # Inductor compiles in each rank's own process. Its default pool of forked compile
    # workers deadlocked a first backward (run 011: a worker inherited a lock held at fork,
    # in concurrent.futures' weakref_cb), and the replay A/B showed no cost to compiling inline.
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    **(
        {"NCCL_ALGO": "Ring", "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        if TRAINING.deterministic_kernels
        else {"NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1"}
    ),
    **(
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        if TRAINING.cuda_allocator == "expandable_segments"
        else {}
    ),
    **(TRAINING.kernel_cache.env() if TRAINING.kernel_cache is not None else {}),
}


def _source_digest() -> str:
    if not modal.is_local():
        return os.environ["PROXIMAL_CODE_SHA256"]
    result = hashlib.sha256()
    paths = [REPO / "train_async.py"]
    for directory in ("miles", "miles_plugins", "scripts/models"):
        paths.extend(path for path in (REPO / directory).rglob("*") if path.suffix in {".py", ".jinja"})
    for path in sorted(paths):
        result.update(str(path.relative_to(REPO)).encode() + b"\0" + path.read_bytes() + b"\0")
    return result.hexdigest()


CODE_SHA256 = _source_digest()

image = add_fork_sources(
    with_configs(
        modal.Image.from_registry(DEPLOYMENT.image)
        .entrypoint([])
        .apt_install("postgresql")
        .pip_install("psycopg[binary]")
        .env({**MEGATRON_ENV, _TRAINING_PATH: _CONTAINER_TRAINING_CONFIG, "PROXIMAL_CODE_SHA256": CODE_SHA256})
    )
    .add_local_file(os.environ[_TRAINING_PATH], _CONTAINER_TRAINING_CONFIG)
    .add_local_file(REPO / "train_async.py", str(FORK / "train_async.py"))
    .add_local_file(REPO / "train.py", str(FORK / "train.py"))
    .add_local_dir(REPO / "scripts/models", str(FORK / "scripts/models"))
    .add_local_file(REPO / TRAINING.train_args, str(FORK / "train_args.txt"))
)

app = modal.App(TRAINING.app_name)


def _wait_healthy(url: str, process: subprocess.Popen[bytes], log: Path, timeout_seconds: float = 300) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            # The log lives in this container; show its end before the container is gone.
            tail = log.read_text(errors="replace")[-4000:] if log.exists() else "(no log)"
            print(f"[training] {log.name} (end):\n{tail}", flush=True)
            raise RuntimeError(f"{process.args!r} exited with {process.returncode} before {url} was healthy")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"{url} did not become healthy")


def _wait_for_registration(platform: RealPlatform) -> None:
    """Create no runs until the platform routes this model to the pool's capture."""
    pool_url = RUN.capture.url
    command = registration_command(RUN, pool_endpoint_name(DEPLOYMENT.app_name, RUN), pool_url)
    api_key = os.environ[RUN.platform.api_key_env]
    deadline = time.monotonic() + platform.registration_timeout_seconds
    announced = 0.0
    while time.monotonic() < deadline:
        try:
            registry = fetch_registry(RUN.platform.url, api_key)
            if routes_to_pool(registry, RUN):
                print(f"[training] {RUN.platform_route.model} routes to {pool_url}", flush=True)
                return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"[training] registry read failed ({exc}); retrying", flush=True)
        if time.monotonic() - announced > 60:
            print(
                f"[training] the platform does not route {RUN.platform_route.model} to {pool_url} with this run's budget yet;"
                f" register the pool from proximal-mono (once per pool):\n  {command}",
                flush=True,
            )
            announced = time.monotonic()
        time.sleep(15)
    raise TimeoutError(f"{RUN.platform_route.model} was not routed to {pool_url} in time")


def training_command(resume_step: int | None) -> list[str]:
    from miles.utils.external_utils.model_args_utils import load_model_args
    from miles_plugins.proximal.e2e.snapshots import iter_dir

    lines = (FORK / "train_args.txt").read_text().splitlines()
    args = [
        token for line in lines if line.strip() and not line.lstrip().startswith("#") for token in shlex.split(line)
    ]
    model_args = shlex.split(load_model_args(TRAINING.model_args, model_script_dir=FORK / "scripts/models"))
    # One W&B run per training run, named for it: a restart continues its history.
    args += ["--wandb-run-id", RUN.run_id, "--wandb-group", RUN.run_id, "--disable-wandb-random-suffix"]
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


def _recovery_context() -> RecoveryContext:
    from miles.utils.external_utils.model_args_utils import load_model_args

    args = tuple(
        token
        for line in (FORK / "train_args.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for token in shlex.split(line)
    )
    return RecoveryContext(
        run_id=RUN.run_id,
        contract_sha256=digest(training_contract(RUN)),
        world_size=TRAINING.num_gpus,
        max_policy_lag=RUN.research.max_policy_lag,
        model_args=tuple(shlex.split(load_model_args(TRAINING.model_args, model_script_dir=FORK / "scripts/models"))),
        train_args=args,
        image=DEPLOYMENT.image,
        code_sha256=CODE_SHA256,
    )


def _record_launch(context: RecoveryContext, launch_id: str, run: RunConfig) -> None:
    root_record = SNAPSHOT / "run.json"
    if root_record.exists():
        previous = RecoveryContext.model_validate_json(root_record.read_bytes())
        for field in ("run_id", "contract_sha256", "world_size", "max_policy_lag", "model_args"):
            if getattr(previous, field) != getattr(context, field):
                raise ValueError(f"Run namespace has incompatible {field}")
    else:
        write_atomic(root_record, canonical_bytes(context))
    # No secrets: configs contain environment-variable names, not their values.
    value = {
        "context": context.model_dump(mode="json"),
        "run": run.model_dump(mode="json"),
        "training": TRAINING.model_dump(mode="json"),
    }
    write_atomic(SNAPSHOT / "launches" / launch_id / "config.json", json.dumps(value, sort_keys=True).encode())
    state_volume.commit()


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
    # Capture's credentials come from the serving pool's capture secret: the trainer's
    # control key, and the platform's rollout key (the stand-in platform's agent, and
    # the preflight canary's model calls).
    required = [RUN.capture.api_key_env, RUN.capture.platform_key_env]
    if isinstance(TRAINING.platform, Gsm8kPlatform):
        # The stand-in platform's own key never leaves this container.
        os.environ[RUN.platform.api_key_env] = secrets.token_hex(16)
    else:
        # Must match the platform's: the node creates runs with it.
        required.append(RUN.platform.api_key_env)
    for name in required:
        if not os.environ.get(name):
            raise RuntimeError(f"{name} is not set; add the secret that provides it to the training deployment")
    os.environ["MILES_GATEWAY_AUTHORIZATION"] = f"Bearer {os.environ[DEPLOYMENT.gateway_key_env]}"


def _check_serving(timeout_seconds: float = 1800) -> None:
    """Before the trainer starts: the replicas were deployed with a contract this run fits.
    Waits for replicas that are still starting; a mismatch fails at once."""
    import asyncio

    import httpx

    from miles_plugins.proximal.authorization import authorize_run
    from miles_plugins.proximal.preflight import check_serving

    authorization = authorize_run(RUN, yes_rollouts=True, yes_publish=True)

    async def check() -> int:
        deadline = time.monotonic() + timeout_seconds
        async with httpx.AsyncClient(timeout=300) as client:
            while True:
                try:
                    return await check_serving(authorization, client)
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    starting = isinstance(exc, httpx.TransportError) or exc.response.status_code >= 500
                    if not starting or time.monotonic() > deadline:
                        raise
                    print(f"[training] waiting for serving replicas ({type(exc).__name__})", flush=True)
                    await asyncio.sleep(15)

    print(f"[training] serving fits this run ({asyncio.run(check())} replica contract(s) answered)", flush=True)


def _stop_pool() -> None:
    """Stop the serving pool's app. Best effort: the training run's outcome stands either way."""
    try:
        modal.experimental.stop_app(DEPLOYMENT.app_name, environment_name=RUN.volume.environment_name)
        print(f"[training] stopped serving pool {DEPLOYMENT.app_name}", flush=True)
    except Exception as exc:
        print(
            f"[training] could not stop serving pool {DEPLOYMENT.app_name} ({type(exc).__name__}); stop it by hand",
            flush=True,
        )


def _service_commands() -> list[tuple[str, list[str], str]]:
    services: list[tuple[str, list[str], str]] = []
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
    # The rollout executor, Ray and Postgres share this container's CPUs; a GPU
    # function otherwise gets about one core and they starve.
    cpu=float(TRAINING.cpu),
    memory=TRAINING.memory_mib,
    volumes={str(DEPLOYMENT.base_mount): base_volume, str(SNAPSHOT_MOUNT): state_volume, **kernel_mounts},
    retries=modal.Retries(max_retries=TRAINING.max_retries, initial_delay=30.0) if TRAINING.max_retries else None,
    secrets=[modal.Secret.from_name(name, environment_name=RUN.volume.environment_name) for name in TRAINING.secrets],
    timeout=24 * 3600,
    max_containers=1,
)
def train() -> int:
    from miles_plugins.proximal import state_artifacts, state_checkpoints
    from miles_plugins.proximal.e2e import snapshots
    from miles_plugins.proximal.e2e.local_postgres import local_postgres
    from miles_plugins.proximal.state_writer import StateWriter, checkpoint_publisher

    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    # Only this composition root supplies run_state: its writer acknowledges the outbox.
    runtime_run = RUN.model_copy(update={"artifact_storage": RunStateArtifacts(kind="run_state")})
    CONFIG.write_text(runtime_run.model_dump_json())
    _set_keys()
    # A retry may land in the container of the failed attempt: clear its Ray and state.
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    snapshots.reset_local_state(STATE)
    logs = STATE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    pg_bin = sorted(Path("/usr/lib/postgresql").glob("*/bin"))[-1]
    context = _recovery_context()
    launch_id = uuid.uuid4().hex
    state_volume.reload()
    with local_postgres(STATE / "postgres") as dsn:
        os.environ[RUN.store_dsn_env] = dsn
        # Before any service connects: the store must be restored into an empty database.
        resume_step, parent = state_checkpoints.restore(
            snapshot_root=SNAPSHOT,
            checkpoints=STATE / "checkpoints",
            artifacts=Path(RUN.artifact_directory),
            dsn=dsn,
            pg_bin=pg_bin,
            context=context,
            selection=TRAINING.resume,
        )
        print(f"[training] {'resuming from step ' + str(resume_step) if resume_step is not None else 'fresh start'}")
        _record_launch(context, launch_id, runtime_run)
        state_artifacts.initialize(dsn)
        writer = StateWriter(
            dsn=dsn,
            run_id=RUN.run_id,
            artifacts=RUN.artifact_directory,
            snapshot_root=SNAPSHOT,
            commit=state_volume.commit,
            publish_checkpoint=checkpoint_publisher(
                dsn=dsn,
                pg_bin=pg_bin,
                checkpoints=STATE / "checkpoints",
                snapshot_root=SNAPSHOT,
                context=context,
                launch_id=launch_id,
                parent=parent,
                taken=resume_step,
                commit=state_volume.commit,
            ),
        )
        completed = False
        try:
            for name, command, health in _service_commands():
                log = (logs / f"{name}.log").open("ab")
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
                _wait_healthy(health, processes[-1], logs / f"{name}.log")
                print(f"[training] {name} ready", flush=True)
            if isinstance(TRAINING.platform, RealPlatform):
                _wait_for_registration(TRAINING.platform)
            _check_serving()
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
            writer.start()
            code = _run_trainer(command)
            if code != 0:  # Raise so Modal retries from the latest snapshot.
                raise RuntimeError(f"Trainer exited with {code}")
            completed = True
            return code
        finally:
            # Stop everything that can still write a checkpoint, then the snapshot thread,
            # before re-raising: a retry must never overlap this attempt's writes.
            subprocess.run(["ray", "stop", "--force"], check=False)
            # Every producer has stopped; drain paid captures and the last complete
            # optimizer boundary before the local Postgres context exits.
            try:
                writer.close()
            finally:
                for process in reversed(processes):
                    process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                # Compiled kernels this attempt built, for the next run; losing them costs only time.
                if kernel_volume is not None:
                    try:
                        kernel_volume.commit()
                    except Exception as exc:
                        print(f"[training] kernel cache commit failed ({type(exc).__name__})", flush=True)
                if stops_pool(TRAINING, completed=completed):
                    _stop_pool()


@app.function(
    image=image,
    cpu=float(TRAINING.cpu),
    memory=TRAINING.memory_mib,
    volumes={
        str(DEPLOYMENT.base_mount): base_volume.with_mount_options(read_only=True),
        str(SNAPSHOT_MOUNT): state_volume,
    },
    secrets=[modal.Secret.from_name(name, environment_name=RUN.volume.environment_name) for name in TRAINING.secrets],
    timeout=24 * 3600,
    max_containers=1,
    nonpreemptible=TRAINING.collection_nonpreemptible,
)
def collect(
    samples: int,
    fresh: bool,
    policy_json: str,
    yes_rollouts: bool,
    yes_publish: bool,
    rollouts_persist_to_volume: bool,
    collection_id: str,
) -> str:
    """CPU-only producer; the persistence flag uses TrainingDeployment.state_volume."""
    import httpx

    from miles_plugins.proximal.authorization import authorize_run
    from miles_plugins.proximal.clients import ServingPoolClient
    from miles_plugins.proximal.collect_batch import claim_collection, collect_persisted, validate_collection_request
    from miles_plugins.proximal.contracts import Policy
    from miles_plugins.proximal.e2e.local_postgres import local_postgres
    from miles_plugins.proximal.initial_policy import prepare_base_policy
    from miles_plugins.proximal.modal_volume import authorize_volume_publication, modal_publish_snapshot
    from miles_plugins.proximal.store import open_store

    policy = validate_collection_request(
        RUN,
        samples=samples,
        fresh=fresh,
        policy_json=policy_json,
        persist_to_volume=rollouts_persist_to_volume,
    )
    runtime_run = RUN.model_copy(update={"artifact_storage": RunStateArtifacts(kind="run_state")})
    authorization = authorize_run(runtime_run, yes_rollouts=yes_rollouts, yes_publish=yes_publish)
    state_volume.reload()
    if claim_collection(
        authorization,
        collection_id=collection_id,
        samples=samples,
        policy=policy,
        snapshot_root=SNAPSHOT,
        commit=state_volume.commit,
    ):
        print(f"Collection already complete: {collection_id}", flush=True)
        return collection_id
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(runtime_run.model_dump_json())
    _set_keys()
    work = STATE / "collections" / collection_id
    base_policy = prepare_base_policy(runtime_run, output=work / "initial-policy") if fresh else None
    collection_root = SNAPSHOT / "collections" / collection_id
    print(f"Rollouts persist to {TRAINING.state_volume.volume_name}:/{RUN.run_id}/artifacts/{RUN.run_id}", flush=True)
    print(f"Collection ID: {collection_id}", flush=True)
    processes: list[subprocess.Popen[bytes]] = []
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        for name, command, health in _service_commands():
            with (logs / f"{name}.log").open("ab") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            _wait_healthy(health, process, logs / f"{name}.log")
        if isinstance(TRAINING.platform, RealPlatform):
            _wait_for_registration(TRAINING.platform)
        _check_serving()
        if base_policy is not None:
            modal_publish_snapshot(authorize_volume_publication(RUN.volume, yes_publish=yes_publish), base_policy)
            policy = Policy(run_id=RUN.run_id, version=1, base_model=RUN.base_model, snapshot=base_policy.reference)
        assert policy is not None
        with local_postgres(work / "postgres") as dsn:
            os.environ[RUN.store_dsn_env] = dsn

            async def run() -> None:
                recipe = _recovery_context()
                store = await open_store(runtime_run)
                try:
                    async with httpx.AsyncClient(timeout=RUN.request_timeout_seconds) as client:
                        await ServingPoolClient(authorization, client).prepare(policy)
                    await store.commit_policy(policy)
                finally:
                    await store.close()
                await collect_persisted(
                    authorization,
                    config_path=CONFIG,
                    policy=policy,
                    num_samples=samples,
                    out=work / "collected",
                    snapshot_root=SNAPSHOT,
                    collection_root=collection_root,
                    commit=state_volume.commit,
                    base_policy=None if base_policy is None else base_policy.directory,
                    train_args=(*recipe.train_args, *recipe.model_args),
                )

            asyncio.run(run())
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print(f"Ready: {samples} rollouts at {collection_root / 'batch'}", flush=True)
    return collection_id


@app.local_entrypoint()
def main(
    collect_rollouts: int | None = None,
    rollouts_persist_to_volume: bool = False,
    fresh: bool = False,
    policy_file: str = "",
    yes_rollouts: bool = False,
    yes_publish: bool = False,
    collection_id: str = "",
) -> None:
    if collect_rollouts is not None:
        from miles_plugins.proximal.collect_batch import validate_collection_request

        if not rollouts_persist_to_volume or not yes_rollouts or not yes_publish:
            raise ValueError("Collection requires --rollouts-persist-to-volume --yes-rollouts --yes-publish")
        policy_json = Path(policy_file).read_text() if policy_file else ""
        validate_collection_request(
            RUN,
            samples=collect_rollouts,
            fresh=fresh,
            policy_json=policy_json,
            persist_to_volume=rollouts_persist_to_volume,
        )
        stable_id = TypeAdapter(SafeId).validate_python(collection_id or uuid.uuid4().hex)
        print(f"Launching collection: {stable_id}", flush=True)
        identity = collect.remote(
            collect_rollouts, fresh, policy_json, yes_rollouts, yes_publish, rollouts_persist_to_volume, stable_id
        )
        print(f"Collection ready: {identity}")
        return
    if fresh or policy_file or yes_rollouts or yes_publish or collection_id:
        raise ValueError("Collection options require --collect-rollouts N")
    # Online training already enables the same acknowledged Volume persistence.
    if rollouts_persist_to_volume:
        print(f"Rollout persistence enabled: {TRAINING.state_volume.volume_name}, run {RUN.run_id}")
    code = train.remote()
    print(f"Trainer exited with {code}")
