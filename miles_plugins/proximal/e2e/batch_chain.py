"""Single-update steps over different frozen batches, per LoRA configuration, on one owned Modal cluster.

Each arm (a LoRA target set) starts fresh from the batches' base policy and applies one
optimizer update per step: step k trains on ``batches[k]`` at policy lag k, continuing
the previous step's native weights, optimizer, scheduler and RNG. Native checkpoints and an
evaluation snapshot commit after every step, and each step's receipt names its predecessor.
With ``gate: "manual"``, every step after the first waits for approval in the chain's Dict.
This diagnostic does not generate rollouts or publish an online policy.

Multi-node allocations are scarce. The local entrypoint only validates and spawns: the
clustered call does not depend on the launching process, and all coordination lives in an
operator-created named Dict. Transient control-plane and Volume errors are retried, failures
stay inside one step or arm, and a step starts only when the remaining function time holds it.

Set the usual PROXIMAL_* configs, SIZING_PROFILE=qwen38, and SIZING_NODES to the plan's nodes.
Run with --plan chain.json --yes-train --yes-publish.
"""

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import modal
import modal.experimental

from miles_plugins.proximal import modal_training as node
from miles_plugins.proximal.authorization import AuthorizedRun, authorize_run, require_authorization
from miles_plugins.proximal.contracts import Policy
from miles_plugins.proximal.e2e import step_sizing as cluster
from miles_plugins.proximal.e2e.batch_chain_artifacts import (
    RECEIPT,
    checkpoint_dir,
    finalize_chain_step,
    stage_previous,
    step_root,
)
from miles_plugins.proximal.e2e.batch_chain_coordination import (
    ChainPublisher,
    ChainRuntime,
    GateAnswer,
    hold_failed_cluster,
    publish_ready,
    resume_confirmed,
    retrying,
    run_chain,
    train_metrics,
)
from miles_plugins.proximal.e2e.batch_chain_inputs import (
    ChainArm,
    ChainPlan,
    step_command,
    validate_batches,
    validate_step_args,
)
from miles_plugins.proximal.e2e.batch_sweep_artifacts import publish_node_files
from miles_plugins.proximal.e2e.batch_sweep_recovery import stage_batch
from miles_plugins.proximal.serving_app import RUN
from miles_plugins.proximal.state_artifacts import StateFile, describe
from miles_plugins.proximal.state_checkpoints import NativeCompletion
from miles_plugins.proximal.storage import write_atomic

VOLUME = node.state_volume
MOUNT = Path("/snapshot")
LOCAL = Path("/work/batch-chain")
CHAIN_TIMEOUT_SECONDS = 24 * 3600  # Modal's maximum; steps are budgeted against it.
STATE_RETRY_SECONDS, VOLUME_RETRY_SECONDS, PUBLISHER_TOLERANCE_SECONDS = 120, 300, 300
app = modal.App("miles-qwen38-batch-chain")
_VOLUME_LOCK = threading.Lock()


class _ResilientState(cluster._State):
    """The cluster's coordination state, retrying transient Dict errors instead of failing the gang."""

    def get(self, key: str) -> Any:
        return retrying(
            lambda: super(_ResilientState, self).get(key), what=f"get {key}", budget_seconds=STATE_RETRY_SECONDS
        )

    def __getitem__(self, key: str) -> Any:
        return retrying(
            lambda: super(_ResilientState, self).__getitem__(key),
            what=f"read {key}",
            budget_seconds=STATE_RETRY_SECONDS,
        )

    def __setitem__(self, key: str, value: Any) -> None:
        retrying(
            lambda: super(_ResilientState, self).__setitem__(key, value),
            what=f"write {key}",
            budget_seconds=STATE_RETRY_SECONDS,
        )


class _Restarted(RuntimeError):
    """Modal replayed this call on a new gang; the original experiment directory already exists."""


def _reload() -> None:
    retrying(VOLUME.reload, what="Volume reload", budget_seconds=VOLUME_RETRY_SECONDS)


def _commit() -> None:
    retrying(VOLUME.commit, what="Volume commit", budget_seconds=VOLUME_RETRY_SECONDS)


def control_name(plan: ChainPlan) -> str:
    """The chain's Dict: coordination, ``go/<arm>/<step>`` approvals and ``stop``.

    The operator creates it, empty, before launch; this integration never creates named objects.
    """
    name = f"batch-chain-{plan.experiment_id}"
    if len(name) > 64:
        raise ValueError("Experiment ID is too long for its control Dict name")
    return name


def _root(plan: ChainPlan) -> Path:
    root = MOUNT / RUN.run_id / "experiments" / plan.experiment_id
    for pinned in plan.batches:
        bundle = MOUNT / pinned.path
        if bundle == root or root in bundle.parents or bundle in root.parents:
            raise ValueError("Experiment output must not overlap immutable batch input")
    return root


def _local_batch(step: int) -> Path:
    return LOCAL / "batches" / f"step-{step + 1}"


def _work(arm: ChainArm, step: int, attempt: int) -> Path:
    return step_root(LOCAL, arm.name, step, attempt)


def _parse(command: list[str], plan: ChainPlan, step: int, *, hardware: bool) -> None:
    from miles.utils.arguments import get_miles_extra_args_provider, hf_validate_args, parse_args
    from miles.utils.hf_utils.config import load_hf_config

    before = sys.argv
    try:
        sys.argv = command[1:]
        if hardware:
            args = parse_args()  # type: ignore[no-untyped-call]
        else:
            # Megatron's full validator queries the CUDA device for TP > 1: parse on CPU,
            # then validate the pinned model; the live gang runs the full validator.
            import megatron.training.arguments as megatron_args  # type: ignore[import-not-found,unused-ignore]

            args = megatron_args.parse_args(extra_args_provider=get_miles_extra_args_provider())  # type: ignore[no-untyped-call]
            hf_validate_args(args, load_hf_config(args.hf_checkpoint))  # type: ignore[no-untyped-call]
        validate_step_args(args, plan, RUN, step)
    finally:
        sys.argv = before


def _check_commands(plan: ChainPlan, *, hardware: bool) -> None:
    """CPU: every step's command. Hardware: first steps only; a continued step's full validation
    needs its staged predecessor (Miles restarts at rollout 0 without one), so it runs after staging."""
    for arm in plan.arms:
        for step in range(1 if hardware else len(plan.batches)):
            command = step_command(
                plan,
                arm,
                step,
                RUN,
                bundle=_local_batch(step),
                save=_work(arm, step, 0) / "checkpoints",
                resume_adapter=None if step == 0 else _work(arm, step, 0) / "resume",
            )
            _parse(command, plan, step, hardware=hardware)


@app.function(
    image=cluster.image,
    env={
        "LD_LIBRARY_PATH": "/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    },
    volumes={**cluster.VOLUMES, str(MOUNT): VOLUME.read_only()},
    cpu=4,
    memory=32768,
    timeout=1800,
)
def preflight(plan_json: str) -> dict[str, object]:
    """Pinned-image batch, lag-window and argument validation before allocating any GPUs."""
    plan = ChainPlan.model_validate_json(plan_json)
    if _root(plan).exists():
        raise FileExistsError("Choose a new experiment ID; implicit chain resume is forbidden")
    batches = validate_batches(MOUNT, plan, RUN)
    _check_commands(plan, hardware=False)
    return {
        "steps": len(batches),
        "arms": [arm.name for arm in plan.arms],
        "behavior_policy": batches[0].policy.model_dump(mode="json"),
        "max_policy_lag": RUN.research.max_policy_lag,
    }


def _gate(control: modal.Dict, plan: ChainPlan, arm: ChainArm, step: int) -> GateAnswer:
    """Wait for ``go/<arm>/<step>`` (or ``go-all``) or ``stop``; report a timeout to the coordinator."""
    approval = f"go/{arm.name}/{step + 1}"
    deadline = time.monotonic() + plan.gate_timeout_seconds
    print(f"[batch-chain] awaiting {approval} in Dict {control_name(plan)}", flush=True)
    while time.monotonic() < deadline:
        try:
            control["awaiting"] = approval
            if control.get("stop"):
                return "stop"
            if control.get(approval) or control.get("go-all"):
                return "go"
        except Exception as exc:  # A control-plane outage is not an answer; keep waiting until the deadline.
            print(f"[batch-chain] control read failed ({type(exc).__name__}: {exc})", flush=True)
        time.sleep(10)
    print(f"[batch-chain] no answer for {approval} within {plan.gate_timeout_seconds}s", flush=True)
    return "timeout"


def _reset_ray(state: _ResilientState, plan: ChainPlan, key: str, rank: int, ips: list[str]) -> None:
    """Stop this node's Ray processes, even stuck ones, then rejoin a fresh cluster."""
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True, timeout=180)
    if subprocess.run(["pgrep", "-f", "raylet|gcs_server"], check=False, capture_output=True).returncode == 0:
        subprocess.run(["pkill", "-9", "-f", "raylet|gcs_server|ray::"], check=False, capture_output=True)
        time.sleep(10)
    state[f"ray-stopped/{key}/{rank}"] = True
    cluster._wait(
        lambda: all(state.get(f"ray-stopped/{key}/{r}") for r in range(plan.nodes)),
        600,
        "failed Ray workers stopped",
        state,
    )
    cluster._start_ray(_ResilientState(state.store, f"{state.scope}/ray/{key}"), rank, ips)


def _run_step(
    authorization: AuthorizedRun,
    plan: ChainPlan,
    state: _ResilientState,
    arm: ChainArm,
    step: int,
    attempt: int,
    bundle: Path,
    adapter: Path | None,
) -> dict[str, Any]:
    require_authorization(authorization)
    work = _work(arm, step, attempt)
    command = step_command(plan, arm, step, RUN, bundle=bundle, save=work / "checkpoints", resume_adapter=adapter)
    if adapter is not None:
        try:  # The full validator, now that the predecessor's native state is staged.
            _parse(command, plan, step, hardware=True)
        except (ValueError, AssertionError, SystemExit) as exc:  # Rejects this arm, not the cluster.
            return {"rejected": f"{type(exc).__name__}: {exc}", "exit_code": None, "train": [], "wall_s": 0}
    result = cluster.run_command(
        command,
        label=f"{arm.name}-step-{step + 1}-attempt-{attempt}",
        nodes=plan.nodes,
        samples=plan.samples,
        state=state,
        log=work / "training.log",
    )
    log = (work / "training.log").read_text(errors="replace")
    result["train"] = train_metrics(log)
    result["targets"], result["policy_lag"] = list(arm.target_modules), step
    if adapter is not None:
        result["resume_confirmed"] = resume_confirmed(log, step)
    return result


def _persist(
    plan: ChainPlan, control: modal.Dict, root: Path, arm: ChainArm, step: int, attempt: int, result: dict[str, Any]
) -> None:
    work = _work(arm, step, attempt)
    with _VOLUME_LOCK:
        destination = step_root(root, arm.name, step, attempt)
        destination.mkdir(parents=True, exist_ok=True)
        if (work / "training.log").is_file():
            shutil.copy2(work / "training.log", destination / "training.log")
        write_atomic(destination / "timings.json", json.dumps(result, indent=2, default=str).encode())
        _commit()
    keys = ("arm", "step", "attempt", "exit_code", "wall_s", "receipt", "policy_lag", "resume_confirmed", "rejected")
    summary = {key: result.get(key) for key in keys} | {"train": result["train"][-1] if result.get("train") else None}
    try:
        control[f"result/{arm.name}/{step + 1}/{attempt}"] = summary
        control["last"] = summary
    except Exception as exc:  # Status is a convenience; the committed timings are the record.
        print(f"[batch-chain] could not publish status ({type(exc).__name__}: {exc})", flush=True)
    print(f"[batch-chain] step result {json.dumps(summary, default=str)}", flush=True)


def _runtime(
    authorization: AuthorizedRun,
    plan: ChainPlan,
    state: _ResilientState,
    control: modal.Dict,
    root: Path,
    rank: int,
    ips: list[str],
    deadline: float,
) -> ChainRuntime:
    def stage(arm: ChainArm, step: int, attempt: int, previous: StateFile | None) -> tuple[Path, Path | None]:
        if previous is None:
            return _local_batch(step), None
        with _VOLUME_LOCK:
            _reload()
            target = _work(arm, step, attempt) / "resume"
            return _local_batch(step), stage_previous(MOUNT, root, plan, arm, step, previous, target)

    def describe_receipt(path: str) -> StateFile:
        with _VOLUME_LOCK:
            _reload()
            return describe(MOUNT / path, relative=path)

    return ChainRuntime(
        rank=rank,
        state=state,
        wait=lambda predicate, timeout, what: cluster._wait(predicate, timeout, what, state),
        stage=stage,
        reset_ray=lambda key: _reset_ray(state, plan, key, rank, ips),
        gate=lambda arm, step: _gate(control, plan, arm, step),
        run_step=lambda arm, step, attempt, bundle, adapter: _run_step(
            authorization, plan, state, arm, step, attempt, bundle, adapter
        ),
        persist=lambda arm, step, attempt, result: _persist(plan, control, root, arm, step, attempt, result),
        describe_receipt=describe_receipt,
        remaining_seconds=lambda: deadline - time.monotonic(),
    )


def _publisher(
    authorization: AuthorizedRun, plan: ChainPlan, state: _ResilientState, root: Path, rank: int, policy: Policy
) -> ChainPublisher:
    def publish_node(arm: ChainArm, step: int, attempt: int, local: Path) -> tuple[StateFile, ...]:
        destination = checkpoint_dir(step_root(root, arm.name, step, attempt), step)
        with _VOLUME_LOCK:
            return publish_node_files(authorization, local=local, destination=destination, commit=_commit)

    def finalize(
        arm: ChainArm, step: int, attempt: int, native: NativeCompletion, receipts: list[tuple[StateFile, ...]]
    ) -> str:
        previous = state.get(f"durable/{arm.name}/{step - 1}") if step > 0 else None
        with _VOLUME_LOCK:
            _reload()
            finalize_chain_step(
                authorization,
                plan=plan,
                arm=arm,
                step=step,
                attempt=attempt,
                behavior_policy=policy,
                previous=describe(MOUNT / previous, relative=previous) if previous else None,
                native=native,
                receipts=receipts,
                root=root,
                commit=_commit,
            )
        receipt = step_root(root, arm.name, step, attempt) / RECEIPT
        print(f"[batch-chain] durable {arm.name} step {step + 1}: {receipt}", flush=True)
        return str(receipt.relative_to(MOUNT))

    return ChainPublisher(
        rank=rank,
        state=state,
        local_adapter=lambda arm, step, attempt: checkpoint_dir(_work(arm, step, attempt), step),
        publish_node=publish_node,
        finalize=finalize,
    )


def _publish_loop(plan: ChainPlan, publisher: ChainPublisher, stop: threading.Event) -> None:
    """Async publication of local native shards; tolerates errors until they persist."""
    done: set[str] = set()
    failing_since: float | None = None
    while True:
        stopping = stop.wait(5)
        try:
            publish_ready(plan, publisher, done)  # Once more after stop: a save may land just before shutdown.
            failing_since = None
        except Exception as exc:
            failing_since = failing_since or time.monotonic()
            print(f"[batch-chain] publisher pass failed ({type(exc).__name__}: {exc})", flush=True)
            if time.monotonic() - failing_since > PUBLISHER_TOLERANCE_SECONDS:
                publisher.state["error"] = f"checkpoint publisher node {publisher.rank}: {type(exc).__name__}: {exc}"
                return
        if stopping:
            return


def _start(plan: ChainPlan, state: _ResilientState, root: Path, rank: int, ips: list[str], cluster_id: str) -> Policy:
    """Rank 0 records and validates the experiment; every node stages every batch locally."""
    _reload()
    if rank == 0:
        if root.exists():
            previous = json.loads((root / "cluster.json").read_text()).get("cluster_id")
            if previous != cluster_id:
                state["restart"] = True
                raise _Restarted(f"Experiment {plan.experiment_id} began on cluster {previous}")
        # A replay of this experiment ID cannot start the paid chain over or overwrite results.
        root.mkdir(parents=True, exist_ok=False)
        write_atomic(root / "plan.json", plan.model_dump_json().encode())
        write_atomic(root / "cluster.json", json.dumps({"cluster_id": cluster_id, "ips": ips}).encode())
        _commit()
        batches = validate_batches(MOUNT, plan, RUN)
        _check_commands(plan, hardware=True)
        state["policy"] = batches[0].policy.model_dump_json()
    cluster._wait(lambda: state.get("policy") or state.get("restart"), 3600, "chain validation", state)
    if state.get("restart"):
        raise _Restarted("Modal restarted the chain on a new gang")
    for step, pinned in enumerate(plan.batches):
        stage_batch(MOUNT / pinned.path, _local_batch(step), batch_sha256=pinned.sha256)
    state[f"batches-staged/{rank}"] = True
    cluster._wait(
        lambda: all(state.get(f"batches-staged/{r}") for r in range(plan.nodes)),
        3600,
        "local immutable batches",
        state,
    )
    return Policy.model_validate_json(state["policy"])


def _report(
    plan: ChainPlan, state: _ResilientState, results: list[dict[str, Any]], policy: Policy, peaks: Any
) -> dict[str, Any]:
    return {
        "behavior_policy": policy.model_dump(mode="json"),
        "batches": [pinned.model_dump(mode="json") for pinned in plan.batches],
        "steps": results,
        "receipts": {
            arm.name: [state.get(f"durable/{arm.name}/{step}") for step in range(len(plan.batches))]
            for arm in plan.arms
        },
        "failed_arms": {
            arm.name: state.get(f"arm-failed/{arm.name}") for arm in plan.arms if state.get(f"arm-failed/{arm.name}")
        },
        "stop_reason": state.get("stop-reason"),
        "peak_memory_mib": peaks,
    }


@app.function(
    image=cluster.image,
    volumes={**cluster.VOLUMES, str(MOUNT): VOLUME},
    gpu="B300:8",
    cpu=32,
    memory=1024 * 1024,
    timeout=CHAIN_TIMEOUT_SECONDS,
    retries=0,
)
@modal.experimental.clustered(size=cluster.NODES, rdma=True)  # type: ignore[untyped-decorator]
def chain(plan_json: str, authorization: AuthorizedRun, store: modal.Dict) -> dict[str, Any]:
    entered = time.monotonic()
    source = require_authorization(authorization)
    if source != RUN:
        raise ValueError("Chain authorization differs from the configured source")
    plan = ChainPlan.model_validate_json(plan_json)
    info = modal.experimental.get_cluster_info()
    rank, ips = info.rank, list(info.container_ipv4_ips)
    if plan.nodes != len(ips):
        raise ValueError("Allocated cluster differs from the validated plan")
    state = _ResilientState(store, info.cluster_id)
    root = _root(plan)
    sampler = cluster._MemorySampler(state, rank)
    publisher_stop = threading.Event()
    publisher: threading.Thread | None = None
    try:
        policy = _start(plan, state, root, rank, ips, info.cluster_id)
        fabric = cluster._fabric()
        state[f"fabric-{rank}"] = fabric
        if not cluster._fabric_ok(fabric):
            raise RuntimeError(f"Node {rank} lacks verified RDMA")
        cluster.prepare_node()
        cluster._wait(lambda: all(state.get(f"fabric-{r}") for r in range(plan.nodes)), 900, "all fabrics", state)
        cluster._start_ray(state, rank, ips)
        sampler.thread.start()
        publisher = threading.Thread(
            target=_publish_loop,
            args=(plan, _publisher(authorization, plan, state, root, rank, policy), publisher_stop),
            daemon=True,
        )
        publisher.start()
        deadline = entered + CHAIN_TIMEOUT_SECONDS - 600
        results = run_chain(plan, _runtime(authorization, plan, state, store, root, rank, ips, deadline))
        if rank != 0:
            cluster._wait(lambda: state.get("stop"), CHAIN_TIMEOUT_SECONDS, "chain completion", state)
            state[f"memory-{rank}"] = sampler.peaks or {"none": {}}
            return {"rank": rank}
        state["stop"] = True
        cluster._wait(
            lambda: all(state.get(f"memory-{r}") for r in range(1, plan.nodes)), 300, "worker reports", state
        )
        peaks = {"0": sampler.peaks, **{str(r): state[f"memory-{r}"] for r in range(1, plan.nodes)}}
        report = _report(plan, state, results, policy, peaks) | {"cluster_id": info.cluster_id}
        write_atomic(root / "completed.json", json.dumps(report, indent=2, default=str).encode())
        _commit()
        return report
    except BaseException as exc:
        if not (isinstance(exc, _Restarted) or state.get("restart")):
            hold_failed_cluster(
                state,
                rank,
                hold_seconds=plan.failure_hold_seconds,
                entered=entered,
                limit_seconds=CHAIN_TIMEOUT_SECONDS,
                exc=exc,
            )
        raise
    finally:
        publisher_stop.set()
        if publisher is not None and publisher.is_alive():
            publisher.join(timeout=1800)
        sampler.stop.set()
        subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
        try:
            _commit()
        except Exception as exc:
            print(f"[batch-chain] final commit failed ({type(exc).__name__}: {exc})", flush=True)


@app.local_entrypoint()
def main(plan: str, yes_train: bool = False, yes_publish: bool = False, out: str = "chain-launch.json") -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12, matching the pinned image and Path serialization")
    specification = ChainPlan.model_validate_json(Path(plan).read_bytes())
    name = control_name(specification)
    if cluster.PROFILE != "qwen38" or specification.nodes != cluster.NODES:
        raise ValueError("Set SIZING_PROFILE=qwen38 and SIZING_NODES to the plan's node count")
    for arm in specification.arms:
        for step in range(len(specification.batches)):
            # Command shapes here; batches, lag windows and parsed arguments in the exact CPU image.
            step_command(
                specification,
                arm,
                step,
                RUN,
                bundle=Path("/validated-by-preflight"),
                save=Path("/unused"),
                resume_adapter=Path("/staged-at-runtime") if step else None,
            )
    store = modal.Dict.from_name(name, environment_name=RUN.volume.environment_name, create_if_missing=False)
    if store.len() != 0:  # Stale approvals or coordination keys must never steer a new chain.
        raise ValueError(f"Chain Dict {name} must be newly created and empty")
    authorization = authorize_run(RUN, yes_rollouts=yes_train, yes_publish=yes_publish)
    print(json.dumps(preflight.remote(specification.model_dump_json()), indent=2))
    # Spawned, not awaited: the clustered call must not depend on this process staying alive.
    call = chain.spawn(specification.model_dump_json(), authorization, store)
    record = {"function_call_id": call.object_id, "chain_dict": name, "experiment_id": specification.experiment_id}
    Path(out).write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"Approve later steps in Dict {name}: go/<arm>/<step> = True (1-based); stop = True ends the chain")
