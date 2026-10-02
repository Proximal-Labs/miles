"""Single-update steps over different frozen batches, per LoRA configuration, on one owned Modal cluster.

Each arm (a LoRA target set) starts fresh from the batches' base policy and applies one
optimizer update per step: step k trains on ``batches[k]`` at policy lag k, continuing
the previous step's native weights, optimizer, scheduler and RNG. Native checkpoints and an
evaluation snapshot commit after every step, and each step's receipt names its predecessor.
With ``gate: "manual"``, every step after the first waits for approval in the control Dict.
This diagnostic does not generate rollouts or publish an online policy.

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
from typing import Any, Literal

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
    hold_failed_cluster,
    publish_ready,
    resume_confirmed,
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
app = modal.App("miles-qwen38-batch-chain")
_VOLUME_LOCK = threading.Lock()


def control_name(plan: ChainPlan) -> str:
    """The operator's Dict: ``go/<arm>/<step>`` approves a step, ``stop`` ends the chain.

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


def _check_commands(plan: ChainPlan, *, hardware: bool) -> None:
    from miles.utils.arguments import get_miles_extra_args_provider, hf_validate_args, parse_args
    from miles.utils.hf_utils.config import load_hf_config

    before = sys.argv
    try:
        for arm in plan.arms:
            for step in range(len(plan.batches)):
                command = step_command(
                    plan,
                    arm,
                    step,
                    RUN,
                    bundle=_local_batch(step),
                    save=_work(arm, step, 0) / "checkpoints",
                    resume_adapter=None if step == 0 else _work(arm, step, 0) / "resume",
                )
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


def _gate(control: modal.Dict, plan: ChainPlan, arm: ChainArm, step: int) -> Literal["go", "stop"]:
    """Wait for ``go/<arm>/<step>`` (or ``go-all``); ``stop`` or no answer before the timeout stops."""
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
        except Exception as exc:  # A control-plane outage is not approval; keep waiting until the deadline.
            print(f"[batch-chain] control read failed ({type(exc).__name__}: {exc})", flush=True)
        time.sleep(10)
    print(f"[batch-chain] no approval for {approval} within {plan.gate_timeout_seconds}s; stopping", flush=True)
    return "stop"


def _runtime(
    authorization: AuthorizedRun,
    plan: ChainPlan,
    state: cluster._State,
    control: modal.Dict | None,
    root: Path,
    rank: int,
    ips: list[str],
) -> ChainRuntime:
    def stage(arm: ChainArm, step: int, attempt: int, previous: StateFile | None) -> tuple[Path, Path | None]:
        if previous is None:
            return _local_batch(step), None
        with _VOLUME_LOCK:
            VOLUME.reload()
            target = _work(arm, step, attempt) / "resume"
            return _local_batch(step), stage_previous(MOUNT, root, plan, arm, step, previous, target)

    def reset_ray(key: str) -> None:
        subprocess.run(["ray", "stop", "--force"], check=True, capture_output=True, timeout=180)
        state[f"ray-stopped/{key}/{rank}"] = True
        cluster._wait(
            lambda: all(state.get(f"ray-stopped/{key}/{r}") for r in range(plan.nodes)),
            300,
            "failed Ray workers stopped",
            state,
        )
        cluster._start_ray(cluster._State(state.store, f"{state.scope}/ray/{key}"), rank, ips)

    def run_step(arm: ChainArm, step: int, attempt: int, bundle: Path, adapter: Path | None) -> dict[str, Any]:
        require_authorization(authorization)
        work = _work(arm, step, attempt)
        command = step_command(plan, arm, step, RUN, bundle=bundle, save=work / "checkpoints", resume_adapter=adapter)
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

    def persist(arm: ChainArm, step: int, attempt: int, result: dict[str, Any]) -> None:
        work = _work(arm, step, attempt)
        with _VOLUME_LOCK:
            destination = step_root(root, arm.name, step, attempt)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work / "training.log", destination / "training.log")
            write_atomic(destination / "timings.json", json.dumps(result, indent=2).encode())
            VOLUME.commit()
        summary = {
            key: result.get(key)
            for key in ("arm", "step", "attempt", "exit_code", "wall_s", "receipt", "policy_lag", "resume_confirmed")
        }
        summary["train"] = result["train"][-1] if result["train"] else None
        try:
            if control is not None:
                control[f"result/{arm.name}/{step + 1}/{attempt}"] = summary
                control["last"] = summary
        except Exception as exc:  # Status is a convenience; the committed timings are the record.
            print(f"[batch-chain] could not publish status ({type(exc).__name__}: {exc})", flush=True)
        print(f"[batch-chain] step result {json.dumps(summary, default=str)}", flush=True)

    def describe_receipt(path: str) -> StateFile:
        with _VOLUME_LOCK:
            VOLUME.reload()
            return describe(MOUNT / path, relative=path)

    return ChainRuntime(
        rank=rank,
        state=state,
        wait=lambda predicate, timeout, what: cluster._wait(predicate, timeout, what, state),
        stage=stage,
        reset_ray=reset_ray,
        gate=lambda arm, step: "go" if control is None else _gate(control, plan, arm, step),
        run_step=run_step,
        persist=persist,
        describe_receipt=describe_receipt,
    )


def _publisher(
    authorization: AuthorizedRun, plan: ChainPlan, state: cluster._State, root: Path, rank: int, policy: Policy
) -> ChainPublisher:
    def publish_node(arm: ChainArm, step: int, attempt: int, local: Path) -> tuple[StateFile, ...]:
        destination = checkpoint_dir(step_root(root, arm.name, step, attempt), step)
        with _VOLUME_LOCK:
            return publish_node_files(authorization, local=local, destination=destination, commit=VOLUME.commit)

    def finalize(
        arm: ChainArm, step: int, attempt: int, native: NativeCompletion, receipts: list[tuple[StateFile, ...]]
    ) -> str:
        previous = state.get(f"durable/{arm.name}/{step - 1}") if step > 0 else None
        with _VOLUME_LOCK:
            VOLUME.reload()
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
                commit=VOLUME.commit,
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
    """Async publication reads only local native shards; no trainer reads its Volume mount."""
    copied: set[str] = set()
    finished: set[str] = set()
    try:
        while not stop.wait(2):
            publish_ready(plan, publisher, copied, finished)
        publish_ready(plan, publisher, copied, finished)  # A save may land just before shutdown.
    except BaseException as exc:
        publisher.state["error"] = f"checkpoint publisher node {publisher.rank}: {type(exc).__name__}: {exc}"


@app.function(
    image=cluster.image,
    volumes={**cluster.VOLUMES, str(MOUNT): VOLUME},
    gpu="B300:8",
    cpu=32,
    memory=1024 * 1024,
    timeout=8 * 3600,
    retries=0,
)
@modal.experimental.clustered(size=cluster.NODES, rdma=True)  # type: ignore[untyped-decorator]
def chain(plan_json: str, authorization: AuthorizedRun, store: modal.Dict) -> dict[str, Any]:
    source = require_authorization(authorization)
    if source != RUN:
        raise ValueError("Chain authorization differs from the configured source")
    plan = ChainPlan.model_validate_json(plan_json)
    info = modal.experimental.get_cluster_info()
    rank, ips = info.rank, list(info.container_ipv4_ips)
    if plan.nodes != len(ips):
        raise ValueError("Allocated cluster differs from the validated plan")
    state = cluster._State(store, info.cluster_id)
    control = (
        modal.Dict.from_name(control_name(plan), environment_name=RUN.volume.environment_name, create_if_missing=False)
        if plan.gate == "manual"
        else None
    )
    root = _root(plan)
    sampler = cluster._MemorySampler(state, rank)
    publisher_stop = threading.Event()
    publisher: threading.Thread | None = None
    entered = time.monotonic()
    try:
        VOLUME.reload()
        if rank == 0:
            # A preempted gang retry sees this marker and cannot start the paid chain over.
            root.mkdir(parents=True, exist_ok=False)
            write_atomic(root / "plan.json", plan.model_dump_json().encode())
            write_atomic(
                root / "cluster.json",
                json.dumps({"cluster_id": info.cluster_id, "ips": ips, "coordination_dict": store.object_id}).encode(),
            )
            VOLUME.commit()
            batches = validate_batches(MOUNT, plan, source)
            _check_commands(plan, hardware=True)
            state["policy"] = batches[0].policy.model_dump_json()
        cluster._wait(lambda: state.get("policy"), 3600, "chain validation", state)
        policy = Policy.model_validate_json(state["policy"])
        for step, pinned in enumerate(plan.batches):
            stage_batch(MOUNT / pinned.path, _local_batch(step), batch_sha256=pinned.sha256)
        state[f"batches-staged/{rank}"] = True
        cluster._wait(
            lambda: all(state.get(f"batches-staged/{r}") for r in range(plan.nodes)),
            3600,
            "local immutable batches",
            state,
        )
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
        results = run_chain(plan, _runtime(authorization, plan, state, control, root, rank, ips))
        if rank != 0:
            cluster._wait(lambda: state.get("stop"), 8 * 3600, "chain completion", state)
            state[f"memory-{rank}"] = sampler.peaks
            return {"rank": rank}
        state["stop"] = True
        cluster._wait(
            lambda: all(state.get(f"memory-{r}") for r in range(1, plan.nodes)), 300, "worker reports", state
        )
        report = {
            "cluster_id": info.cluster_id,
            "behavior_policy": policy.model_dump(mode="json"),
            "batches": [pinned.model_dump(mode="json") for pinned in plan.batches],
            "steps": results,
            "receipts": {
                arm.name: [state.get(f"durable/{arm.name}/{step}") for step in range(len(plan.batches))]
                for arm in plan.arms
            },
            "stopped_by_operator": bool(state.get("stop-all")),
            "peak_memory_mib": {"0": sampler.peaks, **{str(r): state[f"memory-{r}"] for r in range(1, plan.nodes)}},
        }
        write_atomic(root / "completed.json", json.dumps(report, indent=2).encode())
        VOLUME.commit()
        return report
    except BaseException as exc:
        hold_failed_cluster(state, rank, hold_seconds=plan.failure_hold_seconds, entered=entered, exc=exc)
        raise
    finally:
        publisher_stop.set()
        if publisher is not None and publisher.is_alive():
            publisher.join(timeout=1800)
        sampler.stop.set()
        subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
        VOLUME.commit()


@app.local_entrypoint()
def main(plan: str, yes_train: bool = False, yes_publish: bool = False, out: str = "chain.json") -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12, matching the pinned image and Path serialization")
    specification = ChainPlan.model_validate_json(Path(plan).read_bytes())
    control = control_name(specification)
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
    if specification.gate == "manual":
        dictionary = modal.Dict.from_name(
            control, environment_name=RUN.volume.environment_name, create_if_missing=False
        )
        if dictionary.len() != 0:  # Stale approvals must never pass a gate.
            raise ValueError(f"Control Dict {control} must be newly created and empty")
    authorization = authorize_run(RUN, yes_rollouts=yes_train, yes_publish=yes_publish)
    print(json.dumps(preflight.remote(specification.model_dump_json()), indent=2))
    print(f"Approve each later step in Dict {control}: go/<arm>/<step> = True (1-based); stop = True ends the chain")
    with modal.Dict.ephemeral() as coordination:
        result = chain.remote(specification.model_dump_json(), authorization, coordination)
    Path(out).write_text(json.dumps(result, indent=2))
