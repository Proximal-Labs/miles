"""Two updates per LoRA configuration on one continuously allocated Modal cluster.

Compose the tested sizing cluster, stored-batch ReferenceReplay, and native Miles
trainer. Phases declare fresh or certified native initialization. Native
checkpoints and evaluation exports are committed after every update. This diagnostic
does not generate rollouts, publish an online policy, or relax production resume rules.

Set the usual PROXIMAL_* configs, SIZING_PROFILE=qwen38, and SIZING_NODES=8.
Run with --plan plan.json --local-bundle /verified/batch --yes-train --yes-publish.
"""

import json
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import modal
import modal.experimental
from pydantic import TypeAdapter

from miles_plugins.proximal import modal_training as node
from miles_plugins.proximal.authorization import AuthorizedRun, authorize_run, require_authorization
from miles_plugins.proximal.e2e import step_sizing as cluster
from miles_plugins.proximal.e2e.batch_sweep_artifacts import finalize_step, publish_node_files
from miles_plugins.proximal.e2e.batch_sweep_inputs import (
    SweepPhase,
    SweepPlan,
    phase_command,
    validate_phase_args,
    validate_source,
)
from miles_plugins.proximal.e2e.batch_sweep_recovery import (
    completed_training_steps,
    read_resume,
    stage_batch,
    stage_resume,
)
from miles_plugins.proximal.serving_app import RUN
from miles_plugins.proximal.state_artifacts import StateFile, describe
from miles_plugins.proximal.state_checkpoints import NativeCompletion
from miles_plugins.proximal.storage import write_atomic

VOLUME = node.state_volume
MOUNT = Path("/snapshot")
LOCAL = Path("/work/batch-sweep")
app = modal.App("miles-qwen38-batch-sweep")
_VOLUME_LOCK = threading.Lock()


def _paths(plan: SweepPlan) -> tuple[Path, Path]:
    bundle = MOUNT / plan.batch_path
    root = MOUNT / RUN.run_id / "experiments" / plan.experiment_id
    if bundle == root or root in bundle.parents or bundle in root.parents:
        raise ValueError("Experiment output must not overlap immutable batch input")
    return bundle, root


def _check_commands(plan: SweepPlan, bundle: Path, *, hardware: bool) -> None:
    from miles.utils.arguments import get_miles_extra_args_provider, hf_validate_args, parse_args
    from miles.utils.hf_config import load_hf_config

    before = sys.argv
    try:
        for phase in plan.phases:
            resume = read_resume(MOUNT, plan, phase)
            command = phase_command(
                plan,
                phase,
                RUN,
                bundle=bundle,
                save=LOCAL / phase.name / "checkpoints",
                resume_adapter=resume[0] if resume else None,
            )
            sys.argv = command[1:]
            if hardware:
                args = parse_args()  # type: ignore[no-untyped-call]
            else:
                # Megatron's full validator queries the CUDA device architecture
                # for TP > 1. Use its real parser on CPU, then validate the pinned
                # model and sweep contract. Full validation runs on the live gang.
                from megatron.training.arguments import (  # type: ignore[import-not-found,unused-ignore]
                    parse_args as parse_megatron_args,
                )

                args = parse_megatron_args(extra_args_provider=get_miles_extra_args_provider())  # type: ignore[no-untyped-call]
                hf_validate_args(args, load_hf_config(args.hf_checkpoint))  # type: ignore[no-untyped-call]
            validate_phase_args(args, plan, RUN, phase)
            if resume:
                for name, value in resume[1].native.layout.items():
                    if int(getattr(args, name, 1) or 1) != value:
                        raise ValueError(f"Resume changes native {name}")
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
    """Pinned-image argument and artifact validation before allocating any GPUs."""
    plan = SweepPlan.model_validate_json(plan_json)
    bundle, root = _paths(plan)
    if root.exists():
        raise FileExistsError("Choose a new experiment ID; implicit sweep resume is forbidden")
    batch = validate_source(bundle, plan, RUN)
    _check_commands(plan, bundle, hardware=False)
    return {"samples": batch.num_samples, "batch_sha256": plan.batch_sha256, "phases": len(plan.phases)}


def _publish_steps(
    authorization: AuthorizedRun, plan: SweepPlan, state: cluster._State, rank: int, root: Path, stop: threading.Event
) -> None:
    """Async publication reads only local native shards; no trainer reads its Volume mount."""
    copied: set[str] = set()
    finished: set[str] = set()
    try:
        while not stop.wait(2):
            for phase in plan.phases:
                for attempt in range(plan.phase_attempts):
                    attempt_key = f"{phase.name}/{attempt}"
                    if not state.get(f"running/{attempt_key}"):
                        continue
                    attempt_root = root / f"attempt-{attempt}"
                    for step in range(phase.updates):
                        key = f"{attempt_key}/{step}"
                        local = (
                            LOCAL / f"attempt-{attempt}" / phase.name / "checkpoints" / f"iter_{step:07d}" / "adapter"
                        )
                        marker = local / "native_checkpoint.json"
                        if marker.is_file() and state.get(f"save/{key}") is None:
                            native = NativeCompletion.model_validate_json(marker.read_bytes())
                            state[f"save/{key}"] = native.model_dump_json()
                        saved = state.get(f"save/{key}")
                        if saved and key not in copied:
                            destination = attempt_root / phase.name / "checkpoints" / f"iter_{step:07d}" / "adapter"
                            with _VOLUME_LOCK:
                                files = publish_node_files(
                                    authorization, local=local, destination=destination, commit=VOLUME.commit
                                )
                            state[f"files/{key}/{rank}"] = TypeAdapter(tuple[StateFile, ...]).dump_json(files).decode()
                            copied.add(key)
                        if rank == 0 and saved and key not in finished:
                            receipts = [state.get(f"files/{key}/{r}") for r in range(plan.nodes)]
                            if all(receipts):
                                with _VOLUME_LOCK:
                                    VOLUME.reload()
                                    result = finalize_step(
                                        authorization,
                                        plan=plan,
                                        phase=phase,
                                        step=step,
                                        native=NativeCompletion.model_validate_json(saved),
                                        receipts=[
                                            TypeAdapter(tuple[StateFile, ...]).validate_json(r) for r in receipts
                                        ],
                                        root=attempt_root,
                                        commit=VOLUME.commit,
                                    )
                                result["receipt_path"] = str(
                                    (attempt_root / phase.name / f"step-{step + 1}.json").relative_to(MOUNT)
                                )
                                state[f"durable/{phase.name}/{step}"] = result
                                finished.add(key)
                                print(
                                    f"[batch-sweep] durable {phase.name} update {step + 1}: {result['receipt_path']}",
                                    flush=True,
                                )
                    # Only acknowledge after scanning every possible completed save
                    # following subprocess exit, including a save on a non-head node.
                    if state.get(f"exited/{attempt_key}"):
                        state[f"scanned/{attempt_key}/{rank}"] = True
    except BaseException as exc:
        state["error"] = f"checkpoint publisher node {rank}: {type(exc).__name__}: {exc}"


def _all_keys(state: cluster._State, keys: tuple[str, ...]) -> bool:
    return all(state.get(key) for key in keys)


def _run_phases(
    authorization: AuthorizedRun,
    plan: SweepPlan,
    state: cluster._State,
    root: Path,
    bundle: Path,
    rank: int,
    ips: list[str],
) -> list[dict[str, Any]]:
    require_authorization(authorization)
    results = []
    for original in plan.phases:
        for attempt in range(plan.phase_attempts):
            key = f"{original.name}/{attempt}"
            if rank == 0:
                done = state.get(f"phase-complete/{original.name}")
                phase = original
                previous = state.get(f"durable/{original.name}/0")
                if previous:
                    path = MOUNT / previous["receipt_path"]
                    phase = original.model_copy(update={"resume": describe(path, relative=previous["receipt_path"])})
                state[f"reset/{key}"] = bool(state.get("needs-ray-reset"))
                state[f"input/{key}"] = "skip" if done else phase.model_dump_json()
            cluster._wait(partial(state.get, f"input/{key}"), 8 * 3600, "phase decision", state)
            if state[f"input/{key}"] == "skip":
                break
            phase = SweepPhase.model_validate_json(state[f"input/{key}"])
            work = LOCAL / f"attempt-{attempt}" / phase.name
            # The preceding attempt's publishers have drained. Stage every native
            # rank on every node: Ray may assign global ranks differently on retry.
            with _VOLUME_LOCK:
                VOLUME.reload()
                adapter = stage_resume(MOUNT, plan, phase, work / "resume")
            state[f"staged/{key}/{rank}"] = True
            cluster._wait(
                partial(_all_keys, state, tuple(f"staged/{key}/{r}" for r in range(plan.nodes))),
                1800,
                "all resume shards staged",
                state,
            )
            if state[f"reset/{key}"]:
                subprocess.run(["ray", "stop", "--force"], check=True, capture_output=True, timeout=180)
                state[f"ray-stopped/{key}/{rank}"] = True
                cluster._wait(
                    partial(_all_keys, state, tuple(f"ray-stopped/{key}/{r}" for r in range(plan.nodes))),
                    300,
                    "failed Ray workers stopped",
                    state,
                )
                ray_state = cluster._State(state.store, f"{state.scope}/ray/{key}")
                cluster._start_ray(ray_state, rank, ips)
            if rank == 0:
                state["needs-ray-reset"] = False
                state[f"running/{key}"] = True
                command = phase_command(
                    plan, phase, RUN, bundle=bundle, save=work / "checkpoints", resume_adapter=adapter
                )
                result = cluster.run_command(
                    command,
                    label=f"{phase.name}-attempt-{attempt}",
                    nodes=plan.nodes,
                    samples=plan.samples,
                    state=state,
                    log=work / "training.log",
                )
                result["targets"], result["attempt"] = list(phase.target_modules), attempt
                state[f"exited/{key}"] = True
                cluster._wait(
                    partial(_all_keys, state, tuple(f"scanned/{key}/{r}" for r in range(plan.nodes))),
                    1800,
                    "checkpoint publishers drained",
                    state,
                )
                saved = [step for step in range(phase.updates) if state.get(f"save/{key}/{step}")]
                cluster._wait(
                    partial(_all_keys, state, tuple(f"durable/{phase.name}/{step}" for step in saved)),
                    1800,
                    "completed native saves committed",
                    state,
                )
                with _VOLUME_LOCK:
                    destination = root / f"attempt-{attempt}" / phase.name
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(work / "training.log", destination / "training.log")
                    write_atomic(destination / "timings.json", json.dumps(result, indent=2).encode())
                    VOLUME.commit()
                results.append(result)
                start = 1 if phase.resume else 0
                expected = set(range(start, phase.updates))
                complete = expected == completed_training_steps(result["perf"]) and all(
                    state.get(f"durable/{phase.name}/{step}") for step in expected
                )
                state[f"phase-complete/{phase.name}"] = complete
                state["needs-ray-reset"] = result["exit_code"] != 0
                state[f"finished/{key}"] = True
                if not complete:
                    print(f"[batch-sweep] {key} incomplete; preserving nodes and committed updates", flush=True)
                    # A committed final update must never be applied again just
                    # because a trailing metric/cleanup operation failed.
                    if state.get(f"durable/{phase.name}/1"):
                        raise RuntimeError(
                            "Final checkpoint committed but completion metrics missing; inspect held cluster"
                        )
            cluster._wait(partial(state.get, f"finished/{key}"), 8 * 3600, "phase result", state)
    if not all(state.get(f"phase-complete/{phase.name}") for phase in plan.phases):
        raise RuntimeError("Sweep still has missing updates after bounded recovery; holding allocation")
    return results


def _hold_failed_cluster(
    state: cluster._State, rank: int, plan: SweepPlan, entered: float, exc: BaseException
) -> None:
    # Coordination failures must not themselves bypass the allocation hold.
    try:
        if not state.get("error"):
            state["error"] = f"sweep node {rank}: {type(exc).__name__}: {exc}"
        state[f"holding/{rank}"] = str(exc)
    except Exception as coordination_error:
        print(f"[batch-sweep] could not record hold: {coordination_error}", flush=True)
    deadline = min(entered + 8 * 3600 - 60, time.monotonic() + plan.failure_hold_seconds)
    print(f"[batch-sweep] HOLDING node {rank}: {exc}; release through cluster coordination", flush=True)
    while time.monotonic() < deadline:
        try:
            if state.get("release"):
                break
        except Exception:
            # No release acknowledgement means continue the already-authorized
            # bounded hold, even while the control plane is unavailable.
            pass
        time.sleep(5)


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
def sweep(plan_json: str, authorization: AuthorizedRun, store: modal.Dict) -> dict[str, Any]:
    source = require_authorization(authorization)
    if source != RUN:
        raise ValueError("Sweep authorization differs from the configured source")
    plan = SweepPlan.model_validate_json(plan_json)
    info = modal.experimental.get_cluster_info()
    rank, ips = info.rank, list(info.container_ipv4_ips)
    if plan.nodes != len(ips):
        raise ValueError("Allocated cluster differs from the validated plan")
    state = cluster._State(store, info.cluster_id)
    bundle, root = _paths(plan)
    publisher_stop = threading.Event()
    sampler = cluster._MemorySampler(state, rank)
    publisher = threading.Thread(
        target=_publish_steps, args=(authorization, plan, state, rank, root, publisher_stop), daemon=True
    )
    entered = time.monotonic()
    try:
        VOLUME.reload()
        if rank == 0:
            # A preempted gang retry sees this durable marker and cannot silently
            # start the paid experiment over or overwrite completed checkpoints.
            root.mkdir(parents=True, exist_ok=False)
            write_atomic(root / "plan.json", plan.model_dump_json().encode())
            write_atomic(
                root / "cluster.json",
                json.dumps({"cluster_id": info.cluster_id, "ips": ips, "coordination_dict": store.object_id}).encode(),
            )
            for attempt in range(plan.phase_attempts):
                write_atomic(root / f"attempt-{attempt}" / "plan.json", plan.model_dump_json().encode())
            VOLUME.commit()
            validate_source(bundle, plan, source)
            _check_commands(plan, bundle, hardware=True)
            state["validated"] = True
        cluster._wait(lambda: state.get("validated"), 1800, "source validation", state)
        stage_batch(bundle, LOCAL / "batch", plan)
        bundle = LOCAL / "batch"
        state[f"batch-staged/{rank}"] = True
        cluster._wait(
            lambda: all(state.get(f"batch-staged/{r}") for r in range(plan.nodes)),
            1800,
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
        publisher.start()
        results = _run_phases(authorization, plan, state, root, bundle, rank, ips)
        if rank != 0:
            cluster._wait(lambda: state.get("stop"), 8 * 3600, "sweep completion", state)
            state[f"memory-{rank}"] = sampler.peaks
            return {"rank": rank}
        state["stop"] = True
        cluster._wait(
            lambda: all(state.get(f"memory-{r}") for r in range(1, plan.nodes)), 300, "worker reports", state
        )
        report = {
            "cluster_id": info.cluster_id,
            "batch_sha256": plan.batch_sha256,
            "phases": results,
            "peak_memory_mib": {"0": sampler.peaks, **{str(r): state[f"memory-{r}"] for r in range(1, plan.nodes)}},
            "checkpoints": [
                state[f"durable/{p.name}/{s}"] for p in plan.phases for s in range(1 if p.resume else 0, p.updates)
            ],
            "inherited": [p.resume.model_dump(mode="json") for p in plan.phases if p.resume],
        }
        write_atomic(root / "completed.json", json.dumps(report, indent=2).encode())
        VOLUME.commit()
        return report
    except BaseException as exc:
        _hold_failed_cluster(state, rank, plan, entered, exc)
        raise
    finally:
        publisher_stop.set()
        if publisher.is_alive():
            publisher.join(timeout=1800)
        sampler.stop.set()
        subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
        VOLUME.commit()


@app.local_entrypoint()
def main(
    plan: str, local_bundle: str, yes_train: bool = False, yes_publish: bool = False, out: str = "sweep.json"
) -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12, matching the pinned image and Path serialization")
    specification = SweepPlan.model_validate_json(Path(plan).read_bytes())
    if cluster.PROFILE != "qwen38" or specification.nodes != cluster.NODES:
        raise ValueError("Set SIZING_PROFILE=qwen38 and SIZING_NODES to the plan's node count")
    validate_source(Path(local_bundle), specification, RUN)
    for phase in specification.phases:
        # Resume bytes and topology are validated in the exact CPU image before GPUs.
        phase_command(
            specification,
            phase,
            RUN,
            bundle=Path(local_bundle),
            save=Path("/unused") / phase.name,
            resume_adapter=Path("/validated-by-preflight") if phase.resume else None,
        )
    authorization = authorize_run(RUN, yes_rollouts=yes_train, yes_publish=yes_publish)
    # CPU-only validation in the exact training image runs before the GPU function.
    preflight.remote(specification.model_dump_json())
    with modal.Dict.ephemeral() as coordination:
        result = sweep.remote(specification.model_dump_json(), authorization, coordination)
    Path(out).write_text(json.dumps(result, indent=2))
