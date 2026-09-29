"""Two updates per LoRA configuration on one continuously allocated Modal cluster.

Compose the tested sizing cluster, stored-batch ReferenceReplay, and native Miles
trainer. Each phase starts fresh; its two updates share live trainer workers. Native
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
from miles_plugins.proximal.serving_app import RUN
from miles_plugins.proximal.state_artifacts import StateFile
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
            command = phase_command(plan, phase, RUN, bundle=bundle, save=LOCAL / phase.name / "checkpoints")
            sys.argv = command[1:]
            if hardware:
                args = parse_args()  # type: ignore[no-untyped-call]
            else:
                # Megatron's full validator queries the CUDA device architecture
                # for TP > 1. Use its real parser on CPU, then validate the pinned
                # model and sweep contract. Full validation runs on the live gang.
                from megatron.training.arguments import parse_args as parse_megatron_args  # type: ignore[import-not-found]

                args = parse_megatron_args(extra_args_provider=get_miles_extra_args_provider())  # type: ignore[no-untyped-call]
                hf_validate_args(args, load_hf_config(args.hf_checkpoint))  # type: ignore[no-untyped-call]
            validate_phase_args(args, plan, RUN)
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
    """Each node copies only its own completed native shards; rank zero certifies their union."""
    copied: set[str] = set()
    finished: set[str] = set()
    try:
        while not stop.wait(2):
            for phase in plan.phases:
                for step in range(phase.updates):
                    key = f"{phase.name}/{step}"
                    local = LOCAL / phase.name / "checkpoints" / f"iter_{step:07d}" / "adapter"
                    marker = local / "native_checkpoint.json"
                    if marker.is_file() and state.get(f"save/{key}") is None:
                        # The global trainer rank zero can live on any Modal node.
                        native = NativeCompletion.model_validate_json(marker.read_bytes())
                        state[f"save/{key}"] = native.model_dump_json()
                    saved = state.get(f"save/{key}")
                    if saved and key not in copied:
                        destination = root / phase.name / "checkpoints" / f"iter_{step:07d}" / "adapter"
                        with _VOLUME_LOCK:
                            files = publish_node_files(
                                authorization, local=local, destination=destination, commit=VOLUME.commit
                            )
                        receipt = TypeAdapter(tuple[StateFile, ...]).dump_json(files).decode()
                        state[f"files/{key}/{rank}"] = receipt
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
                                    receipts=[TypeAdapter(tuple[StateFile, ...]).validate_json(r) for r in receipts],
                                    root=root,
                                    commit=VOLUME.commit,
                                )
                            state[f"durable/{key}"] = result
                            finished.add(key)
                            print(
                                f"[batch-sweep] durable {phase.name} update {step + 1}: {result['eval_path']}",
                                flush=True,
                            )
    except BaseException as exc:
        state["error"] = f"checkpoint publisher node {rank}: {type(exc).__name__}: {exc}"


def _run_phases(plan: SweepPlan, state: cluster._State, root: Path, bundle: Path) -> list[dict[str, Any]]:
    results = []
    for phase in plan.phases:
        command = phase_command(plan, phase, RUN, bundle=bundle, save=LOCAL / phase.name / "checkpoints")
        # Trainer workers are disposed by train.py after the pair. The Ray cluster,
        # containers, GPUs and local kernel caches remain alive across both phases.
        result = cluster.run_command(
            command,
            label=phase.name,
            nodes=plan.nodes,
            samples=plan.samples,
            state=state,
            log=LOCAL / phase.name / "training.log",
        )
        result["targets"] = list(phase.target_modules)
        results.append(result)
        with _VOLUME_LOCK:
            (root / phase.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(LOCAL / phase.name / "training.log", root / phase.name / "training.log")
            write_atomic(root / phase.name / "timings.json", json.dumps(result, indent=2).encode())
            VOLUME.commit()
        if result["exit_code"] != 0 or len(result["perf"]) != phase.updates:
            raise RuntimeError(f"{phase.name} did not complete exactly {phase.updates} updates; artifacts retained")

        def phase_committed(phase: SweepPhase = phase) -> bool:
            return all(state.get(f"durable/{phase.name}/{step}") for step in range(phase.updates))

        cluster._wait(
            phase_committed,
            1800,
            f"{phase.name} native and eval checkpoints committed",
            state,
        )
    return results


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
    try:
        VOLUME.reload()
        if rank == 0:
            # A preempted gang retry sees this durable marker and cannot silently
            # start the paid experiment over or overwrite completed checkpoints.
            root.mkdir(parents=True, exist_ok=False)
            write_atomic(root / "plan.json", plan.model_dump_json().encode())
            write_atomic(root / "cluster.json", json.dumps({"cluster_id": info.cluster_id, "ips": ips}).encode())
            VOLUME.commit()
            validate_source(bundle, plan, source)
            _check_commands(plan, bundle, hardware=True)
            state["validated"] = True
        cluster._wait(lambda: state.get("validated"), 1800, "source validation", state)
        fabric = cluster._fabric()
        state[f"fabric-{rank}"] = fabric
        if not cluster._fabric_ok(fabric):
            raise RuntimeError(f"Node {rank} lacks verified RDMA")
        cluster.prepare_node()
        cluster._wait(lambda: all(state.get(f"fabric-{r}") for r in range(plan.nodes)), 900, "all fabrics", state)
        cluster._start_ray(state, rank, ips)
        sampler.thread.start()
        publisher.start()
        if rank != 0:
            cluster._wait(lambda: state.get("stop"), 8 * 3600, "sweep completion", state)
            state[f"memory-{rank}"] = sampler.peaks
            return {"rank": rank}
        results = _run_phases(plan, state, root, bundle)
        state["stop"] = True
        cluster._wait(
            lambda: all(state.get(f"memory-{r}") for r in range(1, plan.nodes)), 300, "worker reports", state
        )
        report = {
            "cluster_id": info.cluster_id,
            "batch_sha256": plan.batch_sha256,
            "phases": results,
            "peak_memory_mib": {"0": sampler.peaks, **{str(r): state[f"memory-{r}"] for r in range(1, plan.nodes)}},
            "checkpoints": [state[f"durable/{p.name}/{s}"] for p in plan.phases for s in range(p.updates)],
        }
        write_atomic(root / "completed.json", json.dumps(report, indent=2).encode())
        VOLUME.commit()
        return report
    except BaseException as exc:
        state["error"] = f"sweep node {rank}: {type(exc).__name__}: {exc}"
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
        phase_command(specification, phase, RUN, bundle=Path(local_bundle), save=Path("/unused") / phase.name)
    authorization = authorize_run(RUN, yes_rollouts=yes_train, yes_publish=yes_publish)
    # CPU-only validation in the exact training image runs before the GPU function.
    preflight.remote(specification.model_dump_json())
    with modal.Dict.ephemeral() as coordination:
        result = sweep.remote(specification.model_dump_json(), authorization, coordination)
    Path(out).write_text(json.dumps(result, indent=2))
