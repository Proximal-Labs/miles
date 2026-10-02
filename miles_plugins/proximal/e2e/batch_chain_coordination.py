"""Arm-by-arm, step-by-step decisions for a batch chain on one owned cluster.

Pure coordination over the shared cluster state: no Modal image, GPUs or Volume here, so
tests drive the real decision logic. Every node runs ``run_chain``. Rank 0 decides each
step, waits at the operator gate and runs the trainer; every node stages the step's inputs
and, through ``publish_ready``, commits its own native shards. A committed update is never
applied again: a step is complete once its receipt is durable, whatever the exit code.

Multi-node allocations are scarce, so failures stay as local as possible: a transient
control-plane error is retried, a failed step or arm never stops an independent arm, and a
step starts only when the remaining function time can hold it.
"""

import ast
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import TypeAdapter

from miles_plugins.proximal.e2e.batch_chain_inputs import ChainArm, ChainPlan
from miles_plugins.proximal.state_artifacts import StateFile
from miles_plugins.proximal.state_checkpoints import NativeCompletion

Decision = Literal["go", "stop", "skip"]
GateAnswer = Literal["go", "stop", "timeout"]
_FILES = TypeAdapter(tuple[StateFile, ...])
_TRAIN = re.compile(r"step (\d+): (\{'train/.*\})")
# A step starts only if the remaining function time covers this multiple of the longest
# step so far (or the plan's estimate) plus a fixed margin for save, publish and cleanup.
STEP_TIME_FACTOR, STEP_TIME_MARGIN_S = 1.3, 900
T = TypeVar("T")


class SharedState(Protocol):
    def get(self, key: str) -> Any: ...

    def __getitem__(self, key: str) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...


def retrying(
    call: Callable[[], T],
    *,
    what: str,
    budget_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a transient control-plane or Volume call with backoff; raise only after the budget."""
    deadline, delay = clock() + budget_seconds, 0.5
    while True:
        try:
            return call()
        except Exception as exc:
            if clock() + delay > deadline:
                raise
            print(f"[batch-chain] {what} failed ({type(exc).__name__}: {exc}); retrying in {delay:.1f}s", flush=True)
            sleep(delay)
            delay = min(delay * 2, 15.0)


@dataclass(frozen=True)
class ChainRuntime:
    """What one node can do. Only rank 0's ``gate``, ``run_step`` and ``persist`` are called."""

    rank: int
    state: SharedState
    wait: Callable[[Callable[[], Any], float, str], None]
    stage: Callable[[ChainArm, int, int, StateFile | None], tuple[Path, Path | None]]
    reset_ray: Callable[[str], None]
    gate: Callable[[ChainArm, int], GateAnswer]
    run_step: Callable[[ChainArm, int, int, Path, Path | None], dict[str, Any]]
    persist: Callable[[ChainArm, int, int, dict[str, Any]], None]
    describe_receipt: Callable[[str], StateFile]
    remaining_seconds: Callable[[], float]


@dataclass(frozen=True)
class ChainPublisher:
    """One node's view of its local saves, and rank 0's step finalization."""

    rank: int
    state: SharedState
    local_adapter: Callable[[ChainArm, int, int], Path]
    publish_node: Callable[[ChainArm, int, int, Path], tuple[StateFile, ...]]
    finalize: Callable[[ChainArm, int, int, NativeCompletion, list[tuple[StateFile, ...]]], str]


def train_metrics(log: str) -> list[dict[str, Any]]:
    """The trainer's per-update ``train/*`` metrics (loss, grad norm, TIS, train/rollout KL)."""
    rows: dict[int, dict[str, Any]] = {}
    for match in _TRAIN.finditer(log):
        try:
            rows[int(match.group(1))] = {"step": int(match.group(1)), **ast.literal_eval(match.group(2))}
        except (ValueError, SyntaxError):
            continue
    return [rows[step] for step in sorted(rows)]


def resume_confirmed(log: str, step: int) -> bool:
    """Miles falls back to a fresh adapter when it cannot load one; a continued step must not."""
    if "freshly initialized adapter weights" in log or "No adapter checkpoint found" in log:
        return False
    return "Restored optimizer state from LoRA checkpoint" in log and (
        f"Resuming LoRA training from iteration {step - 1}" in log
    )


def healthy(result: dict[str, Any], step: int) -> bool:
    """A durable update with finite loss and grad norm that, if continued, proved its resume."""
    if not result.get("receipt") or (step > 0 and result.get("resume_confirmed") is not True):
        return False
    rows = result.get("train") or []
    last = rows[-1] if rows else {}
    values = [last.get("train/loss"), last.get("train/grad_norm")]
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _key(arm: ChainArm, step: int, attempt: int) -> str:
    return f"{arm.name}/{step}/{attempt}"


def _rank0_decision(plan: ChainPlan, runtime: ChainRuntime, arm: ChainArm, step: int, first: bool) -> Decision:
    state = runtime.state
    if state.get("stop-all"):
        return "stop"
    if step > 0 and not state.get(f"durable/{arm.name}/{step - 1}"):
        return "skip"  # The arm's previous update never became durable.
    estimate = max(plan.step_estimate_seconds, state.get("max-step-seconds") or 0)
    if runtime.remaining_seconds() < STEP_TIME_FACTOR * estimate + STEP_TIME_MARGIN_S:
        state["stop-reason"] = f"too little function time left for {arm.name} step {step + 1}"
        return "stop"
    if plan.gate == "manual" and not first:
        answer = runtime.gate(arm, step)
        if answer == "timeout":
            if plan.gate_timeout_action == "continue_if_healthy" and state.get("last-healthy"):
                return "go"
            state["stop-reason"] = f"no approval for {arm.name} step {step + 1}"
            return "stop"
        if answer == "stop":
            state["stop-reason"] = "operator stop"
        return answer
    return "go"


def _decide(plan: ChainPlan, runtime: ChainRuntime, arm: ChainArm, step: int, first: bool) -> Decision:
    state = runtime.state
    if runtime.rank == 0:
        decision = _rank0_decision(plan, runtime, arm, step, first)
        if decision == "stop":
            state["stop-all"] = True
        state[f"decision/{arm.name}/{step}"] = decision
    runtime.wait(lambda: runtime.state.get(f"decision/{arm.name}/{step}"), 24 * 3600, "chain step decision")
    result: Decision = state[f"decision/{arm.name}/{step}"]
    return result


def _conclude(
    plan: ChainPlan, runtime: ChainRuntime, arm: ChainArm, step: int, attempt: int, result: dict[str, Any]
) -> dict[str, Any]:
    """Rank 0, after the trainer exits: wait for this step's publication, then record the outcome."""
    state, key = runtime.state, _key(arm, step, attempt)
    state[f"exited/{key}"] = True
    runtime.wait(
        lambda: all(state.get(f"scanned/{key}/{r}") for r in range(plan.nodes)), 1800, "checkpoint publishers"
    )
    if state.get(f"save/{key}"):
        runtime.wait(
            lambda: state.get(f"durable/{arm.name}/{step}") or state.get(f"finalize-failed/{key}"),
            1800,
            "the committed chain step",
        )
    durable = state.get(f"durable/{arm.name}/{step}")
    result = {**result, "arm": arm.name, "step": step + 1, "attempt": attempt, "receipt": durable}
    if state.get(f"finalize-failed/{key}"):
        result["finalize_error"] = state.get(f"finalize-failed/{key}")
    runtime.persist(arm, step, attempt, result)
    state["max-step-seconds"] = max(state.get("max-step-seconds") or 0, result.get("wall_s") or 0)
    state["last-healthy"] = healthy(result, step)
    state["needs-ray-reset"] = result.get("exit_code") not in (0, None)  # None: no trainer was started.
    return result


def _attempt(plan: ChainPlan, runtime: ChainRuntime, arm: ChainArm, step: int, attempt: int) -> dict[str, Any] | None:
    state, rank, key = runtime.state, runtime.rank, _key(arm, step, attempt)
    if rank == 0:
        previous = state.get(f"durable/{arm.name}/{step - 1}") if step > 0 else None
        state[f"reset/{key}"] = bool(state.get("needs-ray-reset"))
        state[f"input/{key}"] = runtime.describe_receipt(previous).model_dump_json() if previous else "fresh"
    runtime.wait(lambda: state.get(f"input/{key}"), 1800, "chain step input")
    source = state[f"input/{key}"]
    previous_receipt = None if source == "fresh" else StateFile.model_validate_json(source)
    bundle, adapter = runtime.stage(arm, step, attempt, previous_receipt)
    state[f"staged/{key}/{rank}"] = True
    runtime.wait(
        lambda: all(state.get(f"staged/{key}/{r}") for r in range(plan.nodes)), 1800, "every node's chain inputs"
    )
    if state[f"reset/{key}"]:
        runtime.reset_ray(key)
    result = None
    if rank == 0:
        state["needs-ray-reset"] = False
        state[f"running/{key}"] = True
        outcome = runtime.run_step(arm, step, attempt, bundle, adapter)
        result = _conclude(plan, runtime, arm, step, attempt, outcome)
        if result.get("rejected"):
            verdict = {"durable": False, "final": True, "reason": f"rejected before training: {result['rejected']}"}
        elif result["receipt"] and step > 0 and result.get("resume_confirmed") is not True:
            # Committed, but without proof it continued its predecessor: never build on it.
            verdict = {"durable": True, "final": True, "reason": "no proof the step resumed its predecessor"}
        elif result.get("finalize_error"):
            verdict = {"durable": False, "final": True, "reason": f"could not certify: {result['finalize_error']}"}
        else:
            verdict = {"durable": bool(result["receipt"]), "final": False, "reason": None}
        state[f"finished/{key}"] = verdict
    runtime.wait(lambda: state.get(f"finished/{key}"), 24 * 3600, "chain step result")
    return result


def run_chain(plan: ChainPlan, runtime: ChainRuntime) -> list[dict[str, Any]]:
    """Run every arm's steps in plan order; rank 0 returns one result per attempt.

    A failed arm stops only itself. Failures are recorded under ``arm-failed/<arm>``.
    """
    state, results = runtime.state, []
    first = True
    for arm in plan.arms:
        for step in range(len(plan.batches)):
            decision = _decide(plan, runtime, arm, step, first)
            first = False
            if decision != "go":
                break
            verdict: dict[str, Any] = {}
            for attempt in range(plan.step_attempts):
                result = _attempt(plan, runtime, arm, step, attempt)
                if result is not None:
                    results.append(result)
                verdict = state[f"finished/{_key(arm, step, attempt)}"]
                if verdict["durable"] or verdict["final"]:
                    break
            if verdict["final"] or not verdict["durable"]:
                if runtime.rank == 0:
                    state[f"arm-failed/{arm.name}"] = verdict["reason"] or f"step {step + 1} never became durable"
                break
        if state.get("stop-all"):
            break
    return results


def publish_ready(plan: ChainPlan, publisher: ChainPublisher, done: set[str]) -> None:
    """One scan: commit this node's completed native saves; rank 0 finalizes complete steps.

    A key is dropped from later scans once it is fully handled on this node.
    """
    state, rank = publisher.state, publisher.rank
    for arm in plan.arms:
        for step in range(len(plan.batches)):
            for attempt in range(plan.step_attempts):
                key = _key(arm, step, attempt)
                if key in done or not state.get(f"running/{key}"):
                    continue
                local = publisher.local_adapter(arm, step, attempt)
                marker = local / "native_checkpoint.json"
                if marker.is_file() and state.get(f"save/{key}") is None:
                    state[f"save/{key}"] = NativeCompletion.model_validate_json(marker.read_bytes()).model_dump_json()
                saved = state.get(f"save/{key}")
                if saved and state.get(f"files/{key}/{rank}") is None:
                    state[f"files/{key}/{rank}"] = _FILES.dump_json(
                        publisher.publish_node(arm, step, attempt, local)
                    ).decode()
                finalized = not saved or rank != 0
                if rank == 0 and saved:
                    finalized = bool(state.get(f"durable/{arm.name}/{step}") or state.get(f"finalize-failed/{key}"))
                    receipts = [state.get(f"files/{key}/{r}") for r in range(plan.nodes)]
                    if not finalized and all(receipts):
                        try:
                            state[f"durable/{arm.name}/{step}"] = publisher.finalize(
                                arm,
                                step,
                                attempt,
                                NativeCompletion.model_validate_json(saved),
                                [_FILES.validate_json(r) for r in receipts],
                            )
                        except ValueError as exc:  # Invalid shards or export: this step's outcome, not the cluster's.
                            state[f"finalize-failed/{key}"] = str(exc)[:2000]
                        finalized = True
                # Acknowledge only after scanning the save that may follow subprocess exit.
                if state.get(f"exited/{key}"):
                    state[f"scanned/{key}/{rank}"] = True
                    # Keys without a save keep being scanned, so a save after exit is still published.
                    if saved and finalized and state.get(f"files/{key}/{rank}"):
                        done.add(key)


def hold_failed_cluster(
    state: SharedState,
    rank: int,
    *,
    hold_seconds: int,
    entered: float,
    limit_seconds: float,
    exc: BaseException,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Keep this node allocated for forensics, bounded by the function limit, until released or expired.

    The same rule as the sweep's hold: coordination failures never bypass the hold.
    """
    try:
        if not state.get("error"):
            state["error"] = f"chain node {rank}: {type(exc).__name__}: {exc}"
        state[f"holding/{rank}"] = str(exc)
    except Exception as coordination_error:
        print(f"[batch-chain] could not record hold: {coordination_error}", flush=True)
    deadline = min(entered + limit_seconds - 60, clock() + hold_seconds)
    print(f"[batch-chain] HOLDING node {rank}: {exc}; release through cluster coordination", flush=True)
    while clock() < deadline:
        try:
            if state.get("release"):
                break
        except Exception:
            # No release acknowledgement means continue the already-authorized bounded hold.
            pass
        sleep(5)
