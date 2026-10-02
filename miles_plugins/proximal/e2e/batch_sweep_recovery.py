"""Stage immutable sweep inputs and restore only certified native update boundaries."""

import hashlib
import json
from pathlib import Path

from miles_plugins.proximal.e2e.batch_sweep_artifacts import EXPORTS, SweepStep
from miles_plugins.proximal.e2e.batch_sweep_inputs import SweepPhase, SweepPlan
from miles_plugins.proximal.initial_policy import copy_base_policy, verify_base_policy
from miles_plugins.proximal.state_artifacts import copy_verified, describe, verify
from miles_plugins.proximal.state_checkpoints import NativeCompletion
from miles_plugins.proximal.storage import write_atomic


def stage_batch(source: Path, target: Path, *, batch_sha256: str) -> None:
    """No trainer process reads the reloadable Volume mount after this boundary."""
    from miles_plugins.proximal.offline_batch import read_batch

    manifest = (source / "batch.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != batch_sha256:
        raise ValueError("Staging source differs from the authorized frozen batch")
    batch = read_batch(source)
    for group in batch.groups:
        relative = f"groups/{group.header.group_id}.bin"
        file = describe(source / relative, relative=relative)
        if file.sha256 != group.payload_sha256:
            raise ValueError("Frozen group changed before local staging")
        copy_verified(source / relative, target / relative, file)
    copy_base_policy(verify_base_policy(batch.source, batch.policy, source / "base_policy"), target / "base_policy")
    # Preserve the exact manifest bytes: its digest is the experiment identity.
    write_atomic(target / "batch.json", manifest)


def read_resume(mount: Path, plan: SweepPlan, phase: SweepPhase) -> tuple[Path, SweepStep] | None:
    if phase.resume is None:
        return None
    path = mount / phase.resume.path
    verify(path, phase.resume)
    receipt = SweepStep.model_validate_json(path.read_bytes())
    if (
        receipt.phase != phase.name
        or receipt.update != 1
        or receipt.batch_sha256 != plan.batch_sha256
        or receipt.targets != phase.target_modules
        or receipt.native.iteration != 0
        or receipt.native.world_size != plan.nodes * 8
        or not (receipt.native.optimizer and receipt.native.scheduler and receipt.native.rng)
    ):
        raise ValueError("Resume receipt changes the phase, batch, targets, topology, or update boundary")
    # Legacy sweep plans had no resume field. Compare their original research
    # fields directly at the JSON boundary rather than silently defaulting a new plan.
    original = json.loads((path.parent.parent / "plan.json").read_bytes())
    for name in ("recipe", "samples", "nodes", "batch_sha256"):
        expected = list(plan.recipe) if name == "recipe" else getattr(plan, name)
        if original[name] != expected:
            raise ValueError(f"Native sweep resume changes {name}")
    adapter = path.parent / "checkpoints" / "iter_0000000" / "adapter"
    names = [file.path for file in receipt.files]
    required = {"adapter_config.json", *(name for name in EXPORTS if name in names)} | {
        f"{prefix}{rank}.pt"
        for rank in range(receipt.native.world_size)
        for prefix in ("adapter_megatron_rank", "training_state_rank")
    }
    if len(set(names)) != len(names) or set(names) != required or len(required & set(EXPORTS)) != 1:
        raise ValueError("Resume receipt must contain exactly every native, optimizer and serving file")
    for file in receipt.files:
        verify(adapter / file.path, file)
    native = NativeCompletion.model_validate_json((adapter / "native_checkpoint.json").read_bytes())
    if native != receipt.native:
        raise ValueError("Resume completion marker differs from its committed receipt")
    return adapter, receipt


def stage_resume(mount: Path, plan: SweepPlan, phase: SweepPhase, target: Path) -> Path | None:
    result = read_resume(mount, plan, phase)
    if result is None:
        return None
    adapter, receipt = result
    for file in receipt.files:
        copy_verified(adapter / file.path, target / file.path, file)
    write_atomic(target / "native_checkpoint.json", receipt.native.model_dump_json().encode())
    return target


def completed_training_steps(perf: list[dict[str, object]]) -> set[int]:
    """Rollout and training emit separate perf rows with the same rollout ID."""
    steps: set[int] = set()
    for row in perf:
        step = row.get("rollout")
        if "perf/actor_train_time" in row and isinstance(step, int):
            steps.add(step)
    return steps
