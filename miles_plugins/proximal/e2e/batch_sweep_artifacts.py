"""Commit native shards and evaluation adapters at each completed sweep update."""

import json
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import Field
from safetensors.torch import load_file

from miles_plugins.proximal.adapter_layout import adapter_layout_problem
from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization
from miles_plugins.proximal.contracts import Contract, SafeId
from miles_plugins.proximal.e2e.batch_sweep_inputs import SweepPhase, SweepPlan
from miles_plugins.proximal.lora_targets import convert_target_modules_to_hf
from miles_plugins.proximal.snapshot import (
    Digest,
    Nonempty,
    SnapshotMetadata,
    SnapshotReference,
    prepare_snapshot,
    read_snapshot,
)
from miles_plugins.proximal.state_artifacts import RelativePath, StateFile, copy_verified, describe, verify
from miles_plugins.proximal.state_checkpoints import NativeCompletion
from miles_plugins.proximal.storage import write_atomic
from miles_plugins.proximal.weight_update import peft_config_json, write_adapter

# The trainer exports safetensors; sweeps before the upstream merge kept adapter_model.bin.
EXPORTS = ("adapter_model.safetensors", "adapter_model.bin")


class SweepStep(Contract):
    phase: SafeId
    update: Literal[1, 2]
    batch_sha256: Digest
    targets: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    native: NativeCompletion
    files: tuple[StateFile, ...]
    snapshot: SnapshotReference
    eval_path: RelativePath


def export_file(names: Iterable[str]) -> str:
    """The one PEFT weight file among a checkpoint's file names."""
    available = set(names)
    found = [name for name in EXPORTS if name in available]
    if len(found) != 1:
        raise ValueError("Missing native/evaluation checkpoint files: need exactly one adapter_model export")
    return found[0]


def publish_node_files(
    authorization: AuthorizedRun, *, local: Path, destination: Path, commit: Callable[[], None]
) -> tuple[StateFile, ...]:
    """Called only after the all-rank native-save barrier; filenames contain global ranks."""
    require_authorization(authorization)
    files = tuple(
        describe(path, relative=path.name)
        for path in sorted(local.iterdir())
        if path.is_file() and path.name != "native_checkpoint.json"
    )
    if not any(f.path.startswith("adapter_megatron_rank") for f in files):
        raise ValueError("A sweep node has no native adapter shards to preserve")
    for file in files:
        copy_verified(local / file.path, destination / file.path, file)
    commit()
    return files


def finalize_step(
    authorization: AuthorizedRun,
    *,
    plan: SweepPlan,
    phase: SweepPhase,
    step: int,
    native: NativeCompletion,
    receipts: Sequence[Sequence[StateFile]],
    root: Path,
    commit: Callable[[], None],
) -> dict[str, object]:
    """All node commits precede the globally complete marker and eval snapshot manifest."""
    source = require_authorization(authorization)
    if native.iteration != step or native.world_size != 8 * plan.nodes:
        raise ValueError("Native completion does not describe this sweep update")
    if not (native.optimizer and native.scheduler and native.rng) or len(receipts) != plan.nodes:
        raise ValueError("Every node and full training state are required")
    files: dict[str, StateFile] = {}
    for receipt in receipts:
        for file in receipt:
            if file.path in files and files[file.path] != file:
                raise ValueError(f"Conflicting checkpoint writers for {file.path}")
            files[file.path] = file
    required = {"adapter_config.json", export_file(files)}
    required.update(
        f"{prefix}{rank}.pt"
        for rank in range(native.world_size)
        for prefix in ("adapter_megatron_rank", "training_state_rank")
    )
    if not required <= files.keys():
        raise ValueError(f"Missing native/evaluation checkpoint files: {sorted(required - files.keys())}")
    adapter = root / phase.name / "checkpoints" / f"iter_{step:07d}" / "adapter"
    for file in files.values():
        verify(adapter / file.path, file)
    config = json.loads((adapter / "adapter_config.json").read_text())
    targets = list(convert_target_modules_to_hf(list(phase.target_modules)))
    lora = source.research.lora
    # The trainer lists every adapted module path; the tensor layout check below ties them to the phase's targets.
    if (config.get("r"), config.get("lora_alpha")) != (lora.rank, lora.alpha):
        raise ValueError("Serving export differs from the phase's LoRA configuration")
    export = adapter / export_file(files)
    tensors = (
        load_file(str(export), device="cpu")
        if export.suffix == ".safetensors"
        else torch.load(export, map_location="cpu", weights_only=True)
    )
    if (
        not isinstance(tensors, dict)
        or not tensors
        or not all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all()
            for name, tensor in tensors.items()
        )
    ):
        raise ValueError("Invalid or nonfinite serving adapter")
    problem = adapter_layout_problem(
        {name: tuple(tensor.shape) for name, tensor in tensors.items()}, serving_targets=targets, rank=lora.rank
    )
    if problem:
        raise ValueError(problem)
    write_atomic(adapter / "native_checkpoint.json", native.model_dump_json().encode())
    # Replicas serve the published form: the phase's serving targets and the pinned base model's name.
    with tempfile.TemporaryDirectory(prefix="sweep-eval-") as staged:
        write_adapter(
            Path(staged),
            tensors=tensors,
            config_json=peft_config_json(
                config | {"target_modules": targets}, rank=lora.rank, base_model_name=source.base_model.name
            ),
        )
        snapshot = prepare_snapshot(
            Path(staged),
            metadata=SnapshotMetadata(
                run_id=f"{plan.experiment_id}-{phase.name}", checkpoint_iteration=step, base_model=source.base_model
            ),
            output_root=root / phase.name / "eval",
        )
    result: dict[str, object] = {
        "phase": phase.name,
        "update": step + 1,
        "batch_sha256": plan.batch_sha256,
        "targets": phase.target_modules,
        "native": native.model_dump(mode="json"),
        "files": [file.model_dump(mode="json") for file in sorted(files.values(), key=lambda f: f.path)],
        "snapshot": snapshot.reference.model_dump(mode="json"),
        "eval_path": str(snapshot.directory.relative_to(root)),
    }
    # All payloads and snapshot bytes commit before the completion receipt.
    commit()
    read_snapshot(snapshot.directory, snapshot.reference)
    receipt_path = root / phase.name / f"step-{step + 1}.json"
    write_atomic(receipt_path, json.dumps(result, sort_keys=True).encode())
    commit()
    return result
