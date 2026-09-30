"""CPU bootstrap of a serving-only, zero-delta LoRA for fresh-base collection.

Uses actual HF checkpoint tensor shapes and Miles's target-name mapping. It does
not load base tensors, initialize an optimizer, or produce a native checkpoint.
The trainable adapter is initialized later by Megatron, with its normal nonzero A.
"""

from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import Field
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from miles_plugins.proximal.contracts import Contract, Policy, RunConfig
from miles_plugins.proximal.snapshot import (
    PreparedSnapshot,
    SnapshotMetadata,
    manifest_bytes,
    prepare_snapshot,
    read_snapshot,
)
from miles_plugins.proximal.state_artifacts import StateFile, copy_verified


class BasePolicyConfig(Contract):
    peft_type: Literal["LORA"]
    task_type: Literal["CAUSAL_LM"]
    r: int
    lora_alpha: int
    lora_dropout: Annotated[float, Field(ge=0, le=0)]
    bias: Literal["none"]
    target_modules: tuple[str, ...]
    base_model_name_or_path: str


def _targets(run: RunConfig) -> tuple[str, ...]:
    from miles_plugins.proximal.lora_targets import convert_target_modules_to_hf

    # A path/wildcard scopes training to particular layers; a leaf-only bootstrap
    # cannot certify that contract. Fail before publishing or launching rollouts.
    if any("." in name or "*" in name for name in run.research.lora.target_modules):
        raise ValueError("Fresh CPU bootstrap requires explicit unscoped LoRA targets")
    return tuple(convert_target_modules_to_hf(list(run.research.lora.target_modules)))


def prepare_base_policy(run: RunConfig, *, output: Path) -> PreparedSnapshot:
    """Read only checkpoint headers; allocate only the zero serving adapter."""
    targets = _targets(run)
    shards = sorted(run.tokenizer_path.glob("*.safetensors"))
    if not shards:
        raise ValueError("Fresh collection needs the pinned base's safetensors checkpoint, not only a tokenizer")
    tensors: dict[str, torch.Tensor] = {}
    found: set[str] = set()
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                parts = key.rsplit(".", 2)
                if len(parts) != 3 or parts[-1] != "weight" or parts[-2] not in targets:
                    continue
                view = reader.get_slice(key)
                shape = view.get_shape()
                if len(shape) != 2 or view.get_dtype() not in {"BF16", "F16", "F32"}:
                    raise ValueError(f"Fresh bootstrap needs an unquantized linear weight: {key}")
                prefix = f"base_model.model.{key.removesuffix('.weight')}"
                a, b = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
                if a in tensors:
                    raise ValueError(f"Repeated base weight {key}")
                tensors[a] = torch.zeros((run.research.lora.rank, shape[1]), dtype=torch.bfloat16)
                tensors[b] = torch.zeros((shape[0], run.research.lora.rank), dtype=torch.bfloat16)
                found.add(parts[-2])
    if found != set(targets):
        raise ValueError(f"Base checkpoint is missing LoRA targets: {sorted(set(targets) - found)}")
    config = BasePolicyConfig(
        peft_type="LORA",
        task_type="CAUSAL_LM",
        r=run.research.lora.rank,
        lora_alpha=run.research.lora.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        base_model_name_or_path=run.base_model.name,
    )
    adapter = output / "zero-adapter"
    adapter.mkdir(parents=True, exist_ok=False)
    (adapter / "adapter_config.json").write_text(config.model_dump_json())
    save_file(tensors, str(adapter / "adapter_model.safetensors"))
    return prepare_snapshot(
        adapter,
        metadata=SnapshotMetadata(run_id=run.run_id, checkpoint_iteration=0, base_model=run.base_model),
        output_root=output,
    )


def verify_base_policy(run: RunConfig, policy: Policy, directory: Path) -> PreparedSnapshot:
    """A version number or a filename alone never proves that the policy is base."""
    snapshot = read_snapshot(directory, policy.snapshot)
    if (policy.run_id, policy.base_model, policy.version) != (run.run_id, run.base_model, 1):
        raise ValueError("Fresh training requires this run's initial base policy")
    if snapshot.manifest.metadata != SnapshotMetadata(
        run_id=run.run_id, checkpoint_iteration=0, base_model=run.base_model
    ):
        raise ValueError("Fresh policy snapshot belongs to another run/base")
    config = BasePolicyConfig.model_validate_json((directory / "adapter_config.json").read_bytes())
    if (config.r, config.lora_alpha, config.target_modules, config.base_model_name_or_path) != (
        run.research.lora.rank,
        run.research.lora.alpha,
        _targets(run),
        run.base_model.name,
    ):
        raise ValueError("Fresh serving adapter differs from the declared LoRA/base contract")
    tensors = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    a_names = {name for name in tensors if name.endswith(".lora_A.weight")}
    b_names = {name for name in tensors if name.endswith(".lora_B.weight")}
    if not a_names or len(a_names) + len(b_names) != len(tensors):
        raise ValueError("Fresh serving policy has unexpected or missing adapter tensors")
    if {name.replace(".lora_A.weight", ".lora_B.weight") for name in a_names} != b_names:
        raise ValueError("Fresh serving policy is missing adapter pairs")
    for name in a_names:
        a, b = tensors[name], tensors[name.replace(".lora_A.weight", ".lora_B.weight")]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != config.r or b.shape[1] != config.r:
            raise ValueError("Fresh serving adapter has invalid rank/shape")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or torch.count_nonzero(b):
            raise ValueError("Fresh training requires a zero-delta serving policy")
    return snapshot


def copy_base_policy(snapshot: PreparedSnapshot, destination: Path) -> None:
    """Keep the proof with detached data; caller commits payloads before readiness."""
    for file in snapshot.manifest.files:
        source = snapshot.directory / file.name
        copy_verified(
            source,
            destination / file.name,
            StateFile(path=file.name, size_bytes=file.size_bytes, sha256=file.sha256),
        )
    source = snapshot.directory / "manifest.json"
    copy_verified(
        source,
        destination / "manifest.json",
        StateFile(
            path="manifest.json",
            size_bytes=len(manifest_bytes(snapshot.manifest)),
            sha256=snapshot.reference.sha256,
        ),
    )
