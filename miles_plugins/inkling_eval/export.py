"""Collective Inkling adapter export, using the existing inference weight mapping."""

import hashlib
import json
from pathlib import Path

import safetensors.torch
import torch.distributed as dist

from miles_plugins.inkling_eval.config import write_json


def export_adapter(args, model, destination):
    # Collective GPU dependencies are unnecessary for CPU snapshot verification.
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator import _gather_pp_full_adapter
    from miles_plugins.models.inkling.lora import export_inkling_lora_hf_named

    if "inkling" not in (args.custom_model_provider_path or "") or args.lora_rank <= 0:
        raise ValueError("Inkling evaluation export requires an Inkling LoRA model")
    named = _gather_pp_full_adapter(export_inkling_lora_hf_named(model))
    if not named or len(dict(named)) != len(named):
        raise ValueError("Adapter export is empty or contains duplicate names")
    error = [None]
    if dist.get_rank() == 0:
        try:
            _write_adapter(args, named, destination)
        except Exception as exc:
            error[0] = str(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"Inkling adapter export failed: {error[0]}")


def _write_adapter(args, named, destination):
    tensors = dict(named)
    for name, tensor in named:
        if ".lora_A.weight" in name:
            counterpart = name.replace(".lora_A.weight", ".lora_B.weight")
            rank_axis = -2
        elif ".lora_B.weight" in name:
            counterpart = name.replace(".lora_B.weight", ".lora_A.weight")
            rank_axis = -1
        else:
            raise ValueError(f"Unexpected adapter tensor: {name}")
        if counterpart not in tensors or tensor.ndim not in {2, 3} or tensor.shape[rank_axis] != args.lora_rank:
            raise ValueError(f"Incomplete or incompatible LoRA pair: {name}")
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".complete").unlink(missing_ok=True)
    tensors = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in named}
    safetensors.torch.save_file(tensors, path / "adapter_model.safetensors")
    write_json(
        path / "adapter_config.json",
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "inference_mode": True,
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": 0,
            "bias": "none",
            "target_modules": "all-linear",
            "base_model_name_or_path": args.hf_checkpoint,
        },
    )
    hashes = {}
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        with (path / filename).open("rb") as stream:
            hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    (path / ".complete").write_text(json.dumps(hashes))
