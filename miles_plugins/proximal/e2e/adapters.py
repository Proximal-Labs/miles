"""Random-weight PEFT LoRA adapters for a base model, built from its config.json only.

Stand-ins for trained versions: each seed gives distinct, small, nonzero weights so
two versions produce different outputs while the base model stays coherent. The
adapter shape comes from the run config's ``research.lora``, like a real export.

    python -m miles_plugins.proximal.e2e.adapters --config run.json \\
        --base-config /models/Qwen3-0.6B/config.json --output /tmp/adapters --count 3
"""

import argparse
import json
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from safetensors.torch import save_file

from miles_plugins.proximal.contracts import RunConfig, read_run_config

# The HF leaf modules Miles's Megatron->HF conversion yields for standard LoRA targets.
_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP = ("gate_proj", "up_proj", "down_proj")


class DenseDecoderShape(BaseModel):
    """The fields of a Qwen3-style dense decoder config that fix LoRA shapes."""

    model_config = ConfigDict(extra="ignore")
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int


def _module_shape(shape: DenseDecoderShape, module: str) -> tuple[int, int]:
    """(in_features, out_features) of one projection."""
    q = shape.num_attention_heads * shape.head_dim
    kv = shape.num_key_value_heads * shape.head_dim
    return {
        "q_proj": (shape.hidden_size, q),
        "k_proj": (shape.hidden_size, kv),
        "v_proj": (shape.hidden_size, kv),
        "o_proj": (q, shape.hidden_size),
        "gate_proj": (shape.hidden_size, shape.intermediate_size),
        "up_proj": (shape.hidden_size, shape.intermediate_size),
        "down_proj": (shape.intermediate_size, shape.hidden_size),
    }[module]


def hf_target_modules(run: RunConfig) -> list[str]:
    from miles.backends.megatron_utils.lora.utils import convert_target_modules_to_hf

    modules = convert_target_modules_to_hf(list(run.research.lora.target_modules))
    unknown = set(modules) - set(_ATTENTION + _MLP)
    if unknown:
        raise ValueError(f"Test adapters support dense attention/MLP targets only, not {sorted(unknown)}")
    return modules


def write_adapter(run: RunConfig, shape: DenseDecoderShape, directory: Path, *, seed: int) -> Path:
    lora = run.research.lora
    modules = hf_target_modules(run)
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(shape.num_hidden_layers):
        for module in modules:
            parent = "self_attn" if module in _ATTENTION else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{parent}.{module}"
            in_features, out_features = _module_shape(shape, module)
            tensors[f"{prefix}.lora_A.weight"] = (torch.randn(lora.rank, in_features, generator=generator) * 0.01).to(
                torch.bfloat16
            )
            # Nonzero B: distinct versions must serve distinguishable outputs.
            tensors[f"{prefix}.lora_B.weight"] = (
                torch.randn(out_features, lora.rank, generator=generator) * 0.001
            ).to(torch.bfloat16)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": lora.rank,
                "lora_alpha": lora.alpha,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": modules,
                "base_model_name_or_path": run.base_model.name,
            },
            sort_keys=True,
        )
    )
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Run config JSON")
    parser.add_argument("--base-config", type=Path, required=True, help="The base model's HF config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    run = read_run_config(args.config)
    shape = DenseDecoderShape.model_validate_json(args.base_config.read_bytes())
    for index in range(args.count):
        print(write_adapter(run, shape, args.output / f"adapter-{index}", seed=index))


if __name__ == "__main__":
    main()
