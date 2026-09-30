"""Stages of the LoRA serving parity check (``lora_parity.py``), run inside the training image.

``export`` (under torchrun, one rank per tensor-parallel GPU) builds the trainer's LoRA
model exactly as a run does, checks that Megatron-Bridge's merged export reproduces the
checkpoint while the adapter is still zero, gives every adapter random nonzero weights,
and writes them twice: as the adapter a run publishes, through the publisher's own
staging, layout check and writer, and merged into a full Hugging Face checkpoint, which
is the trained policy's ground truth. ``logprobs`` scores fixed token sequences with one
SGLang engine, configured like a serving replica's LoRA.

    torchrun --nproc-per-node 4 -m miles_plugins.proximal.e2e.lora_parity_stages export OUT -- <miles args>
    python -m miles_plugins.proximal.e2e.lora_parity_stages logprobs MODEL TOKENS OUT [--adapter NAME=PATH ...]
"""

import argparse
import json
import os
import re
import shutil
import sys
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# Only the adapter's own tensors differ between the checkpoint and the merged export.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def export(out: Path, *, seed: int, strength: float) -> None:
    # megatron and sglang exist only in the training image.
    from megatron.bridge import AutoBridge  # type: ignore[import-not-found,unused-ignore]
    from megatron.core import mpu  # type: ignore[import-not-found,unused-ignore]

    from miles.backends.megatron_utils.checkpoint import _load_checkpoint_hf
    from miles.backends.megatron_utils.initialize import init
    from miles.backends.megatron_utils.lora.bridge import _setup_lora_model_via_bridge
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator
    from miles.utils import megatron_bridge_utils
    from miles.utils.arguments import parse_args
    from miles.utils.distributed_utils import init_gloo_group
    from miles.utils.hf_utils.config import load_hf_config
    from miles.utils.lora.hf_lora_targets import parse_lora_targets
    from miles.utils.lora.utils import LORA_ADAPTER_NAME, build_lora_config
    from miles_plugins.proximal.adapter_layout import adapter_layout_problem
    from miles_plugins.proximal.lora_targets import convert_target_modules_to_hf
    from miles_plugins.proximal.weight_update import (
        ModalVolumeTransfer,
        peft_config_json,
        staged_adapter_tensor,
        write_adapter,
    )

    args = parse_args()  # type: ignore[no-untyped-call]  # Miles's untyped CLI.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    init_gloo_group()  # type: ignore[no-untyped-call]
    args.rank, args.world_size = dist.get_rank(), dist.get_world_size()
    init(args)
    checkpoint = Path(args.hf_checkpoint)
    # The trainer's model: the same builder and HF load as a run's actor.
    model = _setup_lora_model_via_bridge(args)
    _load_checkpoint_hf(model, None, args, str(checkpoint))
    bridge = AutoBridge.from_hf_pretrained(str(checkpoint), trust_remote_code=True)
    report: dict[str, Any] = {"lora_modules": _lora_module_names(model)}

    with megatron_bridge_utils.patch_megatron_model(model):
        report["round_trip"] = _round_trip(
            bridge.export_hf_weights(model, cpu=True, merge_adapter_weights=True), checkpoint
        )
    _randomize_adapters(model, seed=seed, strength=strength, tp_group=mpu.get_tensor_model_parallel_group())

    iterator = get_hf_weight_iterator(
        args,
        model,
        required_placement=ModalVolumeTransfer.required_placement,
        model_name=type(load_hf_config(str(checkpoint))).__name__.lower(),
        quantization_config=None,
    )
    exported = iterator.materialize_adapter(None)
    if dist.get_rank() == 0:
        tensors = dict(staged_adapter_tensor(f"{LORA_ADAPTER_NAME}:{name}", t) for name, t in exported.items())
        targets = convert_target_modules_to_hf(parse_lora_targets(args.target_modules) or [])
        sync_config = build_lora_config(args, target_modules=targets)  # type: ignore[no-untyped-call]
        report["adapter_tensors"] = len(tensors)
        report["layout_problem"] = adapter_layout_problem(
            {name: tuple(t.shape) for name, t in tensors.items()}, serving_targets=targets, rank=args.lora_rank
        )
        report["serving_targets"] = targets
        adapter = out / "adapter"
        adapter.mkdir(parents=True)
        config = peft_config_json(sync_config, rank=args.lora_rank, base_model_name=checkpoint.name)
        write_adapter(adapter, tensors=tensors, config_json=config)

    merged = out / "merged"
    with megatron_bridge_utils.patch_megatron_model(model):
        bridge.save_hf_pretrained(model, str(merged), merge_adapter_weights=True)
    dist.barrier()
    if dist.get_rank() == 0:
        _copy_non_weight_files(checkpoint, merged)
        (out / "export.json").write_text(json.dumps(report, indent=2, default=str))
    dist.destroy_process_group()


def _lora_module_names(model: list[torch.nn.Module]) -> list[str]:
    from megatron.bridge.peft.lora_layers import LoRALinear  # type: ignore[import-not-found,unused-ignore]

    names = (name for chunk in model for name, module in chunk.named_modules() if isinstance(module, LoRALinear))
    return sorted({re.sub(r"\.\d+\.", ".N.", name) for name in names})


def _round_trip(exported: Iterator[Any], checkpoint: Path) -> dict[str, Any]:
    """Compare the merged export, taken while every lora_B is still zero, with the checkpoint."""
    from safetensors import safe_open

    files = {}
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(str(path), framework="pt") as handle:
            files.update({key: path for key in handle.keys()})
    compared, differing, missing, largest = 0, [], [], 0.0
    for item in exported:
        name, tensor = item[0], item[1]
        if dist.get_rank() != 0:
            continue
        if name not in files:
            missing.append(name)
            continue
        with safe_open(str(files[name]), framework="pt") as handle:
            reference = handle.get_tensor(name)
        difference = (
            (tensor.float() - reference.float()).abs().max().item()
            if tensor.shape == reference.shape
            else float("inf")
        )
        compared += 1
        largest = max(largest, difference)
        if difference != 0.0:
            differing.append(name)
    return {
        "compared": compared,
        "differing": differing[:20],
        "n_differing": len(differing),
        "max_abs_difference": largest,
        "not_in_checkpoint": missing[:20],
    }


def _randomize_adapters(model: list[torch.nn.Module], *, seed: int, strength: float, tp_group: Any) -> None:
    """Give every adapter random weights whose update B·A·(alpha/r) is ``strength`` times the base weight's RMS."""
    from megatron.bridge.peft.lora_layers import LoRALinear  # type: ignore[import-not-found,unused-ignore]

    tp_rank = dist.get_rank(group=tp_group)
    for chunk in model:
        for name, module in chunk.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            adapter = module.adapter
            base_ms = module.to_wrap.weight.detach().float().pow(2).mean()
            dist.all_reduce(base_ms, group=tp_group)
            base_rms = (base_ms / dist.get_world_size(group=tp_group)).sqrt().item()
            a, b = adapter.linear_in.weight, adapter.linear_out.weight
            input_sharded = getattr(a, "tensor_model_parallel", False) and getattr(a, "partition_dim", 0) == 1
            fan_in = a.shape[1] * (dist.get_world_size(group=tp_group) if input_sharded else 1)
            # std(delta W) = (alpha / r) * sqrt(r) * std(B) / sqrt(fan_in) with A ~ N(0, 1 / fan_in).
            b_std = strength * base_rms * (adapter.dim / adapter.alpha) * (fan_in / adapter.dim) ** 0.5
            with torch.no_grad():
                for param, std in ((a, fan_in**-0.5), (b, b_std)):
                    sharded = getattr(param, "tensor_model_parallel", False)
                    generator = torch.Generator(device=param.device).manual_seed(
                        seed + zlib.crc32(f"{name}:{param.shape}".encode()) + (7919 * tp_rank if sharded else 0)
                    )
                    param.copy_(
                        torch.randn(param.shape, generator=generator, device=param.device, dtype=torch.float32) * std
                    )


def _copy_non_weight_files(source: Path, merged: Path) -> None:
    """Tokenizer, chat template and processor files, so SGLang can serve the merged checkpoint."""
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith(_WEIGHT_SUFFIXES) and not path.name.endswith(".index.json"):
            if not (merged / path.name).exists():
                shutil.copy2(path, merged / path.name)


def logprobs(model: Path, tokens: Path, out: Path, *, adapters: dict[str, str], rank: int, targets: list[str]) -> None:
    """Per-token logprobs of each sequence in ``tokens``: base, then under each adapter."""
    import sglang  # type: ignore[import-not-found,unused-ignore]

    engine_args: dict[str, Any] = {
        "model_path": str(model),
        "tp_size": 1,
        "dtype": "bfloat16",
        "attention_backend": "trtllm_mha",  # As the Qwen3.8 replicas.
        "page_size": 64,
        "disable_cuda_graph": True,  # Prefill only.
        "mem_fraction_static": 0.6,
    }
    if adapters:
        engine_args |= {
            "enable_lora": True,
            "max_lora_rank": rank,
            "lora_target_modules": targets,
            "lora_strict_loading": True,
            "max_loras_per_batch": 1,
            "max_loaded_loras": len(adapters),
        }
    sequences = json.loads(tokens.read_text())
    engine = sglang.Engine(**engine_args)
    try:
        for name, path in adapters.items():
            engine.load_lora_adapter(name, path)
        results = {
            label: _score(engine, sequences, None if label == "base" else label) for label in ["base", *adapters]
        }
    finally:
        engine.shutdown()
    out.write_text(json.dumps(results))


def _score(engine: Any, sequences: list[list[int]], adapter: str | None) -> list[list[float]]:
    scored = []
    for ids in sequences:
        output = engine.generate(
            input_ids=ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=0,
            lora_path=adapter,
        )
        scored.append([float(entry[0]) for entry in output["meta_info"]["input_token_logprobs"][1:]])
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("out", type=Path)
    exporter.add_argument("--seed", type=int, default=0)
    exporter.add_argument("--strength", type=float, required=True)
    scorer = commands.add_parser("logprobs")
    scorer.add_argument("model", type=Path)
    scorer.add_argument("tokens", type=Path)
    scorer.add_argument("out", type=Path)
    scorer.add_argument("--adapter", action="append", default=[], help="NAME=PATH")
    scorer.add_argument("--rank", type=int, default=0)
    scorer.add_argument("--targets", default="")
    own, miles_args = sys.argv[1:], []
    if "--" in own:  # Everything after -- is Miles's own command line.
        split = own.index("--")
        own, miles_args = own[:split], own[split + 1 :]
    options = parser.parse_args(own)
    if options.command == "export":
        sys.argv = [sys.argv[0], *miles_args]  # Miles's parse_args reads sys.argv.
        export(options.out, seed=options.seed, strength=options.strength)
    else:
        adapters = dict(item.split("=", 1) for item in options.adapter)
        logprobs(
            options.model,
            options.tokens,
            options.out,
            adapters=adapters,
            rank=options.rank,
            targets=[t for t in options.targets.split(",") if t],
        )


if __name__ == "__main__":
    main()
