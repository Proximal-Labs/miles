"""Numerical evidence for the state verification experiment; never silently tolerates drift."""

import json
import shutil
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from pydantic import BaseModel, TypeAdapter


class Comparison(TypedDict):
    equal: bool
    tensors: int
    elements: int
    max_abs: float
    mismatches: list[str]


class EngineMetadata(BaseModel):
    input_token_logprobs: list[tuple[float | None, int, str | None]]
    output_token_logprobs: list[tuple[float, int, str | None]]


class EngineReply(BaseModel):
    meta_info: EngineMetadata


def compare_values(left: object, right: object) -> Comparison:
    result: Comparison = {"equal": True, "tensors": 0, "elements": 0, "max_abs": 0.0, "mismatches": []}

    def visit(a: object, b: object, path: str) -> None:
        same = True
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            result["tensors"] += 1
            result["elements"] += a.numel()
            same = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
            if a.shape == b.shape and a.numel() and a.dtype != torch.bool:
                delta = (a.double() - b.double()).abs().max().item()
                result["max_abs"] = max(result["max_abs"], delta)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            same = np.array_equal(a, b)
        elif isinstance(a, dict) and isinstance(b, dict):
            same = a.keys() == b.keys()
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (tuple, list)) and isinstance(b, type(a)):
            same = len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                visit(x, y, f"{path}/{i}")
        else:
            same = type(a) is type(b) and a == b
        if not same:
            result["equal"] = False
            if len(result["mismatches"]) < 30:
                result["mismatches"].append(path)

    visit(left, right, "root")
    return result


def compare_checkpoints(root: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    reference = root / "reference/checkpoints/iter_0000001/adapter"
    resumed = root / "resumed/checkpoints/iter_0000001/adapter"
    for name in ("adapter_megatron_rank0.pt", "training_state_rank0.pt", "adapter_model.bin"):
        result[name] = compare_values(
            torch.load(reference / name, map_location="cpu", weights_only=False),
            torch.load(resumed / name, map_location="cpu", weights_only=False),
        )
    result["adapter_config"] = compare_values(
        json.loads((reference / "adapter_config.json").read_text()),
        json.loads((resumed / "adapter_config.json").read_text()),
    )
    result["train_data"] = compare_values(
        torch.load(root / "reference/train-data/1_0.pt", map_location="cpu", weights_only=False),
        torch.load(root / "resumed/train-data/1_0.pt", map_location="cpu", weights_only=False),
    )
    initial = torch.load(
        root / "reference/checkpoints/iter_0000000/adapter/adapter_megatron_rank0.pt", weights_only=True
    )
    final = torch.load(reference / "adapter_megatron_rank0.pt", weights_only=True)
    change = compare_values(initial, final)
    result["nonzero_update"] = not change["equal"] and change["max_abs"] > 0
    result["update_max_abs"] = change["max_abs"]
    result["native_keys"] = list(final)[:12]
    peft = torch.load(reference / "adapter_model.bin", weights_only=True)
    result["peft_keys"] = list(peft)[:12]
    result["native_to_peft"] = compare_qwen06_export(final, peft)
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    return result


def compare_qwen06_export(native: dict[str, torch.Tensor], peft: dict[str, torch.Tensor]) -> Comparison:
    """Independent algebraic witness for this recipe, without calling Bridge's exporter.

    Qwen3-0.6B has 28 layers, 16 Q heads, 8 KV groups, head dim 128 and FFN dim
    3072. Megatron interleaves Q,Q,K,V within each group and fuses gate/up.
    This is deliberately a model-specific test, not a second production exporter.
    """
    expected: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def weights(layer: int, module: str) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"module.module.decoder.layers.{layer}.{module}.adapter"
        a, b = f"{prefix}.linear_in.weight", f"{prefix}.linear_out.weight"
        consumed.update((a, b))
        return native[a], native[b]

    def put(layer: int, module: str, a: torch.Tensor, b: torch.Tensor) -> None:
        prefix = f"model.layers.{layer}.{module}"
        expected[f"{prefix}.lora_A.weight"] = a
        expected[f"{prefix}.lora_B.weight"] = b

    for layer in range(28):
        put(layer, "self_attn.o_proj", *weights(layer, "self_attention.linear_proj"))
        put(layer, "mlp.down_proj", *weights(layer, "mlp.linear_fc2"))
        a, b = weights(layer, "mlp.linear_fc1")
        if b.shape[0] != 6144:
            raise ValueError("Unexpected Qwen3-0.6B fused gate/up dimension")
        put(layer, "mlp.gate_proj", a, b[:3072])
        put(layer, "mlp.up_proj", a, b[3072:])
        a, b = weights(layer, "self_attention.linear_qkv")
        grouped = b.reshape(8, 4, 128, a.shape[0])
        put(layer, "self_attn.q_proj", a, grouped[:, :2].reshape(2048, a.shape[0]))
        put(layer, "self_attn.k_proj", a, grouped[:, 2].reshape(1024, a.shape[0]))
        put(layer, "self_attn.v_proj", a, grouped[:, 3].reshape(1024, a.shape[0]))
    if consumed != native.keys():
        raise ValueError("Verification recipe does not cover every native adapter tensor")
    return compare_values(expected, peft)


def serving_check(root: Path) -> dict[str, object]:
    import sglang  # type: ignore[import-not-found]  # Available in the pinned GPU image.
    from safetensors.torch import save_file

    from miles_plugins.proximal.offline_batch import load_training_group, read_batch, training_groups
    from miles_plugins.proximal.snapshot import SnapshotMetadata, prepare_snapshot

    batch = read_batch(root / "batch")
    names = {}
    for role, branch, step in (
        ("reference", "reference", 1),
        ("resumed", "resumed", 1),
        ("before-step", "reference", 0),
    ):
        native = root / branch / f"checkpoints/iter_{step:07d}/adapter"
        export = Path("/work/serve") / role
        export.mkdir(parents=True)
        shutil.copy2(native / "adapter_config.json", export / "adapter_config.json")
        weights = torch.load(native / "adapter_model.bin", map_location="cpu", weights_only=True)
        save_file(
            {k: v.contiguous().clone() for k, v in sorted(weights.items())}, str(export / "adapter_model.safetensors")
        )
        snapshot = prepare_snapshot(
            export,
            metadata=SnapshotMetadata(
                run_id=batch.source.run_id, checkpoint_iteration=step, base_model=batch.source.base_model
            ),
            output_root=root / "serving" / role,
        )
        names[role] = snapshot
    samples = [s for group in training_groups(batch)[:2] for s in load_training_group(root / "batch", batch, group)]
    prompts = [list(s.tokens) for s in samples]
    engine = sglang.Engine(
        model_path=str(batch.source.tokenizer_path),
        dtype="bfloat16",
        tp_size=1,
        enable_lora=True,
        max_lora_rank=batch.source.research.lora.rank,
        lora_target_modules=TypeAdapter(list[str]).validate_python(
            json.loads((names["reference"].directory / "adapter_config.json").read_text())["target_modules"]
        ),
        max_loaded_loras=4,
        max_loras_per_batch=4,
        mem_fraction_static=0.5,
        disable_cuda_graph=True,
        log_level="warning",
    )
    replies: dict[str, list[EngineReply]] = {}
    try:
        for role, snapshot in names.items():
            loaded = engine.load_lora_adapter(role, str(snapshot.directory))
            print(f"[state-check] load {role}: {loaded}", flush=True)
        for adapter in (None, "reference", "resumed", "before-step"):
            replies[adapter or "base"] = TypeAdapter(list[EngineReply]).validate_python(
                engine.generate(
                    input_ids=prompts,
                    sampling_params={"temperature": 0, "max_new_tokens": 1},
                    return_logprob=True,
                    logprob_start_len=0,
                    lora_path=[adapter] * len(prompts) if adapter else None,
                )
            )
    finally:
        engine.shutdown()
    probabilities = {}
    output_ids = {}
    for role, results in replies.items():
        probabilities[role] = torch.tensor(
            [token[0] for reply in results for token in reply.meta_info.input_token_logprobs if token[0] is not None]
        )
        output_ids[role] = [[token[1] for token in reply.meta_info.output_token_logprobs] for reply in results]
        if not all(output_ids[role]):
            raise ValueError("Serving verification must observe actual generated token IDs")
    result = {
        "snapshots": {role: snapshot.reference.sha256 for role, snapshot in names.items()},
        "reference_vs_resumed": compare_values(probabilities["reference"], probabilities["resumed"]),
        "base_vs_adapter": compare_values(probabilities["base"], probabilities["resumed"]),
        "greedy_ids_equal": output_ids["reference"] == output_ids["resumed"],
        "prompts": len(prompts),
    }
    result["before_vs_after_step"] = compare_values(probabilities["before-step"], probabilities["reference"])
    (root / "serving.json").write_text(json.dumps(result, indent=2))
    return result
