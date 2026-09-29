"""LoRA serving parity: does SGLang serve the adapter a run publishes as the trainer trained it?

One node, one tensor-parallel group like a run's trainer (``lora_parity_stages.py``):

1. Build the trainer's LoRA model for ``--targets`` and load the checkpoint. While every
   lora_B is still zero, Megatron-Bridge's merged export must reproduce the checkpoint
   exactly (round trip).
2. Give every adapter random nonzero weights whose update is ``--strength`` times the base
   weight's RMS. Write the adapter a run would publish (the publisher's staging, layout
   check and writer) and the same adapter merged into a full checkpoint in Megatron's own
   layout: the trained policy.
3. Score fixed token sequences with SGLang: the base model with the adapter loaded as a
   replica loads it, the base model alone, and the merged checkpoint alone.

The adapter must reproduce the merged checkpoint's logprobs far more closely than the
base model does. The merged checkpoint is stored in bf16, so its rounding leaves a small
residual; two deliberately broken adapters (GDN q and k slices swapped, attention q and
gate halves swapped) show how large a mapping error is, and must be caught.

    modal run --env main -m miles_plugins.proximal.e2e.lora_parity --model Qwen/Qwen3.5-4B
    modal run --env main -m miles_plugins.proximal.e2e.lora_parity --model qwen38
"""

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import modal

from miles_plugins.proximal.e2e.argv import set_flag
from miles_plugins.proximal.modal_sources import add_fork_sources

REPO = Path(__file__).resolve().parents[3]
FORK = Path("/fork")
WORK = Path("/tmp/parity")
TP = 4  # As a run's trainer.
STAGES = "miles_plugins.proximal.e2e.lora_parity_stages"
QWEN38 = "/models/Qwen3.8-27B-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# Every linear layer of a Qwen3.5-architecture decoder, anchored so the vision tower and MTP stay unadapted.
ALL_LINEAR = ",".join(
    f"language_model.decoder.layers.*.{module}"
    for module in (
        "self_attention.linear_qkv",
        "self_attention.linear_proj",
        "self_attention.in_proj",
        "self_attention.out_proj",
        "mlp.linear_fc1",
        "mlp.linear_fc2",
    )
)
# The training image, as the Qwen3.8 deployments pin it.
IMAGE = json.loads((REPO / "examples/proximal/qwen38/overhead/serving.json").read_text())["image"]

image = (
    add_fork_sources(
        modal.Image.from_registry(IMAGE)
        .entrypoint([])
        .env(
            {
                "PYTHONPATH": f"/root/Megatron-LM:{FORK}",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "PYTHONUNBUFFERED": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
    )
    .add_local_dir(REPO / "scripts/models", str(FORK / "scripts/models"))
    .add_local_file(REPO / "examples/proximal/qwen38/overhead/train_args.txt", str(FORK / "train_args.txt"))
)
base_volume = modal.Volume.from_name("miles-qwen38-base", environment_name="main", create_if_missing=False)
app = modal.App("miles-lora-parity")


def _checkpoint(model: str) -> tuple[Path, str]:
    """(checkpoint directory, Miles model-args script) for ``model``."""
    if model == "qwen38":
        return Path(QWEN38), "qwen3.8-27B"
    from huggingface_hub import snapshot_download

    name = model.split("/")[-1]
    return Path(snapshot_download(model, local_dir=f"/tmp/hf/{name}")), name.replace("Qwen", "qwen")


def _miles_argv(checkpoint: Path, model_args: str, targets: str) -> list[str]:
    from miles.utils.external_utils.model_args_utils import load_model_args

    lines = (FORK / "train_args.txt").read_text().splitlines()
    argv = [t for line in lines if line.strip() and not line.lstrip().startswith("#") for t in shlex.split(line)]
    argv += shlex.split(load_model_args(model_args, model_script_dir=FORK / "scripts/models"))
    argv += [
        "--hf-checkpoint", str(checkpoint),
        "--train-backend", "megatron",
        "--megatron-to-hf-mode", "bridge",
        "--lora-rank", "32", "--lora-alpha", "32", "--lora-dropout", "0",
        "--target-modules", targets,
        "--n-samples-per-prompt", "8",
        "--rollout-max-response-len", "1024",
        "--rollout-max-context-len", "32768",
        "--disable-rollout-global-dataset",
        "--load-debug-rollout-data", "/tmp/unused/{rollout_id}.pt",
    ]  # fmt: skip
    for flag, value in {
        "--actor-num-gpus-per-node": str(TP),
        "--tensor-model-parallel-size": str(TP),
        "--rollout-batch-size": "1",
        "--global-batch-size": "8",
        "--num-rollout": "1",
    }.items():
        argv = set_flag(argv, flag, value)
    for flag in ("--save", "--save-interval", "--use-wandb", "--wandb-project", "--wandb-group"):
        argv = set_flag(argv, flag, None)
    return argv


def _write_tokens(checkpoint: Path, lengths: list[int]) -> Path:
    """Fixed sequences of real code tokens (this fork's own sources), one per length."""
    from transformers import AutoTokenizer

    text = "\n".join(p.read_text() for p in sorted(Path("/root/miles_plugins").rglob("*.py")))
    ids = AutoTokenizer.from_pretrained(str(checkpoint)).encode(text)
    assert len(ids) >= sum(lengths), f"only {len(ids)} tokens of text for {lengths}"
    sequences, start = [], 0
    for length in lengths:
        sequences.append(ids[start : start + length])
        start += length
    path = WORK / "tokens.json"
    path.write_text(json.dumps(sequences))
    return path


def _broken_adapters(adapter: Path, checkpoint: Path) -> dict[str, Path]:
    """The published adapter with one layout mistake each: what a mapping error looks like."""
    import torch
    from safetensors.torch import load_file, save_file

    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    heads, head_dim = text["num_attention_heads"], text["head_dim"]

    def swap_gdn_qk(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not name.endswith("linear_attn.in_proj_qkv.lora_B.weight"):
            return tensor
        q, k, rest = tensor[:key_dim], tensor[key_dim : 2 * key_dim], tensor[2 * key_dim :]
        return torch.cat([k, q, rest]).contiguous()

    def swap_attention_gate(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not name.endswith("self_attn.q_proj.lora_B.weight"):
            return tensor
        per_head = tensor.reshape(heads, 2, head_dim, -1)
        return per_head.flip(1).reshape(tensor.shape).contiguous()

    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    broken = {}
    for label, edit in (("gdn_qk_swapped", swap_gdn_qk), ("attention_gate_swapped", swap_attention_gate)):
        directory = WORK / label
        directory.mkdir()
        (directory / "adapter_config.json").write_text((adapter / "adapter_config.json").read_text())
        save_file(
            {name: edit(name, tensor) for name, tensor in tensors.items()},
            str(directory / "adapter_model.safetensors"),
        )
        broken[label] = directory
    return broken


def _score(model: Path, tokens: Path, name: str, *, adapters: dict[str, Path], targets: list[str]) -> dict[str, Any]:
    out = WORK / f"{name}.json"
    command = ["python", "-m", STAGES, "logprobs", str(model), str(tokens), str(out)]
    if adapters:
        command += ["--rank", "32", "--targets", ",".join(targets)]
        command += [arg for label, path in adapters.items() for arg in ("--adapter", f"{label}={path}")]
    subprocess.run(command, check=True)
    return dict(json.loads(out.read_text()))


def _distance(a: list[list[float]], b: list[list[float]]) -> dict[str, float]:
    import torch

    diff = (torch.tensor([x for s in a for x in s]) - torch.tensor([x for s in b for x in s])).abs()
    return {"mean": diff.mean().item(), "p99": diff.quantile(0.99).item(), "max": diff.max().item()}


def _verdict(distances: dict[str, dict[str, float]]) -> dict[str, Any]:
    effect = distances["base"]["mean"]
    ratios = {label: d["mean"] / effect for label, d in distances.items()}
    checks = {
        "adapter_changes_the_model": effect >= 0.05,
        "adapter_matches_merged": ratios["adapter"] <= 0.3,
        "gdn_mapping_error_detected": ratios["gdn_qk_swapped"] >= 3 * ratios["adapter"],
        "attention_mapping_error_detected": ratios["attention_gate_swapped"] >= 3 * ratios["adapter"],
    }
    return {"ratios_to_adapter_effect": ratios, "checks": checks, "passed": all(checks.values())}


@app.function(
    image=image,
    gpu=f"B300:{TP}",
    cpu=32,
    memory=512 * 1024,
    volumes={"/models": base_volume.read_only()},
    timeout=3 * 3600,
)
def parity(model: str, targets: str, strength: float, seed: int, lengths: list[int]) -> dict[str, Any]:
    checkpoint, model_args = _checkpoint(model)
    WORK.mkdir(parents=True)
    miles = _miles_argv(checkpoint, model_args, targets)
    subprocess.run(
        ["torchrun", f"--nproc-per-node={TP}", "-m", STAGES, "export", str(WORK), "--strength", str(strength), "--seed", str(seed), "--", *miles],
        check=True,
    )  # fmt: skip
    export = json.loads((WORK / "export.json").read_text())
    if export["layout_problem"] is not None:
        return {"export": export, "passed": False}
    tokens = _write_tokens(checkpoint, lengths)
    serving_targets = export["serving_targets"]
    adapters = {"adapter": WORK / "adapter", **_broken_adapters(WORK / "adapter", checkpoint)}
    served = _score(checkpoint, tokens, "served", adapters=adapters, targets=serving_targets)
    merged = _score(WORK / "merged", tokens, "merged", adapters={}, targets=[])["base"]
    distances = {label: _distance(scores, merged) for label, scores in served.items()}
    return {"export": export, "lengths": lengths, "distances_to_merged": distances, **_verdict(distances)}


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-4B",
    targets: str = ALL_LINEAR,
    strength: float = 0.05,
    seed: int = 0,
    lengths: str = "512,4096,16384",
    out: str = "lora_parity.json",
) -> None:
    result = parity.remote(model, targets, strength, seed, [int(n) for n in lengths.split(",")])
    Path(out).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: result.get(k) for k in ("distances_to_merged", "ratios_to_adapter_effect", "checks", "passed")},
            indent=2,
        )
    )
    print(f"round trip: {result['export'].get('round_trip')}")
    print(f"details in {out}")
