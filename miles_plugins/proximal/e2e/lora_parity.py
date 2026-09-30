"""LoRA serving parity: does SGLang serve the adapter a run publishes as the trainer trained it?

One node, one tensor-parallel group like a run's trainer (``lora_parity_stages.py``):

1. Build the trainer's LoRA model for ``--targets`` and load the checkpoint. While every
   lora_B is still zero, Megatron-Bridge's merged export must reproduce the checkpoint
   (round trip).
2. Give every adapter random nonzero weights whose update is ``--strength`` times the base
   weight's RMS. Write the adapter a run would publish (the publisher's staging, layout
   check and writer) and the same adapter merged into a full checkpoint in Megatron's own
   layout: the trained policy.
3. Export mapping, per module: ``merged - base`` must equal ``(alpha / r) B A`` computed
   from the published tensors, up to the merged checkpoint's bf16 rounding.
4. Serving, per module family: SGLang scores fixed token sequences under the published
   adapter and its MLP-only and attention/GDN-only subsets (loaded like a replica, strict),
   and under each one's merged checkpoint. An adapter must match its merged checkpoint far
   more closely than the base model does. Two deliberately broken attention/GDN adapters
   (GDN q and k slices swapped, attention q and gate halves swapped) show what a mapping
   error looks like and must be caught. A plain engine on the base checkpoint gives the
   engine's own noise floor.

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
# The training image, as the Qwen3.8 deployments pin it (read where the app is launched).
IMAGE = (
    json.loads((REPO / "examples/proximal/qwen38/overhead/serving.json").read_text())["image"]
    if modal.is_local()
    else ""
)

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
    from transformers import AutoTokenizer  # type: ignore[import-not-found,unused-ignore]

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


# The adapter's module families, as HF tensor leaves.
FAMILIES = {
    "mlp": frozenset({"gate_proj", "up_proj", "down_proj"}),
    "mixers": frozenset(
        {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
    ),
}


def _leaf(name: str) -> str:
    return name.split(".lora_")[0].rsplit(".", 1)[-1]


def _pairs(adapter: Path) -> dict[str, tuple[Any, Any]]:
    """Adapter module name -> (lora_A, lora_B)."""
    from safetensors.torch import load_file

    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    modules = {name.split(".lora_")[0] for name in tensors}
    return {m: (tensors[f"{m}.lora_A.weight"], tensors[f"{m}.lora_B.weight"]) for m in modules}


def _scale(adapter: Path) -> float:
    config = json.loads((adapter / "adapter_config.json").read_text())
    return float(config["lora_alpha"]) / float(config["r"])


def _weights(checkpoint: Path) -> dict[str, Path]:
    """Tensor name -> the safetensors file holding it."""
    from safetensors import safe_open

    index: dict[str, Path] = {}
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(str(path), framework="pt") as handle:
            index.update({key: path for key in handle.keys()})
    return index


def _base_name(module: str) -> str:
    return module.removeprefix("base_model.model.") + ".weight"


def _export_mapping_errors(adapter: Path, checkpoint: Path, merged: Path) -> dict[str, float]:
    """Per tensor leaf, the largest relative error of ``merged - base`` against ``(alpha / r) B A``."""
    from safetensors import safe_open

    scale, base, after = _scale(adapter), _weights(checkpoint), _weights(merged)
    errors: dict[str, float] = {}
    for module, (a, b) in _pairs(adapter).items():
        name = _base_name(module)
        with safe_open(str(base[name]), framework="pt") as old, safe_open(str(after[name]), framework="pt") as new:
            actual = new.get_tensor(name).float() - old.get_tensor(name).float()
        expected = scale * (b.float() @ a.float())
        error = ((actual - expected).norm() / expected.norm()).item()
        errors[_leaf(module)] = max(errors.get(_leaf(module), 0.0), error)
    return errors


def _subset(adapter: Path, leaves: frozenset[str], directory: Path) -> Path:
    from safetensors.torch import load_file, save_file

    directory.mkdir()
    (directory / "adapter_config.json").write_text((adapter / "adapter_config.json").read_text())
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    save_file({n: t for n, t in tensors.items() if _leaf(n) in leaves}, str(directory / "adapter_model.safetensors"))
    return directory


def _merge(adapter: Path, checkpoint: Path, directory: Path) -> Path:
    """The checkpoint with ``adapter`` merged in HF space: each base weight plus (alpha / r) B A."""
    import shutil

    from safetensors.torch import load_file, save_file

    scale, index = _scale(adapter), _weights(checkpoint)
    updates: dict[Path, dict[str, tuple[Any, Any]]] = {}
    for module, pair in _pairs(adapter).items():
        name = _base_name(module)
        updates.setdefault(index[name], {})[name] = pair
    directory.mkdir()
    for path in checkpoint.iterdir():
        if path.is_file() and path not in updates:
            shutil.copy2(path, directory / path.name)
    for path, changes in updates.items():
        tensors = load_file(str(path))
        for name, (a, b) in changes.items():
            base = tensors[name]
            tensors[name] = (base.float() + scale * (b.float() @ a.float())).to(base.dtype)
        save_file(tensors, str(directory / path.name), metadata={"format": "pt"})
    return directory


def _broken(adapter: Path, checkpoint: Path) -> dict[str, Path]:
    """``adapter`` with one layout mistake each: what a mapping error looks like."""
    import torch
    from safetensors.torch import load_file, save_file

    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    heads, head_dim = text["num_attention_heads"], text["head_dim"]

    def swap_gdn_qk(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not name.endswith("linear_attn.in_proj_qkv.lora_B.weight"):
            return tensor
        return torch.cat([tensor[key_dim : 2 * key_dim], tensor[:key_dim], tensor[2 * key_dim :]]).contiguous()

    def swap_attention_gate(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not name.endswith("self_attn.q_proj.lora_B.weight"):
            return tensor
        return tensor.reshape(heads, 2, head_dim, -1).flip(1).reshape(tensor.shape).contiguous()

    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    broken = {}
    for label, edit in (("gdn_qk_swapped", swap_gdn_qk), ("attention_gate_swapped", swap_attention_gate)):
        directory = WORK / label
        directory.mkdir()
        (directory / "adapter_config.json").write_text((adapter / "adapter_config.json").read_text())
        save_file({n: edit(n, t) for n, t in tensors.items()}, str(directory / "adapter_model.safetensors"))
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


def _distance(a: list[list[float]], b: list[list[float]]) -> dict[str, Any]:
    """Absolute logprob differences, over all tokens and per sequence."""
    import torch

    def stats(x: list[float], y: list[float]) -> dict[str, float]:
        diff = (torch.tensor(x) - torch.tensor(y)).abs()
        return {"mean": diff.mean().item(), "p99": diff.quantile(0.99).item(), "max": diff.max().item()}

    flat = stats([v for s in a for v in s], [v for s in b for v in s])
    return flat | {"per_sequence_mean": [stats(x, y)["mean"] for x, y in zip(a, b, strict=True)]}


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
    adapter = WORK / "adapter"
    mapping = _export_mapping_errors(adapter, checkpoint, WORK / "merged")

    # Each served adapter, and the merged checkpoint it must reproduce.
    adapters, truth = {"all": adapter}, {"all": WORK / "merged"}
    for family, leaves in FAMILIES.items():
        adapters[family] = _subset(adapter, leaves, WORK / f"{family}-adapter")
        truth[family] = _merge(adapters[family], checkpoint, WORK / f"{family}-merged")
    for label, path in _broken(adapters["mixers"], checkpoint).items():
        adapters[label], truth[label] = path, truth["mixers"]

    tokens = _write_tokens(checkpoint, lengths)
    served = _score(checkpoint, tokens, "served", adapters=adapters, targets=export["serving_targets"])
    plain = _score(checkpoint, tokens, "plain", adapters={}, targets=[])["base"]
    merged = {p: _score(p, tokens, p.name, adapters={}, targets=[])["base"] for p in set(truth.values())}
    report: dict[str, Any] = {"export": export, "export_mapping_errors": mapping, "lengths": lengths}
    report["noise_floor"] = _distance(served["base"], plain)
    report["served"] = {
        label: {
            "residual": _distance(served[label], merged[truth[label]]),
            "effect": _distance(served["base"], merged[truth[label]]),
        }
        for label in adapters
    }
    return report | _verdict(report)


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    ratio = {label: r["residual"]["mean"] / r["effect"]["mean"] for label, r in report["served"].items()}
    checks = {
        # A wrong export mapping errs by about 100% of the update; bf16 rounding by a few percent.
        "export_mapping": max(report["export_mapping_errors"].values()) <= 0.1,
        "round_trip": report["export"]["round_trip"]["max_abs_difference"] <= 2**-8,
        "adapters_change_the_model": all(r["effect"]["mean"] >= 0.02 for r in report["served"].values()),
        "served_like_merged": all(ratio[label] <= 0.3 for label in ("all", *FAMILIES)),
        "mapping_errors_caught": all(
            ratio[label] >= 2 * ratio["mixers"] for label in ("gdn_qk_swapped", "attention_gate_swapped")
        ),
    }
    return {"residual_to_effect": ratio, "checks": checks, "passed": all(checks.values())}


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-4B",
    targets: str = ALL_LINEAR,
    strength: float = 0.15,  # Large enough that a mapping error dwarfs the merged checkpoint's bf16 rounding.
    seed: int = 0,
    lengths: str = "512,4096,16384",
    out: str = "lora_parity.json",
) -> None:
    result = parity.remote(model, targets, strength, seed, [int(n) for n in lengths.split(",")])
    Path(out).write_text(json.dumps(result, indent=2))
    summary = {k: result.get(k) for k in ("export_mapping_errors", "residual_to_effect", "checks", "passed")}
    print(json.dumps(summary, indent=2))
    print(f"round trip: {result['export'].get('round_trip')}")
    print(f"details in {out}")
