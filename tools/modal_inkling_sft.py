"""Modal transport for scripts/run_inkling_small_sft.py; no work runs on import."""

import json
import os
import shlex
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

import modal

_ROOT = Path(__file__).resolve().parents[1]
_REMOTE_ROOT = "/opt/inkling-miles"
app = modal.App("inkling-small-sft")
volume = modal.Volume.from_name("inkling-small-rft")
# Pinned linux/amd64 Miles image; the old :inkling tag is ARM64-only.
_DEFAULT_IMAGE = "radixark/miles@sha256:8ee6528fa209dd3bc65ccb40556e6606e3e9e502cd521d994d3ee6da3a58b67d"
image = modal.Image.from_registry(os.environ.get("INKLING_MODAL_IMAGE", _DEFAULT_IMAGE))
image = image.entrypoint([]).env({"PYTHONPATH": f"{_REMOTE_ROOT}:/root/Megatron-LM"})
data_image = image.pip_install_from_requirements(str(_ROOT / "tools/requirements-inkling-sft.txt"))


def _add_sources(container_image):
    # Startup mounts must follow every image build step, including pip installs.
    for directory in ("miles", "miles_plugins", "scripts", "tools"):
        container_image = container_image.add_local_dir(
            _ROOT / directory, f"{_REMOTE_ROOT}/{directory}", ignore=["**/__pycache__/**", "**/*.pyc"]
        )
    return container_image.add_local_file(_ROOT / "train.py", f"{_REMOTE_ROOT}/train.py")


image = _add_sources(image)
data_image = _add_sources(data_image)


def _config(config_json):
    # Miles and its GPU dependencies are supplied by the remote image, not the Modal CLI environment.
    from scripts.run_inkling_small_sft import ScriptArgs

    return ScriptArgs(**json.loads(config_json))


def _gpu_preflight():
    import torch

    cuda = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
    if cuda < (13, 0):
        raise RuntimeError(f"This experimental B300 recipe requires torch CUDA >=13.0; found {torch.version.cuda}")
    if cuda < (13, 1):
        print("EXPERIMENTAL: CUDA 13.0 on B300; Modal documents 13.1+. Kernel compatibility must pass the smoke run.")
    if torch.cuda.device_count() != 8:
        raise RuntimeError("Expected exactly 8 visible B300 GPUs")
    for index in range(8):
        props = torch.cuda.get_device_properties(index)
        if "B300" not in props.name:
            raise RuntimeError(f"Unexpected GPU: {props.name}")
        print(f"GPU {index}: {props.name}, {props.total_memory / 2**30:.1f} GiB")


def _resume_adapter(args):
    candidates = (
        [Path(args.lora_adapter_path)]
        if args.lora_adapter_path
        else sorted(Path(args.save_dir).glob("iter_[0-9]*/adapter"), reverse=True)
    )
    for candidate in candidates:
        required = [
            candidate / f"{prefix}{rank}.pt"
            for rank in range(args.num_gpus_per_node)
            for prefix in ("adapter_megatron_rank", "training_state_rank")
        ]
        iteration = int(candidate.parent.name.removeprefix("iter_"))
        required.append(candidate.parents[1] / f"rollout/global_dataset_state_dict_{iteration}.pt")
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            return str(candidate)
    raise FileNotFoundError(
        "No complete native LoRA checkpoint found with all 8 ranks and the matching dataset cursor"
    )


@app.function(
    image=data_image,
    volumes={"/mnt/inkling": volume},
    secrets=[modal.Secret.from_name("rft_hf_token", required_keys=["HF_TOKEN"])],
    cpu=8,
    memory=32768,
    timeout=86400,
    retries=0,
    max_containers=1,
)
def prepare_data(config_json: str):
    from huggingface_hub import snapshot_download
    from tools.prepare_inkling_sft import _prepare

    args = _config(config_json)
    if not Path(args.source_data).is_file():
        raise FileNotFoundError(f"Upload the raw dataset to {args.source_data} first")
    snapshot_download(
        "thinkingmachines/Inkling-Small",
        local_dir=args.hf_checkpoint,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
    )
    _prepare(Path(args.source_data), Path(args.data_dir) / "train.prepared.jsonl", args.hf_checkpoint, args.max_length)
    # Validate data before downloading hundreds of GB of weights.
    snapshot_download("thinkingmachines/Inkling-Small", local_dir=args.hf_checkpoint)
    volume.commit()


@app.function(
    image=image,
    gpu="B300:8",
    cpu=32,
    memory=524288,
    # Leave room for checkpoint writes alongside the roughly 500 GiB base model.
    ephemeral_disk=3 * 1024 * 1024,
    volumes={"/mnt/inkling": volume},
    secrets=[
        modal.Secret.from_name("rft_hf_token", required_keys=["HF_TOKEN"]),
        modal.Secret.from_name("rft_wandb_api_key", required_keys=["WANDB_API_KEY"]),
    ],
    timeout=86400,
    retries=0,
    max_containers=1,
)
def train(config_json: str):
    from scripts.run_inkling_small_sft import prepare

    import miles.utils.external_utils.command_utils as U

    args = _config(config_json)
    _gpu_preflight()
    if args.mode == "prepare":
        if not (Path(args.hf_checkpoint) / "model.safetensors.index.json").is_file():
            raise FileNotFoundError("Run --mode data first to download the checkpoint")
        prepare(**asdict(args))
        volume.commit()
        return
    if args.mode not in {"smoke", "train"}:
        raise ValueError(f"Unsupported GPU mode: {args.mode}")
    if not Path(args.dataset).is_file():
        raise FileNotFoundError("Run --mode data first")
    if not (Path(args.torch_dist) / "latest_checkpointed_iteration.txt").is_file():
        raise FileNotFoundError("Run --mode prepare first")
    destination = Path(args.save_dir)
    if args.resume:
        saved_config = json.loads((destination / "launch.json").read_text())
        for field in ("lora_rank", "lora_alpha", "num_nodes", "num_gpus_per_node", "lr", "mode"):
            if saved_config.get(field) != getattr(args, field):
                raise ValueError(f"Cannot resume with changed {field}; retain the saved LoRA configuration")
        args = replace(args, lora_adapter_path=_resume_adapter(args))
    if not args.resume and destination.exists():
        raise FileExistsError("Run directory already exists; use --resume or a new --run-id")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "launch.json").write_text(json.dumps(asdict(args), indent=2) + "\n")
    print(
        f"EXPERIMENTAL: LoRA rank {args.lora_rank} / 8 B300 / TP4 PP2 EP4 / cap {args.max_length}; fit is unvalidated"
    )
    try:
        # Bound subprocess time independently of Modal's outer 24-hour timeout,
        # leaving time to commit completed checkpoints after a failure.
        config_path = destination / "launch.json"
        U.exec_command_cpu(
            shlex.join(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=60",
                    str(args.timeout_hours * 3600 - 120),
                    "python",
                    str(U.repo_base_dir / "tools/modal_inkling_sft_worker.py"),
                    str(config_path),
                ]
            )
        )
    finally:
        # Ray workers outlive the submitting CLI. Stop them before committing,
        # including when timeout killed the CLI while its job was still active.
        try:
            U.exec_command_cpu("ray stop --force")
        finally:
            volume.commit()


def _prepared_checkpoint_cached(config):
    model_dir = PurePosixPath(config.get("model_dir", "/mnt/inkling/models"))
    if ".." in model_dir.parts:
        raise ValueError("model_dir cannot contain '..'")
    relative_dir = model_dir.relative_to("/mnt/inkling")
    tracker = relative_dir / "Inkling-Small_torch_dist/latest_checkpointed_iteration.txt"
    try:
        marker = b"".join(volume.read_file(str(tracker)))
    except FileNotFoundError:
        return False
    # Same completion criterion as command_utils.convert_checkpoint. Do not
    # interpret authentication/network failures as cache misses and allocate GPUs.
    return marker.strip() == b"release"


@app.local_entrypoint()
def main(config_json: str):
    config = json.loads(config_json)
    # Record the actual image even when submitting directly with the Modal CLI.
    config["image"] = os.environ.get("INKLING_MODAL_IMAGE", _DEFAULT_IMAGE)
    config_json = json.dumps(config)
    if config["mode"] == "data":
        prepare_data.remote(config_json)
    elif config["mode"] == "prepare" and _prepared_checkpoint_cached(config):
        print("Converted Inkling-Small checkpoint is cached on inkling-small-rft; skipping GPU allocation.")
    else:
        train.remote(config_json)
