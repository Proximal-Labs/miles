"""Modal transport for scripts/run_inkling_small_sft.py; no work runs on import."""

import hashlib
import json
import os
import shlex
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

import modal
import modal.experimental

_ROOT = Path(__file__).resolve().parents[1]
_REMOTE_ROOT = "/opt/inkling-miles"
app = modal.App("inkling-small-sft")
volume = modal.Volume.from_name("inkling-small-rft")
# Pinned linux/amd64 Miles image; the old :inkling tag is ARM64-only.
_DEFAULT_IMAGE = "radixark/miles@sha256:8ee6528fa209dd3bc65ccb40556e6606e3e9e502cd521d994d3ee6da3a58b67d"
_IMAGE_REF = os.environ.get("INKLING_MODAL_IMAGE", _DEFAULT_IMAGE)
_CACHE_ROOT = f"/mnt/inkling/compile-cache/{hashlib.sha256(_IMAGE_REF.encode()).hexdigest()[:16]}"
image = modal.Image.from_registry(_IMAGE_REF)
_EVAL_SECRET = os.environ.get("INKLING_EVAL_SECRET")
if _EVAL_SECRET:
    image = image.pip_install("modal==1.5.5", "httpx==0.28.1").env({"INKLING_EVAL_SECRET": _EVAL_SECRET})
# Set these before importing torch or starting Ray so every worker inherits them.
# The existing final volume.commit() also preserves caches after failed jobs.
image = image.entrypoint([]).env(
    {
        "PYTHONPATH": f"{_REMOTE_ROOT}:/root/Megatron-LM",
        "TORCHINDUCTOR_CACHE_DIR": f"{_CACHE_ROOT}/inductor",
        "TRITON_CACHE_DIR": f"{_CACHE_ROOT}/triton",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
    }
)
data_image = image.pip_install_from_requirements(str(_ROOT / "tools/requirements-inkling-sft.txt"))


def _add_sources(container_image):
    # Startup mounts must follow every image build step, including pip installs.
    for directory in ("miles", "miles_plugins", "scripts", "tools"):
        container_image = container_image.add_local_dir(_ROOT / directory, f"{_REMOTE_ROOT}/{directory}", ignore=["**/__pycache__/**", "**/*.pyc"])
    return container_image.add_local_file(_ROOT / "train.py", f"{_REMOTE_ROOT}/train.py")


image = _add_sources(image)
data_image = _add_sources(data_image)


def _config(config_json):
    # Miles and its GPU dependencies are supplied by the remote image, not the Modal CLI environment.
    from scripts.run_inkling_small_sft import ScriptArgs

    config = json.loads(config_json)
    config.pop("_eval_config", None)
    return ScriptArgs(**config)


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
    candidates = [Path(args.lora_adapter_path)] if args.lora_adapter_path else sorted(Path(args.save_dir).glob("iter_[0-9]*/adapter"), reverse=True)
    for candidate in candidates:
        required = [candidate / f"{prefix}{rank}.pt" for rank in range(args.num_nodes * args.num_gpus_per_node) for prefix in ("adapter_megatron_rank", "training_state_rank")]
        iteration = int(candidate.parent.name.removeprefix("iter_"))
        required.append(candidate.parents[1] / f"rollout/global_dataset_state_dict_{iteration}.pt")
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            return str(candidate)
    raise FileNotFoundError("No complete native LoRA checkpoint found with all training ranks and the matching dataset cursor")


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


_GPU_OPTIONS = dict(
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
        *([modal.Secret.from_name(_EVAL_SECRET)] if _EVAL_SECRET else []),
    ],
    timeout=86400,
    retries=0,
    max_containers=1,
)


@app.function(**_GPU_OPTIONS)
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
        # Older one-node runs predate configurable parallelism.
        saved_config = {"tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 2, "expert_model_parallel_size": 4, "decoder_first_pipeline_num_layers": None, "decoder_last_pipeline_num_layers": None, **saved_config}
        for field in ("lora_rank", "lora_alpha", "num_nodes", "num_gpus_per_node", "lr", "mode", "tensor_model_parallel_size", "pipeline_model_parallel_size", "expert_model_parallel_size", "decoder_first_pipeline_num_layers", "decoder_last_pipeline_num_layers"):
            if saved_config.get(field) != getattr(args, field):
                raise ValueError(f"Cannot resume with changed {field}; retain the saved LoRA configuration")
        args = replace(args, lora_adapter_path=_resume_adapter(args))
    if not args.resume and destination.exists():
        raise FileExistsError("Run directory already exists; use --resume or a new --run-id")
    destination.mkdir(parents=True, exist_ok=True)
    if args.eval_enabled:
        evaluation = json.loads(config_json).get("_eval_config")
        if evaluation is None or not _EVAL_SECRET:
            raise ValueError("Launch evaluations through scripts.run_inkling_small_sft modal with --eval-config")
        config_path = destination / "evaluation-config.json"
        encoded = json.dumps(evaluation, indent=2) + "\n"
        if config_path.exists() and config_path.read_text() != encoded:
            raise ValueError("Evaluation configuration changed on resume")
        config_path.write_text(encoded)
        args = replace(args, eval_config=str(config_path))
    (destination / "launch.json").write_text(json.dumps(asdict(args), indent=2) + "\n")
    print(f"EXPERIMENTAL: LoRA rank {args.lora_rank} / {args.num_nodes} x 8 B300 / TP{args.tensor_model_parallel_size} PP{args.pipeline_model_parallel_size} EP{args.expert_model_parallel_size} DP{args.data_parallel_size} / cap {args.max_length}; fit is unvalidated")
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
        if args.num_nodes == 1:
            try:
                U.exec_command_cpu("ray stop --force")
            finally:
                volume.commit()


@app.function(**{**_GPU_OPTIONS, "max_containers": 2})
@modal.experimental.clustered(size=2, rdma=True)
def train_two_nodes(config_json: str, state: modal.Dict):
    # GPU/Ray dependencies are supplied by the remote image.
    from tools.modal_inkling_sft_cluster import training_cluster

    args = _config(config_json)
    info = modal.experimental.get_cluster_info()
    with training_cluster(args, state, info.rank, info.container_ipv4_ips, volume, _gpu_preflight):
        if info.rank == 0:
            train.local(config_json)


@app.function(image=image, timeout=86400, retries=0, max_containers=1)
def run_cluster(config_json: str):
    # Own the coordination resource remotely so detaching the local CLI cannot
    # delete it underneath a running training job. All containers share this App.
    with modal.Dict.ephemeral() as state:
        call = train_two_nodes.spawn(config_json, state)
        try:
            call.get()
        except BaseException:
            state["error"] = "Cluster coordinator cancelled or failed"
            with suppress(Exception):
                call.get(timeout=180)
            raise
        finally:
            call.cancel(terminate_containers=True)


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
    elif config["mode"] in {"smoke", "train"} and config.get("num_nodes", 1) == 2:
        run_cluster.remote(config_json)
    else:
        train.remote(config_json)
