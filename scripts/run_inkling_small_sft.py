"""Experimental Inkling-Small LoRA SFT on one node of 8 B300s.

Uses Miles/Megatron, GPU-resident Adam, TP4/PP2/EP4,
full activation recomputation and no rollout inference. 262K training is an
unvalidated memory target, not a demonstrated fit. Allows CUDA >=13.0 experimentally
with the Miles Megatron fork. The frozen BF16 base and trainable adapters stay
on GPU; only adapters have gradients and optimizer state.

Args:
  --mode: data (CPU download/render), prepare (GPU conversion), smoke, train.
  --source-data: Raw JSONL with messages and optional tools/reasoning_effort.
  --max-length: Total token cap, including reasoning and tool results.
  --lr: Initial experimental Muon learning rate; no validated Inkling SFT LR.
  --lora-rank / --lora-alpha: Adapter rank and scaling numerator (both default 32).
  --lora-adapter-path: Explicit native adapter checkpoint for resume; Modal can
    select the latest complete adapter checkpoint in the run directory.
  --run-id: Stable identifier; reuse with --resume to restore training state.
  --image: Modal container image; runtime preflight checks CUDA and GPUs.
  --model-dir / --data-dir / --output-dir: Paths inside the Modal Volume.

Examples (local host, Modal credentials for proximal already configured):
  python -m scripts.run_inkling_small_sft modal --mode data
  python -m scripts.run_inkling_small_sft modal --mode prepare
  python -m scripts.run_inkling_small_sft modal --mode smoke --run-id fit-check
  python -m scripts.run_inkling_small_sft modal --mode train --run-id sft-001

Inside a prepared container, use `execute` or `prepare` instead of `modal`.
"""

import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    run_id: str = field(default_factory=U.create_run_id)
    mode: Literal["data", "prepare", "smoke", "train"] = "smoke"
    model_dir: str = "/mnt/inkling/models"
    data_dir: str = "/mnt/inkling/data"
    output_dir: str = "/mnt/inkling/checkpoints"
    source_data: str = "/mnt/inkling/data/train.jsonl"
    megatron_path: str = "/root/Megatron-LM"
    max_length: int = 262144
    lr: float = 1e-5
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_adapter_path: str | None = None
    save_interval: int = 100
    resume: bool = False
    wandb_entity: str = "evan-proximal-proximal"
    wandb_project: str = "inkling-small-rft"
    profile: str = "proximal"
    environment: str = "main"
    image: str = "radixark/miles@sha256:8ee6528fa209dd3bc65ccb40556e6606e3e9e502cd521d994d3ee6da3a58b67d"
    timeout_hours: int = 24

    def __post_init__(self):
        if (self.num_nodes, self.num_gpus_per_node) != (1, 8):
            raise ValueError("This experimental profile requires exactly one node with 8 GPUs")
        if not 1 <= self.max_length <= 1048576:
            raise ValueError("max_length must be between 1 and 1048576")
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be a single directory name")
        if not 0 < self.lr or not 1 <= self.timeout_hours <= 24 or self.save_interval < 1:
            raise ValueError("Invalid LR, timeout or save interval")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.lora_adapter_path and not self.resume:
            raise ValueError("Use --resume with --lora-adapter-path")

    @property
    def hf_checkpoint(self):
        return f"{self.model_dir}/Inkling-Small"

    @property
    def torch_dist(self):
        return f"{self.model_dir}/Inkling-Small_torch_dist"

    @property
    def dataset(self):
        suffix = ".smoke" if self.mode == "smoke" else ""
        return f"{self.data_dir}/train.prepared{suffix}.jsonl"

    @property
    def save_dir(self):
        return f"{self.output_dir}/{self.run_id}"


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    U.convert_checkpoint(
        model_name="Inkling-Small",
        megatron_model_type="inkling-small",
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args="--tensor-model-parallel-size 1 --pipeline-model-parallel-size 8 --decoder-first-pipeline-num-layers 6 --decoder-last-pipeline-num-layers 6 --expert-model-parallel-size 1 --moe-router-dtype fp32 --bf16",
    )


@app.command()
@U.dataclass_cli
def execute(args: ScriptArgs):
    q = shlex.quote
    checkpoint_args = f"--hf-checkpoint {q(args.hf_checkpoint)} --model-name inkling --megatron-to-hf-mode raw --load {q(args.torch_dist)} --save {q(args.save_dir)} --save-interval {args.save_interval} "
    if args.resume:
        if not args.lora_adapter_path:
            raise ValueError("Local execute --resume requires --lora-adapter-path; Modal resolves it automatically")
        checkpoint_args += f"--lora-adapter-path {q(args.lora_adapter_path)} "
    if not args.resume:
        # The release base reports iteration 0, but contains no completed SFT rollout.
        checkpoint_args += "--no-load-optim --no-load-rng --start-rollout-id 0 --finetune "
    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        "--target-modules all-linear --experts-shared-outer-loras "
    )
    sft_args = (
        "--rollout-function-path miles.rollout.inkling_sft.generate_rollout "
        "--data-source-path miles.rollout.inkling_sft_data_source.InklingSFTDataSource "
        f"--prompt-data {q(args.dataset)} --input-key text --metadata-key metadata "
        "--rollout-shuffle --rollout-batch-size 1 --global-batch-size 1 --n-samples-per-prompt 1 "
        "--num-epoch 1 --loss-type sft_loss --calculate-per-token-loss "
        "--disable-compute-advantages-and-returns --debug-train-only "
    )
    perf_args = f"--tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 --context-parallel-size 1 --sequence-parallel --micro-batch-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --seq-length {args.max_length} "
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --min-lr {args.lr * 0.1} "
        "--lr-decay-style cosine --lr-warmup-fraction 0.03 --weight-decay 0.1 --clip-grad 1.0 "
    )
    misc_args = f"--bf16 --moe-router-dtype fp32 --transformer-impl transformer_engine --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --no-bias-dropout-fusion --actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --num-gpus-per-node {args.num_gpus_per_node} "
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    if wandb_args:
        # The shared helper includes the API key in a printed command. Let W&B
        # read the inherited secret instead, and override its generated project.
        parts = shlex.split(wandb_args)
        index = parts.index("--wandb-key")
        del parts[index : index + 2]
        parts[parts.index("--wandb-project") + 1] = args.wandb_project
        wandb_args = shlex.join(parts)
    U.execute_train(
        train_args=f"{checkpoint_args} {lora_args} {sft_args} {perf_args} {optimizer_args} {misc_args} {wandb_args}",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="inkling-small",
        config=args,
        megatron_path=args.megatron_path,
        train_script="train.py",
        extra_env_vars={
            "WANDB_ENTITY": args.wandb_entity,
            "MILES_INKLING_ATTN_BACKEND": "flex",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )


@app.command("modal")
@U.dataclass_cli
def launch(args: ScriptArgs):
    command = shlex.join(
        [
            "modal",
            "run",
            "--detach",
            "--env",
            args.environment,
            str(U.repo_base_dir / "tools/modal_inkling_sft.py"),
            "--config-json",
            json.dumps(asdict(args)),
        ]
    )
    U.exec_command_cpu(
        f"MODAL_PROFILE={shlex.quote(args.profile)} INKLING_MODAL_IMAGE={shlex.quote(args.image)} {command}"
    )


if __name__ == "__main__":
    app()
