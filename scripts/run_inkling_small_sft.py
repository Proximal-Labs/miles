"""Experimental Inkling-Small LoRA SFT on one or two nodes of 8 B300s each.

Uses Miles/Megatron, GPU-resident Adam, TP4/PP2/EP4,
full activation recomputation and no rollout inference. 262K training is an
unvalidated memory target, not a demonstrated fit. Allows CUDA >=13.0 experimentally
with the Miles Megatron fork. The frozen BF16 base and trainable adapters stay
on GPU; only adapters have gradients and optimizer state.

Args:
  --num-nodes: One or two; TP4/PP2 defaults give DP1 or DP2 respectively.
  --tensor-model-parallel-size / --pipeline-model-parallel-size: Default 4 / 2.
  --expert-model-parallel-size: Default 4; expert tensor parallelism stays 1.
  --decoder-first-pipeline-num-layers / --decoder-last-pipeline-num-layers:
    Optional uneven stage sizes. For PP4, set the last stage to 12 (10/10/10/12).
  --mode: data (CPU download/render), prepare (GPU conversion), smoke, train.
  --source-data: Raw JSONL with messages and optional tools/reasoning_effort.
  --max-length: Total token cap, including reasoning and tool results.
  --num-epoch: Passes over the prepared dataset (default 10).
  --global-batch-size: Conversations per optimizer step (default 32).
  --lr: Initial experimental Adam learning rate; no validated Inkling SFT LR.
  --min-lr: Cosine decay floor (default 1e-6).
  --warmup-epoch-fraction: Linear warmup up to one full epoch (default 0.1).
  --distributed-timeout-minutes: GPU communication timeout (default 30), including
    waits while another pipeline stage compiles its first-step kernels.
  --lora-rank / --lora-alpha: Adapter rank and scaling numerator (both default 32).
  --lora-adapter-path: Explicit native adapter checkpoint for resume; Modal can
    select the latest complete adapter checkpoint in the run directory.
  --run-id: Stable identifier; reuse with --resume to restore training state.
  --image: Modal container image; runtime preflight checks CUDA and GPUs.
  --model-dir / --data-dir / --output-dir: Paths inside the Modal Volume.
  --eval-config: JSON named environment sets and Proximal/Modal evaluation settings.
  --eval-every-n-epochs: Evaluate before training and every N epochs (default 1).
    Zero disables all evaluation. Smoke mode never runs environment evaluations.
  --eval-rollouts-per-env: Override the config's rollout count for every environment.

Examples (local host, Modal credentials for proximal already configured):
  python -m scripts.run_inkling_small_sft modal --mode data
  python -m scripts.run_inkling_small_sft modal --mode prepare
  python -m scripts.run_inkling_small_sft modal --mode smoke --run-id fit-check
  python -m scripts.run_inkling_small_sft modal --mode train --run-id sft-001
  python -m scripts.run_inkling_small_sft modal --mode train --num-nodes 2 --run-id sft-dp2
  python -m scripts.run_inkling_small_sft modal --mode train --num-nodes 2 --pipeline-model-parallel-size 4 --decoder-last-pipeline-num-layers 12 --run-id sft-pp4

Inside a prepared container, use `execute` or `prepare` instead of `modal`.
Multi-node `execute` requires an already joined Ray cluster and
MILES_SCRIPT_EXTERNAL_RAY=1. Modal configures this automatically.
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
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 2
    expert_model_parallel_size: int = 4
    decoder_first_pipeline_num_layers: int | None = None
    decoder_last_pipeline_num_layers: int | None = None
    run_id: str = field(default_factory=U.create_run_id)
    mode: Literal["data", "prepare", "smoke", "train"] = "smoke"
    model_dir: str = "/mnt/inkling/models"
    data_dir: str = "/mnt/inkling/data"
    output_dir: str = "/mnt/inkling/checkpoints"
    source_data: str = "/mnt/inkling/data/train.jsonl"
    megatron_path: str = "/root/Megatron-LM"
    max_length: int = 262144
    lr: float = 1e-5
    min_lr: float = 1e-6
    warmup_epoch_fraction: float = 0.1
    num_epoch: int = 10
    global_batch_size: int = 32
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
    distributed_timeout_minutes: int = 30
    eval_config: str | None = None
    eval_every_n_epochs: int = 1
    eval_rollouts_per_env: int | None = None

    def __post_init__(self):
        if self.num_nodes not in (1, 2) or self.num_gpus_per_node != 8:
            raise ValueError("This experimental profile requires one or two nodes with 8 GPUs each")
        self._validate_parallelism()
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
        if self.distributed_timeout_minutes < 1:
            raise ValueError("distributed_timeout_minutes must be positive")
        if self.num_epoch < 1:
            raise ValueError("num_epoch must be positive")
        if self.eval_every_n_epochs < 0:
            raise ValueError("eval_every_n_epochs must be nonnegative")
        if self.eval_rollouts_per_env is not None and self.eval_rollouts_per_env < 1:
            raise ValueError("eval_rollouts_per_env must be positive")
        if self.global_batch_size < 1:
            raise ValueError("global_batch_size must be positive")
        if self.global_batch_size % self.data_parallel_size:
            raise ValueError("global_batch_size must be divisible by data parallel size")
        if not 0 <= self.min_lr <= self.lr:
            raise ValueError("min_lr must be between zero and lr")
        if not 0 <= self.warmup_epoch_fraction <= 1 or self.warmup_epoch_fraction >= self.num_epoch:
            raise ValueError("warmup_epoch_fraction must be in [0, 1] and less than num_epoch")

    @property
    def data_parallel_size(self):
        return self.num_nodes * self.num_gpus_per_node // (self.tensor_model_parallel_size * self.pipeline_model_parallel_size)

    def _validate_parallelism(self):
        tp, pp, ep = self.tensor_model_parallel_size, self.pipeline_model_parallel_size, self.expert_model_parallel_size
        world_size = self.num_nodes * self.num_gpus_per_node
        if min(tp, pp, ep) < 1 or world_size % (tp * pp):
            raise ValueError("Positive TP and PP must divide the total GPU count")
        if 8 % tp or tp < 2:
            raise ValueError("TP must be 2, 4 or 8 for this sequence-parallel Inkling recipe")
        if 256 % ep or (world_size // pp) % ep:
            raise ValueError("EP must divide both 256 experts and the GPUs per pipeline stage (expert TP=1)")
        overrides = [n for n in (self.decoder_first_pipeline_num_layers, self.decoder_last_pipeline_num_layers) if n is not None]
        if overrides and (pp == 1 or any(n < 1 for n in overrides)):
            raise ValueError("Pipeline layer overrides require PP > 1 and positive layer counts")
        stages, layers = pp - len(overrides), 42 - sum(overrides)
        if (stages == 0 and layers != 0) or (stages > 0 and (layers < stages or layers % stages)):
            raise ValueError("42 layers must divide the remaining pipeline stages; for PP4 set --decoder-last-pipeline-num-layers 12")

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

    @property
    def eval_enabled(self):
        return self.mode == "train" and self.eval_every_n_epochs > 0 and self.eval_config is not None


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
    lora_args = f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --target-modules all-linear --experts-shared-outer-loras "
    sft_args = (
        "--rollout-function-path miles.rollout.inkling_sft.generate_rollout "
        "--data-source-path miles.rollout.inkling_sft_data_source.InklingSFTDataSource "
        f"--prompt-data {q(args.dataset)} --input-key text --metadata-key metadata "
        f"--rollout-shuffle --rollout-batch-size {args.global_batch_size} --global-batch-size {args.global_batch_size} --n-samples-per-prompt 1 "
        f"--num-epoch {args.num_epoch} --loss-type sft_loss --calculate-per-token-loss "
        "--disable-compute-advantages-and-returns --debug-train-only "
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} --pipeline-model-parallel-size {args.pipeline_model_parallel_size} --expert-model-parallel-size {args.expert_model_parallel_size} --expert-tensor-parallel-size 1 --context-parallel-size 1 --sequence-parallel --micro-batch-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --seq-length {args.max_length} "
    )
    for name in ("decoder_first_pipeline_num_layers", "decoder_last_pipeline_num_layers"):
        if (value := getattr(args, name)) is not None:
            perf_args += f"--{name.replace('_', '-')} {value} "
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --min-lr {args.min_lr} "
        # Megatron expresses warmup relative to the entire multi-epoch run.
        f"--lr-decay-style cosine --lr-warmup-init 0 --lr-warmup-fraction {args.warmup_epoch_fraction / args.num_epoch} "
        "--weight-decay 0.1 --clip-grad 1.0 "
    )
    misc_args = f"--distributed-timeout-minutes {args.distributed_timeout_minutes} --bf16 --moe-router-dtype fp32 --transformer-impl transformer_engine --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --no-bias-dropout-fusion --actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} --num-gpus-per-node {args.num_gpus_per_node} "
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    eval_args = ""
    if args.eval_enabled:
        eval_args = f"--inkling-eval-config {q(args.eval_config)} --inkling-eval-every-n-epochs {args.eval_every_n_epochs} --inkling-eval-image {q(args.image)} --inkling-eval-environment {q(args.environment)} "
        if args.eval_rollouts_per_env is not None:
            eval_args += f"--inkling-eval-rollouts-per-env {args.eval_rollouts_per_env} "
    if wandb_args:
        # The shared helper includes the API key in a printed command. Let W&B
        # read the inherited secret instead, and override its generated project.
        parts = shlex.split(wandb_args)
        index = parts.index("--wandb-key")
        del parts[index : index + 2]
        parts[parts.index("--wandb-project") + 1] = args.wandb_project
        wandb_args = shlex.join(parts)
    U.execute_train(
        train_args=f"{checkpoint_args} {lora_args} {sft_args} {perf_args} {optimizer_args} {misc_args} {eval_args}{wandb_args}",
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
    config = asdict(args)
    eval_env = ""
    if args.eval_enabled:
        from miles_plugins.inkling_eval.config import EvalConfig

        evaluation = EvalConfig.read(args.eval_config)
        config["_eval_config"] = evaluation.to_dict()
        eval_env = f"INKLING_EVAL_SECRET={shlex.quote(evaluation.modal_secret)} "
    command = shlex.join(
        [
            "modal",
            "run",
            "--detach",
            "--env",
            args.environment,
            str(U.repo_base_dir / "tools/modal_inkling_sft.py"),
            "--config-json",
            json.dumps(config),
        ]
    )
    U.exec_command_cpu(f"{eval_env}MODAL_PROFILE={shlex.quote(args.profile)} INKLING_MODAL_IMAGE={shlex.quote(args.image)} {command}")


if __name__ == "__main__":
    app()
