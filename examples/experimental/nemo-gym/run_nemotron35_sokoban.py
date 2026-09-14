"""Nemotron 3.5 Lightning Sokoban GRPO with separate rollout and training nodes.

Requires two joined eight-GPU Ray nodes, matching model and dataset paths on both,
and the NeMo Gym Sokoban verifier. The model starts from original HF weights.

Args:
    --config: JSON object containing the ScriptArgs fields below.

Example:
    python examples/experimental/nemo-gym/run_nemotron35_sokoban.py \
        --config /scratch/run/launcher_config.json
"""

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from tap import Tap

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    num_nodes: int = 2
    num_gpus_per_node: int = 8
    model_dir: str = "/root/models"
    model_name: str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    data_dir: str = "/root/datasets"
    data_file: str = "sokoban_train.jsonl"
    megatron_path: str = "/root/Megatron-LM"
    verifier_url: str = "http://127.0.0.1:8210"
    hf_cache_dir: str = "/scratch/hf"
    learning_rate: float = 3e-7
    rollout_batch_size: int = 8
    group_size: int = 16
    global_batch_size: int = 128
    num_rollout: int = 1000
    save_interval: int = 200
    response_length: int = 65536
    context_length: int = 81920
    max_tokens_per_gpu: int = 16384
    max_weight_staleness: int = 2
    async_max_concurrent_samples: int = 128
    async_data_buffer_capacity_factor: float = 2.0
    wandb_team: str = "radixarkai"
    wandb_project: str = "nemotron35-sokoban"

    def __post_init__(self) -> None:
        assert self.num_nodes == 2 and self.num_gpus_per_node == 8
        assert self.global_batch_size == self.rollout_batch_size * self.group_size
        assert self.response_length < self.context_length

    @property
    def run_name(self) -> str:
        return f"{self.run_id}-nemotron35-lightning-sokoban-async-2n-bs{self.global_batch_size}-g{self.group_size}"


def _flags(values: dict[str, object]) -> str:
    tokens = []
    for key, value in values.items():
        if value is False or value is None:
            continue
        tokens.append("--" + key.replace("_", "-"))
        if value is not True:
            tokens.append(str(value))
    return shlex.join(tokens)


def _wandb_args(args: ScriptArgs) -> str:
    # Credentials come from W&B's credential store, never the command line.
    defaults = shlex.split(U.get_default_wandb_args(__file__, run_id=args.run_name))
    sanitized = []
    index = 0
    while index < len(defaults):
        token = defaults[index]
        if token in {"--wandb-key", "--wandb-project", "--wandb-group"}:
            index += 2
        else:
            sanitized.append(token)
            index += 1
    return (
        shlex.join(sanitized)
        + " "
        + _flags(
            {
                "use_wandb": True,
                "wandb_team": args.wandb_team,
                "wandb_project": args.wandb_project,
                "wandb_group": args.run_name,
                "wandb_dir": str(Path(args.output_dir) / "wandb"),
                "disable_wandb_random_suffix": True,
            }
        )
    )


def _learning_args(args: ScriptArgs) -> str:
    performance_args = _flags(
        {
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 2,
            "expert_tensor_parallel_size": 1,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "use_dynamic_batch_size": True,
            "max_tokens_per_gpu": args.max_tokens_per_gpu,
            "log_probs_chunk_size": 128,
            "seq_length": args.context_length,
            "optimizer_cpu_offload": True,
            "overlap_cpu_optimizer_d2h_h2d": True,
            "use_precision_aware_optimizer": True,
        }
    )
    algorithm_args = _flags(
        {
            "advantage_estimator": "grpo",
            "use_kl_loss": True,
            "kl_loss_coef": 0,
            "kl_loss_type": "low_var_kl",
            "entropy_coef": 0,
            "eps_clip": 0.2,
            "eps_clip_high": 0.28,
            "fully_async": True,
            "use_rollout_logprobs": True,
            "pause_generation_mode": "retract",
            "max_weight_staleness": args.max_weight_staleness,
            "async_unused_samples_handler": "retry",
            "async_max_concurrent_samples": args.async_max_concurrent_samples,
            "async_data_buffer_capacity_factor": args.async_data_buffer_capacity_factor,
        }
    )
    optimizer_args = _flags(
        {
            "optimizer": "adam",
            "lr": args.learning_rate,
            "lr_decay_style": "constant",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
        }
    )
    return " ".join([performance_args, algorithm_args, optimizer_args])


def execute(args: ScriptArgs) -> None:
    checkpoint = str(Path(args.model_dir) / args.model_name)
    checkpoint_args = _flags(
        {
            "hf_checkpoint": checkpoint,
            "ref_load": checkpoint,
            "save": str(Path(args.output_dir) / "checkpoints"),
            "save_interval": args.save_interval,
            "megatron_to_hf_mode": "bridge",
            "no_load_optim": True,
            "no_load_rng": True,
            "finetune": True,
        }
    )
    rollout_args = _flags(
        {
            "prompt_data": str(Path(args.data_dir) / args.data_file),
            "input_key": "prompt",
            "label_key": "label",
            "apply_chat_template": True,
            "rollout_shuffle": True,
            "rm_type": "deepscaler",
            "custom_rm_path": "sokoban_reward.reward_func",
            "num_rollout": args.num_rollout,
            "rollout_batch_size": args.rollout_batch_size,
            "n_samples_per_prompt": args.group_size,
            "global_batch_size": args.global_batch_size,
            "rollout_max_response_len": args.response_length,
            "rollout_temperature": 1,
            "balance_data": True,
        }
    )
    serving_args = _flags(
        {
            "rollout_num_gpus_per_engine": 1,
            "sglang_mem_fraction_static": 0.7,
            "sglang_context_length": args.context_length,
            "use_rollout_routing_replay": True,
            "sglang_router_policy": "round_robin",
            "sglang_enable_metrics": True,
        }
    )
    miscellaneous_args = _flags(
        {
            "attention_dropout": 0,
            "hidden_dropout": 0,
            "accumulate_allreduce_grads_in_fp32": True,
            "attention_softmax_in_fp32": True,
            "attention_backend": "auto",
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": args.num_gpus_per_node,
            "num_gpus_per_node": args.num_gpus_per_node,
            "rollout_num_gpus": args.num_gpus_per_node,
            "mtp_loss_scaling_factor": 0,
            "custom_megatron_before_train_step_hook_path": "sokoban_training_checks.before_train_step",
            "dump_details": str(Path(args.output_dir) / "traces"),
            "use_miles_dashboard": True,
            "observe_training_entropy": True,
            "use_rollout_entropy": True,
            "use_prometheus": True,
            "prometheus_port": 9090,
            "dashboard_forward_prometheus": True,
        }
    )
    U.execute_train(
        train_args=" ".join(
            [
                checkpoint_args,
                rollout_args,
                _learning_args(args),
                serving_args,
                miscellaneous_args,
                _wandb_args(args),
            ]
        ),
        config=args,
        train_script="train_async.py",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="nemotron-3-nano-30b-a3b",
        megatron_path=args.megatron_path,
        extra_env_vars={
            "MILES_NEMOTRONH_KEEP_MTP": "",
            "NEMO_GYM_SOKOBAN_URL": args.verifier_url,
            "HF_HOME": args.hf_cache_dir,
            "HUGGINGFACE_HUB_CACHE": str(Path(args.hf_cache_dir) / "hub"),
            "PYTHONPATH": str(Path(U.repo_base_dir) / "examples/experimental/nemo-gym"),
        },
    )


class _CLI(Tap):
    config: Path


def main() -> None:
    cli = _CLI().parse_args()
    execute(ScriptArgs(**json.loads(cli.config.read_text())))


if __name__ == "__main__":
    main()
