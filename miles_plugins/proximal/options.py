"""Translate one typed run contract into Miles's existing configuration seams."""

from argparse import ArgumentParser, Namespace

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.contracts import read_run_config

ROLLOUT = "miles_plugins.proximal.rollout.PlatformRolloutFn"
TRANSFER = "miles_plugins.proximal.weight_update.ModalVolumeTransfer"
SOURCE = "miles_plugins.proximal.data_source.PlatformTaskSource"
BUFFER = "miles_plugins.proximal.buffer.PlatformDataBuffer"


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--proximal-config", required=True)
    parser.add_argument("--proximal-yes-rollouts", action="store_true")
    parser.add_argument("--proximal-yes-publish", action="store_true")


def validate_args(args: Namespace) -> None:
    config = read_run_config(args.proximal_config)
    authorize_run(config, yes_rollouts=args.proximal_yes_rollouts, yes_publish=args.proximal_yes_publish)
    required = {
        "train_backend": "megatron",
        "use_rollout_logprobs": True,
        "fully_async": True,
        "rollout_external": True,
        "rollout_num_gpus": 0,
        "update_weights_interval": 1,
        "rollout_function_path": ROLLOUT,
        "custom_weight_transfer_protocol_path": TRANSFER,
        "data_source_path": SOURCE,
        "custom_async_data_buffer_path": BUFFER,
        "rollout_global_dataset": True,
        "rollout_submission_granularity": "group",
        "n_samples_per_prompt": config.research.group_size,
        "max_weight_staleness": config.research.max_policy_lag,
        "async_unused_samples_handler": config.research.unused_groups,
        "async_max_concurrent_samples": config.max_in_flight_samples,
        "rollout_temperature": config.research.sampling.temperature,
        "rollout_top_p": config.research.sampling.top_p,
        "rollout_top_k": config.research.sampling.top_k,
        "rollout_max_response_len": config.research.sampling.max_tokens,
        "rollout_max_context_len": config.research.sampling.max_sequence_tokens,
        "hf_checkpoint": str(config.tokenizer_path),
    }
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"Platform run requires --{name.replace('_', '-')}={expected!r}")
    if args.lora_rank <= 0 or args.lora_train_only or args.lora_dropout != 0:
        raise ValueError("Platform training requires a served LoRA with zero dropout")
    for name in (
        "colocate",
        "use_critic",
        "indep_dp",
        "use_fault_tolerance",
        "rollout_shuffle",
        "use_tis",
        "group_rm",
        "multi_lora",
        "debug_train_only",
        "debug_rollout_only",
        "debug_skip_weight_update",
        "partial_rollout",
        "rollout_sample_filter_path",
        "dynamic_sampling_filter_path",
        "eval_interval",
        "eval_num_gpus",
        "rollout_external_engine_addrs",
        "sglang_speculative_algorithm",
        "use_rollout_routing_replay",
        "use_rollout_indexer_replay",
        "custom_generate_function_path",
    ):
        if getattr(args, name, None):
            raise ValueError(f"Unsupported first-pass platform option: --{name.replace('_', '-')}")
    if not getattr(args, "megatron_to_hf_mode", None) == "bridge":
        raise ValueError("Platform adapter export requires --megatron-to-hf-mode bridge")
    if not config.tokenizer_path.is_dir():
        raise ValueError("Stage the pinned base/tokenizer checkpoint locally before starting")
    if args.rollout_batch_size > config.completed_group_capacity:
        raise ValueError("Completed-group capacity must fit a training batch")
