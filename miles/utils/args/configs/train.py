import json
from typing import Annotated, Any

from miles.utils.args.schema import A, BaseConfig


class TrainConfig(BaseConfig):
    train_backend: Annotated[str, A(
        "--train-backend", type=str, choices=["megatron", "fsdp"], default="megatron",
        help="The backend for training.",
    )]
    qkv_format: Annotated[str, A(
        "--qkv-format", type=str, choices=["thd", "bshd"], default="thd", help="The qkv layout.",
    )]
    linear_attention_backend: Annotated[str, A(
        "--linear-attention-backend", type=str, choices=["fla", "flashqla"], default="fla",
        help=(
            "Backend for Qwen GDN linear-attention layers. "
            "'fla' (flash-linear-attention) is portable and runs on any supported GPU. "
            "'flashqla' (FlashQLA) requires NVIDIA SM90 (Hopper) or newer, CUDA 12.8+, and PyTorch 2.8+."
        ),
    )]
    miles_dsa_topk_backend: Annotated[str, A(
        "--miles-dsa-topk-backend", type=str, choices=["torch", "flashinfer"], default="torch",
        help="Top-k backend for Miles DSA indexer.",
    )]
    true_on_policy_mode: Annotated[bool, A(
        "--true-on-policy-mode", action="store_true", default=False, help="Whether to enable true-on-policy mode.",
    )]
    recompute_logprobs_via_prefill: Annotated[bool, A(
        "--recompute-logprobs-via-prefill", action="store_true", default=False,
        help=(
            "Recompute rollout logprobs via SGLang prefill instead of decode kernels. "
            "Only needed for models whose prefill and decode paths are not numerically identical."
        ),
    )]
    train_env_vars: Annotated[Any, A(
        "--train-env-vars", type=json.loads, default="{}",
        help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
    )]
    train_memory_margin_bytes: Annotated[int, A(
        "--train-memory-margin-bytes", type=int, default=1024**3,
        help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
    )]
    debug_skip_weight_update: Annotated[bool, A(
        "--debug-skip-weight-update", action="store_true", default=False,
        help=(
            "Debug-only: preserve the train/rollout offload-onload schedule, "
            "but skip the actual actor-to-rollout weight update."
        ),
    )]
    debug_disable_optimizer: Annotated[bool, A(
        "--debug-disable-optimizer", action="store_true", default=False,
        help=(
            "Debug-only: do not initialize the Megatron optimizer or LR scheduler. "
            "Training still runs rollout, log-prob forward, and actor forward/backward, "
            "but skips optimizer state allocation and optimizer updates."
        ),
    )]
    rematerialize_param_from_master_weight: Annotated[bool, A(
        "--rematerialize-param-from-master-weight", action="store_true",
        help=(
            "Colocate CPU memory optimization. Drop the actor's parameter weight backup "
            "during inference, and rebuild it from the optimizer's master weights on the "
            "next train step. Reduces peak CPU memory by 2*param per rank (bf16 training). "
            "Works with both the GPU optimizer and the CPU optimizer, but is not compatible "
            "with --use-precision-aware-optimizer on GPU. ref/teacher tags keep their "
            "backups. Recommended for Grace GPU colocate training."
        ),
    )]
    check_rematerialize_param_from_master_weight: Annotated[bool, A(
        "--check-rematerialize-param-from-master-weight", action="store_true",
        help="Debug: SHA256-verify the first two rematerialize cycles are bit-identical.",
    )]
    megatron_to_hf_mode: Annotated[str, A(
        "--megatron-to-hf-mode", choices=["raw", "bridge"], default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )]
    dsa_attention_backend: Annotated[str, A(
        "--dsa-attention-backend", choices=["megatron", "tilelang"], default="tilelang",
        help=(
            "DSA sparse-MLA kernel backend for GLM (glm_moe_dsa) under --megatron-to-hf-mode bridge. "
            "'tilelang' (default) uses the fused TileLang kernels (SparseMLA + lighting_indexer, vendored from slime) for "
            "rollout<->train numerical parity; 'megatron' uses the portable unfused megatron-core "
            "kernels. 'tilelang' requires --qkv-format thd and the optional tilelang dep, and is "
            "training/forward-only (no KV cache, cannot serve inference). Both support GLM-5.1 and "
            "GLM-5.2, full or LoRA. No effect on non-DSA models or the 'raw' path."
        ),
    )]
    extra_high_precision_layers_hf: Annotated[list[str] | tuple[str, ...], A(
        "--extra-high-precision-layers-hf", type=str, nargs="*", default=(),
        help=("Extra substrings for HF weight names to skip quantization " "(e.g. .kv_b_proj.)."),
    )]
    extra_high_precision_layers_megatron: Annotated[list[str] | tuple[str, ...], A(
        "--extra-high-precision-layers-megatron", type=str, nargs="*", default=(),
        help=(
            "Extra substrings for Megatron weight names to skip quantization in Megatron-to-HF paths "
            "(e.g. .linear_kv_up_proj.)."
        ),
    )]
    custom_model_provider_path: Annotated[str | None, A(
        "--custom-model-provider-path", type=str, default=None,
        help=(
            "Path to a custom model provider function. "
            "If set, we will use this function instead of the default model provider. "
            "The function should have the signature "
            "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
            "Example: 'my_module.my_model_provider'."
        ),
    )]
    recompute_loss_function: Annotated[bool, A(
        "--recompute-loss-function", action="store_true",
        help="Whether to enable recompute loss function to save memory during training.",
    )]
    log_probs_chunk_size: Annotated[int, A(
        "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory",
    )]
    indep_dp: Annotated[bool, A(
        "--indep-dp", action="store_true", default=False,
        help="Launch each DP replica as an independent Megatron instance instead of using Megatron-internal data parallelism.",
    )]
    delay_split_train_data_by_dp: Annotated[bool, A(
        "--delay-split-train-data-by-dp", action="store_true", default=False,
        help="Split the rollout batch across DP ranks on the training side instead of the rollout side, "
        "using the training side's own DP size.",
    )]
    allgather_cp: Annotated[bool, A("--allgather-cp", action="store_true", default=False)]
    low_memory_resume: Annotated[bool, A(
        "--low-memory-resume", reset=True, action="store_true", default=False,
        help=("Allocate optimizer states on CPU during checkpoint loading to prevent GPU OOM on memory spike. "),
    )]
    mfu_peak_tflops: Annotated[float | None, A(
        "--mfu-peak-tflops", type=float, default=None,
        help=(
            "Peak dense BF16 TFLOP/s of one training GPU — the denominator of perf/actor_train_mfu. "
            "Defaults to a built-in table keyed on the device name; set this for a device the table "
            "does not know, or to report MFU against another precision's peak. With neither available "
            "the MFU metric is not logged."
        ),
    )]
