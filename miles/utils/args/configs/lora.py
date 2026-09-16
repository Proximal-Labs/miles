from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class LoraConfig(BaseConfig):
    """Add LoRA-related arguments for Megatron backend."""

    lora_rank: Annotated[
        int,
        A("--lora-rank", type=int, default=0, help="LoRA rank. Set to 0 to disable LoRA (default: 0)"),
    ]
    lora_alpha: Annotated[
        int, A("--lora-alpha", type=int, default=16, help="LoRA alpha for scaling (default: 16)")
    ]
    lora_dropout: Annotated[
        float, A("--lora-dropout", type=float, default=0.0, help="LoRA dropout rate (default: 0.0)")
    ]
    lora_type: Annotated[
        str,
        A(
            "--lora-type",
            type=str,
            default="lora",
            choices=["lora", "canonical_lora"],
            help="LoRA variant to use: 'lora' (standard) or 'canonical_lora' (split Q/K/V) (default: lora)",
        ),
    ]
    target_modules: Annotated[
        str | None,
        A(
            "--target-modules",
            type=str,
            default=None,
            help="Target modules for LoRA. Use 'all-linear' or comma-separated module names "
            "(e.g., 'q_proj,k_proj,v_proj,o_proj' for HF naming or 'linear_qkv,linear_proj' for Megatron naming)",
        ),
    ]
    exclude_modules: Annotated[
        str | None,
        A("--exclude-modules", type=str, default=None, help="Modules to exclude from LoRA (comma-separated)"),
    ]
    lora_adapter_path: Annotated[
        str | None,
        A("--lora-adapter-path", type=str, default=None, help="Path to load pre-trained LoRA adapter weights (default: None)"),
    ]
    lora_sync_from_tensor: Annotated[
        bool,
        A(
            "--lora-sync-from-tensor",
            action="store_true",
            default=False,
            help="Sync LoRA weights via tensor instead of file (more efficient)",
        ),
    ]
    lora_base_cpu_backup: Annotated[
        bool,
        A(
            "--lora-base-cpu-backup",
            action="store_true",
            default=False,
            help=(
                "LoRA + colocate: keep SGLang-side CPU mirror of base weights "
                "and skip per-step base sync. Trades host RAM for faster "
                "onload/offload. Ignored unless --colocate and LoRA are both on."
            ),
        ),
    ]
    lora_train_only: Annotated[
        bool,
        A(
            "--lora-train-only",
            action="store_true",
            default=False,
            help=(
                "Train LoRA adapters in Megatron but keep rollout engines on the frozen "
                "base policy: SGLang LoRA serving and adapter weight sync are disabled "
                "(only the base weights are synced). For models without SGLang LoRA "
                "support (e.g. Inkling native LoRA)."
            ),
        ),
    ]
    experts_shared_outer_loras: Annotated[
        bool,
        A(
            "--experts-shared-outer-loras",
            action="store_true",
            default=False,
            help="Enable shared-outer grouped-expert LoRA (gate_up lora_A and "
            "down lora_B shared across experts, expert_dim=1). Matches SGLang "
            "PR #21466's experts_shared_outer_loras=True serving contract.",
        ),
    ]
    multi_lora_n_adapters: Annotated[
        int,
        A(
            "--multi-lora-n-adapters",
            type=int,
            default=0,
            help="Maximum number of concurrent adapter slots for multi-LoRA. Set to 0 to disable multi-LoRA (default: 0)",
        ),
    ]
    multi_lora_adapters: Annotated[
        list[list[str]],
        A("--multi-lora-adapter", nargs=2, action="append", type=str, dest="multi_lora_adapters", default=[]),
    ]
    multi_lora_idle_poll_s: Annotated[
        float,
        A(
            "--multi-lora-idle-poll-s",
            type=float,
            default=5.0,
            help="When no adapter is RUNNING, the trainer polls for new registrations every this many seconds (default: 5.0)",
        ),
    ]
    multi_lora_http_server_path: Annotated[
        str | None,
        A(
            "--multi-lora-http-server-path",
            type=str,
            default=None,
            help=(
                "Dotted path to a MultiLoRAHTTPServer subclass to use for the multi-LoRA "
                "controller's HTTP server (default: MultiLoRAHTTPServer)"
            ),
        ),
    ]
    multi_lora_backend_path: Annotated[
        str | None,
        A(
            "--multi-lora-backend-path",
            type=str,
            default=None,
            help=(
                "Dotted path to a MultiLoRABackend subclass for the multi-LoRA controller, "
                "e.g. to add custom adapter validation via validate_adapter (default: MultiLoRABackend)"
            ),
        ),
    ]
    multi_lora_api_port: Annotated[
        int,
        A(
            "--multi-lora-api-port",
            type=int,
            default=8068,
            help="Port for the multi-LoRA controller's control-plane API, served from the head node (default: 8068)",
        ),
    ]
    multi_lora_service_mode: Annotated[
        bool,
        A(
            "--multi-lora-disable-service-mode",
            action="store_false",
            dest="multi_lora_service_mode",
            help="Disable service mode. By default, the trainer waits indefinitely for new adapters. With this flag, it exits after all adapters have been processed.",
        ),
    ]
    multi_lora_max_adapter_global_batch_size: Annotated[
        int | None,
        A(
            "--multi-lora-max-adapter-global-batch-size",
            type=int,
            default=None,
            help=(
                "Registration-time upper bound on an adapter's samples per optimizer "
                "step (rollout_batch_size x n_samples_per_prompt). Defaults to 4x "
                "--global-batch-size."
            ),
        ),
    ]
    multi_lora_max_coalesce_wait_s: Annotated[
        float,
        A(
            "--multi-lora-max-coalesce-wait-s",
            type=float,
            default=0.5,
            help=(
                "Maximum time ready groups wait for the batch to fill toward "
                "--global-batch-size before training starts on what is ready (default: 0.5)."
            ),
        ),
    ]
    multi_lora_max_empty_wait_s: Annotated[
        float,
        A(
            "--multi-lora-max-empty-wait-s",
            type=float,
            default=30.0,
            help=(
                "How long a generate call waits for the first poppable group before "
                "failing with an empty-batch timeout (default: 30)."
            ),
        ),
    ]
