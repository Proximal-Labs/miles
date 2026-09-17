from miles.utils.args.schema import A, Arg, BaseConfig


class LoraConfig(BaseConfig):
    """Add LoRA-related arguments for Megatron backend."""

    sglang_lora_use_virtual_experts: A[
        bool,
        Arg(
            cli_name="--no-sglang-lora-use-virtual-experts",
            action="store_false",
            help="Serve MoE-expert LoRA through sglang's fused_moe_lora alignment path instead "
            "of the virtual-experts path.",
        ),
    ] = True

    lora_rank: A[int, Arg(help="LoRA rank. Set to 0 to disable LoRA (default: 0)")] = 0
    lora_alpha: A[int, Arg(help="LoRA alpha for scaling (default: 16)")] = 16
    lora_dropout: A[float, Arg(help="LoRA dropout rate (default: 0.0)")] = 0.0
    lora_type: A[
        str,
        Arg(
            choices=["lora", "canonical_lora"],
            help="LoRA variant to use: 'lora' (standard) or 'canonical_lora' (split Q/K/V) (default: lora)",
        ),
    ] = "lora"
    target_modules: A[
        str | None,
        Arg(
            help=(
                "Target modules for LoRA. Use 'all-linear' or comma-separated module names "
                "(e.g., 'q_proj,k_proj,v_proj,o_proj' for HF naming or 'linear_qkv,linear_proj' for Megatron naming)"
            )
        ),
    ] = None
    exclude_modules: A[str | None, Arg(help="Modules to exclude from LoRA (comma-separated)")] = None
    lora_adapter_path: A[str | None, Arg(help="Path to load pre-trained LoRA adapter weights (default: None)")] = None
    lora_sync_from_tensor: A[
        bool,
        Arg(help="Sync LoRA weights via tensor instead of file (more efficient)"),
    ] = False
    lora_base_cpu_backup: A[
        bool,
        Arg(
            help=(
                "LoRA + colocate: keep SGLang-side CPU mirror of base weights "
                "and skip per-step base sync. Trades host RAM for faster "
                "onload/offload. Ignored unless --colocate and LoRA are both on."
            )
        ),
    ] = False
    lora_train_only: A[
        bool,
        Arg(
            help=(
                "Train LoRA adapters in Megatron but keep rollout engines on the frozen "
                "base policy: SGLang LoRA serving and adapter weight sync are disabled "
                "(only the base weights are synced). For models without SGLang LoRA "
                "support (e.g. Inkling native LoRA)."
            )
        ),
    ] = False
    experts_shared_outer_loras: A[
        bool,
        Arg(
            help=(
                "Enable shared-outer grouped-expert LoRA (gate_up lora_A and "
                "down lora_B shared across experts, expert_dim=1). Matches SGLang "
                "PR #21466's experts_shared_outer_loras=True serving contract."
            )
        ),
    ] = False
    multi_lora_n_adapters: A[
        int,
        Arg(
            help="Maximum number of concurrent adapter slots for multi-LoRA. Set to 0 to disable multi-LoRA (default: 0)"
        ),
    ] = 0
    multi_lora_adapters: A[
        list[list[str]],
        Arg(type_parser=str, cli_name="--multi-lora-adapter", nargs=2, action="append"),
    ] = []
    multi_lora_idle_poll_s: A[
        float,
        Arg(
            help="When no adapter is RUNNING, the trainer polls for new registrations every this many seconds (default: 5.0)"
        ),
    ] = 5.0
    multi_lora_http_server_path: A[
        str | None,
        Arg(
            help=(
                "Dotted path to a MultiLoRAHTTPServer subclass to use for the multi-LoRA "
                "controller's HTTP server (default: MultiLoRAHTTPServer)"
            )
        ),
    ] = None
    multi_lora_backend_path: A[
        str | None,
        Arg(
            help=(
                "Dotted path to a MultiLoRABackend subclass for the multi-LoRA controller, "
                "e.g. to add custom adapter validation via validate_adapter (default: MultiLoRABackend)"
            )
        ),
    ] = None
    multi_lora_api_port: A[
        int,
        Arg(help="Port for the multi-LoRA controller's control-plane API, served from the head node (default: 8068)"),
    ] = 8068
    multi_lora_service_mode: A[
        bool,
        Arg(
            cli_name="--multi-lora-disable-service-mode",
            action="store_false",
            help="Disable service mode. By default, the trainer waits indefinitely for new adapters. With this flag, it exits after all adapters have been processed.",
        ),
    ] = True
    multi_lora_max_adapter_global_batch_size: A[
        int | None,
        Arg(
            help=(
                "Registration-time upper bound on an adapter's samples per optimizer "
                "step (rollout_batch_size x n_samples_per_prompt). Defaults to 4x "
                "--global-batch-size."
            )
        ),
    ] = None
    multi_lora_max_coalesce_wait_s: A[
        float,
        Arg(
            help=(
                "Maximum time ready groups wait for the batch to fill toward "
                "--global-batch-size before training starts on what is ready (default: 0.5)."
            )
        ),
    ] = 0.5
    multi_lora_max_empty_wait_s: A[
        float,
        Arg(
            help=(
                "How long a generate call waits for the first poppable group before "
                "failing with an empty-batch timeout (default: 30)."
            )
        ),
    ] = 30.0
