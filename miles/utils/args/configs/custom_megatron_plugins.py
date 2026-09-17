from miles.utils.args.schema import A, Arg, BaseConfig


class CustomMegatronPluginsConfig(BaseConfig):
    """
    Add custom Megatron plugins arguments.
    This is a placeholder for any additional arguments that might be needed.
    """

    freeze_indexer: A[bool, Arg()] = False
    custom_megatron_init_path: A[str | None, Arg()] = None
    custom_megatron_before_log_prob_hook_path: A[str | None, Arg()] = None
    custom_megatron_before_train_step_hook_path: A[str | None, Arg()] = None
    dsv4_impl: A[
        str,
        Arg(
            choices=["miles", "megatron"],
            help=(
                "Which DeepSeek-V4 attention implementation to train with. 'miles' is the plugin path "
                "and 'megatron' is Megatron's native dsv4_hybrid path."
            ),
        ),
    ] = "megatron"
