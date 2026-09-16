import argparse
from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class CustomMegatronPluginsConfig(BaseConfig):
    """
    Add custom Megatron plugins arguments.
    This is a placeholder for any additional arguments that might be needed.
    """

    freeze_indexer: Annotated[bool, A("--freeze-indexer", action="store_true", default=False)]
    custom_megatron_init_path: Annotated[str | None, A("--custom-megatron-init-path", type=str, default=None)]
    custom_megatron_before_log_prob_hook_path: Annotated[
        str | None, A("--custom-megatron-before-log-prob-hook-path", type=str, default=None)
    ]
    custom_megatron_before_train_step_hook_path: Annotated[
        str | None, A("--custom-megatron-before-train-step-hook-path", type=str, default=None)
    ]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        from miles_plugins.models.deepseek_v4.arguments import add_dsv4_arguments

        add_dsv4_arguments(parser)
        super().add_arguments(parser=parser)
