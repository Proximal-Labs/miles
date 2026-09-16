from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class RewardModelConfig(BaseConfig):
    rm_type: Annotated[str | None, A("--rm-type", type=str, default=None, help="Type of the reward model")]
    reward_key: Annotated[
        str | None,
        A(
            "--reward-key",
            type=str,
            default=None,
            help=(
                "Some reward model may return a dict instead of a value, "
                "this is the key to extract the reward value from the dict. "
            ),
        ),
    ]
    eval_reward_key: Annotated[
        str | None,
        A("--eval-reward-key", type=str, default=None, help="The eval variant for --reward-key"),
    ]
    group_rm: Annotated[
        bool, A("--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group.")
    ]
    rm_url: Annotated[
        str | None,
        A(
            "--rm-url",
            type=str,
            default=None,
            help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
        ),
    ]
    custom_rm_path: Annotated[
        str | None,
        A(
            "--custom-rm-path",
            type=str,
            default=None,
            help=(
                "Path to the custom reward model function. "
                "If set, we will use this function to calculate the reward instead of the default one. "
                "The function should have the signature `def custom_rm(args, sample) -> float`."
            ),
        ),
    ]
    custom_reward_post_process_path: Annotated[
        str | None,
        A(
            "--custom-reward-post-process-path",
            type=str,
            default=None,
            help=(
                "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
            ),
        ),
    ]
    custom_convert_samples_to_train_data_path: Annotated[
        str | None,
        A(
            "--custom-convert-samples-to-train-data-path",
            type=str,
            default=None,
            help=(
                "Path to a custom function that converts samples to training data. "
                "If set, this function will replace the default _convert_samples_to_train_data. "
                "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
            ),
        ),
    ]
