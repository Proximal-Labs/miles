from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


# wandb
class WandbConfig(BaseConfig):
    # wandb parameters
    use_wandb: Annotated[bool, A("--use-wandb", action="store_true", default=False)]
    wandb_mode: Annotated[
        str | None,
        A(
            "--wandb-mode",
            type=str,
            default=None,
            choices=["online", "offline", "disabled"],
            help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
        ),
    ]
    wandb_dir: Annotated[
        str | None,
        A(
            "--wandb-dir",
            type=str,
            default=None,
            help="Directory to store wandb logs. Default is ./wandb in current directory.",
        ),
    ]
    wandb_key: Annotated[str | None, A("--wandb-key", type=str, default=None)]
    wandb_host: Annotated[str | None, A("--wandb-host", type=str, default=None)]
    wandb_team: Annotated[str | None, A("--wandb-team", type=str, default=None)]
    wandb_group: Annotated[str | None, A("--wandb-group", type=str, default=None)]
    wandb_project: Annotated[str | None, A("--wandb-project", reset=True, type=str, default=None)]
    wandb_random_suffix: Annotated[
        bool,
        A(
            "--disable-wandb-random-suffix",
            action="store_false",
            dest="wandb_random_suffix",
            default=True,
            help=(
                "Whether to add a random suffix to the wandb run name. "
                "By default, we will add a random 6 length string with characters to the run name."
            ),
        ),
    ]
    wandb_always_use_train_step: Annotated[
        bool,
        A(
            "--wandb-always-use-train-step",
            action="store_true",
            default=False,
            help=(
                "Whether to always use train step as the step metric in wandb. "
                "If set, we will always use the train steps for wandb logging, "
                "otherwise, will use rollout step for most info other than train/*. "
            ),
        ),
    ]
    log_multi_turn: Annotated[
        bool,
        A(
            "--log-multi-turn",
            action="store_true",
            default=False,
            help="Whether to log information for multi-turn rollout.",
        ),
    ]
    log_passrate: Annotated[
        bool,
        A(
            "--log-passrate",
            action="store_true",
            default=False,
            help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
        ),
    ]
    log_reward_category: Annotated[
        str | None,
        A(
            "--log-reward-category",
            type=str,
            default=None,
            help=(
                "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                "Specify the key in the reward dict using this argument."
            ),
        ),
    ]
    log_correct_samples: Annotated[
        bool,
        A(
            "--log-correct-samples",
            action="store_true",
            default=False,
            help="Explicitly log metrics for correct samples.",
        ),
    ]
    wandb_run_id: Annotated[str | None, A("--wandb-run-id", type=str, default=None)]
