from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class AlgoConfig(BaseConfig):
    ref_load: Annotated[
        str | None,
        A(
            "--ref-load",
            type=str,
            default=None,
            help=(
                "The checkpoint for reference model. "
                "When --load is not set, this will be used as the initial checkpoint for training. "
            ),
        ),
    ]
    ref_ckpt_step: Annotated[
        int | None,
        A("--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "),
    ]
    load: Annotated[str | None, A("--load", reset=True, type=str, default=None)]
    save: Annotated[str | None, A("--save", reset=True, type=str, default=None)]
    save_interval: Annotated[int | None, A("--save-interval", reset=True, type=int, default=None)]
    async_save: Annotated[bool, A("--async-save", reset=True, action="store_true")]
    no_save_optim: Annotated[
        bool,
        A(
            "--no-save-optim",
            reset=True,
            action="store_true",
            default=False,
            help=(
                "If set, do not save the optimizer state when saving checkpoints. "
                "This reduces checkpoint size but disables training resumption from the saved checkpoint."
            ),
        ),
    ]
    save_hf: Annotated[
        str | None,
        A(
            "--save-hf",
            type=str,
            default=None,
            help=(
                "Path to save the model in HuggingFace format when using Megatron backend. "
                "The model will be saved to `save_hf.format(rollout_id)`. "
            ),
        ),
    ]
    save_trigger_sentinel: Annotated[
        str | None,
        A(
            "--save-trigger-sentinel",
            type=str,
            default=None,
            help=(
                "Path to a sentinel file for externally-triggered checkpoint saving. If the file "
                "exists at an iteration's save point, a checkpoint is saved and the file is removed."
            ),
        ),
    ]
    custom_megatron_post_save_hook_path: Annotated[
        str | None,
        A(
            "--custom-megatron-post-save-hook-path",
            type=str,
            default=None,
            help=(
                "Path to a custom function invoked on rank 0 after every checkpoint save. "
                "Signature: def hook(args, rollout_id: int, checkpoint_dir: str, "
                "hf_checkpoint_dir: str | None) -> None."
            ),
        ),
    ]
    seed: Annotated[int, A("--seed", reset=True, type=int, default=1234)]
    clip_grad: Annotated[float, A("--clip-grad", reset=True, type=float, default=1.0)]
    calculate_per_token_loss: Annotated[bool, A("--calculate-per-token-loss", reset=True, action="store_true")]
    lr: Annotated[float, A("--lr", reset=True, type=float, default=1e-6)]

    num_critic_only_steps: Annotated[
        int,
        A(
            "--num-critic-only-steps",
            type=int,
            default=0,
            help="Number of initial rollout steps where only the critic trains (value-function warmup) "
            "while the actor stays frozen. Only takes effect when --advantage-estimator is ppo.",
        ),
    ]
    critic_load: Annotated[
        str | None, A("--critic-load", type=str, default=None, help="The checkpoint for critic model.")
    ]
    critic_save: Annotated[
        str | None,
        A(
            "--critic-save",
            type=str,
            default=None,
            help="Where to save critic checkpoints. If not set, it defaults to the --save path with a "
            "'_critic' suffix appended, e.g. --save /ckpts/run1 saves the critic to /ckpts/run1_critic.",
        ),
    ]
    critic_lr: Annotated[float | None, A("--critic-lr", type=float, default=None, help="The lr for critic model")]
    critic_lr_warmup_iters: Annotated[
        int,
        A(
            "--critic-lr-warmup-iters",
            type=int,
            default=0,
            help="number of iterations to linearly warmup for critic model.",
        ),
    ]

    eps_clip: Annotated[float, A("--eps-clip", type=float, default=0.2, help="PPO clip range")]
    eps_clip_high: Annotated[float | None, A("--eps-clip-high", type=float, default=None, help="PPO clip upper range")]
    eps_clip_c: Annotated[
        float | None,
        A(
            "--eps-clip-c",
            type=float,
            default=None,
            help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
        ),
    ]
    value_clip: Annotated[float, A("--value-clip", type=float, default=0.2, help="the clip for value loss")]
    kl_coef: Annotated[
        float,
        A(
            "--kl-coef",
            type=float,
            default=0.00,
            help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
        ),
    ]
    loss_type: Annotated[
        str,
        A(
            "--loss-type",
            type=str,
            choices=["policy_loss", "sft_loss", "custom_loss"],
            default="policy_loss",
            help=(
                "Choose loss type, currently support ppo policy_loss or sft_loss, "
                "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
            ),
        ),
    ]
    custom_loss_function_path: Annotated[
        str | None,
        A(
            "--custom-loss-function-path",
            type=str,
            default=None,
            help=(
                "Path to the custom loss function, if the loss_type is `custom_loss`, "
                "we will use this function to calculate the loss. "
            ),
        ),
    ]
    kl_loss_type: Annotated[
        str,
        A(
            "--kl-loss-type",
            type=str,
            choices=["k1", "k2", "k3", "low_var_kl"],
            default="k1",
            help="Choose KL loss type: kl, k2, k3, low_var_kl",
        ),
    ]
    advantage_estimator: Annotated[
        str,
        A(
            "--advantage-estimator",
            type=str,
            choices=["grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "ppo"],
            default="grpo",
            help=(
                "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
            ),
        ),
    ]
    compute_advantages_and_returns: Annotated[
        bool,
        A(
            "--disable-compute-advantages-and-returns",
            action="store_false",
            dest="compute_advantages_and_returns",
            help=(
                "Whether to disable computing advantages and returns. "
                "If set, we will not compute the advantages and returns, "
                "This is useful for sft or custom loss function."
            ),
        ),
    ]
    use_kl_loss: Annotated[
        bool, A("--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO")
    ]
    kl_loss_coef: Annotated[
        float,
        A(
            "--kl-loss-coef",
            type=float,
            default=0.0,
            help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
        ),
    ]
    use_unbiased_kl: Annotated[
        bool,
        A("--use-unbiased-kl", action="store_true", default=False, help="Whether to enable unbiased KL estimation."),
    ]
    ref_update_interval: Annotated[
        int | None,
        A(
            "--ref-update-interval",
            type=int,
            default=None,
            help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
        ),
    ]
    entropy_coef: Annotated[float, A("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")]
    gamma: Annotated[float, A("--gamma", type=float, default=1.0, help="PPO GAE gamma")]
    lambd: Annotated[float, A("--lambd", type=float, default=1.0, help="PPO GAE lambd")]
    normalize_advantages: Annotated[bool, A("--normalize-advantages", action="store_true", default=False)]
    grpo_std_normalization: Annotated[
        bool,
        A(
            "--disable-grpo-std-normalization",
            action="store_false",
            dest="grpo_std_normalization",
            help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
        ),
    ]
    rewards_normalization: Annotated[
        bool,
        A(
            "--disable-rewards-normalization",
            action="store_false",
            dest="rewards_normalization",
            help="Disable rewards normalization",
        ),
    ]
    use_rollout_entropy: Annotated[
        bool,
        A(
            "--use-rollout-entropy",
            action="store_true",
            default=False,
            help=(
                "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                "This is useful for doing special loss mask."
            ),
        ),
    ]
    observe_training_entropy: Annotated[
        bool,
        A(
            "--observe-training-entropy",
            action="store_true",
            default=False,
            help=(
                "Compute training entropy as a logged metric even when --entropy-coef is 0. "
                "When the coefficient is 0, the observed entropy is detached and does not affect backward."
            ),
        ),
    ]
    get_mismatch_metrics: Annotated[
        bool,
        A("--get-mismatch-metrics", action="store_true", default=False, help="Whether to calculate the mismatch metrics."),
    ]
    reset_optimizer_states: Annotated[
        bool,
        A(
            "--reset-optimizer-states",
            action="store_true",
            default=False,
            help=(
                "Whether to reset optimizer states after each rollout. "
                "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
            ),
        ),
    ]
    use_rollout_logprobs: Annotated[
        bool,
        A(
            "--use-rollout-logprobs",
            action="store_true",
            default=False,
            help=(
                "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                "If not set, we will use the logprobs from the actor model."
            ),
        ),
    ]
    skip_actor_forward_only: Annotated[
        bool,
        A(
            "--skip-actor-forward-only",
            action="store_true",
            default=False,
            help=(
                "Skip the standalone Megatron actor forward-only pass. With --use-rollout-logprobs, "
                "those log-probs remain the old-policy baseline; otherwise detached training log-probs "
                "are reused and the actor importance log-ratio is exactly 0. This requires a single "
                "optimizer step. The skipped pass's rollout/log_probs metric is not emitted."
            ),
        ),
    ]
    # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
    use_tis: Annotated[
        bool,
        A(
            "--use-tis",
            action="store_true",
            default=False,
            help="Enable TIS from https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33.",
        ),
    ]
    tis_clip: Annotated[
        float,
        A(
            "--tis-clip",
            type=float,
            default=2.0,
            help="Clipping threshold C for importance sampling ratios to control variance.",
        ),
    ]
    tis_clip_low: Annotated[
        float,
        A(
            "--tis-clip-low",
            type=float,
            default=0,
            help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
        ),
    ]
    custom_tis_function_path: Annotated[
        str | None,
        A(
            "--custom-tis-function-path",
            type=str,
            default=None,
            help="Path to the custom TIS/RS function (e.g., examples/infra_features/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
        ),
    ]
    custom_pg_loss_reducer_function_path: Annotated[
        str | None,
        A(
            "--custom-pg-loss-reducer-function-path",
            type=str,
            default=None,
            help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/experimental/DrGRPO/custom_reducer.py:get_pg_loss_reducer).",
        ),
    ]

    use_routing_replay: Annotated[
        bool,
        A(
            "--use-routing-replay",
            action="store_true",
            default=False,
            help="The routing replay technique from https://arxiv.org/abs/2507.18071",
        ),
    ]
    use_rollout_routing_replay: Annotated[
        bool,
        A(
            "--use-rollout-routing-replay",
            action="store_true",
            default=False,
            help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370 (R3): "
            "replay the rollout's MoE routing in training. MoE-only; the GLM-5 launchers pass it "
            "explicitly.",
        ),
    ]
    use_indexer_replay: Annotated[
        bool,
        A(
            "--use-indexer-replay",
            action="store_true",
            default=False,
            help="Replay indexer topk decisions for layers with indexers.",
        ),
    ]
    use_rollout_indexer_replay: Annotated[
        bool,
        A(
            "--use-rollout-indexer-replay",
            action="store_true",
            default=False,
            help="Replay indexer topk from rollout during training.",
        ),
    ]
    use_opsm: Annotated[
        bool,
        A(
            "--use-opsm",
            action="store_true",
            default=False,
            help="Whether to enable Off-Policy Sequence Masking (OPSM).",
        ),
    ]
    opsm_delta: Annotated[
        float,
        A("--opsm-delta", type=float, default=1e-4, help="The threshold for Off-Policy Sequence Masking (OPSM)."),
    ]
