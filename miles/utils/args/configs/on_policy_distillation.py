from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class OnPolicyDistillationConfig(BaseConfig):
    """Add on-policy distillation (OPD) related arguments.

    OPD is orthogonal to advantage estimators and can be applied on top of
    any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
    """

    use_opd: Annotated[
        bool,
        A(
            "--use-opd",
            action="store_true",
            default=False,
            help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
        ),
    ]
    opd_type: Annotated[
        str | None,
        A(
            "--opd-type",
            type=str,
            choices=["sglang", "megatron"],
            default=None,
            help=(
                "Type of on-policy distillation. "
                "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
            ),
        ),
    ]
    opd_kl_coef: Annotated[
        float,
        A(
            "--opd-kl-coef",
            type=float,
            default=1.0,
            help="On-policy distillation KL penalty coefficient. Default is 1.0.",
        ),
    ]
    opd_log_prob_top_k: Annotated[
        int,
        A(
            "--opd-log-prob-top-k",
            type=int,
            default=0,
            help=(
                "Number of top-k tokens to use for the re-think OPD token-level reward. "
                "Set to 0 to use sampled-token OPD."
            ),
        ),
    ]
    opd_top_k_strategy: Annotated[
        str,
        A(
            "--opd-top-k-strategy",
            type=str,
            choices=["only-student", "only-teacher", "intersection", "union", "xor"],
            default="only-student",
            help="Token set strategy for top-k OPD.",
        ),
    ]
    opd_reward_weight_mode: Annotated[
        str,
        A(
            "--opd-reward-weight-mode",
            type=str,
            choices=["student_p", "teacher_p", "none"],
            default="student_p",
            help="Weighting scheme for top-k OPD token rewards.",
        ),
    ]
    opd_topk_per_position: Annotated[
        bool,
        A(
            "--opd-topk-per-position",
            action="store_true",
            default=False,
            help=(
                "Send per-position token ids to the teacher/student scoring server "
                "(token_ids_logprob_positions) instead of the global top-k union, so the "
                "response is O(response_len * k) instead of O(response_len * |union|). "
                "Requires a patched sglang server that supports token_ids_logprob_positions; "
                "leave off for an unpatched server."
            ),
        ),
    ]
    opd_teacher_urls: Annotated[
        list[str] | None,
        A(
            "--opd-teacher-urls",
            type=str,
            nargs="+",
            default=None,
            metavar="NAME=URL",
            help=(
                "Multi-teacher routing map for --opd-type=sglang, e.g. "
                "--opd-teacher-urls math=http://h1:30001/generate code=http://h2:30002/generate. "
                "Each sample is routed to the teacher named by "
                "sample.metadata[--opd-teacher-key]; the reserved name 'default' is the "
                "fallback for samples with a missing or unknown name. When unset, all "
                "samples are scored by the single teacher at --rm-url (original behavior)."
            ),
        ),
    ]
    opd_teacher_key: Annotated[
        str,
        A(
            "--opd-teacher-key",
            type=str,
            default="opd_teacher",
            help=(
                "Sample metadata key holding the teacher name used for --opd-teacher-urls "
                "routing. Populated from the dataset's metadata column (see --metadata-key)."
            ),
        ),
    ]
    opd_teacher_load: Annotated[
        str | None,
        A(
            "--opd-teacher-load",
            type=str,
            default=None,
            help=(
                "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                "The teacher model should have the same architecture as policy/ref model."
            ),
        ),
    ]
    opd_teacher_ckpt_step: Annotated[
        int | None,
        A("--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."),
    ]
