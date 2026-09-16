from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class CiConfig(BaseConfig):
    ci_inject_missing_prefetched_batch_bug: Annotated[
        bool,
        A(
            "--ci-inject-missing-prefetched-batch-bug",
            action="store_true",
            help="Discard the restored prefetched batch to test sample ownership failure detection.",
        ),
    ]
    ci_test: Annotated[bool, A("--ci-test", action="store_true")]
    ci_disable_kl_checker: Annotated[bool, A("--ci-disable-kl-checker", action="store_true")]
    ci_disable_logprobs_checker: Annotated[bool, A("--ci-disable-logprobs-checker", action="store_true")]
    ci_disable_weight_update_checker: Annotated[bool, A("--ci-disable-weight-update-checker", action="store_true")]
    ci_metric_checker_key: Annotated[str | None, A("--ci-metric-checker-key", type=str, default=None)]
    ci_metric_checker_threshold: Annotated[float | None, A("--ci-metric-checker-threshold", type=float, default=None)]
    ci_metric_checker_expect_num: Annotated[
        int | None,
        A(
            "--ci-metric-checker-expect-num",
            type=int,
            default=None,
            help="Require exactly this many eval checks, all meeting the CI threshold.",
        ),
    ]
    ci_assert_prefill_lag_max: Annotated[
        int | None,
        A(
            "--ci-assert-prefill-lag-max",
            type=int,
            default=None,
            help="Require every rollout's prompt KV to lag its decode weight version by at most this much.",
        ),
    ]
    ci_save_grad_norm: Annotated[str | None, A("--ci-save-grad-norm", type=str, default=None)]
    ci_load_grad_norm: Annotated[str | None, A("--ci-load-grad-norm", type=str, default=None)]
    ci_save_model_hash: Annotated[bool, A("--ci-save-model-hash", action="store_true")]
    ci_check_model_hash: Annotated[bool, A("--ci-check-model-hash", action="store_true")]
