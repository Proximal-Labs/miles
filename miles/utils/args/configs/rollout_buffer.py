from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class RolloutBufferConfig(BaseConfig):
    rollout_buffer_url: Annotated[
        str | None, A("--rollout-buffer-url", type=str, default=None, help="URL for the rollout buffer")
    ]
    fetch_trajectory_retry_times: Annotated[
        int,
        A(
            "--fetch-trajectory-retry-times",
            type=int,
            default=-1,
            help="Number of times to retry fetching trajectory, -1 means unlimited retry",
        ),
    ]
    min_batch_collection_ratio: Annotated[
        float, A("--min-batch-collection-ratio", type=float, default=1, help="Minimum batch collection ratio")
    ]
    rollout_task_type: Annotated[str, A("--rollout-task-type", type=str, default="math")]
    loss_mask_type: Annotated[
        str,
        A(
            "--loss-mask-type",
            type=str,
            default="qwen",
            choices=["qwen", "qwen3", "distill_qwen"],
            help="Loss mask type",
        ),
    ]
    data_pad_size_multiplier: Annotated[
        int,
        A(
            "--data-pad-size-multiplier",
            type=int,
            default=128,
            help="Multiplier for data padding size in data processing.",
        ),
    ]
    rollout_sample_filter_path: Annotated[
        str | None,
        A(
            "--rollout-sample-filter-path",
            type=str,
            default=None,
            help=(
                "Path to the rollout sample filter function. "
                "This function determines whether a sample will participate in loss calculation. "
                "The function is called as `fn(args, data)` where `data` is `list[list[Sample]]` "
                "(grouped by n_samples_per_prompt), and should return None. "
                "To exclude a sample from the loss, set `sample.remove_sample = True`. "
                "Note: This attribute does not determine whether the sample participates in advantage normalization."
            ),
        ),
    ]
    rollout_all_samples_process_path: Annotated[
        str | None,
        A(
            "--rollout-all-samples-process-path",
            type=str,
            default=None,
            help=(
                "Path to the rollout all samples process function that "
                "can process all samples including filtered ones."
            ),
        ),
    ]
    disable_rollout_trim_samples: Annotated[
        bool,
        A(
            "--disable-rollout-trim-samples",
            action="store_true",
            default=False,
            help="disable trim samples in rollout buffer when converting samples to train data",
        ),
    ]
    use_dynamic_global_batch_size: Annotated[
        bool,
        A(
            "--use-dynamic-global-batch-size",
            action="store_true",
            default=False,
            help="enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data",
        ),
    ]
