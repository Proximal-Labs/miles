import json
from typing import Annotated, Any

from miles.utils.args.schema import A, BaseConfig


# data
class DataConfig(BaseConfig):
    # dataset
    # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
    num_rollout: Annotated[int | None, A(
        "--num-rollout", type=int, default=None,
        help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
    )]
    debug_exit_after_rollout: Annotated[int | None, A(
        "--debug-exit-after-rollout", type=int, default=None,
        help="Exit training after this many rollouts (for testing checkpoint resume with consistent scheduler params).",
    )]
    num_epoch: Annotated[int | None, A(
        "--num-epoch", type=int, default=None,
        help=(
            "Number of epochs for the training. "
            "This is used to calculate the number of rollout steps from the dataset size. "
            "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
            "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
        ),
    )]
    rollout_global_dataset: Annotated[bool, A(
        "--disable-rollout-global-dataset", action="store_false", dest="rollout_global_dataset",
        help=(
            "Disable the global dataset for rollout. By default, Miles loads `--prompt-data` into a global dataset and samples from it for rollout. "
            "Setting this flag turns off this behavior, Use this flag only when providing a custom `--rollout-function-path` (and usually a custom `--data-source-path`) that handles data loading independently."
        ),
    )]
    data_source_path: Annotated[str, A(
        "--data-source-path", type=str, default="miles.rollout.data_source.RolloutDataSource",
        help="The data source class for rollout data.",
    )]
    prompt_data: Annotated[str | None, A(
        "--prompt-data", type=str, default=None,
        help=(
            "The path to the prompt data. "
            "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
            "which will be used as the prompt and the label respectively."
            "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
            "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
        ),
    )]
    apply_chat_template: Annotated[bool, A("--apply-chat-template", action="store_true", default=False)]
    # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
    apply_chat_template_kwargs: Annotated[Any, A("--apply-chat-template-kwargs", type=json.loads, default="{}")]
    chat_template_path: Annotated[str | None, A(
        "--chat-template-path", type=str, default=None,
        help="Path to an explicit custom Jinja chat template file (.jinja). "
        "Sets tokenizer.chat_template when loading via load_tokenizer, "
        "and also sets --sglang-chat-template so the sglang server uses the same template. "
        "For Miles-maintained fixed templates, leave this unset and pass "
        "--tito-model so Miles can auto-resolve the registered template. "
        "The literal value 'autofix' is kept only as a "
        "deprecated compatibility alias for that auto-resolve path. "
        "The path must be accessible on all Ray worker nodes "
        "(e.g. a path inside the miles repo, or a shared filesystem like NFS).",
    )]
    input_key: Annotated[str, A("--input-key", type=str, default="input", help="JSON dataset key")]
    label_key: Annotated[str | None, A("--label-key", type=str, default=None, help="JSON dataset key")]
    multimodal_keys: Annotated[Any, A(
        "--multimodal-keys", type=json.loads, default=None,
        help=('JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''),
    )]
    metadata_key: Annotated[str, A("--metadata-key", type=str, default="metadata", help="JSON dataset key")]
    tool_key: Annotated[str, A(
        "--tool-key", type=str, default="tools",
        help=("When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."),
    )]
    start_rollout_id: Annotated[int | None, A(
        "--start-rollout-id", type=int, default=None,
        help=(
            "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
            "otherwise will be set to 0, meaning training from start."
        ),
    )]

    # batch sizes
    rollout_batch_size: Annotated[int, A(
        "--rollout-batch-size", type=int, required=True,
        help=(
            "The number of prompts in each rollout step. "
            "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
        ),
    )]
    n_samples_per_prompt: Annotated[int, A(
        "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation",
    )]

    # gbs of the training, note that the gbs is of sample, not of prompts,
    # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
    # `rollout_batch_size * n_samples_per_prompt`.
    global_batch_size: Annotated[int | None, A("--global-batch-size", reset=True, type=int, default=None)]
    num_steps_per_rollout: Annotated[int | None, A(
        "--num-steps-per-rollout", type=int, default=None,
        help=(
            "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
            "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
        ),
    )]
    # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
    micro_batch_size: Annotated[int, A("--micro-batch-size", reset=True, type=int, default=1)]
    balance_data: Annotated[bool, A(
        "--balance-data", action="store_true", default=False,
        help=(
            "Repartition each rollout batch so each data-parallel rank gets a similar total token count via Karmarkar-Karp method. "
            "It may be beneficial for training speed but changes per-rank sample grouping and adds a small CPU scheduling overhead."
        ),
    )]
    balance_by_flops: Annotated[bool, A(
        "--balance-by-flops", action="store_true", default=False,
        help=(
            "Use FLOPs-based workload estimation for micro-batch partitioning via "
            "Karmarkar-Karp instead of first-fit token packing, and distribute mbs "
            "across DP ranks by FLOPs. Captures the quadratic attention cost when "
            "sequence lengths vary widely. Requires --use-dynamic-batch-size. NOTE: "
            "FLOPs balancing does not enforce the per-mbs token cap."
        ),
    )]
    allow_partial_train_step: Annotated[bool, A(
        "--allow-partial-train-step", action="store_true", default=False,
        help=(
            "Train the trailing rollouts that don't fill a whole global_batch_size step as one "
            "smaller final step instead of dropping them (rollout-side schedule + dynamic batch "
            "size only). Loss normalization and the LR scheduler use the true per-step count."
        ),
    )]
    use_dynamic_batch_size: Annotated[bool, A(
        "--use-dynamic-batch-size", action="store_true", default=False,
        help=(
            "Because the sample length varies, to maximize the GPU utilization, "
            "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
            "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
            "dynamic batch size, miles will make 2 micro batches, i.e. [100, 200], [300]."
        ),
    )]
    max_tokens_per_gpu: Annotated[int | None, A(
        "--max-tokens-per-gpu", type=int, default=None,
        help=(
            "The maximum number of tokens per GPU for dynamic batch size. "
            "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
            "`max_response_len // cp_size` instead of `max_response_len`."
        ),
    )]
    log_probs_max_tokens_per_gpu: Annotated[int | None, A(
        "--log-probs-max-tokens-per-gpu", type=int, default=None,
        help=(
            "The maximum number of tokens per GPU for calculating log probs. "
            "This is used to calculate the log probs of the responses during rollout, "
            "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
        ),
    )]
