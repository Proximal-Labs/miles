import argparse
import json
from typing import Annotated, Any

from miles.utils.args.schema import A, BaseConfig
from miles.utils.object_store import ObjectStoreBackend


# rollout
class RolloutConfig(BaseConfig):
    hf_checkpoint: Annotated[
        str | None,
        A(
            "--hf-checkpoint",
            type=str,
            default=None,
            help=(
                "The huggingface checkpoint of the trained model. "
                "This is used to initialize sglang and also provide the tokenizer. "
                "Note that, we will always update the parameters in sglang with that of megatron before training, "
                "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                "It doesn't necessary need to contain the most up-to-date parameters."
            ),
        ),
    ]
    model_name: Annotated[
        str | None,
        A(
            "--model-name",
            type=str,
            default=None,
            help=(
                "The name of the model, this is used to convert the megatron weights into huggingface format. "
                "If not set, we will use `type(load_hf_config(args.hf_checkpoint)).__name__.lower()` as model_name. "
                "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
            ),
        ),
    ]
    rollout_function_path: Annotated[
        str | None,
        A(
            "--rollout-function-path",
            type=str,
            default=None,
            help=(
                "Path to the rollout generation function. "
                "Use this to create your own custom rollout function and set this to its path. "
                "The function is called as `fn(args, rollout_id, data_source, evaluation=evaluation)`, "
                "so its signature should be "
                "`def generate_rollout(args, rollout_id, data_source, evaluation=False) "
                "-> RolloutFnTrainOutput | RolloutFnEvalOutput` "
                "(see `miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn` "
                "for the default class-based implementation). "
                "Within each output sample, set at least `tokens`, `response_length`, `reward`, "
                "and `truncated`."
            ),
        ),
    ]
    fully_async: Annotated[
        bool,
        A(
            "--fully-async",
            action="store_true",
            default=False,
            help=(
                "Run fully async rollout: a persistent worker keeps generating while the trainer "
                "drains completed groups. Selects `FullyAsyncRolloutFn` as the rollout function; "
                "evaluation keeps the standard rollout function, which fully async does not serve. "
                "Requires train_async.py."
            ),
        ),
    ]
    namespaced_radix_cache: Annotated[
        bool | None,
        A(
            "--namespaced-radix-cache",
            action=argparse.BooleanOptionalAction,
            help=(
                "Whether every generation request carries a radix cache key naming the rollout call "
                "the sample started under, so prefix KV computed under old weights cannot serve "
                "samples of a later call. Defaults to true when --fully-async is combined with "
                "--pause-generation-mode in_place, where the engine never flushes the cache and the "
                "staleness of a shared prompt is otherwise unbounded; an explicit "
                "--no-namespaced-radix-cache is respected."
            ),
        ),
    ]
    rollout_temperature: Annotated[
        float,
        A("--rollout-temperature", type=float, default=1.0, help="the temperature for the inference engine during rollout."),
    ]
    rollout_top_p: Annotated[
        float, A("--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout.")
    ]
    rollout_top_k: Annotated[
        int, A("--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout.")
    ]
    rollout_max_context_len: Annotated[
        int | None,
        A(
            "--rollout-max-context-len",
            type=int,
            default=None,
            help=(
                "The maximum context size for the inference engine during rollout."
                "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
            ),
        ),
    ]
    rollout_max_prompt_len: Annotated[
        int | None,
        A(
            "--rollout-max-prompt-len",
            type=int,
            default=None,
            help=(
                "The maximum length of the prompt for the inference engine during rollout. "
                "If set, we will filter out the long prompts during initialization of the global dataset. "
                "This is not recommended if the dataset is large."
            ),
        ),
    ]
    rollout_max_response_len: Annotated[
        int | None,
        A(
            "--rollout-max-response-len",
            type=int,
            default=None,
            help=(
                "The maximum length of the response for the inference engine during rollout. "
                "It is basically `max_tokens` in sglang."
            ),
        ),
    ]
    rollout_skip_special_tokens: Annotated[
        bool,
        A(
            "--rollout-skip-special-tokens",
            action="store_true",
            default=False,
            help=(
                "Whether to skip special tokens in the response during rollout. "
                "This is useful when you want to use the response as a prompt for the next rollout."
            ),
        ),
    ]
    rollout_stop: Annotated[
        list[str] | None,
        A(
            "--rollout-stop",
            type=str,
            nargs="+",
            default=None,
            help=(
                "The stop words for the inference engine during rollout. "
                "It can be a list of strings or a single string. "
                "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
            ),
        ),
    ]
    rollout_stop_token_ids: Annotated[
        list[int] | None,
        A(
            "--rollout-stop-token-ids",
            type=int,
            nargs="+",
            default=None,
            help=(
                "The stop token ids for the inference engine during rollout. "
                "It can be a list of integers or a single integer."
            ),
        ),
    ]
    rollout_shuffle: Annotated[
        bool,
        A("--rollout-shuffle", action="store_true", default=False, help=("Whether to shuffle the prompts during rollout.")),
    ]
    rollout_seed: Annotated[
        int,
        A(
            "--rollout-seed",
            type=int,
            default=42,
            help=(
                "The seed for the random number generator during rollout. "
                "This is used to shuffle the prompts and also for the random sampling of the prompts."
            ),
        ),
    ]
    object_store_backend: Annotated[
        str,
        A(
            "--object-store-backend",
            type=str,
            choices=tuple(backend.value for backend in ObjectStoreBackend),
            default="ray",
            help="Backend of the object store used to pass data (e.g. rollout data) between processes.",
        ),
    ]
    mooncake_store_init_kwargs: Annotated[
        Any,
        A(
            "--mooncake-store-init-kwargs",
            type=json.loads,
            default=None,
            help="JSON kwargs used to initialize MooncakeDistributedStore for rollout transfer.",
        ),
    ]
    mooncake_replica_num: Annotated[
        int,
        A("--mooncake-replica-num", type=int, default=1, help="Number of Mooncake memory replicas for each stored object."),
    ]

    # sampling
    over_sampling_batch_size: Annotated[
        int | None,
        A(
            "--over-sampling-batch-size",
            type=int,
            default=None,
            help=(
                "This defines the granularity of the sampling batch in the rollout function. "
                "When the number of available samples falls below the target, a sampling "
                "operation of size over_sampling_batch_size will be triggered."
                "Regardless of whether partial rollout is used or filters are applied, "
                "the sampling granularity is always determined by this value. "
                "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
            ),
        ),
    ]
    dynamic_sampling_filter_path: Annotated[
        str | None,
        A(
            "--dynamic-sampling-filter-path",
            type=str,
            default=None,
            help=(
                "This is the filter function for dynamic sampling. "
                "It should be able to judge whether the result of a prompt should be selected or not."
                "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                "You could use `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
            ),
        ),
    ]
    rollout_submission_granularity: Annotated[
        str | None,
        A(
            "--rollout-submission-granularity",
            type=str,
            choices=["group", "sample"],
            default=None,
            help=(
                "Granularity at which a completed unit frees rollout submission capacity. "
                "`group` holds a slot until the whole prompt group returns; `sample` frees each "
                "slot as its own sample finishes, so a replacement group goes out once "
                "n_samples_per_prompt samples complete, whichever groups they came from. "
                "Prompt groups are submitted whole either way. Unset picks the driver default: "
                "`sample` under --fully-async, where groups completed beyond the batch are queued "
                "for later steps; `group` otherwise, where they are aborted at the end of the step "
                "and, without --partial-rollout, discarded."
            ),
        ),
    ]

    # partial rollout
    partial_rollout: Annotated[
        bool,
        A(
            "--partial-rollout",
            action="store_true",
            default=False,
            help=(
                "Whether to use partial rollout. "
                "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                "This is useful for long responses."
            ),
        ),
    ]
    mask_offpolicy_in_partial_rollout: Annotated[
        bool,
        A(
            "--mask-offpolicy-in-partial-rollout",
            action="store_true",
            default=False,
            help=(
                "Whether to mask previous generation in partial rollout. "
                "If set, only on-policy generated tokens will be used in training"
            ),
        ),
    ]
    max_weight_staleness: Annotated[
        int | None,
        A(
            "--max-weight-staleness",
            type=int,
            default=None,
            help=(
                "Maximum allowed gap between a group's oldest weight version and the current "
                "engine weight version. Groups exceeding this threshold are recycled back to "
                "the data buffer instead of being sent to training. Only effective in fully "
                "async mode. None (default) disables staleness filtering."
            ),
        ),
    ]
    async_max_concurrent_samples: Annotated[
        int | None,
        A(
            "--async-max-concurrent-samples",
            type=int,
            default=None,
            help=(
                "Maximum number of concurrently generating trajectories in fully async mode, "
                "decoupling generation concurrency from the training batch size. None (default) "
                "keeps the legacy bound of one training batch worth of trajectories "
                "(rollout_batch_size groups, i.e. rollout_batch_size * n_samples_per_prompt)."
            ),
        ),
    ]
    async_data_buffer_capacity_factor: Annotated[
        float,
        A(
            "--async-data-buffer-capacity-factor",
            type=float,
            default=2.0,
            help=(
                "Capacity of the finished-group data buffer between rollout production and "
                "training consumption in fully async mode, as a multiple of rollout_batch_size "
                "(floor(factor * rollout_batch_size) groups). When the buffer is full the "
                "producer blocks until training consumes, so generation cannot run "
                "unboundedly ahead of training."
            ),
        ),
    ]
    async_unused_samples_handler: Annotated[
        str,
        A(
            "--async-unused-samples-handler",
            type=str,
            choices=["retry", "drop"],
            default="drop",
            help=(
                "What to do with a finished group fully async mode does not train on "
                "(aborted, or beyond --max-weight-staleness): drop "
                "(default) discards the group; retry recycles its prompts into the data "
                "source for regeneration. Groups rejected by "
                "--dynamic-sampling-filter-path are always dropped."
            ),
        ),
    ]
    custom_async_data_buffer_path: Annotated[
        str | None,
        A(
            "--custom-async-data-buffer-path",
            type=str,
            default=None,
            help=(
                "Path to a custom DataBuffer subclass replacing the fully async finished-group "
                "data buffer (see miles/rollout/fully_async_data_buffer.py). Constructed with "
                "DataBufferConstructorInput; it takes over dataflow/staleness control, so the "
                "--async-data-buffer-* args apply only if the custom class reads them."
            ),
        ),
    ]
    custom_generate_function_path: Annotated[
        str | None,
        A(
            "--custom-generate-function-path",
            type=str,
            default=None,
            help=(
                "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
            ),
        ),
    ]
    custom_rollout_log_function_path: Annotated[
        str | None,
        A(
            "--custom-rollout-log-function-path",
            type=str,
            default=None,
            help=(
                "The custom function for logging rollout data. The signature of the functions is: "
                "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                "The return value indicates whether to skip the default logging. "
            ),
        ),
    ]
    custom_eval_rollout_log_function_path: Annotated[
        str | None,
        A(
            "--custom-eval-rollout-log-function-path",
            type=str,
            default=None,
            help=(
                "The custom function for logging eval rollout data. "
                "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                "The return value indicates whether to skip the default logging. "
            ),
        ),
    ]

    buffer_filter_path: Annotated[
        str | None,
        A(
            "--buffer-filter-path",
            type=str,
            default=None,
            help=(
                "Path to the buffer filter function. "
                "It should be able to select the samples in the buffer. "
                "The function should take list[list[Sample]] and return list[list[Sample]]."
            ),
        ),
    ]
    # update weight
    update_weight_buffer_size: Annotated[
        int,
        A(
            "--update-weight-buffer-size",
            type=int,
            default=512 * 1024**2,
            help=(
                "buffer size for update weight, in bytes. "
                "This is used for updating weights by batch and should be useful for MoE models."
            ),
        ),
    ]
    update_weights_interval: Annotated[
        int, A("--update-weights-interval", type=int, default=1, help="Interval for updating the weights")
    ]
    pause_generation_mode: Annotated[
        str,
        A(
            "--pause-generation-mode",
            type=str,
            choices=["abort", "retract", "in_place"],
            default="retract",
            help=(
                "How SGLang pauses in-flight requests during weight updates. "
                "'abort' immediately terminates all requests (previous default). "
                "'retract' moves running requests back to the waiting queue and "
                "recomputes KV cache after update. "
                "'in_place' freezes requests and resumes with existing KV cache."
            ),
        ),
    ]
    keep_old_actor: Annotated[
        bool, A("--keep-old-actor", action="store_true", help="Whether to keep the rollout model on training process")
    ]

    rollout_data_postprocess_path: Annotated[
        str | None,
        A(
            "--rollout-data-postprocess-path",
            type=str,
            default=None,
            help=(
                "The called after we have all the rollout data including log_probs. "
                "It may be helpful for updating loss mask."
            ),
        ),
    ]
    pin_rollout_manager_to_head: Annotated[
        bool,
        A(
            "--pin-rollout-manager-to-head",
            action="store_true",
            default=False,
            help=(
                "Pin the RolloutExecutor (and the co-located router process) to the Ray head node. "
                "Useful in K8s where the head pod has a stable Service address so that "
                "external agent environments can reliably reach the router."
            ),
        ),
    ]
    rollout_external_engine_addrs: Annotated[
        list[str] | None,
        A(
            "--rollout-external-engine-addrs",
            type=str,
            default=None,
            nargs="+",
            help=(
                "Static addresses of externally launched SGLang engines, one per engine cell "
                "(the node-0 engine url for multi-node engines). Each entry is host:port or "
                "http://host:port. Setting this implies external rollout: Miles launches no "
                "engines and discovers the topology from each engine's /server_info."
            ),
        ),
    ]
    rollout_external_router_pd: Annotated[
        bool,
        A(
            "--rollout-external-router-pd",
            action="store_true",
            default=False,
            help=(
                "Launch the router in PD-disaggregation mode for external rollout engines. "
                "Internally launched engines infer this from the sglang config, but the router "
                "starts before external engines are discovered, so a PD external fleet must "
                "declare it here."
            ),
        ),
    ]
    custom_inference_engine_provider_path: Annotated[
        str | None,
        A(
            "--custom-inference-engine-provider-path",
            type=str,
            default=None,
            help=(
                "Import path of a callable(args, *, capability) returning the BaseWorkerProvider "
                "that reports the inference engine cells. Setting this implies external rollout. "
                "When unset it is filled in automatically: the static discovery provider with "
                "--rollout-external-engine-addrs, the backend's own provider otherwise."
            ),
        ),
    ]
    update_weight_transfer_mode: Annotated[
        str,
        A(
            "--update-weight-transfer-mode",
            choices=["broadcast", "p2p", "disk-delta"],
            default="broadcast",
            help=(
                "The method to transfer weights to remote rollout engines during update weight. "
                "'disk-delta' diffs each sync against a CPU snapshot of the previous one and publishes "
                "only the changed bytes to --update-weight-disk-dir; each engine's /pull_weights applies "
                "them into a host-local checkpoint that the engine reloads from."
            ),
        ),
    ]
    update_weight_disk_dir: Annotated[
        str | None,
        A(
            "--update-weight-disk-dir",
            type=str,
            default=None,
            help=(
                "Filesystem directory disk-delta weight sync publishes to: one delta directory "
                "(changed tensors only) per sync, written by the trainer and read by every "
                "rollout host. Required for --update-weight-transfer-mode=disk-delta."
            ),
        ),
    ]
    update_weight_local_checkpoint_dir: Annotated[
        str | None,
        A(
            "--update-weight-local-checkpoint-dir",
            type=str,
            default=None,
            help=(
                "Rollout-host-local directory (e.g. NVMe) holding a full HF checkpoint kept in "
                "sync by each engine's /pull_weights: every host seeds it from the engine's model "
                "path and patches published deltas in place, and the engines reload from it. "
                "Required for --update-weight-transfer-mode=disk-delta. The read-side counterpart "
                "of --custom-update-weight-post-write-path is the engine's "
                "--sglang-custom-pull-weights-pre-read-hook."
            ),
        ),
    ]
    update_weight_delta_encoding: Annotated[
        str,
        A(
            "--update-weight-delta-encoding",
            choices=["xor", "overwrite"],
            default="xor",
            help=(
                "On-disk delta encoding for disk-delta weight sync. 'xor' (default): new ^ old — "
                "smallest wire and fastest, but an involution that must be applied exactly once "
                "against the correct base (applying it twice reverts). 'overwrite': changed positions "
                "+ new absolute values — larger, but idempotent. Both are byte-level and dtype-blind; "
                "the engine reads the choice from each version's index metadata."
            ),
        ),
    ]
    update_weight_delta_checksum: Annotated[
        str,
        A(
            "--update-weight-delta-checksum",
            choices=["xxh3-128", "blake3", "adler32"],
            default="xxh3-128",
            help=(
                "Per-tensor integrity checksum for disk-delta apply. 'xxh3-128' (default): widest fast "
                "non-cryptographic digest. 'blake3': cryptographic, for untrusted storage. 'adler32': "
                "for interop. The engine reads the choice from each version's index metadata."
            ),
        ),
    ]
    custom_update_weight_post_write_path: Annotated[
        str | None,
        A(
            "--custom-update-weight-post-write-path",
            type=str,
            default=None,
            help=(
                "Path to a custom function called on each trainer rank after a disk-delta sync's "
                "files are written, before the engines read them — to publish the writes on a "
                "non-POSIX filesystem (no cross-host visibility without an explicit sync). "
                "Signature: ``def hook(args, version_dir: str, rollout_engines) -> None``; the hook gates itself."
            ),
        ),
    ]
    p2p_transfer_num_workers: Annotated[
        int,
        A("--p2p-transfer-num-workers", type=int, default=4, help="Number of thread pool workers for P2P weight transfer."),
    ]
    p2p_transfer_timeout: Annotated[
        float,
        A("--p2p-transfer-timeout", type=float, default=30.0, help="Timeout in seconds for each P2P transfer operation."),
    ]
