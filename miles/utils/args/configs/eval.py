from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class EvalConfig(BaseConfig):
    eval_function_path: Annotated[str | None, A(
        "--eval-function-path", type=str, default=None,
        help=(
            "Path to the eval fn. Two kinds fit here. A rollout fn generates against the "
            "engines the framework hands it: the training engines, or the dedicated fleet "
            "when --eval-num-gpus is set. A CheckpointEvalFn subclass gets the snapshot "
            "path instead and owns the rest itself — weight delivery, endpoint, generation. "
            "If not set, defaults to --rollout-function-path."
        ),
    )]

    # change the default value of eval_interval from Megatron to None
    eval_interval: Annotated[int | None, A("--eval-interval", reset=True, type=int, default=None)]

    eval_prompt_data: Annotated[list[str] | None, A(
        "--eval-prompt-data", type=str, default=None, nargs="+",
        help=(
            "Path to the evaluation prompt data, "
            "should first input the name of the eval dataset and then the path, e.g. "
            "aime /path/to/aime.jsonl"
        ),
    )]
    eval_config: Annotated[str | None, A(
        "--eval-config", type=str, default=None,
        help=(
            "Path to an OmegaConf YAML/JSON file describing evaluation datasets, or an "
            "inline `base64:<payload>` carrying the same document. "
            "When provided, this overrides --eval-prompt-data."
        ),
    )]
    skip_eval_before_train: Annotated[bool, A(
        "--skip-eval-before-train", action="store_true", default=False,
        help="Whether to skip evaluation before training.",
    )]

    # The following keys are used to override the rollout version during eval.
    eval_input_key: Annotated[str | None, A("--eval-input-key", type=str, default=None, help="JSON dataset key")]
    eval_label_key: Annotated[str | None, A("--eval-label-key", type=str, default=None, help="JSON dataset key")]
    eval_tool_key: Annotated[str | None, A("--eval-tool-key", type=str, default=None, help="JSON dataset key")]
    n_samples_per_eval_prompt: Annotated[int, A(
        "--n-samples-per-eval-prompt", type=int, default=1, help="number of responses for each prompt in generation",
    )]
    eval_temperature: Annotated[float | None, A("--eval-temperature", type=float, default=None)]
    eval_top_p: Annotated[float | None, A("--eval-top-p", type=float, default=None)]
    eval_top_k: Annotated[int | None, A("--eval-top-k", type=int, default=None)]
    eval_max_response_len: Annotated[int | None, A("--eval-max-response-len", type=int, default=None)]
    eval_max_prompt_len: Annotated[int | None, A("--eval-max-prompt-len", type=int, default=None)]
    eval_min_new_tokens: Annotated[int | None, A("--eval-min-new-tokens", type=int, default=None)]
    eval_max_context_len: Annotated[int | None, A("--eval-max-context-len", type=int, default=None)]
    eval_num_gpus: Annotated[int, A(
        "--eval-num-gpus", type=int, default=0,
        help=(
            "Number of GPUs for a dedicated eval engine fleet. When > 0, eval runs on "
            "its own engines behind its own router, synced by loading HF checkpoint "
            "snapshots (never by joining training weight updates). 0 disables the "
            "fleet and keeps today's shared-engine eval behavior. The fleet's engine "
            "settings inherit every --sglang-* value; override individually with "
            "--eval-sglang-* (e.g. --eval-sglang-mem-fraction-static 0.9)."
        ),
    )]
    eval_num_gpus_per_engine: Annotated[int, A(
        "--eval-num-gpus-per-engine", type=int, default=1,
        help="GPUs per eval engine (TP size), independent of --rollout-num-gpus-per-engine.",
    )]
    eval_hf_dir: Annotated[str | None, A(
        "--eval-hf-dir", type=str, default=None,
        help=(
            "Staging directory for per-eval HF snapshots (written to "
            "`{eval_hf_dir}/step_{rollout_id}`). Point at tmpfs (e.g. /dev/shm/...) to "
            "avoid disk. When unset and --save-hf is set, eval reuses the --save-hf "
            "checkpoints instead of exporting its own snapshots."
        ),
    )]
    eval_max_in_flight: Annotated[int, A(
        "--eval-max-in-flight", type=int, default=2, help="Maximum number of concurrently pending async evals.",
    )]
    eval_overflow_policy: Annotated[str, A(
        "--eval-overflow-policy", type=str, choices=["backpressure", "skip"], default="backpressure",
        help=(
            "What to do when an eval is due but --eval-max-in-flight evals are pending: "
            "'backpressure' awaits the oldest pending eval (deterministic curve, bounded "
            "stall); 'skip' drops the new eval point and logs eval/skipped_busy at that "
            "step (training cadence is never stalled)."
        ),
    )]
    eval_keep_snapshots: Annotated[int, A(
        "--eval-keep-snapshots", type=int, default=2,
        help=(
            "How many snapshot dirs to keep under --eval-hf-dir (consumed snapshots "
            "beyond this are deleted). --save-hf checkpoints are never deleted."
        ),
    )]
