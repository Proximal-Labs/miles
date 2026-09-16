import argparse
import os
from typing import Annotated

from miles.utils.args.schema import A, BaseConfig
from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR


# debug
class DebugConfig(BaseConfig):
    save_debug_rollout_data: Annotated[
        str | None,
        A(
            "--save-debug-rollout-data",
            type=str,
            default=None,
            help=(
                "Save the rollout data to this path for debugging. "
                "The file will be saved to `save_debug_rollout_data.format(rollout_id)`, "
                "so the template must contain the `{rollout_id}` placeholder."
            ),
        ),
    ]
    save_debug_trajectory_data: Annotated[
        str | None,
        A(
            "--save-debug-trajectory-data",
            type=str,
            default=None,
            help=(
                "Save per-sample role-tagged trajectory text (JSONL) next to the rollout "
                "dump. The file will be saved to `save_debug_trajectory_data.format(rollout_id)`, "
                "so the template must contain the `{rollout_id}` placeholder."
            ),
        ),
    ]
    load_debug_rollout_data: Annotated[
        str | None,
        A(
            "--load-debug-rollout-data",
            type=str,
            default=None,
            help=(
                "Load the rollout data from this path for debugging. "
                "The file will be loaded from `load_debug_rollout_data.format(rollout_id)`. "
                "When this is enabled, miles will not instantiate sglang servers."
            ),
        ),
    ]
    load_debug_rollout_data_subsample: Annotated[
        float | None,
        A(
            "--load-debug-rollout-data-subsample",
            type=float,
            default=None,
            help="Subsample a portion of the debug rollout data for faster debugging.",
        ),
    ]
    debug_rollout_only: Annotated[
        bool,
        A(
            "--debug-rollout-only",
            action="store_true",
            default=False,
            help=(
                "Whether to only run the rollout generation without training. "
                "This is useful for debugging the rollout generation function."
            ),
        ),
    ]
    debug_train_only: Annotated[
        bool,
        A(
            "--debug-train-only",
            action="store_true",
            default=False,
            help=(
                "Whether to run training without rollout generation. Rollout engines are "
                "skipped; a snapshot-eval fleet (--eval-num-gpus) still starts when configured."
            ),
        ),
    ]
    save_debug_train_data: Annotated[
        str | None,
        A(
            "--save-debug-train-data",
            type=str,
            default=None,
            help=(
                "Save the train data to this path for debugging. "
                "The file will be saved to `save_debug_train_data.format(rollout_id)`."
            ),
        ),
    ]
    save_debug_event_data: Annotated[
        str | None,
        A(
            "--save-debug-event-data",
            type=str,
            default=None,
            help="Where the audit events of this run go, including the env report. Defaults to <save>/events "
            "(or <dump-details>/events); --ci-test falls back to a run-specific temporary directory.",
        ),
    ]
    dump_details: Annotated[
        str | None,
        A(
            "--dump-details",
            type=str,
            default=None,
            help=("Dump all details of training for post-hoc analysis and visualization."),
        ),
    ]
    dumper_enable: Annotated[
        bool,
        A(
            "--dumper-enable",
            action="store_true",
            default=False,
            help="Enable sglang dumper for all three phases (sglang inference, "
            "megatron forward-only, megatron forward-backward). "
            "Per-phase --dumper-inference/--dumper-fwd-only/--dumper-fwd-bwd can override.",
        ),
    ]
    dumper_dir: Annotated[
        str,
        A(
            "--dumper-dir",
            type=str,
            default="/tmp/dumper",
            help="Base output directory for sglang dumper. Three subdirs are created: "
            "inference/, fwd_only/, fwd_bwd/.",
        ),
    ]
    dumper_inference: Annotated[
        list[str] | None,
        A(
            "--dumper-inference",
            nargs="*",
            default=None,
            help="SGLang inference phase dumper config as key=value pairs. "
            "Keys map to DumperConfig fields (e.g. enable=true filter=whatever).",
        ),
    ]
    dumper_fwd_only: Annotated[
        list[str] | None,
        A(
            "--dumper-fwd-only",
            nargs="*",
            default=None,
            help="Megatron forward-only phase dumper config as key=value pairs.",
        ),
    ]
    dumper_fwd_bwd: Annotated[
        list[str] | None,
        A(
            "--dumper-fwd-bwd",
            nargs="*",
            default=None,
            help="Megatron forward-backward phase dumper config as key=value pairs.",
        ),
    ]
    dumper_source_patcher_config_inference: Annotated[
        str | None,
        A(
            "--dumper-source-patcher-config-inference",
            type=str,
            default=None,
            help="Path to YAML config file for source patcher applied in SGLang inference engines.",
        ),
    ]
    dumper_source_patcher_config_train: Annotated[
        str | None,
        A(
            "--dumper-source-patcher-config-train",
            type=str,
            default=None,
            help="Path to YAML config file for source patcher applied in Megatron training actors.",
        ),
    ]
    # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
    memory_snapshot_dir: Annotated[str, A("--memory-snapshot-dir", type=str, default=".")]
    memory_snapshot_num_steps: Annotated[int | None, A("--memory-snapshot-num-steps", type=int, default=None)]
    profile_target: Annotated[
        list[str],
        A(
            "--profile-target",
            type=str,
            choices=["train_overall", "train_actor", "train_log_probs"],
            default=["train_overall"],
            nargs="+",
        ),
    ]
    memory_recorder: Annotated[
        str, A("--memory-recorder", type=str, choices=["torch", "memray"], default="torch")
    ]
    check_weight_update_equal: Annotated[bool, A("--check-weight-update-equal", action="store_true")]
    check_weight_update_selector: Annotated[
        str,
        A(
            "--check-weight-update-selector",
            type=str,
            default="all",
            choices=["all", "target", "draft"],
            help="Which model the post-update equality check covers: 'all' (target + "
            "draft/MTP), 'target' (target model only; skips the draft, e.g. when MTP "
            "training is off), or 'draft' (draft/MTP worker only).",
        ),
    ]
    check_weight_update_skip_list: Annotated[
        list[str] | None,
        A(
            "--check-weight-update-skip-list",
            type=str,
            nargs="*",
            default=None,
            help="Weight-name substrings to exclude from the post-update equality check; "
            "their mismatches are downgraded to non-fatal info (e.g. MTP/draft layer names "
            "that are absent on the training side).",
        ),
    ]
    check_weight_update_allow_quant_error: Annotated[
        bool,
        A(
            "--check-weight-update-allow-quant-error",
            action="store_true",
            help="When comparing weights after update, allow quantized tensors to differ "
            "by up to 1 ULP of the quantized dtype per side (compared in dequantized space).",
        ),
    ]
    check_lora_weight_equal: Annotated[
        bool,
        A(
            "--check-lora-weight-equal",
            action="store_true",
            default=False,
            help=(
                "Verify the megatron->sglang LoRA adapter weight-sync on the colocated "
                "(from_tensors) path: on every sync the trainer ships a per-tensor sha256 "
                "manifest of the adapter it sends, and each rollout engine hashes the "
                "tensors it received and fails the load on any mismatch/missing/extra "
                "name. The LoRA analogue of --check-weight-update-equal, which only "
                "covers base weights."
            ),
        ),
    ]
    save_local_weight_checksum: Annotated[
        bool,
        A("--save-local-weight-checksum", action="store_true", help="Save per-rank local weight checksum per-step."),
    ]
    enable_event_analyzer: Annotated[
        bool,
        A(
            "--enable-event-analyzer",
            action="store_true",
            help="Enable event analyzer to run sanity checks (e.g. cross-replica checksum consistency) before each training step.",
        ),
    ]
    enable_sample_ownership_checker: Annotated[
        bool | None,
        A(
            "--enable-sample-ownership-checker",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Verify exactly one outcome for every consumed sample and every mature issued sample; "
            "CI enables this unless it is explicitly disabled. Every actor step appends one full "
            "consumption snapshot per replica to the event log, whose size therefore grows with steps "
            "times consumed samples, so this is meant for CI and debugging.",
        ),
    ]
    sample_ownership_grace_steps: Annotated[
        int | None,
        A(
            "--sample-ownership-grace-steps",
            type=int,
            default=None,
            help="Completed rollout training steps before checking an issued sample (default: 10, or 2 in CI).",
        ),
    ]
    enable_witness: Annotated[
        bool, A("--enable-witness", action="store_true", help="Enable forward/backward pass witness.")
    ]
    witness_buffer_size: Annotated[
        int,
        A(
            "--witness-buffer-size",
            type=int,
            default=1048576,
            help="Maximum number of unique witness IDs before recycling.",
        ),
    ]
    ci_ft_test_actions: Annotated[
        str | None,
        A(
            "--ci-ft-test-actions",
            type=str,
            default=None,
            help="JSON array of fault injection actions. Each action: "
            '{"at_rollout": N, "action": "stop_cell_at_end"|"start_cell_at_end"|"crash_before_allreduce", '
            '"cell_id": "trainer-engine-actor-00002", "rank": 0, "attempt": 0}. '
            "cell_id is the full cell id (spec name plus zero-padded cell index) of the target cell. "
            'The action "sleep_forever_at_end" names no cell: it puts the orchestration script itself to sleep '
            "once the step it names is trained and saved, so the run never starts the step after it.",
        ),
    ]
    # TODO ad hoc hack: revert after the args refactor
    ci_ft_test_actions_path: Annotated[
        str | None,
        A(
            "--ci-ft-test-actions-path",
            type=str,
            default=None,
            help="Path of a file holding the same JSON array as --ci-ft-test-actions, read afresh every time "
            "the actions are consulted. A run relaunched in place keeps the arguments its pods were rendered "
            "from, so a plan that has to change from one launch to the next is delivered through this file "
            "instead of through the argument. Mutually exclusive with --ci-ft-test-actions.",
        ),
    ]
    ci_inject_rollout_data_path: Annotated[
        str | None,
        A(
            "--ci-inject-rollout-data-path",
            type=str,
            default=None,
            help="CI comparison tests only: path template (with {rollout_id}) of rollout "
            "data recorded via --save-debug-rollout-data. For rollouts at or after "
            "--ci-inject-rollout-data-start-rollout-id, generation still runs normally "
            "but its result is discarded and the recorded data is used for training "
            "instead. Unlike --load-debug-rollout-data, sglang engines stay alive "
            "(debug_train_only is not forced).",
        ),
    ]
    ci_inject_rollout_data_start_rollout_id: Annotated[
        int | None,
        A(
            "--ci-inject-rollout-data-start-rollout-id",
            type=int,
            default=None,
            help="First rollout_id whose training data is replaced by the "
            "--ci-inject-rollout-data-path recordings.",
        ),
    ]
    ci_inject_rollout_data_min_match_ratio: Annotated[
        float,
        A(
            "--ci-inject-rollout-data-min-match-ratio",
            type=float,
            default=0.9,
            help="Minimum mean response-token match ratio between the discarded generated "
            "data and the injected recording. Below this the engine weights are considered "
            "wrong (legitimate ulp-level drift only flips occasional sampled tokens).",
        ),
    ]
    env_report: Annotated[
        str,
        A(
            "--env-report",
            type=str,
            default_factory=lambda: os.environ.get(LAUNCHER_REPORT_ENV_VAR, ""),
            help="Path to the json record the external launcher wrote about the launch that started "
            "this process.",
        ),
    ]
    env_report_interval_seconds: Annotated[
        float,
        A(
            "--env-report-interval-seconds",
            type=float,
            default=3600.0,
            help="How often every process re-records its environment, so that code loaded later "
            "(lazy imports, a swapped shared disk) is still captured. Non-positive records only at startup.",
        ),
    ]
    debug_unified_grad_fused_logprob: Annotated[
        bool,
        A(
            "--debug-unified-grad-fused-logprob",
            action="store_true",
            default=False,
            help="Debug/test only: compute the stored log probabilities through the same grad-enabled fused "
            "cross entropy the training step uses, then detach the result, so the two invocations of the "
            "fused kernel take one execution path instead of two.",
        ),
    ]
    debug_deterministic_collective: Annotated[
        bool,
        A(
            "--debug-deterministic-collective",
            action="store_true",
            default=False,
            help="Debug/test only: run the training world on the det_nccl backend "
            "(miles.utils.test_utils.det_process_group), which folds order-sensitive SUM/AVG "
            "reductions in a fixed tree order so different reduction topologies become "
            "bitwise-comparable. Slow; never enable in production.",
        ),
    ]
