"""Validated inputs for a two-update-per-configuration frozen-batch experiment.

The source batch never changes. Each phase is a fresh parameterization of its proven
base policy; replay is explicit and confined to the existing diagnostic rollout seam.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from miles_plugins.proximal.contracts import (
    Contract,
    RunConfig,
    SafeId,
    behavior_correction_argv,
    sampling_args,
    sampling_argv,
)
from miles_plugins.proximal.e2e.argv import set_flag
from miles_plugins.proximal.initial_policy import verify_base_policy
from miles_plugins.proximal.snapshot import Digest, Nonempty
from miles_plugins.proximal.state_artifacts import RelativePath, StateFile
from miles_plugins.proximal.training import TrainingDeployment

if TYPE_CHECKING:
    from miles_plugins.proximal.offline_batch import Batch


class SweepPhase(Contract):
    name: SafeId
    updates: Literal[2]
    target_modules: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    resume: StateFile | None


class SweepPlan(Contract):
    experiment_id: SafeId
    batch_path: RelativePath
    batch_sha256: Digest
    samples: Annotated[int, Field(gt=0)]
    nodes: Annotated[int, Field(gt=0)]
    phases: Annotated[tuple[SweepPhase, ...], Field(min_length=1)]
    recipe: Annotated[tuple[str, ...], Field(min_length=1)]
    phase_attempts: Annotated[int, Field(ge=1, le=3)] = 2
    failure_hold_seconds: Annotated[int, Field(ge=0, le=21600)] = 21600

    @model_validator(mode="after")
    def _unique(self) -> "SweepPlan":
        if len({phase.name for phase in self.phases}) != len(self.phases):
            raise ValueError("Sweep phase names must be unique")
        for phase in self.phases:
            if len(set(phase.target_modules)) != len(phase.target_modules):
                raise ValueError("Repeated target module in phase")
        return self


def check_sweep_deployment(deployment: TrainingDeployment) -> None:
    """Check hardware selection before constructing a paid Modal function."""
    if deployment.num_gpus != 8 or deployment.gpu not in {"B200:8", "B300:8"}:
        raise ValueError("The Qwen sweep requires gpu B200:8 or B300:8 and num_gpus=8 per node")
    if deployment.gpu == "B300:8":
        return
    if deployment.model_args != "qwen3.8-27B":
        raise ValueError("The B200 sweep layout is validated only for model_args=qwen3.8-27B")
    if deployment.deterministic_kernels or deployment.cuda_allocator != "expandable_segments":
        raise ValueError("B200 requires deterministic_kernels=false and cuda_allocator=expandable_segments")


def resolve_sweep_plan(plan: SweepPlan, deployment: TrainingDeployment) -> SweepPlan:
    """Resolve the Qwen B200 memory recipe before validation, recording or resume checks.

    B300 keeps the authored recipe. B200 uses the measured 256k BF16 layout; this
    is specific to this Qwen sweep, not a default for other models or launchers.
    The resulting plan is the one preflight, every worker and recovery must use.
    """
    check_sweep_deployment(deployment)
    if deployment.gpu == "B300:8":
        return plan
    flags = {token.split("=", 1)[0] for token in plan.recipe}
    if {"--fp16", "--fp8", "--fp4"} & flags:
        raise ValueError("The B200 sweep layout requires BF16; remove conflicting precision flags")
    # Selecting B200 overrides authored memory/layout flags with the measured
    # Qwen recipe. Return the effective plan so validation and recovery see it too.
    recipe = list(plan.recipe)
    for flag, value in (
        ("--tensor-model-parallel-size", "4"),
        ("--pipeline-model-parallel-size", "1"),
        ("--context-parallel-size", "2"),
        ("--micro-batch-size", "1"),
        ("--max-tokens-per-gpu", "131072"),
        ("--recompute-granularity", "full"),
        ("--recompute-method", "uniform"),
        ("--recompute-num-layers", "1"),
        ("--log-probs-chunk-size", "4096"),
    ):
        recipe = set_flag(recipe, flag, value)
    # If a separate logprob pass is requested, its packing budget must fit too.
    if "--log-probs-max-tokens-per-gpu" in flags:
        recipe = set_flag(recipe, "--log-probs-max-tokens-per-gpu", "131072")
    for flag in ("--bf16", "--sequence-parallel", "--use-dynamic-batch-size", "--recompute-loss-function"):
        recipe = set_flag(recipe, flag, None) + [flag]
    return plan.model_copy(update={"recipe": tuple(recipe)})


def validate_source(bundle: Path, plan: SweepPlan, source: RunConfig) -> "Batch":
    from miles_plugins.proximal.offline_batch import validate_batch  # Requires the trainer dependency set.

    if hashlib.sha256((bundle / "batch.json").read_bytes()).hexdigest() != plan.batch_sha256:
        raise ValueError("Sweep batch differs from the pinned manifest")
    batch = validate_batch(bundle)
    if batch.source != source or batch.num_samples != plan.samples:
        raise ValueError("Sweep must use the authorized source and exact sample count")
    # The original source contract authenticates this proof. The experimental target
    # set is deliberately separate and never substituted into the source metadata.
    verify_base_policy(batch.source, batch.policy, bundle / "base_policy")
    if batch.source.research.max_policy_lag < 1:
        raise ValueError("Two-update replay requires allowance for the older base-policy batch")
    return batch


def phase_command(
    plan: SweepPlan,
    phase: SweepPhase,
    source: RunConfig,
    *,
    bundle: Path,
    save: Path,
    resume_adapter: Path | None = None,
) -> list[str]:
    """Native Miles loop, exact behavior data, fresh optimizer for each phase."""
    forbidden = {
        "--lora-adapter-path",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--finetune",
        "--fully-async",
        "--enable-mtp-training",
        "--mtp-num-layers",
        "--custom-weight-transfer-protocol-path",
        "--custom-reward-post-process-path",
        "--custom-convert-samples-to-train-data-path",
        "--dynamic-sampling-filter-path",
        "--rollout-sample-filter-path",
        "--load-debug-rollout-data",
        "--partial-rollout",
        "--use-critic",
        "--multi-lora",
        "--use-dynamic-global-batch-size",
        "--debug-disable-optimizer",
    }
    if blocked := forbidden & {token.split("=", 1)[0] for token in plan.recipe}:
        raise ValueError(f"Unsupported frozen-base experiment flags: {sorted(blocked)}")
    if not {"--seed", "--optimizer", "--lr"} <= set(plan.recipe):
        raise ValueError("Sweep recipe must explicitly declare optimizer, learning rate and seed")
    args = list(plan.recipe)
    for flag in ("--use-wandb", "--wandb-project", "--wandb-group", "--eval-interval"):
        args = set_flag(args, flag, None)
    research = source.research
    values = {
        "--actor-num-nodes": str(plan.nodes),
        "--actor-num-gpus-per-node": "8",
        "--hf-checkpoint": str(source.tokenizer_path),
        "--load": str(source.tokenizer_path),
        "--train-backend": "megatron",
        "--megatron-to-hf-mode": "bridge",
        "--lora-rank": str(research.lora.rank),
        "--lora-alpha": str(research.lora.alpha),
        "--lora-dropout": "0",
        "--target-modules": ",".join(phase.target_modules),
        "--n-samples-per-prompt": str(research.group_size),
        "--rollout-batch-size": str(plan.samples // research.group_size),
        "--global-batch-size": str(plan.samples),
        "--num-rollout": str(phase.updates),
        "--start-rollout-id": "1" if phase.resume is not None else "0",
        "--rollout-num-gpus": "0",
        "--rollout-max-response-len": str(research.sampling.max_tokens),
        "--rollout-max-context-len": str(research.sampling.max_sequence_tokens),
        "--rollout-function-path": "miles_plugins.proximal.e2e.state_gpu_replay.ReferenceReplay",
        "--verification-batch": str(bundle),
        "--save": str(save),
        "--save-interval": "1",
    }
    if plan.samples % research.group_size:
        raise ValueError("Sweep samples must be a whole number of source training groups")
    for flag, value in values.items():
        args = set_flag(args, flag, value)
    if (phase.resume is None) != (resume_adapter is None):
        raise ValueError("Native resume requires its verified, staged adapter directory")
    if resume_adapter is not None:
        args = set_flag(args, "--lora-adapter-path", str(resume_adapter))
    for flag in ("--use-rollout-logprobs", "--use-tis", "--tis-clip", "--tis-clip-low"):
        args = set_flag(args, flag, None)
    args += behavior_correction_argv(research.behavior_correction)
    # The batch's sampling decides whether Miles replays each token's recorded support.
    for flag in ("--rollout-temperature", "--rollout-top-p", "--rollout-top-k"):
        args = set_flag(args, flag, None)
    args += sampling_argv(research.sampling)
    for flag in ("--debug-train-only", "--disable-rollout-global-dataset"):
        args = set_flag(args, flag, None) + [flag]
    return ["python", "/fork/train.py", *args]


def validate_phase_args(args: object, plan: SweepPlan, source: RunConfig, phase: SweepPhase) -> None:
    """Check parsed model/shape flags before starting Ray or any GPU workers."""
    for name, expected in {
        "num_rollout": 2,
        "start_rollout_id": 1 if phase.resume is not None else 0,
        "actor_num_nodes": plan.nodes,
        "actor_num_gpus_per_node": 8,
        "global_batch_size": plan.samples,
        "n_samples_per_prompt": source.research.group_size,
        "save_interval": 1,
        "lora_A_init_method": "xavier",
        "lora_B_init_method": "zero",
        **sampling_args(source.research.sampling),
    }.items():
        # Bridge's initializer defaults are consumed by the builder, not CLI flags.
        actual = getattr(args, name, expected if name.startswith("lora_") else None)
        if actual != expected:
            raise ValueError(f"Sweep requires {name}={expected!r}, got {actual!r}")
