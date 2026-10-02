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


# Flags that would change initialization, persistence or the data path of a frozen
# base-policy experiment; the native Miles loop must own every optimizer step.
UNSUPPORTED_RECIPE_FLAGS = frozenset(
    {
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
)


def check_recipe(recipe: tuple[str, ...]) -> None:
    if blocked := UNSUPPORTED_RECIPE_FLAGS & {token.split("=", 1)[0] for token in recipe}:
        raise ValueError(f"Unsupported frozen-base experiment flags: {sorted(blocked)}")
    if not {"--seed", "--optimizer", "--lr"} <= set(recipe):
        raise ValueError("Experiment recipe must explicitly declare optimizer, learning rate and seed")


def native_train_argv(
    recipe: tuple[str, ...],
    source: RunConfig,
    *,
    nodes: int,
    samples: int,
    target_modules: tuple[str, ...],
    num_rollout: int,
    start_rollout_id: int,
    rollout_function: str,
    batch_flag: str,
    bundle: Path,
    save: Path,
    resume_adapter: Path | None,
) -> list[str]:
    """Native Miles train-only loop over stored behavior data: the source's sampling, LoRA
    shape and behavior correction, an explicit target set and one save per update."""
    check_recipe(recipe)
    args = list(recipe)
    for flag in ("--use-wandb", "--wandb-project", "--wandb-group", "--eval-interval"):
        args = set_flag(args, flag, None)
    research = source.research
    values = {
        "--actor-num-nodes": str(nodes),
        "--actor-num-gpus-per-node": "8",
        "--hf-checkpoint": str(source.tokenizer_path),
        "--load": str(source.tokenizer_path),
        "--train-backend": "megatron",
        "--megatron-to-hf-mode": "bridge",
        "--lora-rank": str(research.lora.rank),
        "--lora-alpha": str(research.lora.alpha),
        "--lora-dropout": "0",
        "--target-modules": ",".join(target_modules),
        "--n-samples-per-prompt": str(research.group_size),
        "--rollout-batch-size": str(samples // research.group_size),
        "--global-batch-size": str(samples),
        "--num-rollout": str(num_rollout),
        "--start-rollout-id": str(start_rollout_id),
        "--rollout-num-gpus": "0",
        "--rollout-max-response-len": str(research.sampling.max_tokens),
        "--rollout-max-context-len": str(research.sampling.max_sequence_tokens),
        "--rollout-function-path": rollout_function,
        batch_flag: str(bundle),
        "--save": str(save),
        "--save-interval": "1",
    }
    if samples % research.group_size:
        raise ValueError("Samples must be a whole number of source training groups")
    for flag, value in values.items():
        args = set_flag(args, flag, value)
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
    check_recipe(plan.recipe)
    if (phase.resume is None) != (resume_adapter is None):
        raise ValueError("Native resume requires its verified, staged adapter directory")
    return native_train_argv(
        plan.recipe,
        source,
        nodes=plan.nodes,
        samples=plan.samples,
        target_modules=phase.target_modules,
        num_rollout=phase.updates,
        start_rollout_id=1 if phase.resume is not None else 0,
        rollout_function="miles_plugins.proximal.e2e.state_gpu_replay.ReferenceReplay",
        batch_flag="--verification-batch",
        bundle=bundle,
        save=save,
        resume_adapter=resume_adapter,
    )


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
