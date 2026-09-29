"""Validated inputs for a two-update-per-configuration frozen-batch experiment.

The source batch never changes. Each phase is a fresh parameterization of its proven
base policy; replay is explicit and confined to the existing diagnostic rollout seam.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from miles_plugins.proximal.contracts import Contract, Nonempty, RunConfig, SafeId, behavior_correction_argv
from miles_plugins.proximal.e2e.argv import set_flag
from miles_plugins.proximal.initial_policy import verify_base_policy
from miles_plugins.proximal.snapshot import Digest
from miles_plugins.proximal.state_artifacts import RelativePath

if TYPE_CHECKING:
    from miles_plugins.proximal.offline_batch import Batch


class SweepPhase(Contract):
    name: SafeId
    updates: Literal[2]
    target_modules: Annotated[tuple[Nonempty, ...], Field(min_length=1)]


class SweepPlan(Contract):
    experiment_id: SafeId
    batch_path: RelativePath
    batch_sha256: Digest
    samples: Annotated[int, Field(gt=0)]
    nodes: Annotated[int, Field(gt=0)]
    phases: Annotated[tuple[SweepPhase, ...], Field(min_length=1)]
    recipe: Annotated[tuple[str, ...], Field(min_length=1)]

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


def phase_command(plan: SweepPlan, phase: SweepPhase, source: RunConfig, *, bundle: Path, save: Path) -> list[str]:
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
        "--start-rollout-id": "0",
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
    for flag in ("--use-rollout-logprobs", "--use-tis", "--tis-clip", "--tis-clip-low"):
        args = set_flag(args, flag, None)
    args += behavior_correction_argv(research.behavior_correction)
    for flag in ("--debug-train-only", "--disable-rollout-global-dataset"):
        args = set_flag(args, flag, None) + [flag]
    return ["python", "/fork/train.py", *args]


def validate_phase_args(args: object, plan: SweepPlan, source: RunConfig) -> None:
    """Check parsed model/shape flags before starting Ray or any GPU workers."""
    for name, expected in {
        "num_rollout": 2,
        "start_rollout_id": 0,
        "actor_num_nodes": plan.nodes,
        "actor_num_gpus_per_node": 8,
        "global_batch_size": plan.samples,
        "n_samples_per_prompt": source.research.group_size,
        "save_interval": 1,
        "lora_A_init_method": "xavier",
        "lora_B_init_method": "zero",
    }.items():
        # Bridge's initializer defaults are consumed by the builder, not CLI flags.
        actual = getattr(args, name, expected if name.startswith("lora_") else None)
        if actual != expected:
            raise ValueError(f"Sweep requires {name}={expected!r}, got {actual!r}")
