"""Validated inputs for a chain of single-update steps over different frozen batches.

Every arm starts fresh from the batches' proven base policy and applies exactly one
optimizer update per step, continuing the previous step's native weights, optimizer,
scheduler and RNG. All batches were sampled by that base policy, so step k trains on
data k policy versions old: the source's behavior correction (TIS) is the only
off-policy correction, as in online async RL with a staleness window. Arms differ only
in their LoRA target set and see the same batches in the same order.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from miles_plugins.proximal.contracts import Contract, RunConfig, SafeId, sampling_args
from miles_plugins.proximal.e2e.batch_sweep_inputs import check_recipe, native_train_argv
from miles_plugins.proximal.initial_policy import verify_base_policy
from miles_plugins.proximal.snapshot import Digest, Nonempty
from miles_plugins.proximal.state_artifacts import RelativePath

if TYPE_CHECKING:
    from miles_plugins.proximal.offline_batch import Batch

CHAIN_REPLAY = "miles_plugins.proximal.e2e.batch_chain_replay.ChainReplay"


class ChainBatch(Contract):
    path: RelativePath
    sha256: Digest


class ChainArm(Contract):
    name: SafeId
    target_modules: Annotated[tuple[Nonempty, ...], Field(min_length=1)]


class ChainPlan(Contract):
    experiment_id: SafeId
    samples: Annotated[int, Field(gt=0)]
    nodes: Annotated[int, Field(gt=0)]
    batches: Annotated[tuple[ChainBatch, ...], Field(min_length=1)]
    arms: Annotated[tuple[ChainArm, ...], Field(min_length=1)]
    recipe: Annotated[tuple[str, ...], Field(min_length=1)]
    # "manual": every step after the first waits for an operator's approval in the control Dict.
    gate: Literal["manual", "none"]
    # What an unanswered gate does. Stopping releases a scarce allocation, so it is explicit.
    gate_timeout_action: Literal["stop", "continue_if_healthy"]
    gate_timeout_seconds: Annotated[int, Field(ge=60, le=21600)] = 3600
    step_attempts: Annotated[int, Field(ge=1, le=3)] = 2
    # A step starts only if the remaining function time covers it (the longest measured step wins).
    step_estimate_seconds: Annotated[int, Field(ge=60, le=6 * 3600)] = 3600
    failure_hold_seconds: Annotated[int, Field(ge=0, le=21600)] = 1800

    @model_validator(mode="after")
    def _distinct(self) -> "ChainPlan":
        if len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("Chain arm names must be unique")
        if len({frozenset(arm.target_modules) for arm in self.arms}) != len(self.arms):
            raise ValueError("Chain arms must differ in their target modules")
        for arm in self.arms:
            if len(set(arm.target_modules)) != len(arm.target_modules):
                raise ValueError("Repeated target module in chain arm")
        if len({b.path for b in self.batches}) != len(self.batches) or len({b.sha256 for b in self.batches}) != len(
            self.batches
        ):
            raise ValueError("Each chain step trains on a different batch; replay is not a chain step")
        return self


def validate_batches(mount: Path, plan: ChainPlan, source: RunConfig) -> tuple["Batch", ...]:
    """Every step's batch: pinned bytes, the authorized source, one behavior policy, disjoint
    groups, and a policy lag inside the source's staleness window."""
    from miles_plugins.proximal.offline_batch import validate_batch  # Requires the trainer dependency set.

    batches: list[Batch] = []
    seen_groups: set[str] = set()
    for lag, pinned in enumerate(plan.batches):
        bundle = mount / pinned.path
        if hashlib.sha256((bundle / "batch.json").read_bytes()).hexdigest() != pinned.sha256:
            raise ValueError(f"Chain batch {pinned.path} differs from the pinned manifest")
        batch = validate_batch(bundle)
        if batch.source != source or batch.num_samples != plan.samples:
            raise ValueError("Chain batches must use the authorized source and the plan's sample count")
        # The source contract authenticates this proof. Each arm's experimental target set is
        # deliberately separate and never substituted into the source metadata.
        verify_base_policy(batch.source, batch.policy, bundle / "base_policy")
        if batches and batch.policy != batches[0].policy:
            raise ValueError("Every chain batch must come from one exact behavior policy")
        if lag > source.research.max_policy_lag:
            raise ValueError(f"Step {lag + 1} would train at policy lag {lag}, outside max_policy_lag")
        groups = {index.header.group_id for index in batch.groups}
        if groups & seen_groups:
            raise ValueError("Chain batches must not share stored groups")
        seen_groups |= groups
        batches.append(batch)
    return tuple(batches)


def check_chain_recipe(recipe: tuple[str, ...]) -> None:
    """Continuation restores the LR scheduler from a run with a different length; only a constant,
    warmup-free schedule with an explicit override survives that unchanged."""
    check_recipe(recipe)

    def value(flag: str) -> str | None:
        return recipe[recipe.index(flag) + 1] if flag in recipe and recipe.index(flag) + 1 < len(recipe) else None

    if "--override-opt_param-scheduler" not in recipe:
        raise ValueError("A chain recipe needs --override-opt_param-scheduler to continue across steps")
    if value("--lr-decay-style") != "constant":
        raise ValueError("A chain recipe needs --lr-decay-style constant")
    for flag in ("--lr-warmup-iters", "--lr-warmup-samples", "--lr-warmup-fraction"):
        if value(flag) not in (None, "0", "0.0"):
            raise ValueError(f"A chain recipe cannot warm up the learning rate ({flag})")


def step_command(
    plan: ChainPlan,
    arm: ChainArm,
    step: int,
    source: RunConfig,
    *,
    bundle: Path,
    save: Path,
    resume_adapter: Path | None,
) -> list[str]:
    """Exactly one native update: rollout ``step`` of the arm, on ``plan.batches[step]``."""
    check_chain_recipe(plan.recipe)
    if not 0 <= step < len(plan.batches):
        raise ValueError("Chain step outside the plan")
    if (step > 0) != (resume_adapter is not None):
        raise ValueError("Every step after the first continues the previous step's verified native state")
    return native_train_argv(
        plan.recipe,
        source,
        nodes=plan.nodes,
        samples=plan.samples,
        target_modules=arm.target_modules,
        num_rollout=step + 1,
        start_rollout_id=step,
        rollout_function=CHAIN_REPLAY,
        batch_flag="--chain-batch",
        bundle=bundle,
        save=save,
        resume_adapter=resume_adapter,
    )


def validate_step_args(args: object, plan: ChainPlan, source: RunConfig, step: int) -> None:
    """Check parsed model/shape flags before starting Ray or any GPU workers."""
    for name, expected in {
        "num_rollout": step + 1,
        "start_rollout_id": step,
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
            raise ValueError(f"Chain step requires {name}={expected!r}, got {actual!r}")
