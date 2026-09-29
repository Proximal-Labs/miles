"""Freeze complete platform groups, then feed them to Miles without live services.

The bundle contains the existing group codec bytes, not another tensor format.
Its manifest is written last. No database, capture service or replica is needed
to validate or train a completed bundle. See docs/proximal/offline-batches.md.
"""

import argparse
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, assert_never

import torch
from pydantic import Field, TypeAdapter, model_validator

from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput, RolloutFnInput, RolloutFnTrainOutput
from miles.utils.types import Sample
from miles_plugins.proximal.buffer import accepted, validate_group
from miles_plugins.proximal.contracts import (
    Contract,
    Policy,
    Positive,
    RunConfig,
    SafeId,
    behavior_correction_args,
    behavior_correction_argv,
    digest,
    read_run_config,
    training_contract,
)
from miles_plugins.proximal.initial_policy import copy_base_policy, verify_base_policy
from miles_plugins.proximal.state_artifacts import copy_verified, describe
from miles_plugins.proximal.state_checkpoints import CheckpointManifest, read_manifest
from miles_plugins.proximal.store import GroupIndex, GroupRow, StoredGroup, decode_group

ROLLOUT = "miles_plugins.proximal.offline_batch.FrozenBatchRolloutFn"


class _BatchFields(Contract):
    source: RunConfig
    policy: Policy
    num_samples: Positive
    groups: tuple[GroupIndex, ...]

    @model_validator(mode="after")
    def _complete(self) -> "_BatchFields":
        if self.policy.run_id != self.source.run_id or self.policy.base_model != self.source.base_model:
            raise ValueError("Batch policy differs from source run/base")
        if len(self.groups) * self.source.research.group_size != self.num_samples:
            raise ValueError("Batch must contain the requested number of complete groups")
        ids = [g.header.group_id for g in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated group in frozen batch")
        return self


class FrozenBatch(_BatchFields):
    schema_version: Literal[1] = 1

    @model_validator(mode="after")
    def _single_source(self) -> "FrozenBatch":
        contract = digest(training_contract(self.source))
        for group in self.groups:
            if group.header.policy != self.policy or group.header.contract_sha256 != contract:
                raise ValueError("Frozen batch requires one exact behavior policy and training contract")
        return self


class AssembledBatch(_BatchFields):
    schema_version: Literal[2] = 2
    additional_sources: Annotated[tuple[RunConfig, ...], Field(min_length=1)]
    require_nonzero_reward_variance: bool

    @model_validator(mode="after")
    def _compatible_sources(self) -> "AssembledBatch":
        primary = training_contract(self.source)
        contracts = {digest(primary)}
        for source in self.additional_sources:
            contract = training_contract(source)
            # Compare every training field; only membership in the same project
            # may differ. This comparison never changes stored provenance.
            if (
                contract.dataset.project_id != primary.dataset.project_id
                or contract.model_copy(update={"dataset": primary.dataset}) != primary
                or source.research.max_policy_lag != self.source.research.max_policy_lag
            ):
                raise ValueError("Assembly sources differ beyond dataset membership")
            identity = digest(contract)
            if identity in contracts:
                raise ValueError("Repeated assembly source contract")
            contracts.add(identity)
        if {group.header.contract_sha256 for group in self.groups} != contracts:
            raise ValueError("Assembly needs exactly the sources referenced by its groups")
        if any(group.header.policy != self.policy for group in self.groups):
            raise ValueError("Assembly requires one exact behavior policy")
        return self


Batch = Annotated[FrozenBatch | AssembledBatch, Field(discriminator="schema_version")]


def batch_sources(batch: Batch) -> tuple[RunConfig, ...]:
    if isinstance(batch, FrozenBatch):
        return (batch.source,)
    if isinstance(batch, AssembledBatch):
        return (batch.source, *batch.additional_sources)
    assert_never(batch)


def group_source(batch: Batch, index: GroupIndex) -> RunConfig:
    matches = [s for s in batch_sources(batch) if digest(training_contract(s)) == index.header.contract_sha256]
    if len(matches) != 1:
        raise ValueError("Group must name exactly one original source contract")
    return matches[0]


def read_batch(bundle: Path) -> Batch:
    return TypeAdapter(Batch).validate_json((bundle / "batch.json").read_bytes())


def load_group(root: Path, index: GroupIndex, config: RunConfig) -> list[Sample]:
    path = root / "groups" / f"{index.header.group_id}.bin"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular group payload: {path}")
    row = GroupRow(index.header.group_id, index.header.policy.version, str(path), index.payload_sha256)
    header, samples = decode_group(path.read_bytes(), row=row, contract_sha256=digest(training_contract(config)))
    if header != index.header or validate_group(config, samples) != header.policy:
        raise ValueError("Group payload, index and acceptance evidence disagree")
    if any(accepted(s).attempt.group_id != header.group_id for s in samples):
        raise ValueError("Group evidence names a different group")
    return samples


def load_batch_group(root: Path, batch: Batch, index: GroupIndex) -> list[Sample]:
    samples = load_group(root, index, group_source(batch, index))
    if isinstance(batch, AssembledBatch) and batch.require_nonzero_reward_variance:
        # Same threshold and float64/sample std as Miles's standard group filter.
        rewards = torch.tensor([accepted(sample).grade.reward for sample in samples], dtype=torch.float64)
        if len(samples) < 2 or rewards.std().item() <= 1e-8:
            raise ValueError("Assembly selected a zero-variance reward group")
    return samples


def validate_batch(bundle: Path) -> Batch:
    """Validate one group at a time; never materialize the whole 1024-sample batch."""
    batch = read_batch(bundle)
    attempts: set[str] = set()
    for index in batch.groups:
        for sample in load_batch_group(bundle, batch, index):
            attempt_id = accepted(sample).attempt.attempt_id
            if attempt_id in attempts:
                raise ValueError("Repeated rollout attempt in frozen batch")
            attempts.add(attempt_id)
    return batch


def _write_manifest(batch: Batch, destination: Path) -> None:
    # Serialize the validated value, not a source manifest that could have changed.
    # Modal v1 does not support local write_immutable's hardlink operation.
    with tempfile.TemporaryDirectory(prefix="frozen-manifest-") as temporary:
        manifest = Path(temporary) / "batch.json"
        manifest.write_text(batch.model_dump_json())
        copy_verified(manifest, destination / "batch.json", describe(manifest, relative="batch.json"))


def publish_batch(bundle: Path, destination: Path, *, commit: Callable[[], None]) -> Batch:
    """Publish exact referenced bytes, commit, then publish/commit readiness last."""
    batch = validate_batch(bundle)
    for index in batch.groups:
        relative = f"groups/{index.header.group_id}.bin"
        source = bundle / relative
        file = describe(source, relative=relative)
        if file.sha256 != index.payload_sha256:
            raise ValueError("Source group changed during publication")
        copy_verified(source, destination / relative, file)
    if (bundle / "base_policy").exists():
        copy_base_policy(
            verify_base_policy(batch.source, batch.policy, bundle / "base_policy"), destination / "base_policy"
        )
    commit()
    _write_manifest(batch, destination)
    commit()
    return batch


class BatchSelection(Contract):
    bundle: Path
    group_ids: Annotated[tuple[SafeId, ...], Field(min_length=1)]


def assemble_batch(
    *,
    selections: tuple[BatchSelection, ...],
    num_samples: int,
    require_nonzero_reward_variance: bool,
    out: Path,
) -> AssembledBatch:
    """Copy explicit groups, preserving their source contracts and codec bytes."""
    if len(selections) < 2:
        raise ValueError("Assembly requires at least two explicit input selections")
    inputs = [(selection, read_batch(selection.bundle)) for selection in selections]
    sources: dict[str, RunConfig] = {}
    picked: list[tuple[Path, GroupIndex]] = []
    for selection, batch in inputs:
        protected = selection.bundle.resolve()
        target = out.resolve()
        if target == protected or target in protected.parents or protected in target.parents:
            raise ValueError("Assembly output must be separate from immutable inputs")
        if batch.policy != inputs[0][1].policy:
            raise ValueError("Assembly requires one exact behavior policy")
        indexes = {index.header.group_id: index for index in batch.groups}
        for group_id in selection.group_ids:
            if group_id not in indexes:
                raise ValueError(f"Selected group is absent from its input bundle: {group_id}")
            index = indexes[group_id]
            sources.setdefault(index.header.contract_sha256, group_source(batch, index))
            picked.append((selection.bundle, index))
    anchor = inputs[0][1].source
    anchor_digest = digest(training_contract(anchor))
    sources.pop(anchor_digest, None)
    result = AssembledBatch(
        source=anchor,
        additional_sources=tuple(sources.values()),
        policy=inputs[0][1].policy,
        num_samples=num_samples,
        groups=tuple(index for _, index in picked),
        require_nonzero_reward_variance=require_nonzero_reward_variance,
    )
    attempts: set[str] = set()
    # Verify all selected data before writing readiness or copying any payload.
    for root, index in picked:
        for sample in load_batch_group(root, result, index):
            identity = accepted(sample).attempt.attempt_id
            if identity in attempts:
                raise ValueError("Repeated rollout attempt in assembled batch")
            attempts.add(identity)
    base = inputs[0][0].bundle / "base_policy"
    if base.exists():
        copy_base_policy(verify_base_policy(anchor, result.policy, base), out / "base_policy")
    for root, index in picked:
        relative = f"groups/{index.header.group_id}.bin"
        file = describe(root / relative, relative=relative)
        if file.sha256 != index.payload_sha256:
            raise ValueError("Source group changed during assembly")
        copy_verified(root / relative, out / relative, file)
    _write_manifest(result, out)
    return result


def oldest_groups(config: RunConfig, source_root: Path, policy: Policy, num_samples: int) -> tuple[str, ...]:
    """Explicit selection from durable indexes, independent of a live Postgres."""
    if num_samples <= 0 or num_samples % config.research.group_size:
        raise ValueError("Select a positive number of complete groups")
    contract = digest(training_contract(config))
    indexes = [GroupIndex.model_validate_json(path.read_bytes()) for path in (source_root / "groups").glob("*.json")]
    matches = sorted(
        (index for index in indexes if index.header.policy == policy and index.header.contract_sha256 == contract),
        key=lambda index: (index.created_at, index.header.group_id),
    )
    count = num_samples // config.research.group_size
    if len(matches) < count:
        raise ValueError(f"Need {count} complete matching groups; only {len(matches)} are durably indexed")
    return tuple(index.header.group_id for index in matches[:count])


def freeze_batch(
    *,
    config: RunConfig,
    source_root: Path,
    group_ids: tuple[str, ...],
    policy: Policy,
    num_samples: int,
    out: Path,
    base_policy: Path | None = None,
) -> FrozenBatch:
    """Explicit ordered selection, never 'whatever files happen to be present'.

    source_root is artifacts/<run_id> on the state Volume (or the store mount).
    No live-policy or consumption table is changed: this is a new experiment.
    Legacy .bin payloads are supported; their missing indexes are reconstructed
    only in the new bundle, after checking the payload and acceptance evidence.
    """
    ids = TypeAdapter(tuple[SafeId, ...]).validate_python(group_ids)
    if num_samples <= 0 or len(ids) * config.research.group_size != num_samples or len(set(ids)) != len(ids):
        raise ValueError("Select exactly num_samples/group_size distinct complete groups")
    indexes = []
    for group_id in ids:
        path = source_root / "groups" / f"{group_id}.bin"
        file = describe(path, relative=f"groups/{group_id}.bin")
        index_path = path.with_suffix(".json")
        if index_path.exists():
            index = GroupIndex.model_validate_json(index_path.read_bytes())
        else:
            # Pre-#32 groups have the identical codec/header but no durable sidecar.
            with path.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "big")
                if not 0 < size < file.size_bytes - 8 or size > 16 * 1024 * 1024:
                    raise ValueError("Invalid legacy group header length")
                header = StoredGroup.model_validate_json(stream.read(size))
            index = GroupIndex(header=header, payload_sha256=file.sha256, created_at=datetime.fromtimestamp(0, UTC))
        if index.header.group_id != group_id or index.payload_sha256 != file.sha256:
            raise ValueError("Source group index/payload mismatch")
        indexes.append(index)
    batch = FrozenBatch(source=config, policy=policy, num_samples=num_samples, groups=tuple(indexes))
    if base_policy is not None:
        copy_base_policy(verify_base_policy(config, policy, base_policy), out / "base_policy")
    # Validate before copying; a failed selection cannot become a completed bundle.
    attempts: set[str] = set()
    for index in batch.groups:
        for sample in load_group(source_root, index, config):
            attempt = accepted(sample).attempt.attempt_id
            if attempt in attempts:
                raise ValueError("Repeated rollout attempt in frozen batch")
            attempts.add(attempt)
        relative = f"groups/{index.header.group_id}.bin"
        file = describe(source_root / relative, relative=relative)
        if file.sha256 != index.payload_sha256:
            raise ValueError("Source group changed during freeze")
        copy_verified(source_root / relative, out / relative, file)
    _write_manifest(batch, out)
    return batch


class FrozenBatchRolloutFn(BaseRolloutFn):
    """One finite batch at the existing RolloutFn seam; no producer or remote I/O."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--proximal-frozen-batch", type=Path, required=True)
        initialization = parser.add_mutually_exclusive_group(required=True)
        initialization.add_argument("--proximal-frozen-checkpoint", type=Path)
        initialization.add_argument("--proximal-frozen-fresh", action="store_true")

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        self.bundle = Path(input.args.proximal_frozen_batch)
        self.batch = read_batch(self.bundle)
        validate_input_args(input.args, self.batch)
        self._used = False

    def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation or self._used:
            raise ValueError("A frozen batch is one training input, with no eval or implicit replay")
        groups = []
        attempts: set[str] = set()
        for group_index, index in enumerate(self.batch.groups):
            samples = load_batch_group(self.bundle, self.batch, index)
            for offset, sample in enumerate(samples):
                attempt_id = accepted(sample).attempt.attempt_id
                if attempt_id in attempts:
                    raise ValueError("Repeated rollout attempt in frozen batch")
                attempts.add(attempt_id)
                # Batch-local framework IDs: original attempt/sample IDs remain in
                # the immutable evidence. A source resumed run may reuse indices.
                sample.group_index = group_index
                sample.index = group_index * self.batch.source.research.group_size + offset
                sample.rollout_id = None
            groups.append(samples)
        self._used = True
        return RolloutFnTrainOutput(samples=groups, metrics={})


def read_checkpoint(path: Path, batch: Batch) -> CheckpointManifest:
    checkpoint = read_manifest(path.parent.parent, path.name)
    if checkpoint.context.contract_sha256 != digest(training_contract(batch.source)):
        raise ValueError("Native checkpoint and frozen data have different training contracts")
    if checkpoint.context.run_id != batch.source.run_id:
        raise ValueError("Select a native checkpoint from the source run")
    current_version = checkpoint.step + 2
    if not current_version - batch.source.research.max_policy_lag <= batch.policy.version <= current_version:
        raise ValueError("Behavior policy is outside the selected checkpoint's staleness window")
    return checkpoint


def validate_input_args(args: argparse.Namespace, batch: Batch) -> None:
    """Data-plane constraints; model/checkpoint validation belongs to the launcher."""
    required = {
        "debug_train_only": True,
        "rollout_global_dataset": False,
        "rollout_function_path": ROLLOUT,
        "global_batch_size": batch.num_samples,
        "rollout_batch_size": len(batch.groups),
        "n_samples_per_prompt": batch.source.research.group_size,
        **behavior_correction_args(batch.source.research.behavior_correction),
    }
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"Frozen batch requires --{name.replace('_', '-')}={value!r}")
    if args.num_rollout - args.start_rollout_id != 1:
        raise ValueError("A frozen batch requires exactly one training iteration")


def validate_train_args(args: argparse.Namespace, batch: Batch, checkpoint: CheckpointManifest | None) -> None:
    validate_input_args(args, batch)
    fresh = bool(getattr(args, "proximal_frozen_fresh", False))
    if fresh != (checkpoint is None):
        raise ValueError("Select explicit fresh initialization or a verified native checkpoint")
    required = {
        "num_rollout": 1 if checkpoint is None else checkpoint.step + 2,
        "start_rollout_id": 0 if checkpoint is None else checkpoint.step + 1,
        "load": str(batch.source.tokenizer_path),
        "hf_checkpoint": str(batch.source.tokenizer_path),
        "data_source_path": "miles.rollout.data_source.RolloutDataSourceWithBuffer",
        "train_backend": "megatron",
        "lora_rank": batch.source.research.lora.rank,
        "lora_alpha": batch.source.research.lora.alpha,
        "lora_dropout": 0,
        "custom_reward_post_process_path": None,
        "custom_convert_samples_to_train_data_path": None,
        "save_interval": 1,
    }
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"Frozen batch requires --{name.replace('_', '-')}={value!r}")
    for name in (
        "fully_async",
        "rollout_external",
        "custom_weight_transfer_protocol_path",
        "load_debug_rollout_data",
        "load_debug_rollout_data_subsample",
        "dynamic_sampling_filter_path",
        "rollout_sample_filter_path",
        "partial_rollout",
        "eval_interval",
        "use_critic",
        "multi_lora",
        "indep_dp",
        "use_dynamic_global_batch_size",
        "no_load_optim",
        "no_load_rng",
        "no_save_optim",
        "no_save_rng",
        "finetune",
        "debug_disable_optimizer",
        "debug_rollout_only",
        "custom_tis_function_path",
    ):
        if getattr(args, name, None):
            raise ValueError(f"Unsupported frozen-batch option: --{name.replace('_', '-')}")

    if checkpoint is not None:
        if args.actor_num_nodes * args.actor_num_gpus_per_node != checkpoint.native.world_size:
            raise ValueError("Native checkpoint requires its original GPU world size")
        tokens = checkpoint.context.train_args
        if "--optimizer" not in tokens or tokens.index("--optimizer") + 1 == len(tokens):
            raise ValueError("Source checkpoint must record its optimizer algorithm explicitly")
        if args.optimizer != tokens[tokens.index("--optimizer") + 1]:
            raise ValueError("Cannot restore native optimizer state into a different optimizer algorithm")
        for name, value in checkpoint.native.layout.items():
            if int(getattr(args, name, 1) or 1) != value:
                raise ValueError(f"Native checkpoint requires original {name}={value}")
    else:
        verify_base_policy(batch.source, batch.policy, Path(args.proximal_frozen_batch) / "base_policy")
        if args.lora_adapter_path or args.lora_A_init_method != "xavier" or args.lora_B_init_method != "zero":
            raise ValueError(
                "Fresh training initializes a trainable LoRA from base; it never loads the serving adapter"
            )
    targets = args.target_modules
    if isinstance(targets, str):
        targets = targets.split(",")
    if tuple(targets or ()) != batch.source.research.lora.target_modules:
        raise ValueError("LoRA target modules differ from the frozen batch")
    if not args.save:
        raise ValueError("Independent training step requires an explicit --save destination")
    save = Path(args.save).resolve()
    inputs = [Path(args.proximal_frozen_batch).resolve(), Path(batch.source.tokenizer_path).resolve()]
    if checkpoint is not None:
        inputs.append(Path(args.proximal_frozen_checkpoint).resolve())
    for protected in inputs:
        if save == protected or save in protected.parents or protected in save.parents:
            raise ValueError("Write the new checkpoint outside the immutable input directories")
    if save.exists() and any(save.iterdir()):
        raise ValueError("Independent step requires an empty --save destination")
    if checkpoint is not None and Path(args.lora_adapter_path).resolve() != inputs[-1] / "checkpoint/adapter":
        raise ValueError("Training must load the verified native adapter, not a serving export")


def train_argv(bundle: Path, batch: Batch, checkpoint_path: Path | None, *, fresh: bool = False) -> list[str]:
    """Reuse native checkpoint loading and Miles's driver for exactly one update."""
    if fresh == (checkpoint_path is not None):
        raise ValueError("Choose exactly one of fresh base initialization or native resume")
    checkpoint = read_checkpoint(checkpoint_path, batch) if checkpoint_path is not None else None
    if fresh:
        verify_base_policy(batch.source, batch.policy, bundle / "base_policy")
    initialization = (
        [
            "--proximal-frozen-checkpoint",
            str(checkpoint_path),
            "--lora-adapter-path",
            str(checkpoint_path / "checkpoint/adapter"),
        ]
        if checkpoint_path is not None
        else ["--proximal-frozen-fresh"]
    )
    research = batch.source.research
    return [
        "--debug-train-only", "--disable-rollout-global-dataset",
        "--rollout-function-path", ROLLOUT,
        "--proximal-frozen-batch", str(bundle),
        *initialization,
        "--hf-checkpoint", str(batch.source.tokenizer_path),
        "--load", str(batch.source.tokenizer_path),
        "--lora-rank", str(research.lora.rank), "--lora-alpha", str(research.lora.alpha),
        "--lora-dropout", "0", "--target-modules", ",".join(research.lora.target_modules),
        "--train-backend", "megatron", "--megatron-to-hf-mode", "bridge",
        "--save-interval", "1", "--rollout-num-gpus", "0", "--num-rollout", str(1 if checkpoint is None else checkpoint.step + 2),
        "--start-rollout-id", str(0 if checkpoint is None else checkpoint.step + 1),
        "--rollout-batch-size", str(len(batch.groups)), "--global-batch-size", str(batch.num_samples),
        "--n-samples-per-prompt", str(research.group_size),
        "--rollout-max-response-len", str(research.sampling.max_tokens),
        "--rollout-max-context-len", str(research.sampling.max_sequence_tokens),
        *behavior_correction_argv(research.behavior_correction),
    ]  # fmt: skip


def main() -> None:
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--source-root", type=Path, required=True)
    selection = freeze.add_mutually_exclusive_group(required=True)
    selection.add_argument("--group-ids", type=Path, help="JSON array, in requested batch order")
    selection.add_argument("--oldest", action="store_true", help="Select the oldest complete matching groups")
    freeze.add_argument("--policy", type=Path, required=True)
    freeze.add_argument("--samples", type=int, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--base-policy", type=Path)
    check = commands.add_parser("check")
    check.add_argument("--bundle", type=Path, required=True)
    assembly = commands.add_parser("assemble")
    assembly.add_argument("--selection", type=Path, required=True, help="JSON array of bundle/group_ids selections")
    assembly.add_argument("--samples", type=int, required=True)
    assembly.add_argument("--out", type=Path, required=True)
    assembly.add_argument("--require-nonzero-reward-variance", action=argparse.BooleanOptionalAction, required=True)
    train = commands.add_parser("train")
    train.add_argument("--bundle", type=Path, required=True)
    initialization = train.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--checkpoint", type=Path, help="Verified run-state checkpoint bundle")
    initialization.add_argument("--fresh", action="store_true", help="New LoRA and optimizer on the pinned base")
    train.add_argument("--optimizer-state", choices=["resume"])
    train.add_argument("--recipe", type=Path, help="Saved JSON array of explicit Miles model/optimizer arguments")
    train.add_argument("--yes-train", action="store_true", required=True)
    args, extra = parser.parse_known_args()
    if args.command != "train" and extra:
        parser.error(f"Unexpected arguments: {extra}")
    batch: Batch
    if args.command == "freeze":
        config = read_run_config(args.config)
        policy = Policy.model_validate_json(args.policy.read_bytes())
        batch = freeze_batch(
            config=config,
            source_root=args.source_root,
            group_ids=(
                oldest_groups(config, args.source_root, policy, args.samples)
                if args.oldest
                else tuple(json.loads(args.group_ids.read_text()))
            ),
            policy=policy,
            num_samples=args.samples,
            out=args.out,
            base_policy=args.base_policy,
        )
    elif args.command == "assemble":
        batch = assemble_batch(
            selections=TypeAdapter(tuple[BatchSelection, ...]).validate_json(args.selection.read_bytes()),
            num_samples=args.samples,
            out=args.out,
            require_nonzero_reward_variance=args.require_nonzero_reward_variance,
        )
    elif args.command == "check":
        batch = validate_batch(args.bundle)
    else:
        batch = validate_batch(args.bundle)  # Before Ray, GPUs, or any resource initialization.
        extra = (TypeAdapter(list[str]).validate_json(args.recipe.read_bytes()) if args.recipe else []) + (
            extra[1:] if extra[:1] == ["--"] else extra
        )
        if args.fresh:
            if args.optimizer_state is not None or "--seed" not in extra:
                parser.error("--fresh needs an explicit Miles --seed and no --optimizer-state")
        elif args.optimizer_state != "resume":
            parser.error("--checkpoint requires --optimizer-state resume")
        argv = train_argv(args.bundle, batch, args.checkpoint, fresh=args.fresh)
        from miles.utils.arguments import parse_args

        sys.argv = ["train.py", *(extra[1:] if extra[:1] == ["--"] else extra), *argv]
        parsed = parse_args()  # type: ignore[no-untyped-call]
        if args.fresh:
            # These are Bridge's native initialization attributes, not Miles CLI
            # options. Explicit fresh mode always starts at the base function.
            parsed.lora_A_init_method, parsed.lora_B_init_method = "xavier", "zero"
        validate_train_args(parsed, batch, None if args.fresh else read_checkpoint(args.checkpoint, batch))
        # Existing Miles driver owns optimizer, advantage math, partitioning and save.
        import asyncio

        from train import train as run_train

        from miles.utils.tracking_utils.tracking import finish_tracking

        try:
            asyncio.run(run_train(parsed))  # type: ignore[no-untyped-call]
        finally:
            finish_tracking()  # type: ignore[no-untyped-call]
        return
    print(
        f"Validated {batch.num_samples} samples in {len(batch.groups)} groups; policy {batch.policy.snapshot.sha256}"
    )


if __name__ == "__main__":
    main()
