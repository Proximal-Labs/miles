"""Freeze complete platform groups, then feed them to Miles without live services.

The bundle contains the existing group codec bytes, not another tensor format.
Its manifest is written last. No database, capture service or replica is needed
to validate or train a completed bundle. See docs/proximal/offline-batches.md.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, model_validator

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
from miles_plugins.proximal.state_artifacts import copy_verified, describe
from miles_plugins.proximal.state_checkpoints import CheckpointManifest, read_manifest
from miles_plugins.proximal.storage import write_immutable
from miles_plugins.proximal.store import GroupIndex, GroupRow, StoredGroup, decode_group

ROLLOUT = "miles_plugins.proximal.offline_batch.FrozenBatchRolloutFn"


class FrozenBatch(Contract):
    schema_version: Literal[1] = 1
    source: RunConfig
    policy: Policy
    num_samples: Positive
    groups: tuple[GroupIndex, ...]

    @model_validator(mode="after")
    def _complete(self) -> "FrozenBatch":
        if self.policy.run_id != self.source.run_id or self.policy.base_model != self.source.base_model:
            raise ValueError("Batch policy differs from source run/base")
        if len(self.groups) * self.source.research.group_size != self.num_samples:
            raise ValueError("Batch must contain the requested number of complete groups")
        ids = [g.header.group_id for g in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated group in frozen batch")
        contract = digest(training_contract(self.source))
        for group in self.groups:
            if group.header.policy != self.policy or group.header.contract_sha256 != contract:
                raise ValueError("Frozen batch requires one exact behavior policy and training contract")
        return self


def read_batch(bundle: Path) -> FrozenBatch:
    return FrozenBatch.model_validate_json((bundle / "batch.json").read_bytes())


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


def validate_batch(bundle: Path) -> FrozenBatch:
    """Validate one group at a time; never materialize the whole 1024-sample batch."""
    batch = read_batch(bundle)
    attempts: set[str] = set()
    for index in batch.groups:
        for sample in load_group(bundle, index, batch.source):
            attempt_id = accepted(sample).attempt.attempt_id
            if attempt_id in attempts:
                raise ValueError("Repeated rollout attempt in frozen batch")
            attempts.add(attempt_id)
    return batch


def freeze_batch(
    *, config: RunConfig, source_root: Path, group_ids: tuple[str, ...], policy: Policy, num_samples: int, out: Path
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
            index = GroupIndex(
                header=header, payload_sha256=file.sha256, created_at=datetime.fromtimestamp(0, timezone.utc)
            )
        if index.header.group_id != group_id or index.payload_sha256 != file.sha256:
            raise ValueError("Source group index/payload mismatch")
        indexes.append(index)
    batch = FrozenBatch(source=config, policy=policy, num_samples=num_samples, groups=tuple(indexes))
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
    write_immutable(out / "batch.json", batch.model_dump_json().encode())
    return batch


class FrozenBatchRolloutFn(BaseRolloutFn):
    """One finite batch at the existing RolloutFn seam; no producer or remote I/O."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--proximal-frozen-batch", type=Path, required=True)
        parser.add_argument("--proximal-frozen-checkpoint", type=Path, required=True)

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
            samples = load_group(self.bundle, index, self.batch.source)
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


def read_checkpoint(path: Path, batch: FrozenBatch) -> CheckpointManifest:
    checkpoint = read_manifest(path.parent.parent, path.name)
    if checkpoint.context.contract_sha256 != digest(training_contract(batch.source)):
        raise ValueError("Native checkpoint and frozen data have different training contracts")
    if checkpoint.context.run_id != batch.source.run_id:
        raise ValueError("Select a native checkpoint from the source run")
    current_version = checkpoint.step + 2
    if not current_version - batch.source.research.max_policy_lag <= batch.policy.version <= current_version:
        raise ValueError("Behavior policy is outside the selected checkpoint's staleness window")
    return checkpoint


def validate_input_args(args: argparse.Namespace, batch: FrozenBatch) -> None:
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


def validate_train_args(args: argparse.Namespace, batch: FrozenBatch, checkpoint: CheckpointManifest) -> None:
    validate_input_args(args, batch)
    required = {
        "num_rollout": checkpoint.step + 2,
        "start_rollout_id": checkpoint.step + 1,
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
    targets = args.target_modules
    if isinstance(targets, str):
        targets = targets.split(",")
    if tuple(targets or ()) != batch.source.research.lora.target_modules:
        raise ValueError("LoRA target modules differ from the frozen batch")
    if not args.save:
        raise ValueError("Independent training step requires an explicit --save destination")
    source = Path(args.proximal_frozen_checkpoint).resolve()
    save = Path(args.save).resolve()
    for protected in (source, Path(args.proximal_frozen_batch).resolve()):
        if save == protected or save in protected.parents or protected in save.parents:
            raise ValueError("Write the new checkpoint outside the immutable input directories")
    if save.exists() and any(save.iterdir()):
        raise ValueError("Independent step requires an empty --save destination")
    if Path(args.lora_adapter_path).resolve() != source / "checkpoint/adapter":
        raise ValueError("Training must load the verified native adapter, not a serving export")


def train_argv(bundle: Path, batch: FrozenBatch, checkpoint_path: Path) -> list[str]:
    """Reuse native checkpoint loading and Miles's driver for exactly one update."""
    checkpoint = read_checkpoint(checkpoint_path, batch)
    research = batch.source.research
    return [
        "--debug-train-only", "--disable-rollout-global-dataset",
        "--rollout-function-path", ROLLOUT,
        "--proximal-frozen-batch", str(bundle),
        "--proximal-frozen-checkpoint", str(checkpoint_path),
        "--hf-checkpoint", str(batch.source.tokenizer_path),
        "--load", str(batch.source.tokenizer_path),
        "--lora-adapter-path", str(checkpoint_path / "checkpoint/adapter"),
        "--lora-rank", str(research.lora.rank), "--lora-alpha", str(research.lora.alpha),
        "--lora-dropout", "0", "--target-modules", ",".join(research.lora.target_modules),
        "--train-backend", "megatron", "--megatron-to-hf-mode", "bridge",
        "--save-interval", "1", "--rollout-num-gpus", "0", "--num-rollout", str(checkpoint.step + 2),
        "--start-rollout-id", str(checkpoint.step + 1),
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
    freeze.add_argument("--group-ids", type=Path, required=True, help="JSON array, in requested batch order")
    freeze.add_argument("--policy", type=Path, required=True)
    freeze.add_argument("--samples", type=int, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--bundle", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--bundle", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True, help="Verified run-state checkpoint bundle")
    train.add_argument("--optimizer-state", choices=["resume"], required=True)
    train.add_argument("--yes-train", action="store_true", required=True)
    args, extra = parser.parse_known_args()
    if args.command != "train" and extra:
        parser.error(f"Unexpected arguments: {extra}")
    if args.command == "freeze":
        batch = freeze_batch(
            config=read_run_config(args.config),
            source_root=args.source_root,
            group_ids=tuple(json.loads(args.group_ids.read_text())),
            policy=Policy.model_validate_json(args.policy.read_bytes()),
            num_samples=args.samples,
            out=args.out,
        )
    elif args.command == "check":
        batch = validate_batch(args.bundle)
    else:
        batch = validate_batch(args.bundle)  # Before Ray, GPUs, or any resource initialization.
        argv = train_argv(args.bundle, batch, args.checkpoint)
        from miles.utils.arguments import parse_args

        sys.argv = ["train.py", *(extra[1:] if extra[:1] == ["--"] else extra), *argv]
        parsed = parse_args()  # type: ignore[no-untyped-call]
        validate_train_args(parsed, batch, read_checkpoint(args.checkpoint, batch))
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
