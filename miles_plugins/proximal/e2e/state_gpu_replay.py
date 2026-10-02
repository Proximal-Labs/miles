"""Two-update reference experiment using the exact stored complete-group bytes.

Deliberate replay belongs only to this diagnostic. Production FrozenBatchRolloutFn
continues to enforce one use. Miles owns postprocessing and every optimizer step.
"""

import argparse
from pathlib import Path

from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput, RolloutFnInput, RolloutFnTrainOutput
from miles.utils.types import Sample
from miles_plugins.proximal.contracts import pinned_dataset
from miles_plugins.proximal.data_source import ConsumedGroup, Cursor
from miles_plugins.proximal.offline_batch import Batch, load_training_group, training_groups, validate_batch
from miles_plugins.proximal.storage import write_atomic


def replay_groups(root: Path, batch: Batch) -> list[list[Sample]]:
    """The batch's training groups with batch-local framework IDs; evidence keeps the originals."""
    groups = []
    for group_index, selection in enumerate(training_groups(batch)):
        group = load_training_group(root, batch, selection)
        for offset, sample in enumerate(group):
            sample.group_index = group_index
            sample.index = group_index * len(group) + offset
            sample.rollout_id = None
        groups.append(group)
    return groups


def write_replay_cursor(save: Path, batch: Batch, rollout_id: int) -> None:
    cursor = Cursor(
        dataset_sha256=pinned_dataset(batch.source.dataset).sha256,
        next_group=len(training_groups(batch)),
        pending_tasks=(),
        consumed=tuple(
            ConsumedGroup(group_id=g.header.group_id, policy_version=g.header.policy.version) for g in batch.groups
        ),
    )
    write_atomic(save / "rollout" / f"proximal_{rollout_id}.json", cursor.model_dump_json().encode())


class ReferenceReplay(BaseRolloutFn):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--verification-batch", type=Path, required=True)

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        self.args = input.args
        self.root = input.args.verification_batch
        self.batch = validate_batch(self.root)
        start = self.args.start_rollout_id
        if not self.args.debug_train_only or self.args.num_rollout != 2 or start not in (0, 1):
            raise ValueError("Reference replay requires a two-update train-only diagnostic")
        if (start == 1) != bool(self.args.lora_adapter_path):
            raise ValueError("Reference replay continuation requires an explicit native adapter")

    def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation or not self.args.start_rollout_id <= input.rollout_id < 2:
            raise ValueError("Reference experiment is exactly two training updates")
        return RolloutFnTrainOutput(samples=replay_groups(self.root, self.batch), metrics={})

    def save(self, rollout_id: int) -> None:
        write_replay_cursor(Path(self.args.save), self.batch, rollout_id)
