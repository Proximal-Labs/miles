"""Two-update reference experiment using the exact stored complete-group bytes.

Deliberate replay belongs only to this diagnostic. Production FrozenBatchRolloutFn
continues to enforce one use. Miles owns postprocessing and every optimizer step.
"""

import argparse
from pathlib import Path

from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput, RolloutFnInput, RolloutFnTrainOutput
from miles_plugins.proximal.contracts import pinned_dataset
from miles_plugins.proximal.data_source import ConsumedGroup, Cursor
from miles_plugins.proximal.offline_batch import load_batch_group, validate_batch
from miles_plugins.proximal.storage import write_atomic


class ReferenceReplay(BaseRolloutFn):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--verification-batch", type=Path, required=True)

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        self.args = input.args
        self.root = input.args.verification_batch
        self.batch = validate_batch(self.root)
        if not self.args.debug_train_only or self.args.num_rollout != 2 or self.args.start_rollout_id != 0:
            raise ValueError("Reference replay requires a fresh two-update train-only diagnostic")

    def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation or input.rollout_id not in (0, 1):
            raise ValueError("Reference experiment is exactly two training updates")
        groups = []
        for group_index, index in enumerate(self.batch.groups):
            group = load_batch_group(self.root, self.batch, index)
            for offset, sample in enumerate(group):
                sample.group_index = group_index
                sample.index = group_index * len(group) + offset
                sample.rollout_id = None
            groups.append(group)
        return RolloutFnTrainOutput(samples=groups, metrics={})

    def save(self, rollout_id: int) -> None:
        cursor = Cursor(
            dataset_sha256=pinned_dataset(self.batch.source.dataset).sha256,
            next_group=len(self.batch.groups),
            pending_tasks=(),
            consumed=tuple(
                ConsumedGroup(group_id=g.header.group_id, policy_version=g.header.policy.version)
                for g in self.batch.groups
            ),
        )
        write_atomic(
            Path(self.args.save) / "rollout" / f"proximal_{rollout_id}.json", cursor.model_dump_json().encode()
        )
