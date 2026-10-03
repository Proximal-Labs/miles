"""One stored batch for exactly one update: a single step of a batch chain.

The step index is Miles's rollout ID, so native saves land at ``iter_<step>`` and a
continued step starts from the previous step's native state. Each batch is used once.
"""

import argparse
from pathlib import Path

from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput, RolloutFnInput, RolloutFnTrainOutput
from miles_plugins.proximal.e2e.state_gpu_replay import replay_groups, write_replay_cursor
from miles_plugins.proximal.offline_batch import validate_batch


class ChainReplay(BaseRolloutFn):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--chain-batch", type=Path, required=True)

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        self.args = input.args
        self.root = input.args.chain_batch
        self.batch = validate_batch(self.root)
        self.step = self.args.start_rollout_id
        if not self.args.debug_train_only or self.step < 0 or self.args.num_rollout != self.step + 1:
            raise ValueError("A chain step is exactly one train-only update")
        if (self.step > 0) != bool(self.args.lora_adapter_path):
            raise ValueError("A continued chain step requires the previous step's native adapter")
        self._used = False

    def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation or input.rollout_id != self.step or self._used:
            raise ValueError("A chain step trains on its batch exactly once")
        self._used = True
        return RolloutFnTrainOutput(samples=replay_groups(self.root, self.batch), metrics={})

    def save(self, rollout_id: int) -> None:
        write_replay_cursor(Path(self.args.save), self.batch, rollout_id)
