"""Restore the SFT data cursor alongside adapters, independently of the base model."""

import copy
from pathlib import Path

from miles.rollout.data_source import RolloutDataSourceWithBuffer


class InklingSFTDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        source_args = copy.copy(args)
        source_args.load = str(Path(args.lora_adapter_path).parents[1]) if args.lora_adapter_path else None
        super().__init__(source_args)

    def load(self, rollout_id=None):
        if self.args.load is not None:
            state = Path(self.args.load) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
            if not state.is_file():
                raise FileNotFoundError(f"Cannot resume SFT without the matching dataset cursor: {state}")
        super().load(rollout_id)
