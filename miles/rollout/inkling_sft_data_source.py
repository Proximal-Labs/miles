"""Restore the SFT data cursor alongside adapters, independently of the base model."""

import copy
from pathlib import Path

from miles.rollout.data_source import RolloutDataSourceWithBuffer


class InklingSFTDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        source_args = copy.copy(args)
        source_args.load = str(Path(args.lora_adapter_path).parents[1]) if args.lora_adapter_path else None
        # Prepared text-only SFT records already contain tokens and loss masks.
        # Inkling's multimodal processor expects message lists, not their text placeholder.
        super().__init__(source_args, load_multimodal_processor=False)

    def get_samples(self, num_samples):
        groups = super().get_samples(num_samples)
        targets = sum(sum(sample.metadata["loss_mask"]) for group in groups for sample in group)
        self.metadata["sft_target_tokens"] = self.metadata.get("sft_target_tokens", 0) + targets
        epoch = self.epoch_id + self.sample_offset / len(self.dataset)
        progress = {
            "data/epoch": epoch,
            "data/progress_fraction": epoch / self.args.num_epoch,
            "data/cumulative_target_tokens": self.metadata["sft_target_tokens"],
        }
        for group in groups:
            for sample in group:
                sample.metadata["_sft_progress"] = progress
        return groups

    def load(self, rollout_id=None):
        if self.args.load is not None:
            state = Path(self.args.load) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
            if not state.is_file():
                raise FileNotFoundError(f"Cannot resume SFT without the matching dataset cursor: {state}")
        super().load(rollout_id)
        if "sft_target_tokens" not in self.metadata:
            # Recover progress from checkpoints written before this metric existed.
            counts = [sum(sample.metadata["loss_mask"]) for sample in self.dataset.samples]
            self.metadata["sft_target_tokens"] = self.epoch_id * sum(counts) + sum(counts[: self.sample_offset])
