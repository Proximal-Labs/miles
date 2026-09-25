import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from miles.rollout.data_source import RolloutDataSource
from miles.rollout.inkling_sft import generate_rollout
from miles.rollout.inkling_sft_data_source import InklingSFTDataSource


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(rollout_id=3)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


def test_prepared_inkling_sft_bypasses_multimodal_processor(tmp_path, monkeypatch):
    metadata = {"format": "inkling-sft-v2", "tokens": [10, 20, 30], "loss_mask": [0, 1, 1]}
    dataset = tmp_path / "train.prepared.jsonl"
    second = {"format": "inkling-sft-v2", "tokens": [10, 20, 30, 40], "loss_mask": [0, 1, 1, 1]}
    dataset.write_text("\n".join(json.dumps({"text": "", "metadata": row}) for row in (metadata, second)) + "\n")
    processor_loader = Mock(return_value=object())
    monkeypatch.setattr("miles.rollout.data_source.load_processor", processor_loader)
    monkeypatch.setattr("miles.rollout.data_source.load_tokenizer", lambda *a, **kw: object())
    args = _make_args(
        rollout_global_dataset=True,
        hf_checkpoint="inkling",
        chat_template_path=None,
        dump_details=None,
        prompt_data=str(dataset),
        rollout_max_prompt_len=None,
        input_key="text",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_seed=42,
        buffer_filter_path=None,
        lora_adapter_path=None,
        n_samples_per_prompt=1,
        rollout_batch_size=1,
        seq_length=100,
        num_epoch=30,
    )
    source = InklingSFTDataSource(args)
    samples = generate_rollout(args, 0, source)
    processor_loader.assert_not_called()
    assert len(samples) == 1
    assert samples[0].tokens == metadata["tokens"]
    assert samples[0].loss_mask == [1, 1]
    assert samples[0].multimodal_inputs is None
    assert source.sample_offset == 1
    assert samples[0].metadata["_sft_progress"]["data/epoch"] == 0.5
    assert samples[0].metadata["_sft_progress"]["data/cumulative_target_tokens"] == 2

    source.args.save = str(tmp_path)
    source.save(0)
    resume_args = SimpleNamespace(**vars(args))
    resume_args.lora_adapter_path = str(tmp_path / "iter_0000000" / "adapter")
    resumed = InklingSFTDataSource(resume_args)
    resumed.load(0)
    next_sample = generate_rollout(args, 1, resumed)[0]
    assert next_sample.metadata["_sft_progress"]["data/epoch"] == 1.0
    assert next_sample.metadata["_sft_progress"]["data/cumulative_target_tokens"] == 5
    next_epoch = generate_rollout(args, 2, resumed)[0]
    assert next_epoch.metadata["_sft_progress"]["data/epoch"] == 1.5
    assert next_epoch.metadata["_sft_progress"]["data/cumulative_target_tokens"] == 7

    # Older checkpoints reconstruct the counter from the saved dataset cursor.
    source.metadata.clear()
    source.save(0)
    legacy = InklingSFTDataSource(resume_args)
    legacy.load(0)
    assert legacy.metadata["sft_target_tokens"] == 2

    # Ordinary rollout sources still load their multimodal processor.
    dataset.write_text(json.dumps({"text": [], "metadata": {}}) + "\n")
    monkeypatch.setattr("miles.utils.processing_utils.process_vision_info", lambda *a: {"images": []})
    regular_source = RolloutDataSource(args)
    processor_loader.assert_called_once_with("inkling", trust_remote_code=True)
    assert regular_source.dataset[0].multimodal_inputs == {"images": []}
