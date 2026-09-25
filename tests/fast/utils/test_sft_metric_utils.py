from types import SimpleNamespace

import pytest

from miles.utils.sft_metric_utils import data_metrics, is_inkling_sft, perf_metrics


def test_data_counts_targets_not_response_span():
    samples = [
        SimpleNamespace(tokens=[1] * 10, loss_mask=[1, 0, 0, 1], metadata={}),
        SimpleNamespace(tokens=[1] * 20, loss_mask=[1, 1, 1], metadata={"_sft_progress": {"data/epoch": 0.5}}),
    ]
    result = data_metrics(samples, 3, 0.2)
    assert result["data/sequence_tokens/mean"] == 15
    assert result["data/target_tokens/mean"] == 2.5
    assert result["data/target_tokens/min"] == 2
    assert result["data/target_tokens/max"] == 3
    assert result["data/epoch"] == 0.5
    assert result["train/step"] == 3
    assert not any(key.startswith("rollout/") for key in result)


def test_perf_excludes_rl_and_estimated_flops():
    result = perf_metrics(
        {"actor_train": 2, "train": 2.1, "train_wait": 0.9, "update_weights": 5, "save_model": 0.5},
        [10, 20],
        4,
    )
    assert result == {
        "train/step": 4,
        "perf/train_time": 2,
        "perf/train_tok_per_s": 15,
        "perf/train_wait_time": 0.9,
        "perf/save_model_time": 0.5,
        "perf/step_time": 3,
        "perf/wait_time_ratio": pytest.approx(0.3),
    }


def test_metrics_scope_does_not_change_rl_or_other_sft_recipes():
    assert is_inkling_sft(SimpleNamespace(rollout_function_path="miles.rollout.inkling_sft.generate_rollout"))
    assert not is_inkling_sft(SimpleNamespace(loss_type="sft_loss"))
    assert not is_inkling_sft(SimpleNamespace(rollout_function_path="miles.rollout.sglang_rollout.generate_rollout"))
