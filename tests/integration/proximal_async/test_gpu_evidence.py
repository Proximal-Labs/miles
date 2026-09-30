"""Negative controls for the live checkpoint comparison's exactness claims."""

import copy

import numpy as np
import torch

from miles_plugins.proximal.e2e.state_gpu_compare import compare_values


def test_comparison_detects_precision_lost_below_bf16_weights():
    master = torch.tensor([1.0001, -0.10001], dtype=torch.float32)
    state = {"model": master.bfloat16(), "optimizer": {"master": master, "step": 2}}
    assert compare_values(state, copy.deepcopy(state))["equal"]
    rounded = copy.deepcopy(state)
    rounded["optimizer"]["master"] = master.bfloat16().float()
    assert torch.equal(state["model"], rounded["model"])
    result = compare_values(state, rounded)
    assert not result["equal"]
    assert result["max_abs"] > 0
    assert "root/optimizer/master" in result["mismatches"]


def test_comparison_detects_missing_optimizer_and_rng_changes():
    state = {"model": torch.ones(2), "optimizer": {"step": 2}, "rng": np.array([1, 2, 3])}
    assert not compare_values(state, {"model": state["model"]})["equal"]
    changed = copy.deepcopy(state)
    changed["rng"][0] = 7
    assert not compare_values(state, changed)["equal"]


def test_nan_weights_cannot_pass_exact_comparison():
    state = {"model": torch.tensor([float("nan")])}
    assert not compare_values(state, copy.deepcopy(state))["equal"]
