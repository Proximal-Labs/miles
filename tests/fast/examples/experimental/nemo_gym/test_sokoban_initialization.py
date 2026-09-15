"""Reject resumed or MTP-enabled initial weights in the model comparison."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.external_utils.model_args_utils import load_model_args

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "sokoban_checks", ROOT / "examples/experimental/nemo-gym/sokoban_training_checks.py"
)
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


def test_qwen_target_preset_preserves_every_other_model_flag():
    original = load_model_args("qwen3.6-35B-A3B").split()
    index = original.index("--mtp-num-layers")
    del original[index : index + 2]
    assert load_model_args("qwen3.6-35B-A3B-no-mtp").split() == original


def test_converted_initialization_requires_release_and_matching_source(tmp_path):
    args = Namespace(load=str(tmp_path), ref_load=str(tmp_path), hf_checkpoint="/original/hf")
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    provenance = tmp_path / "sokoban_initialization.json"
    tracker.write_text("release")
    provenance.write_text(json.dumps({"hf_checkpoint": args.hf_checkpoint, "mtp_enabled": False}))
    CHECKS._check_fresh_initialization(args)
    tracker.write_text("200")
    with pytest.raises(AssertionError, match="resume"):
        CHECKS._check_fresh_initialization(args)
    tracker.write_text("release")
    provenance.write_text(json.dumps({"hf_checkpoint": "/different/model", "mtp_enabled": False}))
    with pytest.raises(AssertionError, match="source differs"):
        CHECKS._check_fresh_initialization(args)
    provenance.write_text(json.dumps({"hf_checkpoint": args.hf_checkpoint, "mtp_enabled": True}))
    with pytest.raises(AssertionError, match="MTP"):
        CHECKS._check_fresh_initialization(args)
