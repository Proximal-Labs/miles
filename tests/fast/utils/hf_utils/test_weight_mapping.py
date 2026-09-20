import pytest
from transformers import Qwen3MoeConfig

from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora import validate_adapter_export


@pytest.fixture(scope="module")
def hf_mapping():
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
    )
    return HfWeightMapping.from_config(config)


_PREFIX = "model.layers.0.mlp.experts"
_TARGETS = [f"{_PREFIX}.gate_up_proj", f"{_PREFIX}.down_proj"]


def _unpacked_names():
    return {
        f"{_PREFIX}.{expert}.{projection}_proj.weight" for expert in range(2) for projection in ("gate", "up", "down")
    }


def test_checkpoint_formats_resolve_to_the_same_hf_parameters(hf_mapping):
    unpacked = _unpacked_names()
    assert {hf_mapping.model_parameter(name) for name in unpacked} == set(_TARGETS)
    hf_mapping.validate_coverage(unpacked, _TARGETS)
    hf_mapping.validate_coverage(_TARGETS, _TARGETS)


@pytest.mark.parametrize("missing", ["1.gate_proj.weight", "0.up_proj.weight", "1.down_proj.weight"])
def test_missing_expert_or_projection_is_rejected(hf_mapping, missing):
    names = _unpacked_names() - {f"{_PREFIX}.{missing}"}
    with pytest.raises(AssertionError, match="Incomplete HF"):
        hf_mapping.validate_coverage(names, _TARGETS)


def test_missing_entire_projection_is_rejected(hf_mapping):
    names = {name for name in _unpacked_names() if ".up_proj." not in name}
    with pytest.raises(AssertionError, match="Incomplete HF projections"):
        hf_mapping.validate_coverage(names, _TARGETS)


def test_export_uses_hf_coverage_without_changing_adapter_keys(hf_mapping):
    weights = {
        f"base_model.model.{name.removesuffix('.weight')}.lora_{factor}.weight"
        for name in _unpacked_names()
        for factor in ("A", "B")
    }
    validate_adapter_export(weights, _TARGETS, hf_mapping=hf_mapping)
    missing = f"base_model.model.{_PREFIX}.1.up_proj"
    with pytest.raises(AssertionError, match="Incomplete HF"):
        validate_adapter_export(
            {name for name in weights if not name.startswith(missing + ".")}, _TARGETS, hf_mapping=hf_mapping
        )


def test_export_rejects_missing_factor_before_normalizing(hf_mapping):
    weights = {f"base_model.model.{_PREFIX}.0.gate_proj.lora_A.weight"}
    with pytest.raises(AssertionError, match="unpaired A/B"):
        validate_adapter_export(weights, _TARGETS, hf_mapping=hf_mapping)


def test_packed_peft_parameter_wrapper_names(hf_mapping):
    weights = {
        f"base_model.model.{module}.lora_{factor}.weight"
        for module in (_PREFIX + ".base_layer", _PREFIX)
        for factor in ("A", "B")
    }
    validate_adapter_export(weights, _TARGETS, hf_mapping=hf_mapping)
