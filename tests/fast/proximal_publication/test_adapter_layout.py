"""What a published adapter may hold (miles_plugins.proximal.adapter_layout)."""

import pytest

from miles_plugins.proximal.adapter_layout import adapter_layout_problem, exported_leaves

RANK = 32
PREFIX = "base_model.model.model.language_model.layers"
ALL_LINEAR = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkvz",
    "in_proj_ba",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
MLP = ["gate_proj", "up_proj", "down_proj"]
# Qwen3.8-27B: every 4th layer is full attention, the rest Gated DeltaNet.
ATTENTION = {"q_proj": (5120, 12288), "k_proj": (5120, 1024), "v_proj": (5120, 1024), "o_proj": (6144, 5120)}
GDN = {
    "in_proj_qkv": (5120, 10240),
    "in_proj_z": (5120, 6144),
    "in_proj_b": (5120, 48),
    "in_proj_a": (5120, 48),
    "out_proj": (6144, 5120),
}
FFN = {"gate_proj": (5120, 17408), "up_proj": (5120, 17408), "down_proj": (17408, 5120)}


def adapter(layers: int = 8, *, mixers: bool = True, rank: int = RANK) -> dict[str, tuple[int, ...]]:
    """Megatron-Bridge's export of a Qwen3.8-shaped adapter, as tensor shapes."""
    shapes: dict[str, tuple[int, ...]] = {}
    for layer in range(layers):
        modules = {f"mlp.{leaf}": dims for leaf, dims in FFN.items()}
        if mixers:
            parent, mixer = ("self_attn", ATTENTION) if layer % 4 == 3 else ("linear_attn", GDN)
            modules |= {f"{parent}.{leaf}": dims for leaf, dims in mixer.items()}
        for module, (fan_in, fan_out) in modules.items():
            shapes[f"{PREFIX}.{layer}.{module}.lora_A.weight"] = (rank, fan_in)
            shapes[f"{PREFIX}.{layer}.{module}.lora_B.weight"] = (fan_out, rank)
    return shapes


def test_all_linear_and_mlp_only_adapters_serve():
    assert adapter_layout_problem(adapter(), serving_targets=ALL_LINEAR, rank=RANK) is None
    assert adapter_layout_problem(adapter(mixers=False), serving_targets=MLP, rank=RANK) is None


def test_a_dense_decoder_without_language_model_prefix_serves():
    shapes = {}
    for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        shapes[f"base_model.model.model.layers.0.self_attn.{leaf}.lora_A.weight"] = (RANK, 1024)
        shapes[f"base_model.model.model.layers.0.self_attn.{leaf}.lora_B.weight"] = (1024, RANK)
    assert adapter_layout_problem(shapes, serving_targets=["q_proj", "k_proj", "v_proj", "o_proj"], rank=RANK) is None


def test_an_mtp_layer_adapter_is_refused():
    # Runs 004-013 published these: SGLang files them under decoder layer 0 and they win.
    shapes = adapter(mixers=False)
    for leaf, (fan_in, fan_out) in FFN.items():
        shapes[f"base_model.model.mtp.layers.0.mlp.{leaf}.lora_A.weight"] = (RANK, fan_in)
        shapes[f"base_model.model.mtp.layers.0.mlp.{leaf}.lora_B.weight"] = (fan_out, RANK)
    problem = adapter_layout_problem(shapes, serving_targets=MLP, rank=RANK)
    assert problem is not None and "outside the text decoder" in problem and "mtp.layers.0" in problem


def test_a_vision_tower_adapter_is_refused():
    shapes = adapter(mixers=False) | {
        "base_model.model.model.visual.blocks.0.mlp.linear_fc1.lora_A.weight": (RANK, 1152),
        "base_model.model.model.visual.blocks.0.mlp.linear_fc1.lora_B.weight": (4304, RANK),
    }
    assert "outside the text decoder" in adapter_layout_problem(shapes, serving_targets=MLP, rank=RANK)


def test_an_adapter_for_an_untargeted_module_is_refused():
    problem = adapter_layout_problem(adapter(), serving_targets=MLP, rank=RANK)
    assert problem is not None and "outside the serving targets" in problem


@pytest.mark.parametrize(
    "missing", ["linear_attn.in_proj_z", "linear_attn.in_proj_a", "self_attn.k_proj", "mlp.up_proj"]
)
def test_a_partial_stacked_group_is_refused(missing):
    shapes = {name: shape for name, shape in adapter().items() if f".{missing}." not in name}
    assert "partial stacked groups" in adapter_layout_problem(shapes, serving_targets=ALL_LINEAR, rank=RANK)


def test_a_missing_lora_b_half_is_refused():
    shapes = adapter()
    del shapes[f"{PREFIX}.0.linear_attn.out_proj.lora_B.weight"]
    assert "lora_A or lora_B" in adapter_layout_problem(shapes, serving_targets=ALL_LINEAR, rank=RANK)


def test_a_target_that_matched_no_layer_is_refused():
    shapes = {name: shape for name, shape in adapter().items() if ".out_proj." not in name}
    problem = adapter_layout_problem(shapes, serving_targets=ALL_LINEAR, rank=RANK)
    assert problem is not None and "out_proj" in problem and "matched no layer" in problem


def test_a_wrong_rank_is_refused():
    assert "not rank 32" in adapter_layout_problem(adapter(rank=16), serving_targets=ALL_LINEAR, rank=RANK)


def test_an_empty_adapter_is_refused():
    assert adapter_layout_problem({}, serving_targets=MLP, rank=RANK) is not None


def test_one_member_of_a_stack_stands_for_the_stack():
    assert exported_leaves(["q_proj"]) == {"q_proj", "k_proj", "v_proj"}
    assert exported_leaves(["in_proj_qkvz", "out_proj"]) == {"in_proj_qkv", "in_proj_z", "out_proj"}
