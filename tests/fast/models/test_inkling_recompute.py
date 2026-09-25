"""Exercise config wiring and checkpoint autograd without the CUDA-only imports."""

import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _load_functions(filename, names, **dependencies):
    path = Path(__file__).resolve().parents[3] / "miles_plugins/models/inkling" / filename
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, **dependencies}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("granularity,method,count", [(None, None, None), ("full", "uniform", 1)])
def test_provider_passes_recompute_settings_into_transformer_config(tmp_path, monkeypatch, granularity, method, count):
    text_config = dict.fromkeys(
        (
            "intermediate_size",
            "n_shared_experts",
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "n_routed_experts",
            "num_experts_per_tok",
            "rms_norm_eps",
            "vocab_size",
        ),
        8,
    )
    (tmp_path / "config.json").write_text(json.dumps({"text_config": dict(text_config)}))
    args = SimpleNamespace(
        hf_checkpoint=str(tmp_path),
        tensor_model_parallel_size=4,
        expert_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        bf16=True,
        sequence_parallel=True,
        max_position_embeddings=262144,
        recompute_granularity=granularity,
        recompute_method=method,
        recompute_num_layers=count,
    )
    training = ModuleType("megatron.training")
    training.get_args = lambda: args
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    namespace = _load_functions(
        "model.py",
        {"build_inkling_config", "inkling_model_provider"},
        TransformerConfig=SimpleNamespace,
        InklingExtra=lambda _: SimpleNamespace(dense_mlp_idx=0),
        InklingGPTModel=lambda **kwargs: SimpleNamespace(**kwargs),
        get_inkling_block_spec=lambda *args, **kwargs: None,
    )
    model = namespace["inkling_model_provider"]()
    assert model.config.recompute_granularity == granularity
    assert model.config.recompute_method == method
    assert model.config.recompute_num_layers == count


def test_wrapped_provider_preserves_checkpointed_adapter_gradients():
    model = nn.Module()
    model.embedding = nn.Embedding.from_pretrained(torch.ones(8, 4), freeze=True)
    model.adapter = nn.Parameter(torch.ones(4, 4))
    model.config = SimpleNamespace(recompute_granularity="full")
    model.pre_process = True
    namespace = _load_functions(
        "lora.py",
        {"_enable_full_recompute_input_grads", "wrap_model_provider_with_inkling_lora"},
        apply_inkling_lora=lambda model, args: model,
    )
    wrapped = namespace["wrap_model_provider_with_inkling_lora"](lambda: model, None)
    assert wrapped() is model
    hidden = model.embedding(torch.tensor([[1, 2]]))
    checkpoint(lambda inputs: inputs @ model.adapter, hidden, use_reentrant=True).sum().backward()
    torch.testing.assert_close(model.adapter.grad, torch.full_like(model.adapter, 2.0))
    assert model.embedding.weight.grad is None
    with torch.no_grad():
        assert not model.embedding(torch.tensor([[1]])).requires_grad
    model.eval()
    assert not model.embedding(torch.tensor([[1]])).requires_grad


@pytest.mark.parametrize("granularity,pre_process", [(None, True), ("selective", True), ("full", False)])
def test_hook_skips_non_checkpointed_or_non_first_pipeline_stages(granularity, pre_process):
    model = SimpleNamespace(config=SimpleNamespace(recompute_granularity=granularity), pre_process=pre_process)
    namespace = _load_functions("lora.py", {"_enable_full_recompute_input_grads"})
    namespace["_enable_full_recompute_input_grads"](model)
