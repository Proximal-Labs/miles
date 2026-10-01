import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from miles_plugins.inkling_eval.offline import _topology, export_native_adapter
from tools.inkling_checkpoint_recovery import checkpoint_files


@pytest.mark.parametrize('tp,ep,pp', [(8, 8, 2), (4, 4, 2), (4, 8, 2), (8, 4, 2), (2, 2, 4)])
def test_export_selects_tp_and_ep_shards_without_dp_duplicates(tmp_path, tp, ep, pp):
    config = dict(num_nodes=2, num_gpus_per_node=8, tensor_model_parallel_size=tp,
                  pipeline_model_parallel_size=pp, expert_model_parallel_size=ep,
                  model_dir=str(tmp_path / 'models'), lora_rank=1, lora_alpha=1)
    (tmp_path / 'launch.json').write_text(json.dumps(config))
    base = tmp_path / 'models/Inkling-Small'
    base.mkdir(parents=True)
    (base / 'config.json').write_text(json.dumps(dict(num_hidden_layers=4, vocab_size=16)))
    for path in checkpoint_files(0, 16):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'complete')

    expected = {}
    for rank in range(16):
        stage, local_rank = divmod(rank, 16 // pp)
        attention = 'module.decoder.layers.0.self_attention.lora_adapter.'
        experts = 'module.decoder.layers.0.mlp.experts.lora_adapter.'
        # Distinct values make wrong PP offsets, DP duplication and mixing TP
        # with EP groups visible in a bitwise comparison of the entire export.
        shards = {
            attention + 'wq_A': torch.tensor([[100 + stage]], dtype=torch.bfloat16),
            attention + 'wq_B': torch.tensor([[stage * 20 + local_rank % tp]], dtype=torch.bfloat16),
            experts + 'w1_A': torch.tensor([[200 + stage]], dtype=torch.bfloat16),
            experts + 'w1_B': torch.tensor([[[stage * 20 + local_rank % ep]]], dtype=torch.bfloat16),
        }
        torch.save(shards, tmp_path / f'iter_0000000/adapter/adapter_megatron_rank{rank}.pt')
        prefix = f'language_model.layers.{stage * (4 // pp)}.'
        expected[prefix + 'attn.wq_du.lora_A.weight'] = shards[attention + 'wq_A']
        expected[prefix + 'attn.wq_du.lora_B.weight'] = torch.tensor(
            [[stage * 20 + i] for i in range(tp)], dtype=torch.bfloat16)
        expected[prefix + 'mlp.experts.w1.lora_A.weight'] = shards[experts + 'w1_A'].unsqueeze(0)
        expected[prefix + 'mlp.experts.w1.lora_B.weight'] = torch.tensor(
            [[[stage * 20 + i]] for i in range(ep)], dtype=torch.bfloat16)

    reference = tmp_path / 'reference'
    reference.mkdir()
    save_file(expected, reference / 'adapter_model.safetensors')
    destination = tmp_path / 'export'
    export_native_adapter(tmp_path, 0, destination, reference=reference)
    actual = load_file(destination / 'adapter_model.safetensors')
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], tensor) for name, tensor in expected.items())
    assert (destination / '.complete').is_file()

    expected[next(iter(expected))].add_(1)
    save_file(expected, reference / 'adapter_model.safetensors')
    with pytest.raises(ValueError, match='Reference export differs'):
        export_native_adapter(tmp_path, 0, tmp_path / 'bad-export', reference=reference)
    assert not (tmp_path / 'bad-export/.complete').exists()


@pytest.mark.parametrize('override', [
    {'tensor_model_parallel_size': 3}, {'expert_model_parallel_size': 3},
    {'pipeline_model_parallel_size': 3}, {'context_parallel_size': 2},
    {'expert_tensor_parallel_size': 2}, {'use_tp_pp_dp_mapping': True},
    {'decoder_first_pipeline_num_layers': 1}, {'virtual_pipeline_model_parallel_size': 2},
])
def test_rejects_unsupported_layouts(override):
    config = dict(num_nodes=2, num_gpus_per_node=8, tensor_model_parallel_size=4,
                  pipeline_model_parallel_size=2, expert_model_parallel_size=4)
    with pytest.raises(ValueError):
        _topology(config | override, 42)
