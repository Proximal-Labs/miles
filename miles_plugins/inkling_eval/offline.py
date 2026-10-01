"""CPU export of native Inkling-Small adapters using the saved launch topology.

No frozen model weights or optimizer tensors are needed. Before using this on a
run, compare an export with a serving snapshot produced by its training actors.
Supports balanced pipeline stages, CP=ETP=1 and Megatron's tp-cp-ep-dp-pp
rank order. TP and EP groups are selected independently from DP replica zero.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from miles_plugins.inkling_eval.export import _write_adapter
from tools.inkling_checkpoint_recovery import checkpoint_complete


def _topology(config, num_layers):
    world = config['num_nodes'] * config['num_gpus_per_node']
    tp = config['tensor_model_parallel_size']
    pp = config['pipeline_model_parallel_size']
    ep = config['expert_model_parallel_size']
    if any(type(size) is not int or size <= 0 for size in (world, tp, pp, ep, num_layers)):
        raise ValueError('Topology sizes and layer count must be positive integers')
    if world % (tp * pp) or world % (ep * pp) or num_layers % pp:
        raise ValueError('World size must divide into TP/EP groups and layers into balanced PP stages')
    if (any(config.get(k) is not None for k in (
            'decoder_first_pipeline_num_layers', 'decoder_last_pipeline_num_layers',
            'virtual_pipeline_model_parallel_size'))
            or config.get('context_parallel_size', 1) != 1
            or config.get('expert_tensor_parallel_size', 1) != 1
            or config.get('use_tp_pp_dp_mapping', False)):
        raise ValueError('Offline export requires balanced PP, CP=ETP=1 and tp-cp-ep-dp-pp rank order')
    return world, tp, pp, ep


def _merge_parameter(name, shards, layer_offset, unpadded_vocab):
    values = [shard[name] for shard in shards]
    local = values[0]
    if '.lora_lm_head_adapter.' in name:
        parameter = name.rsplit('.', 1)[1]
        value = local if parameter == 'head_A' else torch.cat(values, dim=0)[:unpadded_vocab]
        return f"language_model.lm_head.lora_{parameter[-1]}.weight", value
    match = re.fullmatch(r'(?:module\.)*decoder\.layers\.(\d+)\.(.+)\.lora_adapter\.(\w+)', name)
    if match is None:
        raise ValueError(f'Unsupported adapter parameter: {name}')
    layer, module, parameter = match.groups()
    prefix = f'language_model.layers.{int(layer) + layer_offset}.'
    projection, side = parameter.rsplit('_', 1)
    if module == 'self_attention':
        mapping = {'wq': 'wq_du', 'wk': 'wk_dv', 'wv': 'wv_dv', 'wr': 'wr_du', 'wo': 'wo_ud'}
        target = 'attn.' + mapping[projection]
        sharded = side == ('A' if projection == 'wo' else 'B')
        value = torch.cat(values, dim=1 if projection == 'wo' else 0) if sharded else local
    elif module == 'mlp':
        target = 'mlp.' + {'fc1': 'gate_up_proj', 'fc2': 'down_proj'}[projection]
        if parameter == 'fc1_B':
            halves = [v.chunk(2, dim=0) for v in values]
            value = torch.cat([torch.cat([v[i] for v in halves], dim=0) for i in range(2)], dim=0)
        elif parameter == 'fc2_A':
            value = torch.cat(values, dim=1)
        else:
            value = local
    elif module == 'mlp.experts':
        target = module + '.' + projection
        sharded = parameter in {'w1_B', 'w3_B', 'w2_A'}
        value = torch.cat(values, dim=0) if sharded else local.unsqueeze(0)
    elif module == 'mlp.shared_experts':
        target = module + '.' + projection
        if parameter in {'w1_B', 'w3_B', 'w2_A'}:
            dim = 1 if side == 'A' else 0
            value = torch.cat([torch.cat([v[i] for v in values], dim=dim)
                               for i in range(local.shape[0])], dim=dim)
        else:
            value = local
    else:
        raise ValueError(f'Unsupported adapter module: {module}')
    return f'{prefix}{target}.lora_{side}.weight', value


def export_native_adapter(root, iteration, destination, *, reference=None):
    root = Path(root)
    config = json.loads((root / 'launch.json').read_text())
    base = Path(config['model_dir']) / 'Inkling-Small'
    hf = json.loads((base / 'config.json').read_text())
    hf = hf.get('text_config') or hf
    world, tp, pp, ep = _topology(config, hf['num_hidden_layers'])
    if not checkpoint_complete(root, iteration, world):
        raise ValueError(f'Checkpoint {iteration} is incomplete')
    vocab = hf.get('unpadded_vocab_size') or hf['vocab_size']
    layers_per_stage = hf['num_hidden_layers'] // pp
    ranks_per_stage = world // pp
    tensors = {}
    for stage in range(pp):
        # PP is the slowest-changing rank dimension. Skip DP replicas rather
        # than concatenating them into duplicate tensor/expert partitions.
        first_rank = stage * ranks_per_stage
        shards = [torch.load(root / f'iter_{iteration:07d}/adapter/adapter_megatron_rank{rank}.pt',
                             map_location='cpu', weights_only=True, mmap=True)
                  for rank in range(first_rank, first_rank + max(tp, ep))]
        if any(shard.keys() != shards[0].keys() for shard in shards):
            raise ValueError('Native shard parameter names disagree')
        for name in shards[0]:
            group_size = ep if '.mlp.experts.lora_adapter.' in name else tp
            target, value = _merge_parameter(name, shards[:group_size], stage * layers_per_stage, vocab)
            if target in tensors:
                raise ValueError(f'Duplicate exported tensor: {target}')
            tensors[target] = value.to(torch.bfloat16).contiguous()
    if reference is not None:
        with safe_open(str(Path(reference) / 'adapter_model.safetensors'), framework='pt') as expected:
            if set(expected.keys()) != tensors.keys():
                raise ValueError('Reference export tensor names disagree')
            for name, value in tensors.items():
                if not torch.equal(value, expected.get_tensor(name)):
                    raise ValueError(f'Reference export differs: {name}')
        print(f'Validated all {len(tensors)} tensors against the training export', flush=True)
    args = SimpleNamespace(lora_rank=config['lora_rank'], lora_alpha=config['lora_alpha'], hf_checkpoint=str(base))
    _write_adapter(args, list(tensors.items()), destination)
    return str(destination)
