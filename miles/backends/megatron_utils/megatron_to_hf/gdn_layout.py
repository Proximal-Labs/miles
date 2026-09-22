"""Undo the group-major row layout Megatron uses for head-sharded GDN weights (see
miles_plugins.models.layers.delta_rule_layout) when exporting to HF."""

from functools import cache

import torch

from miles.utils.hf_config import load_hf_config
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads, qkv_group_major_to_flat

# Both Qwen GDN families store the conv group-major; Qwen3.5 also regroups its flat in_proj_qkv,
# while Qwen3-Next's fused in_proj_qkvz is group-major in HF already.
GROUP_MAJOR_GDN_WEIGHTS = ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")


@cache
def _gdn_heads(hf_checkpoint: str) -> DeltaRuleHeads:
    config = load_hf_config(hf_checkpoint)
    text_config = getattr(config, "text_config", config)
    return DeltaRuleHeads(
        num_k_heads=text_config.linear_num_key_heads,
        num_v_heads=text_config.linear_num_value_heads,
        head_k_dim=text_config.linear_key_head_dim,
        head_v_dim=text_config.linear_value_head_dim,
    )


def gdn_weight_to_hf_layout(args, weight_name: str, param: torch.Tensor) -> torch.Tensor:
    """``weight_name`` is the mcore name below ``self_attention.``; non-GDN weights pass through."""
    if weight_name not in GROUP_MAJOR_GDN_WEIGHTS:
        return param
    return qkv_group_major_to_flat(param, _gdn_heads(args.hf_checkpoint))
