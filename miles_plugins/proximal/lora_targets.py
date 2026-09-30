"""Megatron LoRA target names to the HF leaf names SGLang serves.

Miles resolved --target-modules through this conversion until upstream #3318 moved
target selection onto HF model layouts. The run config still names Megatron modules
(anchored paths such as ``language_model.decoder.layers.*.self_attention.in_proj``),
and replicas derive their serving targets from it without a trainer's args, so the
plugin keeps the conversion it was validated with.
"""

_MEGATRON_TO_HF_MODULES = {
    # Standard LoRA (merged layers)
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    "output_layer": ["lm_head"],
    # CanonicalLoRA (split layers)
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
    # GDN linear attention: SGLang serves the fused in_proj as two modules
    "in_proj": ["in_proj_qkvz", "in_proj_ba"],
}

# DeepSeek / Kimi MLA and the DSA indexer: Megatron-Bridge linear_* names to HF leaves.
_MEGATRON_MLA_TO_HF = {
    "linear_q_down_proj": "q_a_proj",
    "linear_kv_down_proj": "kv_a_proj_with_mqa",
    "linear_q_up_proj": "q_b_proj",
    "linear_kv_up_proj": "kv_b_proj",
    "linear_wq_b": "wq_b",
    "linear_wk": "wk",
    "linear_weights_proj": "weights_proj",
}


def convert_target_modules_to_hf(megatron_modules: list[str] | tuple[str, ...]) -> list[str]:
    """HF leaf names for Megatron LoRA targets, deduplicated in order.

    Dotted or wildcard paths map by their last segment; SGLang uses the result to
    choose adapter-buffer types, not to scope by layer. Unknown leaves pass through.
    """
    hf_modules: list[str] = []
    for module in megatron_modules:
        leaf = module.rsplit(".", 1)[-1]
        if leaf in _MEGATRON_MLA_TO_HF:
            hf_modules.append(_MEGATRON_MLA_TO_HF[leaf])
        else:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES.get(leaf, [leaf]))
    return list(dict.fromkeys(hf_modules))
