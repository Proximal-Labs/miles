import copy

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

from miles.utils.hf_config import load_hf_config
from miles_plugins.models.layers.delta_rule_attention import (
    DeltaRuleAttention,
    DeltaRuleInputs,
    GatedDeltaRule,
    LinearAttentionLayer,
)
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


def gdn_heads(text_config) -> DeltaRuleHeads:
    return DeltaRuleHeads(
        num_k_heads=text_config.linear_num_key_heads,
        num_v_heads=text_config.linear_num_value_heads,
        head_k_dim=text_config.linear_key_head_dim,
        head_v_dim=text_config.linear_value_head_dim,
    )


class Qwen3_5GatedDeltaNet(DeltaRuleAttention):
    """Qwen3.5 / 3.6 / 3.8 GDN: separate ``in_proj_qkv`` (group-major rows, see delta_rule_layout),
    ``in_proj_z``, ``in_proj_b`` and ``in_proj_a``."""

    def _build_projections(self):
        hidden = self.config.hidden_size
        self.in_proj_qkv = self.sharded_linear("in_proj_qkv", hidden, self.heads.qkv_dim)
        self.in_proj_z = self.sharded_linear("in_proj_z", hidden, self.heads.value_dim)
        self.in_proj_b = self.sharded_linear("in_proj_b", hidden, self.heads.num_v_heads)
        self.in_proj_a = self.sharded_linear("in_proj_a", hidden, self.heads.num_v_heads)

    def project(self, x):
        mixed, _ = self.in_proj_qkv(x)
        local = self.local
        grouped = mixed.reshape(*mixed.shape[:-1], local.num_k_heads, local.group_qkv_dim)
        q, k, v = grouped.split([local.head_k_dim, local.head_k_dim, local.v_per_k * local.head_v_dim], dim=-1)
        z, _ = self.in_proj_z(x)
        b, _ = self.in_proj_b(x)
        a, _ = self.in_proj_a(x)
        return DeltaRuleInputs(q=q.flatten(-2), k=k.flatten(-2), v=v.flatten(-2), gate=z, beta_logits=b, decay=a)


class Attention(LinearAttentionLayer):
    def __init__(self, args, config, layer_number: int, cp_comm_type=None, pg_collection=None, name=None):
        del layer_number, cp_comm_type, name
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        text_config = _get_text_config(load_hf_config(args.hf_checkpoint))
        linear_attn = Qwen3_5GatedDeltaNet(
            config,
            heads=gdn_heads(text_config),
            rule=GatedDeltaRule(backend=args.linear_attention_backend, norm_activation=text_config.hidden_act),
            conv_kernel_size=text_config.linear_conv_kernel_dim,
            norm_eps=text_config.rms_norm_eps,
            tp_group=pg_collection.tp,
        )
        input_layernorm = Qwen3NextRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        super().__init__(config, linear_attn, input_layernorm, pg_collection, allgather_cp=args.allgather_cp)


def get_qwen3_5_spec(args, config, vp_stage):
    # always use the moe path for MoE models
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    text_config = _get_text_config(hf_config)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
