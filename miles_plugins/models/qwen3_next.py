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
from miles_plugins.models.qwen3_5 import gdn_heads


class Qwen3NextGatedDeltaNet(DeltaRuleAttention):
    """Qwen3-Next GDN: fused ``in_proj_qkvz`` and ``in_proj_ba``. HF already stores both group-major
    (per key head ``[q, k, v_group, z_group]`` and ``[b_group, a_group]``), so the local rows of a
    column-parallel split are whole key-head groups."""

    def _build_projections(self):
        hidden = self.config.hidden_size
        self.in_proj_qkvz = self.sharded_linear("in_proj_qkvz", hidden, self.heads.qkv_dim + self.heads.value_dim)
        self.in_proj_ba = self.sharded_linear("in_proj_ba", hidden, 2 * self.heads.num_v_heads)

    def project(self, x):
        local = self.local
        mixed, _ = self.in_proj_qkvz(x)
        lead = mixed.shape[:-1]
        grouped = mixed.reshape(*lead, local.num_k_heads, local.group_qkv_dim + local.v_per_k * local.head_v_dim)
        v_group = local.v_per_k * local.head_v_dim
        q, k, v, z = grouped.split([local.head_k_dim, local.head_k_dim, v_group, v_group], dim=-1)
        ba, _ = self.in_proj_ba(x)
        b, a = ba.reshape(*lead, local.num_k_heads, 2 * local.v_per_k).split([local.v_per_k, local.v_per_k], dim=-1)
        return DeltaRuleInputs(
            q=q.flatten(-2),
            k=k.flatten(-2),
            v=v.flatten(-2),
            gate=z.flatten(-2),
            beta_logits=b.flatten(-2),
            decay=a.flatten(-2),
        )


class Attention(LinearAttentionLayer):
    def __init__(self, args, config, layer_number: int, cp_comm_type=None, pg_collection=None, name=None):
        del layer_number, cp_comm_type, name
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        hf_config = load_hf_config(args.hf_checkpoint)
        linear_attn = Qwen3NextGatedDeltaNet(
            config,
            heads=gdn_heads(hf_config),
            rule=GatedDeltaRule(backend=args.linear_attention_backend, norm_activation=hf_config.hidden_act),
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            norm_eps=hf_config.rms_norm_eps,
            tp_group=pg_collection.tp,
        )
        input_layernorm = Qwen3NextRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        super().__init__(config, linear_attn, input_layernorm, pg_collection, allgather_cp=args.allgather_cp)


def get_qwen3_next_spec(args, config, vp_stage):
    # always use the moe path
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
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(hf_config, "layer_types"):
        interval = getattr(hf_config, "full_attention_interval", 4)
        n = hf_config.num_hidden_layers
        hf_config.layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)]

    for layer_id in range(num_layers_to_build):
        if hf_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
