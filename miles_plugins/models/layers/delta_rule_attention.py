"""Head-sharded delta-rule attention (GDN, KDA) as Megatron modules.

The recurrence is per-head separable, so tensor parallelism shards heads. Communication follows
Megatron's attention block: one collective on the way in (identity forward / all-reduce backward, or
under sequence parallelism all-gather forward / reduce-scatter backward), local head-sharded input
projections and conv, the fla chunk kernel on local heads, a gated RMSNorm whose replicated weight
gets its gradient summed across TP, and one collective on the way out through the row-parallel
output projection. :class:`LinearAttentionLayer` is the ``self_attention`` drop-in that owns those
collectives and the context-parallel relayout; :class:`DeltaRuleAttention` is the recurrence core.
Models subclass the core to declare their input projections under the HF parameter names and pick a
:class:`DeltaRule`.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.fused_norm_gate import rms_norm_gated
from megatron.core.extensions.transformer_engine import TELinear, TERowParallelLinear
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint

from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype
from miles.backends.training_utils.cp_utils import build_fla_cp_context
from miles.kernels.attention.delta_rule.backend import get_chunk_gated_delta_rule, get_chunk_kda
from miles_plugins.models.cp_utils import packed_shard_to_zigzag, zigzag_to_packed_shard
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads, cat_group_major_qkv, split_group_major_qkv


@dataclass
class DeltaRuleInputs:
    """This rank's projections for one forward. ``q``/``k`` are ``[b, s, Gl * hk]``, ``v`` and ``gate``
    are ``[b, s, Hl * hv]``, ``beta_logits`` is ``[b, s, Hl]``; ``decay`` is ``[b, s, Hl]`` for GDN (the
    ``a`` projection) and ``[b, s, Hl * hv]`` for KDA (the raw forget gate)."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor
    beta_logits: torch.Tensor
    decay: torch.Tensor


def _linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out, bias = module(x)
    assert bias is None
    return out


class DeltaRule(ABC):
    """Which recurrence runs on the local heads and how its per-head parameters are shaped."""

    norm_activation: str
    # None keeps dt_bias in the model's params dtype (Qwen GDN); KDA holds it in fp32 like A_log.
    dt_bias_dtype: torch.dtype | None = None

    @abstractmethod
    def param_sizes(self, local: DeltaRuleHeads) -> tuple[int, int]:
        """Sizes of ``A_log`` and ``dt_bias`` on this rank."""

    @abstractmethod
    def __call__(self, q, k, v, inputs: DeltaRuleInputs, A_log, dt_bias, *, cu_seqlens, cp_context):
        """q/k ``[b, s, Hl, hk]`` (repeated to value heads), v ``[b, s, Hl, hv]`` -> ``[b, s, Hl, hv]``."""


class GatedDeltaRule(DeltaRule):
    def __init__(self, backend: str = "fla", norm_activation: str = "silu"):
        self.backend = backend
        self.kernel = get_chunk_gated_delta_rule(backend)
        self.norm_activation = norm_activation

    def param_sizes(self, local):
        return local.num_v_heads, local.num_v_heads

    def __call__(self, q, k, v, inputs, A_log, dt_bias, *, cu_seqlens, cp_context):
        beta = inputs.beta_logits.sigmoid()
        # fp32 keeps exp(A_log) finite under fp16 params.
        g = -A_log.float().exp() * F.softplus(inputs.decay.float() + dt_bias)
        if cp_context is not None and self.backend != "fla":
            raise NotImplementedError(f"GDN context parallelism requires the 'fla' backend, got {self.backend!r}.")
        if self.backend == "flashqla":
            q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
        out, _ = self.kernel(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            **({"cp_context": cp_context} if cp_context is not None else {}),
        )
        return out


class KimiDeltaRule(DeltaRule):
    norm_activation = "sigmoid"
    dt_bias_dtype = torch.float32

    def __init__(self, gate_lower_bound: float):
        self.kernel = get_chunk_kda()
        self.gate_lower_bound = gate_lower_bound

    def param_sizes(self, local):
        return local.num_v_heads, local.num_v_heads * local.head_v_dim

    def __call__(self, q, k, v, inputs, A_log, dt_bias, *, cu_seqlens, cp_context):
        # cu_seqlens and cp_context are mutually exclusive for chunk_kda; the context carries its own.
        boundaries = {"cp_context": cp_context} if cp_context is not None else {"cu_seqlens": cu_seqlens}
        out, _ = self.kernel(
            q=q,
            k=k,
            v=v,
            g=inputs.decay.reshape(v.shape),
            beta=inputs.beta_logits.float().sigmoid(),
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            **boundaries,
        )
        return out


class _ShardedShortConvolution(ShortConvolution):
    """fla's depthwise conv with its channel dim marked as TP-sharded for checkpoints and export."""

    def __init__(self, *args, tp_group, **kwargs):
        super().__init__(*args, **kwargs)
        self.tp_group = tp_group
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class DeltaRuleAttention(MegatronModule, ABC):
    """Recurrence core on this rank's heads: projections -> conv -> rule -> gated norm -> out_proj.

    Subclasses build their projections in ``_build_projections`` (called before the shared parameters
    so ``self.local`` and the linear helpers are available) and map them in ``project``. Projections
    are *local* linears over this rank's heads: the input already carries the TP collective, so no
    per-projection all-reduce is needed (Megatron fuses qkv for the same reason).
    """

    def __init__(
        self,
        config,
        heads: DeltaRuleHeads,
        rule: DeltaRule,
        conv_kernel_size: int,
        norm_eps: float,
        tp_group,
    ):
        super().__init__(config=config)
        self.tp_group = tp_group
        self.heads = heads
        self.local = heads.local(tp_group.size())
        self.rule = rule
        self.conv_kernel_size = conv_kernel_size
        self.norm_eps = norm_eps
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False
        device = torch.cuda.current_device()
        dtype = config.params_dtype

        self._sharded_linears: list[str] = []
        self._build_projections()

        # One conv over this rank's [q, k, v] heads, group-major (see delta_rule_layout).
        self.conv1d = _ShardedShortConvolution(
            hidden_size=self.local.num_k_heads * self.local.group_qkv_dim,
            kernel_size=conv_kernel_size,
            bias=False,
            activation="silu",
            device=device,
            dtype=dtype,
            tp_group=tp_group,
        )
        a_log_size, dt_bias_size = rule.param_sizes(self.local)
        self.A_log = nn.Parameter(torch.empty(a_log_size, dtype=torch.float32, device=device))
        mark_param_dtype(self.A_log, torch.float32)
        self.dt_bias = nn.Parameter(torch.empty(dt_bias_size, dtype=rule.dt_bias_dtype or dtype, device=device))
        if rule.dt_bias_dtype is not None:
            mark_param_dtype(self.dt_bias, rule.dt_bias_dtype)
        for param in (self.A_log, self.dt_bias):
            set_tensor_model_parallel_attributes(param, True, 0, 1)
        # Holds the replicated norm weight; the forward calls the functional form so the weight can
        # pass through a TP gradient all-reduce first.
        self.norm = FusedRMSNormGated(
            heads.head_v_dim, eps=norm_eps, activation=rule.norm_activation, device=device, dtype=dtype
        )
        # Real ``config`` here: under SP the row-parallel output reduce-scatters straight into the
        # sequence-parallel layout, otherwise it all-reduces.
        self.out_proj = TERowParallelLinear(
            heads.value_dim,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=tp_group,
        )

    def sharded_linear(self, name: str, input_size: int, output_size: int) -> TELinear:
        """A linear whose ``output_size`` rows are the whole tensor; this rank holds its
        ``1 / tp_size`` chunk. No communication: the input already went through the TP collective."""
        tp_size = self.tp_group.size()
        assert output_size % tp_size == 0, (name, output_size, tp_size)
        linear = TELinear(
            input_size,
            output_size // tp_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        # TELinear's duplicated mode marks the weight replicated (parallel_mode, tensor_model_parallel
        # and, under SP, a TP grad all-reduce); this weight is a head shard with a complete gradient,
        # so the export and checkpoint paths must see it as column-sharded on dim 0.
        linear.weight.parallel_mode = "column"
        linear.weight.tensor_model_parallel = True
        linear.weight.partition_dim = 0
        linear.weight.partition_stride = 1
        linear.weight.sequence_parallel = False
        self._sharded_linears.append(name)
        return linear

    def duplicated_linear(self, input_size: int, output_size: int) -> TELinear:
        return TELinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    @abstractmethod
    def _build_projections(self) -> None: ...

    @abstractmethod
    def project(self, x: torch.Tensor) -> DeltaRuleInputs: ...

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        head_sharded = {"A_log": self.A_log, "dt_bias": self.dt_bias}
        head_sharded.update({f"{name}.weight": getattr(self, name).weight for name in self._sharded_linears})
        sharded.update(
            make_sharded_tensors_for_checkpoint(
                head_sharded,
                prefix,
                {key: 0 for key in head_sharded},
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return sharded

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None, cp_context=None) -> torch.Tensor:
        """Standalone (non-SP) path: x ``[b, s, hidden]`` replicated across TP -> ``[b, s, hidden]``."""
        x = copy_to_tensor_model_parallel_region(x, group=self.tp_group)
        out, bias = self.out_proj(self.core(x, cu_seqlens, cp_context))
        assert bias is None
        return out

    def core(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None, cp_context=None) -> torch.Tensor:
        """x ``[b, s, hidden]`` (this rank's tokens, TP collective already applied) -> this rank's
        pre-``out_proj`` activation ``[b, s, local value_dim]``."""
        batch, seq_len, _ = x.shape
        inputs = self.project(x)

        mixed = cat_group_major_qkv(inputs.q, inputs.k, inputs.v, self.local)
        mixed, _ = self.conv1d(x=mixed, cu_seqlens=cu_seqlens, cp_context=cp_context)
        q, k, v = split_group_major_qkv(mixed, self.local)
        if self.local.v_per_k > 1:
            q = q.repeat_interleave(self.local.v_per_k, dim=2)
            k = k.repeat_interleave(self.local.v_per_k, dim=2)

        core = self.rule(q, k, v, inputs, self.A_log, self.dt_bias, cu_seqlens=cu_seqlens, cp_context=cp_context)

        # The norm weight is replicated but applied to this rank's heads only, so its gradient is a
        # partial sum; the copy op is an identity forward and an all-reduce backward.
        weight = copy_to_tensor_model_parallel_region(self.norm.weight, group=self.tp_group)
        core = rms_norm_gated(
            core.reshape(-1, self.heads.head_v_dim),
            inputs.gate.reshape(-1, self.heads.head_v_dim),
            weight,
            self.norm.bias,
            self.rule.norm_activation,
            eps=self.norm_eps,
        )
        return core.reshape(batch, seq_len, -1)


class LinearAttentionLayer(MegatronModule):
    """``self_attention`` drop-in wrapping a :class:`DeltaRuleAttention` with the HF layer's own input
    norm, the TP collectives, and the context-parallel relayout.

    ``allgather_cp``: the data pipeline already provides contiguous CP shards (``--allgather-cp``),
    so no zigzag relayout is needed.
    """

    def __init__(
        self,
        config,
        linear_attn: DeltaRuleAttention,
        input_layernorm: nn.Module,
        pg_collection: ProcessGroupCollection,
        allgather_cp: bool,
    ):
        super().__init__(config=config)
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.allgather_cp = allgather_cp
        self.input_layernorm = input_layernorm
        self.linear_attn = linear_attn
        # Under SP the norm sees this rank's tokens only, so Megatron must sum its weight gradient
        # across TP (the same attribute TE sets on its fused layernorm weights).
        for param in self.input_layernorm.parameters():
            param.sequence_parallel = self.sequence_parallel

    def _global_cu_seqlens(self, hidden_states, packed_seq_params):
        if packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None:
            return packed_seq_params.cu_seqlens_q
        total = hidden_states.shape[0] * self.cp_size
        return torch.tensor([0, total], dtype=torch.int32, device=hidden_states.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        x = self.input_layernorm(hidden_states)
        # One TP collective in: under SP all-gather here and reduce-scatter in backward, otherwise
        # identity here and all-reduce in backward. out_proj does the matching collective out.
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True, group=self.tp_group)
        else:
            x = copy_to_tensor_model_parallel_region(x, group=self.tp_group)

        global_cu_seqlens = self._global_cu_seqlens(x, packed_seq_params)
        relayout = self.cp_size > 1 and not self.allgather_cp
        if relayout:
            # CP tokens sit in ring attention's zigzag order; fla's CP kernels want a contiguous chunk
            x = zigzag_to_packed_shard(x, global_cu_seqlens, self.cp_group, self.cp_group.rank(), self.cp_size)
        cp_context = None
        cu_seqlens = global_cu_seqlens
        if self.cp_size > 1:
            # the context carries the rank-local boundaries; the global ones only located this shard
            cp_context = build_fla_cp_context(
                global_cu_seqlens, self.cp_group, self.linear_attn.conv_kernel_size, x.device
            )
            cu_seqlens = cp_context.cu_seqlens

        core = self.linear_attn.core(x.transpose(0, 1), cu_seqlens, cp_context).transpose(0, 1)

        if relayout:
            core = packed_shard_to_zigzag(core, global_cu_seqlens, self.cp_group, self.cp_group.rank(), self.cp_size)
        output, bias = self.linear_attn.out_proj(core)
        assert bias is None
        return output, None


class KimiDeltaAttention(DeltaRuleAttention):
    """KDA with the Kimi-K3 / GLM-5.3-flash projection layout: separate ``q_proj``/``k_proj``/``v_proj``,
    a low-rank forget gate ``f_b_proj(f_a_proj(x))``, ``b_proj`` for beta and ``g_proj`` for the output
    gate. Use it with :class:`KimiDeltaRule`. HF ships three convs (``q_conv1d`` .. ``v_conv1d``); the
    bridge concatenates them into the single group-major ``conv1d``."""

    def _build_projections(self):
        hidden, size = self.config.hidden_size, self.heads.value_dim
        self.q_proj = self.sharded_linear("q_proj", hidden, size)
        self.k_proj = self.sharded_linear("k_proj", hidden, size)
        self.v_proj = self.sharded_linear("v_proj", hidden, size)
        self.b_proj = self.sharded_linear("b_proj", hidden, self.heads.num_v_heads)
        self.g_proj = self.sharded_linear("g_proj", hidden, size)
        # The low-rank forget gate's first factor is replicated and feeds a head-sharded second
        # factor, so its gradient is a partial sum; the forward routes the weight through the TP
        # copy op (identity forward, all-reduce backward) like the norm weight.
        self.f_a_proj = self.duplicated_linear(hidden, self.heads.head_v_dim)
        self.f_a_proj.weight.sequence_parallel = False
        self.f_b_proj = self.sharded_linear("f_b_proj", self.heads.head_v_dim, size)

    def project(self, x):
        f_a_weight = copy_to_tensor_model_parallel_region(self.f_a_proj.weight, group=self.tp_group)
        forget, _ = self.f_b_proj(F.linear(x, f_a_weight))
        return DeltaRuleInputs(
            q=_linear(self.q_proj, x),
            k=_linear(self.k_proj, x),
            v=_linear(self.v_proj, x),
            gate=_linear(self.g_proj, x),
            beta_logits=_linear(self.b_proj, x),
            decay=forget,
        )
