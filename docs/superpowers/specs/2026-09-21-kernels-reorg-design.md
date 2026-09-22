# Kernel tree reorganization

## Problem

Hand-written training kernels were filed by the model that first needed them, across six
unrelated locations and five backends (TileLang, Triton, CuTe DSL, raw CUDA, TE). The
glm5 / deepseek_v4 TileLang DSA fork is the direct result: the second model copied four kernel
files and diverged. Inkling's Triton activations, the fsdp MoE backward, and the fp8 / NVFP4 /
MXFP8 / INT4 quantization kernels each sat wherever their first caller lived.

## Design

Three layers, distinguishable from the import path alone:

| Layer | Path | Owns | May import |
|---|---|---|---|
| kernels | `miles/kernels/<op>/` | device kernel, `autograd.Function`, one entry function | torch, triton, tilelang, cutlass, fla, sglang kernels |
| shared modules | `miles_plugins/models/layers/` | parallel-aware `MegatronModule`s shared by several models (delta-rule attention, DSA attention) | Megatron, `miles.kernels` |
| model plugins | `miles_plugins/models/<model>/` | what is specific to one model: rope, compressor, layer spec | both above |

Rules for `miles/kernels` are in its README: filed by op, backend is a suffix, one entry
function per op, no Megatron or process group, variants are parameters, torch-reference test
per kernel.

## Phases

1. **Pure moves** (this branch). `git mv` every kernel into `miles/kernels`, rewrite imports,
   split the two Triton activations out of `miles_plugins/models/inkling/ops.py`. No logic
   change; verified by pre-commit and by running the existing kernel tests on a GPU box against
   `main`.
2. **DSA merge** (branch `zhichen/kernels-dsa-merge`, stacked on phase 1). The two copies turned
   out to be the same kernels: the indexer files were byte-identical and the v4 sparse-attention
   kernel is the glm5 one at `kv_group=1, tail_dim=0` plus a sink. `attention/dsa/tilelang/` now
   holds one pair; `tail_dim=0` and `has_sink` are compile-time parameters; the sink gradient is
   computed in torch from `delta` and `lse`, so no kernel accumulates it with atomics. The bshd
   batch loop and the causal-range helpers moved into the wrapper. Gated by torch-reference
   tests under `tests/fast-gpu/kernels/attention/dsa/` and a bitwise comparison against both old
   copies on a GPU box. Flag unification (`--dsv4-impl`, `--dsa-attention-backend`,
   `--dsa-kernel-backend`) touches the Megatron provider and launch scripts and is left for a
   later change.
3. **Delta-rule unification** (branch `zhichen/kernels-delta-rule`, stacked on phase 2).
   `miles_plugins/models/layers/delta_rule_attention.py` holds one head-sharded module for GDN and
   KDA: `LinearAttentionLayer` owns the HF input norm, the TP collectives (one in, one out, the
   Megatron attention pattern: identity/all-reduce or all-gather/reduce-scatter under SP) and the CP
   zigzag relayout; `DeltaRuleAttention` is the core (local head-sharded projections, one
   group-major conv, the fla kernel, a gated RMSNorm whose replicated weight gets a TP grad
   all-reduce through `copy_to_tensor_model_parallel_region`, row-parallel `out_proj`). Models
   subclass the core to declare projections under their HF names: `Qwen3_5GatedDeltaNet`,
   `Qwen3NextGatedDeltaNet`, and `KimiDeltaAttention` (K3 / GLM-5.3-flash layout, no consumer on
   main yet). Fused projection and conv rows are stored group-major in Megatron
   (`delta_rule_layout.py`), so a TP chunk is exactly a rank's heads for any TP size; the bridges
   permute on load and export. This replaces the TP-replicated GDN (`hf_attention.py`, deleted).
   Measured on H200, Qwen3.5-35B-A3B GDN layer, fwd+bwd, vs the replicated module: TP=2 1.7x faster
   and 1.8x less memory at 32k tokens, TP=4 2.7x / 3.2x; at 8k the layer is launch-bound and wall
   time is flat while GPU time still drops 1.7x. Checkpoint note: Megatron checkpoints of Qwen3.5 /
   Qwen3-Next GDN layers saved before this change have a different row layout for `in_proj_qkv`
   (Qwen3.5) and `conv1d` and cannot be resumed.

## File map, phase 1

| From | To |
|---|---|
| `miles_plugins/models/dsa_topk.py` | `miles/kernels/attention/dsa/topk.py` |
| `miles_plugins/models/glm5/ops/*` | `miles/kernels/attention/dsa/glm5/` |
| `miles_plugins/models/deepseek_v4/ops/kernel/tilelang_*` | `miles/kernels/attention/dsa/deepseek_v4/` (`tilelang_indexer.py` -> `indexer.py`, `tilelang_sparse_mla.py` -> `sparse_mla.py`) |
| `miles_plugins/models/deepseek_v4/ops/kernel/act_quant.py` | `miles/kernels/quant/fp8_act_quant.py` |
| `miles_plugins/models/deepseek_v4/ops/kernel/precision_aligned_ops.py` | `miles_plugins/models/deepseek_v4/ops/precision_aligned_ops.py` (model glue, stays in the plugin) |
| `miles_plugins/models/qwen_gdn_backend.py` | `miles/kernels/attention/delta_rule/backend.py` |
| `miles_plugins/models/inkling/ops.py` (Triton swiglu, sconv) | `miles/kernels/activation/{swiglu_fp32,short_conv_fp32}.py` |
| `miles/backends/fsdp_utils/sglang_attn_bridge/triton_attn_bwd.py` | `miles/kernels/attention/dense_bwd/triton_attn_bwd.py` |
| `miles/backends/fsdp_utils/kernels/*` | `miles/kernels/moe/` |
| `miles/utils/fp8_kernel.py` | `miles/kernels/quant/fp8_blockwise.py` |
| `miles/utils/fused_nvfp4_qdq.py` | `miles/kernels/quant/nvfp4_qdq.py` |
| `miles/utils/{nvfp4,nvfp4_fake_qat,mxfp8}.py` | `miles/kernels/quant/` |
| `miles/backends/megatron_utils/kernels/int4_qat/` | `miles/kernels/quant/int4_fake/` |
