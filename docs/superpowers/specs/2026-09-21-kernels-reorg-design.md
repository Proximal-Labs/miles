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
2. **DSA merge.** Fold `attention/dsa/glm5` and `attention/dsa/deepseek_v4` into one TileLang
   kernel pair taking `layout` and `attn_sink`. Gate with an equivalence test against both old
   copies before deleting them. Collapse `--dsv4-impl` and `--dsa-attention-backend` into one
   switch.
3. **Delta-rule unification.** One head-sharded module (Kimi-K3 layout) in
   `miles_plugins/models/layers/` selecting `chunk_gated_delta_rule` or `chunk_kda`; Qwen3.5,
   Qwen3-Next, Kimi-K3, GLM-5.3-flash point at it. Depends on `kimi-k3` landing.

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
