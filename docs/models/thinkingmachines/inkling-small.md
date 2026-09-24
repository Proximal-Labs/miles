---
title: Inkling-Small
description: Launch recipe for Inkling-Small (276 B), the compact sibling of Inkling — same architecture, 4-node H200 footprint.
---

## 1. Model Introduction

[Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) is the compact member of Thinking Machines Lab's Inkling family: a 276 B-total / 12 B-active-parameter, 42-layer multimodal MoE (256 routed + 2 shared experts, top-6 sigmoid routing) that matches — and on some benchmarks exceeds — the flagship 975 B model (e.g. SWEBench Verified 80.2 vs 77.6) at a fraction of the deployment footprint. The architecture is the same as [Inkling](/models/thinkingmachines/inkling) — ShortConv, local/global relative attention, and the shared-expert-sink MoE — so everything on the Inkling page (architecture summary, attention backends, R3 routing replay, LoRA schema, multimodal RL) applies unchanged. This page only covers what differs: the model registry entry and the validated small-cluster launch profile.

## 2. Supported Variants

| Model | Active / Total | Layers | HF ID | Recipe |
|---|---|---|---|---|
| Inkling-Small | 12 B / 276 B | 42 | [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) | this page |

## 3. Quick start

Validated on 4 nodes × 8 H200 (TP4 SP PP8 EP4, DP1):

```bash
cd /root/miles

# Full-parameter GRPO. 276 B fits with the CPU-offloaded optimizer -
# no NVMe streaming needed (unlike the 975 B recipe).
python scripts/run_inkling.py train \
   --model-name Inkling-Small --train-mode full --task dapo_math \
   --num-nodes 4 --num-gpus-per-node 8 \
   --lr 5e-5 --rollout-batch-size 64 --global-batch-size 128 \
   --sglang-context-length 4096 --rollout-max-response-len 2048 \
   --extra-args "--offload-train-target cpu --sglang-mem-fraction-static 0.65 \
      --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer"

# LoRA GRPO (rank 32, all-linear), same cluster; adapter-only weight
# sync swaps in ~4 s per rollout.
python scripts/run_inkling.py train \
   --model-name Inkling-Small --train-mode lora --task dapo_math \
   --num-nodes 4 --num-gpus-per-node 8 \
   --lr 2e-4 --rollout-batch-size 64 --global-batch-size 128 \
   --sglang-context-length 4096 --rollout-max-response-len 2048
```

The model definition lives in `scripts/models/inkling-small.py` (`MODEL_ARGS_NUM_LAYERS` overrides the layer count for sliced smoke/parity checkpoints). HF → `torch_dist` conversion uses the same tool as Inkling with this recipe file — a single 8-GPU node (TP8 EP8) converts it in one pass.

## 4. Validated parallelism

| Hardware | GPUs | TP | SP | PP | EP | expert-TP | Notes |
|---|---|---|---|---|---|---|---|
| H200 | 32 | 4 | on | 8 | 4 | 1 | `--decoder-last-pipeline-num-layers 7` (42 = 7×5 + 7) |

Batch shape is configurable from the launcher (`--rollout-batch-size`, `--global-batch-size`; defaults 32/64). The validated Small runs used 64/128: full with `--lr 5e-5`, LoRA with `--lr 2e-4` — both produce a steadily rising dapo-math reward curve. At the launcher's conservative LoRA default (5e-6) the zero-initialised B factors take hundreds of rollouts to accumulate a visible delta-W, which reads as "not learning".

## 5. Experimental text LoRA SFT on Modal

[`scripts/run_inkling_small_sft.py`](https://github.com/radixark/miles/blob/main/scripts/run_inkling_small_sft.py)
targets **one node of 8 B300s**, text LoRA SFT, one epoch, with
`dist_muon`. The BF16 base is frozen; only adapters are trained. Defaults are
rank **32**, alpha **32**, `all-linear` targets and shared-outer expert adapters,
matching the native Inkling adapter layout. Set `--lora-rank` and `--lora-alpha`
to override them. It starts no inference engines. This is an **unvalidated fit**,
especially at the default **262,144 total tokens per conversation**. The model's
1M context capability does not establish training memory feasibility. TP4/PP2/EP4,
sequence parallelism and full recomputation are enabled. Conversion and training
explicitly request FP32 routing (`--moe-router-dtype fp32`), matching the Inkling
model provider. Base weights, adapters, adapter gradients and optimizer state
stay in GPU memory; CPU and NVMe optimizer
offloading are disabled. LoRA reduces gradient/optimizer memory, but the full
base weights and long-context activations still require GPU memory.
An out-of-memory error fails the run without an offload fallback. No context
parallelism is used in this experimental profile. The current Modal launcher is
single-node; scaling to additional nodes requires a clustered launcher and an
updated parallelism layout.

The provisional LR is `1e-5`, with 3% warmup and cosine decay to `1e-6`; this is
configurable and is not a validated Inkling/Muon LoRA SFT optimum. Global and microbatch
size are both one conversation, so one epoch visits every prepared record without
dropping a partial batch. This favors memory feasibility over throughput.

### Modal configuration

Defaults are workspace/profile `proximal`, environment `main`, Volume
`inkling-small-rft`, and W&B entity/project
`evan-proximal-proximal/inkling-small-rft`. Use these Modal Secrets:

| Secret name | Environment variable | Access |
|---|---|---|
| `rft_hf_token` | `HF_TOKEN` | Read the model and any private dataset |
| `rft_wandb_api_key` | `WANDB_API_KEY` | Write runs to the W&B project |

The local machine needs Miles' CPU-side launcher dependencies and the Modal CLI;
authenticate the `proximal` profile before submitting. No Tinker key is needed.
The GPU function requests 512 GiB host RAM, with a
24-hour ceiling. The default base image is the pinned Linux AMD64 image
`radixark/miles@sha256:8ee6528fa209dd3bc65ccb40556e6606e3e9e502cd521d994d3ee6da3a58b67d`
(registry metadata: CUDA 13.0.3). The former `radixark/miles:inkling` tag is ARM64-only
and cannot run on Modal. Override with `--image` through the Miles launcher, or
`INKLING_MODAL_IMAGE` when invoking the Modal wrapper directly.
Before conversion or training, runtime checks require torch CUDA >=13.0 and eight B300s.
CUDA 13.0 is allowed as an experiment: NVIDIA lists B300 support in CUDA 13.0,
but [Modal documents CUDA 13.1+](https://modal.com/docs/guide/gpu#b300-gpus).
The image's full training stack and memory fit have **not** been verified on B300;
the smoke run must exercise forward/backward, optimizer updates and checkpointing.
The local checkout's training code is copied into the
image, so custom image and checkout versions must be compatible.

### Dataset format and reasoning

Store one complete conversation per JSONL line with `messages`, optional `tools`,
and optional numeric `reasoning_effort` (default `0.99`, max effort). Use standard role/content
messages with `reasoning_content`, `tool_calls`, `tool_call_id`, and tool `name`.
The preparation step uses Thinking Machines' official `TmlV0Renderer`, including
its unshifted per-token SFT weights. Historical reasoning remains in context and
is supervised alongside assistant text and tool calls. The official renderer owns
structural/end-of-turn supervision; system/user/tool-result tokens are masked.
Prepared `inkling-sft-v2` records store renderer versions, source revision when
available, checkpoint tokenizer hashes, and effective effort. Older v1 records
must be regenerated. Preparation checks token-ID round-tripping through the
checkpoint tokenizer; this does not prove fidelity to a historical rollout.
Tools are recorded training data; they are never executed by this recipe.

For readability this example is expanded; write it on one line in the JSONL:

```json
{
  "reasoning_effort": 0.99,
  "tools": [{"type": "function", "function": {
    "name": "lookup", "description": "Look up a value",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}
  }}],
  "messages": [
    {"role": "user", "content": "Look up the status of order 123."},
    {"role": "assistant", "content": null, "reasoning_content": "I need the order record.",
     "tool_calls": [{"id": "call_1", "type": "function",
       "function": {"name": "lookup", "arguments": {"key": "123"}}}]},
    {"role": "tool", "name": "lookup", "tool_call_id": "call_1", "content": "shipped"},
    {"role": "assistant", "content": "Order 123 has shipped."}
  ]
}
```

Text-only string content is required. Conversations must end with an assistant
target. Overlength records cause preparation to fail with a line number; nothing
is silently truncated or dropped. Optional assistant `step_loss_mask: 0` retains
that turn in context without supervising it.

Local CPU preparation requires `pip install -r tools/requirements-inkling-sft.txt`
(Python 3.11+ and PyTorch 2.10+). Modal installs these pinned rendering dependencies
in its data-preparation image; the GPU training image is unchanged.

### Prepare, measure, train

```bash
# Upload raw training conversations to the existing Volume.
MODAL_PROFILE=proximal modal volume put --env main inkling-small-rft train.jsonl /data/train.jsonl

# CPU job: download BF16 weights and prepare token IDs/masks plus length statistics.
python -m scripts.run_inkling_small_sft modal --mode data

# GPU job: convert HF weights into a Megatron distributed checkpoint.
# If the Volume already has the release marker, skip without allocating GPUs.
python -m scripts.run_inkling_small_sft modal --mode prepare

# GPU job: two optimizer steps on the longest real prepared conversation.
python -m scripts.run_inkling_small_sft modal --mode smoke --run-id fit-check

# Only after inspecting smoke memory, timing, checkpoint and loss results:
python -m scripts.run_inkling_small_sft modal --mode train --run-id sft-001

# Resume the same epoch schedule from its latest complete native adapter checkpoint.
python -m scripts.run_inkling_small_sft modal --mode train --run-id sft-001 --resume
```

Each submission is detached. A smoke run measures the longest *actual* record;
short data does not validate the configured 262K cap. Validation data and benchmark
evaluation are not wired yet. Output is native per-rank adapter checkpoints with
optimizer/scheduler state under `iter_XXXXXXX/adapter`, saved every 100 steps and
at the end, plus a launch configuration. The base model is not saved again at
each step. The backend also attempts an HF adapter export; native shards remain
the training-resume format. Resume reloads the original converted base and the
latest adapter directory containing all eight adapter and training-state shards.
The synchronous training loop saves the matching dataset cursor; resume requires
that cursor and reads it from the adapter run rather than the frozen base directory.
Use `--lora-adapter-path` to select a specific adapter directory; local `execute`
requires that path explicitly when resuming. Keep rank, alpha, topology, LR and
mode identical to the saved run. Existing full-parameter runs cannot be resumed
as LoRA runs; use a new run ID. The launcher does not prune checkpoints.
Persistent data and checkpoints use the Volume. Completed files are committed when a job exits; a
hard container failure may lose work after the last persisted checkpoint.
