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

The GPU communication timeout defaults to 30 minutes (`distributed_timeout_minutes`
in the Modal config JSON). First-step FlexAttention compilation and autotuning can
leave later pipeline stages waiting; a cold compile can still exceed this timeout.

Modal persists TorchInductor (including FX graph, AOTAutograd and local autotuning)
and Triton caches under `/mnt/inkling/compile-cache/<image-hash>/` on the
`inkling-small-rft` volume. Ray workers inherit the cache settings from the image.
The existing final volume commit preserves completed cache entries even when
training fails. The first run still compiles; later runs can reuse compatible
entries. New shapes or changed code may compile again, and changing the image
reference selects a separate cache directory. This does not add a kernel warmup.

[`scripts/run_inkling_small_sft.py`](https://github.com/radixark/miles/blob/main/scripts/run_inkling_small_sft.py)
targets **one node of 8 B300s**, text LoRA SFT, 10 epochs by default, with
Miles' standard `adam` distributed optimizer. The BF16 base is frozen;
only adapters are trained. Fresh SFT runs explicitly start at rollout 0; the release
base checkpoint's iteration 0 does not count as a completed training rollout.
Resumed runs use the saved adapter's rollout cursor. Defaults are
rank **32**, alpha **32**, `all-linear` targets and shared-outer expert adapters,
matching the native Inkling adapter layout. Set `--lora-rank` and `--lora-alpha`
to override them. It starts no inference engines. This is an **unvalidated fit**,
especially at the default **262,144 total tokens per conversation**. The model's
1M context capability does not establish training memory feasibility. TP4/PP2/EP4,
sequence parallelism and full recomputation are enabled. The custom model provider
forwards the recomputation granularity, method and layer count into TransformerConfig.
The LoRA provider enables gradients on frozen embedding outputs during training so
reentrant checkpoints preserve adapter gradients without training the embeddings.
Conversion and training
explicitly request FP32 routing (`--moe-router-dtype fp32`), matching the Inkling
model provider. Base weights, adapters, adapter gradients and optimizer state
stay in GPU memory; CPU and NVMe optimizer
offloading are disabled. LoRA reduces gradient/optimizer memory, but the full
base weights and long-context activations still require GPU memory.
An out-of-memory error fails the run without an offload fallback. No context
parallelism is used in this experimental profile. The current Modal launcher is
single-node; scaling to additional nodes requires a clustered launcher and an
updated parallelism layout.

The provisional peak LR is `1e-5`. Linear warmup starts at zero and spans 10%
of the samples in **one epoch**, followed by cosine decay over the rest of the
10-epoch run to a floor of `1e-6`. The launcher passes a whole-run warmup fraction
of `0.1 / num_epoch`, rather than warming up for 10% of all 10 epochs.
Warmup is resolved at optimizer-step boundaries, so very small datasets do not
produce a multi-step ramp. Configure `lr`, `min_lr`, and `warmup_epoch_fraction`
as needed; these defaults are not a validated Inkling/Adam LoRA SFT optimum.

Global batch size and the SFT loader's batch size both default to 32 conversations.
Microbatch size is 1. With TP4, PP2, CP1 on eight GPUs, data parallel size is 1,
so each optimizer step accumulates 32 microbatches. Expert parallel size 4 shares
the existing GPU topology and does not create four independent data replicas.
Set `global_batch_size` and `num_epoch` in the Modal config JSON to override these.
Miles schedules `num_epoch * floor(dataset_size / global_batch_size)` full batches;
use a dataset divisible by 32 for exactly ten full passes. The earlier one-record
dataset is too small for batch size 32; use `global_batch_size: 1` for that test.
Trainer tracking clients flush before normal shutdown so final metrics upload
before Ray exits.

### SFT metrics

SFT logs `train/loss`, `train/grad_norm`, per-parameter-group learning rates, and
`train/step`. All data and performance charts also use `train/step` (zero-based).
Dataset statistics are `data/num_samples`, `data/sequence_tokens/mean`, and
`data/target_tokens/{mean,median,min,max}`; target counts include only loss-masked
assistant tokens. `data/epoch` reports fractional passes through the dataset,
`data/progress_fraction` divides that by the configured epoch count, and
`data/cumulative_target_tokens` counts target tokens read so far, including prior
epochs and resumed progress. These are dataset-consumption metrics, emitted before
the corresponding optimizer update, not proof that an update completed.

Performance metrics include `perf/train_time`, `perf/train_tok_per_s` (full sequence
tokens), `perf/data_load_time`, `perf/data_preprocess_time`, `perf/train_wait_time`,
`perf/step_time`, `perf/wait_time_ratio`, and `perf/save_model_time` when a previous
save was timed. Save timing is emitted with the next training batch. W&B system
telemetry remains enabled. Placeholder rewards, RL group statistics, generation
metrics, inference-cache metrics, duplicate response lengths, inference weight-sync
timing, and approximate FLOP/MFU metrics are excluded for this SFT recipe.
Validation loss, accuracy, and perplexity are not computed by this recipe.

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
short data does not validate the configured 262K cap. Optional Proximal environment
evaluation is described below. Output is native per-rank adapter checkpoints with
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

### Proximal environment evaluation

Supply a local JSON file with named sets of platform environment IDs:

```json
{
  "platform_url": "https://YOUR-PROXIMAL-API-HOST",
  "sets": {
    "coding": [116702, 33736],
    "heldout": [12345, 12346]
  },
  "rollouts_per_environment": 3,
  "max_concurrent_rollouts": 4,
  "modal_secret": "inkling-eval"
}
```

Replace the example IDs with your suites. IDs must be distinct within each set;
an environment may appear in multiple sets. The launcher uploads this configuration
with the run. Each ID resolves once to its latest pushed image with a digest and
source commit. The resolved image/commit and configuration are frozen in
`<save_dir>/evaluation/suite.json`; resume rejects changes to that contract.

```bash
python -m scripts.run_inkling_small_sft modal \
  --mode train --run-id sft-eval-001 \
  --data-dir /mnt/inkling/data/deepseek-distill \
  --eval-config eval.json --eval-every-n-epochs 1 --eval-rollouts-per-env 3
```

`--eval-every-n-epochs 0` disables **all** evaluation, including the baseline,
configuration loading, snapshot exports and platform submissions. Omitting
`--eval-config` also disables evaluation. Smoke mode never evaluates environments.
`--eval-rollouts-per-env` overrides `rollouts_per_environment` for every named set.
When omitted, the configuration value is used (default 1). For two sets of 20
environments, `--eval-rollouts-per-env 3` produces 120 rollouts per evaluation point.

With evaluation enabled, the baseline finishes before the first optimizer update.
After the first optimizer step reaching each scheduled dataset epoch boundary, the trainer saves a resumable
checkpoint and exports an immutable adapter. Training then continues while a
separate Modal inference deployment and Proximal sandboxes evaluate that adapter.
One evaluation point runs at a time; if it is still running at the next scheduled
point, training waits there. Shutdown drains the pending evaluation. There is no
extra off-cadence final evaluation.
For 190 examples and batch size 32, the first evaluation is after step 6
(192 examples consumed, epoch 1.0105); at batch size 1 it is after step 190.
The graphs use actual consumed-example epochs, including batches crossing an
epoch boundary. The existing training loop still determines the total step budget.

W&B logs `eval/coding/mean_reward`, `eval/coding/pass_rate`,
`eval/heldout/mean_reward`, etc., each against its own `eval/<set>/epoch` axis.
Each set also reports scored rollout counts, infrastructure failures, and
per-environment metrics. Verified zero rewards count as scores; failed executions
do not. `evaluation/step_XXXXXXXX/point.json` records every platform run ID and
result incrementally. Deterministic run IDs make submission retries idempotent.
Resume reattaches unfinished evaluations and reuses completed results.
W&B also receives a rollout-results table for each set. Set `platform_ui_url` to
your Proximal frontend URL to include clickable platform run URLs in those tables.

Create the named Modal secret in the selected environment with:

- `PROXIMAL_API_KEY`: access to the selected environments, run creation/querying,
  and admin access to `modal.inference.endpoints` LiveConfig.
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`: deploy/stop evaluation apps and commit
  the training Volume in the same workspace.
- `MODAL_INFERENCE_API_KEY`: the platform's Modal proxy credential in
  `Modal-Key.Modal-Secret` form, used for endpoint readiness checks.

Secrets are inherited by the trainer; they are not stored in the JSON config.
`api_key_env` can override the Proximal key's environment variable name.

Evaluation defaults to one separate `B300:8` BF16 inference replica (TP8), a
1,048,576-token context (matching the platform's Inkling context budget), `default`
harness, `MAX` reasoning effort, and Modal
sandboxes. The config supports `serving_gpu`, `serving_tp`, `context_length`,
`agent_type`, `reasoning_effort`, `deployment_config`, `timeout_seconds` (14,400),
and `poll_seconds` (15). Baseline and later points use the same BF16 base and LoRA
serving path, with strict adapter loading, shared outer LoRAs and virtual experts.
The existing shared NVFP4 endpoint is not modified. Each new endpoint is registered
under a unique name and removed after its runs finish, then its Modal app is stopped.
Snapshots are retained for audit. Failed orchestration retains its endpoint and
state so existing rollouts can finish and a resume can recover it.

This integration requires live validation of full Inkling adapter loading,
trainer/serving numerical agreement, and a real environment rollout before treating
its scores as validated. CPU orchestration tests do not establish GPU memory fit.
