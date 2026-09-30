---
title: "Qwen3.8-27B on real platform rollouts"
description: "Qwen3.8-27B trained on real Proximal platform rollouts (DeepSWE tasks) from a Modal training node and replicas."
# Generated from examples/proximal/qwen38/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
The first run against the real Proximal platform. Miles trains `Qwen/Qwen3.8-27B` on DeepSWE feature tasks. Each rollout is a real mini-swe session in a platform sandbox: agent-px calls capture in the serving replica that holds the rollout's session, and the platform grades the rollout.

| Part | Here |
| --- | --- |
| Training node | Modal 1× 8 H200 (`modal_training` with `training.json`, platform `real`): Megatron LoRA trainer at TP 4, rollout store |
| Serving | Modal 2× H200 (`serving_app`), one replica per GPU. Each replica runs SGLang and a front process with the gateway and capture. |
| Platform | Production: runs created with the run config's platform key; rollout workers call the pool's URL, and sticky routing sends each rollout's calls to one replica |
| Tasks | `tasks-pilot.json`: 41 tasks from project 519 / DeepSWE. It combines "3.8 good passrate" (13), "3.5 flash medium" (8) and the first 20 of "3.5 flash hard". `tasks-hard.json` holds all 348 "3.5 flash hard" tasks: gpt-5.5 solves more than 75% of rollouts on the newest image, and gemini-3.5 solved 1–2 of 8. |

## Settings and why

- **Chat template and parsers:** the template family is `qwen38small`, which Miles's own tests use with the 27B. The replicas use the `qwen3_coder` tool-call parser and the `qwen3` reasoning parser. Thinking is on, with reasoning effort `xhigh`. Capture renders the run's effort through the template, and a run config naming an effort the template can't render (it takes `xhigh`, `medium` or `low`) is rejected.
- **LoRA on every linear layer of the text decoder (rank 32).**
  - **Targets.** The run config names the six Megatron modules by anchored path (`language_model.decoder.layers.*.…`):
    - attention `self_attention.linear_qkv`, which includes the output gate, and `self_attention.linear_proj`;
    - Gated DeltaNet `self_attention.in_proj` and `self_attention.out_proj`;
    - MLP `mlp.linear_fc1` and `mlp.linear_fc2`.

    Bare module names would also match the vision tower and any MTP layer.
  - **Mapping.** In Bridge mode the trainer uses Megatron's fused modules. Megatron-Bridge exports each under the checkpoint's own names: `q_proj` (gate rows included), `k_proj`, `v_proj`, and `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a`. SGLang stacks them back into `qkv_proj`, `in_proj_qkvz` and `in_proj_ba`, in the base weights' order. An earlier note gave "separate HF-style layers" as the reason to stay MLP-only; that described the raw `--spec` model, which Bridge mode never builds.
  - **Proof.** `miles_plugins/proximal/e2e/lora_parity.py` checks the mapping end to end at the trainer's TP 4, with a random nonzero adapter on every module:
    - **Export mapping, per module:** `merged − base` equals `(alpha/r)·B·A` from the published tensors to 1–6%, depending on the adapter's strength. That is bf16 rounding; a wrong mapping errs by about 140%.
    - **Serving:** the attention/GDN adapters reproduce their merged checkpoint exactly as closely as the MLP adapters runs already used (on Qwen3.8, residual 0.112 vs 0.111 of the adapter's effect). Swapping GDN q and k, or attention q and gate, is caught at 3.7× and 5.0× that residual.
    - **Coverage:** Qwen3.5-4B (GDN 2 value heads per key head) and Qwen3.8-27B (3).
  - **Resume.** Runs before this change trained MLP-only. Their checkpoints don't resume under these targets: the adapter parameters differ, and the load fails.
  - **Serving memory.** A replica's LoRA buffers grow by about 1.5 GiB (4 adapter slots), taken from the KV cache.
  - **Speculative decoding.** The NEXTN draft stays unadapted, so watch the speculative accept length.
- **An adapter is checked before it can serve.** SGLang files each adapter tensor under the first `layers.N.` in its name and keeps the last one written. Runs 004–013 published the MTP layer's adapter (`mtp.layers.0.mlp.*`) next to the decoder's, and SGLang served it as decoder layer 0's MLP adapter: those runs sampled from a policy that differed from the trained one at layer 0.
  - The MTP head is now off in training (#23).
  - The publisher refuses any adapter SGLang would serve differently from the trained one, at every publish including the startup one (`adapter_layout.py`). That covers tensors outside the text decoder layers, modules outside the serving targets, a missing lora_A or lora_B, a partial stacked group, and a target that matched no layer.
  - Replicas load adapters with `lora_strict_loading`, so a tensor that matches no target fails the load instead of being dropped.
- **Batch:** 8 tasks × 4 samples = 32 rollouts per step, one optimizer update per step. The Miles recipe uses 1 node × 8 GPUs at TP 4.
- **Limits:** mini-swe with `max_turns` 30, 8k tokens per turn, 32k per sequence, and 64 samples in flight. At that concurrency, per-attempt polling is cheap.
- **Routing:** `platform_route.endpoint_name` is unset, so the platform routes the model's calls to its *default* registry endpoint: the serving pool's URL, registered once per pool.

## Capture in the replicas

Capture keeps each rollout's exact token history and calls SGLang for every turn, so it runs where SGLang runs:
- **Per call**, the path is agent-px → the pool's URL → the replica's front process → SGLang on localhost. The training node is not on it. Measured with capture on the training node instead (which Modal placed in eu-north-1 while the replicas ran in us-west), each call paid about 1.6 s outside SGLang; about 0.9 s of that was capture's round trip to the replica.
- **Sticky routing.** Modal sends requests carrying the same `Modal-Session-Id` to the same container. The trainer and the platform's `rollout_capture` client both send the SHA-256 of the platform rollout ID (`contracts.AFFINITY_HEADER`). A call that reaches a replica without the rollout's session fails with 404 and the rollout is retried; the timing log records each call's container so the routing can be checked. With one replica routing is trivially sticky.
- **Once per rollout**, the trainer seals the session and fetches its samples from the replica, and writes them to its rollout store; it remains the store's only writer. Until fetched, sealed samples live only on the replica's disk: a replica lost in between loses those rollouts, and they are retried.
- **Abandoned sessions expire.** Replicas outlive the trainer, so a trainer that crashes leaves its sessions open. A session older than the rollout timeout, one model request and ten minutes is released when a new session opens.
- **Fixed pool size.** `min_replicas` equals `max_replicas`: scaling a replica down would drop the sessions it holds.
- **CPU.** Each replica requests `cpu` cores for SGLang's processes and the front process, and the front process runs at lower scheduling priority than SGLang, so capture never delays SGLang's step loop.
- **One run per pool.** Capture enforces the run's contract (tasks, harness, sampling), so the pool is deployed with the same run config as the trainer, and a changed run config means redeploying the pool.

## Checked without paid resources

- The run config and training deployment validate, including the chat-template parser check (`training check`).
- Miles's Qwen3.8-27B template and exact-token tests pass (61 tests).
- In the Miles image, Miles's parser accepts the full training command, and SGLang's `ServerArgs` accepts the replica's LoRA engine arguments.

Checked only on GPUs:
- Megatron loading this vision-language model through the bridge;
- exporting the MLP LoRA and the replicas loading it;
- memory at 32k tokens.

## Smoke first: one step on one task (`smoke/`)

Before the pilot, `smoke/` runs one training step with 1 task × 4 samples:
- **Trainer:** 4 H200 at TP 4, which is one data-parallel rank; the frozen 27B is about 14 GB of weights per GPU.
- **Serving:** 1 replica.
- **Task:** the first task of "3.8 good passrate".

The step checks, in order:
1. The replica loads the step-0 adapter (Miles publishes it before any rollout).
2. Qwen3.8's tool calls parse in mini-swe.
3. Platform rollout workers reach capture in the replica.
4. The platform grades the 4 rollouts and they land in the store.
5. Megatron loads the 27B and trains one LoRA step, then publishes version 1 and exits.

The smoke deployment has `max_retries` 0, so a failure stops the run rather than retrying it. Its registration timeout is 20 minutes. It uses its own run id (and so its own snapshots) and W&B group.

Use the `smoke/` files in steps 3–6 below: `smoke/serving.json`, `smoke/run.template.json` with `smoke/tasks.json`, and `smoke/training.json`. Volumes, secrets and the staged base are shared with the pilot.

## Overhead run (`overhead/`)

Measures what capture adds to each model call on real platform traffic, on the topology closest to the final one. The same four "3.5 flash hard" environments (`overhead/tasks.json`) run 8 rollouts each on every step, for 10 steps, with mini-swe allowed 200 turns:
- **Serving:** 8 replicas, each on 1 × B300 with an FP8 KV cache (the 27B plus about 3.2M tokens of KV), one inference GPU per training GPU. Capture runs in each, and the platform's rollout_capture client pins every rollout's calls to one replica (`Modal-Session-Id`, proximal-mono #4726). The 64 rollouts in flight grow about 2k tokens a turn toward 228k, so near their end they need about 12M tokens of KV: 8 per replica keeps each rollout's context cached with room for uneven routing. With 2 replicas (32 each) the cache filled by turn 25 and requests queued behind re-computed prefixes.
- **Serving settings** (`serving.json`, `attention` and `speculative`): trtllm_mha attention with 64-token KV pages, and speculative decoding with Qwen3.8's own MTP head (NEXTN, 3 draft steps) under SGLang's default verification, and an FP8 KV cache (`kv_cache_dtype`) with 2k prefill chunks: the per-GPU recipe of proximal-mono's `infra/modal/qwen3-8-27b` (PR #4841, BF16 weights and fp32 Gated DeltaNet state). FP8 KV is the one setting that changes numerics against the BF16 trainer; watch the rollout/train logprob gap. Left to itself SGLang serves this hybrid model on Blackwell with Triton attention and one-token pages. Measured 2026-09-25 by replaying 150 real agent turns (recorded prefills and output lengths, 1 s tool gaps, prompts ~55k tokens, a non-zero LoRA adapter), ten rollouts per GPU:

  | Serving | Model call p50 / p90 | Output tok/s per GPU |
  | --- | --- | --- |
  | SGLang's choice (run 003) | 9.4 / 46 s | ~91 |
  | trtllm_mha, one GPU | 4.7 / 24 s | 188 |
  | trtllm_mha, TP2 | 4.2 / 22 s | 259 |
  | trtllm_mha + MTP, one GPU | 2.4 / 12 s | 362 |
  | trtllm_mha + MTP, TP2 | 2.1 / 11 s | 526 |

  Sampling is checked, not assumed: 600 five-token samples per prompt against serving without speculation. With default verification no token fell outside what exact sampling produced and sampled-token logprobs stayed within noise. With `--speculative-use-rejection-sampling` about 2% of speculated tokens were ones the model gives logprob -24 to -34, random multilingual tokens mid-sentence ("These tests aren 则表示 about"), so that path is not used. Returned logprobs match teacher-forced ones either way.
- **Trainer:** 8 × B300 at TP 4 with two data-parallel ranks, with sequences up to 256k tokens.
  - **Memory.** A 262k-token micro-batch needs about 155 GB on one GPU, which ran out of memory on H200. The output layer's bf16 logits for a micro-batch do exist: about 30 GiB at TP 4 for 256k tokens. Log-probs are computed from them in 4,096-token fp32 chunks and the loss is recomputed, so there is no full-vocabulary fp32 tensor.
  - **No MTP head in training (#23).** Qwen3.8's HF config declares one. Megatron used to add its loss, undetached, to every training forward. That was a reward-independent self-imitation term reaching the policy's LoRA weights, and its fp32 full-vocabulary logits cost about 60 GiB. Measured on 64 × B300 at TP 4 with 4 samples of 258k tokens per data-parallel rank, dropping it cut a steady step from 173.7 s to 148.8 s (−14%) and peak GPU memory from about 253k to 196k MiB.
  - **What the leak did in production (run 013, 2026-09-28).** Run 013's code predated #23.
    - **The policy degraded.** Over 16 steps at LR 4e-5:
      - mean rollout log-prob of generated tokens fell from −0.66 to −1.45;
      - pass rate fell from 24–28% to 13–15%;
      - on 3 of 5 environments the model stopped submitting and its reasoning turned telegraphic.
    - **The MTP loss did it.** Replaying its step-19 batch with every reward set to 0, so the RL loss is exactly 0, still left 84% of the decoder LoRA's gradient norm. That gradient pointed along the adapter's accumulated drift (cosine 0.69); the RL gradient did not (0.006).
    - **Why it dominated:** after Adam's normalization the two pushes had the same size, but only the MTP push repeated every step. So it accumulated linearly while the RL push behaved like a random walk.
    - **Old checkpoints don't load.** Checkpoints from before #23 carry MTP-layer LoRA weights, so they fail to load on this code.
  - **Serving still uses the base model's MTP head** as the NEXTN draft. A draft never changes the sampled tokens or the returned logprobs, only decode speed, so watch the speculative accept length as the policy drifts from the base. Train MTP detached (`--enable-mtp-training --mtp-num-layers 1`) only if that length falls and serving is shown to load the trained head.
  - **Context parallelism works, but TP 4 stays.**
    - **The blocker is fixed.** Megatron-Bridge 0.7's Qwen3-VL model, which Qwen3.8 uses, needs explicit rank-local 3D MRoPE position ids for CP-sharded inputs, and Miles's Qwen3-VL patch now passes them. The Gated DeltaNet layers support CP (`linear_cp_mode`, chunkwise by default).
    - **Halve `--max-tokens-per-gpu` at CP 2.** Miles gives a micro-batch `max_tokens_per_gpu × cp_size` tokens, so this keeps one 256k sample per micro-batch.
    - **TP 1 × CP 2 × DP 32 does not fit.** On 64 × B300 it ran out of memory in its first train pass, even without the MTP head. Each GPU holds the full 27B in bf16 (~54 GB), 129k tokens of saved activations and 59.7 GiB of unsplit bf16 logits.
    - **Untested:** TP 2 × CP 2 and TP 1 × CP 4, which keep per-GPU memory near TP 4's.
- **Credentials:** replicas take capture's platform key from `miles-platform`. Capture's control credential is the gateway key, which only the trainer and the replicas hold.
- **Failure budget:** 16 consecutive failed groups.
- **Trainer replay before paying for rollouts** (`miles_plugins/proximal/e2e/trainer_replay.py`): the production trainer on mock agent-shaped rollouts (model spans trained, tool outputs masked, mixed rewards per group) through Miles's `--load-debug-rollout-data`. On 8 × B300 (2026-09-26): a realistic batch (32 samples, mean 153k tokens) trained in 17 min and a stress batch (mean 227k) in 13 min, checkpoints included. Without `cuda_allocator: expandable_segments` the first step ran out of memory on a 30 GiB logits buffer with 37 GiB reserved but unallocated.
- **Learning rate.** Run 013 raised it from 1e-5 to 4e-5 at step 4, the rate of the Tinker cookbook's RL recipes.
  - **What happened:** with the MTP loss still attached, the adapter's drift ran about 4× faster, and the policy degraded within about four steps.
  - **Without MTP, 4e-5 is untested.** Raise the LR only while watching mean rollout log-prob per step. The trainer doesn't log it today (`entropy_coef` is 0), so the drift was invisible in W&B. It held near −0.66 on a healthy policy.
- **Gradient attribution** (`miles_plugins/proximal/e2e/grad_attribution.py`) replays one recorded production step from its snapshots, at a learning rate too small to move any weight. It runs three arms:
  1. the batch as trained;
  2. the same samples with every reward 0, which isolates the reward-independent loss terms;
  3. the batch again, as the noise floor.

  Gradients come back from the saved Adam moments and are split by parameter using the names the training state now records. It reproduced run 013's step-19 gradient bit for bit on 8 × B300, in 76 min. Use it when a run drifts in a direction the reward doesn't explain.
- **Step sizing on N clustered nodes** (`miles_plugins/proximal/e2e/step_sizing.py`, #21). It runs the production trainer on fixed-length mock rollouts across N gang-scheduled Modal nodes with RDMA, and records per-step timers plus peak GPU and host memory. `--layouts` compares parallel layouts on one cluster.
  - **1024 samples of 258k tokens per step works on 64 × B300.** With the MTP head on (2026-09-27), the step processed 264M tokens in 45 min: 524 s recompute, 1948 s train, train MFU 0.48.
  - **Memory is independent of batch size.** Peak GPU memory was the same ~253k MiB as a 16-sample step. Each micro-batch is one sample, so a bigger batch only adds gradient accumulation, which costs time.
  - **Step 1 of each invocation carries 100–200 s of warm-up.** At 64 GPUs a cold kernel cache made step 1 straggle for 20+ min, so seed the whole cache.
- **Determinism is off for this trainer** (`training.json`, `deterministic_kernels: false`). FlashAttention's SM100 backward for Qwen3.8's 256-wide heads has no deterministic mode; with it forced (run 004) step 1 failed in the backward pass after a clean forward over ~147k-token samples.
- **Running out of context ends a rollout cleanly.** Capture answers a turn that cannot fit (its own budget check, or SGLang's "maximum context length" / "longer than the model's context length") with OpenAI's `context_length_exceeded` error, which agent-px turns into a budget stop and the platform grades. Before, SGLang's flat error body reached agent-px as "400 status code (no body)", the rollout failed, the platform's retry was refused by capture, and the trainer dropped the whole group (run 004: 2 such rollouts, at turns 128 and 144, cost two groups).

Capture's timing log on each replica, joined with the agent journal by response id, splits each call's time outside SGLang. Use the `overhead/` files in steps 3–6 below.

## Paid steps, in order (workspace `proximal`, environment `main`)

1. **Volumes and secrets.**
   ```bash
   modal volume create miles-qwen38-base --env main
   modal volume create miles-qwen38-adapters --env main
   modal volume create miles-qwen38-state --env main
   modal secret create miles-qwen38-gateway MILES_GATEWAY_KEY=$(openssl rand -hex 32) --env main
   # Capture's credentials, for the replicas and the trainer: the trainer's control key,
   # and the key the platform's rollout workers send (the same value as
   # MILES_CAPTURE_PLATFORM_KEY in the platform's worker secrets).
   modal secret create miles-qwen38-capture MILES_CAPTURE_ADMIN_KEY=$(openssl rand -hex 32) MILES_CAPTURE_PLATFORM_KEY=... --env main
   # The platform API key the trainer creates runs with.
   modal secret create miles-platform PROXIMAL_PLATFORM_API_KEY=... --env main
   ```
   The proxy and W&B secrets from the gsm8k run are reused.
2. **Stage the base model** (about 55 GB), using `e2e.stage_base` with this directory's `serving.json`.
3. **Write the run config** (tasks pinned by the platform's image and commit), with the pool's URL (`https://proximal--miles-qwen38-serving-replica.us-west.modal.direct`) as both `inference_url` and `capture.url`:
   ```bash
   python -m miles_plugins.proximal.training prepare --template examples/proximal/qwen38/run.template.json \
     --tasks examples/proximal/qwen38/tasks-pilot.json --inference-url <pool URL> --out run.json
   python -m miles_plugins.proximal.training check --config run.json --training examples/proximal/qwen38/training.json
   ```
4. **Deploy the replicas** with `serving_app` and a `run.json`. A deployment binds only its serving contract (`contracts.ServingContract`: base model, tokenizer, chat-template family, thinking, model protocol including reasoning effort, LoRA shape, and the sequence ceiling). Run id, tasks, harness and token budgets within the ceiling travel with each session, so later runs that fit reuse the deployment without a redeploy. Changing a contract field needs one, and a redeploy keeps live replicas on their old config: `modal app stop miles-qwen38-serving` first.
5. **Register the pool once:** make the pool's URL the default `rollout_capture` endpoint of `miles/qwen38-27b`, from proximal-mono (it writes the production registry):
   ```bash
   pnpm tsx packages/backend/scripts/modal/switch-endpoint.ts --model miles/qwen38-27b \
     --register miles-qwen38-serving-ctx<max_sequence_tokens>-out<max_tokens> --set-default miles-qwen38-serving-ctx<max_sequence_tokens>-out<max_tokens> \
     --kind rollout_capture --base-url <pool URL> --wire-model Qwen/Qwen3.8-27B --api-key-env MILES_CAPTURE_PLATFORM_KEY \
     --context-window-tokens <max_sequence_tokens> --max-output-tokens <max_tokens> --apply
   ```
   The endpoint carries the run's token budget (proximal-mono #4760): the platform sizes each solve from it, so mini-swe stops at (context − max output) × 0.9, before capture's cap. The node checks both the URL and the budget. Endpoints can't be edited in place, so the name carries both budgets.
6. **Start the training node.**
   ```bash
   PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=examples/proximal/qwen38/serving.json \
   PROXIMAL_TRAINING_CONFIG=examples/proximal/qwen38/training.json \
     modal run --detach --env main -m miles_plugins.proximal.modal_training
   ```
   The node creates platform runs only once `miles/qwen38-27b`'s default endpoint is the pool's URL; until then it prints the registration command from step 5. The registration survives node restarts.
   Before it starts the trainer, the node checks that the replicas it reaches were deployed with a serving contract this run fits (`preflight.check_serving`; it waits for replicas that are still starting). After the first policy is published and before any platform run, a canary opens a real capture session: one model call is sealed as a sample, and a turn past the sequence budget must come back as OpenAI's `context_length_exceeded`, which agent-px ends as a graded rollout (`preflight.canary`). Either failure stops the node within a few minutes of launch instead of deep into a run.
7. **Tear down:** `modal app stop miles-qwen38-training --env main` and `modal app stop miles-qwen38-serving --env main`.
