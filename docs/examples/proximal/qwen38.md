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

- **Chat template and parsers:** the template family is `qwen38small`, which Miles's own tests use with the 27B. The replicas use the `qwen3_coder` tool-call parser and the `qwen3` reasoning parser. Thinking is on, with reasoning effort `high`.
- **MLP-only LoRA (rank 32).** In this Miles version, Qwen3.5 and Qwen3.8 train their attention and Gated DeltaNet projections as separate Hugging Face-style layers, while SGLang serves them fused (`in_proj_qkvz`, `in_proj_ba`, fused QKV). Adapters on those layers are not yet verified to load on the replicas. The MLP layers use the same mapping proven on Qwen3-0.6B.
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

Measures what capture adds to each model call on real platform traffic, on the topology closest to the final one. The same four "3.5 flash hard" environments (`overhead/tasks.json`) run 8 rollouts each on every step, for 10 steps:
- **Serving:** 2 replicas, each on 1 × B300 (288 GB: the 27B plus about 180 GB of KV cache). Capture runs in each, and the platform's rollout_capture client pins every rollout's calls to one replica (`Modal-Session-Id`, proximal-mono #4726). Each replica has room for 16 rollouts growing toward 256k tokens.
- **Trainer:** 8 × B300 at TP 4 with two data-parallel ranks, with sequences up to 256k tokens. A 262k-token micro-batch needs about 155 GB on one GPU, which ran out of memory on H200. Log-probs are chunked and the loss is recomputed, so the 256k × vocabulary logits never exist at once. There is no context parallelism yet: Megatron-Bridge's Qwen3-VL model, which Qwen3.8 uses, needs explicit rank-local 3D MRoPE position ids for pre-sharded CP inputs, and Miles's text-only path does not provide them.
- **Credentials:** replicas take capture's platform key from `miles-platform`. Capture's control credential is the gateway key, which only the trainer and the replicas hold.
- **Failure budget:** 16 consecutive failed groups.

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
4. **Deploy the replicas** with `serving_app` and the same `run.json`: capture in the replicas enforces it.
5. **Register the pool once:** make the pool's URL the default `rollout_capture` endpoint of `miles/qwen38-27b`, from proximal-mono (it writes the production registry):
   ```bash
   pnpm tsx packages/backend/scripts/modal/switch-endpoint.ts --model miles/qwen38-27b \
     --register miles-qwen38-serving --set-default miles-qwen38-serving --kind rollout_capture \
     --base-url <pool URL> --wire-model Qwen/Qwen3.8-27B --api-key-env MILES_CAPTURE_PLATFORM_KEY --apply
   ```
6. **Start the training node.**
   ```bash
   PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=examples/proximal/qwen38/serving.json \
   PROXIMAL_TRAINING_CONFIG=examples/proximal/qwen38/training.json \
     modal run --detach --env main -m miles_plugins.proximal.modal_training
   ```
   The node creates platform runs only once `miles/qwen38-27b`'s default endpoint is the pool's URL; until then it prints the registration command from step 5. The registration survives node restarts.
7. **Tear down:** `modal app stop miles-qwen38-training --env main` and `modal app stop miles-qwen38-serving --env main`.
