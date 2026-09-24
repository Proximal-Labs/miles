---
title: "Qwen3.8-27B on real platform rollouts"
description: "Qwen3.8-27B trained on real Proximal platform rollouts (DeepSWE tasks) from a Modal training node and replicas."
# Generated from examples/proximal/qwen38/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
The first run against the real Proximal platform. Miles trains `Qwen/Qwen3.8-27B` on DeepSWE feature tasks. Each rollout is a real mini-swe session in a platform sandbox: agent-px calls capture through an HTTPS tunnel, and the platform grades the rollout.

| Part | Here |
| --- | --- |
| Training node | Modal 1× 8 H200 (`modal_training` with `training.json`, platform `real`): Megatron LoRA trainer at TP 4, capture, rollout store |
| Serving | Modal 2× H200 (`serving_app`), one replica per GPU |
| Platform | Production: runs created with the run config's platform key; rollout workers call capture through the node's tunnel |
| Tasks | `tasks-pilot.json`: 41 tasks from project 519 / DeepSWE. It combines "3.8 good passrate" (13), "3.5 flash medium" (8) and the first 20 of "3.5 flash hard". `tasks-hard.json` holds all 348 "3.5 flash hard" tasks: gpt-5.5 solves more than 75% of rollouts on the newest image, and gemini-3.5 solved 1–2 of 8. |

## Settings and why

- **Chat template and parsers:** the template family is `qwen38small`, which Miles's own tests use with the 27B. The replicas use the `qwen3_coder` tool-call parser and the `qwen3` reasoning parser. Thinking is on, with reasoning effort `high`.
- **MLP-only LoRA (rank 32).** In this Miles version, Qwen3.5 and Qwen3.8 train their attention and Gated DeltaNet projections as separate Hugging Face-style layers, while SGLang serves them fused (`in_proj_qkvz`, `in_proj_ba`, fused QKV). Adapters on those layers are not yet verified to load on the replicas. The MLP layers use the same mapping proven on Qwen3-0.6B.
- **Batch:** 8 tasks × 4 samples = 32 rollouts per step, one optimizer update per step. The Miles recipe uses 1 node × 8 GPUs at TP 4.
- **Limits:** mini-swe with `max_turns` 30, 8k tokens per turn, 32k per sequence, and 64 samples in flight. At that concurrency, per-attempt polling is cheap.
- **Routing:** `platform_route.endpoint_name` is unset, so the platform routes the model's calls to its *default* registry endpoint. Each start of the node registers a new endpoint as the default.

## Checked without paid resources

- The run config and training deployment validate, including the chat-template parser check (`training check`).
- Miles's Qwen3.8-27B template and exact-token tests pass (61 tests).
- In the Miles image, Miles's parser accepts the full training command, and SGLang's `ServerArgs` accepts the replica's LoRA engine arguments.

Checked only on GPUs:
- Megatron loading this vision-language model through the bridge;
- exporting the MLP LoRA and the replicas loading it;
- memory at 32k tokens.

## Paid steps, in order (workspace `proximal`, environment `main`)

1. **Volumes and secrets.**
   ```bash
   modal volume create miles-qwen38-base --env main
   modal volume create miles-qwen38-adapters --env main
   modal volume create miles-qwen38-state --env main
   modal secret create miles-qwen38-gateway MILES_GATEWAY_KEY=$(openssl rand -hex 32) --env main
   # The platform API key, and the capture key the platform's rollout workers send
   # (the same value as MILES_CAPTURE_PLATFORM_KEY in the platform's worker secrets).
   modal secret create miles-platform PROXIMAL_PLATFORM_API_KEY=... MILES_CAPTURE_PLATFORM_KEY=... --env main
   ```
   The proxy and W&B secrets from the gsm8k run are reused.
2. **Stage the base model** (about 55 GB), using `e2e.stage_base` with this directory's `serving.json`.
3. **Deploy the replicas** with `serving_app`, and note the pool URL.
4. **Write the run config** (tasks pinned by the platform's image and commit):
   ```bash
   python -m miles_plugins.proximal.training prepare --template examples/proximal/qwen38/run.template.json \
     --tasks examples/proximal/qwen38/tasks-pilot.json --inference-url <pool URL> --out run.json
   python -m miles_plugins.proximal.training check --config run.json --training examples/proximal/qwen38/training.json
   ```
5. **Real-SGLang check:** run gsm8k through the pool with this model (a copy of the gsm8k run template with this model, template family and parsers) before creating platform runs.
6. **Start the training node.**
   ```bash
   PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=examples/proximal/qwen38/serving.json \
   PROXIMAL_TRAINING_CONFIG=examples/proximal/qwen38/training.json \
     modal run --detach --env main -m miles_plugins.proximal.modal_training
   ```
   The node opens its capture tunnel and prints a `switch-endpoint.ts` command. Run it from proximal-mono (it writes the production registry, `--apply`). The node polls the registry and creates platform runs only once `miles/qwen38-27b`'s default endpoint is its tunnel. A restart after a crash opens a new tunnel and prints a new command.
7. **Tear down:** `modal app stop miles-qwen38-training --env main` and `modal app stop miles-qwen38-serving --env main`.
