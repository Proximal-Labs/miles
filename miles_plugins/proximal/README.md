# Proximal async RL integration

Miles trains a LoRA while Proximal continuously executes feature tasks against independently served immutable policy versions on Modal. Miles uses one fleet URL; replica count, placement and sandbox/container teardown belong to the platform.

Read the [architecture](../../docs/proximal/architecture.md), [investigation](../../docs/proximal/investigation.md), and exact [remaining platform changes](../../docs/proximal/platform-contract.md). The first pass supports DeepSWE/Qwen3, one Megatron actor cell, text-only linear TITO, complete prompt groups, and rollout-logprob importance ratios. It includes code and CPU tests; live numerical/Modal validation is still required.

## 1. Prepare the environment and explicit run contract

Use Miles's existing Megatron/bridge training environment on your trainer cluster. For the CPU service, use the same Miles checkout with its compatible SGLang Python package, Torch CPU, Transformers, FastAPI, httpx and safetensors. SGLang's schemas currently import runtime dependencies; a bare Python install with only FastAPI is insufficient. Modal publication uses `modal==1.5.5` and existing authenticated Modal credentials.

Copy [run.example.json](../../examples/proximal/run.example.json). Replace project/task/image/harness pins and URLs. Deliberately invalid placeholders prevent accidental execution. The DeepSWE example is pinned to HF revision `4887205c533cd162baac7ba758159cfc3304cf94`; stage that exact base checkpoint/tokenizer on training and serving machines, and the same tokenizer files on capture. Configure Qwen3 thinking/tool parsing consistently. Do not point the base checkpoint at a changing `main` revision.

The config's sampling, group size, staleness and behavior-logprob convention are authoritative. The launcher translates them into existing Miles flags, and validation rejects conflicting overrides. Objective, clipping, optimizer, LoRA rank/targets, learning rate, checkpoint paths and training duration remain ordinary explicit Miles arguments. The example numbers are an initial experiment choice, not an established DeepSWE recipe.

Provide secrets through the named environment variables on processes that need them. `MILES_GATEWAY_AUTHORIZATION` contains `Bearer <gateway-key>`; the gateway's `MILES_GATEWAY_KEY` contains the raw key. Modal proxy headers are optional: remove their entries if your platform endpoint does not use them. Capture's administrator credential never goes to a sandbox; each rollout receives its own scoped credential.

Use a persistent `/artifacts` directory for the capture service. The trainer and CPU rollout executor also need a persistent artifact destination for publication/accepted samples, preferably the same shared mount. Mount the run JSON at the same absolute path on the driver and Ray workers, and provide the required environment variables to those workers through your existing Miles launch environment. The resolved run configuration is immutable within the capture artifact namespace; changing it requires a new run identity.

Free validation and argument inspection:

```bash
python -m miles_plugins.proximal.runtime validate --config /config/run.json
python -m miles_plugins.proximal.runtime train-args --config /config/run.json
```

## 2. Attach the serving gateway to your existing Modal fleet

Mount the **same existing Modal Volume** on each inference replica, for example at `/adapters`. Start SGLang with the pinned base, LoRA enabled, matching targets and sufficient maximum rank/adapter capacity. Give the gateway sole control of its `miles-*` adapters. Keep SGLang loopback-only and restart it together with the gateway.

Inside each platform-owned replica, expose the gateway port through the existing Modal endpoint:

```bash
python -m miles_plugins.proximal.serve_replica \
  --config /config/replica.json \
  --volume-name miles-adapters --environment-name main \
  --volume-mount /adapters --local-cache /tmp/miles-adapter-cache \
  --engine-api-key-env SGLANG_API_KEY \
  --host 0.0.0.0 --port 8092 --yes-load
```

[replica.example.json](../../examples/proximal/replica.example.json) declares the expected base path and adapter capacity. The gateway checks the reported engine path at startup. It serves `/policies/prepare` and `/v1/chat/completions`; authentication is mandatory. It verifies the shared Volume artifact before registering each version and keeps active requests pinned during LRU eviction. Registration ambiguity fails closed; recycle the replica with the platform's normal lifecycle.

This command runs a gateway in an existing container. It does not deploy an app, rent a GPU, create a Volume, or enumerate replicas. Use the resulting single fleet URL in the run config. Keep at least one replica available for the initial policy acknowledgement if your Modal routing requires it.

## 3. Start the CPU capture service

```bash
python -m miles_plugins.proximal.runtime capture \
  --config /config/run.json --host 0.0.0.0 --port 8091 \
  --yes-rollouts --yes-publish
```

Expose it behind HTTPS at `capture.url`, reachable by the platform's solver and Miles. Use **one process/worker** per run; active sessions are process-owned. The service stores sealed safetensors and receipts atomically and supports collection retries after restart. A lost unsealed session must be regenerated. Do not put a non-sticky autoscaling fleet in front of this CPU service.

## 4. First live gate: a frozen-policy rollout without a training GPU

Start with a compatible, already-existing PEFT adapter. The original snapshot commands remain available:

```bash
python -m miles_plugins.proximal prepare \
  --adapter-directory /exports/adapter --config /config/export.json --checkpoint-iteration 0
python -m miles_plugins.proximal publish-modal \
  --snapshot-directory /exports/snapshots/<sha256> --sha256 <sha256> \
  --volume-name miles-adapters --environment-name main --yes-publish
```

`export.json` contains `run_id`, `base_model` (same name/revision), and `output_root`. Use the actual snapshot directory reported by the prepare layout (`<output_root>/snapshots/<sha256>`). Create `policy.json`:

```json
{"run_id":"deepswe-features-001","version":1,"snapshot":{"sha256":"<snapshot digest>"},"base_model":{"name":"agentica-org/DeepSWE-Preview","revision":"4887205c533cd162baac7ba758159cfc3304cf94"}}
```

Then explicitly authorize the live policy registration and one platform task:

```bash
python -m miles_plugins.proximal.runtime commit-policy \
  --config /config/run.json --policy-file /config/policy.json --yes-rollouts --yes-publish
python -m miles_plugins.proximal.runtime rollout \
  --config /config/run.json --task-index 0 --yes-rollouts --yes-publish
```

The second command prints the reward and token counts and persists `accepted/<attempt-id>/{accepted.json,samples.safetensors}`. It runs no optimizer. It **does** use paid serving/sandbox infrastructure, so run it only when those operations are approved. Use a separate smoke run ID before starting training; startup publication must not collide with an unrelated manually registered adapter.

## 5. Start continuous async training

Run the launcher in the existing Miles trainer environment attached to your Ray cluster (`RAY_ADDRESS` as appropriate). It calls the normal `train_async.train`; there is no second optimizer loop.

```bash
python -m miles_plugins.proximal.runtime train \
  --config /config/run.json --yes-rollouts --yes-publish -- \
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 \
  --lora-rank 64 --lora-alpha 128 --lora-dropout 0 \
  --target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.01 \
  --eps-clip 0.2 --eps-clip-high 0.2 \
  --rollout-batch-size 4 --global-batch-size 32 --num-rollout 100 \
  --lr 1e-5 --seed 42 \
  --load /checkpoints/initial-native --save /checkpoints/feature-hillclimb --save-interval 5
```

Supply the normal model/parallelism/optimizer arguments required by your Miles Megatron launch environment as well; the fragment above shows the integration arguments and illustrative research choices, not a universal GPU launch configuration. Convert/load the pinned DeepSWE base using Miles's standard checkpoint workflow. If using KL, provide the matching reference checkpoint/configuration required by that workflow. Inspect the resolved arguments before running. No trainer or inference GPU is provisioned by this launcher.

At startup and after each iteration, the existing weight updater exports/publishes a fresh immutable adapter. The producer keeps generating during training/publication. All members of a group share a policy. Q discards over-stale groups when the trainer drains its next batch. A valid zero score remains trainable. Consecutive execution failures trip the configured circuit breaker.

For resume, restore the latest matching native checkpoint and task cursor. Sealed data remains inspectable, but queue consumption is not durably acknowledged against optimizer updates. In-flight/queued work is regenerated after a process restart. Rolling back behind published history needs a new run ID. Artifact retention is explicit operator maintenance; this integration never deletes shared policy history.

## CPU verification

The dedicated [CPU workflow](../../.github/workflows/proximal-publication.yml) builds the Linux environment in `tests/integration/proximal_async/Dockerfile`, fetches only the pinned Qwen3 tokenizer, and runs tests with networking disabled. Its exact test selection currently passes **475 tests**, including 61 publication/integration tests and the affected upstream argument, async-driver, session/codec and weight-update regressions. Strict mypy covers all 19 adapter modules; Ruff, Black, isort and workflow syntax checks also pass locally.

Remote boundaries use scripted HTTP/Modal fixtures. TITO (including a tool-call/result turn with verified zero loss on tool output), safetensors, the async worker, argument parser, CPU tensors and Gloo weight-updater lifecycle are real. This does not exercise Megatron GPU tensor gathering or numerical adapter equivalence. The dependency-heavy integration tests live outside the generic fast suite and have their own required asset setup; a missing tokenizer fails their dedicated job.

## Validation progression

1. CPU tests: exact TITO/codec, request/grade provenance, failure rejection, staleness/backpressure, adapter concurrency, actual Miles argument and weight-update seams.
2. One frozen policy and one feature task: inspect the saved Sample and platform trace; no training GPU required.
3. Two real replicas and two policies: publish v2 while a v1 task is running, replace a replica, verify every assistant token remains on its selected policy; measure Volume propagation and adapter latency.
4. GPU numerical gate: compare trainer and SGLang logprobs on identical IDs/weights, then one actual LoRA optimizer update and publication. Only then scale task concurrency and assess learning.

No live Modal deployment, sandbox execution, GPU inference, or optimizer update was performed to prepare this PR.
