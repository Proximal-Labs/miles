---
title: "Stage A: end-to-end from a Mac, training faked"
description: "Stage A harness for Proximal platform rollouts: the real rollout, store and capture path from a Mac against Modal replicas, with training faked."
# Generated from examples/proximal/e2e/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Stage A runs every part of the async platform-RL path for real except the optimizer:

| Part | Stage A |
| --- | --- |
| Rollout store (Postgres + payloads), batch query, consumption ledger, policy registry | Real |
| Miles producer, `PlatformRolloutFn`, `PlatformDataBuffer`, `PlatformTaskSource` | Real |
| Capture service (Miles TITO, exact tokens/logprobs/masks) | Real |
| Snapshot preparation, adapter Volume upload, serving-pool verification | Real (offline: local) |
| Serving pool: SGLang + gateway loading immutable LoRA versions on Modal | Real (offline: `fake_pool`) |
| Platform | `stub_platform`: plays agent-px's mini-swe traffic (pinned proximal-mono commit) and acts as the endpoint registry, deterministic mixed grades |
| Training | `fake_trainer`: consumes real batches, publishes pre-made random-weight adapters as new versions |

Everything local runs in one Linux container (the CPU test image), because Miles and SGLang's Python code need Linux and the run config allows plain HTTP only on loopback. Configs here use Qwen3-0.6B at the revision CI pins, LoRA rank 8, groups of 4, `max_policy_lag` 1.

## 0. Local prerequisites (free)

From the repository root:

```bash
docker build -t proximal-cpu -f tests/integration/proximal_async/Dockerfile .
git clone https://github.com/sgl-project/sglang .cpu-sglang && git -C .cpu-sglang checkout 94602c9c2b7cbdb8efd5c52802dac6a1c180089e
REV=c1899de289a04d12100db370d81485cdf75e47ca
mkdir -p .stage-a/Qwen3-0.6B-${REV}
docker run --rm -v "$PWD/.stage-a/Qwen3-0.6B-${REV}:/out" proximal-cpu python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='${REV}', allow_patterns=['tokenizer.json','tokenizer_config.json','config.json','vocab.json','merges.txt'], local_dir='/out')"
```

Use `${REV}` with braces in zsh: `$REV:ro` is parsed as a variable modifier.

A zsh helper for the steps below:

```bash
# Extra docker flags go in STAGE_A_DOCKER, e.g. STAGE_A_DOCKER="--network none".
stage_a() {
  docker run --rm ${=STAGE_A_DOCKER} -v "$PWD:/work:ro" -v "$PWD/.cpu-sglang:/sglang:ro" \
    -v "$PWD/.stage-a:/stage-a" -v "$PWD/.stage-a/Qwen3-0.6B-${REV}:/models/Qwen3-0.6B-${REV}:ro" \
    -e STAGE_A_PLATFORM_KEY -e STAGE_A_CAPTURE_KEY -e STAGE_A_CAPTURE_PLATFORM_KEY -e STAGE_A_GATEWAY_AUTHORIZATION \
    -e MODAL_TOKEN_ID -e MODAL_TOKEN_SECRET -e MODAL_PROXY_KEY -e MODAL_PROXY_SECRET \
    -w /work proximal-cpu "$@"
}
export STAGE_A_PLATFORM_KEY=$(openssl rand -hex 16) STAGE_A_CAPTURE_KEY=$(openssl rand -hex 16) \
  STAGE_A_CAPTURE_PLATFORM_KEY=$(openssl rand -hex 16)
```

Generate three random-weight adapters (distinct seeds, so versions serve distinguishable outputs):

```bash
stage_a sh -c "python -m miles_plugins.proximal.e2e.adapters --config examples/proximal/e2e/run.stage-a.json \
  --base-config /models/Qwen3-0.6B-${REV}/config.json --output /stage-a/adapters --count 3"
```

## 1. Offline rehearsal (free, no network)

Uses `run.stage-a.local.json`: the fake serving pool on loopback, local publication, a persistent local Postgres under the workdir.

```bash
export STAGE_A_GATEWAY_AUTHORIZATION="Bearer $(openssl rand -hex 16)"
STAGE_A_DOCKER="--network none" stage_a sh -c "python -m miles_plugins.proximal.e2e.launch --config examples/proximal/e2e/run.stage-a.local.json --platform stub \
  --adapters /stage-a/adapters --workdir /stage-a/offline --publish local --steps 4 --yes-rollouts --yes-publish"
# Resume from the step-1 checkpoint against the same store:
STAGE_A_DOCKER="--network none" stage_a sh -c "python -m miles_plugins.proximal.e2e.launch --config examples/proximal/e2e/run.stage-a.local.json --platform stub \
  --adapters /stage-a/adapters --workdir /stage-a/offline --publish local --steps 4 --resume-step 1 --yes-rollouts --yes-publish"
```

Expect four steps over policy versions 1 and 2, then a resume at step 2 that never retrains a group consumed by steps 0-1. Reports: `.stage-a/offline/report*.json`; process logs: `.stage-a/offline/logs/`.

## 2. Against the Modal serving pool (PAID, workspace `proximal`, environment `main`)

Every step here creates or uses paid Modal resources. Run them only with explicit approval, in order, and tear down in step 2g. Set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` for the **proximal** workspace (a `MODAL_PROFILE` alone may point at another workspace).

**2a. Create the Volumes (one time).**

```bash
modal volume create miles-stage-a-base --env main
modal volume create miles-stage-a-adapters --env main
```

**2b. Create the gateway credential and Modal proxy auth.** The gateway requires `Authorization: Bearer <key>`; the Modal proxy requires `Modal-Key`/`Modal-Secret` (create a proxy auth token in the workspace settings).

```bash
GATEWAY_KEY=$(openssl rand -hex 32)
modal secret create miles-stage-a-gateway MILES_GATEWAY_KEY=$GATEWAY_KEY --env main
export STAGE_A_GATEWAY_AUTHORIZATION="Bearer $GATEWAY_KEY" MODAL_PROXY_KEY=wk-... MODAL_PROXY_SECRET=ws-...
```

**2c. Stage the base weights (small CPU job).**

```bash
stage_a sh -c "PROXIMAL_RUN_CONFIG=examples/proximal/e2e/run.stage-a.json PROXIMAL_SERVING_CONFIG=examples/proximal/e2e/serving.stage-a.json \
  modal run --env main -m miles_plugins.proximal.e2e.stage_base"
```

**2d. Deploy the serving pool: two L4 replicas kept warm (`min_replicas` 2), billed while up.** Engine arguments are validated at deploy time. Deploy from the container so SGLang is importable.

```bash
stage_a sh -c "PROXIMAL_RUN_CONFIG=examples/proximal/e2e/run.stage-a.json PROXIMAL_SERVING_CONFIG=examples/proximal/e2e/serving.stage-a.json \
  modal deploy --env main -m miles_plugins.proximal.serving_app"
```

If the printed URL differs from `inference_url` in `run.stage-a.json`, update the config (it is part of neither the training contract nor the store identity).

**2e. Run Stage A against the pool.** Publication uploads each version to `miles-stage-a-adapters`; replicas load and verify it on demand.

```bash
stage_a sh -c "python -m miles_plugins.proximal.e2e.launch --config examples/proximal/e2e/run.stage-a.json --platform stub \
  --adapters /stage-a/adapters --workdir /stage-a/modal --publish modal --steps 6 --yes-rollouts --yes-publish"
```

**2f. What to check.**

- `report.json`: every group's `policy_version` within lag of the trainer version; `loss_tokens` > 0; `model_calls` per sample from the real model (it may or may not call the bash tool); mixed rewards.
- Both replicas served: `modal app logs miles-stage-a-serving --env main` shows adapter loads on two containers.
- Two versions at once: while step 2e runs, versions N and N+1 are both served; old groups still carry version N.
- Replica replacement: during a run, `modal container list --env main` then `modal container stop <id>` on one replica. Rollouts on it fail as execution errors (never zero-reward samples), the replacement loads versions on demand, and training continues.
- Resume: rerun 2e with `--resume-step 1` and the same workdir.

**2g. Tear down (stops billing).**

```bash
modal app stop miles-stage-a-serving --env main
```

Volumes and the secret cost little at rest; delete them only when Stage A is finished (`modal volume delete ...`, `modal secret delete ...`).

## 3. Stage B: the real platform (after the proximal-mono change)

Replace the stub with the platform once its endpoint registry supports the Chat Completions per-run route in [platform-contract.md](/proximal/platform-contract). The launcher skips the stub whenever the run config's `platform.url` is not loopback.

1. **Choose where agent-px runs.** Its rollout workers must reach the capture service. The capture URL in the run config must be loopback (for the local launcher) or HTTPS.
   - Local platform worker on this Mac: the capture service must share a network with it (for example `--network host` for the Stage A container, or run the capture service outside Docker), and the registry `baseURL` points at it.
   - Staging: the capture service needs a public HTTPS URL, e.g. deployed as a Modal `app.server` next to the serving pool, or a tunnel.
2. **Register the endpoint** (platform operator): a `rollout_capture` entry under the platform model ID in `platform_route.model` (e.g. `miles/stage-a`) named `platform_route.endpoint_name`, with `baseURL` = the capture URL, wire model `Qwen/Qwen3-0.6B`, and `apiKeyEnv` naming a worker variable that holds `STAGE_A_CAPTURE_PLATFORM_KEY`'s value: `switch-endpoint.ts --model miles/stage-a --register miles-capture --set-default miles-capture --kind rollout_capture --base-url <capture URL> --wire-model Qwen/Qwen3-0.6B --api-key-env <NAME> --apply`.
3. **Point the run config at real tasks and the platform**: `platform.url`, a real project and pinned `dataset.tasks`, the mini-swe `harness.agent_type` and revision, and a small `harness.max_turns` (e.g. 3) to keep sandbox time low.
4. **Run** with `--publish modal` against the deployed serving pool (step 2 above), with a small `--steps`. Each rollout's model calls go agent-px → capture → serving pool; grades come back through the existing run APIs.
