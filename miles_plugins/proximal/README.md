# Proximal async RL integration

Miles trains a LoRA while Proximal continuously executes feature tasks against independently served immutable policy versions on Modal. Miles uses one fleet URL; the serving deployment config controls replica hardware and bounds, and the platform owns rollout sandbox teardown.

For sampling without a training cluster and a separate optimizer step later, start
with the [Modal collect-then-train guide](../../docs/proximal/modal-collect-then-train.md). It explains replica
ownership, policy selection, automatic Volume persistence, batch validation, and
what the later training launcher must provide. The
[offline-batch reference](../../docs/proximal/offline-batches.md) covers detailed
selection and recovery commands. For the training allocation itself, see
[Modal training clusters](../../docs/proximal/modal-training-clusters.md).

Read the [architecture](../../docs/proximal/architecture.md), [investigation](../../docs/proximal/investigation.md), and exact [remaining platform changes](../../docs/proximal/platform-contract.md). The first pass supports DeepSWE/Qwen3, one Megatron actor cell, text-only linear TITO, complete prompt groups, and rollout-logprob importance ratios. It includes code and CPU tests; live numerical/Modal validation is still required.

## 1. Prepare the environment and explicit run contract

Use Miles's existing Megatron/bridge training environment on your trainer cluster. For the CPU service, use the same Miles checkout with its compatible SGLang Python package, Torch CPU, Transformers, FastAPI, httpx and safetensors. SGLang's schemas currently import runtime dependencies; a bare Python install with only FastAPI is insufficient. Modal publication uses `modal==1.5.5` and existing authenticated Modal credentials.

Copy [run.example.json](../../examples/proximal/run.example.json). Replace project/task/image/harness pins and URLs. Deliberately invalid placeholders prevent accidental execution. The DeepSWE example is pinned to HF revision `4887205c533cd162baac7ba758159cfc3304cf94`; stage that exact base checkpoint/tokenizer on training and serving machines, and the same tokenizer files on capture. Configure Qwen3 thinking/tool parsing consistently. Do not point the base checkpoint at a changing `main` revision.

The config's sampling, group size, staleness and behavior-logprob convention are authoritative. The launcher translates them into existing Miles flags, and validation rejects conflicting overrides. LoRA rank, alpha and target modules come from `research.lora` and feed both the trainer and the serving pool. Objective, clipping, optimizer, learning rate, checkpoint paths and training duration remain ordinary explicit Miles arguments. The example numbers are an initial experiment choice, not an established DeepSWE recipe.

Provide secrets through the named environment variables on processes that need them. `MILES_GATEWAY_AUTHORIZATION` contains `Bearer <gateway-key>`; the gateway's `MILES_GATEWAY_KEY` contains the raw key. Modal proxy headers are optional: remove their entries if your platform endpoint does not use them. Capture's administrator credential never goes to a sandbox; each rollout receives its own scoped credential.

Choose `artifact_storage` explicitly: `modal_volume` (the Volume mounted at `artifact_directory` in every container that reads or writes the store) or `shared_disk` (only when all of them share one filesystem). Provide a Postgres database for the rollout store: set the environment variable named by `store_dsn_env` to its DSN on the CPU rollout process, the capture service, and the trainer's rank zero. Tables are created on first connection. Use a persistent `/artifacts` directory for the capture service. The trainer and CPU rollout executor also need a persistent artifact destination for publication/accepted samples, preferably the same shared mount. Mount the run JSON at the same absolute path on the driver and Ray workers, and provide the required environment variables to those workers through your existing Miles launch environment. The resolved run configuration is immutable within the capture artifact namespace; changing it requires a new run identity.

Free validation and argument inspection:

```bash
python -m miles_plugins.proximal.runtime validate --config /config/run.json
python -m miles_plugins.proximal.runtime train-args --config /config/run.json
```

## 2. Deploy the Miles-owned serving pool

Miles owns the inference replicas so serving cannot drift from training. The pool runs the **same Miles image** as the trainer. Its SGLang arguments are derived from the run config's `research.lora` (rank and target modules), `model_protocol` (reasoning and tool-call parsers), base model and tokenizer. They are rendered, parsed by SGLang, and checked setting by setting at deploy time, so a bad flag fails the deploy rather than a GPU replica. Extra engine flags may change only an explicit allowlist of operational settings (memory, scheduling, CUDA graphs, logging); anything that could change the served model or its numerics, such as `--load-format dummy`, is rejected. Deploy from the Miles environment, where SGLang is importable. Operational shape (GPU type, tensor parallelism, replica bounds, adapter slots, the attention kernel, speculative decoding, performance-only flags) lives in a separate serving config. Attention and speculation are typed fields every config states (null leaves them to SGLang); speculation uses SGLang's default verification, which a sampling check showed keeps the served model's distribution (its rejection-sampling path did not). They are not extras: an extra that changes either is rejected. Example: [serving.example.json](../../examples/proximal/serving.example.json).

Prerequisites, created once outside this integration:

- A Modal Volume holding the pinned base weights at `<base_mount>/<basename of tokenizer_path>`.
- The run's adapter Volume (`volume` in the run config).
- A Modal secret named `gateway_secret` that sets `gateway_key_env`.

Deploy:

```bash
PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=serving.json \
  modal deploy --env main -m miles_plugins.proximal.serving_app
```

Each replica starts SGLang on loopback, then its front process (`serve_replica`): the gateway and capture, on one port. If either exits, the replica exits and Modal replaces it. The deployed `*.modal.direct` URL is the run config's `inference_url` and `capture.url`, and it is what the platform endpoint registry records as a `rollout_capture` endpoint. Every replica mounts the same adapter Volume and independently loads and verifies the immutable version each request names. The gateway serves `/policies/prepare` and `/v1/chat/completions`; capture serves `/sessions` and `/rollouts/<rollout>/v1/chat/completions` and calls the gateway in-process. Authentication is mandatory on every route.

Capture keeps each rollout's session in the replica that serves it, so every call about a rollout carries `Modal-Session-Id: sha256(<platform rollout id>)` (`contracts.AFFINITY_HEADER`): the trainer's session calls and the platform's chat calls. Modal routes equal session IDs to one container. Because sessions live in the replicas, the pool runs exactly one spare (`max_replicas` is `min_replicas + 1`, enforced by `ServingDeployment`): Modal replaces a replica on a host it drains by starting the new container before retiring the old one, which needs room above `min_replicas`. A second spare would only let load grow the pool, and a replica started for load drops its sessions when it scales back down, so `target_concurrency` should be high enough that normal load fits in `min_replicas`. The deployment's `capture_secret` provides capture's credentials, `cpu` sizes the replica for SGLang plus the front process (which runs at lower scheduling priority than SGLang), and `modal_proxy_auth` is false when the platform's agents call the pool directly.

## 3. Capture

On the Modal topology capture runs in the serving replicas (above); nothing else to start.

The local Stage A harness runs it as its own process next to the trainer instead, calling the pool over the network and checking the rollout store for each session's policy:

```bash
python -m miles_plugins.proximal.runtime capture \
  --config /config/run.json --host 0.0.0.0 --port 8091 \
  --yes-rollouts --yes-publish
```

Either way, active sessions are process-owned: a call must reach the process that holds its session, so capture never sits behind a non-sticky balancer. It stores sealed safetensors and receipts atomically and supports collection retries after a restart. A lost unsealed session must be regenerated.

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

Started this way, outside the Modal training launcher, the run config must set `"weight_sync_transport": "volume"`. The default, `"http"`, needs the launcher's run-state writer to archive each policy, and without it the trainer stops at startup with an error naming this setting (see [Full-snapshot HTTP weight sync](#full-snapshot-http-weight-sync)).

```bash
python -m miles_plugins.proximal.runtime train \
  --config /config/run.json --yes-rollouts --yes-publish -- \
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.01 \
  --eps-clip 0.2 --eps-clip-high 0.2 \
  --rollout-batch-size 4 --global-batch-size 32 --num-rollout 100 \
  --lr 1e-5 --seed 42 \
  --load /checkpoints/initial-native --save /checkpoints/feature-hillclimb --save-interval 5
```

Supply the normal model/parallelism/optimizer arguments required by your Miles Megatron launch environment as well; the fragment above shows the integration arguments and illustrative research choices, not a universal GPU launch configuration. Convert/load the pinned DeepSWE base using Miles's standard checkpoint workflow. If using KL, provide the matching reference checkpoint/configuration required by that workflow. Inspect the resolved arguments before running. No trainer or inference GPU is provisioned by this launcher.

At startup and after each iteration, the existing weight updater exports/publishes a fresh immutable adapter. The producer keeps generating during training/publication. All members of a group share a policy. Q discards over-stale groups when the trainer drains its next batch. A valid zero score remains trainable. With `unused_groups: retry`, a rollout that fails to execute is relaunched in its group at once (same task, policy and sample, a new attempt), up to `group_size` times per group; a group that still fails is retried whole. Consecutive failed groups trip the configured circuit breaker.

To fill training batches with groups whose verifier rewards have nonzero variance, append the existing Miles option to the training arguments (after `--`):

```bash
--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter
```

This drops whole all-zero/all-one groups from training, while retaining scored-zero samples in mixed-reward groups. The default without the flag retains every valid complete group. Completed rollout payloads are still stored; Postgres records the filter path and rejection reason, and rejected groups take no training-buffer capacity. The async producer continues generating replacement groups until the batch is full. Drops appear in `rollout/platform/dynamic_filtered_groups` and Miles's existing `rollout/dynamic_filter/drop_<reason>` metrics. A filter rejection does not count as an execution failure or immediately retry the same prompt.

Custom hooks use Miles's synchronous `(args, samples)` interface, returning `FilterOutput` or a legacy boolean. They must be deterministic and must not mutate samples; keep their implementation and arguments fixed for a run. Use a new run ID when those change. Filtering also runs before consuming previously stored groups, so pre-existing groups and interrupted writes cannot bypass it. All-rejected data leaves the trainer waiting for an eligible batch; it never silently trains on rejected data. The training-node recipe keeps the Postgres index/rejections and separate payload files locally and snapshots both to its existing Modal state Volume with completed checkpoints.

For resume, restore the latest matching native checkpoint; the task source restores its cursor and consumption ledger from the same checkpoint. Completed groups persist in the rollout store and are selectable after restart if still fresh; only in-flight work is regenerated. The first publication after resume abandons versions newer than the checkpoint, and their groups are never trained on. Artifact retention is explicit operator maintenance; this integration never deletes shared policy history.

For **collection now and one independent training step later**, see the
[offline batch runbook](../../docs/proximal/offline-batches.md). A finite CPU collector
uses the existing platform producer, while a frozen batch enters the ordinary Miles
training driver through `FrozenBatchRolloutFn`. Preserve a compatible native training
checkpoint alongside the data; serving PEFT exports alone cannot initialize this
fork's Megatron resume path.

## CPU verification

The dedicated [CPU workflow](../../.github/workflows/proximal-publication.yml) builds the Linux environment in `tests/integration/proximal_async/Dockerfile`, fetches only the pinned Qwen3 tokenizer, and runs tests with networking disabled. Its exact test selection currently passes **475 tests**, including 61 publication/integration tests and the affected upstream argument, async-driver, session/codec and weight-update regressions. Strict mypy covers all 19 adapter modules; Ruff, Black, isort and workflow syntax checks also pass locally.

Remote boundaries use scripted HTTP/Modal fixtures. TITO (including a tool-call/result turn with verified zero loss on tool output), safetensors, the async worker, argument parser, CPU tensors and Gloo weight-updater lifecycle are real. This does not exercise Megatron GPU tensor gathering or numerical adapter equivalence. The dependency-heavy integration tests live outside the generic fast suite and have their own required asset setup; a missing tokenizer fails their dedicated job.

## Validation progression

1. CPU tests: exact TITO/codec, request/grade provenance, failure rejection, staleness/backpressure, adapter concurrency, actual Miles argument and weight-update seams.
2. One frozen policy and one feature task: inspect the saved Sample and platform trace; no training GPU required.
3. Two real replicas and two policies: publish v2 while a v1 task is running, replace a replica, verify every assistant token remains on its selected policy; measure Volume propagation and adapter latency.
4. GPU numerical gate: compare trainer and SGLang logprobs on identical IDs/weights, then one actual LoRA optimizer update and publication. Only then scale task concurrency and assess learning.

No live Modal deployment, sandbox execution, GPU inference, or optimizer update was performed to prepare this PR.


## Modal run-state recovery

`TrainingDeployment.state_volume` names an **already provisioned** state Volume.
The Modal launcher mounts it at `/snapshot` and gives each `RunConfig.run_id` its
own root. Keep this Volume separate from the serving adapter Volume. The launcher
selects the `run_state` artifact-storage variant in its generated container config;
standalone/shared-disk execution retains its existing behavior.

```text
<run_id>/
  run.json                         # immutable run identity / initial context
  launches/<launch_id>/config.json # configured image, source digest, exact args/config
  artifacts/<run_id>/
    accepted/<attempt_id>/samples.safetensors
    accepted/<attempt_id>/accepted.json
    groups/<group_id>.bin
    groups/<group_id>.json
  checkpoints/<step>-<digest>/
    checkpoint/adapter/            # native rank shards, optimizer/scheduler/RNG
    cursor.json
    launch.json                    # checksummed launch configuration
    store.dump
    manifest.json
  pins/<checkpoint-id>             # optional operator-created empty file
  LATEST                          # convenience pointer to a verified manifest
```

The nested artifact run ID preserves the existing artifact layout. No S3 copy is
involved, and publication staging/serving history is not copied into new state
snapshots. Live Postgres remains local; only `pg_dump` output goes on the Volume.

A validated capture stages locally and waits for two publication commits: bytes
first, typed completion record last. Only then is the replica session released.
A complete group's immutable index similarly precedes its database row. One
writer batches up to 64 records / 256 MiB (one larger record is streamed alone).
The existing in-flight sample/token limits bound waiting rollout work. A Volume
outage backpressures completion and retries publication, without rerunning paid
inference. This adds latency at rollout completion, not at each model turn.
Measure commit latency under the actual load before scaling the number of runs.

The writer snapshots completed native optimizer boundaries and drains explicitly
on shutdown. Restore verifies each file before loading it, restores the cursor's
consumption ledger, reconciles group indexes newer than the database dump, and
copies only eligible group payloads back to local disk. Reconciliation never makes
an abandoned policy live; the existing publisher owns that decision. Even before
the first checkpoint, durable groups can be recovered, but training restarts from
the initial weights. In-flight or partially assembled groups may be regenerated.

Configure resume in the training deployment, for example:

```json
"resume": {
  "kind": "latest",
  "allow_changes": []
}
```

`latest` fails if there is no committed checkpoint. `fresh` requires an unused run
namespace. `checkpoint` requires its exact `<step>-<digest>` in `checkpoint` and
supports rewinding to a retained/pinned bundle. `auto` preserves the old launcher
behavior for existing configs: latest if present, otherwise an initial run.
It is the backward-compatible default; use an explicit mode for experiments.

Base/tokenizer/training-contract, world size, staleness window and model-argument
mismatches are rejected. The staleness window stays fixed because the saved
consumption ledger has already pruned older entries. Changed training flags require explicit `allow_changes`, e.g. `["--lr"]`;
changed fork source or configured base image require `code_sha256` / `image` there.
This is acknowledgement, not an override of Miles's optimizer/scheduler loading
semantics: changing the resumed learning-rate schedule may also require Miles's
scheduler override flags. Parallel-layout and optimizer changes remain unsupported.
No optimizer/RNG save/load opt-outs are accepted on this full-recovery launcher.
The manifest records configured image identity; it is not an attestation of a
running container's installed libraries or numerical behavior.

Legacy integer `LATEST` / `steps/<step>` snapshots still load, with a warning that
they lack the new manifest and RNG evidence. The first subsequent save uses the
new format. Old snapshots and artifact history are not migrated or deleted.
New recovery retention keeps the last two published bundles plus explicit pins,
including across a rewind. It never deletes a Volume or serving policy history.
There is deliberately no implicit capture/group expiration deadline.

**One active trainer per run is required.** The function has one container per
app, but that does not fence a second independently launched app with the same
run ID. Stop the previous app before a manual restart. Cross-app duplicate-owner
fencing/automatic takeover is not implemented; a Volume lock file would not solve
it. Different runs use separate namespaces and can share a Volume.

Start with v1 for durable run state. Modal recommends keeping v1 below 50,000
files/directories, with a hard 500,000 limit and no more than about five concurrent
writers. Budget growth within each run: an accepted attempt adds a directory and
two files, and a group adds two files. Monitor around 30,000 inodes and place new
runs on another Volume before 40,000, allowing for active-run growth. These two
thresholds are our operating recommendation, not provider limits. v2 has different
limits and is currently Beta; do not switch the recovery authority without testing
and deciding on a second durable copy. See [Modal Volumes](https://modal.com/docs/guide/volumes).

CPU tests cover commit failures, missing/corrupt shards, immutable rewinds, pins,
post-checkpoint reconciliation, separate run namespaces, capture release ordering,
and final flush. A CPU Adam replay also checks the next parameters, moments and
scheduler after restart. Before production use, run the separate live v1 Volume
visibility/latency test and fixed-batch Megatron GPU resume comparison. Async order
and nondeterministic kernels mean this is not a bit-identical whole-run promise.
Platform catalog registration and artifact download access remain a separate PR.


## Full-snapshot HTTP weight sync

HTTP is the default weight-sync transport. Update serving replicas before running
the Modal training launcher. Set `"weight_sync_transport": "volume"` to use the
previous full Volume path. The `ModalVolumeTransfer` import path stays compatible.

The default applies to every run config that does not name a transport, and only the
Modal training launcher can run it. A trainer started any other way (`runtime train`
on your own cluster, as in section 5) must set `"weight_sync_transport": "volume"`,
as `examples/proximal/run.example.json` does; otherwise it refuses to start.

Rank zero exports the adapter and scatters disjoint byte ranges to training
ranks, which upload concurrently without shared storage. The receiver checks
part sizes/checksums, reconstructs the snapshot and loads it into SGLang.
Requests use affinity and a receiver ID; misrouting or a failed rank rejects the
update. Retry the whole upload after failure.

After HTTP readiness, the existing `StateWriter` archives the snapshot through
the plain Volume publisher. Only then does the policy become fleet-selectable,
so replacement replicas can recover it. One pending archive is allowed; the
next sync waits with a timeout. Existing writer retries and shutdown drain apply.
The first sync returns before its policy is selectable, so the rollout producer
waits for that first policy instead of failing at startup.
HTTP requires `run_state` artifact storage, supplied by the Modal launcher.

Native checkpoints and restore are unchanged: local capture remains synchronous,
and Volume persistence stays in the existing background writer. Production GPU
validation is pending. Centralized export, inter-rank scatter and archive latency
still need end-to-end measurement; transport timings alone do not show a speedup.
