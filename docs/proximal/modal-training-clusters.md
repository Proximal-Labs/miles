# Modal training clusters: allocation, warm recovery and shutdown

Use this guide to allocate the training side of the
[collect-then-train workflow](modal-collect-then-train.md). It documents the
existing launchers, including the eight-node, 64-B300 frozen-batch sweep. Inference
replicas and platform sandboxes have separate lifetimes; a stored batch can be
trained with both already stopped.

Run commands from the root of this Miles fork. Launch commands below allocate paid
compute. Prepare and validate the inputs first, and agree on the training and
failure-hold budgets before submission.

## Choose the existing launcher

| Need | Entry point | Allocation and result |
| --- | --- | --- |
| Online async RL | [`modal_training`](../../miles_plugins/proximal/modal_training.py) | One GPU container sized by `training.json`; owns the online trainer and recovery snapshots. `SIZING_NODES` does not make this launcher multi-node. |
| Multi-node capacity/timing check | [`e2e.step_sizing`](../../miles_plugins/proximal/e2e/step_sizing.py) | `SIZING_NODES` eight-GPU nodes; mock rollouts, no retained trained checkpoints. Qwen and Inkling profiles exist. |
| Train an existing base-policy batch and retain adapters | [`e2e.batch_sweep`](../../miles_plugins/proximal/e2e/batch_sweep.py) | Qwen profile, `SIZING_NODES` nodes, **B300:8 per node**; two updates per configuration, local staging, native checkpoints and evaluation exports. |
| One update per step over several base-policy batches, per LoRA configuration | [`e2e.batch_chain`](../../miles_plugins/proximal/e2e/batch_chain.py) | Same allocation as the sweep; each step trains a different batch at policy lag = step index, continuing the previous step's native state, with an optional operator gate between steps. See [Chain single updates over different batches](#chain-single-updates-over-different-batches). |
| Exactly one independent update | [`offline_batch train`](offline-batches.md) | Runs inside an already allocated Miles/Ray environment. It does not allocate Modal nodes or commit its output. |

The instructions below use `batch_sweep`, the launcher used for the real stored
1,024-rollout experiment. Its `updates: 2` contract is deliberate. A phase can
instead resume its certified first update and execute only the second. There is
currently no general CLI that allocates an arbitrary cluster and hands it to
`offline_batch train` for one fresh update. A held sweep also has no command queue
for submitting arbitrary new training work. For several single updates over different
base-policy batches, use `e2e.batch_chain` (last section).

## Understand the allocation

`SIZING_NODES=8` plus `gpu="B300:8"` means **8 containers × 8 GPUs = 64 GPUs**.
Each node requests 32 CPUs and 1 TiB of host RAM. The sweep decorator fixes these
resources, an eight-hour function timeout and `retries=0`; changing
`training.json.memory_mib`, `gpu` or `max_retries` does not override that decorator.
The training config still supplies the model, image-related environment, input
argument file, state Volume and kernel-cache configuration.

Modal acquires the colocated nodes as a gang before executing the function on
each node. Its private network carries Ray coordination; `rdma=True` requests the
fabric used by GPU collectives. Allocation may wait for suitable capacity; an app
ID or a successful image build is not evidence that 64 GPUs are running.
See [Modal's cluster guide](https://modal.com/docs/guide/multi-node-clusters).

The launcher then starts Ray on the cluster's private IPv4 addresses: node 0 is
the head on port 6379, and the other nodes join it. It waits for the exact node IP
set, each advertising eight GPUs, before starting Miles. You do not create a
separate Ray head, expose Ray publicly, or manually set `RAY_ADDRESS` on your laptop.
The implemented startup is in [`_start_ray`](../../miles_plugins/proximal/e2e/step_sizing.py).

GPU allocation and model parallelism are separate choices. For the Qwen run,
TP=4, PP=1 and CP=1 across 64 GPUs gave DP=16. Preserve a native checkpoint's
parallel layout when resuming; changing the node count is not automatic optimizer
resharding. Provisioning more nodes also does not fix an oversized per-GPU
microbatch. Use sizing experiments when changing context length or parallelism.

## Prepare the launch host and inputs

Use a persistent launch host/session (for example, a supervised process or `tmux`),
Python **3.12**, and the compatible Miles dependencies described in the
[plugin setup](../../miles_plugins/proximal/README.md#1-prepare-the-environment-and-explicit-run-contract). The sweep explicitly
rejects other Python versions because its pinned image serializes `Path` objects
using 3.12. The integration documents `modal==1.5.5`; a bare Modal-only virtualenv
is insufficient for the local batch validation and SGLang imports.

The code currently uses `modal.experimental.clustered` and
`modal.experimental.get_cluster_info`. Current Modal docs show newer stable API
spellings; keep the SDK and this checkout compatible rather than rewriting the
launcher as part of an experiment.

Authenticate to the intended workspace, select the matching Modal profile, and
keep all Volume environments consistent with `--env main` below. The base-weight,
adapter and state Volumes must already exist. No inference deployment needs to be
started just to run this sweep.

Prepare these files:

| Input | Meaning |
| --- | --- |
| `run.json` | **Exact source config embedded in the frozen batch.** Do not change its run ID or LoRA targets to name the new experiment. |
| `serving.json` | Supplies the pinned image and base Volume/mount, and must remain compatible with the source. Importing it does not deploy serving replicas. |
| `training.json` | Supplies `model_args`, a repository-relative `train_args` file, the state Volume, and kernel settings. |
| Local verified `batch/` | `batch.json`, all referenced group payloads and `base_policy/`. This is the independent readback used for local validation. |
| Same batch on the state Volume | Byte-identical input at the plan's `batch_path`, relative to `/snapshot`. `--local-bundle` does **not** upload it. |
| `plan.json` | New experiment ID, batch hash/count/path, node count, phases, resolved Miles recipe and explicit recovery/hold budgets. |

Keep `training.json.num_gpus` at **8** with `gpu: "B300:8"`, and its referenced
argument file at `--actor-num-nodes 1 --actor-num-gpus-per-node 8`. Import-time
validation checks that single-node template. The sweep overrides the effective
actor node count using `plan.nodes`; both that field and `SIZING_NODES` must be 8
for a 64-GPU run. Putting 64 in `num_gpus` fails before submission.

For B300/Qwen3.8, the successful training configuration used
`deterministic_kernels: false` and `cuda_allocator: "expandable_segments"`.
The example `examples/proximal/qwen38/training.json` is an H200 template and must
be reviewed; its current values are not the successful B300 settings. A populated,
compatible kernel-cache Volume avoids much repeated compilation: the sizing
image mounts it read-only and copies it to node-local disk. See the
[Qwen execution notes](../../examples/proximal/qwen38/README.md) for the measured
SM100 backward, memory and kernel-cache constraints.

Record the checkout commit, local diff, image digest and all input files with the
experiment. The launcher packages the local checkout; it does not fetch GitHub
`main` at runtime. Do not edit that checkout while submitting the app.

## Build and review the sweep plan

The plan schema is [`SweepPlan`](../../miles_plugins/proximal/e2e/batch_sweep_inputs.py).
Its `recipe` is a JSON array of resolved Miles argument tokens, including model
arguments, parallelism, loss, optimizer, LR and seed. It is not a shell command.
Unlike the online launcher, the sweep does not automatically append
`training.model_args` to `plan.recipe`.

To prepare `recipe.json` from your reviewed training argument file and this
checkout's model definition, run this locally after setting the config path:

```bash
export PROXIMAL_TRAINING_CONFIG=/absolute/path/to/training.json
python - <<'PY'
import json
import os
import shlex
from pathlib import Path
from miles.utils.external_utils.model_args_utils import load_model_args

deployment = json.loads(Path(os.environ["PROXIMAL_TRAINING_CONFIG"]).read_text())
lines = Path(deployment["train_args"]).read_text().splitlines()
recipe = [token for line in lines if line.strip() and not line.lstrip().startswith("#")
          for token in shlex.split(line)]
recipe += shlex.split(load_model_args(deployment["model_args"],
                                     model_script_dir=Path("scripts/models").resolve()))
Path("recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
PY
```

Review the resulting array against the intended experiment. In particular, a
256k-context run needs the intended sequence and microbatch token limits; the
template's smaller token budget is not a historical production recipe. No LR or
optimizer choice is implied by allocating 64 GPUs.

The sweep runs `plan.recipe` as written, so a per-step optimization is active only if
its flag is in the array. In particular, `--skip-actor-forward-only` drops the separate
old-policy log-prob pass, which is redundant when each update is one optimizer step with
no KL and no dropout ([#37](https://github.com/Proximal-Labs/miles/pull/37) measured
20–23% of step time on 8 B300s). A recipe built from an argument file without the flag
pays for that pass on every update.

The following JSON is a **plan skeleton**, not a runnable recipe. Replace the
hash/path/recipe and choose the target list deliberately. This example uses the
full Qwen text-decoder targets; it excludes the MTP head.

```json
{
  "experiment_id": "new-full-lora-experiment",
  "batch_path": "SOURCE_RUN/assemblies/ASSEMBLY/batch",
  "batch_sha256": "REPLACE_WITH_SHA256_OF_EXACT_BATCH_JSON_BYTES",
  "samples": 1024,
  "nodes": 8,
  "phases": [{
    "name": "full",
    "updates": 2,
    "target_modules": [
      "language_model.decoder.layers.*.self_attention.linear_qkv",
      "language_model.decoder.layers.*.self_attention.linear_proj",
      "language_model.decoder.layers.*.self_attention.in_proj",
      "language_model.decoder.layers.*.self_attention.out_proj",
      "language_model.decoder.layers.*.mlp.linear_fc1",
      "language_model.decoder.layers.*.mlp.linear_fc2"
    ],
    "resume": null
  }],
  "recipe": ["REPLACE_WITH_CONTENTS_OF_RECIPE_JSON"],
  "phase_attempts": 2,
  "failure_hold_seconds": 21600
}
```

After saving the skeleton as `plan.json`, fill its hash and recipe from the
reviewed local inputs (these operations are local only):

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from miles_plugins.proximal.e2e.batch_sweep_inputs import SweepPlan

plan = json.loads(Path("plan.json").read_text())
plan["batch_sha256"] = hashlib.sha256(
    Path("/absolute/path/to/batch/batch.json").read_bytes()).hexdigest()
plan["recipe"] = json.loads(Path("recipe.json").read_text())
validated = SweepPlan.model_validate_json(json.dumps(plan))
Path("plan.json").write_text(validated.model_dump_json(indent=2) + "\n")
PY
```

`samples` is the number of trajectories, so 1,024 at group size 8 means 128 groups.
The imported Qwen sizing profile currently requires group size 8. The sweep also
requires a verified zero-delta base-policy proof and source `max_policy_lag >= 1`
to allow its second update on the same data. If the collected source doesn't meet
these conditions, do not rewrite its provenance to pass validation.

The phase builder explicitly supplies batch sizing, LoRA rank/alpha from the
source, phase targets, source behavior correction, save-every-update and offline
rollout flags. It removes W&B/evaluation flags and rejects unsupported modes such
as online async, MTP training and dynamic sampling. Review that translation in
[`phase_command`](../../miles_plugins/proximal/e2e/batch_sweep_inputs.py) when
reusing a recipe from another launch path.

## Validate, then submit once

Replace the paths below; `training.train_args` is relative to the checkout, while
these exported config paths may be absolute. Use the same selected profile for
launch, monitoring and shutdown.

```bash
export MODAL_PROFILE=proximal
export PROXIMAL_RUN_CONFIG=/absolute/path/to/run.json
export PROXIMAL_SERVING_CONFIG=/absolute/path/to/serving.json
export PROXIMAL_TRAINING_CONFIG=/absolute/path/to/training.json
export SIZING_PROFILE=qwen38
export SIZING_NODES=8

python -m miles_plugins.proximal.runtime validate --config "$PROXIMAL_RUN_CONFIG"
python -m miles_plugins.proximal.training check \
  --config "$PROXIMAL_RUN_CONFIG" --training "$PROXIMAL_TRAINING_CONFIG"
python -m miles_plugins.proximal.offline_batch check --bundle /absolute/path/to/batch
modal app list --env main --json > apps-before.json
```

Resolve any already-active `miles-qwen38-batch-sweep` invocation before submitting
another. The local batch must match the committed Volume batch and plan hash.
The launch command first validates the plan and local batch, then runs a **paid
CPU-only preflight** in the pinned image against the actual Volume input. That
checks source identity, resume receipts, native argument parsing and HF model
configuration before requesting GPUs. Hardware-dependent Megatron validation
still runs on the allocated gang before model loading.

```bash
modal run --detach --env main -m miles_plugins.proximal.e2e.batch_sweep \
  --plan /absolute/path/to/plan.json \
  --local-bundle /absolute/path/to/batch \
  --yes-train --yes-publish \
  --out /absolute/path/to/sweep-result.json \
  > /absolute/path/to/sweep-launch.log 2>&1
```

Keep this process alive in the persistent session. `--detach` protects the remote
app from client disconnection, but this entrypoint owns a
[`modal.Dict.ephemeral()` context](https://modal.com/docs/reference/modal.Dict#ephemeral)
for coordination. It also writes the local result only after the remote call
returns. Detachment does not make that coordinator durable. See
[Modal app lifetimes](https://modal.com/docs/guide/apps#ephemeral-apps).

The exact invocation's launch log contains its `ap-...` app ID. Save it and
corroborate it against `apps-before.json` and the current inventory. App names are
reused; use the owned **ID** for monitoring and eventual shutdown. A second
invocation can request another gang. Waiting longer does not justify resubmission.

## Follow startup and training

In a second terminal with the same profile:

```bash
modal app list --env main --json
modal app logs ap-REPLACE_WITH_OWNED_APP_ID --env main
```

Distinguish these stages:

1. **Build / CPU preflight / capacity wait:** no training step has started.
2. **Gang entered:** the owner commits `plan.json` and `cluster.json` under
   `RUN_ID/experiments/EXPERIMENT_ID/` on the state Volume. `cluster.json` records
   cluster ID, node IPs and coordination Dict ID.
3. **Input/fabric/Ray preparation:** every node hash-verifies a local batch copy,
   checks RDMA, seeds its kernel cache and joins the exact Ray node set.
4. **Model loading and compilation:** GPUs being allocated or low-utilization
   does not mean a completed optimizer step. Read rank logs for failures/stalls.
5. **Training:** Miles emits `perf/log_probs_time` and `perf/actor_train_time`.
   Recompute and gradient passes can each take many minutes on long contexts.
6. **Durable update:** `[batch-sweep] durable PHASE update N: .../step-N.json`
   appears only after all node shards and the evaluation export are committed
   and verified. The returned report and Volume `completed.json` certify the
   requested update set.

Use both optimizer-step IDs and durable receipts to count progress. A generic
`perf` row, an adapter file appearing, or quiet GPUs is insufficient. For the
current attempt layout, artifacts are under
`RUN_ID/experiments/EXPERIMENT_ID/attempt-N/PHASE/`: `training.log`, `timings.json`,
`checkpoints/iter_0000000/adapter/` (first update), `iter_0000001/adapter/`
(second update), `eval/snapshots/`, and `step-1.json` / `step-2.json`.

The evaluation export is saved on the **state Volume**; `--yes-publish` authorizes
that artifact publication. This sweep does not update the online policy ledger
or load a serving replica. Use the receipt's snapshot identity for later evaluation.

## Keep the allocation through recoverable failures

The run owns the Modal nodes across all phases. It stages batch bytes and resume
shards on local disk so asynchronous checkpoint publication can safely reload the
state Volume. The original failed sweep read training inputs from that reloadable
mount; the current launcher removes that race.

For an incomplete training attempt, the launcher can stop/restart Ray workers on
the **same nodes**, retain local kernel caches, and resume a committed first update.
`phase_attempts` allows 1–3 total attempts per phase. Independent fresh phases can
still run after another exhausts its attempts. Committed updates are not blindly
reapplied.

After an unrecovered exception, surviving functions hold their allocation for
`failure_hold_seconds` (0–21,600), capped at 60 seconds before their eight-hour
function deadline. A six-hour hold is a potentially substantial **paid** budget,
not six extra hours beyond the function lifetime. The hold permits inspection;
it does not provide an interactive retry command or indefinite reservation.

To end a failure hold early, read that experiment's committed `cluster.json` and,
after verifying ownership, set the coordination Dict key
`<cluster_id>/release` to `True` using `modal.Dict.from_id(coordination_dict)`.
This releases waiting functions; it does not resume training. If the coordinator
is gone or inaccessible, the bounded hold remains the fallback, or explicitly
stop the owned app. Do not have a supervisor automatically stop the app merely
because a local CLI or log connection failed.

Provider failures are different. Modal documents that preemption replaces the
entire gang, and GPU functions cannot request `nonpreemptible=True`.
`retries=0` limits configured exception retries; it is not a guarantee against
provider replay. The sweep's committed experiment directory prevents a replay
from silently restarting the training history. Treat any replacement as an
incident and reconcile durable receipts before another launch.
See [cluster fault tolerance](https://modal.com/docs/guide/multi-node-clusters#fault-tolerance)
and [preemption](https://modal.com/docs/guide/preemption).

Do not apply a “64 minimum / 65 maximum” serving-pool rule to this training gang.
These are 8 eight-GPU containers, not 64 independently replaceable workers. Current
Modal cluster scaling limits count containers in multiples of cluster size, and
this launcher sets no warm-pool minimum. See the
[`clustered` API](https://modal.com/docs/sdk/py/latest/clustered).

If the gang has ended, continuation needs a new experiment ID and a hash-pinned
`resume` reference to the prior phase's `step-1.json` (`path`, `size_bytes`,
`sha256`, relative to the state Volume). Keep the same phase name, targets, recipe,
sample count, batch hash and native parallel layout. CPU preflight verifies every
native/optimizer shard. A PEFT export alone cannot resume training. See
[`read_resume`](../../miles_plugins/proximal/e2e/batch_sweep_recovery.py).

## Verify output, then stop the owned app

On success, check the requested phases/updates against `sweep-result.json` and
independently read back `completed.json`, per-step receipts, native shards and
evaluation snapshots from the state Volume. Preserve input/config hashes, launch
and training logs, and the app/cluster identities with the experiment. An abnormal
exit may still have a valid committed first update; inspect before rerunning it.

Once all requested updates are durable, the function returns and stops its Ray
processes. The launcher has no “keep after success” switch. Verify cleanup of this
exact app ID; explicitly stop it if still active and you intend to release it:

```bash
modal app stop ap-REPLACE_WITH_OWNED_APP_ID --env main --yes
modal app list --env main --json > apps-after.json
```

Confirm that ID reports `state: "stopped"` and `tasks: 0`; retain the inventory
as the cleanup record. A disappeared local process or successful stop command
alone is not verification. Stop only the app you own. This operation preserves
the named Volumes and their retained rollouts/checkpoints; it does not stop
separately deployed serving apps.

This guide documents the existing
[architecture and recovery boundaries](architecture.md). It adds no generic
cluster service, automatic gang relaunch, or unbounded allocation retention.

## Chain single updates over different batches

[`e2e.batch_chain`](../../miles_plugins/proximal/e2e/batch_chain.py) takes the same
allocation, staging, publication and failure hold as the sweep, but each arm (a LoRA
target set) applies **one update per step on a different batch**: step k trains
`batches[k]` at rollout ID k, continuing step k-1's native weights, optimizer,
scheduler and RNG. Every batch must come from the same base policy, so step k trains
on data k versions old; the source's behavior correction (TIS) is the only off-policy
correction, and the source's `max_policy_lag` bounds the number of steps. Arms see the
same batches in the same order and start fresh from that base policy.

The plan schema is
[`ChainPlan`](../../miles_plugins/proximal/e2e/batch_chain_inputs.py):

```json
{
  "experiment_id": "p519-chain-20261002",
  "samples": 1024,
  "nodes": 4,
  "batches": [
    {"path": "RUN/batches/a", "sha256": "<sha256 of a/batch.json>"},
    {"path": "RUN/batches/b", "sha256": "<sha256 of b/batch.json>"},
    {"path": "RUN/batches/c", "sha256": "<sha256 of c/batch.json>"}
  ],
  "arms": [
    {"name": "mlp", "target_modules": ["language_model.decoder.layers.*.mlp.linear_fc1", "language_model.decoder.layers.*.mlp.linear_fc2"]},
    {"name": "full", "target_modules": ["language_model.decoder.layers.*.self_attention.linear_qkv", "...", "language_model.decoder.layers.*.mlp.linear_fc2"]}
  ],
  "recipe": ["<the collection's saved training-args.json array>"],
  "gate": "manual"
}
```

Validation refuses, before any GPU: a batch whose manifest hash differs from the plan,
a source or sample count other than the configured run's, a behavior policy that
differs between batches or lacks its zero-delta proof, two batches sharing a stored
group (identical batches are a replay, not a step), and a step beyond
`max_policy_lag`. Assembled batches must name the configured run as their anchor
source; build every batch with the same anchor collection first.

With `"gate": "manual"`, create the empty control Dict before launch (the launcher
only references named objects), then approve each later step after reviewing its
predecessor. Steps are 1-based; `stop` ends the chain at its next gate (a running step
finishes and commits) without holding the cluster, and no answer within
`gate_timeout_seconds` (default one hour) also stops it:

```bash
modal dict create batch-chain-p519-chain-20261002 --env main
python - <<'PY'
import modal
control = modal.Dict.from_name("batch-chain-p519-chain-20261002", environment_name="main")
print(control.get("awaiting"), control.get("last"))  # Next step and the last step's summary.
control["go/mlp/2"] = True  # Or control["stop"] = True.
PY
```

`last` holds the step's exit code, receipt, policy lag, resume proof and its final
`train/*` row (loss, grad norm, TIS mean/clip fraction, train/rollout KL). The launch
uses the sweep's environment and holding rules:

```bash
modal run --detach --env main -m miles_plugins.proximal.e2e.batch_chain \
  --plan /absolute/path/to/chain.json --yes-train --yes-publish \
  --out /absolute/path/to/chain-result.json > /absolute/path/to/chain-launch.log 2>&1
```

Each step commits under `experiments/<id>/<arm>/step-<n>/attempt-<a>/`: native and
optimizer shards for every rank, an evaluation snapshot, `training.log`,
`timings.json` and, last, `receipt.json` (batch hash, behavior policy, policy lag,
the predecessor receipt and every file's digest). A continued step stages exactly
its predecessor's verified files; Miles falls back to a fresh adapter when it cannot
load one, so the chain also requires the trainer's log to show the restored optimizer
and the expected iteration, and holds the cluster otherwise. A step whose native
save committed is never applied again, whatever its exit code. All arms and steps
must fit in the function's eight-hour limit, including gate waits; run arms as
separate experiments on separate allocations to parallelize them.

