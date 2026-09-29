# GPU state verification

This bounded diagnostic checks the independent training path with real historical
GSM8K rollouts and Qwen3-0.6B. It does not launch the platform or generate fresh
rollouts. The fixed batch is deliberately replayed to isolate checkpoint recovery
from online sampling and scheduling; it is not a hill-climbing experiment.

## Experiment

1. `prepare` reads the existing model-independent complete-group codec, selects
   1,024 policy-v1 samples (128 complete groups), and commits a self-contained batch
   to a separate test Volume.
2. `reference` loads the pinned frozen base, initializes rank-32 LoRA, and executes
   two native Miles updates on that batch. Each update uses a global batch of
   1,024. It saves both native checkpoints and wraps the first in a verified
   recovery bundle.
3. `resumed` starts a fresh GPU container, reloads the Volume, and uses the production
   `offline_batch train` command to restore the first native checkpoint and perform
   the second update independently. No platform, capture, serving or database
   credentials are needed by this training command.
4. `compare` compares the second-update native adapter, FP32 optimizer state,
   scheduler, RNG, debug training data and PEFT export with the uninterrupted
   reference. Equality means exact tensor equality, with dtype and shape checks;
   mismatches and maximum absolute differences are reported, never tolerated silently.
   An independent model-specific algebraic check also compares every PEFT tensor
   with the native fused QKV/gate-up representation, without reusing Bridge's exporter.
5. `serve` packages the exports through the existing immutable snapshot preparation
   helper, loads them in a local SGLang engine, and compares token logprobs and greedy
   token IDs for the same recorded sequences. It also compares the trained adapter
   with the base and the first-step adapter to show that the exported updates are live.

The diagnostic recovery database is empty because this experiment has no online
producer or queue. Its explicit cursor identifies the frozen data. Online queue
recovery has separate CPU coverage; this experiment does not claim to reproduce an
asynchronous producer's scheduling.

## Run

Use Python **3.12**, matching the pinned GPU image. Python 3.13's `Path` pickle is
not readable by this image. The launch environment needs Modal and Pydantic, not
local CUDA/Megatron. Use the Proximal profile without changing the global default.

The existing `miles-gsm8k-base` and `miles-gsm8k-state` Volumes are mounted read-only.
Provision the separate `miles-state-verification` output Volume explicitly first;
the launcher never implicitly creates a Volume. Use a unique test ID and the
original run-006 config, preserving its training contract. The recorded source run
uses Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`.

```bash
# Execute phases sequentially, inspecting each result before continuing.
MODAL_PROFILE=proximal python -m miles_plugins.proximal.e2e.state_gpu_check prepare \
  --test-id UNIQUE_TEST_ID --config source-run.json --samples 1024 --yes-gpu-test
# Repeat with: reference, resumed, compare, serve.
```

`--resume-preparation` only resumes an incomplete copy with an identical source
config and no completed batch manifest. It does not resume a failed GPU phase.
Reference/resume/serve functions allocate at most one H100 each, have finite
timeouts and zero automatic retries, and exist only within an ephemeral app run.
The phase ends its Ray/SGLang processes; check the owned app is stopped with zero
tasks afterward. Keep the output Volume as the test evidence.

## Scope of the evidence

A successful result establishes the tested one-GPU, BF16, rank-32, fixed-batch
path. It does not establish bit-identical arbitrary asynchronous continuation,
Megatron/SGLang numerical parity, FP4/FP8 behavior, large-model memory fit, or
optimizer resharding across different GPU topologies. Serving-format conversion
is one-way; training restoration always uses the native checkpoint and pinned base.

## September 28, 2026 evidence

Output Volume: `miles-state-verification`, namespace `qwen06-state-20260928-b`.
The source/model Volumes were mounted read-only. GPU phases used one NVIDIA H100
80GB HBM3 each. Retained evidence includes the frozen group bytes, recovery bundle,
native checkpoints, debug training data, logs, and JSON comparison reports.

- [Reference GPU app](https://modal.com/apps/proximal/main/ap-h81OuzcdnQn70B9Xa6TOyJ)
- [Fresh-container offline step](https://modal.com/apps/proximal/main/ap-9Sdlbex9PttXcYBDwqRUEo)
- [CPU state comparison](https://modal.com/apps/proximal/main/ap-pQKpz3HYDJlHjfsrDGZ31s)
- [SGLang GPU serving check](https://modal.com/apps/proximal/main/ap-BcNh693kPII9aKH74r9QNi)

| Check | Result |
| --- | --- |
| Frozen batch | 1,024 real samples, 128 groups, 297,508 tokens, 202,108 loss tokens |
| Learning signal | 84 mixed-reward groups; the second update changes weights by up to `1.0073184967041016e-05` |
| Native adapter after resume | Exact equality: 224 tensors / 17,432,576 elements |
| Training state after resume | Exact equality: 677 tensors / 52,302,848 elements, including FP32 optimizer state and RNG; non-tensor scheduler/state entries also equal |
| Native debug training data | Exact equality: 5,121 tensors / 1,106,964 elements, plus metadata |
| PEFT export after resume | Exact equality: 392 tensors / 20,185,088 elements; config also equal |
| Native-to-PEFT layout mapping | All 392 PEFT tensors exactly match the independent QKV/gate-up reconstruction |
| SGLang reference vs resumed | All 3,552 input-token logprobs on 16 sequences exactly equal; greedy output IDs equal |
| Active serving updates | Base vs trained and first-step vs second-step logprobs differ; all three adapter loads returned success |

Every equality row above has maximum absolute difference **0**. The second update's
loss is `0.009227827806250248` and gradient norm `0.08930762857198715` in both GPU
processes. Reference warm update took 15.3 seconds; the fresh resumed update took
26.0 seconds. These are actor-training times, excluding container/worker startup.
The reference reported about 16.3 GiB of GPU usage after the update.

Recovery bundle:
`0000000-a9a24dfd862f77f42a6e2abd994687ba0b38437201844bc373a4dd6c7b420376`.
Reports are `preparation.json`, `reference.json`, `comparison.json` and `serving.json` under the
namespace. Native state is under `reference/checkpoints` and `resumed/checkpoints`.
The reference/resumed serving snapshot hash is identical:
`254bde3a33ede7589801b73baea51a5db6ea322589d4c9a0298d3ac62e9ce540`.
The first-step snapshot is
`919d3804d6e8b185ecfe5393360c12778a464b4a1db039c40f858b12061e1fcb`.
The algebraic format check and SGLang-to-SGLang comparison do **not** establish
Megatron-to-SGLang forward-logprob equality; their kernels differ and this test
does not compare those forward passes.

The live test found and fixed a real Modal v1 compatibility bug: final frozen-batch
manifest publication used a hardlink, which the Volume rejects. It now uses the
same verified copy/rename path as recovery artifacts. A regression test covers
unsupported hardlinks, identical retries, and rejection of conflicting manifests.
Two launcher attempts ended before training: Python 3.13/3.12 `Path` serialization,
and an optional mismatch-metrics flag that requires a custom TIS hook. Matching
Python 3.12 and removing that diagnostic preserves the original training recipe.
The first serving attempt required explicit SGLang LoRA target modules for dynamic
loading; the final harness reads them from the actual exported config. The launcher
now rejects a Python version mismatch before allocating any remote resources.

Local validation: 723 CPU tests passed, 4 skipped; strict mypy passed for 50 plugin
modules; Ruff, Black and isort passed. Comparator negative controls detect missing
optimizer state, changed RNG, FP32-only differences hidden by equal BF16 weights,
and NaNs. Test outputs are retained intentionally on the isolated Volume.
All ten apps created for preparation, failed launches and successful verification
were confirmed **stopped with zero tasks** after the final phase. Other apps were
not stopped or modified.
