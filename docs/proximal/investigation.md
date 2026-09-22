# Async DeepSWE implementation investigation

Reviewed Miles `d3ffe916589772bb5c424c9aabd61157b954b710` and proximal-mono `f30a80d7099d68594a7153f148cd5d4d095cba45` on 2026-09-22. This extends the approved topology, using the existing Miles driver, rollout, buffer, TITO and weight-transfer boundaries.

## Decisions and their homes

| Concern | Alternatives examined | First implementation |
| --- | --- | --- |
| Async production/Q | New database scheduler; Miles `FullyAsyncRolloutFn` and `DataBuffer` | Reuse the persistent producer and bounded buffer. Specialize group preparation/acceptance, not the training algorithm. Postgres durability is deferred. |
| Policy distribution | Post-save hook; weight-update protocol | Use `WeightTransferProtocol` with the existing gathered HF adapter iterator. The post-save hook has the wrong cadence and misses initial publication. The protocol writes immutable safetensors, publishes them, then commits the serving policy. No optimizer/resume shards enter serving. |
| Fleet topology | Explicit replica list/broadcast; one logical endpoint | One Modal fleet endpoint. Each replica verifies/loads the requested snapshot at request admission. A load-balanced preparation call is only a warm-up; correctness does not depend on it visiting every replica. |
| Capture | Retokenize platform traces; reuse Miles `SessionCore` | Reuse TITO and sample assembly behind a restricted authenticated HTTP adapter. Preserve exact sampled tokens, tool masks and logprobs; reject uncaptured protocols and mismatched policy evidence. |
| Group policy | Per-sample latest-policy lookup; pin at group submission | One immutable policy per complete prompt group, retained across every turn. |
| Staleness | Check while speculative next-batch draining; check immediately before training | Fully-async production needs no extra prefetched batch. Drain after the preceding publication, so buffer filtering sees the consuming version. Semi-async scheduling retains its existing behavior. |
| Failure/reward | Turn all errors into zero; distinguish grade from infrastructure | Valid scored zero is training data. Missing grade, capture, task provenance or policy evidence makes the entire group ineligible. Bound regeneration failures to avoid endless spend. |
| Restart | Pretend database claims imply exactly-once SGD; checkpoint boundary restart | Resume native Miles checkpoint/data-source state under a new run identity after rollback. Regenerate unfinished/unconsumed work. Persist accepted capture/grade/provenance; do not claim exactly-once optimizer application. |
| Lifetime | Trainer manages sandbox teardown; platform owns physical lifetime | Trainer only cancels logical runs through platform RPC. Platform owns all container/GPU/Volume lifecycle. Replica cache eviction concerns idle adapters, never physical resources. |

## Ground truth driving changes

- `FullyAsyncRolloutFn` already has a persistent worker and bounded `DefaultDataBuffer`. `GenerateState` currently derives concurrency from GPU counts; an opaque external fleet instead needs an explicit sample concurrency limit.
- `train_async.py` drains the next batch before publishing, even with a continuously running producer. That batch was checked against the old consuming version. Fully-async mode should drain on demand while leaving its producer running.
- `WeightTransferProtocol` already separates transport from tensor conversion. Its `use_weight_update_session=False` path can publish immutable artifacts without pausing or overwriting active inference weights. Megatron's HF iterator gathers TP/EP/PP and exports adapters through Bridge.
- Existing `--rollout-external` still represents concrete SGLang engines managed by Miles. An external fleet with no enumerated engines must allocate only trainer Ray GPU slots and omit Miles router/engine workers.
- Generic agentic generation catches agent failure and can collect a partial trace. Platform generation must join grade/provenance with capture before marking a sample eligible.
- `SessionCore` supports recorded Chat Completions, TITO prefix reuse and fake SSE. Generic session proxy routes can bypass recording; the training adapter exposes only recorded inference. Collection currently neither seals nor persists, and the generic tracer deletes after failed retrieval. The training adapter serializes calls/sealing, persists the sealed result and supports retryable retrieval.
- Platform `CreateEnvironmentRun` supports deterministic run IDs, image/source pins, harness options and launch recovery. Existing duplicate checks do not certify the full training request. RPC currently lacks scoped inference injection and acknowledgement. A capabilities check is needed before launching against an older backend.
- Platform terminal `SUCCESS` and `COMPLETED` are accepted only with a scored reward and no execution error; `FAILED`, `ERROR`, `TIMEOUT` and `STOPPED` do not. Reward may exist on a failed run; reward presence alone is insufficient. Proto JSON may omit zero, while `rewardScored` certifies it.
- Modal routing/session affinity is not policy identity. Volume refresh distributes files, not GPU state. Independent replicas can serve a unique adapter selector and return verified artifact identity for every request.

## Research contract and limits

The concrete recipe targets `agentica-org/DeepSWE-Preview` (Qwen3-32B with thinking), pinned by immutable revision. Feature tasks remain platform-authored and are selected from a pinned project task manifest. Harness, budgets, group size, sampling, objective/importance-correction settings and maximum policy lag are explicit. The first supported capture path is text-only linear TITO with no compaction/subagent branching. Unsupported modes fail preflight; broader harness support is capability-driven, not Mini-SWE-specific.

Policy version counts successful publication rounds. With the initial publication and one publication per training iteration it has a documented mapping to Miles rollout iterations; it is not silently interpreted as individual optimizer microsteps. Exact token/provenance contract tests do not establish numerical train/serve equivalence. Real BF16/LoRA forward agreement, serving build/parser compatibility, Modal visibility latency and failure recovery still require the GPU smoke matrix before a meaningful learning run.

Sources: [DeepSWE model card](https://huggingface.co/agentica-org/DeepSWE-Preview), [Modal Volumes](https://modal.com/docs/guide/volumes), [Modal Servers](https://modal.com/docs/guide/servers). Code references above are from the pinned revisions, not the user's dirty platform checkout.
