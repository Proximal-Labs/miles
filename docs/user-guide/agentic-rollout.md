---
title: Agentic Rollout (TITO)
description: Configure an OpenAI-compatible agent loop with Token-In-Token-Out trajectory assembly.
---

Multi-turn agentic rollout in Miles runs through the Token-In-Token-Out (TITO)
session server. Your agent exchanges OpenAI-compatible chat messages, while Miles
preserves the exact token IDs, logprobs, and routed experts produced during
inference and assembles them into training samples. For the design rationale, see
[No Token Left Behind](https://lmsys.org/blog/2026-05-13-no-token-left-behind/).

This page owns the agentic path: wrapper setup, the custom agent contract,
session behavior, token ownership, model-family selection, and verification.
Use [Generate Endpoint](/user-guide/generate-endpoint) for the lower-level,
stateless `/generate` interface.

<Warning>

**No VLM support yet.** Currently the TITO session path cannot carry image or video inputs. For vision-language models, use the
[Generate Endpoint](/user-guide/generate-endpoint) path instead.

</Warning>

## Configure the wrapper

Select `agentic_tool_call.generate` as the custom generate function. The wrapper
registers `--custom-agent-function-path` and `--max-seq-len`, creates a TITO
session for each rollout, invokes your agent, and collects the resulting samples.

```bash
AGENTIC_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path    my_agent.run
   --use-session-server
   --hf-checkpoint                 Qwen/Qwen3-4B
   --tito-model                    qwen3
)
```

<Warning>

**Do not apply the chat template to prompt data manually.** Do not pass
`--apply-chat-template`: `Sample.prompt` must remain a `messages` list. The
session server renders the first turn and incrementally appends later turns with
the selected `--tito-model` implementation.

</Warning>

## Write the agent loop

Use `--custom-agent-function-path` to name an async function with this contract:

```python
async def run_agent(
    base_url: str,
    prompt,
    request_kwargs: dict,
    metadata: dict,
    **kwargs,
) -> dict | None:
    ...
```

Send OpenAI-compatible chat requests to the session-scoped endpoint:

```python
from miles.utils.http_utils import post


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    payload = {"model": "default", "messages": prompt, **request_kwargs}
    await post(f"{base_url}/v1/chat/completions", payload)
    return None
```

- `base_url` already includes `/sessions/<id>`; do not append the session path.
- `prompt` is the input sample's OpenAI `messages` list.
- `request_kwargs` contains the rollout sampling settings in
  `ChatCompletionRequest`-compatible form. For example, Miles maps
  `max_new_tokens` to `max_tokens`.
- `metadata` contains the sample metadata, session identifiers, and configured
  `max_seq_len`. Forward only the fields your environment needs.
- Return a dictionary to merge rewards, reports, or metrics into each output
  sample's metadata, or return `None` when there is nothing to add.

For structured parsing, the payload may use SGLang's
`ChatCompletionRequest`-compatible fields, which extend the OpenAI format.


### Optional teardown hook

The module named by `--custom-agent-function-path` may expose an `abort` function
alongside the agent entry point:

```python
async def abort(args) -> None:
    ...  # cancel this agent's in-flight external work
```

Miles calls this hook during oversampling abort after it stops in-flight SGLang
generation. Use it when the agent drives an external sandbox or agent server that
would otherwise keep issuing completion requests until its own length limit or
timeout. The hook is optional; modules without it continue to work.

See [`swe_agent_function.abort`](https://github.com/radixark/miles/blob/main/examples/swe-agent-harbor-docker/swe_agent_function.py)
for an implementation that flushes the Harbor agent server.

## TITO

### Leave token ownership to Miles

Send the full `messages` history on every turn. On the first request, the
session server renders the selected template into `input_ids`. After a
successful completion, it checkpoints those prompt IDs together with the output
token IDs and logprobs returned by SGLang.

On later requests, the server reuses the deepest applicable checkpoint,
tokenizes only the appended suffix, and sends the joined `input_ids` to SGLang.
During collection, Miles aligns the turn outputs against the accumulated TITO
sequence, trims model-specific boundary tokens, and builds the training sample.

<Warning>

**Do not set TITO control fields.** The session server replaces client
`input_ids` and forces `logprobs=True`, `return_meta_info=True`, and the response
metadata needed for TITO. Do not set `logprob_start_len=0`; scoring the entire
prompt defeats prefix caching and hurts performance.

</Warning>

### Choose the session behavior

History handling depends on the selected server version:

- **v1 is linear.** Each request must extend the previous messages at the tail.
  Retrying the latest turn may roll back one assistant checkpoint, including to
  an empty session when retrying the first turn. Earlier divergence or a larger
  rollback is rejected.
- **v2 (Experimental) is an append-only tree.** A request attaches to the deepest checkpoint
  whose complete message path prefixes the request. Any unmatched suffix creates
  a branch, and existing branches are never deleted. A path whose last generation
  ended with `finish_reason=length` cannot be extended.

Whether a replayed message counts as "the same" as the stored one is decided by
`--session-message-matcher` (default `strict`); see
[Choose replay matching](#choose-replay-matching).

The v1 wrapper returns one `Sample`. The v2 wrapper returns a `list[Sample]`, one
for each selected tree leaf. Both versions reject `--partial-rollout`. With R3
replay, every pause mode except `retract` requests only additional R3 rows.
`retract` returns full R3 data on every turn and emits a warning because the
payloads can become very large.

Set `--max-seq-len` to cap the context length. Miles also includes this value in the
metadata passed to your agent so an external environment can stop early.

### Pick your `--tito-model`

There is no auto-detection. Pick the family matching your model. Each named
family resolves a maintainer-verified `FIXED_TEMPLATE` registration from
`--tito-model` alone. The registration owns the bundled Jinja or
HuggingFace-native template, fixed template arguments, and the bundled SGLang
reasoning and tool-call parsers.

A named family rejects `--chat-template-path` overrides and conflicting fixed
arguments. Use `--tito-model default` for a custom or checkpoint-native renderer,
but treat it as best-effort until it passes the checks below.

| Your model | `--tito-model` |
|---|---|
| Qwen3 | `qwen3` |
| Qwen3.5 | `qwen35` |
| Qwen3.6 | `qwen36` |
| Qwen3.8-27B | `qwen38small` |
| Qwen3.8-Flash-Next | `qwen4exp` |
| Qwen3-Thinking-2507 / Qwen3-Next | `qwennext` |
| GLM-4.7 / 5 / 5.2 | `glm47` |
| GLM-5.3 / GLM-5.3-Flash (text sessions) | `glm53` |
| NVIDIA Nemotron 3 Nano / Super / Ultra | `nemotron3` |
| Kimi K2.5 / K2.6 | `kimi25` / `kimi26` |
| MiniMax M2.5 / M2.7 | `minimax_m25` / `minimax_m27` |
| DeepSeek-V3.2 / V4 | `deepseekv32` / `deepseekv4` |
| Inkling / Inkling-Small | `inkling` |
| Unregistered model or custom template (best-effort) | `default` |

More model families and verification history live in
[issue #712](https://github.com/radixark/miles/issues/712).

### Verify a new model TITO

To add a named family, register its `TITOTokenizer` and `FIXED_TEMPLATE` in
[`tito_tokenizer.py`](https://github.com/radixark/miles/blob/main/miles/utils/chat_template_utils/tito_tokenizer.py),
then run both checks. Either failure blocks support.

```bash
# CPU / fast: rendered token sequences remain append-only
python scripts/tools/verify_chat_template.py \
    --model <hf-id> --tito-model <family>

# GPU / end to end: the invariant holds under real model inference
python scripts/tools/verify_session_tito_tokenizer.py \
    --hf-checkpoint <hf-id> --tito-model <family> \
    --sglang-reasoning-parser <rp> --sglang-tool-call-parser <tcp> \
    --rollout-num-gpus-per-engine 1
```

### Choose replay matching

Some agent harnesses do not replay model messages verbatim: they may reserialize tool-call arguments, replace empty `arguments` with `"{}"`, or omit `reasoning_content` on the next request. Under the default matcher those replays count as divergence — v1 rolls back (or rejects), v2 branches a new lineage.

`--session-message-matcher` is process-wide and defaults to `strict`. It accepts a built-in selector or a trusted dotted import path.

| Selector | Behavior |
|---|---|
| `strict` | Preserves the existing comparison of `role`, `content`, `reasoning_content`, and `tool_calls`, including empty-value and tool-call `index` normalization. |
| `loose_tool_call` | Accepts everything `strict` accepts, plus equivalent JSON-object representations of `tool_calls[].function.arguments`. Call IDs, types, function names, order, unknown fields, and `reasoning_content` still have to match. |
| `role_content_only` | Compares only normalized `role` and `content`. **High risk:** different tool-call or reasoning histories can collapse into one session lineage. |
| dotted import path | Loads a trusted synchronous custom matcher; see [Customization](/user-guide/customization#session-message-matcher). |

The matcher only decides whether the message a client replays and the message stored at the same position in the session count as the same one.

- On a mismatch, the existing paths apply: v1 rolls back (or rejects), v2 branches.
- On a match, the stored messages and token snapshot stay authoritative inside the reusable prefix; only the suffix beyond it is tokenized anew from the client input.

Miles does not reconcile tool-call IDs across that boundary: deployments choosing `role_content_only` must themselves keep a stored call ID `A` followed by a replayed tool result referencing `B` protocol-compatible.

## Finalize a v2 episode

V2 keeps every trajectory leaf by default and masks shared completions so they
contribute to the loss once. Sibling order cannot distinguish a retry from valid
parallel work. Existing experiments can opt into the previous heuristic with
`--session-sample-picker-path miles.rollout.session.v2.picker_hub.drop_retries`.

**The agent function must join its child agents and tool work before returning.**
An idle model endpoint does not imply that a child running a tool has finished.
Miles then closes admission, waits for already-admitted model requests, and
exports a sealed snapshot. Agent exceptions, unresolved generations, and drain
timeouts produce an incomplete trace and an aborted rollout.

Custom clients use:

1. `POST /sessions/{id}/finish` with `{"producer_finished": true, "timeout": 60}`.
   Use `producer_finished: false` when the agent did not finish. The timeout is
   in seconds (0–90). Repeated calls with the same parameters return the same
   `snapshot_id`, completeness flag, and failed/unfinished request counts.
2. `POST /sessions/{id}/samples` with that `snapshot_id`, `max_seq_len`, and
   optional agent `metadata`. The first successful export is cached; retries
   must use the same export parameters. Incomplete sessions return no training
   samples and `empty_reason: "incomplete"`.
3. Decode the reply successfully, then `DELETE /sessions/{id}`.

The built-in v2 tracer retries transport failures once and retains sessions on
collection/decode failure. Sessions expire 15 minutes after finalization starts;
successful collections release them immediately. Inspect incomplete sessions
through `GET /sessions/{id}` before expiry. Snapshots are in memory and do not
survive a server restart. Timeout fencing prevents late commits; it does not
cancel upstream GPU work. Active-session collection without a snapshot remains
a live preview for existing clients. V1 keeps its existing lifecycle.

## Explicit agent contexts (v2)

Send `X-Miles-Agent-Run-Id` and `X-Miles-Context-Id` on each OpenAI or Anthropic
request to isolate continuation matching. A child has its own agent and context
IDs; `X-Miles-Parent-Agent-Run-Id` and optional `X-Miles-Parent-Tool-Call-Id`
describe its execution relationship without sharing the parent's token history.

Contexts register after their first request passes preparation. To register one
before it makes any model call, use `POST /sessions/{id}/contexts`:

```json
{"agent_run_id": "main", "context_id": "main-1"}
```

Repeat the same identity headers for that context. Context identities are immutable within the session; conflicting registrations
return 409. Parent/tool/compaction links are descriptive metadata and do not
require ordered registration. Compaction uses a new context ID with
`X-Miles-Derived-From-Context-Id` referencing the same agent's old context.
Changing the model, adapter, tools, or chat-template options also requires a new
context. These transitions always start a new token root.

Optional `X-Miles-Previous-Response-Id` chooses one predecessor inside the
context. Its stored messages must match the request prefix; an invalid reference
returns 409. Without it, equally deep matches start a fresh root. Requests without
identity headers remain inferred and cannot attach to explicitly identified
contexts. Miles consumes these headers before forwarding to inference.

Session metadata exposes the context registry and each node's stable
`generation_id`, `context_id`, and `identity_source`. Leaf sample metadata carries
the agent/context IDs. All contexts still belong to the same training episode.

## Generation delivery replay (v2)

Use a fresh `X-Miles-Idempotency-Key` for each intended generation and reuse it
only to retry delivery of the same request. Concurrent deliveries share one
operation; later deliveries replay its response, including after finalization.
A client disconnect does not cancel that operation. Reusing a key with different
request inputs returns 409. Errors are also retained: a new attempt needs a new
key. Keys and responses live only for the retained session, with at most 1024
keyed operations; there is no replay guarantee after release or server restart.

Successful responses include `X-Miles-Generation-Id` on both protocol adapters.
Unkeyed requests and fresh keys sample new generations, even when their prompts
are identical. All sampled attempts remain available to the configured picker;
this API does not choose which attempts should contribute to training.

## Instrumented harness completion

An agent function can return `AgentResult(metadata=..., producer_finished=...)`
from `miles.rollout.agentic.harness`. Set `producer_finished=False` when child/tool
completion is unknown. A normal return does not upgrade that explicit result.
Existing dict/None returns retain their contract that returning asserts all work
was joined.

`AgentRun` composes an AnyIO task group with context registration. Submit child
agents and background tool work through `run.create_task(...)`; exiting the
scope joins them, including registered grandchildren. `run.result(...)` is valid
only after that join. A child exception cancels and joins siblings; a cancelled
child prevents a complete result. External processes and tasks created outside
the group require a separate adapter-provided join boundary.

`GenerationRequest(context, previous_response_id=...)` supplies identity and a
stable idempotency key. Reuse the same instance's headers for transport retries;
construct a new instance for each sampling attempt. The
[instrumented example](../../examples/experimental/session_multi_agent/agent.py)
runs a parent and two independent reviewers through these APIs.

Score the final task state after joining producers. The default postprocessor
broadcasts one episode reward across all agent/context rows, and those rows
share one `rollout_id`. Reward normalization counts independent episodes, while
loss normalization uses each episode's total trainable tokens. Child agents do
not become extra GRPO trials. Finer-grained credit assignment requires a separate
objective; this path preserves the existing episode objective.

### Harbor's strict entry point

Use `examples.experimental.harbor.harbor_agent_function.run_with_completion`
to require an explicit producer attestation. It accepts only the boolean
`agent_result.metadata["miles_producer_finished"] == True` together with a
nonempty verifier reward report. Missing attestation, missing outcome, timeout,
and adapter failure remain incomplete and are filtered by v2 finalization.

The Harbor agent must emit that attestation after joining every child and tool
producer, before verification observes the final task state. Existing opaque
Claude Code bindings do not provide this proof or automatic per-child context
IDs. The strict entry point therefore aborts those uninstrumented trials; the
existing `run` entry point retains its legacy completion assumption. Wiring real
CLI child identities and validating that boundary remain integration work.

## Example

[`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker)
wires a multi-turn SWE agent, TITO session server, model-family registration,
reward, length limit, and environment teardown into production launchers.
