# Required proximal-mono change: route a run's model calls to Miles capture

Audited against proximal-mono `origin/main` `593de5e46063`. This is the one platform change Miles needs. Everything else uses run APIs that already exist.

## What Miles sends and reads (no change)

Miles creates one platform run per attempt with the existing `EnvironmentRunService.CreateEnvironmentRun`, using only existing fields:

```json
{
  "runId": "<attempt id: deterministic, retries are idempotent>",
  "environmentId": 123, "imageId": 456, "sourceCommitSha": "<40-hex>",
  "instances": 1,
  "ensureRolloutLaunchWorkflows": true, "autoTriggerAnalysis": false, "autoTriggerPostQa": false,
  "config": {
    "agents": [{
      "agentType": "<mini-swe harness>",
      "agentModel": "<run config platform_route.model>",
      "endpointName": "<run config platform_route.endpoint_name>",
      "agentTimeoutSec": 1800,
      "reasoningEffort": "AGENT_REASONING_EFFORT_HIGH"
    }],
    "harborOptions": {"maxTurns": 60, "maxSessionTokens": 32768, "p2pEnforce": true}
  }
}
```

It reads results with the existing `GetEnvironmentRunContainers` (status, `rolloutIndex`, `agentType`, `rewardScored`, `reward`, `error`) and `GetRunSummary` (`imageId`, `sourceCommitSha`), and checks project membership with `ProjectService.ListProjectEnvironments` before the first run. `StopEnvironmentRun` is a logical cancel; Miles never tears anything down.

Eligible: `SUCCESS`/`COMPLETED`, no execution error, `rewardScored` (a scored zero is valid). Anything else is an execution failure and never becomes a zero-reward sample.

## The change: a Chat Completions registry endpoint with per-run routing

Today the Modal endpoint registry (LiveConfig `modal.inference.endpoints`, `packages/backend/src/core/llmRuntime/modal/endpoints.ts`) only selects the Responses-based Modal provider, on `*.modal.direct` URLs, with one global credential. Miles needs one registry entry that:

1. **Selects the existing agent-px Chat Completions adapter** (`packages/agent-px/provider/openai-chat-completions`), not the Responses provider.
2. **Holds a credential reference**, not the secret: e.g. `{ "secretEnv": "MILES_CAPTURE_PLATFORM_KEY" }`, resolved on the rollout workers and sent as the adapter's API key (`Authorization: Bearer <key>`). It is the capture service's `platform_key_env`. Redact it like any provider key.
3. **Derives a per-run base URL**: `${baseURL}/runs/${runId}/${rolloutIndex}/v1`, where `runId` is the environment run ID and `rolloutIndex` the container's rollout index (always 0 today; Miles sends `instances: 1`). The capture service maps that route to the attempt Miles registered before creating the run.
4. **Declares the wire model** sent as `model` (the base model name, e.g. `Qwen/Qwen3-0.6B`), under a platform model ID such as `miles/<name>` that `agentModel` names. The closed Modal model set and context-window registry need an entry or a registry-driven equivalent.
5. **Allows the capture service's HTTPS host** (the current `*.modal.direct` rule would reject it unless the capture service is deployed as a Modal `app.server`).

A proposed entry shape (the platform owns the final schema):

```json
{
  "kind": "chat_completions",
  "baseURL": "https://<capture host>",
  "routing": "per_run",
  "credential": { "secretEnv": "MILES_CAPTURE_PLATFORM_KEY" },
  "model": "Qwen/Qwen3-0.6B",
  "mode": "dedicated"
}
```

The per-run snapshot (`modal_endpoint_assignments`) already pins the endpoint for a run's lifetime; the derived URL is a function of the snapshot and the run, so nothing new is persisted.

## What the capture service accepts from agent-px

Read from agent-px at the pinned commit; covered by Miles's tests.

- Streaming requests with `stream_options.include_usage`: the reply is one SSE chunk with the full message and usage, then `[DONE]`. The engine call itself is never streamed.
- `prompt_cache_key`, `prompt_cache_retention`: ignored (no effect on sampling).
- `reasoning_effort`: must equal the run config's `model_protocol.reasoning_effort` (sent as `agents[].reasoningEffort`); otherwise 422. Not forwarded: the TITO renderer owns thinking.
- `max_completion_tokens` (or `max_tokens`): the per-turn budget, capped by the training contract.
- Tools with `strict: true`: `strict` is removed before the engine. With it, SGLang constrains decoding to the schema, and behavior logprobs would come from a different distribution than training computes.
- No `temperature`/`top_p`/`top_k`: the capture service applies the training contract's sampling.
- Replayed history as agent-px rebuilds it: `content: null` on tool-only turns, `reasoning_content`, compact re-serialized tool arguments. Miles's `loose_tool_call` matcher accepts JSON-equivalent arguments, so the prefix tokens come from the session's own record and nothing is rolled back.
- A history agent-px rewrote differently (unparseable arguments replayed as `"{}"`, the truncated-reasoning placeholder) does not match. Miles then rolls back one turn and re-renders it as context: that turn's tokens are not trained, the rest are.

## Harness and limits

The first supported harness is **mini-swe**: it disables compaction, so every rollout is one linear token history. `maxTurns` and `maxSessionTokens` come from `harborOptions`. The default harness compacts with a separate summary request and needs multi-segment samples; not supported yet.

## Deliberately not required (follow-ups)

The earlier proposal also had a `GetTrainingCapabilities` RPC, a per-run training binding with a per-session URL and key, and a stored request fingerprint echoed on `CreateEnvironmentRun`/`GetRunSummary`. They are dropped from the first version:

- **Routing is proven by the capture service.** A run whose calls never reach the capture service has no captured tokens, so its sample is rejected, not trained.
- **Harness revision is operator-asserted.** The run config pins it as part of the training contract, but the platform does not certify which revision executed. Certifying it (capability or summary field) is the main follow-up.
- **The run ID binds the grade to the attempt**, together with image and source commit from `GetRunSummary`.

## Reachability

The rollout workers that run agent-px must reach the capture service URL. With a local platform worker on the same host, loopback works (the offline Stage A harness does this). Against staging, the capture service needs a public HTTPS URL, e.g. deployed as a Modal `app.server` next to the serving pool.

## Platform test checklist

- A registry entry of kind `chat_completions` with `routing: per_run` builds `${baseURL}/runs/${runId}/${rolloutIndex}/v1` and uses the credential reference; no fallback to another provider or the global Modal key.
- The rendered key never appears in logs, traces, run metadata or the persisted endpoint snapshot.
- A run with `agents[].endpointName` for that entry sends every mini-swe model call there, with `agents[].reasoningEffort` as `reasoning_effort`.
- Existing Modal Responses endpoints are unchanged.
- One capped live rollout (e.g. `maxTurns` 3) reaches a Miles capture service and returns a scored container.
