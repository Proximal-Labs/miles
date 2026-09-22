# Required proximal-mono integration

Audited against `origin/main` `f30a80d7099d68594a7153f148cd5d4d095cba45`. This document describes **proposed additions**, not already-deployed APIs. Miles fails capability checks before submitting a rollout against an older server. No proximal-mono checkout is modified by this PR.

Two cohesive platform changes are needed, plus deployment configuration for the provided replica gateway. These changes belong at the shared execution/completion boundary, not a mini-SWE-specific branch.

## 1. Scoped training inference binding on the existing run RPC

Extend `packages/proto/proximal/v1/environment_run.proto` and the typed CreateEnvironmentRun domain input with an optional binding. Allocate currently unused protobuf field numbers when implementing; do not reuse retired tags.

```protobuf
message TrainingInferenceBinding {
  uint32 protocol_version = 1; // exactly 1
  string request_sha256 = 2;  // immutable execution request fingerprint
  string session_base_url = 3; // standard OpenAI base URL, ending /sessions/<id>/v1
  string session_api_key = 4; // secret scoped to this session
  string model = 5;           // public base model name, not a mutable alias
  string harness_revision = 6;
}
```

Add `training_binding` to `CreateEnvironmentRunRequest`. For a training-bound run:

- Authenticate and authorize normal project/environment access before resources.
- Validate protocol, approved capture URL origin, nonempty scoped credential, exact supported harness revision, explicit image/source pin and one requested instance.
- Route **all** model calls for this rollout through the supplied Chat Completions client, including reasoning/tool responses. Do not fall back to the standard model catalog, Responses client, or a default provider. Fail before launch for unsupported harness capabilities.
- Persist the training fingerprint and resolved source/image/harness binding with the run. Duplicate `runId` with exactly equal execution inputs and the same session binding is idempotent. A mismatch is `ALREADY_EXISTS`/409. Do not compare just `environmentId` and `instances`.
- Keep `session_api_key` private: secret-backed durable workflow input/reference as appropriate, redacted from logs, UI, tracing, public run metadata and normal RPC responses. Do not place it in model-visible context or a shared endpoint catalog.
- Preserve sampling fields supplied by the harness only when accepted by the session contract. The capture service supplies the required distribution, token budgets, thinking mode, logprobs and token capture. Disable compaction, subagents and provider-native unrecorded tools for this contract; unsupported attempts must fail explicitly.

The client sends the existing fields:

```json
{
  "runId": "unique-attempt-id",
  "environmentId": 123,
  "imageId": 456,
  "sourceCommitSha": "<pinned 40-hex commit>",
  "instances": 1,
  "ensureRolloutLaunchWorkflows": true,
  "autoTriggerAnalysis": false,
  "autoTriggerPostQa": false,
  "config": {
    "agents": [{"agentType": "<supported harness>", "agentModel": "<base model>", "agentTimeoutSec": 1800}],
    "harborOptions": {"maxTurns": 60, "maxSessionTokens": 32768, "p2pEnforce": true}
  },
  "trainingBinding": {
    "protocolVersion": 1,
    "requestSha256": "<sha256 of canonical Miles Attempt JSON>",
    "sessionBaseUrl": "https://capture.example/sessions/<session>/v1",
    "sessionApiKey": "<secret>",
    "model": "agentica-org/DeepSWE-Preview",
    "harnessRevision": "<pinned platform harness commit>"
  }
}
```

Thread the typed binding through the existing run/workflow/solver input path. Relevant audited homes are `core/environmentRun`, `servers/environmentRunServer.ts`, `temporal/workflows/rollout-solver/rolloutSolverAgent.ts`, and `core/llmRuntime/completionClient.ts`. The latter's current Modal path uses Responses. Select a scoped OpenAI Chat Completions adapter at this shared seam; each supported agent-px harness receives the same typed completion capability. An adapter specific to mini-SWE would leave sibling harnesses able to bypass training capture.

## 2. Capability and stored-provenance acknowledgements

Add a read-only `EnvironmentRunService.GetTrainingCapabilities` RPC. JSON request is `{}`; response:

```json
{
  "protocolVersion": 1,
  "harnessRevision": "<40-hex revision>",
  "supportedAgentTypes": ["<actual supported generic harness names>"],
  "pinnedSource": true,
  "linearTito": true
}
```

Return only capabilities the live execution path enforces. `linearTito` means text-only, linear, append-compatible model history using the scoped Chat client, with no compaction/subagent model calls. The harness revision must change when prompts/tools/verifier semantics change, or resolve to an immutable contract digest represented by the revision.

Add `training_request_sha256` to both `CreateEnvironmentRunResponse` and `GetRunSummaryResponse`. Read it from the **stored execution binding**. Never merely reflect the query's input. Submission acknowledges the accepted binding; final summary certifies the binding actually used by execution. Existing `runId`, `imageId`, `sourceCommitSha`, and container `agentType` are also checked by Miles.

`instancesStarted` may be 0 on an idempotent replay or 1 on a new single-instance submission. More than one is rejected. A duplicate must not launch a second rollout. If the execution can substitute inputs after submission, reject that substitution for training-bound runs and surface failure; the original digest must not certify a different execution.

No new result store or teardown API is needed. Existing generic APIs already provide the required result facts:

- `ProjectService.ListProjectEnvironments`: pinned dataset membership validation.
- `EnvironmentRunService.GetEnvironmentRunContainers`: terminal status, rollout ID, agent type, canonical reward, `rewardScored`, and execution `error`.
- `EnvironmentRunService.GetRunSummary`: resolved image/source and the proposed stored fingerprint.
- `EnvironmentRunService.StopEnvironmentRun`: logical cancellation only.

Miles accepts only SUCCESS/COMPLETED with no execution error and a finite scored reward. A scored proto-JSON omitted reward is zero. ERROR/TIMEOUT/STOPPED/FAILED, absent grading, or failed provenance are ineligible. The result reader uses the generic boundary shared by Harbor and taskrunner. Submission currently carries the platform’s existing `harborOptions` execution limits. Advertise a harness only when training-bound execution enforces all those limits; the current taskrunner restriction on these fields must not be bypassed or silently ignored. Supporting another execution engine requires translating the same declared limits at the platform boundary. Miles does not select or tear down sandboxes.

## Serving configuration

Run the provided `ReplicaGateway` beside each existing Modal SGLang process. Mount the same existing Volume on every replica, use a separate local cache, and expose the gateway through the existing authenticated fleet endpoint. It loads immutable adapters against loopback SGLang before inference and returns evidence headers. It must be the sole owner of the `miles-*` namespace; SGLang and the gateway restart together.

Configure the same pinned base, Qwen3 reasoning/tool parsers and adapter target modules/rank on every replica. Set SGLang's adapter capacity at least to the gateway capacity. Miles does not enumerate replicas or call a load-balanced engine-control endpoint as if it were a broadcast. A Modal scale-to-zero fleet may require a warm replica for the first publication; choose keep-warm/autoscaling in the platform deployment.

## Platform test checklist

- Missing/unsupported binding fails before sandbox launch; no fallback model call.
- Exercise at least two agent-px harnesses through the shared completion seam.
- Every model call uses the scoped URL/key; capture admin and inference keys never enter the run.
- Duplicate equal Create is idempotent; source/harness/session/sampling fingerprint conflict fails.
- Final digest comes from the executed binding, with matching source/image and canonical reward.
- Valid zero grade is distinct from missing/failed grading; cleanup state does not affect eligibility.
- Unknown protocol, wrong harness revision, compaction and subagent calls fail explicitly.
- Run ordinary TypeScript checks and the relevant platform unit tests before a live rollout.
