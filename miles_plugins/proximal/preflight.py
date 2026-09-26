"""Checks that fail a run in seconds instead of deep into it.

A run spends minutes starting its trainer and tens of minutes on rollouts before its
first step, so each boundary it crosses is checked before the stage that depends on it:

- ``check_serving``, before the trainer starts: the replicas were deployed with a serving
  contract this run fits. A redeploy keeps live replicas on their old config.
- ``canary``, after the first policy is published and before any platform run: a real
  capture session answers a model call and seals it as a sample, and a turn past the
  sequence budget reaches the agent as OpenAI's context-limit error, which agent-px
  ends as a graded rollout instead of a failed one.
"""

import asyncio
import re
import time
import uuid

import httpx

from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization, secret_env
from miles_plugins.proximal.clients import CaptureClient
from miles_plugins.proximal.contracts import Attempt, Policy, affinity_headers, pinned_dataset, serving_mismatches

# agent-px's recognizer for a provider's context-window rejection
# (packages/agent-px/contract/src/errors.ts, isProviderContextLimitError).
AGENT_PX_CONTEXT_LIMIT = re.compile(r"\bmaximum context length is [\d,]+ tokens\b", re.IGNORECASE)


class PreflightFailed(RuntimeError):
    """The run cannot work as configured; nothing paid has started."""


async def check_serving(authorization: AuthorizedRun, client: httpx.AsyncClient, *, probes: int = 24) -> int:
    """Every replica the probes reach fits this run; returns how many distinct contracts answered.

    Probes carry distinct affinity keys, so they spread over the replicas; a replica
    none reached is still checked by its first session (capture refuses what it can't serve).
    """
    config = require_authorization(authorization)
    capture = CaptureClient(authorization, client)
    found = await asyncio.gather(*(capture.serving_contract(f"preflight-{i}") for i in range(probes)))
    distinct = {contract.model_dump_json(): contract for contract in found}
    for deployed in distinct.values():
        if reasons := serving_mismatches(config, deployed):
            raise PreflightFailed(
                "A serving replica cannot serve this run: "
                + "; ".join(reasons)
                + ". Redeploy serving with a config this run fits; stop the app first, because a redeploy "
                "keeps live replicas on their old config."
            )
    return len(distinct)


async def canary(authorization: AuthorizedRun, client: httpx.AsyncClient, policy: Policy) -> dict[str, float]:
    """One sealed model call and one over-long turn through capture, as the platform's agent makes them."""
    config = require_authorization(authorization)
    capture = CaptureClient(authorization, client)
    platform = {"Authorization": f"Bearer {secret_env(config.capture.platform_key_env)}"}
    sampling = config.research.sampling

    def attempt() -> Attempt:
        return Attempt(
            attempt_id=f"preflight-{uuid.uuid4().hex}",
            run_id=config.run_id,
            group_id=f"preflight-{uuid.uuid4().hex}",
            sample_index=0,
            dataset_sha256=pinned_dataset(config.dataset).sha256,
            task=config.dataset.tasks[0],
            harness=config.harness,
            policy=policy,
            sampling=sampling,
        )

    async def call(rollout_url: str, rollout_id: str, content: str) -> httpx.Response:
        return await client.post(
            f"{rollout_url}/chat/completions",
            headers=platform | affinity_headers(rollout_id),
            json={
                "model": config.base_model.name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 64,
                "reasoning_effort": config.model_protocol.reasoning_effort,
            },
        )

    first = attempt()
    handle = await capture.create(first)
    try:
        started = time.monotonic()
        reply = await call(handle.base_url, handle.rollout_id, "Reply with the single word: ready.")
        seconds = time.monotonic() - started
        if reply.status_code != 200:
            raise PreflightFailed(f"A model call through capture failed ({reply.status_code}): {reply.text[:300]}")
        receipt, _ = await capture.collect(handle, first)
        if receipt.num_calls != 1 or receipt.num_tokens <= 0:
            raise PreflightFailed("Capture sealed no model call")
    finally:
        await capture.release(handle)

    over = attempt()
    handle = await capture.create(over)
    try:
        reply = await call(handle.base_url, handle.rollout_id, "ready " * (sampling.max_sequence_tokens + 1024))
    finally:
        await capture.release(handle)
    error = reply.json().get("error") if reply.headers.get("content-type", "").startswith("application/json") else None
    if (
        reply.status_code != 400
        or not isinstance(error, dict)
        or error.get("code") != "context_length_exceeded"
        or not AGENT_PX_CONTEXT_LIMIT.search(str(error.get("message", "")))
    ):
        raise PreflightFailed(
            f"An over-long turn did not come back as OpenAI's context-limit error ({reply.status_code}): "
            f"{reply.text[:300]}"
        )
    return {"reply_seconds": round(seconds, 1), "sealed_tokens": receipt.num_tokens}
