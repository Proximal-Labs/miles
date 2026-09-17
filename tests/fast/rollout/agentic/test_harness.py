"""Delivery retries must reuse an attempt's identity without affecting later calls."""

from miles.rollout.agentic.harness import GenerationRequest
from miles.rollout.session.v2.contexts import SessionContext


def test_generation_request_identity_is_stable_only_within_one_attempt():
    context = SessionContext(agent_run_id="child", context_id="child", parent_agent_run_id="main")
    attempt = GenerationRequest(context, previous_response_id="previous")
    first_headers = attempt.headers()
    assert attempt.headers() == first_headers
    assert GenerationRequest(context).headers()["X-Miles-Idempotency-Key"] != first_headers["X-Miles-Idempotency-Key"]
    first_headers["X-Miles-Idempotency-Key"] = "changed"
    assert attempt.headers()["X-Miles-Idempotency-Key"] != "changed"
