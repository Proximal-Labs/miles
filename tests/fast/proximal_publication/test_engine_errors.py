"""Engine errors reach the agent in the OpenAI shape, context-window rejections as such.

SGLang answers with a flat body ({"object": "error", "message": ...}); OpenAI clients read
{"error": {...}} and otherwise report "400 status code (no body)". A rollout that outgrows
its context must reach the agent as OpenAI's context-limit error, which agent-px turns into
a clean, graded end rather than a failed rollout (and a failed group).
"""

import json
import re

import orjson
import pytest
from fastapi.responses import Response

from miles_plugins.proximal.capture_server import context_limit_error, engine_error

# agent-px's recognizers (packages/agent-px/contract/src/errors.ts, isProviderContextLimitError).
AGENT_PX_CONTEXT_LIMIT = re.compile(r"\bmaximum context length is [\d,]+ tokens\b", re.IGNORECASE)


def _sglang(message: str, status: int = 400) -> Response:
    body = {"object": "error", "message": message, "type": "BadRequestError", "param": None, "code": status}
    return Response(orjson.dumps(body), status_code=status, media_type="application/json")


@pytest.mark.parametrize(
    "message",
    [
        "Requested token count exceeds the model's maximum context length of 262144 tokens. You requested a total "
        "of 270000 tokens: 261900 tokens from the input messages and 8100 tokens for the completion.",
        "The input (263000 tokens) is longer than the model's context length (262144 tokens).",
    ],
)
def test_a_context_window_rejection_reaches_the_agent_as_openai_context_limit(message):
    response = engine_error(_sglang(message), 262144)
    error = json.loads(response.body)["error"]
    assert response.status_code == 400
    assert error["code"] == "context_length_exceeded" and error["type"] == "invalid_request_error"
    assert AGENT_PX_CONTEXT_LIMIT.search(error["message"]) and message in error["message"]


def test_other_engine_errors_keep_their_message_in_the_openai_shape():
    response = engine_error(_sglang("Unknown LoRA adapter", 400), 262144)
    error = json.loads(response.body)["error"]
    assert response.status_code == 400 and error["message"] == "Unknown LoRA adapter" and error["code"] is None
    assert engine_error(Response(b"upstream down", status_code=502), 262144).status_code == 502


def test_capture_budget_exhaustion_is_a_context_limit_too():
    error = json.loads(context_limit_error(262144, "no room for a reply").body)["error"]
    assert error["code"] == "context_length_exceeded" and AGENT_PX_CONTEXT_LIMIT.search(error["message"])
