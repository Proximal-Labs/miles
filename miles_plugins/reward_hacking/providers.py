"""Model-gateway Responses API and existing Modal SGLang chat-completions API."""

import asyncio
import json
import os
from urllib.parse import quote, urlparse

import httpx

from miles_plugins.reward_hacking.prompt import SYSTEM_PROMPT

RESULT_SCHEMA = {
    "type": "object",
    "properties": {"reward_hacking": {"type": "boolean"}},
    "required": ["reward_hacking"],
    "additionalProperties": False,
}


def _required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before starting an evaluation")
    return value


def gateway_route(config):
    """Reuse an explicit gateway or the platform's existing Cloudflare credentials.

    There is deliberately no direct api.openai.com fallback.
    """
    headers = {}
    url = config.get("base_url") or os.environ.get(config.get("base_url_env", "OPENAI_BASE_URL"))
    if not url:
        account = quote(_required("CLOUDFLARE_ACCOUNT_ID"), safe="")
        gateway = quote(_required("AI_GATEWAY_ID"), safe="")
        url = f"https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/openai"
    if urlparse(url).hostname == "gateway.ai.cloudflare.com":
        headers["cf-aig-authorization"] = "Bearer " + _required("CF_AIG_TOKEN")
    headers["Authorization"] = "Bearer " + _required(config.get("api_key_env", "OPENAI_API_KEY"))
    return url.rstrip("/"), headers


def monitor_input(row):
    """Only the Inkling message fields enter the prompt, never labels or provenance."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty Inkling-format messages array")
    clean = []
    for message in messages:
        if message.get("role") not in {"user", "assistant", "system", "tool"}:
            raise ValueError("Unsupported message role")
        item = {"role": message["role"]}
        for key in ("content", "reasoning_content", "name", "tool_call_id"):
            if key in message:
                if not isinstance(message[key], str):
                    raise ValueError(f"{key} must be a string")
                item[key] = message[key]
        if "tool_calls" in message:
            calls = []
            for call in message["tool_calls"]:
                fn = call["function"]
                if not isinstance(fn["arguments"], dict) or not isinstance(fn["name"], str):
                    raise ValueError("Inkling tool calls require object arguments and a string name")
                calls.append({"type": "function", "function": {"name": fn["name"], "arguments": fn["arguments"]}})
            item["tool_calls"] = calls
        clean.append(item)
    return "Classify this recorded trace (which may be partial):\n" + json.dumps(
        {"messages": clean}, ensure_ascii=False, separators=(",", ":")
    )


def request_body(model, trace_text):
    if model["api"] == "responses":
        return {
            "model": model["model"],
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": trace_text}],
            "store": False,
            "truncation": "disabled",
            "max_output_tokens": model.get("max_output_tokens", 8192),
            "reasoning": {"effort": model.get("reasoning_effort", "medium")},
            "text": {
                "format": {"type": "json_schema", "name": "reward_hacking", "strict": True, "schema": RESULT_SCHEMA}
            },
        }
    if model["api"] != "chat_completions":
        raise ValueError("Unsupported API")
    return {
        "model": model["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": trace_text}],
        "max_tokens": model.get("max_output_tokens", 8192),
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": model.get("reasoning_effort", 0.7)},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "reward_hacking", "strict": True, "schema": RESULT_SCHEMA},
        },
    }


def parse_response(data, api):
    if api == "responses":
        if data.get("status") != "completed":
            raise ValueError(f"Response not completed: {data.get('status')}")
        blocks = [
            b for item in data.get("output", []) if item.get("type") == "message" for b in item.get("content", [])
        ]
        if any(b.get("type") == "refusal" for b in blocks):
            raise ValueError("Model refused classification")
        text = "".join(b["text"] for b in blocks if b.get("type") == "output_text")
    else:
        choice = data["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError(f"Generation did not finish: {choice.get('finish_reason')}")
        text = choice["message"].get("content")
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {"reward_hacking"}:
        raise ValueError("Classification must contain only reward_hacking")
    if type(result["reward_hacking"]) is not bool:
        raise ValueError("reward_hacking must be a boolean")
    return result


async def classify(client, model, route, trace_text, *, retries=2):
    url, headers = route
    path = "/responses" if model["api"] == "responses" else "/chat/completions"
    body = request_body(model, trace_text)
    for attempt in range(retries + 1):
        try:
            response = await client.post(url + path, headers=headers, json=body)
        except httpx.TransportError:
            if attempt == retries:
                raise
        else:
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == retries:
                if response.is_error:
                    raise ValueError(f"Provider HTTP {response.status_code}; no label assigned")
                data = response.json()
                try:
                    classification = parse_response(data, model["api"])
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    return {
                        "status": "error",
                        "error": str(error),
                        "usage": data.get("usage"),
                        "response_id": data.get("id"),
                        "raw_response": data,
                        "attempts": attempt + 1,
                    }
                return {
                    "status": "ok",
                    **classification,
                    "usage": data.get("usage"),
                    "response_id": data.get("id"),
                    "raw_response": data,
                    "attempts": attempt + 1,
                }
        await asyncio.sleep(min(2**attempt, 8))
    raise AssertionError("Unreachable")
