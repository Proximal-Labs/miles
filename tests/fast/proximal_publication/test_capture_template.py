"""Capture renders turns incrementally, so it must use its family's fixed chat template.

Qwen3.8's native template refuses a conversation without a user message, which is what
incremental rendering produces for every turn after the first. Needs the Qwen3.8-27B
tokenizer files (``PROXIMAL_TEST_QWEN38_TOKENIZER``).
"""

import json
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils import get_tito_tokenizer
from miles_plugins.proximal.capture_server import capture_registry, capture_tokenizer, session_config
from miles_plugins.proximal.contracts import RunConfig

TOKENIZER = os.environ.get("PROXIMAL_TEST_QWEN38_TOKENIZER")
pytestmark = pytest.mark.skipif(TOKENIZER is None, reason="PROXIMAL_TEST_QWEN38_TOKENIZER is not set")
QWEN38 = Path(__file__).resolve().parents[3] / "examples" / "proximal" / "qwen38"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]
FIRST_TURN = [
    {"role": "system", "content": "You are a software engineer."},
    {"role": "user", "content": "List the files."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should run ls.",
        "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}
        ],
    },
]
TOOL_RESULT = [{"role": "tool", "tool_call_id": "call_0", "content": "README.md\nsetup.py"}]


def _config() -> RunConfig:
    raw = json.loads((QWEN38 / "smoke" / "run.template.json").read_text())
    raw["dataset"]["tasks"] = json.loads((QWEN38 / "smoke" / "tasks.json").read_text())["tasks"]
    raw["inference_url"] = "https://pool.example"
    raw["tokenizer_path"] = TOKENIZER
    return RunConfig.model_validate_json(json.dumps(raw))


def _tool_turn(tokenizer) -> list[int]:
    tito = get_tito_tokenizer(tokenizer, "qwen38small", chat_template_kwargs={"enable_thinking": True})
    return tito.tokenize_additional_messages(
        FIRST_TURN, FIRST_TURN + TOOL_RESULT, template_args={"tools": TOOLS, "enable_thinking": True}
    )


def test_the_native_template_cannot_render_a_tool_turn_incrementally():
    native = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    with pytest.raises(ValueError, match="No user query found"):
        _tool_turn(native)


def test_capture_renders_a_tool_turn_with_the_fixed_template():
    tokenizer = capture_tokenizer(TOKENIZER, "qwen38small")
    text = tokenizer.decode(_tool_turn(tokenizer))
    assert "README.md" in text and text.rstrip().endswith("<|im_start|>assistant\n<think>")


def test_capture_refuses_a_tokenizer_without_the_fixed_template():
    native = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    with pytest.raises(ValueError, match="fixed chat template"):
        capture_registry(_config(), native)
    capture_registry(_config(), capture_tokenizer(TOKENIZER, "qwen38small"))


def test_capture_renders_the_runs_reasoning_effort():
    # Through capture's own renderer and request path, with the request body capture
    # forwards (it names enable_thinking only).
    from miles.rollout.session.request_args import prepare_chat_request

    tokenizer = capture_tokenizer(TOKENIZER, "qwen38small")

    def system_prompt(config: RunConfig) -> str:
        registry = capture_registry(config, tokenizer)
        request = {"messages": FIRST_TURN[:2], "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": True}}
        prepared = prepare_chat_request(
            request, registry.tito_tokenizer, config=session_config(config), turn_args=None
        )
        return tokenizer.apply_chat_template(
            FIRST_TURN[:2], tokenize=False, add_generation_prompt=True, **prepared.template_args
        )

    config = _config()
    assert config.model_protocol.reasoning_effort == "xhigh"
    assert "Reasoning effort is set to xhigh" in system_prompt(config)
    low = config.model_copy(
        update={"model_protocol": config.model_protocol.model_copy(update={"reasoning_effort": "low"})}
    )
    assert "Reasoning effort is set to low" in system_prompt(low)
