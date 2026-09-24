import json
from pathlib import Path

import pydantic
import pytest

from miles_plugins.proximal.capture_server import check_tito_protocol
from miles_plugins.proximal.contracts import RunConfig

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "proximal" / "e2e" / "run.stage-a.json"


def _config(tito_model: str, reasoning_parser: str, tool_call_parser: str) -> RunConfig:
    raw = json.loads(EXAMPLE.read_text())
    raw["tito_model"] = tito_model
    raw["model_protocol"].update(reasoning_parser=reasoning_parser, tool_call_parser=tool_call_parser)
    return RunConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("tito_model", "reasoning_parser", "tool_call_parser"),
    [
        ("qwen3", "qwen3", "qwen25"),
        ("qwen35", "qwen3", "qwen3_coder"),
        ("qwen36", "qwen3", "qwen3_coder"),
        ("qwen38small", "qwen3", "qwen3_coder"),
        ("qwennext", "qwen3", "qwen25"),
    ],
)
def test_each_family_accepts_the_parsers_it_binds(tito_model, reasoning_parser, tool_call_parser):
    check_tito_protocol(_config(tito_model, reasoning_parser, tool_call_parser))


def test_a_parser_the_family_does_not_bind_is_rejected():
    # Qwen3.8 emits XML tool calls; serving them with the Qwen2.5 JSON parser would
    # hand the agent different tool calls than capture renders.
    with pytest.raises(ValueError, match="qwen38small template family binds 'qwen3_coder'"):
        check_tito_protocol(_config("qwen38small", "qwen3", "qwen25"))


def test_families_capture_cannot_render_yet_are_rejected_by_the_contract():
    with pytest.raises(pydantic.ValidationError):
        _config("inkling", "qwen3", "qwen25")
