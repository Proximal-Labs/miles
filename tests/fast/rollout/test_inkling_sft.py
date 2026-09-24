import copy
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
from miles.rollout.inkling_sft import DEFAULT_EFFORT, generate_rollout, load_renderer, render_example

# The official TMLv0 renderer is a CPU data-preparation dependency
# (tools/requirements-inkling-sft.txt), not part of the training image the CPU suite runs in.
needs_renderer = pytest.mark.skipif(
    find_spec("tinker_cookbook") is None, reason="needs tinker-cookbook (tools/requirements-inkling-sft.txt)"
)


def _conversation():
    return [
        {"role": "system", "content": "SYSTEM_ONLY"},
        {"role": "user", "content": "USER_ONLY"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "FIRST_REASONING",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"abc"}'}}
            ],
        },
        {"role": "tool", "name": "lookup", "tool_call_id": "call_1", "content": "TOOL_ONLY"},
        {"role": "assistant", "content": "FIRST_ANSWER"},
        {"role": "user", "content": "SECOND_USER"},
        {"role": "assistant", "reasoning_content": "SECOND_REASONING", "content": "FINAL_ANSWER"},
    ]


@needs_renderer
def test_reasoning_tools_and_end_tokens_are_supervised_without_prompt_leakage():
    messages = _conversation()
    original = copy.deepcopy(messages)
    tokens, mask = render_example(load_renderer(), messages, [], 8192, DEFAULT_EFFORT)
    text = load_renderer().tokenizer.decode(tokens)
    supervised = load_renderer().tokenizer.decode(
        [token for token, weight in zip(tokens, mask, strict=True) if weight]
    )
    for value in ("FIRST_REASONING", "SECOND_REASONING", "FIRST_ANSWER", "FINAL_ANSWER", '"key":"abc"'):
        assert value in supervised
    for value in ("SYSTEM_ONLY", "USER_ONLY", "SECOND_USER", "TOOL_ONLY", "Thinking effort level"):
        assert value in text
        assert value not in supervised
    assert supervised.count("<|content_model_end_sampling|>") == 3
    assert messages == original


@needs_renderer
def test_length_cap_rejects_instead_of_truncating():
    with pytest.raises(ValueError, match="no truncation"):
        render_example(load_renderer(), _conversation(), [], 10, DEFAULT_EFFORT)


@needs_renderer
def test_masked_assistant_turn_is_kept_in_context():
    messages = _conversation()
    messages[2]["step_loss_mask"] = 0
    tokens, mask = render_example(load_renderer(), messages, [], 8192, DEFAULT_EFFORT)
    assert "FIRST_REASONING" in load_renderer().tokenizer.decode(tokens)
    assert "FIRST_REASONING" not in load_renderer().tokenizer.decode(
        [t for t, m in zip(tokens, mask, strict=True) if m]
    )


@needs_renderer
def test_rollout_preserves_internal_tool_mask_and_checks_sequence_cap():
    tokens, mask = render_example(load_renderer(), _conversation(), [], 8192, DEFAULT_EFFORT)
    sample = SimpleNamespace(metadata={"format": "inkling-sft-v2", "tokens": tokens, "loss_mask": mask})
    buffer = SimpleNamespace(get_samples=lambda _: [[sample]])
    result = generate_rollout(SimpleNamespace(rollout_batch_size=1, seq_length=8192), 0, buffer)
    assert result == [sample]
    assert sample.loss_mask == mask[mask.index(1) :]
    assert 0 in sample.loss_mask
    with pytest.raises(ValueError, match="exceeds"):
        generate_rollout(SimpleNamespace(rollout_batch_size=1, seq_length=10), 0, buffer)


@needs_renderer
def test_matches_official_renderer_with_thinking_tools_and_max_effort():
    from tinker_cookbook.renderers import TrainOnWhat

    renderer = load_renderer()
    messages = _conversation()
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "Lookup a key", "parameters": {"type": "object"}},
        }
    ]
    official = copy.deepcopy(messages)
    for message in official:
        if message.get("reasoning_content"):
            parts = [{"type": "thinking", "thinking": message.pop("reasoning_content")}]
            if message.get("content"):
                parts.append({"type": "text", "text": message["content"]})
            message["content"] = parts
    prefix = renderer.create_conversation_prefix_with_tools([tools[0]["function"]])
    expected, weights = renderer.build_supervised_example(
        prefix + official, TrainOnWhat.ALL_ASSISTANT_MESSAGES, effort=0.99
    )
    tokens, mask = render_example(renderer, messages, tools, 8192, DEFAULT_EFFORT)
    assert (tokens, mask) == (expected.to_ints(), weights.tolist())
    assert "Thinking effort level: 0.99" in renderer.tokenizer.decode(tokens)
    generation = renderer.build_generation_prompt(prefix + official[:-1], effort=0.99).to_ints()
    assert tokens[: len(generation)] == generation


@needs_renderer
@pytest.mark.parametrize("effort", ["max", True, float("nan"), float("inf"), -0.1, 1.0])
def test_invalid_effort_is_rejected(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        render_example(load_renderer(), _conversation(), [], 8192, effort)


def test_legacy_prepared_records_are_rejected():
    sample = SimpleNamespace(metadata={"format": "inkling-sft-v1", "tokens": [1, 2], "loss_mask": [0, 1]})
    buffer = SimpleNamespace(get_samples=lambda _: [[sample]])
    with pytest.raises(ValueError, match="rerun data preparation"):
        generate_rollout(SimpleNamespace(rollout_batch_size=1, seq_length=8192), 0, buffer)
