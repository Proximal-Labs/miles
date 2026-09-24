"""Text-only SFT using Thinking Machines' official TMLv0 tokens and weights."""

import copy
import json
import math
from functools import cache
from importlib.metadata import distribution, version

MODEL = "thinkingmachines/Inkling-Small"
DEFAULT_EFFORT = 0.99


@cache
def load_renderer():
    # Rendering dependencies are CPU preparation-only, not training-worker dependencies.
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_renderer(model_info.get_recommended_renderer_name(MODEL), get_tokenizer(MODEL))


def renderer_provenance():
    direct_url = distribution("tinker-cookbook").read_text("direct_url.json")
    return {
        "model": MODEL,
        "renderer": "tml_v0",
        "tinker_cookbook_version": version("tinker-cookbook"),
        "tinker_cookbook_source": json.loads(direct_url) if direct_url else None,
        "tml_renderers_version": version("tml-renderers"),
        "tokenizer": "tml_renderers.tokenizers.o200k_base_chat",
    }


def render_example(renderer, messages: list[dict], tools: list[dict], max_length: int, effort: float):
    from tinker_cookbook.renderers import TrainOnWhat

    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("An SFT conversation must end with an assistant target")
    if (
        isinstance(effort, bool)
        or not isinstance(effort, (int, float))
        or not math.isfinite(effort)
        or not 0 <= effort < 1
    ):
        raise ValueError("reasoning_effort must be a finite number in [0, 1)")
    messages = copy.deepcopy(messages)
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported message role")
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise ValueError("This recipe accepts text-only string content")
        if message.get("reasoning_content") is not None and not isinstance(message["reasoning_content"], str):
            raise ValueError("reasoning_content must be a string")
        if "content_blocks" in message:
            raise ValueError("Use reasoning_content, content and tool_calls for this dataset")
        if message.get("step_loss_mask", 1) not in (0, 1):
            raise ValueError("step_loss_mask must be 0 or 1")
        if message.get("reasoning_content"):
            content = [{"type": "thinking", "thinking": message.pop("reasoning_content")}]
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            message["content"] = content
        message["trainable"] = message.get("step_loss_mask", 1) == 1

    specs = []
    for tool in tools:
        if tool.get("type", "function") != "function":
            raise ValueError("Only function tools are supported")
        function = tool.get("function", tool)
        specs.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )
    prefix = renderer.create_conversation_prefix_with_tools(specs)
    mode = TrainOnWhat.CUSTOMIZED if any(not m["trainable"] for m in messages) else TrainOnWhat.ALL_ASSISTANT_MESSAGES
    examples = renderer.build_supervised_examples(prefix + messages, train_on_what=mode, effort=effort)
    if len(examples) != 1:
        raise ValueError("Expected one full-conversation SFT example; refusing to concatenate renderer segments")
    model_input, weights = examples[0]
    tokens, mask = model_input.to_ints(), weights.tolist()
    if len(tokens) != len(mask) or any(weight not in (0, 1) for weight in mask):
        raise ValueError("Official renderer returned unsupported loss weights")
    if len(tokens) > max_length:
        raise ValueError(f"Conversation has {len(tokens)} tokens, exceeding {max_length}; no truncation applied")
    if not any(mask):
        raise ValueError("Conversation has no assistant target tokens")
    return tokens, [int(weight) for weight in mask]


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Load pre-rendered samples; no inference engine or tool execution is involved."""
    if evaluation:
        raise ValueError("Validation is not configured for this SFT recipe")
    samples = []
    for group in data_buffer.get_samples(args.rollout_batch_size):
        (sample,) = group
        record = sample.metadata
        tokens, mask = record["tokens"], record["loss_mask"]
        if record.get("format") != "inkling-sft-v2" or len(tokens) != len(mask) or not any(mask):
            raise ValueError("Invalid prepared Inkling SFT record; rerun data preparation")
        if len(tokens) > args.seq_length:
            raise ValueError(f"Prepared record exceeds training seq_length={args.seq_length}")
        sample.tokens = tokens
        sample.response_length = len(tokens) - mask.index(1)
        sample.loss_mask = mask[-sample.response_length :]
        sample.reward = 0
        samples.append(sample)
    return samples
