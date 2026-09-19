import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.ray.rollout import metrics
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.checkpoint_eval import retarget_args
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils.function_registry import function_registry
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.types import Sample


def _input(**overrides):
    args = SimpleNamespace(
        custom_agent_function_path="test.eval_agent",
        sglang_router_ip="rollout-router",
        sglang_router_port=30000,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        apply_chat_template_kwargs={"enable_thinking": False},
        partial_rollout=False,
        group_rm=False,
        use_session_server=overrides.pop("use_session_server", "v2"),
        max_seq_len=1,
        sglang_speculative_algorithm=None,
        **overrides,
    )
    state = SimpleNamespace(
        args=args,
        generate_fn_semaphore=asyncio.Semaphore(2),
        aborted=False,
        generate_function=agentic_tool_call.generate,
    )
    return GenerateFnInput(
        state=state,
        sample=Sample(index=7, prompt=[{"role": "user", "content": "hello"}], metadata={"task": "test"}),
        sampling_params={"temperature": 0.0, "top_p": 0.8, "top_k": -1, "max_new_tokens": 8, "no_stop_trim": True},
        evaluation=True,
    )


@pytest.fixture(autouse=True)
def reject_session_and_reward_calls(monkeypatch):
    monkeypatch.setattr(
        agentic_tool_call.OpenAIEndpointTracer, "create", AsyncMock(side_effect=AssertionError("unexpected session"))
    )
    monkeypatch.setattr(
        inference_rollout_common, "async_rm", AsyncMock(side_effect=AssertionError("unexpected reward model"))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session_server", [False, True, "v2"])
@pytest.mark.parametrize("reward", [0.0, 1.0, {"accuracy": 1.0}])
async def test_eval_uses_retargeted_router_and_agent_verdict(use_session_server, reward):
    input = _input(use_session_server=use_session_server, lora_rank=8)
    input.state.args = retarget_args(input.args, "eval-router", 31000, 1, 1)
    original_sample = deepcopy(input.sample)
    original_sampling_params = deepcopy(input.sampling_params)
    report = {"reward": reward, "eval_report": {"passed": True}, "agent_metrics": {"turns": 3}}
    calls = []

    async def agent(**kwargs):
        calls.append(kwargs)
        kwargs["metadata"]["agent_note"] = "done"
        return report

    with function_registry.temporary("test.eval_agent", agent):
        sample = await inference_rollout_common.generate_and_rm(
            input.state, input.sample, input.sampling_params, evaluation=True
        )

    assert isinstance(sample, Sample)
    assert sample.index == 7
    assert sample.reward == reward
    assert sample.status == Sample.Status.COMPLETED
    assert sample.metadata == {"task": "test", "agent_note": "done", **report}
    assert sample.tokens == []
    assert sample.response == ""
    assert sample.loss_mask is sample.rollout_log_probs is sample.rollout_routed_experts is None
    assert input.sample == original_sample
    assert input.sampling_params == original_sampling_params
    assert len(calls) == 1
    assert calls[0]["base_url"] == "http://eval-router:31000"
    assert calls[0]["request_kwargs"] == {
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": -1,
        "max_tokens": 8,
        "no_stop_trim": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "lora_path": LORA_ADAPTER_NAME,
    }
    assert "max_seq_len" not in calls[0]["metadata"]
    assert "session_server_id" not in calls[0]["metadata"]


@pytest.mark.asyncio
async def test_eval_request_template_args_override_launch_defaults():
    input = _input()
    input.sampling_params["chat_template_kwargs"] = {"enable_thinking": True}

    async def agent(**kwargs):
        assert kwargs["request_kwargs"]["chat_template_kwargs"] == {"enable_thinking": True}
        assert "lora_path" not in kwargs["request_kwargs"]
        return {"reward": 1}

    with function_registry.temporary("test.eval_agent", agent):
        await agentic_tool_call.generate(input)
    assert input.args.apply_chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_metadata", [None, {"eval_report": {"passed": True}}])
async def test_eval_without_reward_uses_existing_reward_function(monkeypatch, agent_metadata):
    input = _input()
    reward_function = AsyncMock(return_value=1.0)
    monkeypatch.setattr(inference_rollout_common, "async_rm", reward_function)

    async def agent(**kwargs):
        return agent_metadata

    with function_registry.temporary("test.eval_agent", agent):
        sample = await inference_rollout_common.generate_and_rm(
            input.state, input.sample, input.sampling_params, evaluation=True
        )

    reward_function.assert_awaited_once_with(input.args, sample)
    assert sample.metadata == {**input.sample.metadata, **(agent_metadata or {})}
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_eval_errors_propagate(error):
    async def agent(**kwargs):
        raise error

    with function_registry.temporary("test.eval_agent", agent), pytest.raises(error):
        await agentic_tool_call.generate(_input())


def test_eval_logs_rewards_without_training_metrics(monkeypatch):
    args = SimpleNamespace(
        reward_key="train_score",
        custom_eval_rollout_log_function_path=None,
        log_passrate=True,
        n_samples_per_eval_prompt=2,
    )
    rewards = [0.0, 1.0]
    samples = [Sample(reward={"accuracy": reward}) for reward in rewards]
    monkeypatch.setattr(metrics, "compute_rollout_step", lambda *_: 0)
    monkeypatch.setattr(metrics.tracking, "log", lambda *_args, **_kwargs: None)
    logged = metrics.log_eval_rollout_data(0, args, {"agent": {"rewards": rewards, "samples": samples}})
    assert logged["eval/agent"] == 0.5
    assert logged["eval/agent-pass@2"] == 1.0
    assert not any("response_len" in key or "num_training_samples" in key for key in logged)
