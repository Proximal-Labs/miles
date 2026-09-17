"""The instrumented harness must preserve context, finalization, and episode weight."""

from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from examples.experimental.session_multi_agent.agent import run as run_example
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.generate_hub.test_agentic_v2 import _generate_input

from miles.ray.rollout.train_data_conversion import _compute_rollout_mask_sums, _normalize_rewards_by_rollout
from miles.rollout.agentic.harness import AgentResult
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.server import SessionServer
from miles.utils import http_utils
from miles.utils.chat_template_utils import resolve_fixed_chat_template
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample


@contextmanager
def _harness_server(monkeypatch):
    with with_mock_server(model_name="Qwen/Qwen3-0.6B", process_fn=lambda _: ProcessResult("answer")) as backend:
        original = backend._compute_chat_completions_response

        def stopped_response(payload):
            response = original(payload)
            # real inference returns the stop token in token IDs but trims it from text
            choice = response["choices"][0]
            choice["meta_info"]["output_token_logprobs"].append((-0.1, backend.tokenizer.eos_token_id))
            choice["logprobs"]["content"].append(
                {
                    "token": backend.tokenizer.eos_token,
                    "token_id": backend.tokenizer.eos_token_id,
                    "logprob": -0.1,
                }
            )
            choice["meta_info"]["completion_tokens"] += 1
            response["usage"]["completion_tokens"] += 1
            response["usage"]["total_tokens"] += 1
            return response

        monkeypatch.setattr(backend, "_compute_chat_completions_response", stopped_response)
        config = make_session_server_config(
            backend_url=backend.url,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            tito_model="qwen3",
            chat_template_path=resolve_fixed_chat_template("qwen3")[0],
            apply_chat_template_kwargs={"enable_thinking": False},
            use_session_server="v2",
            session_sample_picker_path="miles.rollout.session.v2.picker_hub.keep_all",
            session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        )
        port = find_available_port(31000)
        server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.stop()


async def test_example_through_live_http_exports_one_episode_and_releases(monkeypatch):
    snapshots = []
    release = OpenAIEndpointTracer._release

    async def run_scored_example(**kwargs):
        outcome = await run_example(**kwargs)
        return AgentResult(metadata={**outcome.metadata, "reward": 0.75}, producer_finished=outcome.producer_finished)

    async def inspect_then_release(tracer):
        response = await http_utils._http_client.get(tracer.base_url)
        snapshots.append(response.json())
        await release(tracer)
        assert (await http_utils._http_client.get(tracer.base_url)).status_code == 404

    monkeypatch.setattr(agentic_tool_call, "load_function", lambda _: run_scored_example)
    monkeypatch.setattr(OpenAIEndpointTracer, "_release", inspect_then_release)
    with _harness_server(monkeypatch) as session_url:
        async with httpx.AsyncClient(timeout=30) as client:
            monkeypatch.setattr(http_utils, "_http_client", client)
            generation = _generate_input()
            generation.args.session_server_addrs = [session_url.removeprefix("http://")]
            generation.sample.index = 0
            generation.sample.group_index = 0
            output = await agentic_tool_call.generate(generation)

    metadata = snapshots[0]["metadata"]
    assert metadata["finalization"]["complete"]
    assert len(metadata["tree"]["nodes"]) == 4
    assert [node["parent"] for node in metadata["tree"]["nodes"]] == [None, None, None, 0]
    assert len(metadata["contexts"]) == 3
    assert {
        context.get("parent_agent_run_id") for context in metadata["contexts"] if context["agent_run_id"] != "main"
    } == {"main"}
    samples = output.samples
    assert len(samples) == 3
    assert all(sample.rollout_id == 0 and sample.reward == 0.75 for sample in samples)
    assert {sample.metadata["leaf"]["agent_run_id"] for sample in samples} == {"main", "review-a", "review-b"}
    assert all(sample.metadata.get("tito_session_mismatch") == [] for sample in samples), [
        sample.metadata.get("tito_session_mismatch") for sample in samples
    ]
    assert all(len(sample.loss_mask) == sample.response_length for sample in samples)

    samples.append(Sample(index=1, rollout_id=1, group_index=0, reward=0.25, loss_mask=[1]))
    args = SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=False)
    normalized = _normalize_rewards_by_rollout(args, samples, [sample.reward for sample in samples], None)
    assert normalized == pytest.approx([0.25, 0.25, 0.25, -0.25])
    mask_totals = _compute_rollout_mask_sums(
        [sample.rollout_id for sample in samples], [sample.loss_mask for sample in samples]
    )
    assert mask_totals[:3] == [sum(sum(sample.loss_mask) for sample in samples[:3])] * 3
    assert mask_totals[-1] == 1
