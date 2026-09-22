"""Real Miles Qwen3 TITO/codec on CPU; only the remote engine is scripted.

Run in the Linux CPU image with PROXIMAL_TEST_TOKENIZER pointing to the pinned
Qwen3 tokenizer. Missing test assets are failures, not skipped integration tests.
"""

import json
import os
from argparse import Namespace

import httpx
import pytest
from transformers import AutoTokenizer

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.types import Sample
from miles_plugins.proximal.capture_server import CaptureServer
from miles_plugins.proximal.clients import CaptureClient, IneligibleAttempt, PlatformClient
from miles_plugins.proximal.contracts import AcceptedAttempt, digest
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.rollout import execute_attempt


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(os.environ["PROXIMAL_TEST_TOKENIZER"], local_files_only=True)


def registry(tokenizer):
    return SessionRegistry(
        tokenizer,
        tito_tokenizer=get_tito_tokenizer(tokenizer, "qwen3", chat_template_kwargs={"enable_thinking": True}),
    )


def scripted_engine(config, policy, tokenizer, requests, *, tool_turn=False):
    def engine(request):
        if request.url.path == "/policies/prepare":
            return httpx.Response(
                200,
                json={
                    "snapshot": policy.snapshot.model_dump(),
                    "base_model": policy.base_model.model_dump(),
                    "request_model": f"{config.base_model.name}:miles-{policy.snapshot.sha256}",
                },
            )
        body = json.loads(request.content)
        requests.append(body)
        message = {"role": "assistant", "reasoning_content": "r", "content": "ok"}
        text = "r</think>\n\nok<|im_end|>"
        if tool_turn and len(requests) == 1:
            message = {
                "role": "assistant",
                "reasoning_content": "r",
                "content": "",  # Pinned SGLang emits an empty string on tool-only turns.
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                    }
                ],
            }
            text = 'r</think>\n\n<tool_call>\n{"name": "bash", "arguments": {"command":"pwd"}}\n</tool_call><|im_end|>'
        ids = tokenizer.encode(text, add_special_tokens=False)
        return httpx.Response(
            200,
            headers={
                "x-proximal-policy-sha256": policy.snapshot.sha256,
                "x-proximal-base-revision": policy.base_model.revision,
            },
            json={
                "id": "chat-1",
                "created": 1,
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                        "message": message,
                        "meta_info": {
                            "output_token_logprobs": [[-0.2, token, None] for token in ids],
                            "prompt_tokens": len(body["input_ids"]),
                            "completion_tokens": len(ids),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": len(body["input_ids"]),
                    "completion_tokens": len(ids),
                    "total_tokens": len(body["input_ids"]) + len(ids),
                },
            },
        )

    return engine


async def test_real_tito_seal_is_retryable_and_survives_restart(
    config, authorization, policy, attempt, tokenizer, store
):
    requests = []
    engine = scripted_engine(config, policy, tokenizer, requests)
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as backend:
        server = CaptureServer(authorization, registry=registry(tokenizer), client=backend, store=store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
            client = CaptureClient(authorization, http)
            await store.commit_policy(policy)
            handle = await client.create(attempt)
            assert await client.create(attempt) == handle
            url = handle.base_url + "/v1/chat/completions"
            headers = {"Authorization": f"Bearer {handle.api_key.get_secret_value()}"}
            messages = [{"role": "user", "content": "Inspect this feature."}]
            first = await http.post(url, headers=headers, json={"model": config.base_model.name, "messages": messages})
            assert first.status_code == 200, first.text
            messages += [first.json()["choices"][0]["message"], {"role": "user", "content": "Verify the result."}]
            second = await http.post(
                url, headers=headers, json={"model": config.base_model.name, "messages": messages}
            )
            assert second.status_code == 200, second.text
            receipt, payload = await client.collect(handle, attempt)
            assert receipt.num_calls == 2
            assert await client.collect(handle, attempt) == (receipt, payload)
            samples = decode_samples_and_merge_input_sample(payload, Sample()).samples
            assert len(samples) == 1
            sample = samples[0]
            assert 0 in sample.loss_mask and 1 in sample.loss_mask
            assert all(span.version == "1" for span in sample.all_weight_version_spans)
            assert len(sample.rollout_log_probs) == sample.response_length
            sealed = await http.post(
                url, headers=headers, json={"model": config.base_model.name, "messages": messages}
            )
            assert sealed.status_code == 409
            wrong = await http.post(url, headers={"Authorization": "Bearer capture-secret"}, json={})
            assert wrong.status_code == 401
            assert requests[1]["input_ids"][: len(requests[0]["input_ids"])] == requests[0]["input_ids"]
            await client.release(handle)
        replacement = CaptureServer(authorization, registry=registry(tokenizer), client=backend, store=store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=replacement.app)) as http:
            client = CaptureClient(authorization, http)
            assert await client.collect(handle, attempt) == (receipt, payload)
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await client.create(attempt)
            assert caught.value.response.status_code == 410


@pytest.mark.parametrize("mutation", ["identity", "logprobs", "sampling"])
async def test_bad_inference_never_seals(config, authorization, policy, attempt, tokenizer, store, mutation):
    def engine(request):
        if request.url.path == "/policies/prepare":
            return httpx.Response(
                200,
                json={
                    "snapshot": policy.snapshot.model_dump(),
                    "base_model": policy.base_model.model_dump(),
                    "request_model": f"{config.base_model.name}:miles-{policy.snapshot.sha256}",
                },
            )
        body = json.loads(request.content)
        return httpx.Response(
            200,
            headers={
                "x-proximal-policy-sha256": "wrong" if mutation == "identity" else policy.snapshot.sha256,
                "x-proximal-base-revision": policy.base_model.revision,
            },
            json={"model": body["model"], "choices": [{"meta_info": {}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as backend:
        server = CaptureServer(authorization, registry=registry(tokenizer), client=backend, store=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
        ) as http:
            client = CaptureClient(authorization, http)
            await store.commit_policy(policy)
            handle = await client.create(attempt)
            body = {"model": config.base_model.name, "messages": [{"role": "user", "content": "test"}]}
            if mutation == "sampling":
                body["temperature"] = 0.7
            reply = await http.post(
                handle.base_url + "/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {handle.api_key.get_secret_value()}"},
            )
            assert reply.status_code >= 400
            with pytest.raises(httpx.HTTPStatusError):
                await client.collect(handle, attempt)


@pytest.mark.parametrize("graded,tool_turn", [(True, False), (False, False), (True, True)])
async def test_task_to_captured_and_graded_miles_sample(
    config, authorization, policy, attempt, tokenizer, store, graded, tool_turn
):
    config_path = config.artifact_directory.parent / "run.json"
    config_path.write_text(config.model_dump_json())
    source = PlatformTaskSource(Namespace(proximal_config=str(config_path)))
    sample = source.get_samples(1)[0][0]
    methods = []
    requests = []
    engine = scripted_engine(config, policy, tokenizer, requests, tool_turn=tool_turn)
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as backend:
        server = CaptureServer(authorization, registry=registry(tokenizer), client=backend, store=store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as capture_http:
            capture = CaptureClient(authorization, capture_http)
            await store.commit_policy(policy)

            async def platform_rpc(request):
                method = request.url.path.rsplit("/", 1)[-1]
                methods.append(method)
                if method == "GetTrainingCapabilities":
                    body = {
                        "protocolVersion": 1,
                        "harnessRevision": config.harness.revision,
                        "supportedAgentTypes": [config.harness.agent_type],
                        "pinnedSource": True,
                        "linearTito": True,
                    }
                elif method == "ListProjectEnvironments":
                    body = {"memberships": [{"environmentId": attempt.task.environment_id}]}
                elif method == "CreateEnvironmentRun":
                    binding = json.loads(request.content)["trainingBinding"]
                    messages = [{"role": "user", "content": "Implement the feature."}]
                    for turn in range(2):
                        reply = await capture_http.post(
                            binding["sessionBaseUrl"] + "/chat/completions",
                            headers={"Authorization": "Bearer " + binding["sessionApiKey"]},
                            json=(
                                {
                                    "model": binding["model"],
                                    "messages": messages,
                                    "tools": [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "bash",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {"command": {"type": "string"}},
                                                    "required": ["command"],
                                                },
                                            },
                                        }
                                    ],
                                }
                                if tool_turn
                                else {"model": binding["model"], "messages": messages}
                            ),
                        )
                        assert reply.status_code == 200, reply.text
                        assistant = reply.json()["choices"][0]["message"]
                        followup = (
                            {"role": "tool", "tool_call_id": "call-1", "content": "UNIQUE_TOOL_RESULT_42"}
                            if tool_turn and turn == 0
                            else {"role": "user", "content": f"Check {turn}"}
                        )
                        messages += [assistant, followup]
                    body = {
                        "runId": attempt.attempt_id,
                        "instancesStarted": 1,
                        "trainingRequestSha256": digest(attempt),
                    }
                elif method == "GetEnvironmentRunContainers":
                    body = {
                        "runId": attempt.attempt_id,
                        "containers": [
                            {
                                "id": "rollout-1",
                                "agentType": config.harness.agent_type,
                                "status": "ROLLOUT_CONTAINER_STATUS_SUCCESS",
                                "rewardScored": graded,
                            }
                        ],
                    }
                elif method == "GetRunSummary":
                    body = {
                        "runId": attempt.attempt_id,
                        "imageId": attempt.task.image_id,
                        "sourceCommitSha": attempt.task.source_commit_sha,
                        "trainingRequestSha256": digest(attempt),
                    }
                elif method == "StopEnvironmentRun":
                    body = {}
                else:
                    raise AssertionError(method)
                return httpx.Response(200, json=body)

            async with httpx.AsyncClient(transport=httpx.MockTransport(platform_rpc)) as platform_http:
                platform = PlatformClient(authorization, platform_http)
                kwargs = {
                    "capture": capture,
                    "platform": platform,
                    "artifact_root": config.artifact_directory / "accepted",
                }
                directory = kwargs["artifact_root"] / attempt.attempt_id
                if not graded:
                    with pytest.raises(IneligibleAttempt):
                        await execute_attempt(attempt, sample, **kwargs)
                    assert not directory.exists()
                    assert "StopEnvironmentRun" in methods
                else:
                    result = await execute_attempt(attempt, sample, **kwargs)
                    assert result.index == sample.index and result.group_index == sample.group_index
                    assert result.metadata["proximal_task"] == sample.metadata["proximal_task"]
                    proof = AcceptedAttempt.model_validate_json((directory / "accepted.json").read_bytes())
                    assert proof.attempt == attempt and proof.capture.num_calls == 2
                    assert result.reward == 0.0 and sum(result.loss_mask) > 0
                    assert len(result.rollout_log_probs) == result.response_length
                    assert "StopEnvironmentRun" not in methods
                    if tool_turn:
                        assert requests[1]["messages"][-1]["role"] == "tool"
                        suffix = result.tokens[-result.response_length :]
                        context_ids = [token for token, mask in zip(suffix, result.loss_mask, strict=True) if not mask]
                        assert "UNIQUE_TOOL_RESULT_42" in tokenizer.decode(context_ids)
                # Both paths release the session; a lost attempt is never reopened.
                with pytest.raises(httpx.HTTPStatusError) as caught:
                    await capture.create(attempt)
                assert caught.value.response.status_code == 410
