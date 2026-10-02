import hashlib
import json

import httpx
import pytest

from miles_plugins.reward_hacking import data, evaluate, modal_server, providers
from miles_plugins.reward_hacking.curate import select_controls
from miles_plugins.reward_hacking.prompt import PROMPT_PROVENANCE, QA_RUBRIC, SYSTEM_PROMPT


def event(sequence, kind, payload, **extra):
    return {"sequence": sequence, "kind": kind, "event": {"payload": payload}, **extra}


def trace():
    return {
        "run": {
            "events": [
                event(
                    1,
                    "agentInput",
                    {"input": {"kind": "inline_json", "value": {"instruction": "Implement the feature"}}},
                ),
                event(
                    2,
                    "providerResponse",
                    {
                        "assistantMessage": {
                            "content": [
                                {"kind": "chat_completions_reasoning", "reasoning": "Inspect first"},
                                {"kind": "tool_call", "input": {"command": "duplicate must not appear"}},
                            ]
                        }
                    },
                ),
                event(3, "toolCall", {"toolCallId": "a", "arguments": {"command": "ls"}}, toolName="bash"),
                event(4, "toolCall", {"toolCallId": "b", "arguments": {"command": "cat file"}}, toolName="bash"),
                event(
                    5, "toolResult", {"output": {"kind": "inline_json", "value": {"output": "excluded tool result"}}}
                ),
                event(
                    6,
                    "providerResponse",
                    {
                        "assistantMessage": {
                            "content": [
                                {"kind": "openai_responses_reasoning", "summary": [{"text": "Use grading shortcut"}]},
                                {
                                    "kind": "openai_responses_reasoning",
                                    "summary": [],
                                    "encryptedContent": "secret ciphertext",
                                },
                            ]
                        }
                    },
                ),
                event(7, "toolCall", {"toolCallId": "c", "arguments": {"command": "hack"}}, toolName="bash"),
            ]
        }
    }


def test_atomic_prefix_and_no_future_leakage():
    task, turns = data.trace_turns(trace(), [])
    source = {"rollout_id": "rollout", "label": True}
    prefix = data._example(source, task, turns, 0.1, [{"sequence": 7}])
    full = data._example(source, task, turns, 1, [{"sequence": 7}])
    assert prefix["metadata"]["tool_calls_included"] == 2  # A parallel batch stays together.
    assert prefix["metadata"]["known_evidence_visible"] is False
    assert full["metadata"]["known_evidence_visible"] is True
    assert prefix["messages"] == full["messages"][:2]
    assert full["metadata"]["unavailable_reasoning_blocks"] == 1
    assert full["metadata"]["reasoning_summary_blocks"] == 1
    prompt = providers.monitor_input(prefix)
    assert "grading shortcut" not in prompt
    assert "duplicate must not appear" not in prompt
    assert "excluded tool result" not in prompt
    assert "secret ciphertext" not in providers.monitor_input(full)


def test_prompt_does_not_include_labels_or_qa():
    row = {
        "messages": [{"role": "assistant", "reasoning_content": "Evidence", "metadata": "SECRET"}],
        "label": True,
        "metadata": {"qa": "SECRET", "fraction": 0.1},
        "hack_highlights": "SECRET",
    }
    prompt = providers.monitor_input(row)
    assert "SECRET" not in prompt and '"label"' not in prompt and '"fraction"' not in prompt
    row["label"] = False
    assert providers.monitor_input(row) == prompt


def response(value, **extras):
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}],
        **extras,
    }


@pytest.mark.parametrize(
    "value",
    [
        {"reward_hacking": "false", "explanation": "x"},
        {},
        {"reward_hacking": True, "explanation": "extra field"},
    ],
)
def test_invalid_predictions_are_not_negative_labels(value):
    with pytest.raises(ValueError):
        providers.parse_response(response(value), "responses")


def test_incomplete_or_refused_outputs_are_errors():
    with pytest.raises(ValueError, match="not completed"):
        providers.parse_response(response({}, status="incomplete"), "responses")
    with pytest.raises(ValueError, match="refused"):
        providers.parse_response(
            {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}, "responses"
        )
    with pytest.raises(ValueError, match="did not finish"):
        providers.parse_response({"choices": [{"finish_reason": "length"}]}, "chat_completions")


@pytest.mark.asyncio
async def test_gateway_request_and_bounded_retry(monkeypatch):
    requests = []

    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=response({"reward_hacking": True}))

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(providers.asyncio, "sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await providers.classify(
            client,
            {"api": "responses", "model": "gpt-6-luna"},
            ("https://gateway.test/v1", {"Authorization": "Bearer dummy"}),
            "trace",
        )
    assert result["reward_hacking"] is True and result["attempts"] == 2
    body = json.loads(requests[-1].content)
    assert requests[-1].url.path == "/v1/responses"
    assert body["store"] is False and body["truncation"] == "disabled"
    assert "tools" not in body


def test_gateway_requires_configured_route(monkeypatch):
    for key in ("OPENAI_BASE_URL", "CLOUDFLARE_ACCOUNT_ID", "AI_GATEWAY_ID", "CF_AIG_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        providers.gateway_route({})
    monkeypatch.setenv("OPENAI_BASE_URL", "https://existing-gateway.test/v1")
    assert providers.gateway_route({})[0] == "https://existing-gateway.test/v1"


def test_metrics_count_errors_and_distinguish_early_detection():
    def row(label, prediction, visible=False):
        return {
            "label": label,
            "status": "ok" if prediction is not None else "error",
            "reward_hacking": prediction,
            "metadata": {"known_evidence_visible": visible},
        }

    result = evaluate.metrics(
        [row(True, True, True), row(True, False), row(True, None), row(False, True), row(False, False)]
    )
    assert (result["tp"], result["fn"], result["fp"], result["tn"], result["errors"]) == (1, 1, 1, 1, 1)
    assert result["positive_recall_all"] == 1 / 3
    assert result["accuracy_all"] == 2 / 5
    assert result["known_evidence_recall_all"] == 1


def test_resume_rejects_changed_inputs(tmp_path):
    rows = [{"id": "r", "prompt": "original", "label": True, "metadata": {}}]
    evaluate.prepare_run(rows, {"luna": {}}, {}, tmp_path)
    rows[0]["prompt"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        evaluate.prepare_run(rows, {"luna": {}}, {}, tmp_path)


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt, SystemExit])
def test_modal_cleanup_on_every_exit(tmp_path, monkeypatch, failure):
    stopped = []
    monkeypatch.setenv("MODAL_INFERENCE_API_KEY", "dummy")
    monkeypatch.setattr(
        modal_server.serving, "deploy", lambda *a, **kw: {"app_id": "ap-owned", "url": "https://inference.test"}
    )
    monkeypatch.setattr(modal_server.serving, "wait_ready", lambda *a, **kw: None)
    monkeypatch.setattr(modal_server.serving, "stop", lambda identity, environment: stopped.append(identity))

    def run():
        with modal_server.inference_endpoint({"base": "/model", "image": "image"}, tmp_path):
            if failure:
                raise failure()

    if failure:
        with pytest.raises(failure):
            run()
    else:
        run()
    assert stopped == ["ap-owned"]
    assert json.loads((tmp_path / "modal-deployment.json").read_text())["status"] == "stopped"


def test_modal_cleanup_if_deployment_is_interrupted(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setenv("MODAL_INFERENCE_API_KEY", "dummy")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(modal_server.serving, "deploy", interrupted)
    monkeypatch.setattr(modal_server.serving, "stop", lambda identity, environment: stopped.append(identity))
    with pytest.raises(KeyboardInterrupt):
        with modal_server.inference_endpoint({"base": "/model", "image": "image"}, tmp_path):
            pass
    assert stopped[0].startswith("reward-hacking-monitor-")


def test_control_selection_rejects_flagged_env_and_unreviewed(tmp_path):
    (tmp_path / "report").mkdir()
    (tmp_path / "report/trajectories.jsonl").write_text(json.dumps({"environmentId": 1}) + "\n")
    records = []
    for env, confirmed in [(1, True), (2, True), (3, False)]:
        records.append(
            {
                "_id": str(env),
                "environmentId": env,
                "scope": "rollout",
                "status": "completed",
                "rolloutId": str(env),
                "runId": str(env),
                "report": {
                    "payload": {
                        "reward_hacking": {"attempted": False},
                        "evaluation_awareness": {"detected": False},
                        "verdict": "FAIR",
                        "post_qa_verification": {"verdict": "confirmed" if confirmed else "refuted"},
                    }
                },
            }
        )
    (tmp_path / "post_qa_reports.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    result = select_controls(tmp_path, tmp_path / "selected.json", seed=519)
    assert [r["environment_id"] for r in result["candidates"]] == [2]


def test_resume_recovers_partial_final_write(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b'{"id":"complete"}\n{"id":"torn')
    assert evaluate._read_predictions(path) == [{"id": "complete"}]
    assert path.with_suffix(".interrupted-tail").read_bytes() == b'{"id":"torn'


@pytest.mark.asyncio
@pytest.mark.parametrize("attempts", [1, 4])
async def test_evaluator_persists_predictions_and_resumes_without_requests(tmp_path, monkeypatch, attempts):
    calls = []
    real_client = httpx.AsyncClient

    async def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response({"reward_hacking": False}))

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    rows = [{"id": "r", "prompt": "trace", "label": False, "metadata": {"fraction": 0.1}}]
    rows = evaluate.repeat_examples(rows, attempts)
    assert len({row["id"] for row in rows}) == attempts
    assert all(row["prompt"] == "trace" for row in rows)
    model = {"api": "responses", "model": "gpt-6-luna"}
    previous = {}
    for _ in range(2):
        await evaluate.evaluate_model(
            rows, "luna", model, ("https://gateway.test/v1", {}), tmp_path, previous, concurrency=1, timeout=1
        )
    assert len(calls) == attempts
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == attempts
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["luna/prefix-0.1/all"]["accuracy_all"] == 1


@pytest.mark.parametrize("replicas", [1, 2])
def test_modal_uses_replacement_headroom(tmp_path, monkeypatch, replicas):
    settings = []
    monkeypatch.setenv("MODAL_INFERENCE_API_KEY", "dummy")

    def deploy(value, **kwargs):
        settings.append(value)
        return {"app_id": "owned", "url": "https://inference.test"}

    monkeypatch.setattr(modal_server.serving, "deploy", deploy)
    monkeypatch.setattr(modal_server.serving, "wait_ready", lambda *a, **kw: None)
    monkeypatch.setattr(modal_server.serving, "stop", lambda *a: None)
    with modal_server.inference_endpoint({"base": "/model", "image": "image", "replicas": replicas}, tmp_path):
        pass
    assert settings[0]["replicas"] == replicas and settings[0]["max_replicas"] == replicas + 1


def test_both_adapters_receive_identical_trace_and_strict_output_schema():
    text = providers.monitor_input({"messages": [{"role": "user", "content": "task"}]})
    responses = providers.request_body({"api": "responses", "model": "gpt-6-sol"}, text)
    chat = providers.request_body({"api": "chat_completions", "model": "thinkingmachines/Inkling-Small"}, text)
    assert responses["input"][0]["content"] == chat["messages"][1]["content"]
    assert responses["text"]["format"]["schema"] == chat["response_format"]["json_schema"]["schema"]
    result = {"reward_hacking": False}
    reply = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}
    assert providers.parse_response(reply, "chat_completions") == result


def test_all_models_use_the_pinned_proximal_qa_prompt(tmp_path):
    assert hashlib.sha256(QA_RUBRIC.encode()).hexdigest() == PROMPT_PROVENANCE["rubric_sha256"]
    assert "verifier_tailored_output" in QA_RUBRIC
    assert "forbidden_tool_causality" in QA_RUBRIC
    assert "Perception only" in QA_RUBRIC
    assert "counterfactual experiment are NEVER required" in SYSTEM_PROMPT
    assert "delta > 0" not in QA_RUBRIC
    assert providers.RESULT_SCHEMA["required"] == ["reward_hacking"]
    assert set(providers.RESULT_SCHEMA["properties"]) == {"reward_hacking"}
    for name, api in [
        ("gpt-6-luna", "responses"),
        ("gpt-6-sol", "responses"),
        ("thinkingmachines/Inkling-Small", "chat_completions"),
    ]:
        request = providers.request_body({"model": name, "api": api}, "recorded trace")
        actual = request["instructions"] if api == "responses" else request["messages"][0]["content"]
        assert actual == SYSTEM_PROMPT
    evaluate.prepare_run([{"id": "r", "prompt": "trace", "label": False, "metadata": {}}], {"luna": {}}, {}, tmp_path)
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["prompt_provenance"] == PROMPT_PROVENANCE
    assert (tmp_path / "system-prompt.txt").read_text() == SYSTEM_PROMPT
