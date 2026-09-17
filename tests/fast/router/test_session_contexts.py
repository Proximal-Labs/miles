"""Explicit contexts must isolate token ancestry through both protocol adapters."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.fast.router.test_session_samples_op_v2 import _ARGS, _build_core, _fresh_state, _single_turn_record

from miles.rollout.session.errors import SessionConflictError
from miles.rollout.session.sessions import setup_session_routes
from miles.rollout.session.v2.contexts import SessionContext, SessionContexts


class RecordingBackend:
    def __init__(self):
        self.requests = []
        self.headers = []

    async def do_proxy(self, request, path, *, body, headers):
        payload = json.loads(body)
        self.requests.append(payload)
        self.headers.append(headers)
        record = _single_turn_record(payload["input_ids"], [70])
        record.response["id"] = f"response-{len(self.requests)}"
        record.response["model"] = "test"
        record.response["choices"][0]["index"] = 0
        record.response["usage"] = {
            "prompt_tokens": len(payload["input_ids"]),
            "completion_tokens": 1,
            "total_tokens": len(payload["input_ids"]) + 1,
        }
        return {"status_code": 200, "headers": {}, "response_body": json.dumps(record.response).encode()}


@pytest.fixture
def context_core():
    core = _build_core(
        _ARGS.model_copy(update={"session_sample_picker_path": "miles.rollout.session.v2.picker_hub.keep_all"})
    )
    core.backend = RecordingBackend()
    return core


async def chat(core, sid, context, messages, **extra):
    return await core.chat_completions(
        sid,
        method="POST",
        query="",
        headers=context.headers(),
        body=json.dumps({"messages": messages, **extra}).encode(),
    )


async def test_identical_child_histories_cannot_attach_to_parent(context_core):
    core = context_core
    sid, state = await _fresh_state(core)
    parent = SessionContext(agent_run_id="main", context_id="main-1")
    child = SessionContext(agent_run_id="child", context_id="child-1", parent_agent_run_id="main")
    await chat(core, sid, parent, [{"role": "user", "content": "hi"}])
    parent_node = state.tree.nodes[0]
    history = parent_node.path_messages() + [{"role": "user", "content": "continue"}]
    await chat(core, sid, child, history)
    child_node = state.tree.nodes[1]
    assert child_node.parent is None
    await chat(core, sid, child, child_node.path_messages() + [{"role": "user", "content": "next"}])
    assert state.tree.nodes[2].parent is child_node
    assert len({node.generation_id for node in state.tree.nodes}) == 3
    assert all(not key.lower().startswith("x-miles-") for headers in core.backend.headers for key in headers)
    assert state.contexts.contexts["child-1"].parent_agent_run_id == "main"


async def test_compaction_keeps_execution_link_and_starts_token_root(context_core):
    core = context_core
    sid, state = await _fresh_state(core)
    first = SessionContext(agent_run_id="main", context_id="first")
    compacted = SessionContext(agent_run_id="main", context_id="compacted", derived_from_context_id="first")
    await chat(core, sid, first, [{"role": "user", "content": "hi"}])
    await chat(core, sid, compacted, state.active_messages() + [{"role": "user", "content": "next"}])
    assert len(state.tree.roots) == 2
    assert state.contexts.contexts["compacted"].derived_from_context_id == "first"


@pytest.mark.parametrize(
    "change", [{"tools": []}, {"model": "another"}, {"chat_template_kwargs": {"enable_thinking": True}}]
)
async def test_context_rejects_rendering_changes_before_inference(context_core, change):
    core = context_core
    sid, state = await _fresh_state(core)
    context = SessionContext(agent_run_id="main", context_id="first")
    await chat(core, sid, context, [{"role": "user", "content": "hi"}])
    with pytest.raises(SessionConflictError, match="rendering changed"):
        await chat(core, sid, context, state.active_messages(), **change)
    assert len(core.backend.requests) == 1


def test_child_context_can_register_before_its_parent():
    contexts = SessionContexts()
    child = SessionContext(agent_run_id="child", context_id="child", parent_agent_run_id="main")
    contexts.register(child)
    contexts.register(SessionContext(agent_run_id="main", context_id="main"))
    with pytest.raises(SessionConflictError, match="identity changed"):
        contexts.register(SessionContext(agent_run_id="other", context_id="child"))
    assert contexts.contexts["child"] == child


@pytest.mark.parametrize("register_first", [False, True])
async def test_rejected_first_request_does_not_bind_rendering(context_core, register_first):
    core = context_core
    sid, state = await _fresh_state(core)
    context = SessionContext(agent_run_id="main", context_id="first")
    if register_first:
        await core.register_context(sid, context)
    with pytest.raises(SessionConflictError, match="Previous response"):
        await core.chat_completions(
            sid,
            method="POST",
            query="",
            headers={**context.headers(), "X-Miles-Previous-Response-Id": "unknown"},
            body=b'{"model":"mistyped-model","messages":[{"role":"user","content":"hi"}]}',
        )
    assert core.backend.requests == []
    assert len(state.contexts.contexts) == int(register_first)
    await chat(core, sid, context, [{"role": "user", "content": "hi"}], model="correct-model")
    assert len(state.tree.nodes) == 1
    assert core.backend.requests[0]["model"] == "correct-model"


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_routes_preserve_identity_and_reject_invalid_predecessors(context_core, protocol):
    app = FastAPI()
    setup_session_routes(app, context_core.backend, context_core.config)
    with TestClient(app) as client:
        sid = client.post("/sessions").json()["session_id"]
        endpoint = f"/sessions/{sid}"
        context = SessionContext(agent_run_id="main", context_id="first")
        assert client.post(f"{endpoint}/contexts", json=context.model_dump()).status_code == 200
        path = "v1/chat/completions" if protocol == "openai" else "v1/messages"
        payload = {"model": "test", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
        response = client.post(f"{endpoint}/{path}", json=payload, headers=context.headers())
        assert response.status_code == 200, response.text
        metadata = client.get(endpoint).json()["metadata"]
        assert metadata["tree"]["nodes"][0]["context_id"] == "first"
        assert metadata["contexts"] == [context.model_dump(exclude_none=True)]
        invalid = {**context.headers(), "X-Miles-Previous-Response-Id": "missing"}
        assert client.post(f"{endpoint}/{path}", json=payload, headers=invalid).status_code == 409
        invalid["X-Miles-Previous-Response-Id"] = response.json()["id"]
        assert client.post(f"{endpoint}/{path}", json=payload, headers=invalid).status_code == 409
        assert len(context_core.backend.requests) == 1
        assert (
            client.post(f"{endpoint}/{path}", json=payload, headers={"X-Miles-Context-Id": "first"}).status_code == 400
        )
        records = client.get(endpoint).json()["records"]
        payload["messages"] += [records[0]["response"]["choices"][0]["message"], {"role": "user", "content": "next"}]
        foreign = {
            **SessionContext(agent_run_id="other", context_id="other").headers(),
            "X-Miles-Previous-Response-Id": response.json()["id"],
        }
        assert client.post(f"{endpoint}/{path}", json=payload, headers=foreign).status_code == 409
        assert client.post(f"{endpoint}/{path}", json=payload, headers=invalid).status_code == 200
        assert client.get(endpoint).json()["metadata"]["tree"]["nodes"][1]["parent"] == 0
