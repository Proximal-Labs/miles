"""V2 finalization retains parallel work and makes failed collection recoverable."""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.fast.router.test_session_samples_op_v2 import (
    _ARGS,
    _build_core,
    _fabricate_node,
    _fresh_state,
    _single_turn_record,
)

from miles.rollout.session.errors import SessionConflictError, SessionNotFoundError
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.sessions import setup_session_routes
from miles.rollout.session.v2 import core as core_module
from miles.rollout.session.v2 import session_state
from miles.rollout.session.v2.postprocessor_hub import default_postprocess
from miles.utils.types import Sample


class _ControlledBackend:
    def __init__(self):
        self.started = asyncio.Queue()

    async def do_proxy(self, request, path, *, body, headers):
        released = asyncio.get_running_loop().create_future()
        await self.started.put(released)
        await released
        prompt_ids = json.loads(body)["input_ids"]
        record = _single_turn_record(prompt_ids, [70])
        return {"status_code": 200, "headers": {}, "response_body": json.dumps(record.response).encode()}


@pytest.fixture
def core():
    return _build_core(
        _ARGS.model_copy(update={"session_sample_picker_path": "miles.rollout.session.v2.picker_hub.keep_all"})
    )


def _chat(core, sid):
    return asyncio.create_task(
        core.chat_completions(
            sid, method="POST", query="", headers={}, body=b'{"messages":[{"role":"user","content":"hi"}]}'
        )
    )


def _decode(response):
    assert response.status_code == 200
    return decode_samples_and_merge_input_sample(response.body, Sample(), fields=COMPUTED_FIELDS_V2)


async def test_parallel_siblings_keep_both_outputs_and_train_shared_prefix_once(core):
    sid, state = await _fresh_state(core)
    root = _fabricate_node(state, None, _single_turn_record([1, 2], [10]), [1, 2, 10], completion_span=(2, 3))
    for observation, output in [(20, 30), (21, 31)]:
        state.active_leaf = _fabricate_node(
            state,
            root,
            _single_turn_record([1, 2, 10, observation], [output]),
            [1, 2, 10, observation, output],
            completion_span=(4, 5),
        )
    await core.finish_session(sid, producer_finished=True, timeout=0)
    reply = _decode(await core.collect_samples(sid, max_seq_len=None, agent_metadata={"reward": 1.0}))
    assert [sample.tokens[-1] for sample in reply.samples] == [30, 31]
    assert [sample.loss_mask for sample in reply.samples] == [[1, 0, 1], [0, 0, 1]]
    assert [sample.reward for sample in reply.samples] == [1.0, 1.0]


@pytest.mark.parametrize("completion_order", [(0, 1), (1, 0)], ids=["forward", "reverse"])
async def test_finish_waits_for_admitted_calls_and_freezes_export(core, completion_order):
    sid, state = await _fresh_state(core)
    backend = core.backend = _ControlledBackend()
    chats = []
    releases = []
    for _ in range(2):
        chats.append(_chat(core, sid))
        releases.append(await backend.started.get())
    finish = asyncio.create_task(core.finish_session(sid, producer_finished=True, timeout=60))
    await asyncio.sleep(0)
    with pytest.raises(SessionConflictError):
        await _chat(core, sid)
    with pytest.raises(SessionConflictError, match="draining"):
        await core.collect_samples(sid, max_seq_len=None)
    for index in completion_order:
        releases[index].set_result(None)
        assert (await chats[index]).status_code == 200
    finished = json.loads((await finish).body)
    assert finished["complete"]
    assert len(state.tree.nodes) == 2
    response = await core.collect_samples(sid, max_seq_len=None, snapshot_id=finished["snapshot_id"])
    assert len(_decode(response).samples) == 2
    core.sample_picker = lambda *_: pytest.fail("sealed export must not rerun hooks")
    assert (await core.collect_samples(sid, max_seq_len=None)).body == response.body
    with pytest.raises(SessionConflictError, match="parameters changed"):
        await core.collect_samples(sid, max_seq_len=1)
    with pytest.raises(SessionConflictError, match="Snapshot"):
        await core.collect_samples(sid, max_seq_len=None, snapshot_id="wrong")
    core.registry.remove_session(sid)


async def test_deadline_fences_late_backend_output_and_exports_no_training_rows(core):
    sid, state = await _fresh_state(core)
    backend = core.backend = _ControlledBackend()
    chat = _chat(core, sid)
    release = await backend.started.get()
    finished = json.loads((await core.finish_session(sid, producer_finished=True, timeout=0)).body)
    assert not finished["complete"]
    assert finished["unfinished_requests"] == 1
    reply = _decode(await core.collect_samples(sid, max_seq_len=None))
    assert reply.samples == [] and reply.empty_reason == "incomplete"
    release.set_result(None)
    await chat
    assert state.tree.nodes == []
    assert json.loads((await core.finish_session(sid, producer_finished=True, timeout=0)).body) == finished
    core.registry.remove_session(sid)


async def test_cancelled_generation_marks_trace_incomplete(core):
    sid, _ = await _fresh_state(core)
    backend = core.backend = _ControlledBackend()
    chat = _chat(core, sid)
    await backend.started.get()
    chat.cancel()
    with pytest.raises(asyncio.CancelledError):
        await chat
    finished = json.loads((await core.finish_session(sid, producer_finished=True, timeout=0)).body)
    assert not finished["complete"]
    assert finished["failed_requests"] == 1
    core.registry.remove_session(sid)


@pytest.mark.parametrize("status_code,complete", [(400, True), (502, False)])
async def test_upstream_rejection_and_lost_output_have_distinct_outcomes(core, status_code, complete):
    class _ErrorBackend:
        async def do_proxy(self, *args, **kwargs):
            return {"status_code": status_code, "headers": {}, "response_body": b'{"error":"upstream failed"}'}

    sid, _ = await _fresh_state(core)
    core.backend = _ErrorBackend()
    assert (await _chat(core, sid)).status_code == status_code
    finished = json.loads((await core.finish_session(sid, producer_finished=True, timeout=0)).body)
    assert finished["complete"] is complete
    assert finished["failed_requests"] == int(not complete)
    core.registry.remove_session(sid)


async def test_capacity_is_reserved_before_inference(core, monkeypatch):
    monkeypatch.setattr(core_module, "MAX_NODES", 1)
    sid, state = await _fresh_state(core)
    backend = core.backend = _ControlledBackend()
    chat = _chat(core, sid)
    release = await backend.started.get()
    with pytest.raises(SessionConflictError, match="capacity"):
        await _chat(core, sid)
    assert backend.started.empty()
    release.set_result(None)
    await chat
    assert len(state.tree.nodes) == 1


async def test_failed_assembly_can_be_retried_with_corrected_hook(core):
    sid, state = await _fresh_state(core)
    state.active_leaf = _fabricate_node(
        state, None, _single_turn_record([1, 2], [10]), [1, 2, 10], completion_span=(2, 3)
    )
    await core.finish_session(sid, producer_finished=True, timeout=0)
    picker = core.sample_picker

    def fail(*_):
        raise ValueError("invalid selection")

    core.sample_picker = fail
    assert (await core.collect_samples(sid, max_seq_len=None)).status_code == 422
    assert state.sample_export is None
    core.sample_picker = picker
    assert len(_decode(await core.collect_samples(sid, max_seq_len=None)).samples) == 1
    core.registry.remove_session(sid)


async def test_abandoned_finished_session_expires(core, monkeypatch):
    monkeypatch.setattr(session_state, "FINISHED_SESSION_RETENTION_SECONDS", 0)
    sid, _ = await _fresh_state(core)
    await core.finish_session(sid, producer_finished=True, timeout=0)
    await asyncio.sleep(0)
    with pytest.raises(SessionNotFoundError):
        core.registry.get_session(sid)


def test_finish_route_validates_input_and_sealed_snapshot(core):
    app = FastAPI()
    setup_session_routes(app, _ControlledBackend(), core.config)
    with TestClient(app) as client:
        sid = client.post("/sessions").json()["session_id"]
        endpoint = f"/sessions/{sid}"
        assert client.post(f"{endpoint}/finish", json={"producer_finished": True, "timeout": -1}).status_code == 422
        finished = client.post(f"{endpoint}/finish", json={"producer_finished": True, "timeout": 0})
        assert finished.status_code == 200 and finished.json()["complete"]
        assert client.post(f"{endpoint}/v1/chat/completions", json={"messages": []}).status_code == 409
        reply = client.post(f"{endpoint}/samples", json={"snapshot_id": finished.json()["snapshot_id"]})
        assert reply.status_code == 200
        assert client.delete(endpoint).status_code == 204


def test_capped_occurrence_cannot_own_tokens_it_does_not_contain():
    metadata = {"tree": {"nodes": [{"id": 0, "completion_span": [1, 3]}, {"id": 1, "completion_span": [3, 4]}]}}
    short = Sample(
        tokens=[0, 10], response_length=1, loss_mask=[1], metadata={"leaf": {"node_id": 0, "path_node_ids": [0]}}
    )
    full = Sample(
        tokens=[0, 10, 11, 20],
        response_length=3,
        loss_mask=[1, 1, 1],
        metadata={"leaf": {"node_id": 1, "path_node_ids": [0, 1]}},
    )
    samples = default_postprocess([full, short], metadata)
    assert samples == [full, short]
    assert short.loss_mask == [1]
    assert full.loss_mask == [0, 1, 1]
