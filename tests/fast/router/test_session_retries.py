"""Transport replay is one operation; sampled retries remain explicit generations."""

import asyncio
import json

import pytest
from tests.fast.router.test_session_contexts import RecordingBackend
from tests.fast.router.test_session_samples_op_v2 import _ARGS, _build_core, _fresh_state

from miles.rollout.session.errors import SessionConflictError
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.v2.contexts import SessionContext
from miles.rollout.session.v2.postprocessor_hub import default_postprocess, exclude_superseded
from miles.utils.types import Sample

_BODY = b'{"messages":[{"role":"user","content":"hi"}]}'


@pytest.fixture
def context_core():
    core = _build_core(
        _ARGS.model_copy(update={"session_sample_picker_path": "miles.rollout.session.v2.picker_hub.keep_all"})
    )
    core.backend = RecordingBackend()
    return core


class ControlledBackend(RecordingBackend):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Queue()

    async def do_proxy(self, *args, **kwargs):
        release = asyncio.get_running_loop().create_future()
        await self.started.put(release)
        await release
        return await super().do_proxy(*args, **kwargs)


def request(core, sid, *, key=None, body=_BODY, **headers):
    if key is not None:
        headers["X-Miles-Idempotency-Key"] = key
    return asyncio.create_task(core.chat_completions(sid, method="POST", query="", headers=headers, body=body))


async def test_duplicate_delivery_survives_disconnect_and_replays_after_finish(context_core):
    core = context_core
    sid, state = await _fresh_state(core)
    backend = core.backend = ControlledBackend()
    first = request(core, sid, key="one")
    release = await backend.started.get()
    duplicate = request(core, sid, key="one")
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    finish = asyncio.create_task(core.finish_session(sid, producer_finished=True, timeout=60))
    await asyncio.sleep(0)
    assert state.lifecycle.pending_count == 1 and not finish.done()
    release.set_result(None)
    response = await duplicate
    assert json.loads((await finish).body)["complete"]
    replay = await request(core, sid, key="one")
    assert replay.body == response.body and replay.headers == response.headers
    assert len(backend.requests) == len(state.tree.nodes) == 1
    assert response.headers["x-miles-generation-id"] == state.tree.nodes[0].generation_id
    assert all(not name.lower().startswith("x-miles-") for name in backend.headers[0])
    with pytest.raises(SessionConflictError, match="different inputs"):
        await request(core, sid, key="one", body=b'{"messages":[],"temperature":1}')
    with pytest.raises(SessionConflictError, match="finishing"):
        await request(core, sid, key="two")
    core.registry.remove_session(sid)


async def test_failed_operation_is_replayed_without_resampling(context_core):
    class FailingBackend:
        calls = 0

        async def do_proxy(self, *args, **kwargs):
            self.calls += 1
            raise ConnectionError("lost backend response")

    core = context_core
    sid, _ = await _fresh_state(core)
    backend = core.backend = FailingBackend()
    for _ in range(2):
        with pytest.raises(ConnectionError):
            await request(core, sid, key="one")
    finished = json.loads((await core.finish_session(sid, producer_finished=True, timeout=0)).body)
    assert backend.calls == finished["failed_requests"] == 1
    assert not finished["complete"]
    core.registry.remove_session(sid)


async def test_delete_cancels_session_owned_operation(context_core):
    core = context_core
    sid, state = await _fresh_state(core)
    backend = core.backend = ControlledBackend()
    pending = request(core, sid, key="one")
    await backend.started.get()
    await core.delete_session(sid)
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert state.lifecycle.pending_count == 0
    assert state.tree.nodes == []


@pytest.mark.parametrize("key", [None, "fresh-key"])
async def test_identical_prompts_with_fresh_intent_are_distinct_generations(context_core, key):
    core = context_core
    sid, state = await _fresh_state(core)
    first = await request(core, sid, key="first")
    second = await request(core, sid, key=key, **{"X-Miles-Retry-Of": first.headers["x-miles-generation-id"]})
    assert len(core.backend.requests) == 2
    assert first.headers["x-miles-generation-id"] != second.headers["x-miles-generation-id"]
    assert state.tree.nodes[1].retry_of == state.tree.nodes[0].generation_id
    reply = decode_samples_and_merge_input_sample(
        (await core.collect_samples(sid, max_seq_len=None)).body, Sample(), fields=COMPUTED_FIELDS_V2
    )
    assert len(reply.samples) == 2
    assert sum(sum(sample.loss_mask) for sample in reply.samples) == 2


async def test_superseded_ancestor_remains_context_for_trainable_descendant(context_core):
    core = context_core
    sid, state = await _fresh_state(core)
    first = await request(core, sid)
    ancestor = state.active_leaf
    await request(
        core,
        sid,
        body=json.dumps({"messages": ancestor.path_messages() + [{"role": "user", "content": "next"}]}).encode(),
    )
    await request(core, sid, **{"X-Miles-Supersedes": first.headers["x-miles-generation-id"]})
    default = await core.collect_samples(sid, max_seq_len=None)
    decoded = decode_samples_and_merge_input_sample(default.body, Sample(), fields=COMPUTED_FIELDS_V2)
    assert sum(sum(sample.loss_mask) for sample in decoded.samples) == 3
    core.sample_postprocessor = exclude_superseded
    selected = await core.collect_samples(sid, max_seq_len=None, agent_metadata={"reward": 0.5})
    decoded = decode_samples_and_merge_input_sample(selected.body, Sample(), fields=COMPUTED_FIELDS_V2)
    assert len(decoded.samples) == 2
    assert sum(sum(sample.loss_mask) for sample in decoded.samples) == 2
    assert all(sample.reward == 0.5 for sample in decoded.samples)
    assert decoded.samples[0].tokens[: len(ancestor.token_ids)] == ancestor.token_ids
    assert decoded.session_metadata["selection"]["excluded_generation_ids"] == [ancestor.generation_id]


async def test_retry_cannot_reference_another_context(context_core):
    core = context_core
    sid, _ = await _fresh_state(core)
    first = await request(core, sid, **SessionContext(agent_run_id="a", context_id="a").headers())
    with pytest.raises(SessionConflictError, match="same context"):
        await request(
            core,
            sid,
            **SessionContext(agent_run_id="b", context_id="b").headers(),
            **{"X-Miles-Retry-Of": first.headers["x-miles-generation-id"]}
        )
    assert len(core.backend.requests) == 1


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


def test_excluded_only_row_is_removed_after_masking():
    metadata = {
        "tree": {
            "nodes": [
                {"id": 0, "generation_id": "old", "completion_span": [1, 2]},
                {"id": 1, "generation_id": "new", "completion_span": [1, 2], "supersedes": "old"},
            ]
        }
    }
    sample = Sample(
        tokens=[0, 10], response_length=1, loss_mask=[1], metadata={"leaf": {"node_id": 0, "path_node_ids": [0]}}
    )
    assert exclude_superseded([sample], metadata) == []
    assert sample.tokens == [0, 10]
    assert metadata["selection"]["trainable_tokens"] == 0
