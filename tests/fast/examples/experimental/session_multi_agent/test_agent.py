"""Cancellation and reviewer failures must join cleanup before the example exits."""

import asyncio

import anyio
import pytest
from examples.experimental.session_multi_agent import agent


def _reply():
    return {"id": "draft", "choices": [{"message": {"role": "assistant", "content": "answer"}}]}


async def test_parent_cancellation_joins_async_child_cleanup(monkeypatch):
    both_started = asyncio.Event()
    started = set()
    cleaned = set()
    parent_calls = []

    async def complete(client, base_url, context, *args, **kwargs):
        if context.agent_run_id == "main":
            parent_calls.append(context)
            return _reply()
        try:
            started.add(context.agent_run_id)
            if len(started) == 2:
                both_started.set()
            await asyncio.Event().wait()
        finally:
            with anyio.CancelScope(shield=True):
                await asyncio.sleep(0)
                cleaned.add(context.agent_run_id)

    monkeypatch.setattr(agent, "_complete", complete)
    episode = asyncio.create_task(agent.run("http://session", "question", {}, {}))
    await asyncio.wait_for(both_started.wait(), timeout=5)
    episode.cancel()
    with pytest.raises(asyncio.CancelledError):
        await episode
    assert cleaned == {"review-a", "review-b"}
    assert len(parent_calls) == 1


async def test_reviewer_failure_joins_peer_before_propagating(monkeypatch):
    peer_started = asyncio.Event()
    peer_cleaned = asyncio.Event()
    parent_calls = []

    async def complete(client, base_url, context, *args, **kwargs):
        if context.agent_run_id == "main":
            parent_calls.append(context)
            return _reply()
        if context.agent_run_id == "review-a":
            await peer_started.wait()
            raise RuntimeError("review failed")
        try:
            peer_started.set()
            await asyncio.Event().wait()
        finally:
            with anyio.CancelScope(shield=True):
                await asyncio.sleep(0)
                peer_cleaned.set()

    monkeypatch.setattr(agent, "_complete", complete)
    with pytest.raises(Exception) as errors:
        await agent.run("http://session", "question", {}, {})
    assert str(errors.value.exceptions[0]) == "review failed"
    assert peer_cleaned.is_set()
    assert len(parent_calls) == 1
