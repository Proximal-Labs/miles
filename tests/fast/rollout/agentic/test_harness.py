"""A producer barrier must include tool work even while model requests are idle."""

import asyncio

import httpx
import pytest

from miles.rollout.agentic.harness import AgentRun, GenerationRequest
from miles.rollout.session.v2.contexts import SessionContext


async def test_result_waits_for_child_tool_work_and_grandchildren():
    started = asyncio.Event()
    release = asyncio.Event()
    joined = []
    async with httpx.AsyncClient() as client:
        run = AgentRun("http://session", client)

        async def tool():
            started.set()
            await release.wait()
            run.create_task(grandchild())
            joined.append("tool")

        async def grandchild():
            joined.append("grandchild")

        async def episode():
            async with run:
                run.create_task(tool())
                with pytest.raises(RuntimeError, match="Join"):
                    run.result()
            return run.result({"reward": 0.0})

        episode_task = asyncio.create_task(episode())
        await started.wait()
        assert not episode_task.done()
        release.set()
        result = await episode_task
    assert result.producer_finished
    assert joined == ["tool", "grandchild"]
    assert result.metadata["reward"] == 0.0


async def test_cancelled_child_cannot_claim_complete_episode():
    async with httpx.AsyncClient() as client:
        run = AgentRun("http://session", client)
        async with run:
            child = run.create_task(asyncio.Event().wait())
            child.cancel()
        assert not run.result().producer_finished


async def test_failed_child_cancels_and_joins_siblings():
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def sibling():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def failure():
        await started.wait()
        raise RuntimeError("child failed")

    async with httpx.AsyncClient() as client:
        run = AgentRun("http://session", client)
        with pytest.raises(Exception) as errors:
            async with run:
                run.create_task(sibling())
                run.create_task(failure())
        assert len(errors.value.exceptions) == 1
        assert isinstance(errors.value.exceptions[0], RuntimeError)
        assert stopped.is_set()
        assert not run.result().producer_finished


async def test_parent_failure_joins_tasks_whose_joiners_have_not_started():
    async with httpx.AsyncClient() as client:
        run = AgentRun("http://session", client)
        with pytest.raises(Exception) as errors:
            async with run:
                child = run.create_task(asyncio.Event().wait())
                raise RuntimeError("parent failed before yielding")
        assert str(errors.value.exceptions[0]) == "parent failed before yielding"
        assert child.done()
        assert not run.result().producer_finished


def test_generation_request_identity_is_stable_only_within_one_attempt():
    context = SessionContext(agent_run_id="child", context_id="child", parent_agent_run_id="main")
    attempt = GenerationRequest(context, previous_response_id="previous", retry_of="earlier")
    first_headers = attempt.headers()
    assert attempt.headers() == first_headers
    assert GenerationRequest(context).headers()["X-Miles-Idempotency-Key"] != first_headers["X-Miles-Idempotency-Key"]
    first_headers["X-Miles-Idempotency-Key"] = "changed"
    assert attempt.headers()["X-Miles-Idempotency-Key"] != "changed"
