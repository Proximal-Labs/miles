"""Finalization must drain admitted requests and fence late completions."""

import asyncio

import pytest

from miles.rollout.session.errors import SessionConflictError
from miles.rollout.session.v2.lifecycle import SessionLifecycle


async def _start_finish(lifecycle, *, timeout=60, producer_finished=True):
    task = asyncio.create_task(lifecycle.finish(producer_finished=producer_finished, timeout=timeout))
    await asyncio.sleep(0)
    return task


@pytest.mark.parametrize("completion_order", [(0, 1), (1, 0)], ids=["forward", "reverse"])
async def test_finish_drains_both_requests_and_rejects_new_admission(completion_order):
    lifecycle = SessionLifecycle()
    async with lifecycle.lock:
        tickets = [lifecycle.admit(), lifecycle.admit()]
    finish = await _start_finish(lifecycle)
    async with lifecycle.lock:
        with pytest.raises(SessionConflictError):
            lifecycle.admit()
        lifecycle.resolve(tickets[completion_order[0]], failed=False)
    assert not finish.done()
    async with lifecycle.lock:
        assert lifecycle.can_commit(tickets[completion_order[1]])
        lifecycle.resolve(tickets[completion_order[1]], failed=False)
    finished = await finish
    assert finished.complete
    assert await lifecycle.finish(producer_finished=True, timeout=60) is finished


async def test_cancelled_finish_caller_does_not_cancel_shared_drain():
    lifecycle = SessionLifecycle()
    async with lifecycle.lock:
        ticket = lifecycle.admit()
    finish = await _start_finish(lifecycle)
    finish.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finish
    async with lifecycle.lock:
        lifecycle.resolve(ticket, failed=False)
    assert (await lifecycle.finish(producer_finished=True, timeout=60)).complete


async def test_timeout_fences_late_completion_without_changing_snapshot():
    lifecycle = SessionLifecycle()
    async with lifecycle.lock:
        ticket = lifecycle.admit()
    finished = await lifecycle.finish(producer_finished=True, timeout=0)
    assert not finished.complete
    assert finished.unfinished_requests == 1
    async with lifecycle.lock:
        assert not lifecycle.can_commit(ticket)
        lifecycle.resolve(ticket, failed=False)
    assert await lifecycle.finish(producer_finished=True, timeout=0) is finished


@pytest.mark.parametrize("producer_finished,request_failed", [(False, False), (True, True)])
async def test_idle_does_not_imply_a_complete_trace(producer_finished, request_failed):
    lifecycle = SessionLifecycle()
    async with lifecycle.lock:
        ticket = lifecycle.admit()
        lifecycle.resolve(ticket, failed=request_failed)
    finished = await lifecycle.finish(producer_finished=producer_finished, timeout=0)
    assert not finished.complete
    assert finished.unfinished_requests == 0
    assert finished.failed_requests == int(request_failed)


async def test_finish_rejects_conflicting_retry_parameters():
    lifecycle = SessionLifecycle()
    await lifecycle.finish(producer_finished=False, timeout=0)
    with pytest.raises(SessionConflictError, match="parameters changed"):
        await lifecycle.finish(producer_finished=True, timeout=0)
