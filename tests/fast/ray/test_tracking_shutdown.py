"""Check shutdown ordering without importing the GPU training dependencies."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def _method(path, class_name, name, **dependencies):
    source = Path(__file__).resolve().parents[3] / path
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if getattr(node, "name", None) == name)
    namespace = dict(dependencies)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.asyncio
async def test_dispose_waits_for_worker_flush_and_skips_dead_cells():
    started, release = asyncio.Event(), asyncio.Event()
    watcher = AsyncMock()

    async def flush(*args, **kwargs):
        watcher.assert_awaited_once()
        started.set()
        await release.wait()

    alive = SimpleNamespace(is_alive=True, execute=AsyncMock(side_effect=flush))
    dead = SimpleNamespace(is_alive=False, execute=AsyncMock())
    controller = SimpleNamespace(_watcher_disposer=watcher, _cells=[alive, dead])
    dispose = _method("miles/ray/train/group.py", "TrainerController", "dispose", asyncio=asyncio)
    task = asyncio.create_task(dispose(controller))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not task.done()
    release.set()
    await task
    alive.execute.assert_awaited_once_with("finish_tracking", kill_on_failure=False)
    dead.execute.assert_not_called()
    assert controller._watcher_disposer is None


def test_worker_flushes_its_own_tracking_manager():
    tracking = SimpleNamespace(finish_tracking=Mock())
    finish = _method("miles/ray/train_actor.py", "TrainRayActor", "finish_tracking", tracking=tracking)
    finish(SimpleNamespace())
    tracking.finish_tracking.assert_called_once_with()
