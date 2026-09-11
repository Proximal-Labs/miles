from __future__ import annotations

import asyncio

import ray.actor

from miles.utils.test_utils.fault_hooks import FaultHookCommand, FaultHookRecord
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations, FaultTarget
from miles.utils.workers.worker_provider.base import CellInfo


class RayCellOperations(BaseCellOperations):
    def __init__(self, *, worker_manager_handle: ray.actor.ActorHandle) -> None:
        self._worker_manager_handle = worker_manager_handle

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids)

    async def suspend(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.stop_cells.remote([cell_id])

    async def resume(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.start_cells.remote([cell_id])

    async def observe_fault_target(self, *, cell_id: str, sub_index: int) -> FaultTarget:
        return await self._worker_manager_handle.observe_fault_target.remote(cell_id, sub_index=sub_index)

    async def control_fault_hook(self, *, target: FaultTarget, command: FaultHookCommand) -> str | FaultHookRecord:
        return await asyncio.wait_for(
            self._worker_manager_handle.control_fault_hook.remote(target=target, command=command), timeout=10.0
        )

    async def inject_fault(
        self,
        *,
        cell_id: str,
        mode: FailureMode,
        sub_index: int,
        expected_target: FaultTarget | None = None,
    ) -> None:
        if expected_target is None:
            await self._worker_manager_handle.inject_fault.remote(
                cell_id, mode=mode.value, worker_in_cell_index=sub_index
            )
        else:
            await self._worker_manager_handle.inject_fault.remote(
                cell_id, mode=mode.value, worker_in_cell_index=sub_index, expected_target=expected_target
            )
