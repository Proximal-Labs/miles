from __future__ import annotations

import asyncio
import logging

import httpx

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations, FaultTarget, StaleFaultTargetError
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER, HEALTH_PATH, POD_UID_HEADER
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError
from miles.utils.workers.worker_provider.base import CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import RPC_PORT_NAME

logger = logging.getLogger(__name__)

INJECT_FAULT_TIMEOUT_SECONDS = 60.0


class KubernetesCellOperations(BaseCellOperations):
    def __init__(self, *, provider: KubernetesWorkerProvider, namespace: str) -> None:
        self._provider = provider
        self._namespace = namespace
        self._watching: asyncio.Task[StopWatchFn] | None = None

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        await self._ensure_watching()

        wanted = set(pool_ids)
        infos = (self._provider.cell_info(cell_id) for cell_id in self._provider.cell_ids())
        return {info.cell_id: info for info in infos if info is not None and info.pool_id in wanted}

    async def suspend(self, *, cell_id: str) -> None:
        await self._ensure_watching()

        pods = self._provider.pod_names_of_cell(cell_id)
        assert pods, f"cannot suspend {cell_id}, which has no pods"
        await _delete_pods(namespace=self._namespace, pod_names=pods)

    async def resume(self, *, cell_id: str) -> None:
        raise NotImplementedError(
            "a deleted cell comes back when its workload recreates it, so resume has no moment to return at"
        )

    async def observe_fault_target(self, *, cell_id: str, sub_index: int) -> FaultTarget:
        await self._ensure_watching()
        incarnation = self._provider.cell_incarnation(cell_id)
        if incarnation is None:
            raise StaleFaultTargetError(f"Cell {cell_id} has disappeared")
        (infos,) = self._provider.get_worker_infos(cell_ids=[cell_id])
        if not 0 <= sub_index < len(infos):
            raise StaleFaultTargetError(f"Cell {cell_id} has no worker at index {sub_index}")
        info = infos[sub_index]
        if info.worker_class is None:
            raise NotImplementedError(f"Worker {info.name} is not served over RPC")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{info.self_addrs[RPC_PORT_NAME].addr.rstrip('/')}{HEALTH_PATH}")
            response.raise_for_status()
        boot_uuid = response.headers.get(BOOT_UUID_HEADER)
        pod_uid = response.headers.get(POD_UID_HEADER)
        current = self._provider.cell_incarnation(cell_id)
        if (
            not boot_uuid
            or not pod_uid
            or pod_uid not in {pod.uid for pod in incarnation.pods}
            or current is None
            or current.workers_hash != incarnation.workers_hash
        ):
            raise StaleFaultTargetError(f"RPC identity does not match the observed pods of {cell_id}")
        return FaultTarget(
            cell_id=cell_id,
            sub_index=sub_index,
            workers_hash=incarnation.workers_hash,
            boot_uuid=boot_uuid,
            pod_uid=pod_uid,
        )

    async def inject_fault(
        self, *, cell_id: str, mode: FailureMode, sub_index: int, expected_target: FaultTarget | None = None
    ) -> None:
        await self._ensure_watching()
        if expected_target is not None:
            incarnation = self._provider.cell_incarnation(cell_id)
            if (
                expected_target.cell_id != cell_id
                or expected_target.sub_index != sub_index
                or not expected_target.boot_uuid
                or incarnation is None
                or incarnation.workers_hash != expected_target.workers_hash
                or expected_target.pod_uid not in {pod.uid for pod in incarnation.pods}
            ):
                raise StaleFaultTargetError(f"Cell {cell_id} no longer matches the observed fault target")

        (infos,) = self._provider.get_worker_infos(cell_ids=[cell_id])
        assert (
            0 <= sub_index < len(infos)
        ), f"sub_index {sub_index} is out of range for cell {cell_id}, which has {len(infos)} workers"

        worker_name = infos[sub_index].name
        if expected_target is None:
            handles = self._provider.get_handles_of_worker_infos(infos)
            assert (
                worker_name in handles
            ), f"{worker_name} is not served over rpc, so no call can reach the process to crash it"
            handle = handles[worker_name]
        else:
            handle = build_rpc_handle_of_worker_info(infos[sub_index], expected_boot_uuid=expected_target.boot_uuid)
        await _inject_fault_over_rpc(handle=handle, mode=mode, worker_name=worker_name)

    async def _ensure_watching(self) -> None:
        if self._watching is None:
            self._watching = asyncio.ensure_future(self._provider.watch_cells(_ignore_cell))
        try:
            await self._watching
        except BaseException:
            self._watching = None
            raise


async def _inject_fault_over_rpc(*, handle: BaseWorkerHandle, mode: FailureMode, worker_name: str) -> None:
    try:
        await asyncio.wait_for(
            handle.submit_without_result("inject_fault", mode=mode.value), timeout=INJECT_FAULT_TIMEOUT_SECONDS
        )
    except ServerRestartedError as error:
        raise StaleFaultTargetError(f"Worker {worker_name} changed its boot identity") from error
    except (WorkerUnreachableError, TimeoutError, asyncio.TimeoutError):
        logger.info("Injecting %s into %s left it unreachable, which is what was asked for", mode.value, worker_name)


async def _ignore_cell(cell_id: str, info: CellInfo | None) -> None:
    return None


async def _delete_pods(*, namespace: str, pod_names: list[str]) -> None:
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    async with kubernetes_client.ApiClient() as api_client:
        core_v1_api = kubernetes_client.CoreV1Api(api_client)
        await asyncio.gather(
            *(core_v1_api.delete_namespaced_pod(name=pod_name, namespace=namespace) for pod_name in pod_names)
        )
