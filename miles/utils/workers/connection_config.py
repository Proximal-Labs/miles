import json
from typing import Any

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_spec import BaseServeSpec, BaseSpec, PortInfo

WORKER_METADATA_ANNOTATION = "miles.radixark.io/worker-metadata"


class StaticPoolConnInfo(FrozenStrictBaseModel):
    name: str
    port_infos: list[PortInfo]
    worker_class: str | None
    num_cells: int
    num_workers_per_cell: int
    pods_per_cell: int


class StaticConnConfig(FrozenStrictBaseModel):
    static_conn_infos: dict[str, StaticPoolConnInfo] = Field(default_factory=dict)


class WorkerPodMetadata(FrozenStrictBaseModel):
    category: str | None = None
    workers_per_pod: int = Field(gt=0)
    pods_per_cell: int = Field(gt=0)
    gpu_slots_per_worker: int = Field(ge=0)
    gpus_per_cell: int = Field(ge=0)
    worker_class: str | None
    port_infos: list[PortInfo]
    meta: dict[str, Any] = Field(default_factory=dict)
    gpu_offset_base: int | None = None
    gpu_offset_stride_per_cell: int = 0
    include_cell_index: bool = False


def build_static_conn_config(*, specs: list[BaseSpec]) -> StaticConnConfig:
    return StaticConnConfig(
        static_conn_infos={
            spec.name: StaticPoolConnInfo(
                name=spec.name,
                port_infos=spec.port_infos(),
                worker_class=spec.worker_class if isinstance(spec, BaseServeSpec) else None,
                num_cells=spec.scheduling().num_cells,
                num_workers_per_cell=spec.scheduling().num_workers_per_cell,
                pods_per_cell=spec.scheduling().pods_per_cell(),
            )
            for spec in specs
            if spec.scheduling().gpus_per_cell() == 0
        }
    )


def build_worker_annotations(*, spec: BaseSpec) -> dict[str, str]:
    meta = spec.static_meta().copy()
    gpu_offset_base = meta.pop("gpu_offset_base", None)
    gpu_offset_stride_per_cell = meta.pop("gpu_offset_stride_per_cell", 0)
    include_cell_index = meta.pop("include_cell_index", False)
    metadata = WorkerPodMetadata(
        category=spec.category,
        workers_per_pod=spec.scheduling().workers_per_pod(),
        pods_per_cell=spec.scheduling().pods_per_cell(),
        gpu_slots_per_worker=spec.scheduling().num_gpu_slots_per_worker,
        gpus_per_cell=spec.scheduling().gpus_per_cell(),
        worker_class=spec.worker_class if isinstance(spec, BaseServeSpec) else None,
        port_infos=spec.port_infos(),
        meta=meta,
        gpu_offset_base=gpu_offset_base,
        gpu_offset_stride_per_cell=gpu_offset_stride_per_cell,
        include_cell_index=include_cell_index,
    )
    return {
        WORKER_METADATA_ANNOTATION: json.dumps(metadata.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    }
