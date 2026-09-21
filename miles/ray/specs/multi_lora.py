from typing import Any, ClassVar, Self

from miles.ray.specs.inference import compute_router_providers
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime import MultiLoraConfig
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import BaseServeSpec, SchedulingSpec, WorkerCtorContext

MULTI_LORA_CONTROLLER_POOL_ID = "multi-lora-controller"
MULTI_LORA_CONTROLLER_WORKER_CLASS = "miles.ray.multi_lora.controller.MultiLoRAController"


class MultiLoraControllerSpec(BaseServeSpec):
    worker_type: ClassVar[str] = "multi_lora"
    config_class = MultiLoraConfig
    args: MultiLoraConfig
    name: str = MULTI_LORA_CONTROLLER_POOL_ID
    worker_class: str = MULTI_LORA_CONTROLLER_WORKER_CLASS

    @classmethod
    def create(cls, config: MultiLoraConfig) -> Self:
        return cls(args=config)

    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return SchedulingSpec(
            num_cells=1 if is_multi_lora_enabled(self.args) else 0,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=0,
            # Pinned to the head node so the API sits at a port-forwardable address.
            pin_to_head=True,
        )

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(
            args=ctx.args,
            router_providers=compute_router_providers(ctx.args, capability=ctx.capability),
        )


def create_multi_lora_controller_handle(*, capability: BackendCapability) -> BaseWorkerHandle:
    worker_name = multi_lora_controller_worker_name()
    provider = capability.static_worker_provider(pool_id=MULTI_LORA_CONTROLLER_POOL_ID)
    return provider.get_handle(worker_name)


def multi_lora_controller_worker_name() -> str:
    return compute_worker_name(pool_id=MULTI_LORA_CONTROLLER_POOL_ID)


def multi_lora_controller_cell_id() -> str:
    return compute_cell_id(pool_id=MULTI_LORA_CONTROLLER_POOL_ID, cell_index=0)
