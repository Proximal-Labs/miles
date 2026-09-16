import logging
import os
import shlex
from dataclasses import dataclass
from typing import Any, Self

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.rollout.session.config import compute_session_server_config
from miles.router.config import compute_miles_router_config
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.args.runtime import InferenceControllerConfig
from miles.utils.function_registry import load_function
from miles.utils.http_utils import resolve_ip
from miles.utils.workers.argv_utils import config_to_argv, python_argv_prefix
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.registration.reporter import RegistrationReporter
from miles.utils.workers.types import DeployComponent, PlatformAccess
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.static import StaticWorkerProvider, parse_host_and_port
from miles.utils.workers.worker_spec import (
    DEFAULT_RPC_PORT_INFO,
    BaseCommandSpec,
    BaseServeSpec,
    LaunchCommandContext,
    PortInfo,
    SchedulingSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
)

logger = logging.getLogger(__name__)

POOL_CATEGORY_INFERENCE_ENGINE = "inference_engine"

ENGINE_POOL_ID_PREFIX = "inference-engine"
INFERENCE_CONTROLLER_ADDR_FLAG = "--inference-controller-addr"
INFERENCE_CONTROLLER_POOL_ID = "inference-controller"
SESSION_SERVER_POOL_ID = "session-server"
INFERENCE_CONTROLLER_WORKER_CLASS = "miles.ray.rollout.inference_controller.InferenceController"
INFERENCE_REGISTRATION_REPORTER_POOL_ID = "inference-registration-reporter"
INFERENCE_REGISTRATION_REPORTER_WORKER_CLASS = "miles.utils.workers.registration.reporter.RegistrationReporterWorker"


@dataclass(kw_only=True)
class RouterSpec(BaseCommandSpec):
    name: str
    _scheduling: SchedulingSpec
    model_idx: int
    model_cfg: ModelConfig
    router_port: int | None

    def scheduling(self) -> SchedulingSpec:
        return self._scheduling

    def static_meta(self) -> dict[str, Any]:
        return {}

    def port_infos(self) -> list[PortInfo]:
        return [
            _compute_router_primary_port_info(self.router_port, model_idx=self.model_idx),
            PortInfo(name="prometheus", static_port=9000, allow_dynamic=True),
        ]

    @classmethod
    def create(cls, args: Any, *, model_idx: int, model_cfg: ModelConfig) -> Self:
        return _compute_spec_router(args, model_idx=model_idx, model_cfg=model_cfg)

    def slice_config(self, args: Any) -> Any:
        return args

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        args = ctx.args
        model_cfg = self.model_cfg
        interpreter_prefix = python_argv_prefix()
        primary = ctx.self_addrs["primary"]

        has_pd_disaggregation = model_cfg.has_pd_disaggregation or args.rollout_external_router_pd

        if args.use_miles_router:
            assert not has_pd_disaggregation, "miles router does not support PD disaggregation."
            router_config = compute_miles_router_config(
                args, host=primary.host, port=primary.port, num_engines=model_cfg.num_server_cells
            )
            launch_argv = [*interpreter_prefix, "-m", "miles.router.router", *config_to_argv(router_config)]
        else:
            router_args = compute_sglang_router_args(
                args,
                host=resolve_ip(primary.host),
                port=primary.port,
                prometheus_port=ctx.self_addrs["prometheus"].port,
                has_pd_disaggregation=has_pd_disaggregation,
            )
            logger.info(f"Launch router with args: {router_args}")
            launch_argv = [
                *interpreter_prefix,
                "-m",
                "sglang_router.launch_router",
                *router_args_to_argv(router_args),
            ]

        return shlex.join(launch_argv)


@dataclass(kw_only=True)
class SessionServerSpec(BaseCommandSpec):
    name: str
    _scheduling: SchedulingSpec
    session_server_port: int | None
    router_model_idx: int | None

    def scheduling(self) -> SchedulingSpec:
        return self._scheduling

    def static_meta(self) -> dict[str, Any]:
        return {}

    def port_infos(self) -> list[PortInfo]:
        return [_compute_session_server_primary_port_info(self.session_server_port)]

    @classmethod
    def create(cls, args: Any) -> Self:
        return _compute_spec_session_server(args)

    def slice_config(self, args: Any) -> Any:
        return args

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        args = ctx.args
        interpreter_prefix = python_argv_prefix()
        assert self.router_model_idx is not None
        (router_addrs,) = ctx.pool_addrs[compute_router_pool_id(self.router_model_idx)]
        session_config = compute_session_server_config(
            args,
            host=args.session_server_ip or ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            # TODO: make the indexing it k8s native compatible
            instance_id=compute_session_server_instance_id(args, ctx.cell_index),
            backend_url=router_addrs["primary"].addr,
        )
        launch_argv = [*interpreter_prefix, "-m", "miles.rollout.session.server", *config_to_argv(session_config)]
        return shlex.join(launch_argv)


@dataclass(kw_only=True)
class InferenceEngineSpec(BaseCommandSpec):
    name: str
    _scheduling: SchedulingSpec
    _static_meta: dict[str, Any]
    category: str = POOL_CATEGORY_INFERENCE_ENGINE
    deploy_component: DeployComponent = DeployComponent.INFERENCE
    model_idx: int
    group_index: int
    server_group_config: ServerGroupConfig
    dp_size: int

    def scheduling(self) -> SchedulingSpec:
        return self._scheduling

    def static_meta(self) -> dict[str, Any]:
        return self._static_meta

    def port_infos(self) -> list[PortInfo]:
        return [
            PortInfo(name="primary", static_port=8000, allow_dynamic=True),
            PortInfo(
                name="dist_init",
                static_port=9000,
                mode="master",
                allow_dynamic=True,
                num_consecutive=30 + self.dp_size,
            ),
            PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
            *(
                [PortInfo(name="disaggregation_bootstrap", static_port=11000, allow_dynamic=True)]
                if self.server_group_config.worker_type == WorkerType.PREFILL
                else []
            ),
            PortInfo(name="engine_info_bootstrap", static_port=12000, allow_dynamic=True),
            PortInfo(name=GATE_PORT_NAME, static_port=13000, mode="master", allow_dynamic=True),
        ]

    @classmethod
    def create(
        cls,
        args: Any,
        *,
        model_idx: int,
        group_index: int,
        model_cfg: ModelConfig,
        server_group_config: ServerGroupConfig,
    ) -> Self:
        return _compute_spec_inference_engine(
            args,
            model_idx=model_idx,
            group_index=group_index,
            model_cfg=model_cfg,
            server_group_config=server_group_config,
        )

    def slice_config(self, args: Any) -> Any:
        return args

    def env_var(self, ctx: WorkerLaunchContext) -> dict[str, str]:
        return compute_inference_engine_env_vars(ctx.args)

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        args = ctx.args
        server_group_config = self.server_group_config
        num_workers_per_cell = self.scheduling().num_workers_per_cell
        interpreter_prefix = python_argv_prefix()
        dist_init = ctx.self_addrs["dist_init"]
        # TODO: only node 0's seed is used by sglang; node != 0 should get node 0's number
        random_seed = (
            args.seed
            + server_group_config.engine_offset
            + ctx.cell_index * num_workers_per_cell
            + ctx.worker_in_cell_index
        )
        return compute_engine_launch_cmd(
            args=args,
            interpreter_prefix=interpreter_prefix,
            # TODO: make the indexing it k8s native compatible
            node_rank=ctx.worker_in_cell_index,
            worker_type=server_group_config.worker_type,
            base_gpu_id=ctx.local_gpu_ids[0],
            sglang_overrides=server_group_config.overrides,
            num_gpus_per_engine=server_group_config.num_gpus_per_engine,
            dist_init_addr=f"{dist_init.host}:{dist_init.port}",
            nccl_port=ctx.self_addrs["nccl"].port,
            host=ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            disaggregation_bootstrap_port=d.port if (d := ctx.self_addrs.get("disaggregation_bootstrap")) else None,
            engine_info_bootstrap_port=ctx.self_addrs["engine_info_bootstrap"].port,
            gated_launch_port=ctx.self_addrs[GATE_PORT_NAME].port,
            random_seed=random_seed,
        )


@dataclass(kw_only=True)
class InferenceControllerSpec(BaseServeSpec):
    worker_type = "inference_controller"
    args: InferenceControllerConfig
    name: str = INFERENCE_CONTROLLER_POOL_ID
    platform_access: PlatformAccess = PlatformAccess.READ
    worker_class: str = INFERENCE_CONTROLLER_WORKER_CLASS

    def scheduling(self) -> SchedulingSpec:
        return SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
            pin_to_head=self.args.pin_rollout_manager_to_head,
        )

    def static_meta(self) -> dict[str, Any]:
        return {}

    def port_infos(self) -> list[PortInfo]:
        return [DEFAULT_RPC_PORT_INFO]

    @classmethod
    def create(cls, args: Any) -> Self:
        return cls(
            name=INFERENCE_CONTROLLER_POOL_ID,
            platform_access=PlatformAccess.READ,
            args=InferenceControllerConfig.from_config(args),
            worker_class=INFERENCE_CONTROLLER_WORKER_CLASS,
        )

    def slice_config(self, args: Any) -> InferenceControllerConfig:
        return InferenceControllerConfig.from_config(args)

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(
            args=ctx.args,
            engine_provider=_compute_controller_engine_provider(ctx.args, capability=ctx.capability),
            router_providers=compute_router_providers(ctx.args, capability=ctx.capability),
        )


@dataclass(kw_only=True)
class InferenceRegistrationReporterSpec(BaseServeSpec):
    worker_type = "inference_registration_reporter"
    args: InferenceControllerConfig
    name: str = INFERENCE_REGISTRATION_REPORTER_POOL_ID
    deploy_component: DeployComponent = DeployComponent.INFERENCE
    platform_access: PlatformAccess = PlatformAccess.READ
    worker_class: str = INFERENCE_REGISTRATION_REPORTER_WORKER_CLASS

    def scheduling(self) -> SchedulingSpec:
        return SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
            pin_to_head=self.args.pin_rollout_manager_to_head,
        )

    def static_meta(self) -> dict[str, Any]:
        return {}

    def port_infos(self) -> list[PortInfo]:
        return [DEFAULT_RPC_PORT_INFO]

    @classmethod
    def create(cls, args: Any) -> list[Self]:
        if DeployComponent(args.deploy_component) is not DeployComponent.INFERENCE:
            return []
        return [
            cls(
                name=INFERENCE_REGISTRATION_REPORTER_POOL_ID,
                deploy_component=DeployComponent.INFERENCE,
                platform_access=PlatformAccess.READ,
                args=InferenceControllerConfig.from_config(args),
                worker_class=INFERENCE_REGISTRATION_REPORTER_WORKER_CLASS,
            )
        ]

    def slice_config(self, args: Any) -> InferenceControllerConfig:
        return InferenceControllerConfig.from_config(args)

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(
            args=ctx.args,
            reporter=_create_inference_registration_reporter(ctx.args, capability=ctx.capability),
        )


def _compute_spec_router(args, model_idx: int, model_cfg: ModelConfig) -> RouterSpec:
    return RouterSpec(
        model_idx=model_idx,
        model_cfg=model_cfg,
        name=compute_router_pool_id(model_idx),
        router_port=args.sglang_router_port,
        _scheduling=SchedulingSpec.single(
            num_gpus_per_worker=0,
            # TODO: refactor the flag
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
    )


def _compute_router_primary_port_info(router_port: int | None, model_idx: int) -> PortInfo:
    if router_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=router_port + model_idx)


def _compute_spec_session_server(args: Any) -> SessionServerSpec:
    sglang_config = args.sglang  # TODO avoid resolve repeatedly
    router_model_idx = 0 if sglang_config.models else None

    return SessionServerSpec(
        name=SESSION_SERVER_POOL_ID,
        session_server_port=args.session_server_port,
        router_model_idx=router_model_idx,
        _scheduling=SchedulingSpec(
            num_cells=(args.session_server_workers if args.use_session_server and router_model_idx is not None else 0),
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=0,
            pin_to_head=True,
        ),
    )


def _compute_session_server_primary_port_info(session_server_port: int | None) -> PortInfo:
    if session_server_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=session_server_port, offset_by_cell=True)


def _compute_spec_inference_engine(
    args,
    model_idx: int,
    group_index: int,
    model_cfg: ModelConfig,
    server_group_config: ServerGroupConfig,
) -> InferenceEngineSpec:
    num_workers_per_cell = max(1, server_group_config.num_gpus_per_engine // args.num_gpus_per_node)

    num_gpus_per_engine = server_group_config.num_gpus_per_engine
    assert num_gpus_per_engine <= args.num_gpus_per_node or num_gpus_per_engine % args.num_gpus_per_node == 0, (
        f"group '{server_group_config.worker_type.value}' wants {num_gpus_per_engine=} which neither fits in one node of "
        f"{args.num_gpus_per_node} gpus nor tiles whole nodes, so its ranks would never all be launched"
    )

    scheduling = SchedulingSpec(
        num_cells=server_group_config.num_gpus // server_group_config.num_gpus_per_engine,
        num_workers_per_cell=num_workers_per_cell,
        # TODO: may need real num for k8s native mode
        num_gpus_per_worker=0.2,
        num_gpu_slots_per_worker=min(server_group_config.num_gpus_per_engine, args.num_gpus_per_node),
        num_gpus_per_node=args.num_gpus_per_node,
        pg_name="rollout",
        pg_slot_offset=server_group_config.gpu_offset,
    )

    num_workers_total = server_group_config.num_gpus // scheduling.num_gpu_slots_per_worker
    assert num_workers_total % scheduling.num_workers_per_cell == 0, (
        f"group '{server_group_config.worker_type.value}' has {num_workers_total=} which is not a whole number of "
        f"{scheduling.num_workers_per_cell}-worker engines; the trailing engine would have no node to run its "
        f"remaining ranks"
    )

    return InferenceEngineSpec(
        model_idx=model_idx,
        group_index=group_index,
        server_group_config=server_group_config,
        name=compute_engine_pool_id(args, model_idx=model_idx, group_index=group_index),
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        deploy_component=DeployComponent.INFERENCE,
        dp_size=args.sglang.get_value("dp_size", group=server_group_config),
        _scheduling=scheduling,
        # TODO: reduce complexity around passing around configs later during arguments refactor
        _static_meta=dict(
            model_id=model_cfg.name,
            worker_type=server_group_config.worker_type.value,
            num_gpus_per_engine=server_group_config.num_gpus_per_engine,
            sglang_api_key=args.sglang.get_value("api_key", group=server_group_config),
            needs_offload=server_group_config.needs_offload,
            update_weights=model_cfg.update_weights,
            gpu_offset_base=server_group_config.gpu_offset,
            gpu_offset_stride_per_cell=scheduling.num_workers_per_cell * scheduling.num_gpu_slots_per_worker,
        ),
    )


def _compute_controller_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    if DeployComponent(args.deploy_component).deploys_own_inference_engines():
        return compute_engine_provider(args, capability=capability)
    return RegistrationHub(run_uuid=args.run_uuid)


def _create_inference_registration_reporter(args, *, capability: BackendCapability) -> RegistrationReporter:
    controller_provider = compute_inference_controller_provider(args, capability=capability)
    return RegistrationReporter(
        run_uuid=args.run_uuid,
        reporter_id=args.deploy_instance_id,
        hub_endpoint=controller_provider.get_handle(inference_controller_worker_name()),
        worker_provider=compute_engine_provider(args, capability=capability),
    )


def compute_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    path = args.custom_inference_engine_provider_path
    fn = load_function(path)
    fn_args = compute_custom_function_config(args, path)
    return fn(fn_args, capability=capability)


def backend_inference_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    return capability.dynamic_worker_provider(pool_ids=None, category=POOL_CATEGORY_INFERENCE_ENGINE)


def compute_router_providers(args, *, capability: BackendCapability) -> list[BaseWorkerProvider]:
    return [
        capability.static_worker_provider(pool_id=compute_router_pool_id(model_idx))
        for model_idx in range(len(args.sglang.models))
    ]


def spec_inference_controller(args: Any) -> BaseServeSpec:
    return InferenceControllerSpec.create(args)


def specs_inference_registration_reporter(args: Any) -> list[BaseServeSpec]:
    return InferenceRegistrationReporterSpec.create(args)


def create_inference_controller_handle(*, capability: BackendCapability) -> BaseWorkerHandle:
    worker_name = inference_controller_worker_name()
    provider = capability.static_worker_provider(pool_id=INFERENCE_CONTROLLER_POOL_ID)
    return provider.get_handle(worker_name)


def compute_inference_controller_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    if (entry := args.inference_controller_addr) is not None:
        return StaticWorkerProvider.of_rpc_addrs(
            pool_id=INFERENCE_CONTROLLER_POOL_ID,
            addrs=[parse_host_and_port(entry)],
            worker_class=INFERENCE_CONTROLLER_WORKER_CLASS,
        )
    return capability.static_worker_provider(pool_id=INFERENCE_CONTROLLER_POOL_ID)


def session_server_worker_name(cell_index: int) -> str:
    return compute_worker_name(pool_id=SESSION_SERVER_POOL_ID, cell_index=cell_index)


def inference_controller_worker_name() -> str:
    return compute_worker_name(pool_id=INFERENCE_CONTROLLER_POOL_ID)


def specs_router(args) -> list[BaseCommandSpec]:
    sglang_config = args.sglang  # TODO avoid resolve repeatedly
    return [
        RouterSpec.create(args, model_idx=model_idx, model_cfg=model_cfg)
        for model_idx, model_cfg in enumerate(sglang_config.models)
    ]


def compute_router_pool_id(model_idx: int) -> str:
    return f"inference-router-{model_idx}"


def compute_router_worker_name(model_idx: int) -> str:
    return compute_worker_name(pool_id=compute_router_pool_id(model_idx))


def spec_session_server(args: Any) -> BaseCommandSpec:
    return SessionServerSpec.create(args)


def compute_session_server_instance_id(args, instance_index: int) -> str:
    return f"{args.run_uuid}-{instance_index}"


def compute_engine_pool_id(args, *, model_idx: int, group_index: int) -> str:
    segment = args.deploy_instance_id or DeployComponent(args.deploy_component).value
    return f"{ENGINE_POOL_ID_PREFIX}-{segment}-{model_idx}-{group_index}"


def specs_inference_engine(args) -> list[BaseCommandSpec]:
    if args.rollout_external:
        return []

    sglang_config = args.sglang  # TODO avoid resolve repeatedly

    return [
        InferenceEngineSpec.create(
            args,
            model_idx=model_idx,
            group_index=group_index,
            model_cfg=model_cfg,
            server_group_config=server_group_config,
        )
        for model_idx, model_cfg in enumerate(sglang_config.models)
        for group_index, server_group_config in enumerate(model_cfg.server_groups)
        if server_group_config.worker_type != WorkerType.PLACEHOLDER
    ]


def compute_engine_pool_ids(args) -> list[str]:
    if args.rollout_external:
        return []
    return [
        compute_engine_pool_id(args, model_idx=model_idx, group_index=group_index)
        for model_idx, model in enumerate(args.sglang.models)
        for group_index, group in enumerate(model.server_groups)
        if group.worker_type != WorkerType.PLACEHOLDER
    ]


def compute_inference_engine_env_vars(args) -> dict[str, str]:
    env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
        key: os.environ.get(key, default_val)
        for key, default_val in {
            # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
            "NVSHMEM_DISABLE_NCCL": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": (
                "0" if args.colocate and args.rollout_num_gpus_per_engine > 1 else "1"
            ),
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            "SGLANG_EXPOSE_OWN_ENV_VARS": "1",
        }.items()
    }
    if args.dumper_enable or args.dumper_inference:
        from miles.utils import dumper_utils

        env_vars.update(dumper_utils.get_sglang_env(args))
    return env_vars
