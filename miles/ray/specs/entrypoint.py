from miles.ray.specs import inference, multi_lora, rollout, train
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import BaseSpec


def compute_specs(args, *, worker_type: str | None = None) -> list[BaseSpec]:
    match worker_type:
        case "rollout":
            return [rollout.RolloutExecutorSpec(args=args)]
        case "multi_lora":
            return [multi_lora.MultiLoraControllerSpec(args=args)]
        case "inference_controller":
            return [inference.InferenceControllerSpec(args=args)]
        case "inference_registration_reporter":
            return [inference.InferenceRegistrationReporterSpec(args=args)]
        case "trainer_controller":
            return [train.TrainerControllerSpec(args=args)]
        case "trainer":
            return [train.TrainerSpec(args=args)]
        case None:
            pass
        case _:
            raise ValueError(f"Unknown worker type: {worker_type!r}")

    selector = DeployComponent(args.deploy_component)
    return [spec for spec in _compute_all_specs(args) if selector.selects(spec.deploy_component)]


def _compute_all_specs(args) -> list[BaseSpec]:
    return [
        rollout.spec_rollout_executor(args),
        multi_lora.spec_multi_lora_controller(args),
        inference.spec_inference_controller(args),
        *inference.specs_router(args),
        *inference.specs_inference_registration_reporter(args),
        inference.spec_session_server(args),
        *inference.specs_inference_engine(args),
        *train.specs_trainer_controller(args),
        *train.specs_trainer(args),
    ]
