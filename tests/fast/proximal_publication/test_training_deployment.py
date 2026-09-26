import json
from pathlib import Path

import pytest

from miles_plugins.proximal.contracts import RunConfig
from miles_plugins.proximal.training import (
    TrainingDeployment,
    check_deployment,
    check_train_args,
    pool_endpoint_name,
    prepare_run_config,
    read_training_deployment,
    registered_capture_endpoint,
    registration_command,
    routes_to_pool,
)

REPO = Path(__file__).resolve().parents[3]
STAGE_A = REPO / "examples" / "proximal" / "e2e" / "run.stage-a.json"
QWEN38 = REPO / "examples" / "proximal" / "qwen38"
GSM8K = REPO / "examples" / "proximal" / "gsm8k"
TUNNEL = "https://abc123.r5.modal.host"


def _run(**changes: object) -> RunConfig:
    """Stage A's run with capture in the pool, as on the Modal topology."""
    raw = json.loads(STAGE_A.read_text())
    raw["capture"]["url"] = raw["inference_url"]
    for key, value in changes.items():
        raw[key] = value
    return RunConfig.model_validate_json(json.dumps(raw))


def _real_run() -> RunConfig:
    return _run(
        platform={"url": "https://backend.example.com", "api_key_env": "PROXIMAL_PLATFORM_API_KEY"},
        platform_route={"model": "miles/qwen38-27b", "endpoint_name": None},
    )


def _registry(**endpoints: dict[str, str]) -> str:
    return json.dumps(
        {
            "version": 1,
            "models": {"miles/qwen38-27b": {"defaultEndpoint": "capture-2", "endpoints": endpoints}},
        }
    )


def test_registry_resolves_the_default_capture_endpoint():
    registry = _registry(
        **{
            "capture-1": {"kind": "rollout_capture", "baseURL": "https://old.modal.host"},
            "capture-2": {"kind": "rollout_capture", "baseURL": TUNNEL},
        }
    )
    assert registered_capture_endpoint(registry, "miles/qwen38-27b", None)["baseURL"] == TUNNEL
    assert (
        registered_capture_endpoint(registry, "miles/qwen38-27b", "capture-1")["baseURL"] == "https://old.modal.host"
    )
    assert registered_capture_endpoint(registry, "miles/other", None) is None


def test_registry_ignores_endpoints_that_are_not_rollout_capture():
    registry = _registry(**{"capture-2": {"mode": "dedicated", "baseURL": TUNNEL}})
    assert registered_capture_endpoint(registry, "miles/qwen38-27b", None) is None


def test_the_pool_counts_as_registered_only_with_the_runs_budget():
    run = _real_run()
    sampling = run.research.sampling
    pool = {"kind": "rollout_capture", "baseURL": run.capture.url}
    budget = {"contextWindowTokens": sampling.max_sequence_tokens, "maxOutputTokens": sampling.max_tokens}
    assert routes_to_pool(_registry(**{"capture-2": pool | budget}), run)
    assert not routes_to_pool(_registry(**{"capture-2": pool}), run)  # Legacy entry: the platform's default budget.
    stale = budget | {"contextWindowTokens": sampling.max_sequence_tokens // 2}
    assert not routes_to_pool(_registry(**{"capture-2": pool | stale}), run)
    assert not routes_to_pool(_registry(**{"capture-2": budget | {"kind": "rollout_capture", "baseURL": TUNNEL}}), run)


def test_registration_command_names_the_tunnel_and_worker_key():
    command = registration_command(_real_run(), "capture-17", TUNNEL)
    assert "--model miles/qwen38-27b --register capture-17 --set-default capture-17" in command
    assert f"--kind rollout_capture --base-url {TUNNEL}" in command
    sampling = _real_run().research.sampling
    assert (
        f"--api-key-env STAGE_A_CAPTURE_PLATFORM_KEY --context-window-tokens {sampling.max_sequence_tokens} "
        f"--max-output-tokens {sampling.max_tokens} --apply"
    ) in command
    assert (
        pool_endpoint_name("miles-qwen38-serving", _real_run())
        == f"miles-qwen38-serving-ctx{sampling.max_sequence_tokens}"
    )


def test_example_deployments_match_their_platforms():
    gsm8k = read_training_deployment(GSM8K / "training.json")
    qwen38 = read_training_deployment(QWEN38 / "training.json")
    assert gsm8k.platform.kind == "gsm8k" and qwen38.platform.kind == "real"
    check_deployment(_run(), gsm8k)  # Stage A's platform is loopback, like the gsm8k stand-in.
    check_deployment(_real_run(), qwen38)


@pytest.mark.parametrize(
    ("run", "change", "message"),
    [
        (_real_run, {"gpu": "H200:4"}, "provides 4 GPUs"),
        (_run, {}, "needs the platform's URL"),
        (
            lambda: _real_run().model_copy(update={"platform_route": _run().platform_route}),
            {},
            "leave platform_route.endpoint_name unset",
        ),
    ],
)
def test_real_deployments_refuse_mismatched_configs(run, change, message):
    deployment = read_training_deployment(QWEN38 / "training.json").model_copy(update=change)
    with pytest.raises(ValueError, match=message):
        check_deployment(run(), deployment)


def test_deployments_refuse_capture_outside_the_pool():
    beside_trainer = _real_run().model_copy(update={"capture": _real_run().capture.model_copy(update={"url": TUNNEL})})
    with pytest.raises(ValueError, match="capture.url must be the pool's inference_url"):
        check_deployment(beside_trainer, read_training_deployment(QWEN38 / "training.json"))


def test_gsm8k_deployment_refuses_a_remote_platform():
    with pytest.raises(ValueError, match="runs on loopback"):
        check_deployment(_real_run(), read_training_deployment(GSM8K / "training.json"))


@pytest.mark.parametrize("directory", [GSM8K, QWEN38, QWEN38 / "smoke"])
def test_example_train_args_ask_for_the_deployments_gpus(directory):
    deployment = read_training_deployment(directory / "training.json")
    check_train_args(deployment, (REPO / deployment.train_args).read_text())


def test_train_args_for_the_wrong_gpu_count_are_refused():
    smoke = read_training_deployment(QWEN38 / "smoke" / "training.json")
    with pytest.raises(ValueError, match="asks for 8 GPUs, but the deployment provides 4"):
        check_train_args(smoke, (QWEN38 / "train_args.txt").read_text())


def test_prepare_writes_a_valid_qwen38_smoke_config(tmp_path):
    smoke = QWEN38 / "smoke"
    run = prepare_run_config(
        smoke / "run.template.json", smoke / "tasks.json", "https://pool.modal.direct", tmp_path / "run.json"
    )
    assert len(run.dataset.tasks) == 1 and run.research.group_size == 4 and run.max_in_flight_samples == 4
    deployment = read_training_deployment(smoke / "training.json")
    assert deployment.max_retries == 0
    check_deployment(run, deployment)


def test_prepare_writes_a_valid_qwen38_run_config(tmp_path):
    out = tmp_path / "run.json"
    run = prepare_run_config(
        QWEN38 / "run.template.json", QWEN38 / "tasks-pilot.json", "https://pool.modal.direct", out
    )
    assert run.tito_model == "qwen38small" and run.model_protocol.tool_call_parser == "qwen3_coder"
    assert run.platform_route.endpoint_name is None and run.dataset.project_id == 519
    assert run.capture.url == run.inference_url == "https://pool.modal.direct"
    assert len(run.dataset.tasks) == 41
    check_deployment(run, read_training_deployment(QWEN38 / "training.json"))
    assert isinstance(read_training_deployment(QWEN38 / "training.json"), TrainingDeployment)


def test_determinism_is_stated_per_deployment_and_off_where_blackwell_hd256_backward_needs_it():
    # FlashAttention's SM100 backward for 256-wide heads (Qwen3.8 on B300) has no deterministic mode.
    assert read_training_deployment(QWEN38 / "overhead" / "training.json").deterministic_kernels is False
    assert read_training_deployment(GSM8K / "training.json").deterministic_kernels is True
    raw = json.loads((GSM8K / "training.json").read_text())
    del raw["deterministic_kernels"]
    with pytest.raises(ValueError, match="deterministic_kernels"):
        TrainingDeployment.model_validate_json(json.dumps(raw))
