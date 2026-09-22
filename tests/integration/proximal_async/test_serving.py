"""The serving pool's SGLang arguments come from the run config and parse in SGLang."""

import importlib
import json
import sys

import pytest

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles_plugins.proximal.serving import ServingDeployment, engine_argv, gateway_config


def deployment(**overrides):
    data = {
        "app_name": "miles-serving-test",
        "image": "radixark/miles@sha256:" + "0" * 64,
        "gpu": "H100:2",
        "tensor_parallel": 2,
        "routing_region": "us-west",
        "min_replicas": 1,
        "max_replicas": 4,
        "target_concurrency": 8,
        "scaledown_window_seconds": 1200,
        "startup_timeout_seconds": 3600,
        "base_volume": {"volume_name": "base-weights", "environment_name": "dev"},
        "base_mount": "/models",
        "adapter_mount": "/adapters",
        "local_cache": "/cache",
        "max_loaded_adapters": 4,
        "gateway_secret": "miles-gateway",
        "gateway_key_env": "MILES_GATEWAY_KEY",
    }
    return ServingDeployment.model_validate_json(json.dumps(data | overrides))


def test_engine_arguments_follow_the_run_lora_contract(config):
    serving = deployment()
    args = parse_server_args_argv(engine_argv(config, serving))
    lora = config.research.lora
    assert args.enable_lora is True
    assert args.max_lora_rank == lora.rank
    assert set(args.lora_target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert args.max_loras_per_batch == serving.max_loaded_adapters
    assert args.lora_paths is None  # Named versions are loaded at runtime by the gateway.
    assert args.served_model_name == config.base_model.name
    assert args.model_path == f"/models/{config.tokenizer_path.name}"
    assert args.host == "127.0.0.1" and args.tp_size == 2
    gateway = gateway_config(config, serving)
    assert gateway.engine_model_path == args.model_path
    assert gateway.replica.backend_url == f"http://127.0.0.1:{args.port}"


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-lora-rank", "128"],
        ["--model", "/models/other"],  # An alias for --model-path.
        ["--tokenizer-path", "/models/other-tokenizer"],
        ["--host", "0.0.0.0"],
        ["--reasoning-parser", "deepseek-r1"],
        ["--lora-paths", "x=/adapters/x"],
        ["--load-format", "dummy"],  # Random weights behind the configured model path.
        ["--quantization", "fp8"],
    ],
)
def test_extras_may_change_only_operational_settings(config, extra):
    with pytest.raises(ValueError, match="non-operational settings"):
        engine_argv(config, deployment(extra_engine_args=extra))


def test_extras_must_parse_and_performance_flags_pass(config):
    with pytest.raises(SystemExit):
        engine_argv(config, deployment(extra_engine_args=["--not-an-sglang-flag"]))
    ok = deployment(extra_engine_args=["--mem-fraction-static", "0.85"])
    assert parse_server_args_argv(engine_argv(config, ok)).mem_fraction_static == 0.85


def test_modal_app_builds_offline_from_both_configs(config, tmp_path, monkeypatch):
    run_path, serving_path = tmp_path / "run.json", tmp_path / "serving.json"
    run_path.write_text(config.model_dump_json())
    serving_path.write_text(deployment().model_dump_json())
    monkeypatch.setenv("PROXIMAL_RUN_CONFIG", str(run_path))
    monkeypatch.setenv("PROXIMAL_SERVING_CONFIG", str(serving_path))
    monkeypatch.delenv("PROXIMAL_RUN_CONFIG_JSON", raising=False)
    monkeypatch.delenv("PROXIMAL_SERVING_CONFIG_JSON", raising=False)
    sys.modules.pop("miles_plugins.proximal.serving_app", None)
    module = importlib.import_module("miles_plugins.proximal.serving_app")
    assert module.app.name == "miles-serving-test"
    assert module.RUN == config and module.DEPLOYMENT == deployment()
    # Engine arguments are rendered at deploy time; a bad flag fails here, not on a GPU.
    serving_path.write_text(deployment(extra_engine_args=["--model", "/models/other"]).model_dump_json())
    sys.modules.pop("miles_plugins.proximal.serving_app", None)
    with pytest.raises(ValueError, match="non-operational"):
        importlib.import_module("miles_plugins.proximal.serving_app")
