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
        "min_replicas": 4,
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
        "capture_secret": "miles-capture",
        "cpu": 8,
        "memory_mib": 32768,
        "modal_proxy_auth": False,
        "attention": None,
        "speculative": None,
    }
    return ServingDeployment.model_validate_json(json.dumps(data | overrides))


BLACKWELL_MTP = {
    "gpu": "B300",
    "tensor_parallel": 1,
    "attention": {"backend": "trtllm_mha", "page_size": 64},
    "speculative": {"algorithm": "NEXTN", "num_steps": 3, "num_draft_tokens": 4},
}


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


def test_stage_a_example_configs_are_valid():
    from pathlib import Path

    from miles_plugins.proximal.contracts import read_run_config

    examples = Path(__file__).resolve().parents[3] / "examples" / "proximal" / "e2e"
    modal_run = read_run_config(examples / "run.stage-a.json")
    offline = read_run_config(examples / "run.stage-a.local.json")
    serving = ServingDeployment.model_validate_json((examples / "serving.stage-a.json").read_bytes())
    argv = engine_argv(modal_run, serving)
    assert parse_server_args_argv(argv).model_path == f"/models/{modal_run.tokenizer_path.name}"
    # The offline config differs only in where inference goes and its identity.
    differing = {key for key in modal_run.model_fields if getattr(modal_run, key) != getattr(offline, key)}
    assert differing == {"run_id", "inference_url", "inference_header_env"}


def test_a_pool_holding_capture_sessions_has_a_fixed_size():
    with pytest.raises(ValueError, match="set min_replicas equal to max_replicas"):
        deployment(min_replicas=1)


def test_attention_and_speculation_are_rendered_from_the_serving_config(config):
    args = parse_server_args_argv(engine_argv(config, deployment(**BLACKWELL_MTP)))
    assert (args.attention_backend, args.page_size) == ("trtllm_mha", 64)
    assert args.speculative_algorithm in ("NEXTN", "EAGLE")  # SGLang resolves NEXTN to EAGLE.
    assert (args.speculative_num_steps, args.speculative_eagle_topk, args.speculative_num_draft_tokens) == (3, 1, 4)
    # Always on with speculation: accepted tokens are exact samples from the served model.
    assert args.speculative_use_rejection_sampling is True
    plain = parse_server_args_argv(engine_argv(config, deployment()))
    assert plain.speculative_algorithm is None and plain.attention_backend is None


def test_attention_and_speculation_must_be_stated_explicitly():
    stated = deployment().model_dump(mode="json")  # null is a choice; leaving the key out is not.
    for key in ("attention", "speculative"):
        with pytest.raises(ValueError, match=f"{key}\\n  Field required"):
            ServingDeployment.model_validate_json(json.dumps({k: v for k, v in stated.items() if k != key}))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            BLACKWELL_MTP | {"speculative": {"algorithm": "NEXTN", "num_steps": 3, "num_draft_tokens": 5}},
            "num_steps \\+ 1",
        ),
        (BLACKWELL_MTP | {"gpu": "H200"}, "Blackwell"),
        (BLACKWELL_MTP | {"attention": {"backend": "trtllm_mha", "page_size": 1}}, "larger than one token"),
    ],
)
def test_serving_config_rejects_unservable_attention_or_speculation(overrides, message):
    with pytest.raises(ValueError, match=message):
        deployment(**overrides)


def test_extras_cannot_turn_on_speculation_or_change_attention(config):
    for extra in (["--speculative-algorithm", "NEXTN"], ["--attention-backend", "trtllm_mha"]):
        with pytest.raises(ValueError, match="non-operational settings"):
            engine_argv(config, deployment(extra_engine_args=extra))
