import json

import pytest

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.contracts import Attempt, Policy, RunConfig, digest
from miles_plugins.proximal.snapshot import SnapshotReference


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("PX_TEST_KEY", "platform-secret")
    monkeypatch.setenv("CAPTURE_TEST_KEY", "capture-secret")
    monkeypatch.setenv("FLEET_TEST_KEY", "fleet-secret")
    data = {
        "run_id": "test-run",
        "base_model": {"name": "test-qwen3", "revision": "a" * 40},
        "dataset": {"project_id": 12, "tasks": [{"environment_id": 7, "image_id": 8, "source_commit_sha": "b" * 40}]},
        "harness": {
            "agent_type": "native",
            "revision": "c" * 40,
            "max_turns": 4,
            "max_session_tokens": 4096,
            "timeout_seconds": 60,
            "p2p_enforce": True,
        },
        "research": {
            "behavior_correction": "rollout_logprobs",
            "sampling": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 64,
                "max_sequence_tokens": 4096,
                "logprob_semantics": "untransformed",
                "budget_policy": "cap_to_remaining_context",
            },
            "group_size": 2,
            "max_policy_lag": 1,
            "unused_groups": "retry",
            "max_consecutive_failed_groups": 2,
        },
        "platform": {"url": "http://127.0.0.1:9010", "api_key_env": "PX_TEST_KEY"},
        "capture": {"url": "http://127.0.0.1:9011", "api_key_env": "CAPTURE_TEST_KEY"},
        "inference_url": "http://127.0.0.1:9012",
        "inference_header_env": {"Authorization": "FLEET_TEST_KEY"},
        "volume": {"volume_name": "adapters", "environment_name": "dev"},
        "artifact_directory": str(tmp_path / "artifacts"),
        "tokenizer_path": str(tmp_path),
        "tito_model": "qwen3",
        "enable_thinking": True,
        "max_in_flight_samples": 4,
        "completed_group_capacity": 2,
        "request_timeout_seconds": 10,
        "poll_interval_seconds": 0.01,
    }
    return RunConfig.model_validate_json(json.dumps(data))


@pytest.fixture
def authorization(config):
    return authorize_run(config, yes_rollouts=True, yes_publish=True)


@pytest.fixture
def policy(config):
    return Policy(
        run_id=config.run_id, version=1, snapshot=SnapshotReference(sha256="d" * 64), base_model=config.base_model
    )


@pytest.fixture
def attempt(config, policy):
    return Attempt(
        attempt_id="attempt-1",
        run_id=config.run_id,
        group_id="group-1",
        sample_index=0,
        dataset_sha256=digest(config.dataset),
        task=config.dataset.tasks[0],
        harness=config.harness,
        policy=policy,
        sampling=config.research.sampling,
    )
