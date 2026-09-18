import json
import shlex
from pathlib import Path
from typing import Any

import pytest

from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

run = import_launch_script(REPO_ROOT / "examples" / "experimental" / "terminal_universe" / "run.py")


def test_submitted_runtime_explicitly_enables_summarization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("HARBOR_TERMINUS_2_ENABLE_SUMMARIZE", "false")
    monkeypatch.setenv("HARBOR_TERMINUS_2_LINEAR_HISTORY", "false")
    monkeypatch.setattr(run.U, "execute_train", capture)
    args = run.ScriptArgs(
        num_nodes=3,
        run_id="260101-example",
        output_dir=str(tmp_path),
        wandb_key="",
        pause_generation_mode="in_place",
    )

    run.execute(args)

    env = captured["extra_env_vars"]
    assert env["HARBOR_TERMINUS_2_ENABLE_SUMMARIZE"] == "true"
    assert env["HARBOR_TERMINUS_2_LINEAR_HISTORY"] == "true"
    assert env["AGENT_MAX_INPUT_TOKENS"] == "49152"
    assert env["AGENT_MAX_OUTPUT_TOKENS"] == "16384"
    assert env["HARBOR_MAX_SEQ_LEN"] == "65536"
    assert env["MILES_ROUTER_EXTERNAL_HOST"] == ""
    argv = shlex.split(captured["train_args"])
    assert "--use-rollout-routing-replay" in argv
    assert "--use-miles-dashboard" in argv
    assert "--observe-training-entropy" in argv
    assert "--use-rollout-entropy" in argv
    assert argv[argv.index("--pause-generation-mode") + 1] == "in_place"
    assert argv[argv.index("--num-rollout") + 1] == "1000"
    assert argv[argv.index("--save-interval") + 1] == "50"
    manifest = json.loads((tmp_path / args.run_id / "run_manifest.json").read_text())
    assert manifest["harness"] == {
        "enable_summarize": True,
        "linear_history": True,
        "max_input_tokens": 49152,
        "max_output_tokens": 16384,
    }
