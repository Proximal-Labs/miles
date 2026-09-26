"""Validated, secret-free configuration and durable evaluation records."""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalConfig:
    platform_url: str
    sets: dict[str, list[int]]
    platform_ui_url: str | None = None
    rollouts_per_environment: int = 1
    max_concurrent_rollouts: int = 4
    timeout_seconds: int = 14400
    poll_seconds: int = 15
    agent_type: str = "default"
    reasoning_effort: str = "MAX"
    api_key_env: str = "PROXIMAL_API_KEY"
    modal_secret: str = "inkling-eval"
    serving_gpu: str = "B300:8"
    serving_tp: int = 8
    context_length: int = 1048576
    deployment_config: dict = field(default_factory=lambda: {"modal": {}})

    def __post_init__(self):
        if not self.platform_url.startswith("https://"):
            raise ValueError("platform_url must be an HTTPS Proximal API URL")
        if self.platform_ui_url is not None and not self.platform_ui_url.startswith("https://"):
            raise ValueError("platform_ui_url must be HTTPS")
        if not self.sets:
            raise ValueError("Evaluation requires at least one named set")
        for name, ids in self.sets.items():
            if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}", name):
                raise ValueError(f"Invalid evaluation set name: {name}")
            if not ids or any(type(i) is not int or i <= 0 for i in ids) or len(ids) != len(set(ids)):
                raise ValueError(f"{name}: provide distinct positive environment IDs")
        for name in (
            "rollouts_per_environment",
            "max_concurrent_rollouts",
            "timeout_seconds",
            "poll_seconds",
            "serving_tp",
            "context_length",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.reasoning_effort not in {"NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH", "MAX"}:
            raise ValueError("Unsupported reasoning_effort")

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluation_due(completed_steps: int, samples_per_epoch: int, every: int, *, batch_size: int = 1) -> bool:
    if every <= 0 or completed_steps <= 0:
        return False
    interval = samples_per_epoch * every
    return completed_steps * batch_size // interval > (completed_steps - 1) * batch_size // interval


def summarize(results: list[dict]) -> dict:
    scored = [r["reward"] for r in results if r["reward"] is not None]
    metrics = {"rollouts": len(results), "scored": len(scored)}
    if scored:
        metrics["reward/mean"] = sum(scored) / len(scored)
    return metrics
