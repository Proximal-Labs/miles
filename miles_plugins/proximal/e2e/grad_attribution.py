"""Gradient attribution: which loss terms move the LoRA, measured on one real recorded step.

Replays step N of a run from its snapshots through the production trainer, at a learning
rate too small to move any weight, starting from step N-1's checkpoint and optimizer state:

    step N    the recorded batch, as trained        -> g_full
    step N+1  the same samples with every reward 0  -> every GRPO advantage is 0, so the
              policy loss contributes nothing: g_aux is the reward-independent loss terms
    step N+2  the recorded batch again              -> the numerical noise floor

Each step's gradient is read back from the Adam moments it saves, g = (m' - b1 m) / (1 - b1),
and split by parameter (decoder, MTP block, vision tower) with the saved parameter names.
The production step's own gradient, from the two snapshots' moments, checks the replay.

    PROXIMAL_RUN_CONFIG=run.json \\
    PROXIMAL_SERVING_CONFIG=examples/proximal/qwen38/overhead/serving.json \\
    PROXIMAL_TRAINING_CONFIG=examples/proximal/qwen38/overhead/training.json \\
      modal run --env main -m miles_plugins.proximal.e2e.grad_attribution --step 19
"""

import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

from miles_plugins.proximal import modal_training as node
from miles_plugins.proximal.e2e.trainer_replay import MOCK, _wait_or_stall, replay_command
from miles_plugins.proximal.serving_app import DEPLOYMENT, RUN, base_volume

# Small enough that no fp32 master weight moves (Adam steps are ~lr), so every arm sees the same weights.
FROZEN_LR = "1e-30"
CHECKPOINTS = Path("/state/checkpoints")  # --save in the fork's train_args.
RESULTS_MOUNT = Path("/results")
# Created once by hand (`modal volume create miles-forensics --env main`); referenced, never created here.
results_volume = modal.Volume.from_name(
    "miles-forensics", environment_name=node.TRAINING.state_volume.environment_name, create_if_missing=False
)

app = modal.App(f"{node.TRAINING.app_name}-grad-attribution")


def recorded_batch(step: int) -> list[Any]:
    """The samples production trained on at ``step``: groups its cursor consumed after step - 1."""
    from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
    from miles.utils.types import Sample
    from miles_plugins.proximal.data_source import Cursor
    from miles_plugins.proximal.store import StoredGroup

    steps = node.SNAPSHOT / "steps"
    before = {c.group_id for c in _cursor(steps / f"{step - 1:07d}", Cursor).consumed}
    groups = [c for c in _cursor(steps / f"{step:07d}", Cursor).consumed if c.group_id not in before]
    samples: list[Any] = []
    for consumed in groups:
        payload = (node.SNAPSHOT / "artifacts" / RUN.run_id / "groups" / f"{consumed.group_id}.bin").read_bytes()
        size = int.from_bytes(payload[:8], "big")
        header = StoredGroup.model_validate_json(payload[8 : 8 + size])
        decoded = decode_samples_and_merge_input_sample(payload[8 + size :], Sample(), fields=COMPUTED_FIELDS_V2)
        for sample, identity in zip(decoded.samples, header.identities, strict=True):
            sample.index, sample.group_index = identity.index, identity.group_index
            samples.append(sample)
    print(f"[attribution] step {step}: {len(groups)} groups, {len(samples)} samples", flush=True)
    return samples


def _cursor(step_dir: Path, cursor_type: Any) -> Any:
    return cursor_type.model_validate_json((step_dir / "cursor.json").read_bytes())


def write_arms(step: int, samples: list[Any]) -> None:
    import torch

    real = [s.to_dict() for s in samples]
    zero = copy.deepcopy(real)
    for sample in zero:
        sample["reward"] = 0.0
    MOCK.mkdir(parents=True, exist_ok=True)
    for rollout_id, arm in ((step, real), (step + 1, zero), (step + 2, real)):
        torch.save({"rollout_id": rollout_id, "metadata": {}, "samples": arm}, MOCK / f"{rollout_id}.pt")


def _states(directory: Path) -> tuple[Any, ...]:
    """Flatten the 8 ranks' parameter state: fp32 params, Adam moments, element names and sizes."""
    import torch

    params, avg, avg_sq, names, sizes = [], [], [], [], []
    beta1 = None
    for rank in range(node.TRAINING.num_gpus):
        state = torch.load(directory / f"training_state_rank{rank}.pt", map_location="cpu", weights_only=False)
        beta1 = state["optimizer"]["optimizer"]["param_groups"][0]["betas"][0]
        rank_names = state.get("optimizer_parameter_names")
        for part_index, part in enumerate(state["optimizer_parameter_state"]):
            for gbuf_idx, dtype_state in ((k, v) for k, v in part.items() if isinstance(k, int)):
                for bucket_idx, bucket in enumerate(next(iter(dtype_state.values()))):
                    for element_idx, element in enumerate(bucket):
                        params.append(element["param"].flatten().float())
                        avg.append(element["exp_avg"].flatten().float())
                        avg_sq.append(element["exp_avg_sq"].flatten().float())
                        sizes.append(params[-1].numel())
                        names.append(rank_names[part_index][gbuf_idx][bucket_idx][element_idx] if rank_names else None)
    return torch.cat(params), torch.cat(avg), torch.cat(avg_sq), names, sizes, beta1


def _category(name: str) -> str:
    if ".mtp." in name:
        return "mtp"
    if "vision" in name or "visual" in name:
        return "vision"
    return "decoder"


def analyse(step: int, out: Path) -> dict[str, Any]:
    import torch

    torch.set_num_threads(32)
    snap = node.SNAPSHOT / "steps"

    def replay_dir(rollout_id: int) -> Path:
        return CHECKPOINTS / f"iter_{rollout_id:07d}" / "adapter"

    p_prev, m_prev, v_prev, _, prod_sizes, beta1 = _states(snap / f"{step - 1:07d}" / "checkpoint" / "adapter")
    _, m_true, _, _, _, _ = _states(snap / f"{step:07d}" / "checkpoint" / "adapter")
    p_a, m_a, _, names, sizes, _ = _states(replay_dir(step))
    _, m_b, _, _, _, _ = _states(replay_dir(step + 1))
    p_a2, m_a2, _, _, _, _ = _states(replay_dir(step + 2))
    if sizes != prod_sizes:
        raise RuntimeError("Replay parameter layout differs from production's")

    def derive(new: Any, old: Any) -> Any:
        return (new - beta1 * old) / (1 - beta1)

    g = {
        "true": derive(m_true, m_prev),
        "full": derive(m_a, m_prev),
        "aux": derive(m_b, m_a),
        "full_again": derive(m_a2, m_b),
    }
    g["policy"] = g["full"] - g["aux"]
    # Adam divides each element by sqrt(v): what an element's gradient does to the update.
    scale = v_prev.sqrt() + 1e-8
    element_category = [_category(n or "?") for n in names]
    layer = [int(m.group(1)) if (m := re.search(r"layers\.(\d+)\.", n or "")) else -1 for n in names]
    masks: dict[str, Any] = {}
    offsets = torch.tensor([0, *sizes]).cumsum(0)
    for category in ("decoder", "mtp", "vision"):
        mask = torch.zeros(len(p_prev), dtype=torch.bool)
        for i, c in enumerate(element_category):
            if c == category:
                mask[offsets[i] : offsets[i + 1]] = True
        masks[category] = mask
    masks["all"] = torch.ones(len(p_prev), dtype=torch.bool)

    def cos(a: Any, b: Any) -> float:
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    report: dict[str, Any] = {
        "beta1": beta1,
        "elements": len(sizes),
        "numel": int(len(p_prev)),
        "weights_moved_max_abs": float(max((p_a - p_prev).abs().max(), (p_a2 - p_prev).abs().max())),
        "categories": {},
    }
    for category, mask in masks.items():
        if not mask.any():
            continue
        x = {k: v[mask] for k, v in g.items()}
        norm = {k: v / scale[mask] for k, v in x.items()}
        drift = m_prev[mask]
        nz = drift != 0
        report["categories"][category] = {
            "numel": int(mask.sum()),
            "norm": {k: float(v.norm()) for k, v in x.items()},
            "adam_normalized_norm": {k: float(v.norm()) for k, v in norm.items()},
            "cos_replay_vs_production": cos(x["full"], x["true"]),
            "cos_noise_floor": cos(x["full"], x["full_again"]),
            "cos_aux_vs_full": cos(x["aux"], x["full"]),
            "cos_policy_vs_full": cos(x["policy"], x["full"]),
            "cos_aux_vs_policy": cos(x["aux"], x["policy"]),
            # The persistent direction: the momentum carried into this step.
            "cos_vs_momentum": {k: cos(v, drift) for k, v in x.items()},
            "adam_normalized_cos_vs_momentum": {k: cos(v, drift / scale[mask]) for k, v in norm.items()},
            "sign_agreement_with_momentum": {
                k: float((torch.sign(v[nz]) == torch.sign(drift[nz])).float().mean()) for k, v in x.items()
            },
        }
    per_layer: dict[int, dict[str, float]] = {}
    for i, (category, index) in enumerate(zip(element_category, layer, strict=True)):
        if category != "decoder" or index < 0:
            continue
        span = slice(int(offsets[i]), int(offsets[i + 1]))
        row = per_layer.setdefault(index, {"aux": 0.0, "policy": 0.0})
        row["aux"] += float(g["aux"][span].pow(2).sum())
        row["policy"] += float(g["policy"][span].pow(2).sum())
    report["decoder_layers"] = {index: {k: v**0.5 for k, v in row.items()} for index, row in sorted(per_layer.items())}
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "sizes": sizes, "m_prev": m_prev, "v_prev": v_prev, **g}, out / "gradients.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


@app.function(
    image=node.image.add_local_file(node.REPO / "train.py", str(node.FORK / "train.py")),
    gpu=node.TRAINING.gpu,
    cpu=float(node.TRAINING.cpu),
    memory=node.TRAINING.memory_mib,
    volumes={
        str(DEPLOYMENT.base_mount): base_volume,
        str(node.SNAPSHOT_MOUNT): node.state_volume,
        str(RESULTS_MOUNT): results_volume,
        **node.kernel_mounts,
    },
    timeout=4 * 3600,
)
def attribute(step: int) -> dict[str, Any]:
    started = time.monotonic()
    write_arms(step, recorded_batch(step))
    command = [
        *replay_command(step + 3),
        "--lora-adapter-path", str(node.SNAPSHOT / "steps" / f"{step - 1:07d}" / "checkpoint" / "adapter"),
        "--lr", FROZEN_LR,
    ]  # fmt: skip
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    subprocess.run(
        ["ray", "start", "--head", "--num-gpus", str(node.TRAINING.num_gpus), "--disable-usage-stats"], check=True
    )
    log = Path("/tmp/attribution.log")
    with log.open("w") as out:
        trainer = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        code, _ = _wait_or_stall(trainer, log, 0)
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    text = log.read_text(errors="replace")
    steps_logged = [line[:600] for line in text.splitlines() if re.search(r"model\.py:\d+ - step \d+:", line)]
    result: dict[str, Any] = {
        "exit_code": code,
        "trainer_seconds": round(time.monotonic() - started),
        "steps": steps_logged,
    }
    destination = RESULTS_MOUNT / RUN.run_id / f"step{step}"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "trainer.log").write_text(text)
    if code == 0:
        result["report"] = analyse(step, destination)
    else:
        result["tail"] = text[-15000:]
    results_volume.commit()
    return result


@app.local_entrypoint()
def main(step: int, out: str = "grad_attribution.json") -> None:
    result = attribute.remote(step)
    Path(out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "tail"}, indent=2)[:6000])
    if "tail" in result:
        print(result["tail"][-6000:], file=sys.stderr)
