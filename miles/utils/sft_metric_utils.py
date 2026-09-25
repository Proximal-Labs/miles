"""Metrics for the offline Inkling SFT recipe, without RL rollout diagnostics."""

from statistics import mean, median


def is_inkling_sft(args):
    return getattr(args, "rollout_function_path", None) == "miles.rollout.inkling_sft.generate_rollout"


def data_metrics(samples, rollout_id, elapsed):
    targets = [sum(sample.loss_mask) for sample in samples]
    metrics = {
        "train/step": rollout_id,
        "data/num_samples": len(samples),
        "data/sequence_tokens/mean": mean(len(sample.tokens) for sample in samples),
        "data/target_tokens/mean": mean(targets),
        "data/target_tokens/median": median(targets),
        "data/target_tokens/min": min(targets),
        "data/target_tokens/max": max(targets),
        "perf/data_load_time": elapsed,
    }
    metrics.update(samples[-1].metadata.get("_sft_progress", {}))
    return metrics


def perf_metrics(times, seq_lens, rollout_id):
    metrics = {"train/step": rollout_id}
    for name in ("data_preprocess", "train_wait", "save_model"):
        if name in times:
            metrics[f"perf/{name}_time"] = times[name]
    if "actor_train" in times:
        duration = times["actor_train"]
        metrics["perf/train_time"] = duration
        if duration > 0:
            metrics["perf/train_tok_per_s"] = sum(seq_lens) / duration
    if "train_wait" in times and "train" in times:
        duration = times["train_wait"] + times["train"]
        if duration > 0:
            metrics["perf/step_time"] = duration
            metrics["perf/wait_time_ratio"] = times["train_wait"] / duration
    return metrics
