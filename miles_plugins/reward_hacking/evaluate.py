"""Resumable, bounded-concurrency evaluation with explicit missing/error outcomes."""

import asyncio
import hashlib
import json
import os
import time
from collections import defaultdict

import httpx

from miles_plugins.reward_hacking.curate import read_jsonl, write_json
from miles_plugins.reward_hacking.prompt import FINAL_INSTRUCTION, PROMPT_PROVENANCE, SYSTEM_PROMPT
from miles_plugins.reward_hacking.providers import classify, monitor_input, request_body


def load_examples(paths, limit=None):
    rows = []
    seen = set()
    for path in paths:
        for index, record in enumerate(read_jsonl(path)):
            if limit is not None and index >= limit:
                break
            prompt = monitor_input(record)
            identity = record.get("id") or hashlib.sha256(prompt.encode()).hexdigest()[:24]
            if identity in seen:
                raise ValueError(f"Duplicate example ID: {identity}")
            if "label" in record and type(record["label"]) is not bool:
                raise ValueError("Ground-truth label must be boolean or absent")
            seen.add(identity)
            rows.append(
                {
                    "id": identity,
                    "prompt": prompt,
                    "label": record.get("label"),
                    "metadata": record.get("metadata", {}),
                }
            )
    if not rows:
        raise ValueError("No examples to evaluate")
    return rows


def repeat_examples(rows, attempts):
    """Give each requested classification its own stable resume identity."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if attempts == 1:
        return rows
    return [
        {
            **row,
            "id": f"{row['id']}/attempt-{attempt}",
            "metadata": {**row["metadata"], "source_example_id": row["id"], "attempt": attempt},
        }
        for row in rows
        for attempt in range(1, attempts + 1)
    ]


def dry_run(rows, models):
    return {
        "examples": len(rows),
        "models": list(models),
        "requests": len(rows) * len(models),
        "positive": sum(r["label"] is True for r in rows),
        "negative": sum(r["label"] is False for r in rows),
        "max_input_characters": max(len(r["prompt"]) for r in rows),
        "input_truncation": "disabled; provider context errors are recorded, never silently shortened",
    }


def _divide(a, b):
    return a / b if b else None


def metrics(records):
    labeled = [r for r in records if type(r.get("label")) is bool]
    valid = [r for r in labeled if r["status"] == "ok"]
    tp = sum(r["label"] and r["reward_hacking"] for r in valid)
    tn = sum(not r["label"] and not r["reward_hacking"] for r in valid)
    fp = sum(not r["label"] and r["reward_hacking"] for r in valid)
    fn = sum(r["label"] and not r["reward_hacking"] for r in valid)
    positives = sum(r["label"] for r in labeled)
    known = [r for r in labeled if r["label"] and r["metadata"].get("known_evidence_visible")]
    return {
        "examples": len(records),
        "labeled": len(labeled),
        "successful": len(valid),
        "errors": sum(r["status"] != "ok" for r in records),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_successful": _divide(tp + tn, len(valid)),
        "accuracy_all": _divide(tp + tn, len(labeled)),
        "positive_recall_all": _divide(tp, positives),
        "positive_recall_successful": _divide(tp, tp + fn),
        "false_positive_rate_successful": _divide(fp, fp + tn),
        "precision": _divide(tp, tp + fp),
        "known_evidence_positive_count": len(known),
        "known_evidence_recall_all": _divide(
            sum(r["status"] == "ok" and r["reward_hacking"] for r in known), len(known)
        ),
    }


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        prefix = record["metadata"].get("fraction", "custom")
        model = record["model_alias"]
        groups[f"{model}/prefix-{prefix}/all"].append(record)
        role = record["metadata"].get("role", "unspecified")
        groups[f"{model}/prefix-{prefix}/{role}"].append(record)
    return {key: metrics(value) for key, value in sorted(groups.items())}


def _read_predictions(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        # A process kill can tear only the final append. Save it before retrying.
        end = raw.rfind(b"\n") + 1
        path.with_suffix(".interrupted-tail").write_bytes(raw[end:])
        path.write_bytes(raw[:end])
    return list(read_jsonl(path))


def prepare_run(rows, models, config, output):
    """An output directory is bound to exact inputs, model settings and monitor prompt."""
    contract = {
        "version": 1,
        "config": config,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "prompt_provenance": PROMPT_PROVENANCE,
        "final_instruction_sha256": hashlib.sha256(FINAL_INSTRUCTION.encode()).hexdigest(),
        "examples": [
            {
                "id": r["id"],
                "label": r["label"],
                "metadata": r["metadata"],
                "input_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest(),
            }
            for r in rows
        ],
        "models": models,
    }
    path = output / "run.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Evaluation inputs/config changed; use a new output directory")
    write_json(path, contract)
    (output / "system-prompt.txt").write_text(SYSTEM_PROMPT)
    (output / "final-instruction.txt").write_text(FINAL_INSTRUCTION)
    prior = _read_predictions(output / "predictions.jsonl")
    keyed = {(r["model_alias"], r["id"]): r for r in prior}
    expected = {(m, r["id"]) for m in models for r in rows}
    if not keyed.keys() <= expected:
        raise ValueError("Predictions contain rows outside this run")
    return keyed


async def evaluate_model(rows, alias, model, route, output, previous, *, concurrency, timeout):
    semaphore = asyncio.Semaphore(concurrency)
    path = output / "predictions.jsonl"

    async def evaluate_one(row, client):
        key = (alias, row["id"])
        if key in previous and previous[key]["status"] == "ok":
            return
        async with semaphore:
            started = time.monotonic()
            try:
                result = await classify(client, model, route, row["prompt"])
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                result = {"status": "error", "error": type(error).__name__ + ": " + str(error)}
            record = {
                "id": row["id"],
                "label": row["label"],
                "metadata": row["metadata"],
                "model_alias": alias,
                "model": model["model"],
                "elapsed_seconds": time.monotonic() - started,
                **result,
            }
            # No awaits between append and update: one event-loop owner writes complete lines.
            with path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            previous[key] = record
            write_json(output / "summary.json", summarize(list(previous.values())))
            print(f"{alias}: {row['id']} {record['status']}", flush=True)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [asyncio.create_task(evaluate_one(row, client)) for row in rows]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return previous


def validate_models(models):
    if not models:
        raise ValueError("Select at least one model")
    for alias, model in models.items():
        if "concurrency" in model and (type(model["concurrency"]) is not int or model["concurrency"] < 1):
            raise ValueError(f"{alias}: concurrency must be a positive integer")
        if model.get("route") not in {"gateway", "modal"}:
            raise ValueError(f"{alias}: unsupported route")
        if not isinstance(model.get("model"), str) or not model["model"]:
            raise ValueError(f"{alias}: model ID is required")
        request_body(model, "schema validation")
