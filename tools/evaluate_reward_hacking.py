"""Evaluate Luna, Sol and a temporary Modal-served Inkling Small monitor.

python -m tools.evaluate_reward_hacking --datasets data/reward-hacking-monitor/prefix-*.jsonl --dry-run
python -m tools.evaluate_reward_hacking --datasets data/reward-hacking-monitor/prefix-*.jsonl --output outputs/reward-hacking-monitor

Gateway credentials and MODAL_INFERENCE_API_KEY come from the environment.
An interrupted run resumes successful predictions and retries recorded errors.
Use --cleanup with the same output directory to stop an orphaned owned Modal app.
"""

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path

from miles_plugins.reward_hacking.curate import write_json
from miles_plugins.reward_hacking.dataset_source import dataset_paths
from miles_plugins.reward_hacking.evaluate import (
    dry_run,
    evaluate_model,
    load_examples,
    prepare_run,
    repeat_examples,
    summarize,
    validate_models,
)
from miles_plugins.reward_hacking.modal_server import cancellation_handlers, inference_endpoint
from miles_plugins.reward_hacking.providers import gateway_route
from miles_plugins.reward_hacking.result_store import result_uploads


def _cleanup(output):
    # Optional Modal lifecycle dependencies are loaded only for execution, never dataset preparation.
    from miles_plugins.inkling_eval.serving import stop

    path = output / "modal-deployment.json"
    state = json.loads(path.read_text())
    if not state["name"].startswith("reward-hacking-monitor-"):
        raise ValueError("Refusing to stop an app not created by this evaluator")
    if state["status"] != "stopped":
        stop(state.get("app_id", state["name"]), state["environment"])
        state["status"] = "stopped"
        write_json(path, state)


def _execute(args, config, rows, models):
    # Validate all routes before allocating GPUs or spending inference requests.
    routes = {alias: gateway_route(config["gateway"]) for alias, m in models.items() if m["route"] == "gateway"}
    if any(m["route"] == "modal" for m in models.values()) and not os.environ.get("MODAL_INFERENCE_API_KEY"):
        raise ValueError("Set MODAL_INFERENCE_API_KEY before starting an evaluation that includes Inkling")
    contract_config = {**config, "resolved_gateway_urls": {alias: route[0] for alias, route in routes.items()}}
    previous = prepare_run(rows, models, contract_config, args.output)
    deployment_path = args.output / "modal-deployment.json"
    if deployment_path.exists() and json.loads(deployment_path.read_text())["status"] != "stopped":
        raise ValueError("An earlier Modal deployment needs cleanup; run with --cleanup first")
    for alias, model in models.items():
        pending = [r for r in rows if previous.get((alias, r["id"]), {}).get("status") != "ok"]
        if not pending:
            continue
        if model["route"] == "modal":
            with inference_endpoint(config["modal"], args.output) as (wire_model, route):
                asyncio.run(
                    evaluate_model(
                        pending,
                        alias,
                        {**model, "model": wire_model},
                        route,
                        args.output,
                        previous,
                        concurrency=model.get("concurrency", args.concurrency),
                        timeout=args.timeout,
                    )
                )
        else:
            asyncio.run(
                evaluate_model(
                    pending,
                    alias,
                    model,
                    routes[alias],
                    args.output,
                    previous,
                    concurrency=model.get("concurrency", args.concurrency),
                    timeout=args.timeout,
                )
            )
    summary = summarize(list(previous.values()))
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if any(r["status"] != "ok" for r in previous.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("run-configs/reward-hacking-monitor.json"))
    parser.add_argument(
        "--datasets", nargs="+", help="Local files or modal://volume/path URIs; defaults to config datasets"
    )
    parser.add_argument("--models", nargs="+", default=["luna", "sol", "inkling-small"])
    parser.add_argument("--output", type=Path, default=Path("outputs/reward-hacking-monitor"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=1, help="Classifications per example per model")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--limit", type=int, help="First N examples from each input file, for smoke runs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without inference or GPUs; Modal datasets require download credentials",
    )
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1 or args.concurrency < 1 or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("attempts, concurrency, timeout and limit must be positive")
    if args.cleanup and args.dry_run:
        parser.error("--cleanup and --dry-run cannot be combined")
    if args.cleanup:
        with (args.output / ".lock").open("w") as lock, cancellation_handlers():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with result_uploads(args.output):
                _cleanup(args.output)
        return
    config = json.loads(args.config.read_text())
    models = {alias: config["models"][alias] for alias in args.models}
    validate_models(models)
    sources = args.datasets or config.get("datasets")
    if not sources:
        parser.error("--datasets or config datasets is required")
    with dataset_paths(sources, environment=config.get("modal", {}).get("environment", "main")) as paths:
        rows = repeat_examples(load_examples(paths, args.limit), args.attempts)
    if args.dry_run:
        print(json.dumps(dry_run(rows, models), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock, cancellation_handlers():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with result_uploads(args.output):
            _execute(args, config, rows, models)


if __name__ == "__main__":
    main()
