"""Build paired cross-model additions and a combined, model-balanced monitor dataset.

Uses selected sources and downloaded traces; makes no inference or network calls.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from miles_plugins.reward_hacking.curate import write_json
from miles_plugins.reward_hacking.data import FRACTIONS, _example, _load_trace, trace_turns
from miles_plugins.reward_hacking.tool_only import build_tool_only


def _new_sources(path):
    fetched = json.loads(path.read_text())
    if fetched["rejected"]:
        raise ValueError("Resolve rejected candidates before building")
    sources = []
    for record in fetched["accepted"]:
        source = {
            key: record[key]
            for key in (
                "rollout_id",
                "run_id",
                "environment_id",
                "agent_run_id",
                "model",
                "role",
                "trace_status",
                "label",
            )
        }
        if record["label"]:
            source["qa"] = [
                {
                    "outcome": evidence["finding"].get("attempt_outcome", "attempted_unspecified"),
                    "description": evidence["finding"]["detail"],
                    "cause": evidence["finding"].get("cause"),
                    "finding": evidence["finding"],
                    "qa_sources": evidence["references"],
                }
                for evidence in record["qa"]
            ]
        else:
            source["qa"] = record["qa"]
        source["cohort"] = "cross-model-addition"
        sources.append(source)
    if len(sources) != 44 or sum(s["label"] for s in sources) != 22:
        raise ValueError("Expected 22 positives and 22 controls")
    if len({s["environment_id"] for s in sources}) != 44:
        raise ValueError("Additional environments must all be distinct")
    return sources


def _build(sources, roots, highlights, output):
    output.mkdir(parents=True, exist_ok=True)
    datasets = {fraction: [] for fraction in FRACTIONS}
    manifest = {
        "schema_version": 2,
        "project_id": 519,
        "positive_definition": "QA-supported reward-hacking attempt; successful reward increase is not required",
        "negative_definition": "Confirmed FAIR QA: attempted=false and evaluation_awareness=false; no flagged environment in archived audit",
        "fraction_unit": "complete assistant turns with reasoning and/or calls; ceil",
        "converter_note": "Includes google_genai_thought blocks omitted by the original converter",
        "sources": [],
        "files": {},
    }
    for source in sources:
        trace, digest = _load_trace(source, roots)
        task, turns = trace_turns(trace, roots)
        manifest["sources"].append({**source, "trace_sha256": digest, "total_turns": len(turns)})
        for fraction in FRACTIONS:
            row = _example(source, task, turns, fraction, highlights.get(source["rollout_id"], []))
            if source["label"] and source["rollout_id"] not in highlights:
                row["metadata"]["known_evidence_visible"] = None
            datasets[fraction].append(row)
    for fraction, rows in datasets.items():
        rows.sort(key=lambda r: hashlib.sha256(r["metadata"]["rollout_id"].encode()).hexdigest())
        path = output / f"prefix-{round(fraction * 100):03d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["files"][path.name] = {
            "examples": len(rows),
            "positive": sum(row["label"] for row in rows),
            "negative": sum(not row["label"] for row in rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest["model_counts"] = {
        label: dict(Counter(s["model"] for s in sources if s["label"] == value))
        for label, value in [("positive", True), ("negative", False)]
    }
    write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("data/reward-hacking-monitor"))
    parser.add_argument("--additional", type=Path, default=Path("data/reward-hacking-monitor-cross-model"))
    parser.add_argument("--combined", type=Path, default=Path("data/reward-hacking-monitor-balanced"))
    args = parser.parse_args()
    original = json.loads((args.original / "manifest.json").read_text())["sources"]
    original = [{**s, "cohort": "original"} for s in original]
    new = _new_sources(args.additional / "sources/fetched.json")
    if {s["environment_id"] for s in original} & {s["environment_id"] for s in new}:
        raise ValueError("New environments overlap the original set")
    if len({s["rollout_id"] for s in original + new}) != 88:
        raise ValueError("Duplicate rollout IDs")
    roots = [args.original / "raw", args.additional / "raw"]
    highlights = json.loads((args.original / "sources/reward-hacking-highlights.json").read_text())["rollouts"]
    for sources, output in [(new, args.additional), (original + new, args.combined)]:
        _build(sources, roots, highlights, output)
        build_tool_only(output, output.with_name(output.name + "-tools-only"))
        print(f"Built {len(sources)} paired rollouts at {output}")


if __name__ == "__main__":
    main()
