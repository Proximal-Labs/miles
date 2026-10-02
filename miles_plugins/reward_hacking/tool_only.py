"""Create a paired tool-call ablation without moving the original prefix cutoffs."""

import hashlib
import json

from miles_plugins.reward_hacking.curate import read_jsonl, write_json


def tool_only_row(row):
    messages = []
    for message in row["messages"]:
        if message["role"] in {"user", "system"}:
            messages.append({"role": message["role"], "content": message["content"]})
        elif message["role"] == "assistant" and message.get("tool_calls"):
            messages.append({"role": "assistant", "content": "", "tool_calls": message["tool_calls"]})
    metadata = {
        **row["metadata"],
        "input_variant": "tool_calls_only",
        "source_known_evidence_visible": row["metadata"].get("known_evidence_visible"),
        # Original annotations can point to removed reasoning, so do not reuse them as tool-only evidence.
        "known_evidence_visible": None,
        "source_reasoning_summary_blocks": row["metadata"].get("reasoning_summary_blocks"),
        "source_unavailable_reasoning_blocks": row["metadata"].get("unavailable_reasoning_blocks"),
        "reasoning_summary_blocks": 0,
        "unavailable_reasoning_blocks": 0,
    }
    return {**row, "messages": messages, "metadata": metadata}


def build_tool_only(source, output):
    if source.resolve() == output.resolve():
        raise ValueError("Tool-only output must differ from the source directory")
    files = sorted(source.glob("prefix-*.jsonl"))
    if not files:
        raise ValueError("No prefix datasets found")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "input_variant": "tool_calls_only",
        "source": str(source),
        "prefix_policy": "Preserve source cutoffs; remove reasoning and empty assistant turns afterward",
        "task_context": "Original task retained; no assistant prose or tool results",
        "evidence_policy": "Original visibility saved as source metadata; tool-only evidence visibility is unknown",
        "files": {},
    }
    for path in files:
        rows = [tool_only_row(row) for row in read_jsonl(path)]
        destination = output / path.name
        destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["files"][path.name] = {
            "examples": len(rows),
            "positive": sum(row["label"] for row in rows),
            "negative": sum(not row["label"] for row in rows),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    write_json(output / "manifest.json", manifest)
    return manifest
