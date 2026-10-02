"""Convert recorded model reasoning and tool calls to the Inkling SFT message schema."""

import hashlib
import json
import math
from pathlib import Path

from miles_plugins.reward_hacking.curate import write_json

FRACTIONS = (0.1, 0.3, 0.5, 1.0)


def _payload(value, roots):
    if value.get("kind") == "inline_json":
        return value["value"]
    if value.get("kind") == "payload_ref":
        sha = value["ref"]["sha256"]
        for root in roots:
            path = root / f"tool-payload-{sha}.json"
            if path.exists():
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != sha:
                    raise ValueError(f"Corrupt payload {sha}")
                return json.loads(raw)
        raise FileNotFoundError(f"Missing full payload {sha}; previews are not accepted")
    raise ValueError("Unknown payload encoding")


def _reasoning(block):
    kind = block.get("kind")
    if kind == "anthropic_thinking":
        return block.get("thinking", ""), "reasoning"
    if kind == "chat_completions_reasoning":
        return block.get("reasoning", ""), "reasoning"
    if kind == "openai_responses_reasoning":
        summary = "\n\n".join(p.get("text", "") for p in block.get("summary", []))
        content = "\n\n".join(p.get("text", "") for p in block.get("content", []))
        return summary or content, "summary" if summary else "reasoning"
    return None, None


def _task(events, roots):
    for event in events:
        if event["kind"] == "agentInput":
            value = _payload(event["event"]["payload"]["input"], roots)
            for key in ("instruction", "prompt", "task"):
                if isinstance(value.get(key), str) and value[key]:
                    return value[key]
    raise ValueError("No original task instruction available")


def trace_turns(trace, roots):
    """Use actual toolCall events once; provider responses supply readable reasoning only.

    Assistant prose, results, QA reports and encrypted reasoning are not model
    inputs. An entire provider response and its tool calls form one atomic turn.
    """
    events = sorted(trace["run"]["events"], key=lambda e: e["sequence"])
    turns, current = [], None
    for event in events:
        payload = event["event"]["payload"]
        if event["kind"] == "providerResponse":
            if current is not None and (current["reasoning"] or current["calls"]):
                turns.append(current)
            current = {
                "sequence": event["sequence"],
                "last_sequence": event["sequence"],
                "reasoning": [],
                "calls": [],
                "unavailable": 0,
                "summaries": 0,
            }
            for block in payload.get("assistantMessage", {}).get("content", []):
                text, kind = _reasoning(block)
                if text is None:
                    continue
                if text:
                    current["reasoning"].append(text)
                    current["summaries"] += kind == "summary"
                else:
                    current["unavailable"] += 1
        elif event["kind"] == "toolCall":
            if current is None:
                raise ValueError("Tool call precedes the first provider response")
            name = event.get("toolName") or event["event"].get("toolName")
            arguments = payload["arguments"]
            if not name or not isinstance(arguments, dict):
                raise ValueError("Tool calls require a name and object arguments")
            current["calls"].append(
                {"id": payload["toolCallId"], "type": "function", "function": {"name": name, "arguments": arguments}}
            )
            current["last_sequence"] = event["sequence"]
    if current is not None and (current["reasoning"] or current["calls"]):
        turns.append(current)
    if not turns or not any(t["calls"] for t in turns):
        raise ValueError("Trace has no usable assistant/tool turns")
    return _task(events, roots), turns


def _example(source, task, turns, fraction, highlights):
    count = max(1, math.ceil(len(turns) * fraction))
    selected = turns[:count]
    messages = [{"role": "user", "content": task}]
    for turn in selected:
        message = {"role": "assistant", "content": ""}
        if turn["reasoning"]:
            message["reasoning_content"] = "\n\n".join(turn["reasoning"])
        if turn["calls"]:
            message["tool_calls"] = turn["calls"]
        messages.append(message)
    cutoff = selected[-1]["last_sequence"]
    visible = any(h["sequence"] <= cutoff for h in highlights)
    return {
        "id": hashlib.sha256(f"{source['rollout_id']}:{fraction}".encode()).hexdigest()[:24],
        "messages": messages,
        "tools": [],
        "reasoning_effort": 0.99,
        "label": source["label"],
        "metadata": {
            **source,
            "fraction": fraction,
            "turns_included": count,
            "total_turns": len(turns),
            "last_sequence": cutoff,
            "tool_calls_included": sum(len(t["calls"]) for t in selected),
            "reasoning_summary_blocks": sum(t["summaries"] for t in selected),
            "unavailable_reasoning_blocks": sum(t["unavailable"] for t in selected),
            "known_evidence_visible": visible if source["label"] else False,
            "prefix_label_note": "Label describes full rollout. No visible annotated evidence does not establish a clean prefix.",
        },
    }


def _load_trace(source, roots):
    for root in roots:
        path = root / f"agent-trace-{source['agent_run_id']}.json"
        if path.exists():
            raw = path.read_bytes()
            trace = json.loads(raw)
            metadata = trace["run"]["run"]
            refs = [r for m in metadata.get("scopeMemberships", []) for r in m["scope"].get("subjectRefs", [])]
            if not any(r.get("type") == "rollout" and r.get("id") == source["rollout_id"] for r in refs):
                raise ValueError("Source trace does not belong to selected rollout")
            return trace, hashlib.sha256(raw).hexdigest()
    raise FileNotFoundError(source["agent_run_id"])


def build_datasets(positives, controls, roots, highlights_path, output):
    positives = json.loads(Path(positives).read_text())
    controls = json.loads(Path(controls).read_text())["controls"]
    if len(positives) != 22 or len(controls) != 22 or len({r["environment_id"] for r in controls}) != 22:
        raise ValueError("Expected 22 positives and 22 controls from distinct environments")
    positive_envs = {r["environment_id"] for r in positives}
    if positive_envs & {r["environment_id"] for r in controls}:
        raise ValueError("Positive and control environments overlap")
    sources = [
        {
            **{k: r[k] for k in ("rollout_id", "environment_id", "agent_run_id", "model", "role", "trace_status")},
            "label": True,
            "qa": r["qa_findings"],
        }
        for r in positives
    ] + controls
    if len({r["rollout_id"] for r in sources}) != 44:
        raise ValueError("Duplicate rollout IDs")
    highlights = json.loads(Path(highlights_path).read_text())["rollouts"]
    datasets = {f: [] for f in FRACTIONS}
    manifest = {
        "schema_version": 1,
        "fraction_unit": "complete assistant turns with reasoning and/or calls; ceil",
        "sources": [],
        "files": {},
    }
    for source in sources:
        trace, digest = _load_trace(source, roots)
        task, turns = trace_turns(trace, roots)
        manifest["sources"].append({**source, "trace_sha256": digest, "total_turns": len(turns)})
        for fraction in FRACTIONS:
            datasets[fraction].append(
                _example(source, task, turns, fraction, highlights.get(source["rollout_id"], []))
            )
    output.mkdir(parents=True, exist_ok=True)
    for fraction, rows in datasets.items():
        # Shuffle independently of the class while retaining the same rollout order across prefixes.
        rows.sort(key=lambda r: hashlib.sha256(r["metadata"]["rollout_id"].encode()).hexdigest())
        path = output / f"prefix-{round(fraction * 100):03d}.jsonl"
        raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        path.write_bytes(raw)
        manifest["files"][path.name] = {
            "examples": len(rows),
            "positive": 22,
            "negative": 22,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    write_json(output / "manifest.json", manifest)
    return manifest
