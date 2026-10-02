"""Select QA-negative controls and retrieve their recorded solver traces."""

import hashlib
import json
from pathlib import Path

import httpx


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _rank(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_controls(archive: Path, output: Path, *, seed: int):
    """Uniform seeded ordering of environments, then rollouts within each environment."""
    excluded = {r["environmentId"] for r in read_jsonl(archive / "report/trajectories.jsonl")}
    candidates = {}
    scanned = 0
    for report in read_jsonl(archive / "post_qa_reports.jsonl"):
        scanned += 1
        if scanned % 50000 == 0:
            print(f"Scanned {scanned} historical QA records; {len(candidates)} eligible environments", flush=True)
        env = report.get("environmentId")
        if env is None or env in excluded or report.get("status") != "completed" or report.get("scope") != "rollout":
            continue
        payload = report.get("report", {}).get("payload", {})
        if not isinstance(payload, dict):
            continue
        hacking = payload.get("reward_hacking", {})
        verification = payload.get("post_qa_verification", {})
        awareness = payload.get("evaluation_awareness", {})
        if (
            hacking.get("attempted") is not False
            or awareness.get("detected") is not False
            or payload.get("verdict") != "FAIR"
            or verification.get("verdict") != "confirmed"
        ):
            continue
        rid = report["rolloutId"]
        candidate = {
            "rollout_id": rid,
            "environment_id": env,
            "run_id": report["runId"],
            "label": False,
            "role": "solver_or_qualification",
            "qa": {
                "record_id": report["_id"],
                "collection": "post_qa_reports",
                "created_at": report.get("createdAt"),
                "model": report.get("model"),
                "outcome": "no_reward_hacking_detected",
                "description": hacking.get("detail", ""),
                "evaluation_awareness": awareness,
                "verification": verification,
            },
        }
        current = candidates.get(env)
        if current is None or _rank(seed, rid) < _rank(seed, current["rollout_id"]):
            candidates[env] = candidate
    ordered = sorted(candidates.values(), key=lambda c: _rank(seed, c["environment_id"]))
    result = {
        "seed": seed,
        "source_snapshot": (
            json.loads((archive / "manifest-mongo-final.json").read_text())
            if (archive / "manifest-mongo-final.json").exists()
            else None
        ),
        "qa_records_scanned": scanned,
        "excluded_environments_with_any_flagged_trajectory": len(excluded),
        "eligible_environments": len(ordered),
        "selection": "SHA256 seeded environment order; one SHA256 seeded rollout per environment; skip unavailable traces",
        "label_scope": "No recorded reward-hacking finding in the archived project audit; not proof of absence.",
        "candidates": ordered,
    }
    write_json(output, result)
    return result


def _rpc(client, service, method, body):
    response = client.post(f"/proximal.v1.{service}/{method}", json=body)
    response.raise_for_status()
    return response.json()


def _solver_runs(client, rollout_id):
    result = _rpc(
        client,
        "LlmRuntimeService",
        "ListAgentRuns",
        {
            "subjects": [{"type": "rollout", "id": rollout_id}],
            "scopeFamilies": ["rollout-solver"],
            "limit": 100,
        },
    )
    if result.get("hasMore"):
        raise ValueError("Ambiguous solver run pagination")
    return [r for r in result.get("runs", []) if r.get("kind") == "agent" and r.get("status") == "succeeded"]


def resolve_payloads(client, trace, cache):
    """Download full payload references used by the data converter; never use previews."""
    wanted = []
    for event in trace["run"]["events"]:
        if event["kind"] not in {"toolResult", "agentInput", "agentConfiguration", "systemPrompt"}:
            continue
        payload = event["event"]["payload"]
        wanted.extend(v["ref"] for v in payload.values() if isinstance(v, dict) and v.get("kind") == "payload_ref")
    for ref in wanted:
        sha = ref["sha256"]
        path = cache / f"tool-payload-{sha}.json"
        if path.exists():
            raw = path.read_bytes()
        else:
            result = _rpc(client, "PayloadDownloadService", "GetDownloadUrl", {"sha256": sha})
            # Presigned download must not receive the platform session header.
            response = httpx.get(result["url"], timeout=120)
            response.raise_for_status()
            raw = response.content
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Payload digest mismatch")
        path.write_bytes(raw)


def fetch_controls(candidates_path, output, cache, *, count, base_url, session_file):
    selection = json.loads(Path(candidates_path).read_text())
    cache.mkdir(parents=True, exist_ok=True)
    accepted, rejected = [], []
    with httpx.Client(
        base_url=base_url, headers={"x-session-id": session_file.read_text().strip()}, timeout=120
    ) as client:
        for candidate in selection["candidates"]:
            rid = candidate["rollout_id"]
            runs = _solver_runs(client, rid)
            if len(runs) != 1:
                rejected.append({"rollout_id": rid, "reason": f"Expected one completed solver; found {len(runs)}"})
                continue
            run_id = runs[0]["runId"]
            path = cache / f"agent-trace-{run_id}.json"
            trace = (
                json.loads(path.read_text())
                if path.exists()
                else _rpc(client, "LlmRuntimeService", "GetAgentRun", {"runId": run_id})
            )
            metadata = trace["run"]["run"]
            refs = [ref for m in metadata.get("scopeMemberships", []) for ref in m["scope"].get("subjectRefs", [])]
            if not any(r.get("type") == "rollout" and r.get("id") == rid for r in refs):
                raise ValueError("Trace scope does not match selected rollout")
            if not any(e["kind"] == "toolCall" for e in trace["run"]["events"]):
                rejected.append({"rollout_id": rid, "reason": "No tool calls"})
                continue
            write_json(path, trace)
            resolve_payloads(client, trace, cache)
            accepted.append(
                {
                    **candidate,
                    "agent_run_id": run_id,
                    "model": metadata.get("model"),
                    "trace_status": metadata["status"],
                }
            )
            write_json(
                output,
                {
                    "selection": {k: v for k, v in selection.items() if k != "candidates"},
                    "controls": accepted,
                    "rejected": rejected,
                },
            )
            print(f"Fetched control {len(accepted)}/{count}: environment {candidate['environment_id']}", flush=True)
            if len(accepted) == count:
                return
    raise ValueError(f"Only {len(accepted)} eligible traces were available, need {count}")
