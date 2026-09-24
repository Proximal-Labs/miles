"""Prepare JSONL {messages, tools?, reasoning_effort?} for Inkling SFT, without GPUs.

Uses Thinking Machines' official TMLv0 renderer; retains reasoning and masks all
non-assistant tokens. Writes a longest-example smoke dataset and token statistics.
"""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from miles.rollout.inkling_sft import DEFAULT_EFFORT, load_renderer, render_example, renderer_provenance


def _prepare(source: Path, output: Path, checkpoint: str, max_length: int):
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must differ")
    if json.loads((Path(checkpoint) / "config.json").read_text()).get("model_type") != "inkling_mm_model":
        raise ValueError("Expected a downloaded Inkling checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer is None:
        raise ValueError("Checkpoint tokenizer could not be loaded")
    renderer = load_renderer()
    provenance = renderer_provenance()
    provenance["checkpoint_tokenizer_sha256"] = {
        name: hashlib.sha256((Path(checkpoint) / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    count = total = target_total = longest_length = 0
    longest = None
    with source.open() as src, temporary.open("w") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                effort = row.get("reasoning_effort", DEFAULT_EFFORT)
                tokens, mask = render_example(
                    renderer,
                    row["messages"],
                    row.get("tools", []),
                    max_length,
                    effort,
                )
                if tokenizer.encode(renderer.tokenizer.decode(tokens), add_special_tokens=False) != tokens:
                    raise ValueError("Official renderer token IDs do not round-trip through the checkpoint tokenizer")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            record = {
                "text": "prepared Inkling SFT example",
                "metadata": {
                    "format": "inkling-sft-v2",
                    "rendering": provenance,
                    "reasoning_effort": effort,
                    "tokens": tokens,
                    "loss_mask": mask,
                    "source_line": line_number,
                },
            }
            encoded = json.dumps(record, separators=(",", ":")) + "\n"
            dst.write(encoded)
            count += 1
            total += len(tokens)
            target_total += sum(mask)
            if len(tokens) > longest_length:
                longest_length, longest = len(tokens), encoded
    if longest is None:
        raise ValueError("Dataset is empty")
    temporary.replace(output)
    output.with_suffix(".smoke.jsonl").write_text(longest * 2)
    stats = {"examples": count, "tokens": total, "target_tokens": target_total, "max_tokens": longest_length, "configured_cap": max_length}
    output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-length", type=int, default=262144)
    args = parser.parse_args()
    _prepare(args.source, args.output, args.checkpoint, args.max_length)


if __name__ == "__main__":
    main()
