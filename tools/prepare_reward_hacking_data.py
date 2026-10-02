"""Curate balanced Inkling-format reward-hacking monitor datasets, without GPUs.

Run from the checkout: python -m tools.prepare_reward_hacking_data --help.
select-controls scans a prior project audit; fetch-controls performs read-only
trace retrieval; build creates 10/30/50/100 percent prefixes and provenance.
"""

import argparse
from pathlib import Path

from miles_plugins.reward_hacking.curate import fetch_controls, select_controls
from miles_plugins.reward_hacking.data import build_datasets
from miles_plugins.reward_hacking.tool_only import build_tool_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select-controls")
    select.add_argument("--archive", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--seed", type=int, default=519)
    fetch = commands.add_parser("fetch-controls")
    fetch.add_argument("--candidates", type=Path, required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--cache", type=Path, required=True)
    fetch.add_argument("--count", type=int, default=22)
    fetch.add_argument("--base-url", required=True)
    fetch.add_argument("--session-file", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--positives", type=Path, required=True)
    build.add_argument("--controls", type=Path, required=True)
    build.add_argument("--trace-root", type=Path, action="append", required=True)
    build.add_argument("--highlights", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    tools = commands.add_parser("tool-only", help="Remove reasoning while preserving existing prefix boundaries")
    tools.add_argument("--source", type=Path, required=True)
    tools.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select-controls":
        result = select_controls(args.archive, args.output, seed=args.seed)
        print(f"Eligible environments: {result['eligible_environments']}")
    elif args.command == "fetch-controls":
        fetch_controls(
            args.candidates,
            args.output,
            args.cache,
            count=args.count,
            base_url=args.base_url,
            session_file=args.session_file,
        )
    elif args.command == "tool-only":
        print(build_tool_only(args.source, args.output)["files"])
    else:
        result = build_datasets(args.positives, args.controls, args.trace_root, args.highlights, args.output)
        print(result["files"])


if __name__ == "__main__":
    main()
