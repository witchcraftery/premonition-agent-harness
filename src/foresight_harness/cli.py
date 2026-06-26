from __future__ import annotations

import argparse
import json
from pathlib import Path

from foresight_harness.evaluator import load_replay_turns, run_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Foresight replay harness against a JSONL replay file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/queueahead_sample.jsonl"),
        help="Path to a JSONL replay file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of next-event branches to generate per turn.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    turns = load_replay_turns(args.input)
    report = run_replay(turns, top_k=args.top_k)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
