from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from foresight_harness.evaluator import load_replay_turns, run_replay, run_replay_turn_log
from foresight_harness.experiments import load_trial_config
from foresight_harness.learning import analyze_harness_misses


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def default_sample_path() -> Path:
    return Path(
        files("foresight_harness").joinpath("data", "queueahead_sample.jsonl")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Foresight replay harness against a JSONL replay file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_sample_path(),
        help="Path to a JSONL replay file. Defaults to the bundled QueueAhead sample.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=3,
        help="Number of next-event branches to generate per turn.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON trial config. Overrides --input and --top-k.",
    )
    parser.add_argument(
        "--turn-log",
        type=Path,
        help="Optional JSONL path for per-turn variant and branch outcomes.",
    )
    parser.add_argument(
        "--miss-report",
        type=Path,
        help="Optional JSON path for harness miss analysis.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input
    top_k = args.top_k
    if args.config:
        config = load_trial_config(args.config)
        input_path = config.input_path
        top_k = config.top_k

    turns = load_replay_turns(input_path)
    report = run_replay(turns, top_k=top_k)
    turn_log = run_replay_turn_log(turns, top_k=top_k)

    if args.turn_log:
        with args.turn_log.open("w", encoding="utf-8") as handle:
            for row in turn_log:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    if args.miss_report:
        with args.miss_report.open("w", encoding="utf-8") as handle:
            json.dump(analyze_harness_misses(turn_log), handle, indent=2, sort_keys=True)
            handle.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
