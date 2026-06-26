from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from foresight_harness.evaluator import load_replay_turns, run_replay, run_replay_turn_log
from foresight_harness.experiments import load_trial_config
from foresight_harness.guidance import run_guidance_loop
from foresight_harness.learning import analyze_harness_misses
from foresight_harness.split_benchmark import run_split_benchmark


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
        "--train-config",
        type=Path,
        help="Path to a JSON train trial config for split benchmark mode.",
    )
    parser.add_argument(
        "--test-config",
        type=Path,
        help="Path to a JSON test trial config for split benchmark mode.",
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
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=1,
        help="Number of guidance loop iterations to run.",
    )
    parser.add_argument(
        "--loop-report",
        type=Path,
        help="Optional JSON path for the full guidance loop report.",
    )
    parser.add_argument(
        "--guidance-markdown",
        type=Path,
        help="Optional Markdown path for learned guidance notes.",
    )
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        help="Optional JSON path for train/test split benchmark output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.train_config or args.test_config:
        if not (args.train_config and args.test_config):
            raise SystemExit("--train-config and --test-config must be provided together")
        train_config = load_trial_config(args.train_config)
        test_config = load_trial_config(args.test_config)
        train_turns = load_replay_turns(train_config.input_path)
        test_turns = load_replay_turns(test_config.input_path)
        top_k = train_config.top_k
        report = run_split_benchmark(
            train_turns=train_turns,
            test_turns=test_turns,
            iterations=args.iterations,
            top_k=top_k,
        )
        if args.benchmark_report:
            with args.benchmark_report.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    input_path = args.input
    top_k = args.top_k
    if args.config:
        config = load_trial_config(args.config)
        input_path = config.input_path
        top_k = config.top_k

    turns = load_replay_turns(input_path)
    if args.iterations > 1:
        loop_report = run_guidance_loop(turns, iterations=args.iterations, top_k=top_k)
        if args.loop_report:
            with args.loop_report.open("w", encoding="utf-8") as handle:
                json.dump(loop_report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        if args.guidance_markdown:
            args.guidance_markdown.write_text(
                str(loop_report["guidance_markdown"]),
                encoding="utf-8",
            )
        print(json.dumps(loop_report, indent=2, sort_keys=True))
        return

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
