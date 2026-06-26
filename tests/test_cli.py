import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_outputs_json_report():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--input",
            "data/queueahead_sample.jsonl",
            "--top-k",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)

    assert "harness" in report
    assert report["harness"]["total_turns"] == 5


def test_cli_default_input_works_outside_repo_root(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}

    completed = subprocess.run(
        [sys.executable, "-m", "foresight_harness.cli", "--top-k", "3"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    report = json.loads(completed.stdout)

    assert report["harness"]["total_turns"] == 5


def test_cli_rejects_non_positive_top_k():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--input",
            "data/queueahead_sample.jsonl",
            "--top-k",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "positive integer" in completed.stderr


def test_cli_writes_turn_log_and_miss_report(tmp_path):
    turn_log = tmp_path / "turn-log.jsonl"
    miss_report = tmp_path / "miss-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--config",
            "experiments/queueahead_v1.json",
            "--turn-log",
            str(turn_log),
            "--miss-report",
            str(miss_report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    miss_summary = json.loads(miss_report.read_text(encoding="utf-8"))

    assert report["harness"]["total_turns"] == 5
    assert turn_log.read_text(encoding="utf-8").count("\n") == 25
    assert miss_summary["harness_turns"] == 5


def test_cli_runs_guidance_loop(tmp_path):
    loop_report = tmp_path / "loop-report.json"
    guidance_markdown = tmp_path / "guidance.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--config",
            "experiments/queueahead_challenge_loop.json",
            "--iterations",
            "3",
            "--loop-report",
            str(loop_report),
            "--guidance-markdown",
            str(guidance_markdown),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    loop_summary = json.loads(loop_report.read_text(encoding="utf-8"))

    assert report["iterations"][-1]["report"]["harness"]["p_at_1"] >= 0.5
    assert loop_summary == report
    assert guidance_markdown.read_text(encoding="utf-8").startswith("# Premonition Guidance")


def test_cli_runs_split_benchmark(tmp_path):
    output = tmp_path / "split-benchmark.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--train-config",
            "experiments/queueahead_challenge_train.json",
            "--test-config",
            "experiments/queueahead_challenge_test.json",
            "--iterations",
            "3",
            "--benchmark-report",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == report
    assert report["generalization"]["test_p_at_1_gain"] > 0
    assert report["promote_guidance"] is True


def test_cli_runs_cross_fold_benchmark(tmp_path):
    output = tmp_path / "cross-benchmark.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--fold-config",
            "experiments/queueahead_enriched_folds.json",
            "--folds",
            "5",
            "--iterations",
            "3",
            "--benchmark-report",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == report
    assert report["summary"]["fold_count"] == 5
    assert report["aggregates"]["test"]["harness"]["p_at_1"]["guided_mean"] > 0
    assert report["weak_segments"]


def test_console_entrypoint_is_declared():
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    assert 'foresight-replay = "foresight_harness.cli:main"' in pyproject
