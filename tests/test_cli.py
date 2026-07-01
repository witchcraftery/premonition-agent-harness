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
    dashboard = tmp_path / "dashboard.html"

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
            "--dashboard-report",
            str(dashboard),
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
    assert dashboard.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_exports_empatheticdialogues_sample(tmp_path):
    input_path = tmp_path / "empathetic.csv"
    output = tmp_path / "empathetic.jsonl"
    input_path.write_text(
        "conv_id,utterance_idx,context,prompt,speaker_idx,utterance\n"
        "c1,1,grateful,A friend helped me.,0,My friend helped me today.\n"
        "c1,2,grateful,A friend helped me.,1,I would send them a thank you note.\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--empatheticdialogues-input",
            str(input_path),
            "--conversation-output",
            str(output),
            "--conversation-limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    exported = json.loads(output.read_text(encoding="utf-8"))

    assert report["exported_turns"] == 1
    assert exported["expected_act"] == "commissive"
    assert exported["expected_emotion"] == "happiness"


def test_cli_exports_esconv_sample(tmp_path):
    input_path = tmp_path / "ESConv.json"
    output = tmp_path / "esconv.jsonl"
    input_path.write_text(
        json.dumps(
            [
                {
                    "emotion_type": "sadness",
                    "problem_type": "family",
                    "dialog": [
                        {
                            "speaker": "seeker",
                            "annotation": {},
                            "content": "I miss my sister.",
                        },
                        {
                            "speaker": "supporter",
                            "annotation": {
                                "strategy": "Affirmation and Reassurance"
                            },
                            "content": "It makes sense that you miss her.",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--esconv-input",
            str(input_path),
            "--conversation-output",
            str(output),
            "--conversation-limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    exported = json.loads(output.read_text(encoding="utf-8"))

    assert report["exported_turns"] == 1
    assert exported["expected_response_mode"] == "reassure"


def test_cli_runs_response_mode_ranker_bakeoff(tmp_path):
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    test_path = tmp_path / "test.jsonl"
    output = tmp_path / "response-mode-bakeoff.json"
    rows = {
        train_path: [
            ("train-validate-1", "drainingcue heavy", "validate"),
            ("train-validate-2", "drainingcue overwhelming", "validate"),
            ("train-ask", "unclear details", "ask_followup"),
        ],
        dev_path: [
            ("dev-validate", "drainingcue difficult", "validate"),
            ("dev-ask", "unclear plan", "ask_followup"),
        ],
        test_path: [
            ("test-validate", "drainingcue hard", "validate"),
            ("test-ask", "unclear situation", "ask_followup"),
        ],
    }
    for path, split_rows in rows.items():
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "turn_id": turn_id,
                        "conversation": [
                            {"role": "speaker_a", "content": context}
                        ],
                        "next_speaker": "speaker_b",
                        "actual_next_utterance": "Okay.",
                        "expected_act": "inform",
                        "expected_emotion": "no_emotion",
                        "expected_response_mode": mode,
                    }
                )
                for turn_id, context, mode in split_rows
            )
            + "\n",
            encoding="utf-8",
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--conversation-train-input",
            str(train_path),
            "--conversation-dev-input",
            str(dev_path),
            "--conversation-test-input",
            str(test_path),
            "--response-mode-bakeoff-report",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == report
    assert report["selected_variant"]["test"]["p_at_1"] == 1.0


def test_cli_runs_response_mode_recovery_stress_report(tmp_path):
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    test_path = tmp_path / "test.jsonl"
    output = tmp_path / "response-mode-stress.json"
    rows = {
        train_path: [
            ("train-validate-1", "drainingcue heavy", "validate"),
            ("train-validate-2", "drainingcue overwhelming", "validate"),
            ("train-ask-1", "unclear details", "ask_followup"),
            ("train-ask-2", "unclear next step", "ask_followup"),
        ],
        dev_path: [
            ("dev-validate-1", "drainingcue difficult", "validate"),
            ("dev-ask-1", "unclear plan", "ask_followup"),
            ("dev-validate-2", "drainingcue hard", "validate"),
            ("dev-ask-2", "unclear situation", "ask_followup"),
        ],
        test_path: [
            ("test-validate-1", "drainingcue moment", "validate"),
            ("test-ask-1", "unclear question", "ask_followup"),
            ("test-validate-2", "drainingcue day", "validate"),
            ("test-ask-2", "unclear details", "ask_followup"),
        ],
    }
    for path, split_rows in rows.items():
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "turn_id": turn_id,
                        "conversation": [
                            {"role": "speaker_a", "content": context}
                        ],
                        "next_speaker": "speaker_b",
                        "actual_next_utterance": "Okay.",
                        "expected_act": "inform",
                        "expected_emotion": "no_emotion",
                        "expected_response_mode": mode,
                    }
                )
                for turn_id, context, mode in split_rows
            )
            + "\n",
            encoding="utf-8",
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--conversation-train-input",
            str(train_path),
            "--conversation-dev-input",
            str(dev_path),
            "--conversation-test-input",
            str(test_path),
            "--response-mode-stress-report",
            str(output),
            "--response-mode-stress-seeds",
            "1",
            "--folds",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == report
    assert report["summary"]["run_count"] == 3
    assert "prepared_hit_gain" in report["aggregates"]


def test_console_entrypoint_is_declared():
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    assert 'foresight-replay = "foresight_harness.cli:main"' in pyproject


def test_cli_parser_accepts_live_shadow_app_mode():
    from foresight_harness.cli import build_parser

    args = build_parser().parse_args(["--live-shadow-app", "--port", "8787"])

    assert args.live_shadow_app is True
    assert args.port == 8787
