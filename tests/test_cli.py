import json
import subprocess
import sys


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


def test_console_entrypoint_is_declared():
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    assert 'foresight-replay = "foresight_harness.cli:main"' in pyproject
