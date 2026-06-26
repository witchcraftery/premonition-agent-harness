from pathlib import Path

from foresight_harness.evaluator import load_replay_turns, run_replay


def test_load_replay_turns_from_jsonl():
    turns = load_replay_turns(Path("data/queueahead_sample.jsonl"))

    assert len(turns) == 5
    assert turns[0].turn_id == "qa-001"


def test_run_replay_reports_harness_metrics():
    turns = load_replay_turns(Path("data/queueahead_sample.jsonl"))
    report = run_replay(turns, top_k=3)

    harness = report["harness"]

    assert harness["total_turns"] == 5
    assert harness["top_3_recall"] >= 0.8
    assert harness["cache_hit_rate"] >= 0.8
    assert harness["stale_artifact_rate"] == 0.0
    assert "retrieval_plus_draft" in report


def test_semantic_cache_reports_cache_hits():
    turns = load_replay_turns(Path("data/queueahead_sample.jsonl"))
    report = run_replay(turns, top_k=3)

    assert report["semantic_cache"]["cache_hit_rate"] == 0.6


def test_empty_replay_keeps_report_schema():
    report = run_replay((), top_k=3)

    assert set(report) == {
        "harness",
        "live_agent",
        "prediction_only",
        "retrieval_plus_draft",
        "semantic_cache",
    }
    assert report["harness"]["total_turns"] == 0
    assert report["semantic_cache"]["cache_hit_rate"] == 0.0
