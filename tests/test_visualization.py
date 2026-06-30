from pathlib import Path

from foresight_harness.cross_benchmark import run_cross_fold_benchmark
from foresight_harness.evaluator import load_replay_turns
from foresight_harness.visualization import render_benchmark_dashboard, write_benchmark_dashboard


def test_render_benchmark_dashboard_includes_headline_metrics():
    turns = load_replay_turns(Path("data/queueahead_enriched.jsonl"))
    report = run_cross_fold_benchmark(turns, fold_count=5, iterations=3, top_k=3)

    html = render_benchmark_dashboard(report)

    assert "<title>Premonition Benchmark</title>" in html
    assert "Overall p@1" in html
    assert "Environment p@1" in html
    assert "Weakest Segments" in html
    assert "Fold 1" in html
    assert "carrier_exception_hold" in html


def test_render_response_mode_dashboard_includes_quality_ready_recovery_panel():
    report = {
        "summary": {"train_turns": 10, "dev_turns": 5, "test_turns": 5, "top_k": 3},
        "background_recovery_calibration": {
            "min_quality_score": 0.75,
            "selected_policy": {"name": "recover_disclose_inform_other"},
        },
        "background_recovery_evaluation": {"promoted": True},
        "probability_pack_replay_baseline": {
            "prepared_hit_rate": 0.577,
            "semantic_prepared_hit_rate": 0.031,
            "quality_ready_rate": 0.546,
            "background_recovery_hit_rate": 0.0,
            "segments": {
                "expected_response_mode": {
                    "disclose": {"prepared_hit_rate": 0.0, "quality_ready_rate": 0.0},
                    "inform": {"prepared_hit_rate": 0.0, "quality_ready_rate": 0.0},
                }
            },
        },
        "probability_pack_replay": {
            "prepared_hit_rate": 0.765,
            "semantic_prepared_hit_rate": 0.0,
            "quality_ready_rate": 0.765,
            "background_recovery_hit_rate": 0.219,
            "segments": {
                "expected_response_mode": {
                    "disclose": {"prepared_hit_rate": 0.449, "quality_ready_rate": 0.449},
                    "inform": {"prepared_hit_rate": 0.739, "quality_ready_rate": 0.739},
                }
            },
        },
    }

    html = render_benchmark_dashboard(report)

    assert "<title>Premonition Response-Mode Recovery</title>" in html
    assert "Quality-Ready Recovery" in html
    assert "Raw Semantic Coverage" in html
    assert "recover_disclose_inform_other" in html
    assert "disclose" in html


def test_write_benchmark_dashboard_creates_static_html(tmp_path):
    turns = load_replay_turns(Path("data/queueahead_enriched.jsonl"))
    report = run_cross_fold_benchmark(turns, fold_count=5, iterations=3, top_k=3)
    output = tmp_path / "dashboard.html"

    write_benchmark_dashboard(report, output)

    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
