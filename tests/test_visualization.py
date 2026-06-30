from pathlib import Path

from foresight_harness.cross_benchmark import run_cross_fold_benchmark
from foresight_harness.evaluator import load_replay_turns
from foresight_harness.visualization import (
    render_benchmark_dashboard,
    render_premonition_outcome_dashboard,
    write_benchmark_dashboard,
)


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
        "probability_pack_replay_baseline_quality_aware": {
            "prepared_hit_rate": 0.546,
            "semantic_prepared_hit_rate": 0.0,
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
    assert "Quality-Aware Gate" in html
    assert "0.546 -> 0.765" in html
    assert "recover_disclose_inform_other" in html
    assert "disclose" in html


def test_render_premonition_outcome_dashboard_compares_base_and_swarm():
    bakeoff_report = {
        "summary": {"test_turns": 1748, "top_k": 3},
        "probability_pack_replay_baseline": {
            "first_speech_hit_rate": 0.217,
            "prepared_hit_rate": 0.577,
            "quality_ready_rate": 0.546,
            "background_hit_rate": 0.360,
            "average_quality_score": 0.974,
            "median_latency_saved_ms": 560,
            "segments": {
                "expected_response_mode": {
                    "disclose": {"prepared_hit_rate": 0.0},
                    "inform": {"prepared_hit_rate": 0.0},
                    "other": {"prepared_hit_rate": 0.0},
                }
            },
        },
        "probability_pack_replay": {
            "first_speech_hit_rate": 0.217,
            "prepared_hit_rate": 0.765,
            "quality_ready_rate": 0.765,
            "background_hit_rate": 0.547,
            "background_recovery_hit_rate": 0.219,
            "average_quality_score": 1.0,
            "median_latency_saved_ms": 560,
            "segments": {
                "expected_response_mode": {
                    "disclose": {"prepared_hit_rate": 0.449},
                    "inform": {"prepared_hit_rate": 0.739},
                    "other": {"prepared_hit_rate": 0.723},
                }
            },
        },
        "background_recovery_evaluation": {
            "promoted": True,
            "raw_prepared_hit_gain": 0.188,
            "quality_ready_gain": 0.219,
            "prepared_hit_floor_met": True,
            "target_mode_results": {
                "disclose": {"prepared_hit_gain": 0.449},
                "inform": {"prepared_hit_gain": 0.739},
                "other": {"prepared_hit_gain": 0.723},
            },
        },
    }
    stress_report = {
        "summary": {"seed_count": 3, "fold_count": 5, "run_count": 15},
        "aggregates": {
            "promotion_rate": 0.867,
            "prepared_hit_gain": {"mean": 0.032, "min": 0.0, "max": 0.132},
            "quality_ready_gain": {"mean": 0.068, "min": 0.0, "max": 0.164},
            "background_recovery_hit_rate": {"mean": 0.068, "min": 0.0, "max": 0.164},
            "selected_policy_counts": {
                "recover_disclose_inform": 8,
                "recover_inform": 6,
                "recover_other": 1,
            },
        },
    }

    html = render_premonition_outcome_dashboard(bakeoff_report, stress_report)

    assert "<title>Premonition Swarm Outcome</title>" in html
    assert "Base State" in html
    assert "Current Guarded Swarm" in html
    assert "0.217 -> 0.765" in html
    assert "13 / 15" in html
    assert "recover_inform" in html
    assert "Not prophecy" in html


def test_write_benchmark_dashboard_creates_static_html(tmp_path):
    turns = load_replay_turns(Path("data/queueahead_enriched.jsonl"))
    report = run_cross_fold_benchmark(turns, fold_count=5, iterations=3, top_k=3)
    output = tmp_path / "dashboard.html"

    write_benchmark_dashboard(report, output)

    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
