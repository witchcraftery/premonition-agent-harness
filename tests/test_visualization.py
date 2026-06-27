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


def test_write_benchmark_dashboard_creates_static_html(tmp_path):
    turns = load_replay_turns(Path("data/queueahead_enriched.jsonl"))
    report = run_cross_fold_benchmark(turns, fold_count=5, iterations=3, top_k=3)
    output = tmp_path / "dashboard.html"

    write_benchmark_dashboard(report, output)

    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
