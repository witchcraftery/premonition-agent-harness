from pathlib import Path

from foresight_harness.analytics import classify_event
from foresight_harness.cross_benchmark import run_cross_fold_benchmark
from foresight_harness.evaluator import load_replay_turns


def test_enriched_dataset_includes_hard_events_and_decoys():
    turns = load_replay_turns(Path("data/queueahead_enriched.jsonl"))
    actors = {classify_event(turn)["actor"] for turn in turns}
    event_types = {classify_event(turn)["event_type"] for turn in turns}

    assert len(turns) >= 20
    assert actors == {"environment", "user"}
    assert "fulfillment" in event_types
    assert "billing" in event_types
    assert any("decoy" in turn.turn_id for turn in turns)


def test_cross_fold_benchmark_reports_dev_gate_and_aggregate_results():
    turns = load_replay_turns(Path("data/queueahead_enriched.jsonl"))

    report = run_cross_fold_benchmark(
        turns=turns,
        fold_count=5,
        iterations=3,
        top_k=3,
    )

    assert report["summary"]["fold_count"] == 5
    assert report["summary"]["total_turns"] == len(turns)
    assert len(report["folds"]) == 5
    assert report["aggregates"]["promotion_rate"] >= 0.6
    assert all("guidance_delta" in fold for fold in report["folds"])
    assert all(fold["test"]["guided"]["harness"]["total_turns"] > 0 for fold in report["folds"])
    assert all(fold["dev"]["guided"]["harness"]["total_turns"] > 0 for fold in report["folds"])

    test_metrics = report["aggregates"]["test"]["harness"]
    assert test_metrics["p_at_1"]["guided_mean"] >= test_metrics["p_at_1"]["baseline_mean"]
    assert test_metrics["usefulness_rate"]["guided_mean"] >= test_metrics["usefulness_rate"]["baseline_mean"]

    environment = report["aggregates"]["test_segments"]["by_actor"]["environment"]
    assert environment["p_at_1"]["guided_mean"] >= environment["p_at_1"]["baseline_mean"]
    assert environment["usefulness_rate"]["guided_mean"] >= environment["usefulness_rate"]["baseline_mean"]
    user = report["aggregates"]["test_segments"]["by_actor"]["user"]
    assert user["p_at_1"]["guided_mean"] >= user["p_at_1"]["baseline_mean"]
    assert user["usefulness_rate"]["guided_mean"] >= user["usefulness_rate"]["baseline_mean"]
    assert "by_profile" in report["aggregates"]["test_segments"]
    assert report["aggregates"]["test_segments"]["by_profile"]["carrier_exception_hold"]["p_at_1"]["guided_mean"] == 1.0
    assert report["aggregates"]["test_segments"]["by_profile"]["payment_gateway_update"]["p_at_1"]["guided_mean"] == 1.0
    assert report["aggregates"]["test_segments"]["by_profile"]["policy_update"]["p_at_1"]["guided_mean"] == 1.0
    assert report["aggregates"]["test_segments"]["by_profile"]["fraud_review_lock"]["p_at_1"]["guided_mean"] == 1.0
    assert report["aggregates"]["test_segments"]["by_profile"]["inventory_backorder"]["p_at_1"]["guided_mean"] == 1.0
    assert report["aggregates"]["test_segments"]["by_profile"]["refund_request"]["p_at_1"]["guided_mean"] == 1.0
    assert report["aggregates"]["test_segments"]["by_profile"]["address_change"]["p_at_1"]["guided_mean"] == 1.0
    assert environment["p_at_1"]["guided_mean"] == 1.0
    assert report["guidance_delta_summary"]["regressed_turn_count"] == 0
    assert report["weak_segments"]
