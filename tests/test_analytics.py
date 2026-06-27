from pathlib import Path

from foresight_harness.analytics import classify_event, summarize_segments
from foresight_harness.evaluator import load_replay_turns, run_replay_turn_log
from foresight_harness.guidance import Guidance


def test_classify_event_identifies_actor_type_and_topic():
    turn = load_replay_turns(Path("data/queueahead_challenge_test.jsonl"))[1]

    metadata = classify_event(turn)

    assert metadata == {
        "actor": "user",
        "event_type": "account_update",
        "profile": "address_change",
        "topic": "address_change",
    }


def test_classify_event_identifies_environment_event():
    turn = load_replay_turns(Path("data/queueahead_challenge_test.jsonl"))[-1]

    metadata = classify_event(turn)

    assert metadata == {
        "actor": "environment",
        "event_type": "fulfillment",
        "profile": "carrier_exception_hold",
        "topic": "shipment_status_update",
    }


def test_turn_log_includes_event_metadata():
    turns = load_replay_turns(Path("data/queueahead_challenge_test.jsonl"))

    row = run_replay_turn_log(turns)[0]

    assert row["actor"] == "user"
    assert row["event_type"] == "escalation"
    assert row["profile"] == "escalation_request"
    assert row["topic"] == "escalation_request"


def test_classify_event_identifies_environment_profiles():
    turns = load_replay_turns(Path("data/queueahead_enriched.jsonl"))
    profiles = {
        turn.turn_id: classify_event(turn)["profile"]
        for turn in turns
    }

    assert profiles["qe-006"] == "carrier_exception_hold"
    assert profiles["qe-010"] == "inventory_backorder"
    assert profiles["qe-014"] == "fraud_review_lock"
    assert profiles["qe-016"] == "policy_update"
    assert profiles["qe-025"] == "payment_gateway_update"


def test_summarize_segments_reports_topic_and_actor_performance():
    turns = load_replay_turns(Path("data/queueahead_challenge_test.jsonl"))
    guidance = Guidance(
        intent_keywords={
            "escalation_request": ("bounced", "case", "supervisor"),
            "address_change": ("address", "fulfillment", "order"),
            "billing_refund_timing": ("billing", "card", "duplicate", "refund"),
            "shipment_status_update": (
                "carrier",
                "exception",
                "fulfillment",
                "hold",
                "locked",
                "shipment",
                "warehouse",
            ),
            "troubleshooting_loop": ("connect", "recovery", "speaker", "step"),
        }
    )
    baseline_rows = run_replay_turn_log(turns)
    guided_rows = run_replay_turn_log(turns, guidance=guidance)

    summary = summarize_segments(baseline_rows, guided_rows)

    assert summary["by_actor"]["user"]["baseline"]["p_at_1"] >= 0.5
    assert summary["by_actor"]["user"]["guided"]["p_at_1"] == 1.0
    assert summary["by_actor"]["environment"]["guided"]["p_at_1"] == 1.0
    assert summary["by_actor"]["environment"]["delta"]["p_at_1"] >= 0
    assert summary["by_topic"]["address_change"]["guided"]["usefulness_rate"] == 1.0
    assert summary["focus_areas"][0]["segment"] in {"by_topic", "by_event_type", "by_actor"}
