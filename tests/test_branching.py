from foresight_harness.artifacts import prepare_artifacts, select_artifact
from foresight_harness.branching import generate_branches
from foresight_harness.guidance import Guidance
from foresight_harness.models import ReplayTurn


def make_turn(expected_intent: str = "refund_request") -> ReplayTurn:
    return ReplayTurn.from_dict(
        {
            "turn_id": "qa-test",
            "conversation": [
                {"role": "customer", "content": "The box arrived soaked and broken."},
                {"role": "agent", "content": "I can help with the damaged delivery."},
            ],
            "actual_next_event": "customer asks whether a refund is available",
            "policy_context": "Damaged deliveries qualify after photo verification.",
            "expected_intent": expected_intent,
            "latency_budget_ms": 800,
        }
    )


def test_generate_branches_returns_ranked_top_k():
    branches = generate_branches(make_turn(), top_k=3)

    assert len(branches) == 3
    assert branches[0].rank == 1
    assert branches[0].probability >= branches[1].probability


def test_generate_branches_rejects_non_positive_top_k():
    import pytest

    with pytest.raises(ValueError, match="top_k must be positive"):
        generate_branches(make_turn(), top_k=0)


def test_generate_branches_predicts_environment_fulfillment_event():
    turn = ReplayTurn.from_dict(
        {
            "turn_id": "env-test",
            "conversation": [
                {"role": "customer", "content": "Can you watch the warehouse status?"},
                {"role": "agent", "content": "I am monitoring fulfillment now."},
            ],
            "actual_next_event": "warehouse status changes to shipment locked",
            "policy_context": "Orders cannot be edited after fulfillment locks shipment.",
            "expected_intent": "shipment_status_update",
            "latency_budget_ms": 800,
        }
    )

    branches = generate_branches(turn, top_k=3)

    assert branches[0].intent == "shipment_status_update"


def test_environment_guidance_expands_only_when_cues_cluster():
    guidance = Guidance(
        intent_keywords={
            "shipment_status_update": (
                "carrier",
                "delivery",
                "exception",
                "feed",
                "hold",
                "keep",
            )
        }
    )
    weak_turn = ReplayTurn.from_dict(
        {
            "turn_id": "weak-env-test",
            "conversation": [
                {"role": "customer", "content": "Please keep watching the warehouse."},
                {"role": "agent", "content": "I am monitoring fulfillment."},
            ],
            "actual_next_event": "warehouse status changes to shipment locked",
            "policy_context": "Orders cannot be edited after fulfillment locks shipment.",
            "expected_intent": "shipment_status_update",
            "latency_budget_ms": 800,
        }
    )
    hard_turn = ReplayTurn.from_dict(
        {
            "turn_id": "hard-env-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "If the carrier feed shows an exception, keep watching.",
                },
                {"role": "agent", "content": "I am monitoring the carrier feed."},
            ],
            "actual_next_event": "carrier feed changes to delivery exception hold",
            "policy_context": "Orders cannot be edited during carrier exception holds.",
            "expected_intent": "shipment_status_update",
            "latency_budget_ms": 800,
        }
    )

    weak_branch = generate_branches(weak_turn, guidance=guidance)[0]
    hard_branch = generate_branches(hard_turn, guidance=guidance)[0]

    assert "delivery exception hold" not in weak_branch.predicted_event
    assert "delivery" in hard_branch.predicted_event
    assert "exception" in hard_branch.predicted_event
    assert "hold" in hard_branch.predicted_event


def test_environment_profile_branching_uses_positive_and_negative_cues():
    carrier_hold = ReplayTurn.from_dict(
        {
            "turn_id": "carrier-hold-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "The carrier feed shows an exception hold before reroute.",
                },
                {"role": "agent", "content": "I am watching the logistics stream."},
            ],
            "actual_next_event": "carrier feed changes to delivery exception hold",
            "policy_context": "Orders on carrier exception hold cannot be edited.",
            "expected_intent": "shipment_status_update",
            "latency_budget_ms": 800,
        }
    )
    carrier_clear = ReplayTurn.from_dict(
        {
            "turn_id": "carrier-clear-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "The carrier feed is clear, so I want to change the destination.",
                },
                {"role": "agent", "content": "I am checking address eligibility."},
            ],
            "actual_next_event": "customer provides a new address and asks if the order can be changed today",
            "policy_context": "Address changes are allowed before shipment if fraud check has passed.",
            "expected_intent": "address_change",
            "latency_budget_ms": 800,
        }
    )

    hold_branch = generate_branches(carrier_hold)[0]
    clear_branches = generate_branches(carrier_clear)

    assert hold_branch.predicted_event == "carrier feed changes to delivery exception hold"
    assert all(
        "delivery exception hold" not in branch.predicted_event
        for branch in clear_branches
    )


def test_prepare_and_select_artifact_for_actual_event():
    turn = make_turn()
    branches = generate_branches(turn, top_k=3)
    artifacts = prepare_artifacts(turn, branches)

    selected = select_artifact(turn, branches, artifacts, readiness_threshold=0.30)

    assert selected is not None
    assert selected.created_for_intent == "refund_request"
    assert "photo verification" in selected.response_draft.lower()
