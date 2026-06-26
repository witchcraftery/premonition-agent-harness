from foresight_harness.artifacts import prepare_artifacts, select_artifact
from foresight_harness.branching import generate_branches
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


def test_prepare_and_select_artifact_for_actual_event():
    turn = make_turn()
    branches = generate_branches(turn, top_k=3)
    artifacts = prepare_artifacts(turn, branches)

    selected = select_artifact(turn, branches, artifacts, readiness_threshold=0.30)

    assert selected is not None
    assert selected.created_for_intent == "refund_request"
    assert "photo verification" in selected.response_draft.lower()
