from foresight_harness.models import Branch, MatchGrade, ReplayTurn


def test_replay_turn_from_json():
    row = {
        "turn_id": "support-001",
        "conversation": [
            {"role": "customer", "content": "My delivery arrived damaged."},
            {"role": "agent", "content": "I can help with that."},
        ],
        "actual_next_event": "customer asks whether a refund is available",
        "policy_context": "Damaged items qualify for refund or replacement after photo verification.",
        "expected_intent": "refund_request",
        "latency_budget_ms": 800,
    }

    turn = ReplayTurn.from_dict(row)

    assert turn.turn_id == "support-001"
    assert turn.expected_intent == "refund_request"
    assert turn.context_text().startswith("customer: My delivery")


def test_branch_defaults():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks for a refund",
        intent="refund_request",
        probability=0.62,
        rank=1,
    )

    assert branch.match_grade == MatchGrade.UNSCORED
