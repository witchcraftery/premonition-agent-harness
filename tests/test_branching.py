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


def test_inventory_profile_does_not_get_shadowed_by_carrier_hold_cues():
    inventory_hold = ReplayTurn.from_dict(
        {
            "turn_id": "inventory-hold-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "Inventory feed says allocated stock disappeared and the shipment is on hold.",
                },
                {"role": "agent", "content": "I am watching inventory service before promising delivery."},
            ],
            "actual_next_event": "inventory service changes order to backorder hold",
            "policy_context": "Backordered items cannot receive delivery promises until inventory is allocated.",
            "expected_intent": "shipment_status_update",
            "latency_budget_ms": 800,
        }
    )

    branch = generate_branches(inventory_hold)[0]

    assert branch.predicted_event == "inventory service changes order to backorder hold"


def test_address_change_is_not_shadowed_by_shipment_lock_decoys():
    not_locked = ReplayTurn.from_dict(
        {
            "turn_id": "not-locked-address-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "The order is going to the old apartment and fulfillment has not locked it.",
                },
                {"role": "agent", "content": "I am checking whether the destination can still be changed."},
            ],
            "actual_next_event": "customer provides a new address and asks if the order can be changed today",
            "policy_context": "Address changes are allowed before shipment if the fraud check has passed.",
            "expected_intent": "address_change",
            "latency_budget_ms": 800,
        }
    )
    cleared = ReplayTurn.from_dict(
        {
            "turn_id": "cleared-address-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "Fraud review cleared, and I need the order address changed before shipment.",
                },
                {"role": "agent", "content": "I am checking whether fulfillment still allows edits."},
            ],
            "actual_next_event": "customer provides a new address and asks if the order can be changed today",
            "policy_context": "Address changes are allowed before shipment if the fraud check has passed.",
            "expected_intent": "address_change",
            "latency_budget_ms": 800,
        }
    )
    old_place = ReplayTurn.from_dict(
        {
            "turn_id": "old-place-address-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "It is going to my old place and has not left the warehouse.",
                },
                {"role": "agent", "content": "I can check whether fulfillment has locked it yet."},
            ],
            "actual_next_event": "customer provides a new address and asks if the order can be changed today",
            "policy_context": "Address changes are allowed before shipment if the fraud check has passed.",
            "expected_intent": "address_change",
            "latency_budget_ms": 800,
        }
    )

    assert generate_branches(not_locked)[0].intent == "address_change"
    assert generate_branches(cleared)[0].intent == "address_change"
    assert generate_branches(old_place)[0].intent == "address_change"


def test_returns_portal_profile_predicts_environment_refund_update():
    returns_update = ReplayTurn.from_dict(
        {
            "turn_id": "returns-update-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "The returns portal marked the delivery damaged and replacement eligible.",
                },
                {"role": "agent", "content": "I am checking whether the refund path is available."},
            ],
            "actual_next_event": "returns portal changes damaged delivery to replacement eligible",
            "policy_context": "Damaged deliveries qualify after photo verification.",
            "expected_intent": "refund_request",
            "latency_budget_ms": 800,
        }
    )

    branch = generate_branches(returns_update)[0]

    assert branch.intent == "refund_request"
    assert branch.predicted_event == "returns portal changes damaged delivery to replacement eligible"


def test_targeted_environment_profiles_use_their_own_topic_intents():
    payment_update = ReplayTurn.from_dict(
        {
            "turn_id": "payment-update-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "Payment service shows duplicate charge settled and refund timing pending.",
                },
                {"role": "agent", "content": "I am checking billing records before quoting timing."},
            ],
            "actual_next_event": "payment gateway posts duplicate charge refund timing update",
            "policy_context": "Duplicate subscription charges can be refunded after transaction ID verification.",
            "expected_intent": "billing_refund_timing",
            "latency_budget_ms": 750,
        }
    )
    policy_update = ReplayTurn.from_dict(
        {
            "turn_id": "policy-update-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "The policy feed says photo verification is now required for damaged delivery refunds.",
                },
                {"role": "agent", "content": "I am checking the refund policy before I promise anything."},
            ],
            "actual_next_event": "policy feed changes damaged delivery refund to require photo verification",
            "policy_context": "Damaged deliveries qualify after photo verification.",
            "expected_intent": "refund_request",
            "latency_budget_ms": 800,
        }
    )
    fraud_lock = ReplayTurn.from_dict(
        {
            "turn_id": "fraud-lock-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "If fraud review locks the destination, do not tell me the address change is done.",
                },
                {"role": "agent", "content": "I am watching the account review status before changing the order."},
            ],
            "actual_next_event": "fraud review changes order address edit to locked",
            "policy_context": "Address changes are allowed before shipment if the fraud check has passed.",
            "expected_intent": "address_change",
            "latency_budget_ms": 800,
        }
    )

    assert generate_branches(payment_update)[0].predicted_event == (
        "payment gateway posts duplicate charge refund timing update"
    )
    assert generate_branches(payment_update)[0].intent == "billing_refund_timing"
    assert generate_branches(policy_update)[0].predicted_event == (
        "policy feed changes damaged delivery refund to require photo verification"
    )
    assert generate_branches(policy_update)[0].intent == "refund_request"
    assert generate_branches(fraud_lock)[0].predicted_event == (
        "fraud review changes order address edit to locked"
    )
    assert generate_branches(fraud_lock)[0].intent == "address_change"


def test_targeted_environment_profiles_respect_negative_cues():
    photo_uploaded = ReplayTurn.from_dict(
        {
            "turn_id": "photo-uploaded-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "The damaged delivery photo is uploaded and I need refund or replacement options.",
                },
                {"role": "agent", "content": "I am checking eligibility after verification."},
            ],
            "actual_next_event": "customer asks whether a refund or replacement is available",
            "policy_context": "Damaged deliveries qualify after photo verification.",
            "expected_intent": "refund_request",
            "latency_budget_ms": 800,
        }
    )
    fraud_cleared = ReplayTurn.from_dict(
        {
            "turn_id": "fraud-cleared-test",
            "conversation": [
                {
                    "role": "customer",
                    "content": "Fraud review cleared, and I need the order address changed before shipment.",
                },
                {"role": "agent", "content": "I am checking whether fulfillment still allows edits."},
            ],
            "actual_next_event": "customer provides a new address and asks if the order can be changed today",
            "policy_context": "Address changes are allowed before shipment if the fraud check has passed.",
            "expected_intent": "address_change",
            "latency_budget_ms": 800,
        }
    )

    assert "policy feed changes" not in generate_branches(photo_uploaded)[0].predicted_event
    assert "fraud review changes" not in generate_branches(fraud_cleared)[0].predicted_event


def test_prepare_and_select_artifact_for_actual_event():
    turn = make_turn()
    branches = generate_branches(turn, top_k=3)
    artifacts = prepare_artifacts(turn, branches)

    selected = select_artifact(turn, branches, artifacts, readiness_threshold=0.30)

    assert selected is not None
    assert selected.created_for_intent == "refund_request"
    assert "photo verification" in selected.response_draft.lower()
