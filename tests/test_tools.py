from foresight_harness.models import PremonitionPacket, ReplayTurn
from foresight_harness.tools import build_premonition_packet


def test_build_premonition_packet_selects_prepared_artifact():
    turn = ReplayTurn.from_dict(
        {
            "turn_id": "qa-test",
            "conversation": [
                {"role": "customer", "content": "The delivery arrived damaged."},
                {"role": "agent", "content": "I can help with the damaged delivery."},
            ],
            "actual_next_event": "customer asks whether a refund is available",
            "policy_context": "Damaged deliveries qualify after photo verification.",
            "expected_intent": "refund_request",
            "latency_budget_ms": 800,
        }
    )

    packet = build_premonition_packet(turn, top_k=3)

    assert isinstance(packet, PremonitionPacket)
    assert packet.matched_intent == "refund_request"
    assert packet.prepared_artifact is not None
    assert packet.freshness == "valid"
    assert packet.unsafe is False
