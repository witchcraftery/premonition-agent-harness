from __future__ import annotations

from collections import Counter

from foresight_harness.models import Branch, ReplayTurn
from foresight_harness.similarity import normalized_tokens

INTENT_PATTERNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "refund_request": (
        "customer asks whether a refund or replacement is available",
        ("refund", "damaged", "delivery", "broken", "soaked", "replacement"),
    ),
    "troubleshooting_loop": (
        "customer reports the result of troubleshooting and asks for the next step",
        ("reset", "device", "pair", "light", "amber", "troubleshooting"),
    ),
    "billing_refund_timing": (
        "customer asks how long a duplicate charge refund will take",
        ("charged", "twice", "billing", "subscription", "duplicate", "refund"),
    ),
    "escalation_request": (
        "customer asks to escalate the unresolved case to a supervisor",
        ("third", "frustrating", "supervisor", "escalate", "unresolved"),
    ),
    "address_change": (
        "customer provides a new shipping address and asks if the order can be changed",
        ("address", "order", "shipped", "shipping", "change", "editable"),
    ),
    "shipment_status_update": (
        "warehouse status changes to shipment locked",
        ("warehouse", "status", "fulfillment", "shipment", "locked", "monitoring"),
    ),
}


def generate_branches(turn: ReplayTurn, top_k: int = 3, guidance=None) -> tuple[Branch, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    context_tokens = normalized_tokens(turn.context_text())
    scored: list[tuple[str, str, float]] = []

    for intent, (event, keywords) in INTENT_PATTERNS.items():
        guidance_keywords = guidance.keywords_for(intent) if guidance else tuple()
        effective_keywords = tuple(dict.fromkeys((*keywords, *guidance_keywords)))
        keyword_counts = Counter(
            keyword for keyword in effective_keywords if keyword in context_tokens
        )
        keyword_score = sum(keyword_counts.values()) / max(len(keywords), 1)
        prior = 0.18
        probability = min(0.85, prior + keyword_score)
        scored.append((intent, event, round(probability, 3)))

    ranked = sorted(scored, key=lambda item: item[2], reverse=True)[:top_k]
    return tuple(
        Branch(
            branch_id=f"{turn.turn_id}-br-{index}",
            predicted_event=event,
            intent=intent,
            probability=probability,
            rank=index,
        )
        for index, (intent, event, probability) in enumerate(ranked, start=1)
    )
