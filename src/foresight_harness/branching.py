from __future__ import annotations

from collections import Counter

from foresight_harness.models import Branch, ReplayTurn
from foresight_harness.similarity import normalized_tokens

MIN_CLUSTERED_GUIDANCE_CUES = 3
ENVIRONMENT_PROFILE_PATTERNS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "carrier feed changes to delivery exception hold",
        ("carrier", "exception", "hold", "feed", "logistics"),
        ("clear", "cleared"),
    ),
    (
        "inventory service changes order to backorder hold",
        ("inventory", "backorder", "stock", "allocated"),
        ("available", "clear", "cleared"),
    ),
    (
        "fraud review changes order address edit to locked",
        ("fraud", "review", "locked", "lock", "address"),
        ("cleared", "approved"),
    ),
    (
        "warehouse status changes to shipment locked",
        ("warehouse", "fulfillment", "shipment", "locked", "locks"),
        ("open", "editable"),
    ),
)

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
        matched_guidance = tuple(
            keyword for keyword in guidance_keywords if keyword in context_tokens
        )
        predicted_event = event
        profile_event = profile_event_for_intent(intent, context_tokens)
        if profile_event:
            predicted_event = profile_event
            probability = max(probability, 0.85)
        if matched_guidance and intent == "shipment_status_update":
            cue_words = (
                guidance_keywords
                if len(matched_guidance) >= MIN_CLUSTERED_GUIDANCE_CUES
                else matched_guidance
            )
            predicted_event = f"{predicted_event} {' '.join(cue_words)}"
        scored.append((intent, predicted_event, round(probability, 3)))

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


def profile_event_for_intent(intent: str, context_tokens: set[str]) -> str | None:
    if intent != "shipment_status_update":
        return None

    for event, positive_tokens, negative_tokens in ENVIRONMENT_PROFILE_PATTERNS:
        if any(token in context_tokens for token in negative_tokens):
            continue
        if len(set(positive_tokens) & context_tokens) >= 2:
            return event

    return None
