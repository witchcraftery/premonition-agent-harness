from __future__ import annotations

from collections import Counter

from foresight_harness.models import Branch, ReplayTurn
from foresight_harness.similarity import normalized_tokens

MIN_CLUSTERED_GUIDANCE_CUES = 3
PROFILE_EVENT_PATTERNS: tuple[
    tuple[str, str, tuple[str, ...], tuple[str, ...], int],
    ...
] = (
    (
        "shipment_status_update",
        "carrier feed changes to delivery exception hold",
        ("carrier", "exception", "hold", "feed", "logistics"),
        ("allocated", "inventory", "stock", "clear", "cleared"),
        2,
    ),
    (
        "shipment_status_update",
        "inventory service changes order to backorder hold",
        ("inventory", "backorder", "stock", "allocated"),
        ("available", "clear", "cleared"),
        2,
    ),
    (
        "address_change",
        "fraud review changes order address edit to locked",
        ("fraud", "review", "locked", "locks", "lock", "destination", "account"),
        ("cleared", "approved", "real"),
        3,
    ),
    (
        "shipment_status_update",
        "warehouse status changes to shipment locked",
        ("warehouse", "fulfillment", "shipment", "locked", "locks"),
        ("allows", "changed", "cleared", "edits", "not", "open", "editable"),
        2,
    ),
    (
        "refund_request",
        "returns portal changes damaged delivery to replacement eligible",
        ("returns", "portal", "marked", "eligible"),
        (),
        2,
    ),
    (
        "refund_request",
        "policy feed changes damaged delivery refund to require photo verification",
        ("policy", "feed", "required", "require", "verification", "photo"),
        ("uploaded", "eligible"),
        3,
    ),
    (
        "billing_refund_timing",
        "payment gateway posts duplicate charge refund timing update",
        ("gateway", "service", "pending", "settled", "reversal", "timing"),
        ("clear", "cleared"),
        2,
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
        (
            "address",
            "apartment",
            "changed",
            "cleared",
            "destination",
            "editable",
            "edits",
            "fraud",
            "left",
            "old",
            "order",
            "place",
            "shipped",
            "shipping",
            "change",
        ),
    ),
    "shipment_status_update": (
        "warehouse status changes to shipment locked",
        ("warehouse", "status", "fulfillment", "shipment", "locked", "monitoring"),
    ),
}

INTENT_NEGATIVE_CUES: dict[str, tuple[str, ...]] = {
    "shipment_status_update": (
        "allows",
        "changed",
        "cleared",
        "destination",
        "edits",
        "left",
        "not",
        "old",
        "place",
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
        negative_hits = sum(
            1
            for keyword in INTENT_NEGATIVE_CUES.get(intent, ())
            if keyword in context_tokens
        )
        if negative_hits:
            probability = max(prior, probability - (0.16 * negative_hits))
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
    for profile_intent, event, positive_tokens, negative_tokens, minimum_hits in PROFILE_EVENT_PATTERNS:
        if profile_intent != intent:
            continue
        if any(token in context_tokens for token in negative_tokens):
            continue
        if len(set(positive_tokens) & context_tokens) >= minimum_hits:
            return event

    return None
