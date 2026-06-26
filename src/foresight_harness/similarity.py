from __future__ import annotations

import re

from foresight_harness.models import Branch, MatchGrade

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "if",
    "is",
    "item",
    "it",
    "of",
    "on",
    "or",
    "please",
    "the",
    "to",
    "whether",
}

UNSAFE_TERMS = {
    "guaranteed refund",
    "refund guaranteed",
    "ignore policy",
    "skip verification",
}


def normalized_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOP_WORDS}


def semantic_overlap(left: str, right: str) -> float:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def contains_unsafe_claim(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in UNSAFE_TERMS)


def grade_branch_match(
    branch: Branch,
    actual_next_event: str,
    expected_intent: str,
) -> Branch:
    if contains_unsafe_claim(branch.predicted_event):
        branch.match_grade = MatchGrade.UNSAFE
        branch.match_score = 0.0
        return branch

    if branch.intent == expected_intent:
        branch.match_grade = MatchGrade.EXACT_INTENT
        branch.match_score = 1.0
        return branch

    overlap = semantic_overlap(branch.predicted_event, actual_next_event)
    branch.match_score = round(overlap, 3)

    if overlap >= 0.55:
        branch.match_grade = MatchGrade.SEMANTIC_EQUIVALENT
    elif overlap >= 0.30:
        branch.match_grade = MatchGrade.USEFUL_PARTIAL
    else:
        branch.match_grade = MatchGrade.MISS

    return branch
