from __future__ import annotations

from collections import Counter
from typing import Iterable


def analyze_harness_misses(turn_log: Iterable[dict[str, object]]) -> dict[str, object]:
    harness_rows = [row for row in turn_log if row.get("variant") == "harness"]
    reasons: Counter[str] = Counter()

    for row in harness_rows:
        branches = row.get("branches", [])
        grades = {
            branch.get("match_grade")
            for branch in branches
            if isinstance(branch, dict)
        }

        if row.get("unsafe_leak"):
            reasons["unsafe_branch"] += 1
        elif not row.get("selected_artifact_id"):
            reasons["no_prepared_artifact"] += 1
        elif "exact_intent" in grades:
            reasons["exact_hit"] += 1
        elif "semantic_equivalent" in grades or "useful_partial" in grades:
            reasons["partial_hit"] += 1
        else:
            reasons["missed_next_event"] += 1

    return {
        "harness_turns": len(harness_rows),
        "reason_counts": dict(sorted(reasons.items())),
        "recommendations": recommendations_from_reasons(reasons),
    }


def recommendations_from_reasons(reasons: Counter[str]) -> list[str]:
    recommendations: list[str] = []
    if reasons["missed_next_event"]:
        recommendations.append(
            "Improve branch generation with more domain examples or a model-backed generator."
        )
    if reasons["no_prepared_artifact"]:
        recommendations.append(
            "Lower readiness threshold carefully or improve artifact templates for partial matches."
        )
    if reasons["unsafe_branch"]:
        recommendations.append(
            "Tighten safety filters before expanding top-k or deeper branch horizons."
        )
    if reasons["partial_hit"] and not reasons["exact_hit"]:
        recommendations.append(
            "Add a reranker that promotes semantically close branches with stronger intent evidence."
        )
    if not recommendations:
        recommendations.append(
            "Use a larger holdout replay set; this run has no obvious harness failure cluster."
        )
    return recommendations
