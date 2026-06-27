from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from foresight_harness.models import ReplayTurn

TOPIC_EVENT_TYPES = {
    "address_change": "account_update",
    "billing_refund_timing": "billing",
    "escalation_request": "escalation",
    "refund_request": "refund",
    "shipment_status_update": "fulfillment",
    "troubleshooting_loop": "troubleshooting",
}


def classify_event(turn: ReplayTurn) -> dict[str, str]:
    return {
        "actor": classify_actor(turn.actual_next_event),
        "event_type": TOPIC_EVENT_TYPES.get(turn.expected_intent, "unknown"),
        "profile": classify_profile(turn),
        "topic": turn.expected_intent,
    }


def classify_actor(event_text: str) -> str:
    lowered = event_text.lower()
    if lowered.startswith("customer") or "customer " in lowered:
        return "user"
    if lowered.startswith("agent") or "agent " in lowered:
        return "agent"
    return "environment"


def classify_profile(turn: ReplayTurn) -> str:
    lowered = turn.actual_next_event.lower()
    if "carrier" in lowered and "exception" in lowered and "hold" in lowered:
        return "carrier_exception_hold"
    if "warehouse" in lowered and ("locked" in lowered or "lock" in lowered):
        return "warehouse_lock"
    if "inventory" in lowered and ("backorder" in lowered or "hold" in lowered):
        return "inventory_backorder"
    if "fraud" in lowered and ("locked" in lowered or "lock" in lowered):
        return "fraud_review_lock"
    if "policy" in lowered and ("feed" in lowered or "changes" in lowered):
        return "policy_update"
    if "payment gateway" in lowered or "gateway posts" in lowered:
        return "payment_gateway_update"
    return turn.expected_intent


def summarize_segments(
    baseline_rows: Iterable[dict[str, object]],
    guided_rows: Iterable[dict[str, object]],
) -> dict[str, object]:
    baseline_harness = [row for row in baseline_rows if row.get("variant") == "harness"]
    guided_harness = [row for row in guided_rows if row.get("variant") == "harness"]

    return {
        "by_topic": summarize_dimension(baseline_harness, guided_harness, "topic"),
        "by_event_type": summarize_dimension(baseline_harness, guided_harness, "event_type"),
        "by_actor": summarize_dimension(baseline_harness, guided_harness, "actor"),
        "by_profile": summarize_dimension(baseline_harness, guided_harness, "profile"),
        "focus_areas": focus_areas(baseline_harness, guided_harness),
    }


def summarize_dimension(
    baseline_rows: list[dict[str, object]],
    guided_rows: list[dict[str, object]],
    dimension: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    segment_names = sorted(
        {
            str(row.get(dimension, "unknown"))
            for row in baseline_rows + guided_rows
        }
    )
    summary: dict[str, dict[str, dict[str, float | int]]] = {}

    for segment in segment_names:
        baseline = metric_summary(
            row for row in baseline_rows if row.get(dimension) == segment
        )
        guided = metric_summary(
            row for row in guided_rows if row.get(dimension) == segment
        )
        summary[segment] = {
            "baseline": baseline,
            "guided": guided,
            "delta": {
                "p_at_1": round(guided["p_at_1"] - baseline["p_at_1"], 3),
                "usefulness_rate": round(
                    guided["usefulness_rate"] - baseline["usefulness_rate"],
                    3,
                ),
            },
        }

    return summary


def metric_summary(rows: Iterable[dict[str, object]]) -> dict[str, float | int]:
    rows_tuple = tuple(rows)
    total = len(rows_tuple)
    if total == 0:
        return {"total_turns": 0, "p_at_1": 0.0, "usefulness_rate": 0.0}

    top_1_hits = 0
    useful = 0
    for row in rows_tuple:
        branches = [
            branch for branch in row.get("branches", [])
            if isinstance(branch, dict)
        ]
        top_branch = next((branch for branch in branches if branch.get("rank") == 1), None)
        if top_branch and top_branch.get("match_grade") == "exact_intent":
            top_1_hits += 1
        if row.get("selected_artifact_id"):
            useful += 1

    return {
        "total_turns": total,
        "p_at_1": round(top_1_hits / total, 3),
        "usefulness_rate": round(useful / total, 3),
    }


def focus_areas(
    baseline_rows: list[dict[str, object]],
    guided_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dimension in ("topic", "event_type", "actor", "profile"):
        dimension_summary = summarize_dimension(baseline_rows, guided_rows, dimension)
        for name, summary in dimension_summary.items():
            rows.append(
                {
                    "segment": f"by_{dimension}",
                    "name": name,
                    "guided_p_at_1": summary["guided"]["p_at_1"],
                    "guided_usefulness_rate": summary["guided"]["usefulness_rate"],
                    "p_at_1_delta": summary["delta"]["p_at_1"],
                    "usefulness_delta": summary["delta"]["usefulness_rate"],
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            float(row["guided_p_at_1"]),
            float(row["guided_usefulness_rate"]),
            -float(row["p_at_1_delta"]),
        ),
    )
