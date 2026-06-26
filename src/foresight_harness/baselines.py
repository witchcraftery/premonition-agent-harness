from __future__ import annotations

from foresight_harness.models import Artifact, ReplayTurn, RunResult


def live_agent(turn: ReplayTurn) -> RunResult:
    return RunResult(
        turn_id=turn.turn_id,
        variant="live_agent",
        latency_ms=turn.latency_budget_ms,
        token_cost=90,
        useful=True,
    )


def retrieval_plus_draft(turn: ReplayTurn) -> RunResult:
    return RunResult(
        turn_id=turn.turn_id,
        variant="retrieval_plus_draft",
        latency_ms=max(250, int(turn.latency_budget_ms * 0.75)),
        token_cost=120,
        useful=True,
    )


def semantic_cache(turn: ReplayTurn) -> RunResult:
    predictable = turn.expected_intent in {
        "refund_request",
        "billing_refund_timing",
        "address_change",
    }
    selected_artifact = None
    if predictable:
        selected_artifact = Artifact(
            artifact_id=f"{turn.turn_id}-semantic-cache",
            branch_id=f"{turn.turn_id}-semantic-cache",
            response_draft="Cached semantically similar support response.",
            policy_checks=(turn.policy_context,),
            readiness_score=0.7,
            token_cost=35,
            created_for_intent=turn.expected_intent,
        )

    return RunResult(
        turn_id=turn.turn_id,
        variant="semantic_cache",
        selected_artifact=selected_artifact,
        latency_ms=180 if predictable else turn.latency_budget_ms,
        token_cost=35 if predictable else 90,
        useful=predictable,
    )


def prediction_only(turn: ReplayTurn) -> RunResult:
    return RunResult(
        turn_id=turn.turn_id,
        variant="prediction_only",
        latency_ms=turn.latency_budget_ms,
        token_cost=45,
        useful=False,
    )
