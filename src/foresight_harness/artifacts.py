from __future__ import annotations

from foresight_harness.models import Artifact, Branch, MatchGrade, ReplayTurn
from foresight_harness.similarity import grade_branch_match


def prepare_artifacts(turn: ReplayTurn, branches: tuple[Branch, ...]) -> tuple[Artifact, ...]:
    artifacts: list[Artifact] = []
    for branch in branches:
        response = (
            f"If the next event is: {branch.predicted_event}. "
            f"Use this policy context: {turn.policy_context} "
            "Respond with the next best support step and do not present predicted facts as observed."
        )
        artifacts.append(
            Artifact(
                artifact_id=f"{branch.branch_id}-artifact",
                branch_id=branch.branch_id,
                response_draft=response,
                policy_checks=(turn.policy_context,),
                readiness_score=round(branch.probability * 0.9, 3),
                token_cost=len(response.split()),
                created_for_intent=branch.intent,
            )
        )
    return tuple(artifacts)


def select_artifact(
    turn: ReplayTurn,
    branches: tuple[Branch, ...],
    artifacts: tuple[Artifact, ...],
    readiness_threshold: float = 0.35,
) -> Artifact | None:
    artifacts_by_branch = {artifact.branch_id: artifact for artifact in artifacts}
    graded = [
        grade_branch_match(branch, turn.actual_next_event, turn.expected_intent)
        for branch in branches
    ]

    usable_grades = {
        MatchGrade.EXACT_INTENT,
        MatchGrade.SEMANTIC_EQUIVALENT,
        MatchGrade.USEFUL_PARTIAL,
    }

    for branch in sorted(graded, key=lambda item: item.rank):
        artifact = artifacts_by_branch.get(branch.branch_id)
        if artifact is None:
            continue
        if branch.match_grade in usable_grades and artifact.readiness_score >= readiness_threshold:
            return artifact

    return None
