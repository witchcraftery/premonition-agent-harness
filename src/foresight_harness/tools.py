from __future__ import annotations

from foresight_harness.artifacts import prepare_artifacts, select_artifact
from foresight_harness.branching import generate_branches
from foresight_harness.models import MatchGrade, PremonitionPacket, ReplayTurn
from foresight_harness.similarity import grade_branch_match


def build_premonition_packet(turn: ReplayTurn, top_k: int = 3) -> PremonitionPacket:
    branches = generate_branches(turn, top_k=top_k)
    graded = tuple(
        grade_branch_match(branch, turn.actual_next_event, turn.expected_intent)
        for branch in branches
    )
    artifacts = prepare_artifacts(turn, graded)
    selected = select_artifact(turn, graded, artifacts)
    matched_branch = next(
        (
            branch
            for branch in graded
            if selected is not None and branch.branch_id == selected.branch_id
        ),
        None,
    )

    return PremonitionPacket(
        packet_id=f"{turn.turn_id}-premonition",
        turn_id=turn.turn_id,
        observed_context=turn.context_text(),
        matched_branch_id=matched_branch.branch_id if matched_branch else None,
        matched_intent=matched_branch.intent if matched_branch else None,
        confidence=matched_branch.probability if matched_branch else 0.0,
        prepared_artifact=selected.response_draft if selected else None,
        policy_checks=selected.policy_checks if selected else tuple(),
        freshness="valid" if selected else "no_match",
        unsafe=any(branch.match_grade == MatchGrade.UNSAFE for branch in graded),
    )
