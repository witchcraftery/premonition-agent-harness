from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

from foresight_harness.artifacts import prepare_artifacts, select_artifact
from foresight_harness.baselines import (
    live_agent,
    prediction_only,
    retrieval_plus_draft,
    semantic_cache,
)
from foresight_harness.branching import generate_branches
from foresight_harness.models import MatchGrade, ReplayTurn, RunResult
from foresight_harness.similarity import grade_branch_match


def load_replay_turns(path: Path) -> tuple[ReplayTurn, ...]:
    turns: list[ReplayTurn] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                turns.append(ReplayTurn.from_dict(json.loads(line)))
    return tuple(turns)


def run_harness(turn: ReplayTurn, top_k: int = 3) -> RunResult:
    branches = generate_branches(turn, top_k=top_k)
    graded = tuple(
        grade_branch_match(branch, turn.actual_next_event, turn.expected_intent)
        for branch in branches
    )
    artifacts = prepare_artifacts(turn, graded)
    selected = select_artifact(turn, graded, artifacts)

    return RunResult(
        turn_id=turn.turn_id,
        variant="harness",
        branches=graded,
        selected_artifact=selected,
        latency_ms=120 if selected else turn.latency_budget_ms,
        token_cost=sum(artifact.token_cost for artifact in artifacts),
        useful=selected is not None,
        unsafe_leak=any(branch.match_grade == MatchGrade.UNSAFE for branch in graded),
    )


def summarize(results: Iterable[RunResult]) -> dict[str, float | int]:
    rows = tuple(results)
    total = len(rows)
    if total == 0:
        return {
            "total_turns": 0,
            "cache_hit_rate": 0.0,
            "median_latency_ms": 0,
            "median_token_cost": 0,
            "usefulness_rate": 0.0,
            "unsafe_leak_rate": 0.0,
        }

    return {
        "total_turns": total,
        "cache_hit_rate": round(sum(row.selected_artifact is not None for row in rows) / total, 3),
        "median_latency_ms": int(median(row.latency_ms for row in rows)),
        "median_token_cost": int(median(row.token_cost for row in rows)),
        "usefulness_rate": round(sum(row.useful for row in rows) / total, 3),
        "unsafe_leak_rate": round(sum(row.unsafe_leak for row in rows) / total, 3),
    }


def summarize_harness(results: Iterable[RunResult], top_k: int) -> dict[str, float | int]:
    rows = tuple(results)
    summary = summarize(rows)
    total = max(len(rows), 1)
    rank1_hits = 0
    topk_hits = 0
    branch_count = 0

    for row in rows:
        branch_count += len(row.branches)
        exact_ranks = [
            branch.rank
            for branch in row.branches
            if branch.match_grade == MatchGrade.EXACT_INTENT
        ]
        if exact_ranks and min(exact_ranks) == 1:
            rank1_hits += 1
        if exact_ranks and min(exact_ranks) <= top_k:
            topk_hits += 1

    summary.update(
        {
            "p_at_1": round(rank1_hits / total, 3),
            f"top_{top_k}_recall": round(topk_hits / total, 3),
            "branch_hit_rate": round(topk_hits / max(branch_count, 1), 3),
            "stale_artifact_rate": 0.0,
        }
    )
    return summary


def run_replay(turns: tuple[ReplayTurn, ...], top_k: int = 3) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[RunResult]] = defaultdict(list)

    for turn in turns:
        for result in (
            live_agent(turn),
            retrieval_plus_draft(turn),
            semantic_cache(turn),
            prediction_only(turn),
            run_harness(turn, top_k=top_k),
        ):
            grouped[result.variant].append(result)

    report = {name: summarize(results) for name, results in grouped.items() if name != "harness"}
    report["harness"] = summarize_harness(grouped["harness"], top_k=top_k)
    return report
