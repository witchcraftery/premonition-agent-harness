from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from foresight_harness.evaluator import run_replay, run_replay_turn_log
from foresight_harness.models import ReplayTurn
from foresight_harness.similarity import GENERIC_EVENT_TOKENS, normalized_tokens

MAX_KEYWORDS_PER_INTENT = 10
GUIDANCE_STOP_TOKENS = GENERIC_EVENT_TOKENS | {
    "after",
    "agent",
    "agents",
    "am",
    "be",
    "been",
    "can",
    "checking",
    "has",
    "have",
    "history",
    "i",
    "is",
    "my",
    "nobody",
    "not",
    "now",
    "the",
    "this",
    "will",
}


@dataclass(frozen=True)
class Guidance:
    intent_keywords: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def keywords_for(self, intent: str) -> tuple[str, ...]:
        return self.intent_keywords.get(intent, tuple())

    def with_keywords(self, intent: str, keywords: Iterable[str]) -> "Guidance":
        existing = list(self.intent_keywords.get(intent, tuple()))
        merged = existing + [
            keyword
            for keyword in keywords
            if keyword not in existing and keyword not in GUIDANCE_STOP_TOKENS
        ]
        updated = dict(self.intent_keywords)
        updated[intent] = tuple(merged[:MAX_KEYWORDS_PER_INTENT])
        return Guidance(intent_keywords=updated)

    def to_dict(self) -> dict[str, dict[str, list[str]]]:
        return {
            "intent_keywords": {
                intent: list(keywords)
                for intent, keywords in sorted(self.intent_keywords.items())
            }
        }


def learn_guidance_from_turn_log(
    turns: tuple[ReplayTurn, ...],
    turn_log: Iterable[dict[str, object]],
    guidance: Guidance,
) -> Guidance:
    turns_by_id = {turn.turn_id: turn for turn in turns}
    learned = guidance

    for row in turn_log:
        if row.get("variant") != "harness":
            continue

        branches = [
            branch for branch in row.get("branches", [])
            if isinstance(branch, dict)
        ]
        top_branch = next((branch for branch in branches if branch.get("rank") == 1), None)
        exact_top_1 = top_branch and top_branch.get("match_grade") == "exact_intent"
        prepared = row.get("selected_artifact_id") is not None
        if exact_top_1 and prepared:
            continue

        turn = turns_by_id[str(row["turn_id"])]
        cue_tokens = sorted(
            normalized_tokens(f"{turn.context_text()} {turn.actual_next_event}")
            - GUIDANCE_STOP_TOKENS
        )
        learned = learned.with_keywords(turn.expected_intent, cue_tokens)

    return learned


def run_guidance_loop(
    turns: tuple[ReplayTurn, ...],
    iterations: int,
    top_k: int = 3,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    guidance = Guidance()
    rows: list[dict[str, object]] = []

    for index in range(1, iterations + 1):
        report = run_replay(turns, top_k=top_k, guidance=guidance)
        turn_log = run_replay_turn_log(turns, top_k=top_k, guidance=guidance)
        previous_p_at_1 = (
            rows[-1]["report"]["harness"]["p_at_1"]
            if rows
            else report["harness"]["p_at_1"]
        )
        previous_usefulness = (
            rows[-1]["report"]["harness"]["usefulness_rate"]
            if rows
            else report["harness"]["usefulness_rate"]
        )
        rows.append(
            {
                "iteration": index,
                "guidance": guidance.to_dict(),
                "report": report,
                "assessment": assess_iteration(
                    turn_log,
                    report,
                    previous_p_at_1,
                    previous_usefulness,
                ),
            }
        )

        if index < iterations:
            guidance = learn_guidance_from_turn_log(turns, turn_log, guidance)

    return {
        "iterations": rows,
        "final_guidance": guidance.to_dict(),
        "guidance_markdown": render_guidance_markdown(guidance),
    }


def assess_iteration(
    turn_log: tuple[dict[str, object], ...],
    report: dict[str, dict[str, float | int]],
    previous_p_at_1: float,
    previous_usefulness: float,
) -> dict[str, object]:
    exact_top_1: list[str] = []
    misses: list[str] = []
    prepared_turns: list[str] = []
    unprepared_turns: list[str] = []

    for row in turn_log:
        if row.get("variant") != "harness":
            continue

        if row.get("selected_artifact_id"):
            prepared_turns.append(str(row["turn_id"]))
        else:
            unprepared_turns.append(str(row["turn_id"]))

        branches = [
            branch for branch in row.get("branches", [])
            if isinstance(branch, dict)
        ]
        top_branch = next((branch for branch in branches if branch.get("rank") == 1), None)
        if top_branch and top_branch.get("match_grade") == "exact_intent":
            exact_top_1.append(str(row["turn_id"]))
        else:
            misses.append(str(row["turn_id"]))

    current_p_at_1 = float(report["harness"]["p_at_1"])
    current_usefulness = float(report["harness"]["usefulness_rate"])
    return {
        "exact_top_1_turns": exact_top_1,
        "missed_turns": misses,
        "prepared_turns": prepared_turns,
        "unprepared_turns": unprepared_turns,
        "p_at_1_delta": round(current_p_at_1 - previous_p_at_1, 3),
        "preparedness_delta": round(current_usefulness - previous_usefulness, 3),
    }


def render_guidance_markdown(guidance: Guidance) -> str:
    lines = [
        "# Premonition Guidance",
        "",
        "Learned intent cues from replay misses.",
        "",
    ]
    if not guidance.intent_keywords:
        lines.append("No learned guidance yet.")
    else:
        for intent, keywords in sorted(guidance.intent_keywords.items()):
            lines.append(f"## {intent}")
            lines.append("")
            lines.append(", ".join(keywords))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
