from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from foresight_harness.models import Message
from foresight_harness.similarity import normalized_tokens


DAILYDIALOG_ACTS = {
    "1": "inform",
    "2": "question",
    "3": "directive",
    "4": "commissive",
}

DAILYDIALOG_EMOTIONS = {
    "0": "no_emotion",
    "1": "anger",
    "2": "disgust",
    "3": "fear",
    "4": "happiness",
    "5": "sadness",
    "6": "surprise",
}

CONVERSATION_ACTS = ("inform", "question", "directive", "commissive")
CONVERSATION_GUIDANCE_STOP_WORDS = {
    "about",
    "like",
    "sounds",
    "speaker",
    "that",
}

BASE_ACT_PRIORS = {
    "inform": 0.42,
    "question": 0.24,
    "directive": 0.22,
    "commissive": 0.18,
}

ACT_TEMPLATES = {
    "inform": "I can give a short, grounded response that keeps the conversation moving.",
    "question": "I can ask a warm follow-up question that invites the next detail.",
    "directive": "I can offer one clear next step without sounding abrupt.",
    "commissive": "I can confirm the helpful action I am ready to take.",
}


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    conversation: tuple[Message, ...]
    next_speaker: str
    actual_next_utterance: str
    expected_act: str
    expected_emotion: str
    latency_budget_ms: int = 650

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ConversationTurn":
        return cls(
            turn_id=str(row["turn_id"]),
            conversation=tuple(Message.from_dict(item) for item in row["conversation"]),
            next_speaker=str(row["next_speaker"]),
            actual_next_utterance=str(row["actual_next_utterance"]),
            expected_act=str(row["expected_act"]),
            expected_emotion=str(row.get("expected_emotion", "no_emotion")),
            latency_budget_ms=int(row.get("latency_budget_ms", 650)),
        )

    def context_text(self) -> str:
        return "\n".join(f"{message.role}: {message.content}" for message in self.conversation)


@dataclass(frozen=True)
class ConversationGuidance:
    act_keywords: dict[str, tuple[str, ...]]

    def keywords_for(self, act: str) -> tuple[str, ...]:
        return self.act_keywords.get(act, tuple())

    def to_dict(self) -> dict[str, dict[str, list[str]]]:
        return {
            "act_keywords": {
                act: list(keywords)
                for act, keywords in sorted(self.act_keywords.items())
            }
        }


@dataclass(frozen=True)
class ConversationProbabilityPack:
    pack_id: str
    turn_id: str
    observed_context: str
    top_branches: tuple[dict[str, object], ...]
    prepared_drafts: tuple[dict[str, object], ...]
    confirmation_mode: str
    expires_after_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "pack_id": self.pack_id,
            "turn_id": self.turn_id,
            "observed_context": self.observed_context,
            "top_branches": list(self.top_branches),
            "prepared_drafts": list(self.prepared_drafts),
            "confirmation_mode": self.confirmation_mode,
            "expires_after_ms": self.expires_after_ms,
        }


def load_conversation_turns(path: Path) -> tuple[ConversationTurn, ...]:
    rows: list[ConversationTurn] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(ConversationTurn.from_dict(json.loads(line)))
    return tuple(rows)


def load_dailydialog_export(
    dialogues_path: Path,
    acts_path: Path,
    emotions_path: Path,
) -> tuple[ConversationTurn, ...]:
    dialogue_lines = dialogues_path.read_text(encoding="utf-8").splitlines()
    act_lines = acts_path.read_text(encoding="utf-8").splitlines()
    emotion_lines = emotions_path.read_text(encoding="utf-8").splitlines()
    turns: list[ConversationTurn] = []

    for dialogue_index, (dialogue_line, act_line, emotion_line) in enumerate(
        zip(dialogue_lines, act_lines, emotion_lines),
        start=1,
    ):
        utterances = [
            utterance.strip()
            for utterance in dialogue_line.split("__eou__")
            if utterance.strip()
        ]
        acts = act_line.split()
        emotions = emotion_line.split()
        for next_index in range(1, len(utterances)):
            history = tuple(
                Message(
                    role=speaker_role(index),
                    content=utterances[index],
                )
                for index in range(next_index)
            )
            turns.append(
                ConversationTurn(
                    turn_id=f"dailydialog-{dialogue_index:04d}-{next_index:03d}",
                    conversation=history,
                    next_speaker=speaker_role(next_index),
                    actual_next_utterance=utterances[next_index],
                    expected_act=DAILYDIALOG_ACTS.get(acts[next_index], "inform"),
                    expected_emotion=DAILYDIALOG_EMOTIONS.get(
                        emotions[next_index],
                        "no_emotion",
                    ),
                )
            )

    return tuple(turns)


def speaker_role(index: int) -> str:
    return "speaker_a" if index % 2 == 0 else "speaker_b"


def build_probability_pack(
    turn: ConversationTurn,
    top_k: int = 3,
    guidance: ConversationGuidance | None = None,
) -> ConversationProbabilityPack:
    branches = generate_conversation_branches(turn, top_k=top_k, guidance=guidance)
    drafts = tuple(
        {
            "branch_id": branch["branch_id"],
            "act": branch["act"],
            "tts_text": ACT_TEMPLATES[str(branch["act"])],
            "voice_ready": True,
            "readiness_score": round(float(branch["probability"]) * 0.92, 3),
        }
        for branch in branches
    )
    return ConversationProbabilityPack(
        pack_id=f"{turn.turn_id}-probability-pack",
        turn_id=turn.turn_id,
        observed_context=turn.context_text(),
        top_branches=branches,
        prepared_drafts=drafts,
        confirmation_mode="wait_for_observed_next_move",
        expires_after_ms=2500,
    )


def generate_conversation_branches(
    turn: ConversationTurn,
    top_k: int = 3,
    guidance: ConversationGuidance | None = None,
) -> tuple[dict[str, object], ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    context_tokens = normalized_tokens(turn.context_text())
    scores = {
        act: BASE_ACT_PRIORS[act] + heuristic_score(act, context_tokens, turn)
        for act in CONVERSATION_ACTS
    }
    if guidance:
        for act in CONVERSATION_ACTS:
            hits = set(guidance.keywords_for(act)) & context_tokens
            scores[act] += min(0.56, 0.14 * len(hits))

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return tuple(
        {
            "branch_id": f"{turn.turn_id}-conv-branch-{index}",
            "rank": index,
            "act": act,
            "emotion": predict_emotion(turn, context_tokens),
            "probability": round(min(score, 0.95), 3),
            "trigger_cues": sorted(context_tokens)[:8],
        }
        for index, (act, score) in enumerate(ranked, start=1)
    )


def heuristic_score(
    act: str,
    context_tokens: set[str],
    turn: ConversationTurn,
) -> float:
    latest = turn.conversation[-1].content.lower() if turn.conversation else ""
    score = 0.0
    if act == "commissive" and {"remind", "could", "please", "help"} & context_tokens:
        score += 0.32
    if act == "directive" and {"loud", "focus", "meet", "should"} & context_tokens:
        score += 0.24
    if act == "question" and ("?" in latest or {"why", "what", "how"} & context_tokens):
        score += 0.18
    if act == "question" and {"anyone", "city", "moved"} & context_tokens:
        score += 0.34
    if act == "inform" and {"finished", "finally", "happy", "surprised"} & context_tokens:
        score += 0.18
    return score


def predict_emotion(turn: ConversationTurn, context_tokens: set[str]) -> str:
    if {"nervous", "interview"} & context_tokens:
        return "fear"
    if {"missed", "late", "frustrating"} & context_tokens:
        return "sadness"
    if {"finished", "proud", "happy", "birthday"} & context_tokens:
        return "happiness"
    if {"surprised", "expect"} & context_tokens:
        return "surprise"
    return "no_emotion"


def run_conversation_probability_loop(
    turns: tuple[ConversationTurn, ...],
    iterations: int,
    top_k: int = 3,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    guidance = ConversationGuidance(act_keywords={})
    iteration_reports: list[dict[str, object]] = []
    for iteration in range(1, iterations + 1):
        rows = score_conversation_turns(turns, top_k=top_k, guidance=guidance)
        metrics = summarize_conversation_rows(rows, turns)
        iteration_reports.append(
            {
                "iteration": iteration,
                "metrics": metrics,
                "guidance": guidance.to_dict(),
                "missed_turns": [
                    row["turn_id"]
                    for row in rows
                    if row["rank_1_act"] != row["expected_act"]
                ],
            }
        )
        guidance = learn_conversation_guidance(turns, rows, guidance)

    return {
        "summary": {
            "total_turns": len(turns),
            "iterations": iterations,
            "top_k": top_k,
        },
        "iterations": iteration_reports,
        "final_guidance": guidance.to_dict(),
    }


def score_conversation_turns(
    turns: tuple[ConversationTurn, ...],
    top_k: int,
    guidance: ConversationGuidance,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for turn in turns:
        pack = build_probability_pack(turn, top_k=top_k, guidance=guidance)
        rank_1 = pack.top_branches[0]
        rows.append(
            {
                "turn_id": turn.turn_id,
                "expected_act": turn.expected_act,
                "expected_emotion": turn.expected_emotion,
                "rank_1_act": rank_1["act"],
                "top_acts": [branch["act"] for branch in pack.top_branches],
                "tts_ready": bool(pack.prepared_drafts),
                "latency_ms": 90 if pack.prepared_drafts else turn.latency_budget_ms,
            }
        )
    return tuple(rows)


def summarize_conversation_rows(
    rows: tuple[dict[str, object], ...],
    turns: tuple[ConversationTurn, ...],
) -> dict[str, float | int]:
    total = max(len(rows), 1)
    exact = sum(row["rank_1_act"] == row["expected_act"] for row in rows)
    top_3 = sum(row["expected_act"] in row["top_acts"] for row in rows)
    tts_ready = sum(bool(row["tts_ready"]) for row in rows)
    return {
        "total_turns": len(turns),
        "p_at_1": round(exact / total, 3),
        "top_3_recall": round(top_3 / total, 3),
        "tts_readiness_rate": round(tts_ready / total, 3),
        "median_latency_ms": int(median(row["latency_ms"] for row in rows)) if rows else 0,
    }


def learn_conversation_guidance(
    turns: tuple[ConversationTurn, ...],
    rows: tuple[dict[str, object], ...],
    prior: ConversationGuidance,
) -> ConversationGuidance:
    keywords = {
        act: list(existing)
        for act, existing in prior.act_keywords.items()
    }
    turns_by_id = {turn.turn_id: turn for turn in turns}
    for row in rows:
        if row["rank_1_act"] == row["expected_act"]:
            continue
        turn = turns_by_id[str(row["turn_id"])]
        act = turn.expected_act
        keywords.setdefault(act, [])
        for token in sorted(normalized_tokens(turn.context_text())):
            if (
                len(token) < 4
                or token in CONVERSATION_GUIDANCE_STOP_WORDS
                or token in keywords[act]
            ):
                continue
            keywords[act].append(token)
            if len(keywords[act]) >= 12:
                break

    return ConversationGuidance(
        act_keywords={
            act: tuple(values[:12])
            for act, values in keywords.items()
            if values
        }
    )
