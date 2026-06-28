from __future__ import annotations

import json
import math
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
class ConversationActRanker:
    act_log_priors: dict[str, float]
    feature_log_likelihoods: dict[str, dict[str, float]]
    default_feature_log_likelihoods: dict[str, float]
    vocabulary: tuple[str, ...]
    act_counts: dict[str, int]

    def score(self, turn: ConversationTurn) -> dict[str, float]:
        features = conversation_features(turn)
        return {
            act: self.act_log_priors[act]
            + sum(
                self.feature_log_likelihoods.get(act, {}).get(
                    feature,
                    self.default_feature_log_likelihoods[act],
                )
                for feature in features
            )
            for act in CONVERSATION_ACTS
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "act_counts": dict(sorted(self.act_counts.items())),
            "vocabulary_size": len(self.vocabulary),
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


def load_dailydialog_split(split_dir: Path) -> tuple[ConversationTurn, ...]:
    split_name = split_dir.name
    turns = load_dailydialog_export(
        split_dir / "dialogues.txt",
        split_dir / "dialogues_act.txt",
        split_dir / "dialogues_emotion.txt",
    )
    return tuple(
        ConversationTurn(
            turn_id=turn.turn_id.replace("dailydialog-", f"dailydialog-{split_name}-"),
            conversation=turn.conversation,
            next_speaker=turn.next_speaker,
            actual_next_utterance=turn.actual_next_utterance,
            expected_act=turn.expected_act,
            expected_emotion=turn.expected_emotion,
            latency_budget_ms=turn.latency_budget_ms,
        )
        for turn in turns
    )


def write_conversation_turns(
    turns: tuple[ConversationTurn, ...],
    output_path: Path,
    limit: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = turns[:limit] if limit is not None else turns
    with output_path.open("w", encoding="utf-8") as handle:
        for turn in selected:
            handle.write(json.dumps(conversation_turn_to_dict(turn), sort_keys=True) + "\n")


def conversation_turn_to_dict(turn: ConversationTurn) -> dict[str, object]:
    return {
        "turn_id": turn.turn_id,
        "conversation": [
            {"role": message.role, "content": message.content}
            for message in turn.conversation
        ],
        "next_speaker": turn.next_speaker,
        "actual_next_utterance": turn.actual_next_utterance,
        "expected_act": turn.expected_act,
        "expected_emotion": turn.expected_emotion,
        "latency_budget_ms": turn.latency_budget_ms,
    }


def speaker_role(index: int) -> str:
    return "speaker_a" if index % 2 == 0 else "speaker_b"


def build_probability_pack(
    turn: ConversationTurn,
    top_k: int = 3,
    guidance: ConversationGuidance | None = None,
    act_ranker: ConversationActRanker | None = None,
    learned_weight: float = 0.0,
    scoring_variant: str = "heuristic",
) -> ConversationProbabilityPack:
    branches = generate_conversation_branches(
        turn,
        top_k=top_k,
        guidance=guidance,
        act_ranker=act_ranker,
        learned_weight=learned_weight,
        scoring_variant=scoring_variant,
    )
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
    act_ranker: ConversationActRanker | None = None,
    learned_weight: float = 0.0,
    scoring_variant: str = "heuristic",
) -> tuple[dict[str, object], ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if learned_weight < 0 or learned_weight > 1:
        raise ValueError("learned_weight must be between 0 and 1")
    if scoring_variant == "heuristic" and act_ranker and learned_weight == 1.0:
        scoring_variant = "learned"

    context_tokens = normalized_tokens(turn.context_text())
    scores = heuristic_act_scores(turn, context_tokens)
    if guidance:
        for act in CONVERSATION_ACTS:
            hits = set(guidance.keywords_for(act)) & context_tokens
            scores[act] += min(0.56, 0.14 * len(hits))
    if act_ranker and learned_weight > 0:
        scores = blended_act_scores(
            heuristic_scores=scores,
            learned_scores=act_ranker.score(turn),
            learned_weight=learned_weight,
        )

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return tuple(
        {
            "branch_id": f"{turn.turn_id}-conv-branch-{index}",
            "rank": index,
            "act": act,
            "emotion": predict_emotion(turn, context_tokens),
            "probability": round(min(score, 0.95), 3),
            "trigger_cues": sorted(context_tokens)[:8],
            "scoring_variant": scoring_variant,
        }
        for index, (act, score) in enumerate(ranked, start=1)
    )


def heuristic_act_scores(
    turn: ConversationTurn,
    context_tokens: set[str],
) -> dict[str, float]:
    return {
        act: BASE_ACT_PRIORS[act] + heuristic_score(act, context_tokens, turn)
        for act in CONVERSATION_ACTS
    }


def blended_act_scores(
    heuristic_scores: dict[str, float],
    learned_scores: dict[str, float],
    learned_weight: float,
) -> dict[str, float]:
    heuristic = normalize_act_scores(heuristic_scores)
    learned = normalize_act_scores(learned_scores)
    return {
        act: (1 - learned_weight) * heuristic[act] + learned_weight * learned[act]
        for act in CONVERSATION_ACTS
    }


def normalize_act_scores(scores: dict[str, float]) -> dict[str, float]:
    values = [scores[act] for act in CONVERSATION_ACTS]
    low = min(values)
    high = max(values)
    if high == low:
        return {act: 0.5 for act in CONVERSATION_ACTS}
    return {
        act: (scores[act] - low) / (high - low)
        for act in CONVERSATION_ACTS
    }


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


def train_conversation_act_ranker(
    turns: tuple[ConversationTurn, ...],
) -> ConversationActRanker:
    if not turns:
        raise ValueError("training turns are required")

    act_counts = {act: 0 for act in CONVERSATION_ACTS}
    feature_counts = {act: {} for act in CONVERSATION_ACTS}
    feature_totals = {act: 0 for act in CONVERSATION_ACTS}
    vocabulary: set[str] = set()

    for turn in turns:
        act = turn.expected_act if turn.expected_act in CONVERSATION_ACTS else "inform"
        act_counts[act] += 1
        features = conversation_features(turn)
        vocabulary.update(features)
        for feature in features:
            feature_counts[act][feature] = feature_counts[act].get(feature, 0) + 1
            feature_totals[act] += 1

    total_turns = len(turns)
    vocabulary_size = max(len(vocabulary), 1)
    act_log_priors = {
        act: math.log((act_counts[act] + 1) / (total_turns + len(CONVERSATION_ACTS)))
        for act in CONVERSATION_ACTS
    }
    default_feature_log_likelihoods = {
        act: math.log(1 / (feature_totals[act] + vocabulary_size))
        for act in CONVERSATION_ACTS
    }
    feature_log_likelihoods = {
        act: {
            feature: math.log(
                (count + 1) / (feature_totals[act] + vocabulary_size)
            )
            for feature, count in counts.items()
        }
        for act, counts in feature_counts.items()
    }
    return ConversationActRanker(
        act_log_priors=act_log_priors,
        feature_log_likelihoods=feature_log_likelihoods,
        default_feature_log_likelihoods=default_feature_log_likelihoods,
        vocabulary=tuple(sorted(vocabulary)),
        act_counts=act_counts,
    )


def conversation_features(turn: ConversationTurn) -> tuple[str, ...]:
    features = set(normalized_tokens(turn.context_text()))
    latest = turn.conversation[-1].content.strip().lower() if turn.conversation else ""
    if "?" in latest:
        features.add("latest_has_question_mark")
    for wh_word in ("what", "why", "how", "where", "when", "who"):
        if latest.startswith(wh_word):
            features.add(f"latest_starts_{wh_word}")
    features.add(f"history_turns_{min(len(turn.conversation), 4)}")
    features.add(f"next_speaker_{turn.next_speaker}")
    return tuple(sorted(features))


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
        candidate_guidance = learn_conversation_guidance(turns, rows, guidance)
        candidate_rows = score_conversation_turns(
            turns,
            top_k=top_k,
            guidance=candidate_guidance,
        )
        candidate_metrics = summarize_conversation_rows(candidate_rows, turns)
        guidance_promoted = (
            float(candidate_metrics["p_at_1"]) > float(metrics["p_at_1"])
            and float(candidate_metrics["tts_readiness_rate"]) >= float(metrics["tts_readiness_rate"])
        )
        iteration_reports.append(
            {
                "iteration": iteration,
                "metrics": metrics,
                "candidate_metrics": candidate_metrics,
                "guidance_promoted": guidance_promoted,
                "guidance": guidance.to_dict(),
                "missed_turns": [
                    row["turn_id"]
                    for row in rows
                    if row["rank_1_act"] != row["expected_act"]
                ],
            }
        )
        if guidance_promoted:
            guidance = candidate_guidance

    return {
        "summary": {
            "total_turns": len(turns),
            "iterations": iterations,
            "top_k": top_k,
        },
        "iterations": iteration_reports,
        "final_guidance": guidance.to_dict(),
    }


def run_conversation_train_dev_test_loop(
    train_turns: tuple[ConversationTurn, ...],
    dev_turns: tuple[ConversationTurn, ...],
    test_turns: tuple[ConversationTurn, ...],
    iterations: int,
    top_k: int = 3,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not train_turns or not dev_turns or not test_turns:
        raise ValueError("train, dev, and test turns are required")

    empty_guidance = ConversationGuidance(act_keywords={})
    selected_guidance = empty_guidance
    iteration_reports: list[dict[str, object]] = []

    for iteration in range(1, iterations + 1):
        train_rows = score_conversation_turns(
            train_turns,
            top_k=top_k,
            guidance=selected_guidance,
        )
        train_metrics = summarize_conversation_rows(train_rows, train_turns)
        candidate_guidance = learn_conversation_guidance(
            train_turns,
            train_rows,
            selected_guidance,
        )
        candidate_train_rows = score_conversation_turns(
            train_turns,
            top_k=top_k,
            guidance=candidate_guidance,
        )
        candidate_train_metrics = summarize_conversation_rows(
            candidate_train_rows,
            train_turns,
        )
        dev_rows = score_conversation_turns(
            dev_turns,
            top_k=top_k,
            guidance=selected_guidance,
        )
        dev_metrics = summarize_conversation_rows(dev_rows, dev_turns)
        candidate_dev_rows = score_conversation_turns(
            dev_turns,
            top_k=top_k,
            guidance=candidate_guidance,
        )
        candidate_dev_metrics = summarize_conversation_rows(
            candidate_dev_rows,
            dev_turns,
        )
        dev_segment_regressions = find_conversation_act_segment_regressions(
            baseline_rows=dev_rows,
            candidate_rows=candidate_dev_rows,
        )
        dev_promote_guidance = should_promote_conversation_guidance(
            current_dev=dev_metrics,
            candidate_dev=candidate_dev_metrics,
            dev_segment_regressions=dev_segment_regressions,
        )
        iteration_reports.append(
            {
                "iteration": iteration,
                "train": {
                    "selected": train_metrics,
                    "candidate": candidate_train_metrics,
                },
                "dev": {
                    "selected": dev_metrics,
                    "candidate": candidate_dev_metrics,
                },
                "dev_promote_guidance": dev_promote_guidance,
                "dev_segment_regressions": dev_segment_regressions,
                "selected_guidance": selected_guidance.to_dict(),
                "candidate_guidance": candidate_guidance.to_dict(),
            }
        )
        if dev_promote_guidance:
            selected_guidance = candidate_guidance

    train_baseline_rows = score_conversation_turns(
        train_turns,
        top_k=top_k,
        guidance=empty_guidance,
    )
    train_guided_rows = score_conversation_turns(
        train_turns,
        top_k=top_k,
        guidance=selected_guidance,
    )
    dev_baseline_rows = score_conversation_turns(
        dev_turns,
        top_k=top_k,
        guidance=empty_guidance,
    )
    dev_guided_rows = score_conversation_turns(
        dev_turns,
        top_k=top_k,
        guidance=selected_guidance,
    )
    test_baseline_rows = score_conversation_turns(
        test_turns,
        top_k=top_k,
        guidance=empty_guidance,
    )
    test_guided_rows = score_conversation_turns(
        test_turns,
        top_k=top_k,
        guidance=selected_guidance,
    )
    train_baseline = summarize_conversation_rows(train_baseline_rows, train_turns)
    train_guided = summarize_conversation_rows(train_guided_rows, train_turns)
    dev_baseline = summarize_conversation_rows(dev_baseline_rows, dev_turns)
    dev_guided = summarize_conversation_rows(dev_guided_rows, dev_turns)
    test_baseline = summarize_conversation_rows(test_baseline_rows, test_turns)
    test_guided = summarize_conversation_rows(test_guided_rows, test_turns)

    return {
        "summary": {
            "train_turns": len(train_turns),
            "dev_turns": len(dev_turns),
            "test_turns": len(test_turns),
            "iterations": iterations,
            "top_k": top_k,
        },
        "iterations": iteration_reports,
        "train": {
            "baseline": train_baseline,
            "guided": train_guided,
        },
        "dev": {
            "baseline": dev_baseline,
            "guided": dev_guided,
        },
        "test": {
            "baseline": test_baseline,
            "guided": test_guided,
        },
        "efficacy": compare_conversation_efficacy(
            train_baseline=train_baseline,
            train_guided=train_guided,
            dev_baseline=dev_baseline,
            dev_guided=dev_guided,
            test_baseline=test_baseline,
            test_guided=test_guided,
        ),
        "analytics": {
            "train_segments": summarize_conversation_segments(
                train_baseline_rows,
                train_guided_rows,
            ),
            "dev_segments": summarize_conversation_segments(
                dev_baseline_rows,
                dev_guided_rows,
            ),
            "test_segments": summarize_conversation_segments(
                test_baseline_rows,
                test_guided_rows,
            ),
        },
        "guidance_delta": compare_conversation_guidance_delta(
            test_baseline_rows,
            test_guided_rows,
        ),
        "final_guidance": selected_guidance.to_dict(),
    }


def score_conversation_turns(
    turns: tuple[ConversationTurn, ...],
    top_k: int,
    guidance: ConversationGuidance,
    act_ranker: ConversationActRanker | None = None,
    learned_weight: float = 0.0,
    scoring_variant: str = "heuristic",
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for turn in turns:
        pack = build_probability_pack(
            turn,
            top_k=top_k,
            guidance=guidance,
            act_ranker=act_ranker,
            learned_weight=learned_weight,
            scoring_variant=scoring_variant,
        )
        rank_1 = pack.top_branches[0]
        rows.append(
            {
                "turn_id": turn.turn_id,
                "expected_act": turn.expected_act,
                "expected_emotion": turn.expected_emotion,
                "next_speaker": turn.next_speaker,
                "rank_1_act": rank_1["act"],
                "top_acts": [branch["act"] for branch in pack.top_branches],
                "tts_ready": bool(pack.prepared_drafts),
                "latency_ms": 90 if pack.prepared_drafts else turn.latency_budget_ms,
            }
        )
    return tuple(rows)


def run_conversation_act_ranker_bakeoff(
    train_turns: tuple[ConversationTurn, ...],
    dev_turns: tuple[ConversationTurn, ...],
    test_turns: tuple[ConversationTurn, ...],
    top_k: int = 3,
) -> dict[str, object]:
    if not train_turns or not dev_turns or not test_turns:
        raise ValueError("train, dev, and test turns are required")

    ranker = train_conversation_act_ranker(train_turns)
    empty_guidance = ConversationGuidance(act_keywords={})
    train_baseline_rows = score_conversation_turns(
        train_turns,
        top_k=top_k,
        guidance=empty_guidance,
    )
    dev_baseline_rows = score_conversation_turns(
        dev_turns,
        top_k=top_k,
        guidance=empty_guidance,
    )
    test_baseline_rows = score_conversation_turns(
        test_turns,
        top_k=top_k,
        guidance=empty_guidance,
    )
    train_baseline = summarize_conversation_rows(train_baseline_rows, train_turns)
    dev_baseline = summarize_conversation_rows(dev_baseline_rows, dev_turns)
    test_baseline = summarize_conversation_rows(test_baseline_rows, test_turns)

    variants: dict[str, dict[str, object]] = {}
    for name, learned_weight in conversation_bakeoff_variants():
        train_rows = score_conversation_turns(
            train_turns,
            top_k=top_k,
            guidance=empty_guidance,
            act_ranker=ranker,
            learned_weight=learned_weight,
            scoring_variant=name,
        )
        dev_rows = score_conversation_turns(
            dev_turns,
            top_k=top_k,
            guidance=empty_guidance,
            act_ranker=ranker,
            learned_weight=learned_weight,
            scoring_variant=name,
        )
        test_rows = score_conversation_turns(
            test_turns,
            top_k=top_k,
            guidance=empty_guidance,
            act_ranker=ranker,
            learned_weight=learned_weight,
            scoring_variant=name,
        )
        variants[name] = {
            "learned_weight": learned_weight,
            "train": summarize_conversation_rows(train_rows, train_turns),
            "dev": summarize_conversation_rows(dev_rows, dev_turns),
            "test": summarize_conversation_rows(test_rows, test_turns),
            "dev_segment_regressions": find_conversation_act_segment_regressions(
                baseline_rows=dev_baseline_rows,
                candidate_rows=dev_rows,
            ),
            "test_segment_regressions": find_conversation_act_segment_regressions(
                baseline_rows=test_baseline_rows,
                candidate_rows=test_rows,
            ),
        }

    selected_name = select_conversation_bakeoff_variant(variants)
    selected = variants[selected_name]
    selected_weight = float(selected["learned_weight"])
    selected_train_rows = score_conversation_turns(
        train_turns,
        top_k=top_k,
        guidance=empty_guidance,
        act_ranker=ranker,
        learned_weight=selected_weight,
        scoring_variant=selected_name,
    )
    selected_dev_rows = score_conversation_turns(
        dev_turns,
        top_k=top_k,
        guidance=empty_guidance,
        act_ranker=ranker,
        learned_weight=selected_weight,
        scoring_variant=selected_name,
    )
    selected_test_rows = score_conversation_turns(
        test_turns,
        top_k=top_k,
        guidance=empty_guidance,
        act_ranker=ranker,
        learned_weight=selected_weight,
        scoring_variant=selected_name,
    )
    train_guided = summarize_conversation_rows(selected_train_rows, train_turns)
    dev_guided = summarize_conversation_rows(selected_dev_rows, dev_turns)
    test_guided = summarize_conversation_rows(selected_test_rows, test_turns)

    return {
        "summary": {
            "train_turns": len(train_turns),
            "dev_turns": len(dev_turns),
            "test_turns": len(test_turns),
            "top_k": top_k,
        },
        "ranker": ranker.to_dict(),
        "variants": variants,
        "selected_variant": {
            "name": selected_name,
            "learned_weight": selected_weight,
            "train": train_guided,
            "dev": dev_guided,
            "test": test_guided,
            "dev_segment_regressions": selected["dev_segment_regressions"],
            "test_segment_regressions": selected["test_segment_regressions"],
        },
        "train": {
            "baseline": train_baseline,
            "guided": train_guided,
        },
        "dev": {
            "baseline": dev_baseline,
            "guided": dev_guided,
        },
        "test": {
            "baseline": test_baseline,
            "guided": test_guided,
        },
        "efficacy": compare_conversation_efficacy(
            train_baseline=train_baseline,
            train_guided=train_guided,
            dev_baseline=dev_baseline,
            dev_guided=dev_guided,
            test_baseline=test_baseline,
            test_guided=test_guided,
        ),
        "analytics": {
            "train_segments": summarize_conversation_segments(
                train_baseline_rows,
                selected_train_rows,
            ),
            "dev_segments": summarize_conversation_segments(
                dev_baseline_rows,
                selected_dev_rows,
            ),
            "test_segments": summarize_conversation_segments(
                test_baseline_rows,
                selected_test_rows,
            ),
        },
        "guidance_delta": compare_conversation_guidance_delta(
            test_baseline_rows,
            selected_test_rows,
        ),
    }


def conversation_bakeoff_variants() -> tuple[tuple[str, float], ...]:
    return (
        ("heuristic", 0.0),
        ("hybrid_25", 0.25),
        ("hybrid_50", 0.5),
        ("hybrid_75", 0.75),
        ("learned", 1.0),
    )


def select_conversation_bakeoff_variant(
    variants: dict[str, dict[str, object]],
) -> str:
    safe_variants = {
        name: row
        for name, row in variants.items()
        if not row["dev_segment_regressions"]
    }
    candidates = safe_variants if safe_variants else variants
    return sorted(
        candidates,
        key=lambda name: (
            float(candidates[name]["dev"]["p_at_1"]),
            float(candidates[name]["dev"]["top_3_recall"]),
            -len(candidates[name]["dev_segment_regressions"]),
            -float(candidates[name]["learned_weight"]),
        ),
        reverse=True,
    )[0]


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


def should_promote_conversation_guidance(
    current_dev: dict[str, float | int],
    candidate_dev: dict[str, float | int],
    dev_segment_regressions: list[dict[str, object]],
) -> bool:
    return (
        not dev_segment_regressions
        and float(candidate_dev["p_at_1"]) > float(current_dev["p_at_1"])
        and float(candidate_dev["top_3_recall"]) >= float(current_dev["top_3_recall"])
        and float(candidate_dev["tts_readiness_rate"]) >= float(current_dev["tts_readiness_rate"])
    )


def find_conversation_act_segment_regressions(
    baseline_rows: tuple[dict[str, object], ...],
    candidate_rows: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    regressions: list[dict[str, object]] = []
    acts = sorted({str(row["expected_act"]) for row in baseline_rows})
    for act in acts:
        baseline = summarize_conversation_row_subset(
            tuple(row for row in baseline_rows if str(row["expected_act"]) == act)
        )
        candidate = summarize_conversation_row_subset(
            tuple(row for row in candidate_rows if str(row["expected_act"]) == act)
        )
        delta = round(float(candidate["p_at_1"]) - float(baseline["p_at_1"]), 3)
        if delta < 0:
            regressions.append(
                {
                    "segment": "expected_act",
                    "name": act,
                    "baseline_p_at_1": baseline["p_at_1"],
                    "candidate_p_at_1": candidate["p_at_1"],
                    "p_at_1_delta": delta,
                }
            )
    return regressions


def compare_conversation_efficacy(
    train_baseline: dict[str, float | int],
    train_guided: dict[str, float | int],
    dev_baseline: dict[str, float | int],
    dev_guided: dict[str, float | int],
    test_baseline: dict[str, float | int],
    test_guided: dict[str, float | int],
) -> dict[str, float]:
    train_p_gain = conversation_metric_gain(train_baseline, train_guided, "p_at_1")
    dev_p_gain = conversation_metric_gain(dev_baseline, dev_guided, "p_at_1")
    test_p_gain = conversation_metric_gain(test_baseline, test_guided, "p_at_1")
    return {
        "train_p_at_1_gain": train_p_gain,
        "dev_p_at_1_gain": dev_p_gain,
        "test_p_at_1_gain": test_p_gain,
        "train_top_3_recall_gain": conversation_metric_gain(
            train_baseline,
            train_guided,
            "top_3_recall",
        ),
        "dev_top_3_recall_gain": conversation_metric_gain(
            dev_baseline,
            dev_guided,
            "top_3_recall",
        ),
        "test_top_3_recall_gain": conversation_metric_gain(
            test_baseline,
            test_guided,
            "top_3_recall",
        ),
        "train_tts_readiness_gain": conversation_metric_gain(
            train_baseline,
            train_guided,
            "tts_readiness_rate",
        ),
        "dev_tts_readiness_gain": conversation_metric_gain(
            dev_baseline,
            dev_guided,
            "tts_readiness_rate",
        ),
        "test_tts_readiness_gain": conversation_metric_gain(
            test_baseline,
            test_guided,
            "tts_readiness_rate",
        ),
        "train_test_p_at_1_gap": round(train_p_gain - test_p_gain, 3),
        "dev_test_p_at_1_gap": round(dev_p_gain - test_p_gain, 3),
    }


def conversation_metric_gain(
    baseline: dict[str, float | int],
    guided: dict[str, float | int],
    metric: str,
) -> float:
    return round(float(guided[metric]) - float(baseline[metric]), 3)


def summarize_conversation_segments(
    baseline_rows: tuple[dict[str, object], ...],
    guided_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    dimensions = {
        "expected_act": sorted({str(row["expected_act"]) for row in baseline_rows}),
        "expected_emotion": sorted({str(row["expected_emotion"]) for row in baseline_rows}),
        "next_speaker": sorted({str(row["next_speaker"]) for row in baseline_rows}),
    }
    summary: dict[str, object] = {
        dimension: {
            value: {
                "baseline": summarize_conversation_row_subset(
                    tuple(row for row in baseline_rows if str(row[dimension]) == value)
                ),
                "guided": summarize_conversation_row_subset(
                    tuple(row for row in guided_rows if str(row[dimension]) == value)
                ),
            }
            for value in values
        }
        for dimension, values in dimensions.items()
    }
    summary["focus_areas"] = conversation_focus_areas(summary)
    return summary


def summarize_conversation_row_subset(
    rows: tuple[dict[str, object], ...],
) -> dict[str, float | int]:
    total = max(len(rows), 1)
    exact = sum(row["rank_1_act"] == row["expected_act"] for row in rows)
    top_3 = sum(row["expected_act"] in row["top_acts"] for row in rows)
    tts_ready = sum(bool(row["tts_ready"]) for row in rows)
    return {
        "total_turns": len(rows),
        "p_at_1": round(exact / total, 3),
        "top_3_recall": round(top_3 / total, 3),
        "tts_readiness_rate": round(tts_ready / total, 3),
    }


def conversation_focus_areas(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dimension, dimension_rows in summary.items():
        if dimension == "focus_areas" or not isinstance(dimension_rows, dict):
            continue
        for value, metrics in dimension_rows.items():
            if not isinstance(metrics, dict):
                continue
            baseline = metrics["baseline"]
            guided = metrics["guided"]
            if not isinstance(baseline, dict) or not isinstance(guided, dict):
                continue
            rows.append(
                {
                    "segment": dimension,
                    "name": value,
                    "guided_p_at_1": guided["p_at_1"],
                    "p_at_1_gain": round(
                        float(guided["p_at_1"]) - float(baseline["p_at_1"]),
                        3,
                    ),
                    "guided_top_3_recall": guided["top_3_recall"],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            float(row["guided_p_at_1"]),
            float(row["guided_top_3_recall"]),
            float(row["p_at_1_gain"]),
        ),
    )[:10]


def compare_conversation_guidance_delta(
    baseline_rows: tuple[dict[str, object], ...],
    guided_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    baseline = {
        str(row["turn_id"]): row["rank_1_act"] == row["expected_act"]
        for row in baseline_rows
    }
    guided = {
        str(row["turn_id"]): row["rank_1_act"] == row["expected_act"]
        for row in guided_rows
    }
    turn_ids = sorted(set(baseline) | set(guided))
    improved = [
        turn_id
        for turn_id in turn_ids
        if not baseline.get(turn_id, False) and guided.get(turn_id, False)
    ]
    regressed = [
        turn_id
        for turn_id in turn_ids
        if baseline.get(turn_id, False) and not guided.get(turn_id, False)
    ]
    unchanged = [
        turn_id
        for turn_id in turn_ids
        if baseline.get(turn_id, False) == guided.get(turn_id, False)
    ]
    return {
        "improved_turns": improved,
        "regressed_turns": regressed,
        "unchanged_turns": unchanged,
        "improved_turn_count": len(improved),
        "regressed_turn_count": len(regressed),
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
    token_counts = conversation_token_act_counts(turns)
    turns_by_id = {turn.turn_id: turn for turn in turns}
    for row in rows:
        if row["rank_1_act"] == row["expected_act"]:
            continue
        turn = turns_by_id[str(row["turn_id"])]
        act = turn.expected_act
        keywords.setdefault(act, [])
        turn_tokens = sorted(
            normalized_tokens(turn.context_text()),
            key=lambda token: (
                -conversation_token_discrimination_score(token, act, token_counts),
                -token_counts.get(token, {}).get(act, 0),
                token,
            ),
        )
        for token in turn_tokens:
            if (
                len(token) < 4
                or token in CONVERSATION_GUIDANCE_STOP_WORDS
                or token in keywords[act]
                or conversation_token_discrimination_score(token, act, token_counts) <= 0
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


def conversation_token_act_counts(
    turns: tuple[ConversationTurn, ...],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for turn in turns:
        for token in normalized_tokens(turn.context_text()):
            counts.setdefault(token, {})
            counts[token][turn.expected_act] = counts[token].get(turn.expected_act, 0) + 1
    return counts


def conversation_token_discrimination_score(
    token: str,
    target_act: str,
    token_counts: dict[str, dict[str, int]],
) -> int:
    counts = token_counts.get(token, {})
    target_count = counts.get(target_act, 0)
    other_count = max(
        (count for act, count in counts.items() if act != target_act),
        default=0,
    )
    return target_count - other_count
