from __future__ import annotations

import csv
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

EMPATHETIC_EMOTION_MAP = {
    "afraid": "fear",
    "angry": "anger",
    "annoyed": "anger",
    "anticipating": "surprise",
    "anxious": "fear",
    "apprehensive": "fear",
    "ashamed": "sadness",
    "caring": "happiness",
    "confident": "happiness",
    "content": "happiness",
    "devastated": "sadness",
    "disappointed": "sadness",
    "disgusted": "disgust",
    "embarrassed": "sadness",
    "excited": "happiness",
    "faithful": "happiness",
    "furious": "anger",
    "grateful": "happiness",
    "guilty": "sadness",
    "hopeful": "happiness",
    "impressed": "happiness",
    "jealous": "anger",
    "joyful": "happiness",
    "lonely": "sadness",
    "nostalgic": "sadness",
    "prepared": "happiness",
    "proud": "happiness",
    "sad": "sadness",
    "sentimental": "sadness",
    "surprised": "surprise",
    "terrified": "fear",
    "trusting": "happiness",
}

CONVERSATION_ACTS = ("inform", "question", "directive", "commissive")
MAX_CONVERSATION_BAKEOFF_TRAIN_DEV_GAP = 0.12
MAX_CONVERSATION_SPECIALIST_TRAIN_DEV_GAP = 0.14
CONVERSATION_BAKEOFF_DEV_TIE_MARGIN = 0.005
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

RESPONSE_MODES = (
    "ask_followup",
    "validate",
    "reassure",
    "disclose",
    "suggest",
    "encourage",
    "inform",
    "commit",
    "apologize",
    "redirect",
    "other",
)

BASE_RESPONSE_MODE_PRIORS = {
    "ask_followup": 0.24,
    "validate": 0.22,
    "reassure": 0.2,
    "suggest": 0.18,
    "inform": 0.16,
    "disclose": 0.12,
    "encourage": 0.1,
    "commit": 0.08,
    "redirect": 0.07,
    "apologize": 0.05,
    "other": 0.04,
}

RESPONSE_MODE_TEMPLATES = {
    "ask_followup": "I can ask one warm follow-up question.",
    "validate": "I can reflect the feeling back clearly and gently.",
    "reassure": "I can offer grounded reassurance without overpromising.",
    "disclose": "I can share a brief relatable disclosure when it is useful.",
    "suggest": "I can offer one practical suggestion.",
    "encourage": "I can encourage the next small step.",
    "inform": "I can provide relevant information.",
    "commit": "I can commit to a helpful next action.",
    "apologize": "I can acknowledge harm or frustration with care.",
    "redirect": "I can redirect toward a safer or more useful path.",
    "other": "I can keep a neutral supportive response ready.",
}

ESCONV_STRATEGY_RESPONSE_MODES = {
    "question": "ask_followup",
    "affirmation and reassurance": "reassure",
    "reflection of feelings": "validate",
    "restatement or paraphrasing": "validate",
    "self-disclosure": "disclose",
    "providing suggestions": "suggest",
    "information": "inform",
    "others": "other",
    "other": "other",
}

ESCONV_EMOTION_MAP = {
    "anger": "anger",
    "anxiety": "fear",
    "depression": "sadness",
    "disgust": "disgust",
    "fear": "fear",
    "guilt": "sadness",
    "jealousy": "anger",
    "nervousness": "fear",
    "pain": "sadness",
    "sadness": "sadness",
    "shame": "sadness",
}


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    conversation: tuple[Message, ...]
    next_speaker: str
    actual_next_utterance: str
    expected_act: str
    expected_emotion: str
    expected_response_mode: str = "inform"
    observed_acts: tuple[str, ...] = tuple()
    observed_response_modes: tuple[str, ...] = tuple()
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
            expected_response_mode=str(
                row.get(
                    "expected_response_mode",
                    response_mode_from_act(str(row["expected_act"])),
                )
            ),
            observed_acts=tuple(str(act) for act in row.get("observed_acts", [])),
            observed_response_modes=tuple(
                str(mode) for mode in row.get("observed_response_modes", [])
            ),
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
class ConversationTransitionRanker:
    transitions: dict[str, dict[str, float]]
    global_scores: dict[str, float]

    def score(self, turn: ConversationTurn) -> dict[str, float]:
        previous_act = turn.observed_acts[-1] if turn.observed_acts else "__none__"
        return self.transitions.get(previous_act, self.global_scores)

    def to_dict(self) -> dict[str, object]:
        return {
            "transition_count": len(self.transitions),
            "observed_previous_acts": sorted(self.transitions),
        }


@dataclass(frozen=True)
class ConversationHistoryRanker:
    window_size: int
    transitions: dict[tuple[str, ...], dict[str, float]]
    global_scores: dict[str, float]

    def score(self, turn: ConversationTurn) -> dict[str, float]:
        key = tuple(turn.observed_acts[-self.window_size:])
        return self.transitions.get(key, self.global_scores)

    def to_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "transition_count": len(self.transitions),
            "observed_history_lengths": sorted(
                {len(history) for history in self.transitions}
            ),
        }


@dataclass(frozen=True)
class ConversationQuestionEvidenceRanker:
    feature_log_odds: dict[str, float]
    question_turn_count: int
    non_question_turn_count: int

    def score(self, turn: ConversationTurn) -> float:
        evidence = sorted(
            (
                self.feature_log_odds.get(feature, 0.0)
                for feature in conversation_features(turn)
            ),
            reverse=True,
        )
        return round(sum(score for score in evidence[:4] if score > 0), 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_count": len(self.feature_log_odds),
            "question_turn_count": self.question_turn_count,
            "non_question_turn_count": self.non_question_turn_count,
        }


@dataclass(frozen=True)
class ResponseModeRanker:
    mode_log_priors: dict[str, float]
    feature_log_likelihoods: dict[str, dict[str, float]]
    default_feature_log_likelihoods: dict[str, float]
    vocabulary: tuple[str, ...]
    mode_counts: dict[str, int]

    def score(self, turn: ConversationTurn) -> dict[str, float]:
        features = conversation_features(turn)
        return {
            mode: self.mode_log_priors[mode]
            + sum(
                self.feature_log_likelihoods.get(mode, {}).get(
                    feature,
                    self.default_feature_log_likelihoods[mode],
                )
                for feature in features
            )
            for mode in RESPONSE_MODES
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "mode_counts": dict(sorted(self.mode_counts.items())),
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


def load_esconv_export(path: Path, split: str = "train") -> tuple[ConversationTurn, ...]:
    if split not in {"train", "validation", "dev", "test", "all"}:
        raise ValueError("split must be train, validation, dev, test, or all")

    conversations = json.loads(path.read_text(encoding="utf-8"))
    selected = select_esconv_conversations(conversations, split)
    turns: list[ConversationTurn] = []
    split_name = "validation" if split == "dev" else split

    for conversation_index, conversation in selected:
        dialog = conversation.get("dialog", [])
        observed_modes: list[str] = []
        observed_acts: list[str] = []
        supporter_index = 0
        emotion = esconv_emotion(conversation.get("emotion_type", ""))
        for dialog_index, row in enumerate(dialog):
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            speaker = str(row.get("speaker", "")).strip().lower()
            annotation = row.get("annotation", {})
            strategy = ""
            if isinstance(annotation, dict):
                strategy = str(annotation.get("strategy", "")).strip()
            act = infer_conversation_act(content)

            if speaker == "supporter" and strategy:
                supporter_index += 1
                mode = esconv_strategy_response_mode(strategy)
                history = tuple(
                    Message(
                        role=str(prior.get("speaker", "unknown")).strip().lower()
                        or "unknown",
                        content=str(prior.get("content", "")).strip(),
                    )
                    for prior in dialog[:dialog_index]
                    if str(prior.get("content", "")).strip()
                )
                if history:
                    turns.append(
                        ConversationTurn(
                            turn_id=(
                                f"esconv-{split_name}-{conversation_index + 1:04d}-"
                                f"{supporter_index:03d}"
                            ),
                            conversation=history,
                            next_speaker="supporter",
                            actual_next_utterance=content,
                            expected_act=act,
                            expected_emotion=emotion,
                            expected_response_mode=mode,
                            observed_acts=tuple(observed_acts),
                            observed_response_modes=tuple(observed_modes),
                        )
                    )
                observed_modes.append(mode)

            observed_acts.append(act)

    return tuple(turns)


def select_esconv_conversations(
    conversations: list[dict[str, Any]],
    split: str,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    if split == "all":
        return tuple(enumerate(conversations))
    normalized_split = "validation" if split == "dev" else split
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, conversation in enumerate(conversations):
        bucket = index % 10
        conversation_split = (
            "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        )
        if conversation_split == normalized_split:
            selected.append((index, conversation))
    return tuple(selected)


def esconv_strategy_response_mode(strategy: str) -> str:
    key = " ".join(strategy.strip().lower().split())
    return ESCONV_STRATEGY_RESPONSE_MODES.get(key, "other")


def esconv_emotion(emotion_type: str) -> str:
    return ESCONV_EMOTION_MAP.get(emotion_type.strip().lower(), "no_emotion")


def load_empatheticdialogues_export(path: Path) -> tuple[ConversationTurn, ...]:
    rows = load_empatheticdialogues_rows(path)
    conversations: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        conv_id = row.get("conv_id", "").strip()
        if not conv_id:
            continue
        conversations.setdefault(conv_id, []).append(row)

    turns: list[ConversationTurn] = []
    for conv_id, conversation_rows in sorted(conversations.items()):
        ordered = sorted(
            conversation_rows,
            key=lambda row: int(row.get("utterance_idx", "0") or 0),
        )
        utterances = [decode_empathetic_text(row.get("utterance", "")) for row in ordered]
        acts = tuple(infer_conversation_act(utterance) for utterance in utterances)
        for next_index in range(1, len(ordered)):
            history = tuple(
                Message(
                    role=empathetic_speaker_role(ordered[index], index),
                    content=utterances[index],
                )
                for index in range(next_index)
                if utterances[index]
            )
            if not history or not utterances[next_index]:
                continue
            next_utterance_idx = int(
                ordered[next_index].get("utterance_idx", str(next_index + 1)) or next_index + 1
            )
            turns.append(
                ConversationTurn(
                    turn_id=f"empathetic-{conv_id}-{next_utterance_idx:03d}",
                    conversation=history,
                    next_speaker=empathetic_speaker_role(
                        ordered[next_index],
                        next_index,
                    ),
                    actual_next_utterance=utterances[next_index],
                    expected_act=acts[next_index],
                    expected_emotion=empathetic_emotion(
                        ordered[next_index].get("context", ""),
                    ),
                    expected_response_mode=response_mode_from_act(acts[next_index]),
                    observed_acts=acts[:next_index],
                    observed_response_modes=tuple(
                        response_mode_from_act(act)
                        for act in acts[:next_index]
                    ),
                )
            )
    return tuple(turns)


def load_empatheticdialogues_rows(path: Path) -> tuple[dict[str, str], ...]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(
                        {
                            str(key): str(value)
                            for key, value in json.loads(line).items()
                        }
                    )
        return tuple(rows)

    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(
            {
                str(key): str(value)
                for key, value in row.items()
                if key is not None and value is not None
            }
            for row in csv.DictReader(handle)
        )


def decode_empathetic_text(value: str) -> str:
    return value.replace("_comma_", ",").strip()


def empathetic_speaker_role(row: dict[str, str], index: int) -> str:
    speaker_idx = row.get("speaker_idx", "")
    try:
        parsed = int(speaker_idx)
    except ValueError:
        parsed = index
    return speaker_role(parsed)


def empathetic_emotion(context: str) -> str:
    return EMPATHETIC_EMOTION_MAP.get(context.strip().lower(), "no_emotion")


def infer_conversation_act(utterance: str) -> str:
    stripped = utterance.strip()
    lowered = stripped.lower()
    tokens = normalized_tokens(stripped)
    if stripped.endswith("?"):
        return "question"
    if tokens & {"please", "should", "check", "try", "tell", "make", "go", "stop"}:
        return "directive"
    if lowered.startswith(
        (
            "i can ",
            "i will ",
            "i'll ",
            "i would ",
            "i'd ",
            "i am going to ",
            "i'm going to ",
            "we can ",
            "we will ",
            "we'll ",
            "we would ",
            "we're going to ",
        )
    ):
        return "commissive"
    return "inform"


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
                    expected_response_mode=response_mode_from_act(
                        DAILYDIALOG_ACTS.get(acts[next_index], "inform")
                    ),
                    observed_acts=tuple(
                        DAILYDIALOG_ACTS.get(act, "inform")
                        for act in acts[:next_index]
                    ),
                    observed_response_modes=tuple(
                        response_mode_from_act(DAILYDIALOG_ACTS.get(act, "inform"))
                        for act in acts[:next_index]
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
            expected_response_mode=turn.expected_response_mode,
            observed_acts=turn.observed_acts,
            observed_response_modes=turn.observed_response_modes,
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
        "expected_response_mode": turn.expected_response_mode,
        "observed_acts": list(turn.observed_acts),
        "observed_response_modes": list(turn.observed_response_modes),
        "latency_budget_ms": turn.latency_budget_ms,
    }


def response_mode_from_act(act: str) -> str:
    return {
        "question": "ask_followup",
        "directive": "suggest",
        "commissive": "commit",
        "inform": "inform",
    }.get(act, "other")


def speaker_role(index: int) -> str:
    return "speaker_a" if index % 2 == 0 else "speaker_b"


def build_probability_pack(
    turn: ConversationTurn,
    top_k: int = 3,
    guidance: ConversationGuidance | None = None,
    act_ranker: ConversationActRanker | None = None,
    learned_weight: float = 0.0,
    transition_ranker: ConversationTransitionRanker | None = None,
    transition_weight: float = 0.0,
    transition_overlay_act: str | None = None,
    transition_overlay_margin: float = 0.0,
    transition_protected_acts: tuple[str, ...] = tuple(),
    history_ranker: ConversationHistoryRanker | None = None,
    history_margin: float = 0.0,
    history_overlay_acts: tuple[str, ...] = tuple(),
    history_preserved_acts: tuple[str, ...] = tuple(),
    question_evidence_ranker: ConversationQuestionEvidenceRanker | None = None,
    question_evidence_margin: float = 0.0,
    question_evidence_preserved_acts: tuple[str, ...] = tuple(),
    scoring_variant: str = "heuristic",
) -> ConversationProbabilityPack:
    branches = generate_conversation_branches(
        turn,
        top_k=top_k,
        guidance=guidance,
        act_ranker=act_ranker,
        learned_weight=learned_weight,
        transition_ranker=transition_ranker,
        transition_weight=transition_weight,
        transition_overlay_act=transition_overlay_act,
        transition_overlay_margin=transition_overlay_margin,
        transition_protected_acts=transition_protected_acts,
        history_ranker=history_ranker,
        history_margin=history_margin,
        history_overlay_acts=history_overlay_acts,
        history_preserved_acts=history_preserved_acts,
        question_evidence_ranker=question_evidence_ranker,
        question_evidence_margin=question_evidence_margin,
        question_evidence_preserved_acts=question_evidence_preserved_acts,
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
    transition_ranker: ConversationTransitionRanker | None = None,
    transition_weight: float = 0.0,
    transition_overlay_act: str | None = None,
    transition_overlay_margin: float = 0.0,
    transition_protected_acts: tuple[str, ...] = tuple(),
    history_ranker: ConversationHistoryRanker | None = None,
    history_margin: float = 0.0,
    history_overlay_acts: tuple[str, ...] = tuple(),
    history_preserved_acts: tuple[str, ...] = tuple(),
    question_evidence_ranker: ConversationQuestionEvidenceRanker | None = None,
    question_evidence_margin: float = 0.0,
    question_evidence_preserved_acts: tuple[str, ...] = tuple(),
    scoring_variant: str = "heuristic",
) -> tuple[dict[str, object], ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if learned_weight < 0 or learned_weight > 1:
        raise ValueError("learned_weight must be between 0 and 1")
    if transition_weight < 0 or transition_weight > 1:
        raise ValueError("transition_weight must be between 0 and 1")
    if transition_overlay_margin < 0:
        raise ValueError("transition_overlay_margin must be non-negative")
    if history_margin < 0:
        raise ValueError("history_margin must be non-negative")
    if question_evidence_margin < 0:
        raise ValueError("question_evidence_margin must be non-negative")
    if any(act not in CONVERSATION_ACTS for act in transition_protected_acts):
        raise ValueError("transition_protected_acts must be known conversation acts")
    if any(act not in CONVERSATION_ACTS for act in history_overlay_acts):
        raise ValueError("history_overlay_acts must be known conversation acts")
    if any(act not in CONVERSATION_ACTS for act in history_preserved_acts):
        raise ValueError("history_preserved_acts must be known conversation acts")
    if any(act not in CONVERSATION_ACTS for act in question_evidence_preserved_acts):
        raise ValueError("question_evidence_preserved_acts must be known conversation acts")
    if scoring_variant == "heuristic" and act_ranker and learned_weight == 1.0:
        scoring_variant = "learned"
    if scoring_variant == "heuristic" and transition_ranker and transition_weight == 1.0:
        scoring_variant = "contextual_transition"

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
    if transition_ranker and transition_weight > 0:
        scores = blended_act_scores(
            heuristic_scores=scores,
            learned_scores=transition_ranker.score(turn),
            learned_weight=transition_weight,
        )
        if transition_protected_acts:
            scores = guarded_transition_scores(
                heuristic_scores=heuristic_act_scores(turn, context_tokens),
                transition_scores=scores,
                protected_acts=transition_protected_acts,
            )
    if history_ranker:
        scores = act_history_overlay_scores(
            current_scores=scores,
            history_scores=history_ranker.score(turn),
            margin_min=history_margin,
            overlay_acts=history_overlay_acts,
            preserved_acts=history_preserved_acts,
        )
    if question_evidence_ranker:
        scores = question_evidence_overlay_scores(
            current_scores=scores,
            question_evidence_score=question_evidence_ranker.score(turn),
            margin_min=question_evidence_margin,
            preserved_acts=question_evidence_preserved_acts,
        )
    if transition_ranker and transition_overlay_act:
        scores = transition_overlay_scores(
            heuristic_scores=scores,
            transition_scores=transition_ranker.score(turn),
            overlay_act=transition_overlay_act,
            margin_min=transition_overlay_margin,
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


def transition_overlay_scores(
    heuristic_scores: dict[str, float],
    transition_scores: dict[str, float],
    overlay_act: str,
    margin_min: float,
) -> dict[str, float]:
    ranked_transition = sorted(
        transition_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ranked_transition) < 2:
        return heuristic_scores
    top_act, top_score = ranked_transition[0]
    second_score = ranked_transition[1][1]
    if top_act != overlay_act or (top_score - second_score) < margin_min:
        return heuristic_scores
    return dict(transition_scores)


def guarded_transition_scores(
    heuristic_scores: dict[str, float],
    transition_scores: dict[str, float],
    protected_acts: tuple[str, ...],
) -> dict[str, float]:
    heuristic = normalize_act_scores(heuristic_scores)
    heuristic_top_act = max(heuristic, key=heuristic.get)
    if heuristic_top_act not in protected_acts:
        return transition_scores

    scores = dict(transition_scores)
    scores[heuristic_top_act] = max(scores.values()) + 0.001
    return scores


def act_history_overlay_scores(
    current_scores: dict[str, float],
    history_scores: dict[str, float],
    margin_min: float,
    overlay_acts: tuple[str, ...] = tuple(),
    preserved_acts: tuple[str, ...] = tuple(),
) -> dict[str, float]:
    ranked_history = sorted(
        history_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ranked_history) < 2:
        return current_scores

    top_act, top_score = ranked_history[0]
    second_score = ranked_history[1][1]
    if (top_score - second_score) < margin_min:
        return current_scores
    if overlay_acts and top_act not in overlay_acts:
        return current_scores
    current_top_act = max(current_scores, key=current_scores.get)
    if current_top_act in preserved_acts and top_act != current_top_act:
        return current_scores

    scores = dict(current_scores)
    scores[top_act] = max(scores.values()) + 0.002
    return scores


def question_evidence_overlay_scores(
    current_scores: dict[str, float],
    question_evidence_score: float,
    margin_min: float,
    preserved_acts: tuple[str, ...] = tuple(),
) -> dict[str, float]:
    if question_evidence_score < margin_min:
        return current_scores
    current_top_act = max(current_scores, key=current_scores.get)
    if current_top_act in preserved_acts and current_top_act != "question":
        return current_scores

    scores = dict(current_scores)
    scores["question"] = max(scores.values()) + 0.003
    return scores


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


def train_conversation_transition_ranker(
    turns: tuple[ConversationTurn, ...],
) -> ConversationTransitionRanker:
    if not turns:
        raise ValueError("training turns are required")

    global_counts = {act: 1 for act in CONVERSATION_ACTS}
    transition_counts: dict[str, dict[str, int]] = {}
    for turn in turns:
        expected = turn.expected_act if turn.expected_act in CONVERSATION_ACTS else "inform"
        previous = turn.observed_acts[-1] if turn.observed_acts else "__none__"
        global_counts[expected] += 1
        transition_counts.setdefault(previous, {act: 1 for act in CONVERSATION_ACTS})
        transition_counts[previous][expected] += 1

    return ConversationTransitionRanker(
        transitions={
            previous: normalize_counts(counts)
            for previous, counts in transition_counts.items()
        },
        global_scores=normalize_counts(global_counts),
    )


def train_conversation_history_ranker(
    turns: tuple[ConversationTurn, ...],
    window_size: int = 4,
) -> ConversationHistoryRanker:
    if not turns:
        raise ValueError("training turns are required")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    global_counts = {act: 1 for act in CONVERSATION_ACTS}
    history_counts: dict[tuple[str, ...], dict[str, int]] = {}
    for turn in turns:
        expected = turn.expected_act if turn.expected_act in CONVERSATION_ACTS else "inform"
        history = tuple(turn.observed_acts[-window_size:])
        global_counts[expected] += 1
        history_counts.setdefault(history, {act: 1 for act in CONVERSATION_ACTS})
        history_counts[history][expected] += 1

    return ConversationHistoryRanker(
        window_size=window_size,
        transitions={
            history: normalize_counts(counts)
            for history, counts in history_counts.items()
        },
        global_scores=normalize_counts(global_counts),
    )


def train_conversation_question_evidence_ranker(
    turns: tuple[ConversationTurn, ...],
) -> ConversationQuestionEvidenceRanker:
    if not turns:
        raise ValueError("training turns are required")

    feature_counts: dict[str, dict[str, int]] = {}
    question_turn_count = 0
    non_question_turn_count = 0
    for turn in turns:
        is_question = turn.expected_act == "question"
        if is_question:
            question_turn_count += 1
        else:
            non_question_turn_count += 1
        bucket = "question" if is_question else "other"
        for feature in conversation_features(turn):
            feature_counts.setdefault(feature, {"question": 0, "other": 0})
            feature_counts[feature][bucket] += 1

    question_denominator = question_turn_count + 2
    other_denominator = non_question_turn_count + 2
    feature_log_odds = {
        feature: math.log((counts["question"] + 1) / question_denominator)
        - math.log((counts["other"] + 1) / other_denominator)
        for feature, counts in feature_counts.items()
    }
    return ConversationQuestionEvidenceRanker(
        feature_log_odds=feature_log_odds,
        question_turn_count=question_turn_count,
        non_question_turn_count=non_question_turn_count,
    )


def train_response_mode_ranker(
    turns: tuple[ConversationTurn, ...],
    class_balanced: bool = False,
) -> ResponseModeRanker:
    if not turns:
        raise ValueError("training turns are required")

    mode_counts = {mode: 0 for mode in RESPONSE_MODES}
    feature_counts = {mode: {} for mode in RESPONSE_MODES}
    feature_totals = {mode: 0 for mode in RESPONSE_MODES}
    vocabulary: set[str] = set()

    for turn in turns:
        mode = (
            turn.expected_response_mode
            if turn.expected_response_mode in RESPONSE_MODES
            else "other"
        )
        mode_counts[mode] += 1
        features = conversation_features(turn)
        vocabulary.update(features)
        for feature in features:
            feature_counts[mode][feature] = feature_counts[mode].get(feature, 0) + 1
            feature_totals[mode] += 1

    total_turns = len(turns)
    vocabulary_size = max(len(vocabulary), 1)
    if class_balanced:
        observed_modes = {mode for mode, count in mode_counts.items() if count > 0}
        observed_prior = math.log(1 / max(len(observed_modes), 1))
        unobserved_prior = math.log(1 / (total_turns + len(RESPONSE_MODES)))
        mode_log_priors = {
            mode: observed_prior if mode in observed_modes else unobserved_prior
            for mode in RESPONSE_MODES
        }
    else:
        mode_log_priors = {
            mode: math.log((mode_counts[mode] + 1) / (total_turns + len(RESPONSE_MODES)))
            for mode in RESPONSE_MODES
        }
    default_feature_log_likelihoods = {
        mode: math.log(1 / (feature_totals[mode] + vocabulary_size))
        for mode in RESPONSE_MODES
    }
    feature_log_likelihoods = {
        mode: {
            feature: math.log(
                (count + 1) / (feature_totals[mode] + vocabulary_size)
            )
            for feature, count in counts.items()
        }
        for mode, counts in feature_counts.items()
    }
    return ResponseModeRanker(
        mode_log_priors=mode_log_priors,
        feature_log_likelihoods=feature_log_likelihoods,
        default_feature_log_likelihoods=default_feature_log_likelihoods,
        vocabulary=tuple(sorted(vocabulary)),
        mode_counts=mode_counts,
    )


def normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {
        act: counts.get(act, 0) / total
        for act in CONVERSATION_ACTS
    }


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
    if turn.observed_response_modes:
        features.add(f"previous_mode_{turn.observed_response_modes[-1]}")
        for mode in turn.observed_response_modes[-3:]:
            features.add(f"recent_mode_{mode}")
    return tuple(sorted(features))


def generate_response_mode_branches(
    turn: ConversationTurn,
    top_k: int = 3,
    ranker: ResponseModeRanker | None = None,
    learned_weight: float = 1.0,
    coverage_modes: tuple[str, ...] = tuple(),
    coverage_min_score: float = 0.0,
    scoring_variant: str = "heuristic_response_mode",
) -> tuple[dict[str, object], ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if learned_weight < 0 or learned_weight > 1:
        raise ValueError("learned_weight must be between 0 and 1")
    if coverage_min_score < 0:
        raise ValueError("coverage_min_score must be non-negative")
    if any(mode not in RESPONSE_MODES for mode in coverage_modes):
        raise ValueError("coverage_modes must be known response modes")
    context_tokens = normalized_tokens(turn.context_text())
    scores = heuristic_response_mode_scores(turn, context_tokens)
    if ranker and learned_weight > 0:
        scores = blended_label_scores(
            baseline_scores=scores,
            learned_scores=ranker.score(turn),
            labels=RESPONSE_MODES,
            learned_weight=learned_weight,
        )
        if scoring_variant == "heuristic_response_mode":
            scoring_variant = "learned_response_mode"

    ranked = apply_response_mode_coverage(
        ranked=sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k],
        scores=scores,
        coverage_modes=coverage_modes,
        coverage_min_score=coverage_min_score,
        top_k=top_k,
    )
    return tuple(
        {
            "branch_id": f"{turn.turn_id}-mode-branch-{index}",
            "rank": index,
            "response_mode": mode,
            "tts_text": RESPONSE_MODE_TEMPLATES[mode],
            "probability": round(min(score, 0.95), 3),
            "trigger_cues": sorted(context_tokens)[:8],
            "scoring_variant": scoring_variant,
        }
        for index, (mode, score) in enumerate(ranked, start=1)
    )


def apply_response_mode_coverage(
    ranked: list[tuple[str, float]],
    scores: dict[str, float],
    coverage_modes: tuple[str, ...],
    coverage_min_score: float,
    top_k: int,
) -> list[tuple[str, float]]:
    if not coverage_modes or top_k < 2 or len(ranked) < top_k:
        return ranked

    ranked_modes = {mode for mode, _score in ranked}
    candidates = [
        (mode, scores[mode])
        for mode in coverage_modes
        if mode not in ranked_modes and scores[mode] >= coverage_min_score
    ]
    if not candidates:
        return ranked

    coverage_mode, coverage_score = max(candidates, key=lambda item: item[1])
    preserved_top = ranked[0]
    replaceable = ranked[1:]
    replace_index, _weakest = min(
        enumerate(replaceable, start=1),
        key=lambda item: item[1][1],
    )
    balanced = list(ranked)
    balanced[replace_index] = (coverage_mode, min(coverage_score, preserved_top[1] - 0.001))
    return sorted(
        balanced,
        key=lambda item: (item[1], item[0] == preserved_top[0]),
        reverse=True,
    )


def heuristic_response_mode_scores(
    turn: ConversationTurn,
    context_tokens: set[str],
) -> dict[str, float]:
    scores = dict(BASE_RESPONSE_MODE_PRIORS)
    latest = turn.conversation[-1].content.lower() if turn.conversation else ""
    if "?" in latest:
        scores["ask_followup"] += 0.1
    if {"sad", "hard", "overwhelmed", "upset", "stressful", "draining"} & context_tokens:
        scores["validate"] += 0.12
        scores["reassure"] += 0.08
    if {"should", "try", "could", "maybe", "plan"} & context_tokens:
        scores["suggest"] += 0.08
    if {"sorry", "apologize", "fault"} & context_tokens:
        scores["apologize"] += 0.1
    return scores


def blended_label_scores(
    baseline_scores: dict[str, float],
    learned_scores: dict[str, float],
    labels: tuple[str, ...],
    learned_weight: float,
) -> dict[str, float]:
    baseline = normalize_label_scores(baseline_scores, labels)
    learned = normalize_label_scores(learned_scores, labels)
    return {
        label: (1 - learned_weight) * baseline[label] + learned_weight * learned[label]
        for label in labels
    }


def normalize_label_scores(
    scores: dict[str, float],
    labels: tuple[str, ...],
) -> dict[str, float]:
    values = [scores[label] for label in labels]
    low = min(values)
    high = max(values)
    if high == low:
        return {label: 0.5 for label in labels}
    return {
        label: (scores[label] - low) / (high - low)
        for label in labels
    }


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
    transition_ranker: ConversationTransitionRanker | None = None,
    transition_weight: float = 0.0,
    transition_overlay_act: str | None = None,
    transition_overlay_margin: float = 0.0,
    transition_protected_acts: tuple[str, ...] = tuple(),
    history_ranker: ConversationHistoryRanker | None = None,
    history_margin: float = 0.0,
    history_overlay_acts: tuple[str, ...] = tuple(),
    history_preserved_acts: tuple[str, ...] = tuple(),
    question_evidence_ranker: ConversationQuestionEvidenceRanker | None = None,
    question_evidence_margin: float = 0.0,
    question_evidence_preserved_acts: tuple[str, ...] = tuple(),
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
            transition_ranker=transition_ranker,
            transition_weight=transition_weight,
            transition_overlay_act=transition_overlay_act,
            transition_overlay_margin=transition_overlay_margin,
            transition_protected_acts=transition_protected_acts,
            history_ranker=history_ranker,
            history_margin=history_margin,
            history_overlay_acts=history_overlay_acts,
            history_preserved_acts=history_preserved_acts,
            question_evidence_ranker=question_evidence_ranker,
            question_evidence_margin=question_evidence_margin,
            question_evidence_preserved_acts=question_evidence_preserved_acts,
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
    transition_ranker = train_conversation_transition_ranker(train_turns)
    history_rankers: dict[int, ConversationHistoryRanker] = {}

    def history_ranker_for(window_size: int) -> ConversationHistoryRanker:
        if window_size <= 0:
            raise ValueError("history_window_size must be positive")
        if window_size not in history_rankers:
            history_rankers[window_size] = train_conversation_history_ranker(
                train_turns,
                window_size=window_size,
            )
        return history_rankers[window_size]

    question_evidence_ranker = train_conversation_question_evidence_ranker(train_turns)
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
    for variant in conversation_bakeoff_variants():
        name = str(variant["name"])
        learned_weight = float(variant["learned_weight"])
        transition_weight = float(variant["transition_weight"])
        transition_overlay_act = variant.get("transition_overlay_act")
        transition_overlay_margin = float(variant.get("transition_overlay_margin", 0.0))
        transition_protected_acts = tuple(
            str(act)
            for act in variant.get("transition_protected_acts", ())
        )
        history_margin = float(variant.get("history_margin", 0.0))
        history_overlay_acts = tuple(
            str(act)
            for act in variant.get("history_overlay_acts", ())
        )
        history_preserved_acts = tuple(
            str(act)
            for act in variant.get("history_preserved_acts", ())
        )
        history_window_size = int(variant.get("history_window_size", 4))
        variant_history_ranker = history_ranker_for(history_window_size)
        use_question_evidence_ranker = bool(
            variant.get("use_question_evidence_ranker", False)
        )
        question_evidence_margin = float(variant.get("question_evidence_margin", 0.0))
        question_evidence_preserved_acts = tuple(
            str(act)
            for act in variant.get("question_evidence_preserved_acts", ())
        )
        use_history_ranker = bool(variant.get("use_history_ranker", False))
        train_rows = score_conversation_turns(
            train_turns,
            top_k=top_k,
            guidance=empty_guidance,
            act_ranker=ranker,
            learned_weight=learned_weight,
            transition_ranker=transition_ranker,
            transition_weight=transition_weight,
            transition_overlay_act=str(transition_overlay_act) if transition_overlay_act else None,
            transition_overlay_margin=transition_overlay_margin,
            transition_protected_acts=transition_protected_acts,
            history_ranker=variant_history_ranker if use_history_ranker else None,
            history_margin=history_margin,
            history_overlay_acts=history_overlay_acts,
            history_preserved_acts=history_preserved_acts,
            question_evidence_ranker=(
                question_evidence_ranker if use_question_evidence_ranker else None
            ),
            question_evidence_margin=question_evidence_margin,
            question_evidence_preserved_acts=question_evidence_preserved_acts,
            scoring_variant=name,
        )
        dev_rows = score_conversation_turns(
            dev_turns,
            top_k=top_k,
            guidance=empty_guidance,
            act_ranker=ranker,
            learned_weight=learned_weight,
            transition_ranker=transition_ranker,
            transition_weight=transition_weight,
            transition_overlay_act=str(transition_overlay_act) if transition_overlay_act else None,
            transition_overlay_margin=transition_overlay_margin,
            transition_protected_acts=transition_protected_acts,
            history_ranker=variant_history_ranker if use_history_ranker else None,
            history_margin=history_margin,
            history_overlay_acts=history_overlay_acts,
            history_preserved_acts=history_preserved_acts,
            question_evidence_ranker=(
                question_evidence_ranker if use_question_evidence_ranker else None
            ),
            question_evidence_margin=question_evidence_margin,
            question_evidence_preserved_acts=question_evidence_preserved_acts,
            scoring_variant=name,
        )
        test_rows = score_conversation_turns(
            test_turns,
            top_k=top_k,
            guidance=empty_guidance,
            act_ranker=ranker,
            learned_weight=learned_weight,
            transition_ranker=transition_ranker,
            transition_weight=transition_weight,
            transition_overlay_act=str(transition_overlay_act) if transition_overlay_act else None,
            transition_overlay_margin=transition_overlay_margin,
            transition_protected_acts=transition_protected_acts,
            history_ranker=variant_history_ranker if use_history_ranker else None,
            history_margin=history_margin,
            history_overlay_acts=history_overlay_acts,
            history_preserved_acts=history_preserved_acts,
            question_evidence_ranker=(
                question_evidence_ranker if use_question_evidence_ranker else None
            ),
            question_evidence_margin=question_evidence_margin,
            question_evidence_preserved_acts=question_evidence_preserved_acts,
            scoring_variant=name,
        )
        cross_validation = (
            cross_validate_conversation_variant(
                train_turns,
                variant,
                fold_count=min(5, len(train_turns)),
                top_k=top_k,
            )
            if len(train_turns) >= 2
            else empty_conversation_cross_validation()
        )
        variants[name] = {
            "learned_weight": learned_weight,
            "transition_weight": transition_weight,
            "transition_overlay_act": transition_overlay_act,
            "transition_overlay_margin": transition_overlay_margin,
            "transition_protected_acts": transition_protected_acts,
            "use_history_ranker": use_history_ranker,
            "history_window_size": history_window_size,
            "history_margin": history_margin,
            "history_overlay_acts": history_overlay_acts,
            "history_preserved_acts": history_preserved_acts,
            "use_question_evidence_ranker": use_question_evidence_ranker,
            "question_evidence_margin": question_evidence_margin,
            "question_evidence_preserved_acts": question_evidence_preserved_acts,
            "train": summarize_conversation_rows(train_rows, train_turns),
            "dev": summarize_conversation_rows(dev_rows, dev_turns),
            "test": summarize_conversation_rows(test_rows, test_turns),
            "cross_validation": cross_validation,
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
    selected_transition_weight = float(selected["transition_weight"])
    selected_overlay_act = selected.get("transition_overlay_act")
    selected_overlay_margin = float(selected.get("transition_overlay_margin", 0.0))
    selected_protected_acts = tuple(
        str(act)
        for act in selected.get("transition_protected_acts", ())
    )
    selected_use_history_ranker = bool(selected.get("use_history_ranker", False))
    selected_history_margin = float(selected.get("history_margin", 0.0))
    selected_history_overlay_acts = tuple(
        str(act)
        for act in selected.get("history_overlay_acts", ())
    )
    selected_history_preserved_acts = tuple(
        str(act)
        for act in selected.get("history_preserved_acts", ())
    )
    selected_history_window_size = int(selected.get("history_window_size", 4))
    selected_history_ranker = history_ranker_for(selected_history_window_size)
    selected_use_question_evidence_ranker = bool(
        selected.get("use_question_evidence_ranker", False)
    )
    selected_question_evidence_margin = float(
        selected.get("question_evidence_margin", 0.0)
    )
    selected_question_evidence_preserved_acts = tuple(
        str(act)
        for act in selected.get("question_evidence_preserved_acts", ())
    )
    selected_train_rows = score_conversation_turns(
        train_turns,
        top_k=top_k,
        guidance=empty_guidance,
        act_ranker=ranker,
        learned_weight=selected_weight,
        transition_ranker=transition_ranker,
        transition_weight=selected_transition_weight,
        transition_overlay_act=str(selected_overlay_act) if selected_overlay_act else None,
        transition_overlay_margin=selected_overlay_margin,
        transition_protected_acts=selected_protected_acts,
        history_ranker=selected_history_ranker if selected_use_history_ranker else None,
        history_margin=selected_history_margin,
        history_overlay_acts=selected_history_overlay_acts,
        history_preserved_acts=selected_history_preserved_acts,
        question_evidence_ranker=(
            question_evidence_ranker if selected_use_question_evidence_ranker else None
        ),
        question_evidence_margin=selected_question_evidence_margin,
        question_evidence_preserved_acts=selected_question_evidence_preserved_acts,
        scoring_variant=selected_name,
    )
    selected_dev_rows = score_conversation_turns(
        dev_turns,
        top_k=top_k,
        guidance=empty_guidance,
        act_ranker=ranker,
        learned_weight=selected_weight,
        transition_ranker=transition_ranker,
        transition_weight=selected_transition_weight,
        transition_overlay_act=str(selected_overlay_act) if selected_overlay_act else None,
        transition_overlay_margin=selected_overlay_margin,
        transition_protected_acts=selected_protected_acts,
        history_ranker=selected_history_ranker if selected_use_history_ranker else None,
        history_margin=selected_history_margin,
        history_overlay_acts=selected_history_overlay_acts,
        history_preserved_acts=selected_history_preserved_acts,
        question_evidence_ranker=(
            question_evidence_ranker if selected_use_question_evidence_ranker else None
        ),
        question_evidence_margin=selected_question_evidence_margin,
        question_evidence_preserved_acts=selected_question_evidence_preserved_acts,
        scoring_variant=selected_name,
    )
    selected_test_rows = score_conversation_turns(
        test_turns,
        top_k=top_k,
        guidance=empty_guidance,
        act_ranker=ranker,
        learned_weight=selected_weight,
        transition_ranker=transition_ranker,
        transition_weight=selected_transition_weight,
        transition_overlay_act=str(selected_overlay_act) if selected_overlay_act else None,
        transition_overlay_margin=selected_overlay_margin,
        transition_protected_acts=selected_protected_acts,
        history_ranker=selected_history_ranker if selected_use_history_ranker else None,
        history_margin=selected_history_margin,
        history_overlay_acts=selected_history_overlay_acts,
        history_preserved_acts=selected_history_preserved_acts,
        question_evidence_ranker=(
            question_evidence_ranker if selected_use_question_evidence_ranker else None
        ),
        question_evidence_margin=selected_question_evidence_margin,
        question_evidence_preserved_acts=selected_question_evidence_preserved_acts,
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
        "transition_ranker": transition_ranker.to_dict(),
        "history_ranker": history_ranker_for(4).to_dict(),
        "history_rankers": {
            str(window_size): ranker.to_dict()
            for window_size, ranker in sorted(history_rankers.items())
        },
        "question_evidence_ranker": question_evidence_ranker.to_dict(),
        "variants": variants,
        "selected_variant": {
            "name": selected_name,
            "learned_weight": selected_weight,
            "transition_weight": selected_transition_weight,
            "transition_overlay_act": selected_overlay_act,
            "transition_overlay_margin": selected_overlay_margin,
            "transition_protected_acts": selected_protected_acts,
            "use_history_ranker": selected_use_history_ranker,
            "history_window_size": selected_history_window_size,
            "history_margin": selected_history_margin,
            "history_overlay_acts": selected_history_overlay_acts,
            "history_preserved_acts": selected_history_preserved_acts,
            "use_question_evidence_ranker": selected_use_question_evidence_ranker,
            "question_evidence_margin": selected_question_evidence_margin,
            "question_evidence_preserved_acts": selected_question_evidence_preserved_acts,
            "cross_validation": selected["cross_validation"],
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


def run_response_mode_ranker_bakeoff(
    train_turns: tuple[ConversationTurn, ...],
    dev_turns: tuple[ConversationTurn, ...],
    test_turns: tuple[ConversationTurn, ...],
    top_k: int = 3,
) -> dict[str, object]:
    if not train_turns or not dev_turns or not test_turns:
        raise ValueError("train, dev, and test turns are required")

    ranker = train_response_mode_ranker(train_turns)
    balanced_prior_ranker = train_response_mode_ranker(
        train_turns,
        class_balanced=True,
    )
    train_baseline_rows = score_response_mode_turns(train_turns, top_k=top_k)
    dev_baseline_rows = score_response_mode_turns(dev_turns, top_k=top_k)
    test_baseline_rows = score_response_mode_turns(test_turns, top_k=top_k)
    variant_rows: dict[str, dict[str, tuple[dict[str, object], ...]]] = {}
    variants: dict[str, dict[str, object]] = {}
    for variant in response_mode_bakeoff_variants():
        name = str(variant["name"])
        learned_weight = float(variant["learned_weight"])
        coverage_modes = tuple(str(mode) for mode in variant.get("coverage_modes", ()))
        coverage_min_score = float(variant.get("coverage_min_score", 0.0))
        class_balanced_prior = bool(variant.get("class_balanced_prior", False))
        variant_ranker = balanced_prior_ranker if class_balanced_prior else ranker
        use_ranker = learned_weight > 0
        train_rows = score_response_mode_turns(
            train_turns,
            top_k=top_k,
            ranker=variant_ranker if use_ranker else None,
            learned_weight=learned_weight,
            coverage_modes=coverage_modes,
            coverage_min_score=coverage_min_score,
            scoring_variant=name,
        )
        dev_rows = score_response_mode_turns(
            dev_turns,
            top_k=top_k,
            ranker=variant_ranker if use_ranker else None,
            learned_weight=learned_weight,
            coverage_modes=coverage_modes,
            coverage_min_score=coverage_min_score,
            scoring_variant=name,
        )
        test_rows = score_response_mode_turns(
            test_turns,
            top_k=top_k,
            ranker=variant_ranker if use_ranker else None,
            learned_weight=learned_weight,
            coverage_modes=coverage_modes,
            coverage_min_score=coverage_min_score,
            scoring_variant=name,
        )
        variant_rows[name] = {
            "train": train_rows,
            "dev": dev_rows,
            "test": test_rows,
        }
        variants[name] = {
            "learned_weight": learned_weight,
            "coverage_modes": coverage_modes,
            "coverage_min_score": coverage_min_score,
            "class_balanced_prior": class_balanced_prior,
            "train": summarize_response_mode_rows(train_rows, train_turns),
            "dev": summarize_response_mode_rows(dev_rows, dev_turns),
            "test": summarize_response_mode_rows(test_rows, test_turns),
            "dev_segment_regressions": find_response_mode_segment_regressions(
                baseline_rows=dev_baseline_rows,
                candidate_rows=dev_rows,
            ),
            "test_segment_regressions": find_response_mode_segment_regressions(
                baseline_rows=test_baseline_rows,
                candidate_rows=test_rows,
            ),
        }
    selected_name = select_response_mode_bakeoff_variant(variants)
    selected_rows = variant_rows[selected_name]["test"]
    selected_train_rows = variant_rows[selected_name]["train"]
    selected_dev_rows = variant_rows[selected_name]["dev"]

    return {
        "summary": {
            "train_turns": len(train_turns),
            "dev_turns": len(dev_turns),
            "test_turns": len(test_turns),
            "top_k": top_k,
        },
        "ranker": ranker.to_dict(),
        "balanced_prior_ranker": balanced_prior_ranker.to_dict(),
        "variants": variants,
        "selected_variant": {
            "name": selected_name,
            **variants[selected_name],
        },
        "promotion": response_mode_promotion_summary(
            selected_name=selected_name,
            baseline=variants["heuristic_response_mode"]["test"],
            selected=summarize_response_mode_rows(selected_rows, test_turns),
            test_segment_regressions=variants[selected_name][
                "test_segment_regressions"
            ],
        ),
        "train": {
            "baseline": variants["heuristic_response_mode"]["train"],
            "guided": summarize_response_mode_rows(selected_train_rows, train_turns),
        },
        "dev": {
            "baseline": variants["heuristic_response_mode"]["dev"],
            "guided": summarize_response_mode_rows(selected_dev_rows, dev_turns),
        },
        "test": {
            "baseline": variants["heuristic_response_mode"]["test"],
            "guided": summarize_response_mode_rows(selected_rows, test_turns),
        },
        "efficacy": compare_conversation_efficacy(
            train_baseline=variants["heuristic_response_mode"]["train"],
            train_guided=summarize_response_mode_rows(selected_train_rows, train_turns),
            dev_baseline=variants["heuristic_response_mode"]["dev"],
            dev_guided=summarize_response_mode_rows(selected_dev_rows, dev_turns),
            test_baseline=variants["heuristic_response_mode"]["test"],
            test_guided=summarize_response_mode_rows(selected_rows, test_turns),
        ),
        "analytics": {
            "train_segments": summarize_response_mode_segments(
                train_baseline_rows,
                selected_train_rows,
            ),
            "dev_segments": summarize_response_mode_segments(
                dev_baseline_rows,
                selected_dev_rows,
            ),
            "test_segments": summarize_response_mode_segments(
                test_baseline_rows,
                selected_rows,
            ),
        },
        "coverage_projection": response_mode_coverage_projection(
            variant_rows=variant_rows,
            test_turns=test_turns,
        ),
        "guidance_delta": compare_response_mode_guidance_delta(
            test_baseline_rows,
            selected_rows,
        ),
    }


def response_mode_promotion_summary(
    selected_name: str,
    baseline: dict[str, float | int],
    selected: dict[str, float | int],
    test_segment_regressions: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "dev_selected_variant": selected_name,
        "heldout_promotable": (
            not test_segment_regressions
            and float(selected["p_at_1"]) >= float(baseline["p_at_1"])
            and float(selected["top_3_recall"]) >= float(baseline["top_3_recall"])
        ),
        "reason": (
            "held-out test passed top-line and segment checks"
            if not test_segment_regressions
            else "held-out test exposed response-mode segment regressions"
        ),
        "test_segment_regression_count": len(test_segment_regressions),
    }


def response_mode_coverage_projection(
    variant_rows: dict[str, dict[str, tuple[dict[str, object], ...]]],
    test_turns: tuple[ConversationTurn, ...],
) -> dict[str, object]:
    baseline_rows = variant_rows["heuristic_response_mode"]["test"]
    modes = sorted({turn.expected_response_mode for turn in test_turns})
    projection: dict[str, object] = {}
    for mode in modes:
        baseline = summarize_response_mode_row_subset(
            tuple(
                row
                for row in baseline_rows
                if str(row["expected_response_mode"]) == mode
            )
        )
        variant_metrics = {
            name: summarize_response_mode_row_subset(
                tuple(
                    row
                    for row in rows["test"]
                    if str(row["expected_response_mode"]) == mode
                )
            )
            for name, rows in variant_rows.items()
        }
        best_top_1 = max(
            variant_metrics,
            key=lambda name: (
                float(variant_metrics[name]["p_at_1"]),
                float(variant_metrics[name]["top_3_recall"]),
            ),
        )
        best_top_3 = max(
            variant_metrics,
            key=lambda name: (
                float(variant_metrics[name]["top_3_recall"]),
                float(variant_metrics[name]["p_at_1"]),
            ),
        )
        projection[mode] = {
            "baseline": baseline,
            "best_top_1_variant": best_top_1,
            "best_top_1": variant_metrics[best_top_1],
            "best_top_1_gain": round(
                float(variant_metrics[best_top_1]["p_at_1"])
                - float(baseline["p_at_1"]),
                3,
            ),
            "best_top_3_variant": best_top_3,
            "best_top_3": variant_metrics[best_top_3],
            "best_top_3_gain": round(
                float(variant_metrics[best_top_3]["top_3_recall"])
                - float(baseline["top_3_recall"]),
                3,
            ),
        }
    return {"expected_response_mode": projection}


def response_mode_bakeoff_variants() -> tuple[dict[str, object], ...]:
    return (
        {"name": "heuristic_response_mode", "learned_weight": 0.0},
        {"name": "response_mode_hybrid_25", "learned_weight": 0.25},
        {"name": "response_mode_hybrid_50", "learned_weight": 0.5},
        {"name": "response_mode_hybrid_75", "learned_weight": 0.75},
        {
            "name": "balanced_response_mode_50",
            "learned_weight": 0.5,
            "coverage_modes": ("disclose", "inform", "other", "reassure"),
            "coverage_min_score": 0.16,
        },
        {
            "name": "balanced_response_mode_75",
            "learned_weight": 0.75,
            "coverage_modes": ("disclose", "inform", "other", "reassure"),
            "coverage_min_score": 0.16,
        },
        {
            "name": "balanced_prior_response_mode_50",
            "learned_weight": 0.5,
            "class_balanced_prior": True,
        },
        {
            "name": "balanced_prior_response_mode_75",
            "learned_weight": 0.75,
            "class_balanced_prior": True,
        },
        {
            "name": "balanced_prior_coverage_response_mode_50",
            "learned_weight": 0.5,
            "class_balanced_prior": True,
            "coverage_modes": ("disclose", "inform", "other", "reassure"),
            "coverage_min_score": 0.16,
        },
        {"name": "learned_response_mode", "learned_weight": 1.0},
    )


def score_response_mode_turns(
    turns: tuple[ConversationTurn, ...],
    top_k: int,
    ranker: ResponseModeRanker | None = None,
    learned_weight: float = 0.0,
    coverage_modes: tuple[str, ...] = tuple(),
    coverage_min_score: float = 0.0,
    scoring_variant: str = "heuristic_response_mode",
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for turn in turns:
        branches = generate_response_mode_branches(
            turn,
            top_k=top_k,
            ranker=ranker,
            learned_weight=learned_weight,
            coverage_modes=coverage_modes,
            coverage_min_score=coverage_min_score,
            scoring_variant=scoring_variant,
        )
        rank_1 = branches[0]
        rows.append(
            {
                "turn_id": turn.turn_id,
                "expected_act": turn.expected_act,
                "expected_emotion": turn.expected_emotion,
                "expected_response_mode": turn.expected_response_mode,
                "next_speaker": turn.next_speaker,
                "rank_1_response_mode": rank_1["response_mode"],
                "top_response_modes": [
                    branch["response_mode"] for branch in branches
                ],
                "tts_ready": bool(branches),
                "latency_ms": 90 if branches else turn.latency_budget_ms,
            }
        )
    return tuple(rows)


def select_response_mode_bakeoff_variant(
    variants: dict[str, dict[str, object]],
) -> str:
    heuristic = variants["heuristic_response_mode"]
    candidates = {
        name: row
        for name, row in variants.items()
        if (
            not row["dev_segment_regressions"]
            and float(row["dev"]["p_at_1"]) >= float(heuristic["dev"]["p_at_1"])
            and float(row["dev"]["top_3_recall"]) >= float(heuristic["dev"]["top_3_recall"])
        )
    }
    if not candidates:
        return "heuristic_response_mode"
    return sorted(
        candidates,
        key=lambda name: (
            float(candidates[name]["dev"]["p_at_1"]),
            float(candidates[name]["dev"]["top_3_recall"]),
            -float(candidates[name]["learned_weight"]),
        ),
        reverse=True,
    )[0]


def conversation_bakeoff_variants() -> tuple[dict[str, object], ...]:
    return (
        {"name": "heuristic", "learned_weight": 0.0, "transition_weight": 0.0},
        {"name": "hybrid_25", "learned_weight": 0.25, "transition_weight": 0.0},
        {"name": "hybrid_50", "learned_weight": 0.5, "transition_weight": 0.0},
        {"name": "hybrid_75", "learned_weight": 0.75, "transition_weight": 0.0},
        {"name": "learned", "learned_weight": 1.0, "transition_weight": 0.0},
        {
            "name": "contextual_transition",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
        },
        {
            "name": "guarded_contextual_transition",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
        },
        {
            "name": "act_rhythm_contextual",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_margin": 0.25,
        },
        {
            "name": "act_rhythm_contextual_strict",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_margin": 0.6,
        },
        {
            "name": "protected_act_rhythm_contextual",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_margin": 0.25,
            "history_overlay_acts": ("directive", "question"),
        },
        {
            "name": "question_act_rhythm_contextual",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_margin": 0.25,
            "history_overlay_acts": ("question",),
        },
        {
            "name": "safe_question_act_rhythm_contextual",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_margin": 0.25,
            "history_overlay_acts": ("question",),
            "history_preserved_acts": ("directive",),
        },
        {
            "name": "question_evidence_act_rhythm_contextual",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_margin": 0.25,
            "history_overlay_acts": ("question",),
            "use_question_evidence_ranker": True,
            "question_evidence_margin": 4.0,
        },
        {
            "name": "safe_question_evidence_act_rhythm_contextual",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_margin": 0.25,
            "history_overlay_acts": ("question",),
            "history_preserved_acts": ("directive",),
            "use_question_evidence_ranker": True,
            "question_evidence_margin": 4.0,
            "question_evidence_preserved_acts": ("directive",),
        },
        {
            "name": "deep_protected_act_rhythm_contextual",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_window_size": 8,
            "history_margin": 0.25,
            "history_overlay_acts": ("directive", "question"),
            "history_preserved_acts": ("directive",),
        },
        {
            "name": "directive_act_rhythm_contextual",
            "learned_weight": 0.0,
            "transition_weight": 1.0,
            "transition_protected_acts": ("directive", "question"),
            "use_history_ranker": True,
            "history_margin": 0.25,
            "history_overlay_acts": ("directive",),
        },
        {
            "name": "contextual_hybrid_50",
            "learned_weight": 0.0,
            "transition_weight": 0.5,
        },
        {
            "name": "contextual_inform_overlay",
            "learned_weight": 0.0,
            "transition_weight": 0.0,
            "transition_overlay_act": "inform",
            "transition_overlay_margin": 0.25,
        },
    )


def cross_validate_conversation_variant(
    turns: tuple[ConversationTurn, ...],
    variant: dict[str, object],
    fold_count: int = 5,
    top_k: int = 3,
) -> dict[str, object]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(turns) < fold_count:
        raise ValueError("turn count must be at least fold_count")

    folds = tuple(
        score_conversation_variant_fold(
            turns=turns,
            variant=variant,
            fold_index=fold_index,
            fold_count=fold_count,
            top_k=top_k,
        )
        for fold_index in range(fold_count)
    )
    p_at_1_gains = [float(fold["p_at_1_gain"]) for fold in folds]
    top_3_gains = [float(fold["top_3_recall_gain"]) for fold in folds]
    return {
        "fold_count": fold_count,
        "folds": list(folds),
        "mean_p_at_1_gain": round(sum(p_at_1_gains) / len(p_at_1_gains), 3),
        "min_p_at_1_gain": round(min(p_at_1_gains), 3),
        "mean_top_3_recall_gain": round(sum(top_3_gains) / len(top_3_gains), 3),
        "min_top_3_recall_gain": round(min(top_3_gains), 3),
        "segment_regression_count": sum(
            len(fold["segment_regressions"]) for fold in folds
        ),
    }


def empty_conversation_cross_validation() -> dict[str, object]:
    return {
        "fold_count": 0,
        "folds": [],
        "mean_p_at_1_gain": 0.0,
        "min_p_at_1_gain": 0.0,
        "mean_top_3_recall_gain": 0.0,
        "min_top_3_recall_gain": 0.0,
        "segment_regression_count": 0,
    }


def score_conversation_variant_fold(
    turns: tuple[ConversationTurn, ...],
    variant: dict[str, object],
    fold_index: int,
    fold_count: int,
    top_k: int,
) -> dict[str, object]:
    train_turns = tuple(
        turn
        for index, turn in enumerate(turns)
        if index % fold_count != fold_index
    )
    validation_turns = tuple(
        turn
        for index, turn in enumerate(turns)
        if index % fold_count == fold_index
    )
    ranker = train_conversation_act_ranker(train_turns)
    transition_ranker = train_conversation_transition_ranker(train_turns)
    history_ranker = train_conversation_history_ranker(
        train_turns,
        window_size=int(variant.get("history_window_size", 4)),
    )
    question_evidence_ranker = train_conversation_question_evidence_ranker(train_turns)
    empty_guidance = ConversationGuidance(act_keywords={})

    baseline_rows = score_conversation_turns(
        validation_turns,
        top_k=top_k,
        guidance=empty_guidance,
    )
    candidate_rows = score_conversation_turns(
        validation_turns,
        top_k=top_k,
        guidance=empty_guidance,
        act_ranker=ranker,
        learned_weight=float(variant["learned_weight"]),
        transition_ranker=transition_ranker,
        transition_weight=float(variant["transition_weight"]),
        transition_overlay_act=(
            str(variant["transition_overlay_act"])
            if variant.get("transition_overlay_act")
            else None
        ),
        transition_overlay_margin=float(variant.get("transition_overlay_margin", 0.0)),
        transition_protected_acts=tuple(
            str(act)
            for act in variant.get("transition_protected_acts", ())
        ),
        history_ranker=(
            history_ranker
            if bool(variant.get("use_history_ranker", False))
            else None
        ),
        history_margin=float(variant.get("history_margin", 0.0)),
        history_overlay_acts=tuple(
            str(act)
            for act in variant.get("history_overlay_acts", ())
        ),
        history_preserved_acts=tuple(
            str(act)
            for act in variant.get("history_preserved_acts", ())
        ),
        question_evidence_ranker=(
            question_evidence_ranker
            if bool(variant.get("use_question_evidence_ranker", False))
            else None
        ),
        question_evidence_margin=float(variant.get("question_evidence_margin", 0.0)),
        question_evidence_preserved_acts=tuple(
            str(act)
            for act in variant.get("question_evidence_preserved_acts", ())
        ),
        scoring_variant=str(variant["name"]),
    )
    baseline = summarize_conversation_rows(baseline_rows, validation_turns)
    candidate = summarize_conversation_rows(candidate_rows, validation_turns)
    return {
        "fold": fold_index + 1,
        "validation_turns": len(validation_turns),
        "baseline": baseline,
        "candidate": candidate,
        "p_at_1_gain": round(
            float(candidate["p_at_1"]) - float(baseline["p_at_1"]),
            3,
        ),
        "top_3_recall_gain": round(
            float(candidate["top_3_recall"]) - float(baseline["top_3_recall"]),
            3,
        ),
        "segment_regressions": find_conversation_act_segment_regressions(
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows,
        ),
    }


def select_conversation_bakeoff_variant(
    variants: dict[str, dict[str, object]],
) -> str:
    safe_variants = {
        name: row
        for name, row in variants.items()
        if not row["dev_segment_regressions"]
    }
    candidates = safe_variants if safe_variants else variants
    robust_candidates = {
        name: row
        for name, row in candidates.items()
        if conversation_variant_robust_enough(row)
    }
    candidates = robust_candidates if robust_candidates else candidates
    cross_validated_candidates = {
        name: row
        for name, row in candidates.items()
        if conversation_cross_validation_ok(row)
    }
    candidates = cross_validated_candidates if cross_validated_candidates else candidates
    candidates = conversation_stability_tie_candidates(candidates)
    return sorted(
        candidates,
        key=lambda name: (
            conversation_preservation_score(candidates[name]),
            float(candidates[name]["dev"]["p_at_1"]),
            float(candidates[name]["dev"]["top_3_recall"]),
            -len(candidates[name]["dev_segment_regressions"]),
            -float(candidates[name]["learned_weight"]),
        ),
        reverse=True,
    )[0]


def conversation_stability_tie_candidates(
    candidates: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    best_dev = max(float(row["dev"]["p_at_1"]) for row in candidates.values())
    best_overlay = tuple(
        next(
            row
            for row in candidates.values()
            if float(row["dev"]["p_at_1"]) == best_dev
        ).get("history_overlay_acts", ())
    )
    near_best = {
        name: row
        for name, row in candidates.items()
        if best_dev - float(row["dev"]["p_at_1"]) <= CONVERSATION_BAKEOFF_DEV_TIE_MARGIN
        and tuple(row.get("history_overlay_acts", ())) == best_overlay
    }
    return near_best


def conversation_preservation_score(row: dict[str, object]) -> int:
    return len(tuple(row.get("history_preserved_acts", ())))


def conversation_variant_robust_enough(row: dict[str, object]) -> bool:
    train_dev_gap = conversation_train_dev_gap(row)
    if train_dev_gap <= MAX_CONVERSATION_BAKEOFF_TRAIN_DEV_GAP:
        return True
    return (
        bool(row.get("history_overlay_acts"))
        and train_dev_gap <= MAX_CONVERSATION_SPECIALIST_TRAIN_DEV_GAP
        and conversation_cross_validation_ok(row)
    )


def conversation_cross_validation_ok(row: dict[str, object]) -> bool:
    cross_validation = row.get("cross_validation")
    if not cross_validation:
        return True
    return (
        float(cross_validation["min_p_at_1_gain"]) >= 0
        and int(cross_validation["segment_regression_count"]) == 0
    )


def conversation_train_dev_gap(row: dict[str, object]) -> float:
    train = row["train"]
    dev = row["dev"]
    return round(
        max(0.0, float(train["p_at_1"]) - float(dev["p_at_1"])),
        3,
    )


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


def summarize_response_mode_rows(
    rows: tuple[dict[str, object], ...],
    turns: tuple[ConversationTurn, ...],
) -> dict[str, float | int]:
    total = max(len(rows), 1)
    exact = sum(
        row["rank_1_response_mode"] == row["expected_response_mode"]
        for row in rows
    )
    top_3 = sum(
        row["expected_response_mode"] in row["top_response_modes"]
        for row in rows
    )
    tts_ready = sum(bool(row["tts_ready"]) for row in rows)
    return {
        "total_turns": len(turns),
        "p_at_1": round(exact / total, 3),
        "top_3_recall": round(top_3 / total, 3),
        "tts_readiness_rate": round(tts_ready / total, 3),
        "median_latency_ms": int(median(row["latency_ms"] for row in rows)) if rows else 0,
    }


def find_response_mode_segment_regressions(
    baseline_rows: tuple[dict[str, object], ...],
    candidate_rows: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    regressions: list[dict[str, object]] = []
    modes = sorted({str(row["expected_response_mode"]) for row in baseline_rows})
    for mode in modes:
        baseline = summarize_response_mode_row_subset(
            tuple(
                row
                for row in baseline_rows
                if str(row["expected_response_mode"]) == mode
            )
        )
        candidate = summarize_response_mode_row_subset(
            tuple(
                row
                for row in candidate_rows
                if str(row["expected_response_mode"]) == mode
            )
        )
        delta = round(float(candidate["p_at_1"]) - float(baseline["p_at_1"]), 3)
        if delta < 0:
            regressions.append(
                {
                    "segment": "expected_response_mode",
                    "name": mode,
                    "baseline_p_at_1": baseline["p_at_1"],
                    "candidate_p_at_1": candidate["p_at_1"],
                    "p_at_1_delta": delta,
                }
            )
    return regressions


def summarize_response_mode_segments(
    baseline_rows: tuple[dict[str, object], ...],
    guided_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    dimensions = {
        "expected_response_mode": sorted(
            {str(row["expected_response_mode"]) for row in baseline_rows}
        ),
        "expected_act": sorted({str(row["expected_act"]) for row in baseline_rows}),
        "expected_emotion": sorted(
            {str(row["expected_emotion"]) for row in baseline_rows}
        ),
        "next_speaker": sorted({str(row["next_speaker"]) for row in baseline_rows}),
    }
    summary: dict[str, object] = {
        dimension: {
            value: {
                "baseline": summarize_response_mode_row_subset(
                    tuple(row for row in baseline_rows if str(row[dimension]) == value)
                ),
                "guided": summarize_response_mode_row_subset(
                    tuple(row for row in guided_rows if str(row[dimension]) == value)
                ),
            }
            for value in values
        }
        for dimension, values in dimensions.items()
    }
    summary["focus_areas"] = response_mode_focus_areas(summary)
    return summary


def summarize_response_mode_row_subset(
    rows: tuple[dict[str, object], ...],
) -> dict[str, float | int]:
    total = max(len(rows), 1)
    exact = sum(
        row["rank_1_response_mode"] == row["expected_response_mode"]
        for row in rows
    )
    top_3 = sum(
        row["expected_response_mode"] in row["top_response_modes"]
        for row in rows
    )
    tts_ready = sum(bool(row["tts_ready"]) for row in rows)
    return {
        "total_turns": len(rows),
        "p_at_1": round(exact / total, 3),
        "top_3_recall": round(top_3 / total, 3),
        "tts_readiness_rate": round(tts_ready / total, 3),
    }


def response_mode_focus_areas(summary: dict[str, object]) -> list[dict[str, object]]:
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


def compare_response_mode_guidance_delta(
    baseline_rows: tuple[dict[str, object], ...],
    guided_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    baseline = {
        str(row["turn_id"]): (
            row["rank_1_response_mode"] == row["expected_response_mode"]
        )
        for row in baseline_rows
    }
    guided = {
        str(row["turn_id"]): (
            row["rank_1_response_mode"] == row["expected_response_mode"]
        )
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
