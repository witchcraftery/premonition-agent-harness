from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from foresight_harness.conversation_probability import (
    ConversationProbabilityPack,
    ConversationTurn,
    RESPONSE_MODE_TEMPLATES,
    build_response_mode_probability_pack,
    generate_response_mode_branches,
    score_response_mode_probability_pack_turn,
)
from foresight_harness.models import Message


LIVE_SHADOW_POLICY = {
    "first_speech_variant": "live_shadow_heuristic_response_mode",
    "first_speech_delivery": "confirm_before_delivery",
    "background_readiness_variant": "live_shadow_background_readiness",
    "background_preparation": "prewarm_tts",
    "confirmation_mode": "wait_for_reality_grade",
}


def infer_response_mode_from_text(text: str) -> str:
    lowered = text.lower()
    tokens = set(lowered.replace("?", " ? ").replace(".", " ").split())
    if {"sorry", "apologize", "apologise"} & tokens:
        return "apologize"
    if "?" in lowered or {"tell", "more", "clarify"} & tokens:
        return "ask_followup"
    if {"will", "promise", "handle"} & tokens or "take care" in lowered:
        return "commit"
    if {"reassure", "okay", "safe"} & tokens or "not alone" in lowered:
        return "reassure"
    if {"should", "try", "step", "option", "options"} & tokens:
        return "suggest"
    if {"heavy", "valid", "understand", "sounds"} & tokens:
        return "validate"
    if {"because", "means", "information", "details"} & tokens:
        return "inform"
    return "inform"


@dataclass
class LiveShadowSession:
    session_id: str = "live-shadow"
    top_k: int = 3
    messages: list[dict[str, object]] = field(default_factory=list)
    packs: list[ConversationProbabilityPack] = field(default_factory=list)
    grades: list[dict[str, object]] = field(default_factory=list)
    timeline: list[dict[str, object]] = field(default_factory=list)

    def observe(self, role: str, content: str) -> dict[str, object]:
        role = role.strip().lower()
        content = content.strip()
        if role not in {"user", "assistant", "system", "environment"}:
            raise ValueError("role must be user, assistant, system, or environment")
        if not content:
            raise ValueError("content is required")

        message = {
            "index": len(self.messages) + 1,
            "role": role,
            "content": content,
        }
        self.messages.append(message)
        self._append_timeline("observed", f"{role} turn observed")

        pack = self._build_pack()
        self.packs.append(pack)
        self._append_timeline("drafted", f"{len(pack.prepared_drafts)} drafts prepared")
        return self.to_dict()

    def grade_reality(
        self,
        content: str,
        actual_response_mode: str | None = None,
    ) -> dict[str, object]:
        content = content.strip()
        if not content:
            raise ValueError("content is required")
        if not self.packs:
            raise ValueError("observe at least one turn before grading reality")

        actual_mode = actual_response_mode or infer_response_mode_from_text(content)
        if actual_mode not in RESPONSE_MODE_TEMPLATES:
            raise ValueError("actual_response_mode must be a known response mode")

        pack = self.packs[-1]
        turn = self._turn_for_actual(content, actual_mode)
        row = score_response_mode_probability_pack_turn(
            turn,
            pack,
            prepared_latency_ms=90,
            min_quality_score=0.0,
        )
        grade = {
            "grade_id": f"{self.session_id}-grade-{len(self.grades) + 1}",
            "pack_id": pack.pack_id,
            "actual_content": content,
            "actual_response_mode": actual_mode,
            "match_grade": row["match_grade"],
            "prepared_response_mode": row["response_mode"],
            "preparation_role": row["preparation_role"],
            "quality_score": row["quality_score"],
            "quality_ready": bool(float(row["quality_score"]) >= 0.75),
            "latency_ms": row["latency_ms"],
            "latency_saved_ms": row["latency_saved_ms"],
        }
        self.grades.append(grade)
        self._append_timeline("reality", "actual next move captured")
        self._append_timeline("graded", f"{grade['match_grade']} against reality")
        return self.to_dict()

    def reset(self) -> dict[str, object]:
        self.messages.clear()
        self.packs.clear()
        self.grades.clear()
        self.timeline.clear()
        return self.to_dict()

    def export_jsonl(self) -> str:
        rows = []
        for grade in self.grades:
            pack = next(
                pack for pack in self.packs if pack.pack_id == str(grade["pack_id"])
            )
            rows.append(
                json.dumps(
                    {
                        "session_id": self.session_id,
                        "pack_id": pack.pack_id,
                        "observed_context": pack.observed_context,
                        "prepared_drafts": list(pack.prepared_drafts),
                        "top_branches": list(pack.top_branches),
                        **grade,
                    },
                    sort_keys=True,
                )
            )
        return "\n".join(rows)

    def to_dict(self) -> dict[str, object]:
        active_pack = self.packs[-1].to_dict() if self.packs else None
        return {
            "session_id": self.session_id,
            "messages": list(self.messages),
            "active_pack": active_pack,
            "grades": list(self.grades),
            "metrics": self._metrics(),
            "timeline": list(self.timeline),
            "export_rows": len(self.grades),
        }

    def _build_pack(self) -> ConversationProbabilityPack:
        turn = self._turn_for_prediction()
        branches = generate_response_mode_branches(turn, top_k=self.top_k)
        background = branches[1:] if len(branches) > 1 else branches
        return build_response_mode_probability_pack(
            turn=turn,
            first_speech_branches=branches[:1],
            background_readiness_branches=background,
            policy=LIVE_SHADOW_POLICY,
            top_k=self.top_k,
        )

    def _turn_for_prediction(self) -> ConversationTurn:
        return ConversationTurn(
            turn_id=f"{self.session_id}-turn-{len(self.messages)}",
            conversation=tuple(
                Message(role=str(message["role"]), content=str(message["content"]))
                for message in self.messages
            ),
            next_speaker="assistant",
            actual_next_utterance="",
            expected_act="inform",
            expected_emotion="no_emotion",
            expected_response_mode="inform",
            latency_budget_ms=650,
        )

    def _turn_for_actual(self, content: str, actual_mode: str) -> ConversationTurn:
        return ConversationTurn(
            turn_id=f"{self.session_id}-actual-{len(self.grades) + 1}",
            conversation=tuple(
                Message(role=str(message["role"]), content=str(message["content"]))
                for message in self.messages
            ),
            next_speaker="assistant",
            actual_next_utterance=content,
            expected_act="inform",
            expected_emotion="no_emotion",
            expected_response_mode=actual_mode,
            latency_budget_ms=650,
        )

    def _append_timeline(self, event_type: str, label: str) -> None:
        self.timeline.append(
            {
                "index": len(self.timeline) + 1,
                "event_type": event_type,
                "label": label,
            }
        )

    def _metrics(self) -> dict[str, object]:
        total = len(self.grades)
        if total == 0:
            return {
                "graded_turns": 0,
                "prepared_hit_rate": 0.0,
                "exact_hit_rate": 0.0,
                "quality_ready_rate": 0.0,
                "average_quality_score": 0.0,
                "median_latency_saved_ms": 0,
            }
        prepared_hits = sum(grade["match_grade"] != "miss" for grade in self.grades)
        exact_hits = sum(grade["match_grade"] == "exact" for grade in self.grades)
        quality_ready = sum(bool(grade["quality_ready"]) for grade in self.grades)
        quality_scores = [float(grade["quality_score"]) for grade in self.grades]
        latency_saved = [int(grade["latency_saved_ms"]) for grade in self.grades]
        return {
            "graded_turns": total,
            "prepared_hit_rate": round(prepared_hits / total, 3),
            "exact_hit_rate": round(exact_hits / total, 3),
            "quality_ready_rate": round(quality_ready / total, 3),
            "average_quality_score": round(sum(quality_scores) / total, 3),
            "median_latency_saved_ms": int(median(latency_saved)),
        }


def handle_live_shadow_api_request(
    session: LiveShadowSession,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> tuple[int, dict[str, object]]:
    try:
        if method == "GET" and path == "/api/session":
            return 200, session.to_dict()
        if method == "GET" and path == "/api/export":
            return 200, {"jsonl": session.export_jsonl()}
        if method == "POST" and path == "/api/reset":
            return 200, session.reset()
        if method == "POST" and path == "/api/observe":
            data = payload or {}
            return 200, session.observe(str(data.get("role", "user")), str(data.get("content", "")))
        if method == "POST" and path == "/api/grade":
            data = payload or {}
            return 200, session.grade_reality(
                str(data.get("content", "")),
                actual_response_mode=(
                    str(data["actual_response_mode"])
                    if data.get("actual_response_mode")
                    else None
                ),
            )
        return 404, {"error": "unknown live shadow route"}
    except ValueError as error:
        return 400, {"error": str(error)}
