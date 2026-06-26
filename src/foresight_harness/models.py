from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MatchGrade(str, Enum):
    EXACT_INTENT = "exact_intent"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    USEFUL_PARTIAL = "useful_partial"
    MISS = "miss"
    UNSAFE = "unsafe"
    UNSCORED = "unscored"


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Message":
        return cls(role=str(row["role"]), content=str(row["content"]))


@dataclass(frozen=True)
class ReplayTurn:
    turn_id: str
    conversation: tuple[Message, ...]
    actual_next_event: str
    policy_context: str
    expected_intent: str
    latency_budget_ms: int = 800

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ReplayTurn":
        return cls(
            turn_id=str(row["turn_id"]),
            conversation=tuple(Message.from_dict(item) for item in row["conversation"]),
            actual_next_event=str(row["actual_next_event"]),
            policy_context=str(row["policy_context"]),
            expected_intent=str(row["expected_intent"]),
            latency_budget_ms=int(row.get("latency_budget_ms", 800)),
        )

    def context_text(self) -> str:
        return "\n".join(f"{message.role}: {message.content}" for message in self.conversation)


@dataclass
class Branch:
    branch_id: str
    predicted_event: str
    intent: str
    probability: float
    rank: int
    match_grade: MatchGrade = MatchGrade.UNSCORED
    match_score: float = 0.0


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    branch_id: str
    response_draft: str
    policy_checks: tuple[str, ...]
    readiness_score: float
    token_cost: int
    created_for_intent: str


@dataclass(frozen=True)
class PremonitionPacket:
    packet_id: str
    turn_id: str
    observed_context: str
    matched_branch_id: str | None
    matched_intent: str | None
    confidence: float
    prepared_artifact: str | None
    policy_checks: tuple[str, ...]
    freshness: str
    unsafe: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "turn_id": self.turn_id,
            "observed_context": self.observed_context,
            "matched_branch_id": self.matched_branch_id,
            "matched_intent": self.matched_intent,
            "confidence": self.confidence,
            "prepared_artifact": self.prepared_artifact,
            "policy_checks": list(self.policy_checks),
            "freshness": self.freshness,
            "unsafe": self.unsafe,
        }


@dataclass(frozen=True)
class RunResult:
    turn_id: str
    variant: str
    branches: tuple[Branch, ...] = field(default_factory=tuple)
    selected_artifact: Artifact | None = None
    latency_ms: int = 0
    token_cost: int = 0
    useful: bool = False
    unsafe_leak: bool = False
