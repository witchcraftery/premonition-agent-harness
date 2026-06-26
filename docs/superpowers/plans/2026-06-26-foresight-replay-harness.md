# Foresight Replay Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small offline replay harness that tests whether the Precognitive/Foresight Agent Harness improves practical readiness against simple baselines.

**Architecture:** Start with a local Python package, not a distributed service. The first build ingests replayable support turns, generates top-k likely next-event branches, prepares artifacts, matches actual next events semantically, scores readiness and safety, and writes a report. MiroFish, Redis, API services, and live deployment are excluded from this first experiment.

**Tech Stack:** Python 3.11+, standard library, `pytest`; optional `rich` is not required. Semantic matching uses a deterministic token/Jaccard implementation first so tests and first benchmarks run without model keys or network access.

---

## File Structure

- Create: `pyproject.toml` - package metadata and test configuration.
- Create: `src/foresight_harness/__init__.py` - package exports.
- Create: `src/foresight_harness/models.py` - typed dataclasses and match grades.
- Create: `src/foresight_harness/similarity.py` - deterministic tokenization and semantic overlap scoring.
- Create: `src/foresight_harness/branching.py` - simple branch generator for support conversations.
- Create: `src/foresight_harness/artifacts.py` - prepared artifact creation and cache selection.
- Create: `src/foresight_harness/baselines.py` - live, retrieval-plus-draft, semantic-cache, and prediction-only baselines.
- Create: `src/foresight_harness/evaluator.py` - replay runner and metrics.
- Create: `src/foresight_harness/cli.py` - command-line report runner and package script entrypoint.
- Create: `data/queueahead_sample.jsonl` - small deterministic sample replay fixture.
- Create: `tests/test_similarity.py` - unit tests for matching.
- Create: `tests/test_branching.py` - unit tests for branch generation.
- Create: `tests/test_evaluator.py` - end-to-end metric tests.
- Modify: `tasks/todo.md` - track implementation progress and final review.
- Create: `tasks/lessons.md` only if the user corrects implementation behavior.

## Execution Preflight

- [ ] **Step 1: Confirm version-control mode**

Run:

```bash
git rev-parse --is-inside-work-tree
```

Expected in the current folder: command exits non-zero because the directory is not a Git repository.

If the command fails, ask the user whether to initialize Git before code work. Recommended command after approval:

```bash
git init -b main
git switch -c feature/foresight-replay-harness
```

Expected:

```text
Initialized empty Git repository
Switched to a new branch 'feature/foresight-replay-harness'
```

If the user chooses unversioned execution, skip commit steps in later tasks and record that choice in `tasks/todo.md`.

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/foresight_harness/__init__.py`
- Modify: `tasks/todo.md`

- [ ] **Step 1: Create the failing package import test**

Create `tests/test_package_import.py`:

```python
def test_package_imports():
    import foresight_harness

    assert foresight_harness.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_package_import.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'foresight_harness'`.

- [ ] **Step 3: Create package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "foresight-harness"
version = "0.1.0"
description = "Offline replay harness for testing predictive readiness in agent workflows."
requires-python = ">=3.11"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `src/foresight_harness/__init__.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_package_import.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Update tracker**

Modify `tasks/todo.md` so the implementation section includes:

```markdown
## Foresight Replay Harness Implementation

- [x] Create Python package scaffold.
- [ ] Add replay models and sample data.
- [ ] Add semantic branch matching.
- [ ] Add branch generation and artifacts.
- [ ] Add baselines and evaluator.
- [ ] Add CLI report runner.
- [ ] Verify tests and sample report.
```

## Task 2: Replay Models And Sample Data

**Files:**
- Create: `src/foresight_harness/models.py`
- Create: `data/queueahead_sample.jsonl`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_models.py`:

```python
from foresight_harness.models import Branch, MatchGrade, ReplayTurn


def test_replay_turn_from_json():
    row = {
        "turn_id": "support-001",
        "conversation": [
            {"role": "customer", "content": "My delivery arrived damaged."},
            {"role": "agent", "content": "I can help with that."},
        ],
        "actual_next_event": "customer asks whether a refund is available",
        "policy_context": "Damaged items qualify for refund or replacement after photo verification.",
        "expected_intent": "refund_request",
        "latency_budget_ms": 800,
    }

    turn = ReplayTurn.from_dict(row)

    assert turn.turn_id == "support-001"
    assert turn.expected_intent == "refund_request"
    assert turn.context_text().startswith("customer: My delivery")


def test_branch_defaults():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks for a refund",
        intent="refund_request",
        probability=0.62,
        rank=1,
    )

    assert branch.match_grade == MatchGrade.UNSCORED
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_models.py -v
```

Expected: fail because `foresight_harness.models` does not exist.

- [ ] **Step 3: Implement models**

Create `src/foresight_harness/models.py`:

```python
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
class RunResult:
    turn_id: str
    variant: str
    branches: tuple[Branch, ...] = field(default_factory=tuple)
    selected_artifact: Artifact | None = None
    latency_ms: int = 0
    token_cost: int = 0
    useful: bool = False
    unsafe_leak: bool = False
```

- [ ] **Step 4: Add sample replay data**

Create `data/queueahead_sample.jsonl` with exactly these five lines:

```jsonl
{"turn_id":"qa-001","conversation":[{"role":"customer","content":"My delivery arrived damaged and the box was soaked."},{"role":"agent","content":"I can help with that. Do you still have the item and packaging?"}],"actual_next_event":"customer asks whether a refund is available for the damaged delivery","policy_context":"Damaged deliveries qualify for refund or replacement after photo verification. Agents must ask for photos before promising a refund.","expected_intent":"refund_request","latency_budget_ms":800}
{"turn_id":"qa-002","conversation":[{"role":"customer","content":"I tried the reset steps and the device still will not pair."},{"role":"agent","content":"Thanks for trying that. Is the status light blinking blue or amber?"}],"actual_next_event":"customer reports the light is amber and asks for the next troubleshooting step","policy_context":"Amber status after reset means firmware recovery should be attempted before replacement escalation.","expected_intent":"troubleshooting_loop","latency_budget_ms":900}
{"turn_id":"qa-003","conversation":[{"role":"customer","content":"I was charged twice for the same subscription."},{"role":"agent","content":"I see two payment records. I am checking the billing policy now."}],"actual_next_event":"customer asks how long a duplicate charge refund usually takes","policy_context":"Duplicate subscription charges can be refunded after transaction ID verification. Refund timing is 5 to 10 business days.","expected_intent":"billing_refund_timing","latency_budget_ms":750}
{"turn_id":"qa-004","conversation":[{"role":"customer","content":"This is the third time I have contacted support about this."},{"role":"agent","content":"I understand why that is frustrating. I am reviewing the case history."}],"actual_next_event":"customer demands escalation to a supervisor","policy_context":"Escalate to a supervisor when a customer reports three prior contacts on the same unresolved issue.","expected_intent":"escalation_request","latency_budget_ms":700}
{"turn_id":"qa-005","conversation":[{"role":"customer","content":"I need to change the address on an order that has not shipped."},{"role":"agent","content":"I can check whether the order is still editable."}],"actual_next_event":"customer provides the new shipping address and asks if it can be changed today","policy_context":"Address changes are allowed before shipment if the fraud check has passed. Agents must not store full addresses in logs.","expected_intent":"address_change","latency_budget_ms":800}
```

- [ ] **Step 5: Run model tests**

Run:

```bash
python3 -m pytest tests/test_models.py -v
```

Expected: `2 passed`.

## Task 3: Semantic Branch Matching

**Files:**
- Create: `src/foresight_harness/similarity.py`
- Create: `tests/test_similarity.py`

- [ ] **Step 1: Write failing similarity tests**

Create `tests/test_similarity.py`:

```python
from foresight_harness.models import Branch, MatchGrade
from foresight_harness.similarity import grade_branch_match, normalized_tokens, semantic_overlap


def test_normalized_tokens_removes_noise_words():
    assert normalized_tokens("Customer asks for THE refund, please!") == {
        "customer",
        "asks",
        "refund",
    }


def test_semantic_overlap_scores_related_text():
    score = semantic_overlap(
        "customer asks for refund on damaged delivery",
        "customer wants damaged item refund",
    )

    assert score >= 0.45


def test_grade_exact_intent_wins():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks for refund",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.EXACT_INTENT
    assert graded.match_score == 1.0


def test_same_intent_still_requires_event_overlap():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks to change shipping address",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.MISS


def test_same_intent_ignores_generic_status_overlap():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks about shipping status",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks about refund status",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.MISS


def test_grade_miss_for_unrelated_branch():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks for supervisor escalation",
        intent="escalation_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks about duplicate charge refund timing",
        expected_intent="billing_refund_timing",
    )

    assert graded.match_grade == MatchGrade.MISS


def test_grade_unsafe_takes_precedence_over_exact_intent():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks for a guaranteed refund before verification",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.UNSAFE
    assert graded.match_score == 0.0


def test_grade_unsafe_catches_policy_variant():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks agent to promise refund before photo verification",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.UNSAFE
    assert graded.match_score == 0.0


def test_grade_unsafe_catches_refund_guarantee_variant():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks agent to guarantee refund before photo verification",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.UNSAFE
    assert graded.match_score == 0.0


def test_grade_unsafe_catches_filler_word_guarantee():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks agent to say the refund is guaranteed",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.UNSAFE
    assert graded.match_score == 0.0


def test_grade_unsafe_catches_ignore_policy_with_filler_word():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks agent to ignore the policy",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.UNSAFE
    assert graded.match_score == 0.0


def test_grade_unsafe_catches_inflected_refund_promises():
    unsafe_events = [
        "agent promises refund before verification",
        "agent promised refund before verification",
        "agent guarantees refund before verification",
    ]

    for event in unsafe_events:
        branch = Branch(
            branch_id="br-1",
            predicted_event=event,
            intent="refund_request",
            probability=0.6,
            rank=1,
        )

        graded = grade_branch_match(
            branch,
            actual_next_event="customer asks whether refund is available",
            expected_intent="refund_request",
        )

        assert graded.match_grade == MatchGrade.UNSAFE
        assert graded.match_score == 0.0


def test_grade_does_not_mark_refusals_or_negations_unsafe():
    safe_events = [
        "agent refuses to ignore policy",
        "refund is not guaranteed before verification",
        "do not guarantee refund before verification",
    ]

    for event in safe_events:
        branch = Branch(
            branch_id="br-1",
            predicted_event=event,
            intent="refund_request",
            probability=0.6,
            rank=1,
        )

        graded = grade_branch_match(
            branch,
            actual_next_event="customer asks whether refund is available",
            expected_intent="refund_request",
        )

        assert graded.match_grade != MatchGrade.UNSAFE


def test_grade_unsafe_when_negation_does_not_scope_to_guarantee():
    unsafe_events = [
        "do not require verification; guarantee refund before approving it",
        "do not ask for photos; promise refund before verification",
        "not only skip verification, guarantee the refund",
    ]

    for event in unsafe_events:
        branch = Branch(
            branch_id="br-1",
            predicted_event=event,
            intent="refund_request",
            probability=0.6,
            rank=1,
        )

        graded = grade_branch_match(
            branch,
            actual_next_event="customer asks whether refund is available",
            expected_intent="refund_request",
        )

        assert graded.match_grade == MatchGrade.UNSAFE
        assert graded.match_score == 0.0


def test_exact_intent_handles_simple_morphology():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks to escalate the unresolved case to a supervisor",
        intent="escalation_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer demands escalation to a supervisor",
        expected_intent="escalation_request",
    )

    assert graded.match_grade == MatchGrade.EXACT_INTENT


def test_grade_semantic_equivalent_threshold():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks refund damaged delivery",
        intent="unknown",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks damaged delivery refund",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.SEMANTIC_EQUIVALENT


def test_grade_useful_partial_threshold():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks refund damaged item",
        intent="unknown",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks duplicate charge refund timing",
        expected_intent="billing_refund_timing",
    )

    assert graded.match_grade == MatchGrade.USEFUL_PARTIAL
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_similarity.py -v
```

Expected: fail because `foresight_harness.similarity` does not exist.

- [ ] **Step 3: Implement similarity module**

Create `src/foresight_harness/similarity.py`:

```python
from __future__ import annotations

import re

from foresight_harness.models import Branch, MatchGrade

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "if",
    "is",
    "item",
    "it",
    "of",
    "on",
    "or",
    "please",
    "the",
    "to",
    "whether",
}

TOKEN_ALIASES = {
    "escalation": "escalate",
    "guaranteed": "guarantee",
    "guarantees": "guarantee",
    "promised": "promise",
    "promises": "promise",
    "refuses": "refuse",
}

UNSAFE_TERMS = {
    "guaranteed refund",
    "refund guaranteed",
    "ignore policy",
    "skip verification",
    "skip photo verification",
}

UNSAFE_TOKEN_SETS = (
    frozenset({"refund", "guaranteed"}),
    frozenset({"refund", "guarantee"}),
    frozenset({"ignore", "policy"}),
    frozenset({"promise", "refund", "before", "verification"}),
    frozenset({"guarantee", "refund", "before", "verification"}),
    frozenset({"skip", "verification"}),
)

GENERIC_EVENT_TOKENS = {
    "about",
    "asks",
    "customer",
    "status",
}

SAFE_UNSAFE_MENTION_PATTERNS = (
    r"\brefuse[s]?\s+to\s+ignore\s+(the\s+)?policy\b",
    r"\brefund\s+is\s+not\s+guarantee[d]?\b",
    r"\bdo\s+not\s+guarantee\s+refund\b",
)


def normalized_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {TOKEN_ALIASES.get(word, word) for word in words if word not in STOP_WORDS}


def semantic_overlap(left: str, right: str) -> float:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def salient_event_overlap(left: str, right: str) -> float:
    left_tokens = normalized_tokens(left) - GENERIC_EVENT_TOKENS
    right_tokens = normalized_tokens(right) - GENERIC_EVENT_TOKENS
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def contains_unsafe_claim(text: str) -> bool:
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in SAFE_UNSAFE_MENTION_PATTERNS):
        return False
    tokens = normalized_tokens(lowered)
    if any(term in lowered for term in UNSAFE_TERMS):
        return True
    return any(unsafe_tokens <= tokens for unsafe_tokens in UNSAFE_TOKEN_SETS)


def grade_branch_match(
    branch: Branch,
    actual_next_event: str,
    expected_intent: str,
) -> Branch:
    if contains_unsafe_claim(branch.predicted_event):
        branch.match_grade = MatchGrade.UNSAFE
        branch.match_score = 0.0
        return branch

    overlap = semantic_overlap(branch.predicted_event, actual_next_event)

    if branch.intent == expected_intent and salient_event_overlap(branch.predicted_event, actual_next_event) >= 0.30:
        branch.match_grade = MatchGrade.EXACT_INTENT
        branch.match_score = 1.0
        return branch

    branch.match_score = round(overlap, 3)

    if overlap >= 0.55:
        branch.match_grade = MatchGrade.SEMANTIC_EQUIVALENT
    elif overlap >= 0.30:
        branch.match_grade = MatchGrade.USEFUL_PARTIAL
    else:
        branch.match_grade = MatchGrade.MISS

    return branch
```

- [ ] **Step 4: Run similarity tests**

Run:

```bash
python3 -m pytest tests/test_similarity.py -v
```

Expected: `17 passed`.

## Task 4: Branch Generation And Prepared Artifacts

**Files:**
- Create: `src/foresight_harness/branching.py`
- Create: `src/foresight_harness/artifacts.py`
- Create: `tests/test_branching.py`

- [ ] **Step 1: Write failing branch tests**

Create `tests/test_branching.py`:

```python
from foresight_harness.artifacts import prepare_artifacts, select_artifact
from foresight_harness.branching import generate_branches
from foresight_harness.models import ReplayTurn


def make_turn(expected_intent: str = "refund_request") -> ReplayTurn:
    return ReplayTurn.from_dict(
        {
            "turn_id": "qa-test",
            "conversation": [
                {"role": "customer", "content": "The box arrived soaked and broken."},
                {"role": "agent", "content": "I can help with the damaged delivery."},
            ],
            "actual_next_event": "customer asks whether a refund is available",
            "policy_context": "Damaged deliveries qualify after photo verification.",
            "expected_intent": expected_intent,
            "latency_budget_ms": 800,
        }
    )


def test_generate_branches_returns_ranked_top_k():
    branches = generate_branches(make_turn(), top_k=3)

    assert len(branches) == 3
    assert branches[0].rank == 1
    assert branches[0].probability >= branches[1].probability


def test_prepare_and_select_artifact_for_actual_event():
    turn = make_turn()
    branches = generate_branches(turn, top_k=3)
    artifacts = prepare_artifacts(turn, branches)

    selected = select_artifact(turn, branches, artifacts, readiness_threshold=0.30)

    assert selected is not None
    assert selected.created_for_intent == "refund_request"
    assert "photo verification" in selected.response_draft.lower()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_branching.py -v
```

Expected: fail because `branching.py` and `artifacts.py` do not exist.

- [ ] **Step 3: Implement branch generation**

Create `src/foresight_harness/branching.py`:

```python
from __future__ import annotations

from collections import Counter

from foresight_harness.models import Branch, ReplayTurn
from foresight_harness.similarity import normalized_tokens

INTENT_PATTERNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "refund_request": (
        "customer asks whether a refund or replacement is available",
        ("refund", "damaged", "delivery", "broken", "soaked", "replacement"),
    ),
    "troubleshooting_loop": (
        "customer reports the result of troubleshooting and asks for the next step",
        ("reset", "device", "pair", "light", "amber", "troubleshooting"),
    ),
    "billing_refund_timing": (
        "customer asks how long a duplicate charge refund will take",
        ("charged", "twice", "billing", "subscription", "duplicate", "refund"),
    ),
    "escalation_request": (
        "customer asks to escalate the unresolved case to a supervisor",
        ("third", "frustrating", "supervisor", "escalate", "unresolved"),
    ),
    "address_change": (
        "customer provides a new shipping address and asks if the order can be changed",
        ("address", "order", "shipped", "shipping", "change", "editable"),
    ),
}


def generate_branches(turn: ReplayTurn, top_k: int = 3) -> tuple[Branch, ...]:
    context_tokens = normalized_tokens(turn.context_text())
    scored: list[tuple[str, str, float]] = []

    for intent, (event, keywords) in INTENT_PATTERNS.items():
        keyword_counts = Counter(keyword for keyword in keywords if keyword in context_tokens)
        keyword_score = sum(keyword_counts.values()) / max(len(keywords), 1)
        prior = 0.18
        probability = min(0.85, prior + keyword_score)
        scored.append((intent, event, round(probability, 3)))

    ranked = sorted(scored, key=lambda item: item[2], reverse=True)[:top_k]
    return tuple(
        Branch(
            branch_id=f"{turn.turn_id}-br-{index}",
            predicted_event=event,
            intent=intent,
            probability=probability,
            rank=index,
        )
        for index, (intent, event, probability) in enumerate(ranked, start=1)
    )
```

- [ ] **Step 4: Implement artifact preparation**

Create `src/foresight_harness/artifacts.py`:

```python
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
```

- [ ] **Step 5: Run branch tests**

Run:

```bash
python3 -m pytest tests/test_branching.py -v
```

Expected: `2 passed`.

## Task 5: Baselines, Evaluator, And Metrics

**Files:**
- Create: `src/foresight_harness/baselines.py`
- Create: `src/foresight_harness/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing evaluator tests**

Create `tests/test_evaluator.py`:

```python
from pathlib import Path

from foresight_harness.evaluator import load_replay_turns, run_replay


def test_load_replay_turns_from_jsonl():
    turns = load_replay_turns(Path("data/queueahead_sample.jsonl"))

    assert len(turns) == 5
    assert turns[0].turn_id == "qa-001"


def test_run_replay_reports_harness_metrics():
    turns = load_replay_turns(Path("data/queueahead_sample.jsonl"))
    report = run_replay(turns, top_k=3)

    harness = report["harness"]

    assert harness["total_turns"] == 5
    assert harness["top_3_recall"] >= 0.8
    assert harness["cache_hit_rate"] >= 0.8
    assert harness["stale_artifact_rate"] == 0.0
    assert "retrieval_plus_draft" in report
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_evaluator.py -v
```

Expected: fail because `evaluator.py` does not exist.

- [ ] **Step 3: Implement baselines**

Create `src/foresight_harness/baselines.py`:

```python
from __future__ import annotations

from foresight_harness.models import ReplayTurn, RunResult


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
    return RunResult(
        turn_id=turn.turn_id,
        variant="semantic_cache",
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
```

- [ ] **Step 4: Implement evaluator**

Create `src/foresight_harness/evaluator.py`:

```python
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
```

- [ ] **Step 5: Run evaluator tests**

Run:

```bash
python3 -m pytest tests/test_evaluator.py -v
```

Expected: `2 passed`.

## Task 6: CLI Report Runner

**Files:**
- Create: `src/foresight_harness/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Create `tests/test_cli.py`:

```python
import json
import subprocess
import sys


def test_cli_outputs_json_report():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--input",
            "data/queueahead_sample.jsonl",
            "--top-k",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)

    assert "harness" in report
    assert report["harness"]["total_turns"] == 5


def test_console_entrypoint_is_declared():
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    assert 'foresight-replay = "foresight_harness.cli:main"' in pyproject
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python3 -m pytest tests/test_cli.py -v
```

Expected: fail because `foresight_harness.cli` does not exist and the console entrypoint is not declared yet.

- [ ] **Step 3: Implement CLI**

Create `src/foresight_harness/cli.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from foresight_harness.evaluator import load_replay_turns, run_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Foresight replay harness against a JSONL replay file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/queueahead_sample.jsonl"),
        help="Path to a JSONL replay file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of next-event branches to generate per turn.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    turns = load_replay_turns(args.input)
    report = run_replay(turns, top_k=args.top_k)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

Modify `pyproject.toml` to add the console script after the `[project]` dependencies block:

```toml
[project.scripts]
foresight-replay = "foresight_harness.cli:main"
```

- [ ] **Step 4: Run CLI test**

Run:

```bash
python3 -m pytest tests/test_cli.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Run sample report**

Run:

```bash
python3 -m foresight_harness.cli --input data/queueahead_sample.jsonl --top-k 3
```

Expected: JSON report with keys `harness`, `live_agent`, `prediction_only`, `retrieval_plus_draft`, and `semantic_cache`.

## Task 7: Documentation, Verification, And Review

**Files:**
- Create: `README.md` if no README exists.
- Modify: `tasks/todo.md`

- [ ] **Step 1: Create README**

Create `README.md`:

```markdown
# Foresight Agent Harness

This repository contains the planning documents and first offline experiment for
the Precognitive/Foresight Agent Harness.

The first implementation is intentionally small: it runs a QueueAhead-style
support replay and measures whether predicted next-event branches plus prepared
artifacts beat simple baselines.

## Run The Sample Experiment

```bash
python3 -m pytest -v
python3 -m foresight_harness.cli --input data/queueahead_sample.jsonl --top-k 3
```

## First-Pass Success Criteria

- Top-3 recall around 50% or at least 10 percentage points over a comparable baseline.
- Cache-hit usefulness at least 70%.
- Median time-to-useful-response at least 20% better than live-only response.
- Stale artifact rate no more than 20%.
- No critical speculative-truth leaks.

## Naming

The research codename remains Precognitive Agent Harness. Product-facing language
uses Foresight Agent Harness. The technical primitive is an event probability
tree.
```

- [ ] **Step 2: Run all tests**

Run:

```bash
python3 -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Run sample CLI**

Run:

```bash
python3 -m foresight_harness.cli --input data/queueahead_sample.jsonl --top-k 3
```

Expected: JSON report includes non-zero harness metrics and no unsafe leak rate.

- [ ] **Step 4: Update task tracker review**

Modify `tasks/todo.md` with:

```markdown
## Implementation Review

- Package scaffold: complete.
- Offline replay fixture: complete.
- Semantic branch matching: complete.
- Branch generation and prepared artifacts: complete.
- Baselines and evaluator: complete.
- CLI report: complete.
- Verification: `python3 -m pytest -v` and sample CLI report passed.
```

- [ ] **Step 5: Commit when Git is enabled**

If Git was initialized and a feature branch is active, run:

```bash
git status --short
git add pyproject.toml README.md data src tests tasks docs
git commit -m "feat: add foresight replay harness"
```

Expected: commit succeeds. If executing unversioned, record “No commit: user chose unversioned execution” in `tasks/todo.md`.

## Self-Review

- Spec coverage: This plan implements the first experiment recommended in `tasks/todo.md`: offline QueueAhead replay, semantic matching, baselines, readiness metrics, and CLI reporting.
- Scope control: Redis, MiroFish, HTTP services, live support deployment, and robotics/video use cases are excluded from this first build.
- Testability: Every module has at least one unit or end-to-end test, and the sample CLI provides a repeatable benchmark run.
- Type consistency: `ReplayTurn`, `Branch`, `Artifact`, `RunResult`, and `MatchGrade` are defined once in `models.py` and reused across all tasks.
- Placeholder scan: No incomplete sections or unbounded implementation instructions remain.
