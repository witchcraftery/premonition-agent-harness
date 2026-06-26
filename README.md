# Precognitive Agent Harness

An offline replay harness for testing whether an agent can prepare useful next
moves before the user asks for them.

This first build focuses on the **Foresight / QueueAhead** version of the idea:
given a conversation turn, generate the top likely next-event branches, prepare
draft artifacts for those branches, then replay the actual next event and score
whether the prepared work was useful, safe, and faster than baseline behavior.

## What It Measures

The harness compares five variants:

- `live_agent`: waits for the next user message, then responds.
- `retrieval_plus_draft`: retrieves context and drafts after the next event is known.
- `semantic_cache`: reuses a semantically similar cached answer when available.
- `prediction_only`: predicts likely next events but prepares no usable artifact.
- `harness`: predicts top-k branches, prepares artifacts, and selects the best match during replay.

The report includes:

- `p_at_1`: how often the top predicted branch exactly matches the actual next intent.
- `top_3_recall`: how often the actual next intent appears in the top three branches.
- `cache_hit_rate`: how often a usable prepared artifact is selected.
- `median_latency_ms`: simulated response readiness latency.
- `median_token_cost`: simulated preparation/runtime cost.
- `usefulness_rate`: how often the variant produces a usable result.
- `unsafe_leak_rate`: how often a branch violates a safety or policy constraint.
- `stale_artifact_rate`: placeholder for future freshness checks.

## Quick Start

```bash
python3 -m pip install -e .
foresight-replay --top-k 3
```

You can also run against the readable sample fixture in the repo:

```bash
python3 -m foresight_harness.cli --input data/queueahead_sample.jsonl --top-k 3
```

Run the test suite:

```bash
python3 -m pytest -v
```

## Replay Data Format

Replay input is JSONL. Each line is one conversation turn:

```json
{
  "turn_id": "qa-001",
  "conversation": [
    {"role": "customer", "content": "My delivery arrived damaged."},
    {"role": "agent", "content": "I can help with that."}
  ],
  "actual_next_event": "customer asks whether a refund is available",
  "policy_context": "Damaged deliveries qualify for refund or replacement after photo verification.",
  "expected_intent": "refund_request",
  "latency_budget_ms": 800
}
```

## Next Experiments

The sample data is intentionally tiny and deterministic. The next meaningful
step is to replay 300-500 support turns, label branch-match grades, and compare
the harness against the baseline variants using the same report schema.

Useful expansion points:

- Replace deterministic keyword branch generation with model-generated branches.
- Add human labels for `exact_intent`, `semantic_equivalent`, `useful_partial`, `miss`, and `unsafe`.
- Track stale artifact rates when policies, user state, or account context changes.
- Add a real semantic scorer after the deterministic benchmark is stable.
