![Premonition Foresight Probability Tree](assets/premonition-foresight-tree.svg)

# Premonition / Foresight Agent Harness

Premonition, in this experiment, does not mean prophecy. It means an agent
learning to stand ready: sensing the present context, imagining the next few
likely moves, preparing useful work for each branch, and letting observed truth
choose the path.

This first build focuses on the **Foresight / QueueAhead** version of the idea.
Given a conversation turn, it generates the top likely next-event branches,
prepares draft artifacts for those branches, then replays the actual next event
and scores whether the prepared work was useful, safe, and faster than baseline
behavior.

The aspiration is a practical kind of knowing: not certainty, but readiness.
The benchmark asks whether that readiness can be measured.

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

## Backend Architecture

The frontend LLM performs the conversation. The Premonition Backend performs the
rehearsal.

The backend is organized around five pieces:

- **Branch generator**: predicts likely next events.
- **Artifact builder**: prepares one draft, policy check, or tool plan per useful branch.
- **Safety filter**: blocks unsafe or speculative claims from becoming frontend context.
- **Premonition packet**: hands the frontend compact readiness context only after a branch matches.
- **Replay evaluator**: compares predictions, preparedness, latency, cost, and safety across runs.

The packet format is intentionally small:

```json
{
  "turn_id": "qa-001",
  "matched_intent": "refund_request",
  "confidence": 0.85,
  "prepared_artifact": "I can help with a refund or replacement after photo verification.",
  "policy_checks": ["Damaged deliveries qualify after photo verification."],
  "freshness": "valid",
  "unsafe": false
}
```

See `PREMONITION.md` for the backend playbook.

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

Run the first trial loop and write per-turn evidence:

```bash
foresight-replay \
  --config experiments/queueahead_v1.json \
  --turn-log runs/queueahead_v1.turns.jsonl \
  --miss-report runs/queueahead_v1.misses.json
```

Run an iterative guidance loop on the challenge split:

```bash
foresight-replay \
  --config experiments/queueahead_challenge_loop.json \
  --iterations 3 \
  --loop-report runs/queueahead_challenge_loop.json \
  --guidance-markdown runs/queueahead_challenge_guidance.md
```

The current challenge loop improves the harness from `p_at_1=0.25` and
`usefulness_rate=0.0` on iteration 1 to `p_at_1=1.0` and
`usefulness_rate=1.0` on iteration 2, then holds those scores on iteration 3.

Run a train/test split benchmark:

```bash
foresight-replay \
  --train-config experiments/queueahead_challenge_train.json \
  --test-config experiments/queueahead_challenge_test.json \
  --iterations 3 \
  --benchmark-report runs/queueahead_split_benchmark.json
```

The current split benchmark learns guidance on the train split and improves the
held-out test split from `p_at_1=0.5` to `p_at_1=1.0`, with `test_p_at_1_gain=0.5`
and `overfit_gap=0.25`.

The split report also includes analytics by:

- `actor`: user, agent, or environment event source.
- `event_type`: escalation, billing, troubleshooting, account update, refund, or unknown.
- `topic`: the expected intent/topic label.

Use these sections to see which areas improved, which stayed weak, and whether
the backend is predicting user/environment events instead of its own next move.

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

## Benchmark Loop

The experimental loop is:

1. Freeze a replay split.
2. Generate top-k premonition branches.
3. Prepare artifacts for useful branches.
4. Reveal the actual next event.
5. Grade branch match and packet usefulness.
6. Analyze misses.
7. Change one backend lever.
8. Rerun the same split and compare.

The first guided loop changes one lever: learned intent cues. The loop records
which turns were exact top-1 hits, which remained missed, which prepared useful
artifacts, and which were still unprepared.

The split benchmark adds a generalization gate: guidance is learned on a train
split, evaluated on a separate test split, and promoted only when held-out
accuracy improves without unsafe leakage.

Each split report also includes segment analytics and focus areas. These make it
possible to track whether improvements are broad or concentrated in a few topics,
and to identify the next areas that deserve new data or better guidance.

Candidate benchmark families are tracked in `docs/dataset-catalog.md`.
