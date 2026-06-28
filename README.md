![Premonition Foresight Probability Tree](assets/premonition-foresight-tree.svg)

# Premonition / Foresight Agent Harness

Premonition, in this experiment, does not mean prophecy. It means an agent learning to stand ready: sensing the present context, imagining the next few likely moves, preparing useful work for each branch, and letting observed truth choose the path.

This first build focuses on the **Foresight / QueueAhead** version of the idea.
Given a conversation turn, it generates the top likely next-event branches, prepares draft artifacts for those branches, then replays the actual next event and scores whether the prepared work was useful, safe, and faster than baseline behavior.

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

The frontend LLM performs the conversation. The Premonition Backend performs the rehearsal.

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

The current challenge loop improves the harness from `p_at_1=0.333` and
`usefulness_rate=0.167` on iteration 1 to `p_at_1=1.0` and
`usefulness_rate=1.0` on iteration 2, then holds those scores on iteration 3.

Run a train/test split benchmark:

```bash
foresight-replay \
  --train-config experiments/queueahead_challenge_train.json \
  --test-config experiments/queueahead_challenge_test.json \
  --iterations 3 \
  --benchmark-report runs/queueahead_split_benchmark.json
```

The current split benchmark learns guidance on the train split and improves the held-out test split from `p_at_1=0.5` to `p_at_1=1.0`, with `test_p_at_1_gain=0.5` and `overfit_gap=0.167`.

The split report also includes analytics by:

- `actor`: user, agent, or environment event source.
- `event_type`: escalation, billing, troubleshooting, account update, refund, or unknown.
- `topic`: the expected intent/topic label.

Use these sections to see which areas improved, which stayed weak, and whether the backend is predicting user/environment events instead of its own next move. The challenge split now includes two non-user fulfillment events under `shipment_status_update`, classified as actor `environment`: a warehouse lock event and a harder carrier exception hold event. Held-out environment-event `p_at_1` improves from `0.5` to `1.0`.

Run the enriched 5-fold benchmark:

```bash
foresight-replay \
  --fold-config experiments/queueahead_enriched_folds.json \
  --folds 5 \
  --iterations 3 \
  --benchmark-report runs/queueahead_enriched_cross_benchmark.json \
  --dashboard-report runs/queueahead_enriched_dashboard.html
```

The enriched loop uses 30 synthetic replay turns with hard environment events,
user events, and decoy cues. Each fold trains on three-fifths of the data, uses
one-fifth as a dev promotion gate, then scores the final held-out fifth.

The current targeted-support 5-fold report shows held-out `p_at_1` at
`0.967 -> 0.967`, environment-event `p_at_1` at `1.000 -> 1.000`, and
user-event `p_at_1` at `0.950 -> 0.950`. The flat guided gain is now useful
signal: the profile and negative-cue rules have moved the known support wins
into the default backend behavior, so learned guidance only promotes on folds
where it still has something to add. The guidance delta reports `0` improved
and `0` regressed held-out turns.

The original weak profiles are now solved in this synthetic set:
`payment_gateway_update`, `policy_update`, `fraud_review_lock`,
`carrier_exception_hold`, `inventory_backorder`, `refund_request`, and
`address_change` all report `1.0` held-out profile `p_at_1`. The remaining weak
area is escalation wording at `0.875`, which is high enough that the next useful
test is probably a harder dataset or a domain-shift benchmark rather than more
cue polishing on this small support set.

Open `runs/queueahead_enriched_dashboard.html` in a browser for a quick visual
reference of overall accuracy, actor performance, weakest segments, profile
performance, guidance deltas, and fold-by-fold results.

Run the first human-conversation probability loop:

```bash
foresight-replay \
  --conversation-input data/human_conversation_sample.jsonl \
  --iterations 3 \
  --conversation-report runs/human_conversation_probability_loop.json
```

This starts the voice-agent swarm-mind path. Instead of predicting support
ticket events, it predicts the next ordinary conversational act, prepares
voice-ready drafts for likely branches, and waits for observed confirmation
before a draft would be spoken. The first tiny DailyDialog-style fixture
improves `p_at_1` from `0.75` to `1.0` by iteration 2, keeps `top_3_recall` at
`1.0`, and keeps `tts_readiness_rate` at `1.0`.

The current Probability Pack includes:

- `top_branches`: likely next conversational acts with probabilities.
- `prepared_drafts`: speakable TTS-ready draft templates.
- `confirmation_mode`: `wait_for_observed_next_move`.
- `expires_after_ms`: a short freshness window for live voice use.

Run the first real DailyDialog sample:

```bash
foresight-replay \
  --dailydialog-dir data/external/dailydialog/train \
  --conversation-output data/dailydialog_train_sample.jsonl \
  --conversation-limit 500

foresight-replay \
  --conversation-input data/dailydialog_train_sample.jsonl \
  --iterations 3 \
  --conversation-report runs/dailydialog_train_probability_loop.json
```

The external DailyDialog files are not committed; keep them under
`data/external/`. The committed 500-turn sample reports `p_at_1=0.472`,
`top_3_recall=0.920`, and `tts_readiness_rate=1.0`. Candidate guidance was
rejected because it would have reduced `p_at_1` to `0.412`, which is the right
behavior for a looped backend: do not promote learned probability rules that
make held replay worse.

Run the true held-out DailyDialog loop:

```bash
foresight-replay \
  --dailydialog-dir data/external/dailydialog/validation \
  --conversation-output data/dailydialog_validation_sample.jsonl \
  --conversation-limit 500

foresight-replay \
  --dailydialog-dir data/external/dailydialog/test \
  --conversation-output data/dailydialog_test_sample.jsonl \
  --conversation-limit 500

foresight-replay \
  --conversation-train-input data/dailydialog_train_sample.jsonl \
  --conversation-dev-input data/dailydialog_validation_sample.jsonl \
  --conversation-test-input data/dailydialog_test_sample.jsonl \
  --iterations 3 \
  --conversation-report runs/dailydialog_heldout_probability_loop.json
```

This is the more honest efficacy loop: train learns candidate guidance, the
validation split decides whether to promote it, and the untouched test split
measures whether the promoted guidance actually generalized. The current
refined loop uses act-specific token learning and a segment-aware promotion
gate. On the 500/500/500 DailyDialog samples, the candidate guidance would have
raised validation `p_at_1` from `0.324` to `0.402`, but it also dropped
validation `top_3_recall` from `0.856` to `0.844` and regressed the
`commissive`, `directive`, and `question` act segments. The gate rejected it in
all three iterations, so held-out test stayed at the baseline:
`p_at_1=0.368`, `top_3_recall=0.858`, and `0` improved / `0` regressed test
turns.

That is useful signal, not a victory lap: the act-specific learner is now safer,
but not yet more capable on held-out DailyDialog. The next refinement should
learn smaller segment-local patches or branch-calibration weights so an
aggregate validation win cannot depend on hurting weak conversational acts.

Run the learned act-ranker bake-off:

```bash
foresight-replay \
  --conversation-train-input data/dailydialog_train_sample.jsonl \
  --conversation-dev-input data/dailydialog_validation_sample.jsonl \
  --conversation-test-input data/dailydialog_test_sample.jsonl \
  --conversation-bakeoff-report runs/dailydialog_act_ranker_bakeoff.json
```

This compares the current heuristic brancher against a transparent learned
act-ranker and hybrid blends. On the committed 500/500/500 samples, validation
selected the current heuristic brancher. The learned-only variant overfit train
(`p_at_1=0.656`) but fell to `p_at_1=0.260` on validation and `0.274` on test;
hybrid variants also regressed `inform`, `commissive`, or `directive` segments.
The result is a useful negative finding: a bag-of-features act classifier is not
the bigger-improvement lever yet. The next likely lever is a contextual brancher
that reads the latest turn structure, not just token counts.

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

The sample data is intentionally tiny and deterministic. For support, the next
meaningful step is to replay 300-500 real or tau-bench-style support turns,
label branch-match grades, and compare the harness against the baseline variants
using the same report schema.

For conversational voice-agent foresight, the next meaningful step is to import
larger and more representative DailyDialog slices, then improve act-specific
learning until validation candidates become clear held-out test gains without
segment regressions. After that, add EmpatheticDialogues for emotional readiness
and Taskmaster or SpokenWOZ for practical spoken-assistant flows.

Useful expansion points:

- Replace deterministic keyword branch generation with model-generated branches.
- Add human labels for `exact_intent`, `semantic_equivalent`, `useful_partial`, `miss`, and `unsafe`.
- Track stale artifact rates when policies, user state, or account context changes.
- Add a real semantic scorer after the deterministic benchmark is stable.
- Add perceived-latency metrics for TTS prewarming once a voice runtime is attached.
- Add segment-aware promotion gates for conversational acts, so aggregate gains do not hide brittle regressions.
- Replace bag-of-features act ranking with contextual branch generation that can inspect the latest utterance role, dialogue rhythm, and candidate reply mode.

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

The first guided loop changes one lever: learned intent cues. The loop records which turns were exact top-1 hits, which remained missed, which prepared useful artifacts, and which were still unprepared.

The split benchmark adds a generalization gate: guidance is learned on a train split, evaluated on a separate test split, and promoted only when held-out accuracy improves without unsafe leakage.

The enriched cross-fold benchmark adds a dev gate before the final held-out test: guidance is learned on train, checked on dev for no-regression promotion, then scored on test. It reports aggregate mean/min/max values and weakest segments across folds so we can see whether gains are broad or concentrated.

The dashboard report turns that same JSON into a static HTML page for quick
review. It is intentionally built from the benchmark artifact, not a separate
data path, so visual peeks and JSON analysis stay aligned. When the backend
baseline absorbs previous guidance wins, the dashboard makes that visible
through flat guided gains, high baseline scores, lower promotion rate, and
zero-regression fold rows.

Each split report also includes segment analytics and focus areas. These make it possible to track whether improvements are broad or concentrated in a few topics, and to identify the next areas that deserve new data or better guidance.

Candidate benchmark families are tracked in `docs/dataset-catalog.md`.
