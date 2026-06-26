# Precognitive Agent Harness Review

## Plan

- [x] Read the concept sheet, system spec, and implementation plan.
- [x] Assess the core idea for product merit, technical novelty, and risks.
- [x] Compare the plan against current agent evaluation and benchmarking practices.
- [x] Recommend clearer MVP scope, experiments, metrics, and benchmarks.
- [x] Document review results and final recommendation.

## Review

The concept has merit if framed as predictive readiness caching, not literal
precognition. The strongest MVP is QueueAhead in one support domain, measured
through offline replay against retrieval-plus-draft, semantic cache, and live
agent baselines. SceneSense remains useful as an offline evaluation sandbox, but
QueueAhead has clearer product value and benchmark fit.

Required plan changes before building:

- Move evaluation and branch-match labeling into the first phase.
- Make semantic matching a core MVP feature; exact content-hash lookup is only a
  throwaway spike.
- Define branch-match grades: exact intent, semantic equivalent, useful partial,
  miss, unsafe.
- Treat MiroFish/deep simulation as an optional ablation after a simpler branch
  generator earns its complexity.
- Add explicit baseline-to-beat sections for each product concept.
- Fix unresolved citations and malformed math formatting in the system spec.

Suggested first experiment:

- Run a two-week offline replay using 300-500 support turns or tau-bench-style
  tasks.
- Compare live agent, retrieval-plus-draft, semantic cache, prediction-only, and
  harness variants.
- Pass only if the harness improves practical readiness without safety or
  quality regression: top-3 recall around 50% or +10pp over baseline, cache-hit
  usefulness at least 70%, median time-to-useful-response at least 20% better,
  stale artifacts no more than 20%, and no critical speculative-truth leaks.

## Foresight Replay Harness Implementation

- [x] Write implementation plan.
- [x] Confirm version-control mode before code work.
- [x] Create Python package scaffold.
- [x] Add replay models and sample data.
- [x] Add semantic branch matching.
- [x] Add branch generation and artifacts.
- [x] Add baselines and evaluator.
- [x] Add CLI report runner.
- [x] Add README and final implementation notes.
- [x] Verify tests and sample report.

## Implementation Review

Implemented a first-pass Foresight / QueueAhead replay harness as a local Python
package. The harness loads replay turns, generates ranked next-event branches,
prepares policy-aware artifacts, grades branch matches, compares four baselines,
and emits a JSON metrics report through the `foresight-replay` command.

Final verification:

- `python3 -m pytest -v`: 32 passed.
- `foresight-replay --top-k 3` from `/tmp`: emitted the bundled sample report.
- `python3 -m foresight_harness.cli --input data/queueahead_sample.jsonl --top-k 3`: emitted the repo sample report.

Sample harness result: `p_at_1=1.0`, `top_3_recall=1.0`,
`cache_hit_rate=1.0`, `median_latency_ms=120`, `unsafe_leak_rate=0.0`.

## Premonition Backend Trial Scaffolding

- [x] Define backend packet/playbook shape.
- [x] Add experiment config structure.
- [x] Add per-turn replay logs and miss analysis hooks.
- [x] Add dataset catalog for benchmark candidates.
- [x] Commit scaffolding before trial execution.
- [x] Run tests and first configured trial.
- [x] Push verified trial scaffolding to GitHub.

Verification:

- `python3 -m pytest -v`: 36 passed.
- `foresight-replay --config experiments/queueahead_v1.json --turn-log runs/queueahead_v1.turns.jsonl --miss-report runs/queueahead_v1.misses.json`: completed.
- First configured trial emitted 25 per-turn rows and a miss report with 5 exact harness hits.

## Guided Premonition Loop

- [x] Add challenge replay split for harder QueueAhead turns.
- [x] Add learned guidance loop with configurable iteration count.
- [x] Assess exact top-1 hits, misses, prepared turns, and unprepared turns per iteration.
- [x] Filter low-signal guidance tokens.
- [x] Regenerate challenge loop report and guidance markdown.

Verification:

- `python3 -m pytest -v`: 41 passed.
- `foresight-replay --config experiments/queueahead_challenge_loop.json --iterations 3 --loop-report runs/queueahead_challenge_loop.json --guidance-markdown runs/queueahead_challenge_guidance.md`: completed.
- Challenge loop improved `p_at_1` from `0.25` to `1.0` and `usefulness_rate` from `0.0` to `1.0` by iteration 2, then held steady on iteration 3.

## Train/Test Split Benchmark

- [x] Add held-out QueueAhead challenge test split.
- [x] Add train/test experiment configs.
- [x] Add split benchmark runner.
- [x] Add CLI split benchmark mode.
- [x] Record split benchmark report.

Verification:

- `python3 -m pytest -v`: 44 passed.
- `foresight-replay --train-config experiments/queueahead_challenge_train.json --test-config experiments/queueahead_challenge_test.json --iterations 3 --benchmark-report runs/queueahead_split_benchmark.json`: completed.
- Train `p_at_1` improved from `0.25` to `1.0`; test `p_at_1` improved from `0.5` to `1.0`; overfit gap was `0.25`; guidance was promotable.

## Segment Analytics

- [x] Classify replay turns by actor, event type, and topic.
- [x] Include event metadata in turn logs.
- [x] Summarize split benchmark performance by actor, event type, and topic.
- [x] Add focus-area reporting for strongest/weakest segments.

Verification:

- `python3 -m pytest -v`: 47 passed.
- Regenerated `runs/queueahead_split_benchmark.json` with `analytics.test_segments`, `analytics.train_segments`, and `analytics.focus_areas`.
- Initial held-out analytics showed only user events; escalation and troubleshooting improved by `+1.0` `p_at_1`, motivating explicit environment-event coverage next.

## Environment Event Coverage

- [x] Add non-user fulfillment/status turns to train and test challenge splits.
- [x] Add `shipment_status_update` branch pattern.
- [x] Classify fulfillment events as actor `environment`.
- [x] Regenerate split benchmark with user and environment actor segments.

Verification:

- `python3 -m pytest -v`: 49 passed.
- Current held-out test analytics include `user` and `environment` actor segments.
- Initial environment coverage added a warehouse-lock external event and confirmed both `user` and `environment` actor segments were reported.
- The following hard-environment pass added carrier exception hold events and required the environment segment to improve on held-out data.

## Hard Environment Event Coverage

- [x] Add carrier exception hold turns to train and test challenge splits.
- [x] Require the held-out environment segment to improve, not merely stay high.
- [x] Make shipment-status branch text use clustered learned cues for hard external events.
- [x] Regenerate challenge loop and split benchmark reports.

Verification:

- `python3 -m pytest -v`: 49 passed.
- `foresight-replay --config experiments/queueahead_challenge_loop.json --iterations 3 --loop-report runs/queueahead_challenge_loop.json --guidance-markdown runs/queueahead_challenge_guidance.md`: completed.
- `foresight-replay --train-config experiments/queueahead_challenge_train.json --test-config experiments/queueahead_challenge_test.json --iterations 3 --benchmark-report runs/queueahead_split_benchmark.json`: completed.
- Challenge loop improved from `p_at_1=0.333` and `usefulness_rate=0.167` on iteration 1 to `p_at_1=1.0` and `usefulness_rate=1.0` on iteration 2.
- Held-out test `p_at_1` improved from `0.5` to `1.0`; environment-event `p_at_1` improved from `0.5` to `1.0`; overfit gap was `0.167`; guidance remained promotable.

## Enriched Benchmark Loop

- [x] Add a richer hard-event pack with environment events, user events, and decoy/near-miss cues.
- [x] Add dev/test promotion so guidance can be selected before the final held-out test.
- [x] Add a cross-fold benchmark runner for repeated train/dev/test trials.
- [x] Report aggregate mean/min/max metrics and weakest segments across folds.
- [x] Add CLI support for running the enriched benchmark loop.
- [x] Regenerate benchmark artifacts and document the new loop.

Verification:

- `python3 -m pytest -v`: 53 passed.
- `foresight-replay --fold-config experiments/queueahead_enriched_folds.json --folds 5 --iterations 3 --benchmark-report runs/queueahead_enriched_cross_benchmark.json`: completed.
- Enriched 5-fold held-out `p_at_1` improved from `0.567` to `0.667`; usefulness improved from `0.567` to `0.667`.
- Environment-event `p_at_1` improved from `0.067` to `0.317`; user-event `p_at_1` improved from `0.91` to `0.95`.
- Promotion rate was `1.0`; weakest segments were environment events, refund, billing, and fulfillment/shipment status.
