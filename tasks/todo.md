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
