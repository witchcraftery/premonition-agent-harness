# Dataset Catalog

This catalog lists candidate datasets for turning the Premonition Backend into a
repeatable benchmark.

## First Trial

### QueueAhead Sample

- Status: included in this repository.
- File: `data/queueahead_sample.jsonl`.
- Purpose: deterministic smoke test for the replay loop, packet schema, turn logs, and miss analysis.
- Limitation: too small to make performance claims.

### QueueAhead Enriched

- Status: included in this repository.
- File: `data/queueahead_enriched.jsonl`.
- Config: `experiments/queueahead_enriched_folds.json`.
- Reports: `runs/queueahead_enriched_cross_benchmark.json` and `runs/queueahead_enriched_dashboard.html`.
- Purpose: synthetic 5-fold train/dev/test benchmark with user events, hard environment events, and decoy cues.
- Current use: measures whether the Premonition Backend improves held-out performance across actors, topics, and event profiles while surfacing weak segments for the next data or brancher pass.
- Limitation: still synthetic; use it to harden the loop before importing larger external datasets.

## Primary External Candidates

### tau-bench / tau2

- Fit: customer-service and tool-agent conversations with policies and user goals.
- Why it matters: closest public shape to QueueAhead-style support premonition.
- Link: https://github.com/sierra-research/tau-bench
- Trial question: can the backend predict the next user/tool intent and prepare useful policy-grounded artifacts?

### Berkeley Function Calling Leaderboard

- Fit: tool-call selection and function-call correctness.
- Why it matters: useful for testing whether premonition predicts the next tool action.
- Link: https://gorilla.cs.berkeley.edu/leaderboard.html
- Trial question: can branch preparation improve next-tool readiness without increasing invalid calls?

### SWE-bench Lite

- Fit: coding-agent issue resolution.
- Why it matters: useful once the packet format supports code patches, file plans, and test predictions.
- Link: https://www.swebench.com/lite.html
- Trial question: can the backend predict likely files, tests, and patch direction before the frontend agent edits?

### OSWorld

- Fit: desktop/computer-use agents.
- Why it matters: later-stage test for long-horizon environment foresight.
- Link: https://os-world.github.io/
- Trial question: can the backend prepare next UI actions or recovery branches in interactive desktop tasks?

## Benchmark Promotion Rule

Do not promote a dataset from candidate to benchmark until it has:

- a replay adapter,
- a known ground-truth next event,
- a branch-match rubric,
- baseline runs,
- at least one holdout split that is not tuned during development.
