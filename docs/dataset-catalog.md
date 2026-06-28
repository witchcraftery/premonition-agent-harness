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

### Human Conversation Sample

- Status: included in this repository.
- File: `data/human_conversation_sample.jsonl`.
- Report: `runs/human_conversation_probability_loop.json`.
- Purpose: tiny DailyDialog-style fixture for testing ordinary next-turn probability, TTS-ready draft preparation, and conversational guidance refinement.
- Current use: proves the Probability Pack path before importing external human-dialogue datasets.
- Limitation: synthetic and tiny; useful only as a mechanics check.

### DailyDialog Train Sample

- Status: derived sample included in this repository; raw source files are excluded under `data/external/`.
- File: `data/dailydialog_train_sample.jsonl`.
- Report: `runs/dailydialog_train_probability_loop.json`.
- Source mirror used for import: `https://github.com/snakeztc/NeuralDialog-LAED/tree/master/data/daily_dialog/train`.
- Purpose: first real human-dialogue sample for ordinary dialogue-act probability and TTS-ready draft readiness.
- Current use: baseline check for DailyDialog import mechanics and no-regression guidance promotion.
- Current result: `p_at_1=0.472`, `top_3_recall=0.920`, `tts_readiness_rate=1.0`; candidate guidance was rejected because it would reduce `p_at_1` to `0.412`.
- Limitation: first 500 exported train turns only; useful as an import and same-split mechanics check.

### DailyDialog Held-Out Samples

- Status: derived samples included in this repository; raw source files are excluded under `data/external/`.
- Files: `data/dailydialog_train_sample.jsonl`, `data/dailydialog_validation_sample.jsonl`, and `data/dailydialog_test_sample.jsonl`.
- Reports: `runs/dailydialog_heldout_probability_loop.json` and `runs/dailydialog_act_ranker_bakeoff.json`.
- Source mirror used for import: `https://github.com/snakeztc/NeuralDialog-LAED/tree/master/data/daily_dialog`.
- Purpose: true train/dev/test efficacy loop for ordinary conversational act prediction and TTS-ready branch preparation.
- Current use: train learns candidate guidance, validation gates promotion, and untouched test measures generalization.
- Current result: act-specific candidate guidance raised validation `p_at_1` from `0.324` to `0.402`, but reduced validation `top_3_recall` from `0.856` to `0.844`, so the segment-aware gate rejected promotion.
- Segment signal: rejected candidates regressed validation `commissive` from `0.213` to `0.066`, `directive` from `0.147` to `0.010`, and `question` from `0.040` to `0.008`.
- Held-out result after gating: the keyword-guidance loop remains at the baseline, `p_at_1=0.368` and `top_3_recall=0.858`, with `0` improved and `0` regressed held-out test turns.
- Contextual bake-off: cross-validation selected `directive_act_rhythm_contextual`, which keeps guarded contextual scoring but only lets longer act-history override the guard for `directive` turns. It improved validation `p_at_1` from `0.324` to `0.466` and held-out test `p_at_1` from `0.368` to `0.504`, with no act-segment regressions.
- Cross-validation signal: the selected directive specialist has mean internal `p_at_1` gain `0.104`, minimum fold gain `0.090`, and `0` segment regressions.
- Raw transition signal: full contextual transition reached test `p_at_1=0.534` and `top_3_recall=0.970`, but regressed `directive` and `question` top-1 accuracy to `0.0`, so it remains diagnostic rather than promoted behavior.
- Diagnostic signal: `act_rhythm_contextual_strict` reached held-out test `p_at_1=0.508`, cross-validation mean gain `0.112`, and minimum fold gain `0.100`, but it showed `1` internal act-segment regression, so it remains diagnostic rather than promoted behavior.
- Specialist signal: the combined protected-act specialist reached held-out test `p_at_1=0.514`, and the question-only specialist reached `0.512`, but both showed `1` internal cross-validation segment regression. The directive-only specialist is promoted because it is smaller but stable.
- Limitation: first 500 turns from each split; the next lever is stabilizing the question specialist without reducing directive accuracy.

## Human Conversation Candidates

### DailyDialog

- Fit: everyday multi-turn human dialogue with dialogue-act and emotion labels.
- Why it matters: best first external dataset for ordinary conversational next-move prediction.
- Link: https://arxiv.org/abs/1710.03957
- Trial question: can the backend predict the next conversational act and prewarm speakable draft branches, then promote only guidance that improves held-out dialogue-act readiness?

### EmpatheticDialogues

- Fit: emotionally grounded conversations.
- Why it matters: voice presence depends on emotional readiness, not only fast answers.
- Link: https://github.com/facebookresearch/EmpatheticDialogues
- Trial question: can the backend prepare emotionally appropriate response modes without sounding premature?

### Blended Skill Talk

- Fit: blends personality, empathy, and knowledge in conversation.
- Why it matters: tests whether the backend can predict conversational mode shifts.
- Link: https://parl.ai/docs/tasks.html
- Trial question: can probability packs choose between empathy, personality, and knowledge branches?

### Taskmaster

- Fit: spoken and written task-oriented assistant conversations.
- Why it matters: bridge between ordinary conversation and practical voice-agent tasks.
- Link: https://github.com/google-research-datasets/Taskmaster
- Trial question: can the backend prewarm likely task responses while waiting for user confirmation?

### SpokenWOZ

- Fit: speech-text task-oriented dialogue with audio.
- Why it matters: later-stage realism for ASR messiness, spoken timing, and TTS prewarming.
- Link: https://spokenwoz.github.io/
- Trial question: can probability refinement improve readiness in spoken assistant flows?

## Support And Agent Candidates

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
