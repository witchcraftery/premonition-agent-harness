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
- Files: `data/dailydialog_train_sample.jsonl`, `data/dailydialog_validation_sample.jsonl`, `data/dailydialog_test_sample.jsonl`, plus larger 2k, 5k, and 6,740-turn samples.
- Reports: `runs/dailydialog_heldout_probability_loop.json`, `runs/dailydialog_act_ranker_bakeoff.json`, `runs/dailydialog_2k_act_ranker_bakeoff.json`, `runs/dailydialog_5k_act_ranker_bakeoff.json`, and `runs/dailydialog_6740_act_ranker_bakeoff.json`.
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
- Specialist signal: the combined protected-act specialist reached held-out test `p_at_1=0.514`, and the question-only specialist reached `0.512`, but both showed `1` internal cross-validation segment regression. A safer question-only variant that preserves current `directive` reads reached held-out test `p_at_1=0.514` with no held-out act segment regressions, but it still regressed `inform` in one internal fold, so it remains diagnostic.
- Larger-slice result: on the 2000/2000/2000 samples, `safe_question_act_rhythm_contextual` is selected. Held-out test `p_at_1` improves from `0.335` to `0.491`, `top_3_recall` improves from `0.850` to `0.974`, cross-validation mean gain is `0.092`, minimum fold gain is `0.078`, and cross-fold segment regressions stay at `0`. Held-out `question` top-1 improves from `0.025` to `0.116`, while `directive` top-1 is preserved at `0.052`.
- 5k scale result: on the 5000/5000/5000 samples, `protected_act_rhythm_contextual` is selected. Held-out test `p_at_1` improves from `0.368` to `0.515`, `top_3_recall` improves from `0.863` to `0.977`, cross-validation mean gain is `0.108`, minimum fold gain is `0.099`, and cross-fold segment regressions stay at `0`. Held-out `question` top-1 improves from `0.021` to `0.123`, and held-out `directive` top-1 improves from `0.076` to `0.115`.
- Full-test-depth result: on balanced 6740/6740/6740 samples, `deep_protected_act_rhythm_contextual` is selected. Held-out test `p_at_1` improves from `0.408` to `0.545`, `top_3_recall` improves from `0.881` to `0.982`, cross-validation mean gain is `0.105`, minimum fold gain is `0.089`, and cross-fold segment regressions stay at `0`. Held-out `question` top-1 improves from `0.026` to `0.163`, while `directive` top-1 improves from `0.077` to `0.090`. Conservative question-evidence variants were safe but did not beat the protected rhythm baseline, so they remain diagnostic.
- Limitation: balanced scale is capped by the 6,740-turn DailyDialog test split; the next lever is adding a second human-conversation dataset to test whether protected sequence memory generalizes beyond DailyDialog.

### EmpatheticDialogues Held-Out Samples

- Status: derived samples included in this repository; raw source files are excluded under `data/external/`.
- Files: `data/empatheticdialogues_train_6740_sample.jsonl`, `data/empatheticdialogues_validation_6740_sample.jsonl`, and `data/empatheticdialogues_test_6740_sample.jsonl`.
- Report: `runs/empatheticdialogues_6740_act_ranker_bakeoff.json`.
- Source used for import: `https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz`.
- Purpose: second held-out human-conversation benchmark focused on emotionally grounded dialogue.
- Current use: imports visible prior utterances as context, maps EmpatheticDialogues `context` labels into coarse emotion groups, and infers conversation acts from the target utterance so the same act bake-off can run.
- Current result: on balanced 6740/6740/6740 samples, the selector stays at `heuristic`. Held-out test `p_at_1` remains `0.608`, `top_3_recall` remains `0.981`, and selected segment regressions stay at `0`. Higher-headline contextual variants reach roughly `0.69` held-out `p_at_1`, but they regress sparse `commissive` and other protected act slices, so the strict gate blocks promotion.
- Limitation: inferred act labels are too coarse for empathy-mode work. The next lever is a richer response-mode taxonomy or class-balanced act protection before treating EmpatheticDialogues gains as real.

### ESConv Response-Mode Samples

- Status: derived samples included in this repository; raw source file is excluded under `data/external/`.
- Files: `data/esconv_train_response_modes_sample.jsonl`, `data/esconv_validation_response_modes_sample.jsonl`, and `data/esconv_test_response_modes_sample.jsonl`.
- Report: `runs/esconv_response_mode_bakeoff.json`.
- Source used for import: `https://github.com/thu-coai/Emotional-Support-Conversation`.
- Purpose: held-out benchmark for emotional-support response-mode prediction using source supporter strategy labels.
- Current use: maps ESConv strategies into modes: `ask_followup`, `validate`, `reassure`, `disclose`, `suggest`, `inform`, and `other`; then compares heuristic, hybrid, and learned response-mode branchers.
- Current result: validation selects `response_mode_hybrid_75`; dev `p_at_1` improves from `0.198` to `0.204`, and held-out test `p_at_1` improves from `0.206` to `0.217`.
- Promotion result: not held-out promotable yet. The untouched test split has one response-mode segment regression: `suggest` top-1 slips from `0.291` to `0.284`.
- Weak mode signal: selected behavior still leaves `disclose`, `inform`, and `other` at `0.0` top-1 and top-3 recall; `reassure` has top-3 coverage but no top-1 hits.
- Class-balanced coverage result: balanced-prior and balanced-coverage variants were added, plus `coverage_projection` reporting. They did not displace `response_mode_hybrid_75`, but they exposed useful reachable signal. Learned-only reaches `disclose` at `0.091` top-1 / `0.449` top-3, `inform` at `0.152` top-1, `other` at `0.381` top-1 / `0.691` top-3, and `reassure` at `0.105` top-1. Balanced coverage raises `inform` top-3 to `0.739` and `reassure` top-3 to `0.751`, but remains diagnostic because it disturbs stronger protected modes.
- Protected specialist result: ESConv source metadata is now preserved as features, and one-vs-rest specialists are trained for `other`, `inform`, `disclose`, and `reassure`. Top-1 promotion remains diagnostic, but protected top-3 coverage is useful: `protected_minority_specialist_coverage` keeps held-out test `p_at_1` at `0.206`, raises top-3 recall from `0.526` to `0.532`, adds `224` previously missed prepared hits, and reports `0` held-out segment regressions. The strongest slice is `other`, where top-3 recall moves from `0.0` to `0.723`.
- Validation-calibrated specialist result: per-mode thresholds accepted `reassure` at `-0.25` and rejected `disclose`, `inform`, and `other` because their best dev slice gains lowered aggregate dev top-3. The calibrated coverage variant keeps held-out test `p_at_1=0.206`, raises held-out top-3 from `0.526` to `0.546`, adds `87` previously missed prepared hits, and reports `0` held-out segment regressions. `reassure` held-out top-3 reaches `0.993`, a `+0.305` gain over baseline.
- Dual recommendation result: the report now separates `recommendations.first_speech` from `recommendations.background_readiness`. First speech points to `response_mode_hybrid_75` with dev `p_at_1=0.204` and held-out test `p_at_1=0.217`, but remains not held-out promotable because `suggest` regresses. Background readiness points to `calibrated_minority_specialist_coverage` with dev top-3 `0.542`, held-out test top-3 `0.546`, `heldout_promotable=true`, and `0` held-out segment regressions.
- Probability Pack policy result: `probability_pack_policy` uses `response_mode_hybrid_75` for first speech with `confirm_before_delivery`, uses `calibrated_minority_specialist_coverage` for background readiness with `prewarm_tts`, and sets `confirmation_mode=confirm_first_speech_then_stream_prepared_background`.
- Probability Pack replay result: held-out ESConv packs now report `prepared_hit_rate=0.577`, `exact_prepared_hit_rate=0.546`, `semantic_prepared_hit_rate=0.031`, `first_speech_hit_rate=0.217`, `background_hit_rate=0.360`, `median_latency_ms=90`, and `median_latency_saved_ms=560`.
- Per-mode quality result: prepared-hit rate is `1.000` for `reassure` and `validate`, `0.833` for `ask_followup`, `0.651` for `suggest`, and `0.000` for `disclose`, `inform`, and `other`. Average quality on prepared hits is `0.974`; quality-ready coverage is `0.546`. `reassure` is the clearest background-readiness win with `background_hit_rate=1.000`, `first_speech_hit_rate=0.000`, and `average_quality_score=0.997`.
- Zero-hit recovery result: diagnostic background recovery targets `disclose`, `inform`, and `other` using their best top-3 variants while keeping first speech locked. The recovery candidate lifts held-out prepared-hit rate from `0.577` to `0.843`, preserves first-speech hit rate at `0.217`, and moves `disclose` to `0.858`, `inform` to `0.812`, and `other` to `0.723` prepared-hit rate. It is not promoted because average quality drops from `0.974` to `0.955`, missing the quality floor even though target modes improve.
- Per-mode recovery calibration result: validation-generated recovery subsets promote `recover_other` and reject `recover_inform` plus `recover_inform_other` because their average draft quality misses the validation floor. Held-out test accepts `recover_other`: active prepared-hit rate rises from `0.577` to `0.692`, background-hit rate rises from `0.360` to `0.475`, quality-ready coverage rises from `0.546` to `0.661`, average prepared-draft quality rises from `0.974` to `0.978`, and first-speech hit rate stays locked at `0.217`. The `other` slice reaches `0.723` prepared-hit rate; `disclose` and `inform` remain at `0.000`.
- Quality-aware recovery result: recovery candidate scoring now applies a `0.75` TTS-readiness floor so low-quality semantic-family drafts do not count as prepared speech during promotion. Validation promotes `recover_disclose_inform_other`. Held-out test accepts it: active prepared-hit rate rises from `0.577` to `0.765`, quality-ready coverage from `0.546` to `0.765`, background-hit rate from `0.360` to `0.547`, background-recovery hit rate from `0.000` to `0.219`, and average prepared-draft quality from `0.974` to `1.000`; first-speech hit rate stays locked at `0.217`. Weak-slice exact quality-ready recovery: `disclose=0.449`, `inform=0.739`, and `other=0.723`.
- Stress/dashboard result: `runs/esconv_response_mode_dashboard.html` now renders the response-mode quick-reference view, contrasting raw semantic coverage, the quality-aware gate, and active quality-ready recovery. The held-out promotion gate now compares recovery candidates against `probability_pack_replay_baseline_quality_aware` and a raw prepared-hit floor, while the raw baseline remains available for honest gain reporting. A 3-seed x 5-fold stress pass in `runs/esconv_response_mode_recovery_stress.json` promotes on `15/15` shuffled folds, with mean prepared-hit gain `+0.059`, mean quality-ready gain `+0.101`, minimum quality-ready gain `+0.078`, max quality-ready gain `+0.201`, and minimum raw prepared-hit gain `+0.034`. Selected policies: `recover_disclose_inform` on `8` folds, `recover_inform_buffer_reassure_validate` on `6`, and `recover_other_buffer_reassure_disclose` on `1`. The quality-gap buffer pass keeps each recovery target fixed while adding prewarmed modes that explain the raw-vs-quality floor gap.
- Next lever: test whether the quality-gap buffer rule survives more seeds and a second conversational dataset before treating it as a universal baseline.
- Limitation: deterministic 80/10/10 split over ESConv's single JSON source is useful for iteration; the stricter stress pass now removes raw prepared regressions in the 3x5 run, but broader universality still needs more seeds and at least one second conversational dataset.

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

### ESConv

- Fit: emotional-support conversations with supporter strategy labels.
- Why it matters: it directly labels the response mechanism the backend should prewarm for a voice agent.
- Link: https://github.com/thu-coai/Emotional-Support-Conversation
- Trial question: can probability packs predict whether the next support move should reassure, validate, ask a follow-up, disclose, suggest, inform, or hold a neutral fallback?

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
