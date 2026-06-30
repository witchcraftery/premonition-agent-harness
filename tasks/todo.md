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

## Profile-Aware Benchmark Visualization

- [x] Add event profiles for carrier exception, warehouse lock, inventory backorder, fraud review lock, policy update, and payment gateway update.
- [x] Add guarded environment branch profile selection with negative cue handling.
- [x] Add guidance comparison reporting for improved and regressed turns.
- [x] Generate a static benchmark dashboard page from the cross-fold JSON report.
- [x] Verify the dashboard visually in a browser and document the quick-reference workflow.

Verification:

- `python3 -m pytest -v`: 57 passed.
- `foresight-replay --config experiments/queueahead_challenge_loop.json --iterations 3 --loop-report runs/queueahead_challenge_loop.json --guidance-markdown runs/queueahead_challenge_guidance.md`: completed.
- `foresight-replay --train-config experiments/queueahead_challenge_train.json --test-config experiments/queueahead_challenge_test.json --iterations 3 --benchmark-report runs/queueahead_split_benchmark.json`: completed.
- `foresight-replay --fold-config experiments/queueahead_enriched_folds.json --folds 5 --iterations 3 --benchmark-report runs/queueahead_enriched_cross_benchmark.json --dashboard-report runs/queueahead_enriched_dashboard.html`: completed.
- Browser verification loaded `runs/queueahead_enriched_dashboard.html` and captured desktop/mobile screenshots at `runs/queueahead_enriched_dashboard_desktop.png` and `runs/queueahead_enriched_dashboard_mobile.png`.
- Current profile-aware 5-fold report: held-out `p_at_1` is `0.633 -> 0.633`; environment-event `p_at_1` is `0.483 -> 0.483`; user-event `p_at_1` is `0.810 -> 0.810`; `carrier_exception_hold` is `1.0`; guidance delta is `0` improved and `0` regressed turns.
- Next focus areas are now explicit: `payment_gateway_update`, `policy_update`, and `fraud_review_lock`.

## Targeted Support Baseline Expansion

- [x] Add regression tests for payment gateway, policy update, and fraud review profile branches.
- [x] Expand profile-aware branch selection beyond shipment-status events.
- [x] Require cross-fold weak-profile improvement without regressions.
- [x] Regenerate benchmark JSON, dashboard HTML, and visual proof screenshots.
- [x] Document the new baseline and next pivot decision.

Verification:

- `python3 -m pytest -v`: 62 passed.
- `foresight-replay --config experiments/queueahead_challenge_loop.json --iterations 3 --loop-report runs/queueahead_challenge_loop.json --guidance-markdown runs/queueahead_challenge_guidance.md`: completed.
- `foresight-replay --train-config experiments/queueahead_challenge_train.json --test-config experiments/queueahead_challenge_test.json --iterations 3 --benchmark-report runs/queueahead_split_benchmark.json`: completed.
- `foresight-replay --fold-config experiments/queueahead_enriched_folds.json --folds 5 --iterations 3 --benchmark-report runs/queueahead_enriched_cross_benchmark.json --dashboard-report runs/queueahead_enriched_dashboard.html`: completed.
- Browser verification loaded `runs/queueahead_enriched_dashboard.html` and captured refreshed desktop/mobile screenshots at `runs/queueahead_enriched_dashboard_desktop.png` and `runs/queueahead_enriched_dashboard_mobile.png`.
- Targeted support 5-fold report: held-out `p_at_1` is `0.967 -> 0.967`; environment-event `p_at_1` is `1.000 -> 1.000`; user-event `p_at_1` is `0.950 -> 0.950`; promotion rate is `0.6`; guidance delta is `0` improved and `0` regressed turns.
- Solved profiles in this synthetic set: `payment_gateway_update`, `policy_update`, `fraud_review_lock`, `carrier_exception_hold`, `inventory_backorder`, `refund_request`, and `address_change` all report `1.0` held-out profile `p_at_1`.
- Next recommended move: use this as the support calibration baseline and pivot to a harder dataset or domain-shift benchmark rather than continuing to polish this small synthetic set.

## Human Conversation Probability Pack

- [x] Add a DailyDialog-style fixture for ordinary human next-turn probability.
- [x] Define a model-agnostic Probability Pack shape for voice-agent readiness.
- [x] Add an adapter that turns labeled dialogue turns into replayable probability turns.
- [x] Add a first benchmark report for conversational act, emotion, and speakable draft readiness.
- [x] Document the dataset path and next loop for conversational swarm-mind refinement.

Verification:

- `python3 -m pytest tests/test_conversation_probability.py -v`: 4 passed.
- `foresight-replay --conversation-input data/human_conversation_sample.jsonl --iterations 3 --conversation-report runs/human_conversation_probability_loop.json`: completed.
- First tiny human-conversation loop improved `p_at_1` from `0.75` to `1.0` by iteration 2, held `top_3_recall=1.0`, held `tts_readiness_rate=1.0`, and learned filtered act guidance without transcript boilerplate tokens.

## Real DailyDialog Import

- [x] Add a DailyDialog split importer for `dialogues.txt`, `dialogues_act.txt`, and `dialogues_emotion.txt`.
- [x] Add CLI export support for bounded DailyDialog-derived JSONL samples.
- [x] Download the real DailyDialog train split into ignored external data storage.
- [x] Export and commit a 500-turn DailyDialog train sample.
- [x] Run the human-conversation probability loop on the real sample.
- [x] Add a no-regression promotion gate for conversational guidance.
- [x] Document the result and next training-loop improvement.

Verification:

- `python3 -m pytest tests/test_conversation_probability.py -v`: 7 passed.
- `foresight-replay --dailydialog-dir data/external/dailydialog/train --conversation-output data/dailydialog_train_sample.jsonl --conversation-limit 500`: exported 500 of 76052 available train turns.
- `foresight-replay --conversation-input data/dailydialog_train_sample.jsonl --iterations 3 --conversation-report runs/dailydialog_train_probability_loop.json`: completed.
- Real DailyDialog 500-turn sample: `p_at_1=0.472`, `top_3_recall=0.920`, `tts_readiness_rate=1.0`.
- Candidate guidance was rejected in every iteration because it would reduce `p_at_1` to `0.412`; the no-regression gate correctly preserved the stronger baseline.

## Held-Out DailyDialog Efficacy Loop

- [x] Add a true train/dev/test conversational probability loop.
- [x] Promote conversational guidance only when validation metrics do not regress.
- [x] Report final efficacy on untouched test turns.
- [x] Add conversational segment analytics by act, emotion, and speaker.
- [x] Export bounded DailyDialog train, validation, and test samples.
- [x] Run the held-out loop and document the result.

Verification:

- `python3 -m pytest tests/test_conversation_probability.py -v`: 11 passed.
- `foresight-replay --dailydialog-dir data/external/dailydialog/validation --conversation-output data/dailydialog_validation_sample.jsonl --conversation-limit 500`: exported 500 of 7069 available validation turns.
- `foresight-replay --dailydialog-dir data/external/dailydialog/test --conversation-output data/dailydialog_test_sample.jsonl --conversation-limit 500`: exported 500 of 6740 available test turns.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --iterations 3 --conversation-report runs/dailydialog_heldout_probability_loop.json`: completed.
- Validation improved from `p_at_1=0.324` to `0.382` and from `top_3_recall=0.856` to `0.868`.
- Untouched test moved only from `p_at_1=0.368` to `0.370`; `top_3_recall` slipped from `0.858` to `0.856`.
- Test guidance deltas were nearly even: `50` improved turns and `49` regressed turns. `question` improved from `p_at_1=0.047` to `0.273`, while `commissive` regressed from `0.507` to `0.217`.

## Act-Specific Conversation Refinement

- [x] Add act-specific guidance learning that favors cues discriminative to one expected act.
- [x] Add a segment-aware validation gate so no meaningful act segment regresses during promotion.
- [x] Report validation segment regressions in each iteration.
- [x] Rerun the DailyDialog train/dev/test loop and compare efficacy.
- [x] Document whether the refined loop improves held-out accuracy or mainly prevents brittle promotion.

Verification:

- `python3 -m pytest tests/test_conversation_probability.py -v`: 13 passed.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --iterations 3 --conversation-report runs/dailydialog_heldout_probability_loop.json`: completed.
- Candidate guidance improved validation `p_at_1` from `0.324` to `0.402`, but reduced validation `top_3_recall` from `0.856` to `0.844`.
- Segment-aware promotion rejected the candidate in all three iterations because validation `commissive` regressed from `0.213` to `0.066`, `directive` regressed from `0.147` to `0.010`, and `question` regressed from `0.040` to `0.008`.
- Held-out test stayed at the baseline after gating: `p_at_1=0.368`, `top_3_recall=0.858`, and `0` improved / `0` regressed test turns.

## Learned Conversation Act-Ranker Bake-Off

- [x] Add a transparent learned act-ranker for ordinary conversation turns.
- [x] Blend learned scores with the current heuristic brancher at multiple weights.
- [x] Select the best variant on validation only.
- [x] Score the selected variant on untouched test data.
- [x] Report segment regressions and guidance deltas for the selected variant.
- [x] Document whether the learned brancher discovers larger held-out improvements.

Verification:

- `python3 -m pytest tests/test_conversation_probability.py -v`: 16 passed.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --conversation-bakeoff-report runs/dailydialog_act_ranker_bakeoff.json`: completed.
- Validation selected the current heuristic brancher: dev `p_at_1=0.324`, test `p_at_1=0.368`, with no segment regressions.
- Learned-only overfit the train sample: train `p_at_1=0.656`, validation `p_at_1=0.260`, test `p_at_1=0.274`; validation `inform` regressed from `0.608` to `0.160`.
- The best hybrid did not beat baseline: `hybrid_25` had validation `p_at_1=0.254` and test `p_at_1=0.324`, with validation regressions in `commissive`, `directive`, and `inform`.
- Diagnostic runs with larger local train slices improved learned-only somewhat, but still selected the heuristic baseline through 20k train turns; this suggests the next larger-gain lever is richer contextual branch generation, not bag-of-features act ranking.

## Contextual Conversation Brancher

- [x] Preserve observed dialogue-act history in DailyDialog-derived turns.
- [x] Add a transition/contextual brancher that learns next-act probabilities from prior observed acts.
- [x] Add contextual variants to the learned-ranker bake-off.
- [x] Regenerate DailyDialog samples with act-history fields.
- [x] Run the bake-off and measure whether contextual structure improves held-out test.
- [x] Document the result and next lever.

Verification:

- `python3 -m pytest tests/test_conversation_probability.py -v`: 18 passed.
- `python3 -m pytest`: 80 passed.
- `foresight-replay --dailydialog-dir data/external/dailydialog/train --conversation-output data/dailydialog_train_sample.jsonl --conversation-limit 500`: exported 500 of 76052 available train turns with observed act history.
- `foresight-replay --dailydialog-dir data/external/dailydialog/validation --conversation-output data/dailydialog_validation_sample.jsonl --conversation-limit 500`: exported 500 of 7069 available validation turns with observed act history.
- `foresight-replay --dailydialog-dir data/external/dailydialog/test --conversation-output data/dailydialog_test_sample.jsonl --conversation-limit 500`: exported 500 of 6740 available test turns with observed act history.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --iterations 3 --conversation-report runs/dailydialog_heldout_probability_loop.json`: completed; keyword guidance remained blocked by segment regressions.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --conversation-bakeoff-report runs/dailydialog_act_ranker_bakeoff.json`: completed.
- Selected variant: `contextual_inform_overlay`; validation `p_at_1` improved from `0.324` to `0.416`, held-out test `p_at_1` improved from `0.368` to `0.448`, held-out test `top_3_recall` improved from `0.858` to `0.876`, with `40` improved and `0` regressed test turns.
- Diagnostic full contextual transition was stronger but unsafe: held-out test `p_at_1=0.534` and `top_3_recall=0.970`, but `directive` and `question` top-1 accuracy regressed to `0.0`.
- Next lever: protect low-frequency acts inside the contextual brancher so more of the raw transition gain can be promoted safely.

## Guarded Contextual Brancher

- [x] Add a low-frequency act guard for contextual transition scoring.
- [x] Prove the guard preserves heuristic `directive` and `question` branches when transition confidence would erase them.
- [x] Add guarded contextual variants to the bake-off.
- [x] Rerun the DailyDialog bake-off and compare against `contextual_inform_overlay`.
- [x] Document whether guarded context captures more raw transition gain without segment regressions.

Verification:

- `python3 -m pytest tests/test_conversation_probability.py -v`: 19 passed.
- `python3 -m pytest`: 81 passed.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --conversation-bakeoff-report runs/dailydialog_act_ranker_bakeoff.json`: completed.
- Selected variant: `guarded_contextual_transition`; validation `p_at_1` improved from `0.324` to `0.464`, held-out test `p_at_1` improved from `0.368` to `0.502`, and held-out test `top_3_recall` improved from `0.858` to `0.970`.
- The guarded variant improved beyond `contextual_inform_overlay`, which had validation `p_at_1=0.416`, held-out test `p_at_1=0.448`, and held-out test `top_3_recall=0.876`.
- No act segment regressed on validation or test; full contextual transition remains stronger but unsafe because it regresses `directive` and `question` top-1 accuracy to `0.0`.
- Held-out guidance delta: `73` improved turns and `6` regressed turns. Next lever: make protected `directive` and `question` modes improve under context instead of only avoiding collapse.

## Act-Rhythm Context Specialist

- [x] Add an act-history specialist that learns next-act probabilities from longer observed dialogue rhythms.
- [x] Prove the specialist can promote a `directive` or `question` pattern when the one-step transition model would flatten it.
- [x] Add act-rhythm contextual variants to the bake-off.
- [x] Add an overfit-aware selector so high train/dev-gap variants remain diagnostic.
- [x] Rerun the DailyDialog bake-off and measure whether protected acts improve without segment regressions.
- [x] Document the result and the next refinement lever.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py -v` failed because `train_conversation_history_ranker` did not exist.
- Selector red check: `python3 -m pytest tests/test_conversation_probability.py -v` failed because `select_conversation_bakeoff_variant` still chose a large train/dev-gap variant.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 21 passed.
- `python3 -m pytest`: 83 passed.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --conversation-bakeoff-report runs/dailydialog_act_ranker_bakeoff.json`: completed.
- Selected variant: `act_rhythm_contextual_strict`; validation `p_at_1` improved from `0.324` to `0.476`, held-out test `p_at_1` improved from `0.368` to `0.508`, and held-out test `top_3_recall` improved from `0.858` to `0.970`.
- The strict act-rhythm variant improved beyond `guarded_contextual_transition`, which had validation `p_at_1=0.464` and held-out test `p_at_1=0.502`.
- No act segment regressed on validation or test. Held-out guidance delta: `76` improved turns and `6` regressed turns.
- Diagnostic loose act-rhythm variant improved validation `question` top-1 from `0.040` to `0.136`, but its train/dev gap was too wide to promote; next lever is cross-validated protected-act specialization.

## Cross-Validated Protected-Act Specialists

- [x] Add internal cross-validation diagnostics for conversation bake-off variants.
- [x] Require selector support for cross-validated stability before promoting specialist variants.
- [x] Report cross-validation mean/min gains and segment regressions for protected-act specialists.
- [x] Rerun the DailyDialog bake-off and compare selected behavior against `act_rhythm_contextual_strict`.
- [x] Document whether cross-validation improves reliability, changes selected behavior, or mainly adds transparency.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py -v` failed during collection because `cross_validate_conversation_variant` did not exist.
- Selector red check: `python3 -m pytest tests/test_conversation_probability.py::test_conversation_act_ranker_bakeoff_selects_on_dev_and_scores_test -v` failed because the selected variant did not report cross-validation evidence.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 23 passed.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --conversation-bakeoff-report runs/dailydialog_act_ranker_bakeoff.json`: completed.
- Selected variant: `guarded_contextual_transition`; validation `p_at_1` improved from `0.324` to `0.464`, held-out test `p_at_1` improved from `0.368` to `0.502`, and held-out test `top_3_recall` improved from `0.858` to `0.970`.
- Selected cross-validation: mean `p_at_1` gain `0.096`, minimum fold gain `0.080`, mean `top_3_recall` gain `0.076`, minimum fold gain `0.040`, and `0` segment regressions.
- Diagnostic strict act-rhythm variant: held-out test `p_at_1=0.508`, cross-validation mean `p_at_1` gain `0.112`, and minimum fold gain `0.100`, but `1` internal segment regression, so it was not promoted.
- Result: cross-validation improved reliability and transparency, and changed selected behavior back to the safer guarded contextual variant.

## Per-Act Specialist Expansion

- [x] Add targeted `directive` and `question` specialist variants to the conversation bake-off.
- [x] Prove specialist variants can promote protected acts without bypassing the cross-validation stability gate.
- [x] Rerun the DailyDialog bake-off and compare promoted behavior against `guarded_contextual_transition`.
- [x] Report whether the lever improves held-out accuracy, segment performance, or only adds diagnostic transparency.
- [x] Document the next lever based on the benchmark result.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py -v` failed because `history_overlay_acts` and targeted specialist variants did not exist.
- Selector red check: `python3 -m pytest tests/test_conversation_probability.py::test_bakeoff_selection_allows_small_gap_for_cross_validated_specialist -v` failed because the train/dev cutoff blocked a stable per-act specialist.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 28 passed.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --conversation-bakeoff-report runs/dailydialog_act_ranker_bakeoff.json`: completed.
- Selected variant: `directive_act_rhythm_contextual`; validation `p_at_1` improved from `0.324` to `0.466`, held-out test `p_at_1` improved from `0.368` to `0.504`, and held-out test `top_3_recall` improved from `0.858` to `0.970`.
- Selected cross-validation: mean `p_at_1` gain `0.104`, minimum fold gain `0.090`, mean `top_3_recall` gain `0.076`, minimum fold gain `0.040`, and `0` segment regressions.
- Segment signal: held-out `directive` top-1 improved from `0.053` to `0.116`, while held-out `question` top-1 stayed at `0.047` and question top-3 recall rose to `1.0`.
- Diagnostic higher-score variants remain blocked: `protected_act_rhythm_contextual` reached test `p_at_1=0.514` and `question_act_rhythm_contextual` reached `0.512`, but each had `1` internal segment regression.
- Result: the lever produced a small stable promoted gain and identified the next focus as question-specialist stability.

## Safe Question Specialist Stabilization

- [x] Add a question-rhythm specialist that preserves current `directive` reads.
- [x] Prove the specialist can boost question rhythm without reducing directive accuracy.
- [x] Rerun the DailyDialog bake-off and compare against `directive_act_rhythm_contextual`.
- [x] Report whether the safe question specialist promotes, stays diagnostic, or exposes the next weak segment.
- [x] Document the next lever based on benchmark evidence.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py -v` failed because `history_preserved_acts` and `safe_question_act_rhythm_contextual` did not exist.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 29 passed.
- `foresight-replay --conversation-train-input data/dailydialog_train_sample.jsonl --conversation-dev-input data/dailydialog_validation_sample.jsonl --conversation-test-input data/dailydialog_test_sample.jsonl --conversation-bakeoff-report runs/dailydialog_act_ranker_bakeoff.json`: completed.
- Selected variant stayed `directive_act_rhythm_contextual`; validation `p_at_1=0.466`, held-out test `p_at_1=0.504`, held-out top-3 recall `0.970`, and cross-validation segment regressions `0`.
- Safe question specialist result: held-out test `p_at_1=0.514`, no held-out act segment regressions, and no directive regression, but `1` internal cross-validation `inform` segment regression.
- Margin/preservation sweep found stable question configurations only at `p_at_1=0.502`, below the current promoted `0.504`, so the safe question specialist remains diagnostic.
- Result: the lever removed the obvious directive failure but exposed `inform` stability as the blocker. Next focus is question-specific evidence/features, not another guard.

## Larger-Slice Question Breakthrough

- [x] Probe larger DailyDialog train/dev/test slices to see whether question specialists stabilize with more examples.
- [x] Add selector support for preferring preserved-act specialists when validation scores are effectively tied.
- [x] Generate and benchmark a larger committed DailyDialog sample.
- [x] Report whether the question specialist becomes promotable without held-out segment regressions.
- [x] Document the next benchmark lever.

Verification:

- Probe: temporary 2000/2000/2000 DailyDialog bake-off selected a question specialist and showed that the raw question variant was cross-fold stable but had a tiny held-out `directive` regression.
- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_bakeoff_selection_prefers_preserved_specialist_inside_dev_tie -v` failed because selector sorting chose the raw question specialist over the preserved safe specialist.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 30 passed.
- Exported committed 2k samples: `data/dailydialog_train_2k_sample.jsonl`, `data/dailydialog_validation_2k_sample.jsonl`, and `data/dailydialog_test_2k_sample.jsonl`.
- `foresight-replay --conversation-train-input data/dailydialog_train_2k_sample.jsonl --conversation-dev-input data/dailydialog_validation_2k_sample.jsonl --conversation-test-input data/dailydialog_test_2k_sample.jsonl --conversation-bakeoff-report runs/dailydialog_2k_act_ranker_bakeoff.json`: completed.
- Selected 2k variant: `safe_question_act_rhythm_contextual`; validation `p_at_1=0.488`, held-out test `p_at_1=0.491`, held-out top-3 recall `0.974`, cross-validation mean gain `0.092`, minimum fold gain `0.078`, and `0` cross-fold segment regressions.
- Segment result: held-out `question` top-1 improved from `0.025` to `0.116`, held-out `directive` top-1 stayed at `0.052`, and both `question` and `directive` top-3 recall reached `1.0`.
- Result: breakthrough achieved through more examples plus a stability-biased tie selector. Next lever is scaling beyond 2k and adding question-specific evidence beyond act history.

## 5k DailyDialog Scale-Up

- [x] Export bounded 5k DailyDialog train/dev/test samples.
- [x] Run the same act-ranker bake-off on 5k/5k/5k.
- [x] Compare selected variant, held-out score, cross-fold stability, and act-segment regressions against the 2k result.
- [x] Document whether scaling strengthens question-specialist promotion or exposes a new failure mode.
- [x] Commit and push reproducible 5k artifacts if the run is useful.

Verification:

- Initial 5k run exposed an over-conservative selector tie rule: `safe_question_act_rhythm_contextual` displaced a stronger stable protected-act specialist.
- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_bakeoff_selection_keeps_stronger_stable_protected_specialist -v` failed because the preserved-question tie rule overrode the stronger protected-act variant.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 31 passed.
- Exported committed 5k samples: `data/dailydialog_train_5k_sample.jsonl`, `data/dailydialog_validation_5k_sample.jsonl`, and `data/dailydialog_test_5k_sample.jsonl`.
- `foresight-replay --conversation-train-input data/dailydialog_train_5k_sample.jsonl --conversation-dev-input data/dailydialog_validation_5k_sample.jsonl --conversation-test-input data/dailydialog_test_5k_sample.jsonl --conversation-bakeoff-report runs/dailydialog_5k_act_ranker_bakeoff.json`: completed.
- Selected 5k variant: `protected_act_rhythm_contextual`; held-out test `p_at_1` improved from `0.368` to `0.515`, held-out top-3 recall improved from `0.863` to `0.977`, cross-validation mean gain `0.108`, minimum fold gain `0.099`, and `0` cross-fold segment regressions.
- Segment result: held-out `question` top-1 improved from `0.021` to `0.123`, and held-out `directive` top-1 improved from `0.076` to `0.115`, with no held-out act-segment regressions.
- Result: scaling strengthened the specialist from question-only at 2k to protected directive+question at 5k. Next lever is scaling as far as validation/test split size allows, then adding question-specific evidence beyond act history.

## Full-Test-Depth DailyDialog Scale-Up

- [x] Export balanced 6,740-turn DailyDialog train/dev/test samples.
- [x] Run the same act-ranker bake-off at full test-split depth.
- [x] Compare selected variant, held-out score, cross-fold stability, and act-segment regressions against the 5k result.
- [x] Document whether the protected specialist keeps strengthening or starts to plateau.
- [x] Commit and push reproducible full-depth artifacts if the run is useful.

Verification:

- Exported balanced full-test-depth samples: `data/dailydialog_train_6740_sample.jsonl`, `data/dailydialog_validation_6740_sample.jsonl`, and `data/dailydialog_test_6740_sample.jsonl`.
- `foresight-replay --conversation-train-input data/dailydialog_train_6740_sample.jsonl --conversation-dev-input data/dailydialog_validation_6740_sample.jsonl --conversation-test-input data/dailydialog_test_6740_sample.jsonl --conversation-bakeoff-report runs/dailydialog_6740_act_ranker_bakeoff.json`: completed.
- Selected full-depth variant: `safe_question_act_rhythm_contextual`; held-out test `p_at_1` improved from `0.408` to `0.541`, held-out top-3 recall improved from `0.881` to `0.982`, cross-validation mean gain `0.104`, minimum fold gain `0.083`, and `0` cross-fold segment regressions.
- Segment result: held-out `question` top-1 improved from `0.026` to `0.134`, and held-out `directive` top-1 stayed at `0.077`, with no held-out act-segment regressions.
- Diagnostic note: `protected_act_rhythm_contextual` reached a slightly higher held-out `p_at_1=0.542`, but had tiny `directive` regressions on dev, held-out test, and internal folds, so it was not promoted.
- Result: scaling to full test depth improved the selected safe specialist score, but the broader protected specialist started to plateau against the directive guard. Next lever is adding question-specific evidence beyond act history.

## Question Evidence Lever

- [x] Add a learned question-evidence ranker that uses language cues beyond act-history rhythm.
- [x] Add a safe question-evidence overlay that can promote likely questions while preserving current directive reads.
- [x] Include question-evidence variants in the train/dev/test bake-off and cross-validation path.
- [x] Add protected deep-rhythm variants after the evidence distribution showed raw language cues were weak on DailyDialog.
- [x] Rerun the full-depth 6,740 DailyDialog benchmark and compare selected variant, question segment, directive segment, and regressions.
- [x] Update README/catalog/tracker with whether the lever produced a real held-out improvement.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_question_evidence_ranker_promotes_question_language_beyond_history tests/test_conversation_probability.py::test_question_evidence_overlay_preserves_current_directive_read tests/test_conversation_probability.py::test_bakeoff_variants_include_protected_act_specialists -v` failed because `train_conversation_question_evidence_ranker` and the safe evidence variant did not exist.
- Second red check: `python3 -m pytest tests/test_conversation_probability.py::test_bakeoff_variants_include_protected_act_specialists tests/test_conversation_probability.py::test_variant_fold_scoring_uses_deeper_history_window -v` failed because `deep_protected_act_rhythm_contextual` and variant-specific `history_window_size` support did not exist.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 34 passed.
- `foresight-replay --conversation-train-input data/dailydialog_train_6740_sample.jsonl --conversation-dev-input data/dailydialog_validation_6740_sample.jsonl --conversation-test-input data/dailydialog_test_6740_sample.jsonl --conversation-bakeoff-report runs/dailydialog_6740_act_ranker_bakeoff.json`: completed.
- Selected full-depth variant: `deep_protected_act_rhythm_contextual`; held-out test `p_at_1` improved from `0.408` to `0.545`, held-out top-3 recall improved from `0.881` to `0.982`, cross-validation mean gain `0.105`, minimum fold gain `0.089`, and `0` cross-fold segment regressions.
- Segment result: held-out `question` top-1 improved from `0.026` to `0.163`, held-out `directive` top-1 improved from `0.077` to `0.090`, and no held-out act segment regressed.
- Diagnostic note: question-evidence variants are implemented and safe at conservative margins, but did not beat the protected rhythm baseline on DailyDialog. The real lift came from extending protected dialogue rhythm to an 8-act history window.
- Result: the harness now supports language-evidence overlays and variant-specific history depth. DailyDialog suggests the next dataset should test whether protected sequence memory generalizes to emotional or task-oriented spoken conversations.

## EmpatheticDialogues Generalization Lever

- [x] Add an EmpatheticDialogues importer for CSV/JSONL rows with `conv_id`, `utterance_idx`, `context`, `prompt`, `speaker_idx`, and `utterance`.
- [x] Export bounded train/dev/test conversation samples from the second human-dialogue dataset.
- [x] Run the same conversation bake-off and compare selected variant, held-out score, emotion segments, and act regressions against DailyDialog.
- [x] Document whether deep protected sequence memory generalizes beyond DailyDialog.
- [x] Commit and push reproducible samples, report, and docs if the run is useful.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_load_empatheticdialogues_export_creates_next_turn_examples tests/test_conversation_probability.py::test_load_empatheticdialogues_export_supports_jsonl_rows -v` failed because `load_empatheticdialogues_export` did not exist.
- Label-quality red check: `python3 -m pytest tests/test_conversation_probability.py::test_load_empatheticdialogues_export_detects_common_commitment_phrases -v` failed because common commitment phrases like `I would...` were labeled `inform` instead of `commissive`.
- `python3 -m pytest tests/test_cli.py::test_cli_exports_empatheticdialogues_sample tests/test_conversation_probability.py::test_load_empatheticdialogues_export_detects_common_commitment_phrases tests/test_conversation_probability.py::test_load_empatheticdialogues_export_creates_next_turn_examples tests/test_conversation_probability.py::test_load_empatheticdialogues_export_supports_jsonl_rows -v`: 4 passed.
- Downloaded raw source archive from `https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz` into ignored `data/external/empatheticdialogues/`.
- Exported balanced 6,740-turn samples: `data/empatheticdialogues_train_6740_sample.jsonl`, `data/empatheticdialogues_validation_6740_sample.jsonl`, and `data/empatheticdialogues_test_6740_sample.jsonl`.
- `foresight-replay --conversation-train-input data/empatheticdialogues_train_6740_sample.jsonl --conversation-dev-input data/empatheticdialogues_validation_6740_sample.jsonl --conversation-test-input data/empatheticdialogues_test_6740_sample.jsonl --conversation-bakeoff-report runs/empatheticdialogues_6740_act_ranker_bakeoff.json`: completed.
- Selected EmpatheticDialogues variant: `heuristic`; held-out test `p_at_1` stayed at `0.608`, held-out top-3 recall stayed at `0.981`, and selected dev/test/cross-fold segment regressions stayed at `0`.
- Diagnostic note: contextual variants reached about `0.69` held-out `p_at_1`, but regressed sparse protected act slices, especially `commissive`, so the strict gate correctly blocked promotion.
- Result: deep protected rhythm did not generalize safely to EmpatheticDialogues under coarse inferred act labels. Next lever is richer empathy response-mode labeling or class-balanced protection before promoting emotional conversation gains.

## ESConv Response-Mode Benchmark

- [x] Add response-mode labels to conversation turns without breaking existing act/emotion samples.
- [x] Import ESConv supporter strategy turns as response-mode probability examples.
- [x] Add a response-mode ranker bake-off with validation selection and held-out test reporting.
- [x] Export bounded ESConv train/dev/test samples from the ignored external source file.
- [x] Run the held-out response-mode benchmark and document exact hits, weak modes, and next levers.
- [x] Verify the full suite, commit, and push the reproducible scaffolding and artifacts.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_conversation_turn_round_trips_response_mode_fields tests/test_conversation_probability.py::test_load_esconv_export_creates_response_mode_examples tests/test_conversation_probability.py::test_learned_response_mode_ranker_predicts_repeated_support_mode tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments tests/test_cli.py::test_cli_exports_esconv_sample tests/test_cli.py::test_cli_runs_response_mode_ranker_bakeoff -v` failed because `generate_response_mode_branches` and related response-mode functions did not exist.
- `python3 -m pytest tests/test_conversation_probability.py::test_conversation_turn_round_trips_response_mode_fields tests/test_conversation_probability.py::test_load_esconv_export_creates_response_mode_examples tests/test_conversation_probability.py::test_learned_response_mode_ranker_predicts_repeated_support_mode tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments tests/test_cli.py::test_cli_exports_esconv_sample tests/test_cli.py::test_cli_runs_response_mode_ranker_bakeoff -v`: 6 passed.
- Exported ESConv response-mode samples: `data/esconv_train_response_modes_sample.jsonl` with 6,740 turns, `data/esconv_validation_response_modes_sample.jsonl` with 1,800 turns, and `data/esconv_test_response_modes_sample.jsonl` with 1,748 turns.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Selected validation variant: `response_mode_hybrid_75`. Dev `p_at_1` improved from `0.198` to `0.204`, and dev `top_3_recall` improved from `0.532` to `0.540`.
- Untouched test moved from `p_at_1=0.206` to `0.217` and from `top_3_recall=0.526` to `0.527`, but `heldout_promotable=false` because the `suggest` segment regressed from `0.291` to `0.284`.
- Weak response-mode slices are explicit: `disclose`, `inform`, and `other` remain at `0.0` top-1 and top-3 recall; `reassure` has top-3 coverage but no top-1 hits. Next lever: class-balanced response-mode coverage/protection.
- `python3 -m pytest -v`: 106 passed.
- `git diff --check`: passed.

## Class-Balanced Response-Mode Coverage

- [x] Add tests for minority-mode top-3 coverage that preserve stronger top-1 modes.
- [x] Add a class-balanced response-mode brancher variant to improve coverage for `disclose`, `inform`, `other`, and `reassure`.
- [x] Gate balanced variants through validation selection and held-out segment promotion.
- [x] Rerun ESConv response-mode benchmark and compare coverage, top-1, and weak-mode movement.
- [x] Update README/catalog/tracker with whether balanced coverage produces a promotable result.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_balanced_response_mode_brancher_adds_minority_mode_to_top_three tests/test_conversation_probability.py::test_balanced_response_mode_brancher_preserves_strong_top_mode tests/test_conversation_probability.py::test_response_mode_bakeoff_variants_include_balanced_coverage -v` failed because `generate_response_mode_branches` did not support `coverage_modes` and the bake-off had no balanced variants.
- Balanced-prior red check: `python3 -m pytest tests/test_conversation_probability.py::test_class_balanced_response_mode_ranker_uses_uniform_priors tests/test_conversation_probability.py::test_response_mode_bakeoff_variants_include_balanced_coverage -v` failed because `train_response_mode_ranker` did not support `class_balanced=True` and the bake-off had no balanced-prior variant.
- Coverage projection red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments -v` failed because the response-mode report had no `coverage_projection` section.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 45 passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Selected validation variant remains `response_mode_hybrid_75`; held-out test stays at `p_at_1=0.217`, `top_3_recall=0.527`, and `heldout_promotable=false` because `suggest` still regresses from `0.291` to `0.284`.
- Balanced coverage is diagnostic, not promoted: `balanced_response_mode_50` lifts held-out `inform` top-3 to `0.739` and `reassure` top-3 to `0.751`, but it reduces aggregate top-3 and validation blocks it.
- Coverage projection now shows the reachable weak-mode signal: learned-only reaches `disclose` at `0.091` top-1 / `0.449` top-3, `inform` at `0.152` top-1, `other` at `0.381` top-1 / `0.691` top-3, and `reassure` at `0.105` top-1. Next lever: protected minority-mode promotion with richer features.
- `python3 -m pytest -v`: 110 passed.
- `git diff --check`: passed.

## Protected Minority Response-Mode Specialists

- [x] Preserve ESConv source metadata (`problem_type`, `emotion_type`, and `experience_type`) in conversation turns as response-mode features.
- [x] Add one-vs-rest response-mode specialist scoring for `other`, `inform`, `disclose`, and `reassure`.
- [x] Add protected specialist variants that can promote weak modes only when confidence margins clear validation gates.
- [x] Report specialist eligibility, blocked counts, and protected-mode regressions in the response-mode bake-off.
- [x] Rerun ESConv response-mode benchmark and compare weak-mode top-1/top-3 movement against `response_mode_hybrid_75`.
- [x] Update README/catalog/tracker with the exact result and next lever.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_specialist_score_includes_mode_prior tests/test_conversation_probability.py::test_response_mode_specialist_top_three_preserves_first_branch -v` failed because specialists did not include a calibrated mode prior and the brancher had no top-3 specialist insertion mode.
- `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_specialist_score_includes_mode_prior tests/test_conversation_probability.py::test_response_mode_specialist_top_three_preserves_first_branch tests/test_conversation_probability.py::test_response_mode_specialist_promotes_target_mode_from_metadata tests/test_conversation_probability.py::test_response_mode_specialist_preserves_protected_top_mode tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments -v`: 5 passed.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 49 passed.
- `python3 -m pytest -v`: 114 passed.
- `git diff --check`: passed.
- Re-exported ESConv train/validation/test response-mode samples so each turn includes source metadata.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Result: direct top-1 protected minority promotion remains diagnostic, but top-3 specialist coverage is useful. `protected_minority_specialist_coverage` kept held-out test `p_at_1=0.206`, raised top-3 recall from `0.526` to `0.532`, added `224` previously missed prepared top-3 hits, and had `0` held-out segment regressions.
- Strongest weak-mode movement: `other` top-3 recall improved from `0.0` to `0.723` under `protected_minority_specialist_coverage_low_margin`.
- Current selected variant remains `response_mode_hybrid_75`, but it is still not promotable because held-out `suggest` top-1 slips from `0.291` to `0.284`.
- Next lever: per-mode validation-calibrated specialist thresholds for `disclose` and `inform`, so protected minority coverage improves without lowering dev top-3.

## Validation-Calibrated Minority Specialists

- [x] Add per-mode specialist thresholds so `disclose`, `inform`, `other`, and `reassure` are not forced through one global gate.
- [x] Add validation calibration that accepts a specialist threshold only when it improves that mode's dev top-3 coverage without lowering aggregate dev top-3.
- [x] Report accepted/rejected specialist thresholds, dev mode gain, and dev aggregate gain in the bake-off.
- [x] Add a calibrated protected specialist coverage variant to the ESConv response-mode bake-off.
- [x] Rerun the ESConv benchmark and compare calibrated coverage against protected specialist coverage and `response_mode_hybrid_75`.
- [x] Update README/catalog/tracker with the exact result and next lever.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_specialist_uses_per_mode_thresholds tests/test_conversation_probability.py::test_response_mode_specialist_calibration_rejects_dev_top_three_drop -v` failed because `calibrate_response_mode_specialist_thresholds` did not exist and the brancher had no per-mode specialist threshold support.
- `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_specialist_uses_per_mode_thresholds tests/test_conversation_probability.py::test_response_mode_specialist_calibration_rejects_dev_top_three_drop -v`: 2 passed.
- `python3 -m pytest tests/test_conversation_probability.py::test_load_esconv_export_creates_response_mode_examples tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments tests/test_conversation_probability.py::test_response_mode_specialist_uses_per_mode_thresholds tests/test_conversation_probability.py::test_response_mode_specialist_calibration_rejects_dev_top_three_drop tests/test_conversation_probability.py::test_response_mode_specialist_top_three_preserves_first_branch tests/test_conversation_probability.py::test_response_mode_bakeoff_variants_include_balanced_coverage -v`: 6 passed.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 51 passed.
- `python3 -m pytest -v`: 116 passed.
- `git diff --check`: passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Calibration result: accepted `reassure` at threshold `-0.25`; rejected `disclose`, `inform`, and `other` because their best dev slice gains lowered aggregate dev top-3.
- Held-out result: `calibrated_minority_specialist_coverage` kept `p_at_1=0.206`, improved top-3 recall from `0.526` to `0.546`, added `87` previously missed prepared hits, and had `0` held-out segment regressions.
- Best calibrated slice: `reassure` top-3 recall improved from `0.688` to `0.993`, a `+0.305` gain over baseline.
- Current first-speech selector still chooses `response_mode_hybrid_75`, but that path remains not promotable because held-out `suggest` top-1 slips from `0.291` to `0.284`.
- Next lever: split reporting into first-speech accuracy recommendation and background-readiness recommendation, because the best TTS preparedness variant is no longer the same as the rank-1 selector.

## Dual Recommendation Reporting

- [x] Add a response-mode recommendation layer that reports first-speech and background-readiness winners separately.
- [x] Keep first-speech selection focused on validation top-1 with segment-safety checks.
- [x] Select background readiness by validation top-3, requiring no dev segment regressions and no drop below heuristic first-speech accuracy.
- [x] Report held-out promotability separately for each recommendation.
- [x] Rerun the ESConv benchmark and confirm background readiness selects the calibrated specialist path.
- [x] Update README/catalog/tracker with the exact result and next lever.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_recommendations_split_speech_and_readiness_winners -v` failed because `response_mode_recommendations` did not exist.
- `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_recommendations_split_speech_and_readiness_winners tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments -v`: 2 passed.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 52 passed.
- `python3 -m pytest -v`: 117 passed.
- `git diff --check`: passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- First-speech recommendation: `response_mode_hybrid_75`, dev `p_at_1=0.204`, held-out test `p_at_1=0.217`, `heldout_promotable=false`, and `1` held-out segment regression.
- Background-readiness recommendation: `calibrated_minority_specialist_coverage`, dev top-3 `0.542`, held-out test top-3 `0.546`, `heldout_promotable=true`, and `0` held-out segment regressions.
- Next lever: wire the background-readiness recommendation into Probability Pack preparation policy while keeping first speech governed by the stricter rank-1 selector.

## Response-Mode Probability Pack Prep Policy

- [x] Add a response-mode Probability Pack policy derived from first-speech and background-readiness recommendations.
- [x] Mark first-speech delivery as confirmation-gated when the first-speech recommendation is not held-out promotable.
- [x] Mark background readiness as TTS-prewarm eligible when the readiness recommendation is held-out promotable.
- [x] Add a response-mode Probability Pack builder that keeps the first branch from the first-speech selector while preparing background-readiness branches.
- [x] Include the policy in the ESConv response-mode bake-off report.
- [x] Update README/catalog/tracker with the exact policy result and next lever.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_probability_pack_policy_uses_recommendation_promotability tests/test_conversation_probability.py::test_response_mode_probability_pack_prepares_background_readiness_branches -v` failed because `build_response_mode_probability_pack` and `response_mode_probability_pack_policy` did not exist.
- `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_probability_pack_policy_uses_recommendation_promotability tests/test_conversation_probability.py::test_response_mode_probability_pack_prepares_background_readiness_branches tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments -v`: 3 passed.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 54 passed.
- `python3 -m pytest -v`: 119 passed.
- `git diff --check`: passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Policy result: first-speech variant `response_mode_hybrid_75`, first-speech delivery `confirm_before_delivery`, background-readiness variant `calibrated_minority_specialist_coverage`, background preparation `prewarm_tts`, confirmation mode `confirm_first_speech_then_stream_prepared_background`.
- Next lever: replay response-mode Probability Packs turn-by-turn and score whether prepared background drafts would reduce TTS latency on exact/semantic hits.

## Response-Mode Probability Pack Replay Scoring

- [x] Add response-mode match grading for exact and semantic-equivalent prepared drafts.
- [x] Add a Probability Pack replay scorer with prepared hit rate, exact hit rate, semantic hit rate, background hit rate, first-speech hit rate, and estimated latency saved.
- [x] Generate response-mode packs from the first-speech and background-readiness recommendations inside the ESConv bake-off.
- [x] Report pack replay metrics in the ESConv response-mode bake-off artifact.
- [x] Rerun the ESConv benchmark and document whether warmed background drafts produce measurable latency readiness.
- [x] Update README/catalog/tracker with the exact result and next lever.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_match_grade_counts_exact_and_semantic_hits tests/test_conversation_probability.py::test_response_mode_probability_pack_replay_scores_tts_readiness_hits -v` failed because `response_mode_match_grade` did not exist.
- `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments tests/test_conversation_probability.py::test_response_mode_match_grade_counts_exact_and_semantic_hits tests/test_conversation_probability.py::test_response_mode_probability_pack_replay_scores_tts_readiness_hits -v`: 3 passed.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 56 passed.
- `python3 -m pytest -v`: 121 passed.
- `git diff --check`: passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Probability Pack replay result on the untouched ESConv test split: `prepared_hit_rate=0.577`, `exact_prepared_hit_rate=0.546`, `semantic_prepared_hit_rate=0.031`, `first_speech_hit_rate=0.217`, `background_hit_rate=0.360`, `median_latency_ms=90`, and `median_latency_saved_ms=560`.
- Next lever: add per-mode prepared-hit analytics and semantic response-quality scoring so the background swarm can strengthen weak response mechanisms without overclaiming first-speech readiness.

## Per-Mode Pack Analytics And Draft Quality

- [x] Add per-response-mode Probability Pack replay analytics for prepared hits, exact hits, semantic hits, first-speech hits, background hits, and latency saved.
- [x] Add deterministic semantic draft-quality scoring for response-mode prepared drafts.
- [x] Include per-mode analytics and quality summaries in the ESConv response-mode bake-off artifact.
- [x] Rerun the ESConv benchmark and document the strongest/weakest prepared response modes.
- [x] Update README/catalog/tracker with exact held-out results and the next lever.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_draft_quality_scores_mode_specific_prepared_speech tests/test_conversation_probability.py::test_response_mode_probability_pack_replay_reports_per_mode_quality -v` failed because `response_mode_draft_quality_score` did not exist.
- `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_draft_quality_scores_mode_specific_prepared_speech tests/test_conversation_probability.py::test_response_mode_probability_pack_replay_reports_per_mode_quality -v`: 2 passed.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 58 passed.
- `python3 -m pytest -v`: 123 passed.
- `git diff --check`: passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Overall replay result: `prepared_hit_rate=0.577`, `average_quality_score=0.974`, `quality_ready_rate=0.546`, `background_hit_rate=0.360`, and `median_latency_saved_ms=560`.
- Strong prepared modes: `reassure` and `validate` at `1.000` prepared-hit rate, `ask_followup` at `0.833`, and `suggest` at `0.651`.
- Weak prepared modes: `disclose`, `inform`, and `other` remain at `0.000` prepared-hit rate.
- Next lever: targeted background specialists for `disclose`, `inform`, and `other`, promoted only when they improve per-mode prepared hits without lowering aggregate quality or protected first-speech behavior.

## Zero-Hit Background Specialist Recovery

- [x] Add a protected background recovery policy for `disclose`, `inform`, and `other`.
- [x] Require recovery to improve zero-hit prepared modes while preserving first-speech selector behavior.
- [x] Report recovered zero-hit modes and any quality/protection regressions in the ESConv bake-off artifact.
- [x] Rerun the ESConv benchmark and document whether zero-hit response modes become prepared.
- [x] Update README/catalog/tracker with exact held-out results and the next lever.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_background_recovery_policy_targets_zero_hit_modes_only -v` failed because `response_mode_background_recovery_policy` did not exist.
- Red routing check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_background_recovery_policy_uses_best_top_3_variants tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments -v` failed because the policy did not accept `coverage_projection` and the report had no baseline/recovery fields.
- Red metric check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_probability_pack_replay_counts_background_recovery_hits -v` failed because background recovery hits were not counted as background hits.
- Red promotion-gate check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_background_recovery_evaluation_blocks_quality_drop -v` failed because `response_mode_background_recovery_evaluation` did not exist.
- `python3 -m pytest tests/test_conversation_probability.py -v`: 62 passed.
- `python3 -m pytest -v`: 127 passed.
- `git diff --check`: passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Baseline active pack replay stays at `prepared_hit_rate=0.577`, `average_quality_score=0.974`, `quality_ready_rate=0.546`, and `first_speech_hit_rate=0.217`.
- Diagnostic recovery candidate reaches `prepared_hit_rate=0.843`, `quality_ready_rate=0.765`, and preserves `first_speech_hit_rate=0.217`.
- Recovered target-mode prepared-hit rates: `disclose=0.858`, `inform=0.812`, and `other=0.723`.
- Recovery is not promoted because average quality drops from `0.974` to `0.955`, missing the quality floor.
- Next lever: per-mode recovery calibration, especially improving `disclose` draft quality from `0.768`, so the broader recovery pack can clear the quality gate.

## Per-Mode Recovery Calibration

- [x] Add recovery policy subset candidates so weak modes can be promoted selectively instead of as one all-or-nothing bundle.
- [x] Select recovery policy on dev data using target-mode improvement, first-speech preservation, and the baseline quality floor.
- [x] Apply the dev-selected recovery policy to held-out test replay and report candidate calibration transparency.
- [x] Rerun the ESConv response-mode benchmark and document whether any recovery subset is promoted.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_background_recovery_policy_candidates_include_mode_subsets tests/test_conversation_probability.py::test_select_response_mode_background_recovery_candidate_prefers_quality_safe_subset -v` failed because `response_mode_background_recovery_policy_candidates` did not exist.
- Integration red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments -v` failed because the report had no `background_recovery_calibration` field.
- Split-floor red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_recovery_policy_for_replay_uses_local_baseline_quality_floor -v` failed because `response_mode_recovery_policy_for_replay` did not exist.
- Focused green check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_recovery_policy_for_replay_uses_local_baseline_quality_floor tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments tests/test_conversation_probability.py::test_select_response_mode_background_recovery_candidate_prefers_quality_safe_subset -v`: 3 passed.
- Full pre-benchmark suite: `python3 -m pytest -v`: 129 passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Validation calibration promoted `recover_other`; `recover_inform` and `recover_inform_other` stayed diagnostic because their average draft quality missed the validation quality floor.
- Held-out test accepted `recover_other`: active `prepared_hit_rate` rose from `0.577` to `0.692`, `background_hit_rate` from `0.360` to `0.475`, `quality_ready_rate` from `0.546` to `0.661`, and `average_quality_score` from `0.974` to `0.978`; `first_speech_hit_rate` stayed locked at `0.217`.
- Recovered target mode: `other` prepared-hit rate is now `0.723`; `disclose` and `inform` remain at `0.000`.
- Final suite: `python3 -m pytest -v`: 130 passed.
- `git diff --check`: passed.
- Next lever: draft-quality-aware recovery generation for `disclose` and `inform`.

## Quality-Aware Recovery Promotion

- [x] Add recovery scoring that ignores low-quality semantic-family drafts below the TTS readiness floor.
- [x] Use quality-aware scoring for validation recovery candidates and held-out recovery replay while preserving the existing raw replay metrics.
- [x] Rerun ESConv and document whether `inform` or `inform+other` can now promote without `disclose` quality spillover.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_probability_pack_replay_can_ignore_low_quality_semantic_hits -v` failed because `score_response_mode_probability_pack_replay` did not accept `min_quality_score`.
- Integration red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments -v` failed because `background_recovery_calibration` had no `min_quality_score` field.
- Focused green check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_probability_pack_replay_can_ignore_low_quality_semantic_hits tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments -v`: 2 passed.
- Conversation suite: `python3 -m pytest tests/test_conversation_probability.py -v`: 66 passed.
- Full pre-benchmark suite: `python3 -m pytest -v`: 131 passed.
- `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json`: completed.
- Calibration uses `min_quality_score=0.75` and promotes `recover_disclose_inform_other`; lower-quality semantic-family spillover no longer counts as prepared speech in recovery promotion.
- Held-out test accepted the promoted recovery policy: active `prepared_hit_rate` rose from `0.577` to `0.765`, `quality_ready_rate` from `0.546` to `0.765`, `background_hit_rate` from `0.360` to `0.547`, `background_recovery_hit_rate` from `0.000` to `0.219`, and `average_quality_score` from `0.974` to `1.000`; `first_speech_hit_rate` stayed locked at `0.217`.
- Weak-slice exact quality-ready recovery: `disclose=0.449`, `inform=0.739`, and `other=0.723`.
- Final suite: `python3 -m pytest -v`: 131 passed.
- `git diff --check`: passed.
- Next lever: seed/fold stress test plus dashboard panel contrasting raw semantic coverage with quality-ready prepared coverage.

## Response-Mode Recovery Stress Dashboard

- [x] Add a seed/fold stress runner for quality-aware response-mode recovery promotion.
- [x] Expose the stress runner through the CLI and write a JSON artifact.
- [x] Add a response-mode dashboard panel that contrasts raw semantic coverage with quality-ready prepared coverage.
- [x] Run an ESConv stress pass and render the quick-reference HTML dashboard.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_recovery_stress_test_reports_seed_fold_stability tests/test_visualization.py::test_render_response_mode_dashboard_includes_quality_ready_recovery_panel -v` failed because `run_response_mode_recovery_stress_test` did not exist.
- Focused green check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_recovery_stress_test_reports_seed_fold_stability tests/test_visualization.py::test_render_response_mode_dashboard_includes_quality_ready_recovery_panel tests/test_cli.py::test_cli_runs_response_mode_recovery_stress_report -v`: 3 passed.
- Full pre-artifact suite: `python3 -m pytest -v`: 134 passed.
- Dashboard generation: `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json --dashboard-report runs/esconv_response_mode_dashboard.html`: completed.
- Browser verification: Playwright loaded `http://127.0.0.1:8765/runs/esconv_response_mode_dashboard.html`, confirmed title `Premonition Response-Mode Recovery`, and showed the quality-ready recovery panel. The only console error was a local `favicon.ico` 404.
- Stress run: `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-stress-report runs/esconv_response_mode_recovery_stress.json --response-mode-stress-seeds 2 --folds 3`: completed.
- Stress result: promotion rate `0.5` across `6` shuffled folds; mean prepared-hit gain `+0.018`, mean quality-ready gain `+0.038`, max quality-ready gain `+0.079`, worst-fold gain `0.000`, and selected policies `recover_disclose_inform=3`, `recover_inform=2`, `none=1`.
- Final suite: `python3 -m pytest -v`: 134 passed.
- `git diff --check`: passed.
- Next lever: improve `recover_disclose_inform` stability on shuffled zero-gain folds before expanding to more seeds/folds.

## Quality-Aware Stress Stabilization

- [x] Add failing checks that bakeoff and stress reports expose the quality-aware held-out baseline.
- [x] Compare held-out recovery candidates against the quality-aware baseline used by the recovery gate, while preserving raw baseline metrics for gain reporting.
- [x] Add the quality-aware gate to the response-mode dashboard.
- [x] Regenerate ESConv bakeoff, stress, and dashboard artifacts.
- [x] Verify full suite, commit, and push.

Verification:

- Initial red check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments tests/test_conversation_probability.py::test_response_mode_recovery_stress_test_reports_seed_fold_stability -v` failed because reports did not expose the quality-aware held-out baseline.
- Focused green check: `python3 -m pytest tests/test_conversation_probability.py::test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments tests/test_conversation_probability.py::test_response_mode_recovery_stress_test_reports_seed_fold_stability -v`: 2 passed.
- Dashboard red check: `python3 -m pytest tests/test_visualization.py::test_render_response_mode_dashboard_includes_quality_ready_recovery_panel -v` failed because the quick-reference page did not show the quality-aware gate.
- Dashboard green check: `python3 -m pytest tests/test_visualization.py::test_render_response_mode_dashboard_includes_quality_ready_recovery_panel -v`: 1 passed.
- Stress run: `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-stress-report runs/esconv_response_mode_recovery_stress.json --response-mode-stress-seeds 2 --folds 3`: completed.
- Stress result: promotion rate `0.833` across `6` shuffled folds; mean prepared-hit gain `+0.020`, mean quality-ready gain `+0.054`, max quality-ready gain `+0.079`, worst-fold gain `0.000`, and selected policies `recover_disclose_inform=3`, `recover_inform=2`, `none=1`.
- Dashboard generation: `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-bakeoff-report runs/esconv_response_mode_bakeoff.json --dashboard-report runs/esconv_response_mode_dashboard.html`: completed.
- Dashboard static check: `rg -n "Quality-Aware Gate|0\\.546 -> 0\\.765|Premonition Response-Mode Recovery" runs/esconv_response_mode_dashboard.html`: passed.
- Final suite: `python3 -m pytest -v`: 134 passed.
- `git diff --check`: passed.
- Next lever: run a larger seed/fold stress pass and target the single no-policy fold with better validation-time target discovery.

## Larger Recovery Stress Pass

- [x] Run a larger response-mode recovery stress pass with more seeds and more folds.
- [x] Inspect aggregate stability, weakest folds, and selected recovery policy distribution.
- [x] Update README, dataset notes, and the canonical stress artifact with the larger result.
- [x] Verify, commit, and push.

Verification:

- Clean-start check: `git status --short`: clean.
- Stress run: `foresight-replay --conversation-train-input data/esconv_train_response_modes_sample.jsonl --conversation-dev-input data/esconv_validation_response_modes_sample.jsonl --conversation-test-input data/esconv_test_response_modes_sample.jsonl --response-mode-stress-report runs/esconv_response_mode_recovery_stress.json --response-mode-stress-seeds 3 --folds 5`: completed.
- Stress result: promotion rate `1.000` across `15` shuffled folds; mean prepared-hit gain `+0.032`, mean quality-ready gain `+0.074`, minimum quality-ready gain `+0.040`, max quality-ready gain `+0.164`, and selected policies `recover_disclose_inform=8`, `recover_inform=6`, `recover_other=1`.
- Weakest quality-ready fold: seed `1`, fold `1`, selected `recover_inform`, prepared-hit gain `-0.006`, quality-ready gain `+0.040`, and background-recovery hit rate `0.039`.
- Interpretation: every fold improved quality-ready coverage. The small raw prepared-hit dip happens because the active pack refuses low-quality semantic drafts that the raw baseline previously counted as prepared.
- Final suite: `python3 -m pytest -v`: 134 passed.
- `git diff --check`: passed.
