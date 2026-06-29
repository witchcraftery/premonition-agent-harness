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
