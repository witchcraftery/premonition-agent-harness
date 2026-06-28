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
