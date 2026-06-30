import json
import subprocess
import sys
from pathlib import Path

from foresight_harness.conversation_probability import (
    build_probability_pack,
    build_response_mode_probability_pack,
    ConversationGuidance,
    ConversationTurn,
    conversation_bakeoff_variants,
    conversation_features,
    conversation_turn_to_dict,
    cross_validate_conversation_variant,
    generate_conversation_branches,
    generate_response_mode_branches,
    load_esconv_export,
    load_empatheticdialogues_export,
    learn_conversation_guidance,
    load_dailydialog_split,
    load_conversation_turns,
    load_dailydialog_export,
    calibrate_response_mode_specialist_thresholds,
    run_conversation_act_ranker_bakeoff,
    run_conversation_probability_loop,
    run_conversation_train_dev_test_loop,
    run_response_mode_ranker_bakeoff,
    run_response_mode_recovery_stress_test,
    response_mode_bakeoff_variants,
    response_mode_background_recovery_evaluation,
    response_mode_background_recovery_policy,
    response_mode_background_recovery_policy_candidates,
    response_mode_recovery_policy_for_replay,
    response_mode_draft_quality_score,
    response_mode_match_grade,
    response_mode_probability_pack_policy,
    response_mode_recommendations,
    score_response_mode_probability_pack_replay,
    score_conversation_variant_fold,
    select_conversation_bakeoff_variant,
    select_response_mode_background_recovery_candidate,
    train_conversation_act_ranker,
    train_conversation_history_ranker,
    train_conversation_question_evidence_ranker,
    train_conversation_transition_ranker,
    train_response_mode_specialists,
    train_response_mode_ranker,
    write_conversation_turns,
)
from foresight_harness.models import Message


def test_load_dailydialog_export_creates_next_turn_examples(tmp_path):
    text_path = tmp_path / "dialogues_text.txt"
    act_path = tmp_path / "dialogues_act.txt"
    emotion_path = tmp_path / "dialogues_emotion.txt"
    text_path.write_text(
        "Hi there.__eou__ How are you today?__eou__ I am doing well, thanks.__eou__\n",
        encoding="utf-8",
    )
    act_path.write_text("1 2 1\n", encoding="utf-8")
    emotion_path.write_text("0 0 4\n", encoding="utf-8")

    turns = load_dailydialog_export(text_path, act_path, emotion_path)

    assert len(turns) == 2
    assert turns[0].turn_id == "dailydialog-0001-001"
    assert turns[0].actual_next_utterance == "How are you today?"
    assert turns[0].expected_act == "question"
    assert turns[0].observed_acts == ("inform",)
    assert turns[1].expected_act == "inform"
    assert turns[1].expected_emotion == "happiness"
    assert turns[1].observed_acts == ("inform", "question")


def test_load_dailydialog_split_writes_limited_jsonl_sample(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    (split_dir / "dialogues.txt").write_text(
        "I found a quiet cafe.__eou__ That sounds peaceful.__eou__ Should we meet there tomorrow?__eou__\n"
        "I am worried about the storm.__eou__ What time is it expected to arrive?__eou__\n",
        encoding="utf-8",
    )
    (split_dir / "dialogues_act.txt").write_text("1 1 3\n1 2\n", encoding="utf-8")
    (split_dir / "dialogues_emotion.txt").write_text("0 0 0\n3 0\n", encoding="utf-8")
    output = tmp_path / "sample.jsonl"

    turns = load_dailydialog_split(split_dir)
    write_conversation_turns(turns, output, limit=3)
    saved = load_conversation_turns(output)

    assert len(turns) == 3
    assert len(saved) == 3
    assert saved[0].turn_id == "dailydialog-train-0001-001"
    assert saved[0].observed_acts == ("inform",)
    assert saved[1].expected_act == "directive"
    assert saved[1].observed_acts == ("inform", "inform")
    assert saved[2].expected_act == "question"


def test_load_empatheticdialogues_export_creates_next_turn_examples(tmp_path):
    input_path = tmp_path / "empathetic.csv"
    input_path.write_text(
        "conv_id,utterance_idx,context,prompt,speaker_idx,utterance\n"
        "c1,1,proud,I finished the mural.,0,I finished the mural today.\n"
        "c1,2,proud,I finished the mural.,1,That is wonderful news!\n"
        "c1,3,proud,I finished the mural.,0,Can I show it to you later?\n"
        "c2,1,afraid,I heard a noise.,0,I heard a noise outside.\n"
        "c2,2,afraid,I heard a noise.,1,Please check the lock.\n",
        encoding="utf-8",
    )

    turns = load_empatheticdialogues_export(input_path)

    assert len(turns) == 3
    assert turns[0].turn_id == "empathetic-c1-002"
    assert turns[0].next_speaker == "speaker_b"
    assert turns[0].actual_next_utterance == "That is wonderful news!"
    assert turns[0].expected_act == "inform"
    assert turns[0].expected_emotion == "happiness"
    assert turns[0].observed_acts == ("inform",)
    assert turns[1].expected_act == "question"
    assert turns[2].expected_act == "directive"
    assert turns[2].expected_emotion == "fear"


def test_load_empatheticdialogues_export_supports_jsonl_rows(tmp_path):
    input_path = tmp_path / "empathetic.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "conv_id": "j1",
                        "utterance_idx": 1,
                        "context": "grateful",
                        "prompt": "A friend helped me move.",
                        "speaker_idx": 0,
                        "utterance": "My friend helped me move.",
                    }
                ),
                json.dumps(
                    {
                        "conv_id": "j1",
                        "utterance_idx": 2,
                        "context": "grateful",
                        "prompt": "A friend helped me move.",
                        "speaker_idx": 1,
                        "utterance": "I can send them a thank you note.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    turns = load_empatheticdialogues_export(input_path)

    assert len(turns) == 1
    assert turns[0].expected_act == "commissive"
    assert turns[0].expected_emotion == "happiness"


def test_load_empatheticdialogues_export_detects_common_commitment_phrases(tmp_path):
    input_path = tmp_path / "empathetic.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "conv_id": "j2",
                        "utterance_idx": 1,
                        "context": "caring",
                        "prompt": "A friend needed help.",
                        "speaker_idx": 0,
                        "utterance": "My friend needed help.",
                    }
                ),
                json.dumps(
                    {
                        "conv_id": "j2",
                        "utterance_idx": 2,
                        "context": "caring",
                        "prompt": "A friend needed help.",
                        "speaker_idx": 1,
                        "utterance": "I would stay with them until they felt safe.",
                    }
                ),
                json.dumps(
                    {
                        "conv_id": "j2",
                        "utterance_idx": 3,
                        "context": "caring",
                        "prompt": "A friend needed help.",
                        "speaker_idx": 0,
                        "utterance": "I'm going to call them tonight.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    turns = load_empatheticdialogues_export(input_path)

    assert turns[0].expected_act == "commissive"
    assert turns[1].expected_act == "commissive"


def test_conversation_turn_round_trips_response_mode_fields():
    turn = ConversationTurn(
        turn_id="mode-001",
        conversation=(Message(role="speaker_a", content="I feel overwhelmed."),),
        next_speaker="speaker_b",
        actual_next_utterance="That sounds really heavy.",
        expected_act="inform",
        expected_emotion="sadness",
        expected_response_mode="validate",
        observed_acts=("inform",),
        observed_response_modes=("ask_followup",),
    )

    saved = conversation_turn_to_dict(turn)
    loaded = ConversationTurn.from_dict(saved)

    assert saved["expected_response_mode"] == "validate"
    assert saved["observed_response_modes"] == ["ask_followup"]
    assert loaded.expected_response_mode == "validate"
    assert loaded.observed_response_modes == ("ask_followup",)


def test_load_esconv_export_creates_response_mode_examples(tmp_path):
    input_path = tmp_path / "ESConv.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "emotion_type": "anxiety",
                    "experience_type": "Previous Experience",
                    "problem_type": "job crisis",
                    "dialog": [
                        {
                            "speaker": "seeker",
                            "annotation": {},
                            "content": "I hate my job.",
                        },
                        {
                            "speaker": "supporter",
                            "annotation": {"strategy": "Question"},
                            "content": "What makes it so stressful?",
                        },
                        {
                            "speaker": "seeker",
                            "annotation": {},
                            "content": "The clients are in hard situations.",
                        },
                        {
                            "speaker": "supporter",
                            "annotation": {
                                "strategy": "Reflection of feelings"
                            },
                            "content": "That sounds emotionally draining.",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    turns = load_esconv_export(input_path, split="train")

    assert len(turns) == 2
    assert turns[0].turn_id == "esconv-train-0001-001"
    assert turns[0].next_speaker == "supporter"
    assert turns[0].expected_response_mode == "ask_followup"
    assert turns[0].expected_act == "question"
    assert turns[0].expected_emotion == "fear"
    assert turns[1].expected_response_mode == "validate"
    assert turns[1].observed_response_modes == ("ask_followup",)
    assert turns[0].source_metadata == (
        ("emotion_type", "anxiety"),
        ("experience_type", "Previous Experience"),
        ("problem_type", "job crisis"),
    )
    assert "problem_type_crisis_job" in conversation_features(turns[0])
    assert "emotion_type_anxiety" in conversation_features(turns[0])


def test_learned_response_mode_ranker_predicts_repeated_support_mode():
    train_turns = (
        conversation_turn(
            "train-validate-1",
            "drainingcue emotionally heavy",
            "inform",
            expected_response_mode="validate",
        ),
        conversation_turn(
            "train-validate-2",
            "drainingcue feels overwhelming",
            "inform",
            expected_response_mode="validate",
        ),
        conversation_turn(
            "train-ask",
            "unclear details",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    ranker = train_response_mode_ranker(train_turns)
    test_turn = conversation_turn(
        "test-validate",
        "drainingcue hard day",
        "inform",
        expected_response_mode="validate",
    )

    branches = generate_response_mode_branches(test_turn, top_k=3, ranker=ranker)

    assert branches[0]["response_mode"] == "validate"
    assert branches[0]["scoring_variant"] == "learned_response_mode"


def test_class_balanced_response_mode_ranker_uses_uniform_priors():
    train_turns = (
        conversation_turn(
            "train-ask-1",
            "unclear details",
            "question",
            expected_response_mode="ask_followup",
        ),
        conversation_turn(
            "train-ask-2",
            "what happened?",
            "question",
            expected_response_mode="ask_followup",
        ),
        conversation_turn(
            "train-disclose",
            "storycue similar experience",
            "inform",
            expected_response_mode="disclose",
        ),
    )

    standard = train_response_mode_ranker(train_turns)
    balanced = train_response_mode_ranker(train_turns, class_balanced=True)

    assert standard.mode_log_priors["ask_followup"] > standard.mode_log_priors["disclose"]
    assert balanced.mode_log_priors["ask_followup"] == balanced.mode_log_priors["disclose"]


def test_response_mode_bakeoff_selects_on_dev_and_reports_test_segments():
    train_turns = (
        conversation_turn(
            "train-validate-1",
            "drainingcue emotionally heavy",
            "inform",
            expected_response_mode="validate",
        ),
        conversation_turn(
            "train-validate-2",
            "drainingcue feels overwhelming",
            "inform",
            expected_response_mode="validate",
        ),
        conversation_turn(
            "train-ask-1",
            "unclear details",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    dev_turns = (
        conversation_turn(
            "dev-validate",
            "drainingcue hard day",
            "inform",
            expected_response_mode="validate",
        ),
        conversation_turn(
            "dev-ask",
            "unclear plan",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    test_turns = (
        conversation_turn(
            "test-validate",
            "drainingcue difficult moment",
            "inform",
            expected_response_mode="validate",
        ),
        conversation_turn(
            "test-ask",
            "unclear situation",
            "question",
            expected_response_mode="ask_followup",
        ),
    )

    report = run_response_mode_ranker_bakeoff(
        train_turns=train_turns,
        dev_turns=dev_turns,
        test_turns=test_turns,
        top_k=3,
    )

    assert report["selected_variant"]["name"] in report["variants"]
    assert report["selected_variant"]["test"]["p_at_1"] == 1.0
    assert "expected_response_mode" in report["analytics"]["test_segments"]
    assert "coverage_projection" in report
    assert "validate" in report["coverage_projection"]["expected_response_mode"]
    assert "specialist_diagnostics" in report
    assert "protected_minority_specialists" in report["variants"]
    assert "specialist_calibration" in report
    assert "calibrated_minority_specialist_coverage" in report["variants"]
    assert "recommendations" in report
    assert "first_speech" in report["recommendations"]
    assert "background_readiness" in report["recommendations"]
    assert "probability_pack_policy" in report
    assert "background_preparation" in report["probability_pack_policy"]
    assert "probability_pack_replay" in report
    assert "prepared_hit_rate" in report["probability_pack_replay"]
    assert "probability_pack_replay_baseline" in report
    assert "probability_pack_replay_baseline_quality_aware" in report
    assert "background_recovery_policy" in report
    assert "background_recovery_calibration" in report
    assert "candidate_evaluations" in report["background_recovery_calibration"]
    assert "selected_policy" in report["background_recovery_calibration"]
    assert report["background_recovery_calibration"]["min_quality_score"] == 0.75


def test_response_mode_recovery_stress_test_reports_seed_fold_stability():
    turns = tuple(
        conversation_turn(
            f"stress-validate-{index}",
            "drainingcue emotionally heavy",
            "inform",
            expected_response_mode="validate",
        )
        for index in range(6)
    ) + tuple(
        conversation_turn(
            f"stress-ask-{index}",
            "unclear details",
            "question",
            expected_response_mode="ask_followup",
        )
        for index in range(6)
    )

    report = run_response_mode_recovery_stress_test(
        turns,
        seeds=(0, 1),
        fold_count=3,
        top_k=3,
    )

    assert report["summary"]["seed_count"] == 2
    assert report["summary"]["fold_count"] == 3
    assert report["summary"]["run_count"] == 6
    assert len(report["runs"]) == 6
    assert "prepared_hit_gain" in report["aggregates"]
    assert "selected_policy_counts" in report["aggregates"]
    assert all("quality_ready_gain" in run for run in report["runs"])
    assert all("baseline_quality_aware" in run for run in report["runs"])


def test_response_mode_specialist_promotes_target_mode_from_metadata():
    train_turns = (
        conversation_turn(
            "train-other-1",
            "The situation is complicated.",
            "inform",
            expected_response_mode="other",
            source_metadata=(("problem_type", "family conflict"),),
        ),
        conversation_turn(
            "train-other-2",
            "There are many pieces here.",
            "inform",
            expected_response_mode="other",
            source_metadata=(("problem_type", "family conflict"),),
        ),
        conversation_turn(
            "train-ask",
            "What happened next?",
            "question",
            expected_response_mode="ask_followup",
            source_metadata=(("problem_type", "job crisis"),),
        ),
    )
    specialists = train_response_mode_specialists(train_turns, target_modes=("other",))
    test_turn = conversation_turn(
        "test-other",
        "The situation has many pieces.",
        "inform",
        expected_response_mode="other",
        source_metadata=(("problem_type", "family conflict"),),
    )

    branches = generate_response_mode_branches(
        test_turn,
        top_k=3,
        specialists=specialists,
        specialist_modes=("other",),
        specialist_min_score=0.0,
        scoring_variant="protected_minority_specialists",
    )

    assert branches[0]["response_mode"] == "other"
    assert branches[0]["scoring_variant"] == "protected_minority_specialists"


def test_response_mode_specialist_preserves_protected_top_mode():
    train_turns = (
        conversation_turn(
            "train-other",
            "familycue complicated",
            "inform",
            expected_response_mode="other",
        ),
        conversation_turn(
            "train-ask-1",
            "What happened next?",
            "question",
            expected_response_mode="ask_followup",
        ),
        conversation_turn(
            "train-ask-2",
            "Why did it change?",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    specialists = train_response_mode_specialists(train_turns, target_modes=("other",))
    test_turn = conversation_turn(
        "test-ask",
        "What happened next?",
        "question",
        expected_response_mode="ask_followup",
    )

    branches = generate_response_mode_branches(
        test_turn,
        top_k=3,
        specialists=specialists,
        specialist_modes=("other",),
        specialist_min_score=0.0,
        specialist_preserved_modes=("ask_followup",),
        scoring_variant="protected_minority_specialists",
    )

    assert branches[0]["response_mode"] == "ask_followup"


def test_response_mode_specialist_score_includes_mode_prior():
    train_turns = (
        conversation_turn(
            "train-other",
            "familycue complicated",
            "inform",
            expected_response_mode="other",
            source_metadata=(("problem_type", "family conflict"),),
        ),
        conversation_turn(
            "train-ask-1",
            "What happened next?",
            "question",
            expected_response_mode="ask_followup",
        ),
        conversation_turn(
            "train-ask-2",
            "Why did it change?",
            "question",
            expected_response_mode="ask_followup",
        ),
        conversation_turn(
            "train-ask-3",
            "How are you handling it?",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    specialist = train_response_mode_specialists(
        train_turns,
        target_modes=("other",),
    )["other"]
    positive_turn = conversation_turn(
        "test-other",
        "familycue complicated",
        "inform",
        expected_response_mode="other",
        source_metadata=(("problem_type", "family conflict"),),
    )
    negative_turn = conversation_turn(
        "test-ask",
        "What happened next?",
        "question",
        expected_response_mode="ask_followup",
    )

    assert specialist.mode_log_prior < 0
    assert specialist.score(positive_turn) > specialist.score(negative_turn)


def test_response_mode_specialist_top_three_preserves_first_branch():
    train_turns = (
        conversation_turn(
            "train-other-1",
            "familycue complicated",
            "inform",
            expected_response_mode="other",
            source_metadata=(("problem_type", "family conflict"),),
        ),
        conversation_turn(
            "train-other-2",
            "familycue many pieces",
            "inform",
            expected_response_mode="other",
            source_metadata=(("problem_type", "family conflict"),),
        ),
        conversation_turn(
            "train-ask-1",
            "What happened next?",
            "question",
            expected_response_mode="ask_followup",
            source_metadata=(("problem_type", "job crisis"),),
        ),
        conversation_turn(
            "train-ask-2",
            "Why did it change?",
            "question",
            expected_response_mode="ask_followup",
            source_metadata=(("problem_type", "job crisis"),),
        ),
    )
    specialists = train_response_mode_specialists(train_turns, target_modes=("other",))
    test_turn = conversation_turn(
        "test-ask-other",
        "What happened next?",
        "question",
        expected_response_mode="other",
        source_metadata=(("problem_type", "family conflict"),),
    )

    branches = generate_response_mode_branches(
        test_turn,
        top_k=3,
        specialists=specialists,
        specialist_modes=("other",),
        specialist_min_score=0.0,
        specialist_preserved_modes=("ask_followup",),
        specialist_insert_mode="top_3",
        scoring_variant="protected_minority_specialist_coverage",
    )

    response_modes = [branch["response_mode"] for branch in branches]
    assert response_modes[0] == "ask_followup"
    assert "other" in response_modes[1:]


def test_response_mode_specialist_uses_per_mode_thresholds():
    train_turns = (
        conversation_turn(
            "train-other",
            "familycue complicated",
            "inform",
            expected_response_mode="other",
            source_metadata=(("problem_type", "family conflict"),),
        ),
        conversation_turn(
            "train-inform",
            "factcue practical details",
            "inform",
            expected_response_mode="inform",
            source_metadata=(("problem_type", "job crisis"),),
        ),
        conversation_turn(
            "train-ask",
            "What happened next?",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    specialists = train_response_mode_specialists(
        train_turns,
        target_modes=("other", "inform"),
    )
    test_turn = conversation_turn(
        "test",
        "familycue factcue practical details",
        "question",
        source_metadata=(("problem_type", "job crisis"),),
    )

    branches = generate_response_mode_branches(
        test_turn,
        top_k=3,
        specialists=specialists,
        specialist_modes=("other", "inform"),
        specialist_min_score=-10.0,
        specialist_mode_min_scores={"other": 99.0, "inform": -10.0},
        specialist_insert_mode="top_3",
        scoring_variant="calibrated_minority_specialist_coverage",
    )

    response_modes = [branch["response_mode"] for branch in branches]
    assert "inform" in response_modes
    assert "other" not in response_modes


def test_response_mode_specialist_calibration_rejects_dev_top_three_drop():
    train_turns = (
        conversation_turn(
            "train-other",
            "familycue complicated",
            "inform",
            expected_response_mode="other",
            source_metadata=(("problem_type", "family conflict"),),
        ),
        conversation_turn(
            "train-ask",
            "What happened next?",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    dev_turns = (
        conversation_turn(
            "dev-reassure",
            "What happened next?",
            "inform",
            expected_response_mode="reassure",
            source_metadata=(("problem_type", "family conflict"),),
        ),
    )
    specialists = train_response_mode_specialists(train_turns, target_modes=("other",))
    baseline_rows = (
        {
            "turn_id": "dev-reassure",
            "expected_response_mode": "reassure",
            "rank_1_response_mode": "ask_followup",
            "top_response_modes": ["ask_followup", "validate", "reassure"],
        },
    )

    calibration = calibrate_response_mode_specialist_thresholds(
        dev_turns=dev_turns,
        baseline_rows=baseline_rows,
        specialists=specialists,
        specialist_modes=("other",),
        candidate_min_scores=(-10.0,),
        top_k=3,
    )

    assert calibration["accepted_mode_min_scores"] == {}
    assert calibration["modes"]["other"]["accepted"] is False
    assert calibration["modes"]["other"]["rejection_reason"] == "aggregate_dev_top_3_drop"


def test_response_mode_recommendations_split_speech_and_readiness_winners():
    variants = {
        "heuristic_response_mode": {
            "learned_weight": 0.0,
            "dev": {"p_at_1": 0.2, "top_3_recall": 0.53},
            "test": {"p_at_1": 0.2, "top_3_recall": 0.53},
            "dev_segment_regressions": [],
            "test_segment_regressions": [],
        },
        "response_mode_hybrid_75": {
            "learned_weight": 0.75,
            "dev": {"p_at_1": 0.21, "top_3_recall": 0.54},
            "test": {"p_at_1": 0.22, "top_3_recall": 0.53},
            "dev_segment_regressions": [],
            "test_segment_regressions": [{"segment": "expected_response_mode"}],
        },
        "calibrated_minority_specialist_coverage": {
            "learned_weight": 0.0,
            "dev": {"p_at_1": 0.2, "top_3_recall": 0.55},
            "test": {"p_at_1": 0.2, "top_3_recall": 0.56},
            "dev_segment_regressions": [],
            "test_segment_regressions": [],
        },
        "learned_response_mode": {
            "learned_weight": 1.0,
            "dev": {"p_at_1": 0.27, "top_3_recall": 0.62},
            "test": {"p_at_1": 0.27, "top_3_recall": 0.61},
            "dev_segment_regressions": [{"segment": "expected_response_mode"}],
            "test_segment_regressions": [],
        },
    }

    recommendations = response_mode_recommendations(
        variants,
        first_speech_variant_name="response_mode_hybrid_75",
    )

    assert recommendations["first_speech"]["name"] == "response_mode_hybrid_75"
    assert recommendations["first_speech"]["heldout_promotable"] is False
    assert (
        recommendations["background_readiness"]["name"]
        == "calibrated_minority_specialist_coverage"
    )
    assert recommendations["background_readiness"]["heldout_promotable"] is True
    assert recommendations["background_readiness"]["reason"] == (
        "held-out test passed background readiness checks"
    )


def test_response_mode_probability_pack_policy_uses_recommendation_promotability():
    recommendations = {
        "first_speech": {
            "name": "response_mode_hybrid_75",
            "heldout_promotable": False,
        },
        "background_readiness": {
            "name": "calibrated_minority_specialist_coverage",
            "heldout_promotable": True,
        },
    }

    policy = response_mode_probability_pack_policy(recommendations)

    assert policy["first_speech_variant"] == "response_mode_hybrid_75"
    assert policy["first_speech_delivery"] == "confirm_before_delivery"
    assert (
        policy["background_readiness_variant"]
        == "calibrated_minority_specialist_coverage"
    )
    assert policy["background_preparation"] == "prewarm_tts"
    assert policy["confirmation_mode"] == (
        "confirm_first_speech_then_stream_prepared_background"
    )


def test_response_mode_probability_pack_prepares_background_readiness_branches():
    turn = conversation_turn(
        "pack-turn",
        "I am scared this will happen again.",
        "inform",
        expected_response_mode="reassure",
    )
    policy = {
        "first_speech_variant": "response_mode_hybrid_75",
        "first_speech_delivery": "confirm_before_delivery",
        "background_readiness_variant": "calibrated_minority_specialist_coverage",
        "background_preparation": "prewarm_tts",
        "confirmation_mode": "confirm_first_speech_then_stream_prepared_background",
    }
    first_speech_branches = (
        {
            "branch_id": "pack-turn-mode-branch-1",
            "rank": 1,
            "response_mode": "ask_followup",
            "tts_text": "I can ask a focused follow-up question.",
            "probability": 0.7,
            "trigger_cues": ["scared"],
            "scoring_variant": "response_mode_hybrid_75",
        },
    )
    background_readiness_branches = (
        {
            "branch_id": "pack-turn-mode-branch-1",
            "rank": 1,
            "response_mode": "ask_followup",
            "tts_text": "I can ask a focused follow-up question.",
            "probability": 0.55,
            "trigger_cues": ["scared"],
            "scoring_variant": "calibrated_minority_specialist_coverage",
        },
        {
            "branch_id": "pack-turn-mode-branch-2",
            "rank": 2,
            "response_mode": "reassure",
            "tts_text": "I can offer calm reassurance without overpromising.",
            "probability": 0.54,
            "trigger_cues": ["scared"],
            "scoring_variant": "calibrated_minority_specialist_coverage",
        },
    )

    pack = build_response_mode_probability_pack(
        turn,
        first_speech_branches=first_speech_branches,
        background_readiness_branches=background_readiness_branches,
        policy=policy,
    )

    assert pack.confirmation_mode == (
        "confirm_first_speech_then_stream_prepared_background"
    )
    assert pack.top_branches[0]["response_mode"] == "ask_followup"
    assert pack.top_branches[0]["preparation_role"] == "first_speech"
    assert pack.top_branches[0]["source_variant"] == "response_mode_hybrid_75"
    reassure_draft = next(
        draft for draft in pack.prepared_drafts if draft["response_mode"] == "reassure"
    )
    assert reassure_draft["preparation_role"] == "background_readiness"
    assert reassure_draft["delivery_policy"] == "prewarm_tts"
    assert reassure_draft["source_variant"] == "calibrated_minority_specialist_coverage"


def test_response_mode_match_grade_counts_exact_and_semantic_hits():
    assert response_mode_match_grade("reassure", "reassure") == "exact"
    assert response_mode_match_grade("validate", "reassure") == "semantic_equivalent"
    assert response_mode_match_grade("suggest", "reassure") == "miss"


def test_response_mode_probability_pack_replay_scores_tts_readiness_hits():
    exact_turn = conversation_turn(
        "exact-pack-turn",
        "I am scared this will happen again.",
        "inform",
        expected_response_mode="reassure",
    )
    semantic_turn = conversation_turn(
        "semantic-pack-turn",
        "I feel completely overwhelmed.",
        "inform",
        expected_response_mode="reassure",
    )
    policy = {
        "first_speech_variant": "response_mode_hybrid_75",
        "first_speech_delivery": "confirm_before_delivery",
        "background_readiness_variant": "calibrated_minority_specialist_coverage",
        "background_preparation": "prewarm_tts",
        "confirmation_mode": "confirm_first_speech_then_stream_prepared_background",
    }
    first_speech = (
        response_mode_branch("ask_followup", "response_mode_hybrid_75", 0.7),
    )
    exact_background = (
        response_mode_branch("ask_followup", "calibrated_minority_specialist_coverage", 0.5),
        response_mode_branch("reassure", "calibrated_minority_specialist_coverage", 0.49),
    )
    semantic_background = (
        response_mode_branch("ask_followup", "calibrated_minority_specialist_coverage", 0.5),
        response_mode_branch("validate", "calibrated_minority_specialist_coverage", 0.49),
    )
    packs = (
        build_response_mode_probability_pack(
            exact_turn,
            first_speech_branches=first_speech,
            background_readiness_branches=exact_background,
            policy=policy,
        ),
        build_response_mode_probability_pack(
            semantic_turn,
            first_speech_branches=first_speech,
            background_readiness_branches=semantic_background,
            policy=policy,
        ),
    )

    summary = score_response_mode_probability_pack_replay(
        (exact_turn, semantic_turn),
        packs,
        prepared_latency_ms=90,
    )

    assert summary["prepared_hit_rate"] == 1.0
    assert summary["exact_prepared_hit_rate"] == 0.5
    assert summary["semantic_prepared_hit_rate"] == 0.5
    assert summary["background_hit_rate"] == 1.0
    assert summary["first_speech_hit_rate"] == 0.0
    assert summary["median_latency_ms"] == 90
    assert summary["median_latency_saved_ms"] == 560


def test_response_mode_draft_quality_scores_mode_specific_prepared_speech():
    assert (
        response_mode_draft_quality_score(
            {
                "response_mode": "reassure",
                "tts_text": "I can offer grounded reassurance without overpromising.",
                "voice_ready": True,
            },
            expected_response_mode="reassure",
        )
        == 1.0
    )
    assert (
        response_mode_draft_quality_score(
            {
                "response_mode": "ask_followup",
                "tts_text": "I can ask one warm follow-up question.",
                "voice_ready": True,
            },
            expected_response_mode="reassure",
        )
        < 0.7
    )


def test_response_mode_probability_pack_replay_reports_per_mode_quality():
    reassure_turn = conversation_turn(
        "reassure-pack-turn",
        "I am scared this will happen again.",
        "inform",
        expected_response_mode="reassure",
    )
    ask_turn = conversation_turn(
        "ask-pack-turn",
        "I do not know what to do next.",
        "question",
        expected_response_mode="ask_followup",
    )
    policy = {
        "first_speech_variant": "response_mode_hybrid_75",
        "first_speech_delivery": "confirm_before_delivery",
        "background_readiness_variant": "calibrated_minority_specialist_coverage",
        "background_preparation": "prewarm_tts",
        "confirmation_mode": "confirm_first_speech_then_stream_prepared_background",
    }
    packs = (
        build_response_mode_probability_pack(
            reassure_turn,
            first_speech_branches=(
                response_mode_branch("ask_followup", "response_mode_hybrid_75", 0.7),
            ),
            background_readiness_branches=(
                response_mode_branch("ask_followup", "calibrated_minority_specialist_coverage", 0.5),
                response_mode_branch("reassure", "calibrated_minority_specialist_coverage", 0.49),
            ),
            policy=policy,
        ),
        build_response_mode_probability_pack(
            ask_turn,
            first_speech_branches=(
                response_mode_branch("ask_followup", "response_mode_hybrid_75", 0.7),
            ),
            background_readiness_branches=(
                response_mode_branch("ask_followup", "calibrated_minority_specialist_coverage", 0.5),
                response_mode_branch("reassure", "calibrated_minority_specialist_coverage", 0.49),
            ),
            policy=policy,
        ),
    )

    summary = score_response_mode_probability_pack_replay(
        (reassure_turn, ask_turn),
        packs,
        prepared_latency_ms=90,
    )

    assert summary["average_quality_score"] == 1.0
    assert summary["quality_ready_rate"] == 1.0
    reassure_segment = summary["segments"]["expected_response_mode"]["reassure"]
    ask_segment = summary["segments"]["expected_response_mode"]["ask_followup"]
    assert reassure_segment["prepared_hit_rate"] == 1.0
    assert reassure_segment["background_hit_rate"] == 1.0
    assert reassure_segment["average_quality_score"] == 1.0
    assert ask_segment["first_speech_hit_rate"] == 1.0
    assert ask_segment["median_latency_saved_ms"] == 560


def test_response_mode_probability_pack_replay_counts_background_recovery_hits():
    disclose_turn = conversation_turn(
        "disclose-recovery-turn",
        "I feel alone in this.",
        "inform",
        expected_response_mode="disclose",
    )
    policy = {
        "first_speech_variant": "response_mode_hybrid_75",
        "first_speech_delivery": "confirm_before_delivery",
        "background_readiness_variant": "calibrated_minority_specialist_coverage",
        "background_preparation": "prewarm_tts",
        "confirmation_mode": "confirm_first_speech_then_stream_prepared_background",
    }
    pack = build_response_mode_probability_pack(
        disclose_turn,
        first_speech_branches=(
            response_mode_branch("ask_followup", "response_mode_hybrid_75", 0.7),
        ),
        background_readiness_branches=(
            response_mode_branch("ask_followup", "calibrated_minority_specialist_coverage", 0.5),
            response_mode_branch("reassure", "calibrated_minority_specialist_coverage", 0.49),
        ),
        background_recovery_branches=(
            response_mode_branch("disclose", "learned_response_mode", 0.41),
        ),
        policy=policy,
    )

    summary = score_response_mode_probability_pack_replay(
        (disclose_turn,),
        (pack,),
        prepared_latency_ms=90,
    )

    assert summary["background_hit_rate"] == 1.0
    assert summary["background_recovery_hit_rate"] == 1.0
    assert (
        summary["segments"]["expected_response_mode"]["disclose"][
            "background_recovery_hit_rate"
        ]
        == 1.0
    )


def test_response_mode_probability_pack_replay_can_ignore_low_quality_semantic_hits():
    disclose_turn = conversation_turn(
        "low-quality-disclose-semantic",
        "I feel alone in this.",
        "inform",
        expected_response_mode="disclose",
    )
    inform_turn = conversation_turn(
        "quality-inform-exact",
        "What should I know?",
        "inform",
        expected_response_mode="inform",
    )
    policy = {
        "first_speech_variant": "response_mode_hybrid_75",
        "first_speech_delivery": "confirm_before_delivery",
        "background_readiness_variant": "calibrated_minority_specialist_coverage",
        "background_preparation": "prewarm_tts",
        "confirmation_mode": "confirm_first_speech_then_stream_prepared_background",
    }
    packs = (
        build_response_mode_probability_pack(
            disclose_turn,
            first_speech_branches=(
                response_mode_branch("ask_followup", "response_mode_hybrid_75", 0.7),
            ),
            background_readiness_branches=(
                response_mode_branch("ask_followup", "calibrated_minority_specialist_coverage", 0.5),
                response_mode_branch("inform", "calibrated_minority_specialist_coverage", 0.49),
            ),
            policy=policy,
        ),
        build_response_mode_probability_pack(
            inform_turn,
            first_speech_branches=(
                response_mode_branch("ask_followup", "response_mode_hybrid_75", 0.7),
            ),
            background_readiness_branches=(
                response_mode_branch("ask_followup", "calibrated_minority_specialist_coverage", 0.5),
                response_mode_branch("inform", "calibrated_minority_specialist_coverage", 0.49),
            ),
            policy=policy,
        ),
    )

    raw_summary = score_response_mode_probability_pack_replay(
        (disclose_turn, inform_turn),
        packs,
        prepared_latency_ms=90,
    )
    quality_aware_summary = score_response_mode_probability_pack_replay(
        (disclose_turn, inform_turn),
        packs,
        prepared_latency_ms=90,
        min_quality_score=0.75,
    )

    assert raw_summary["prepared_hit_rate"] == 1.0
    assert raw_summary["semantic_prepared_hit_rate"] == 0.5
    assert quality_aware_summary["prepared_hit_rate"] == 0.5
    assert quality_aware_summary["semantic_prepared_hit_rate"] == 0.0
    assert quality_aware_summary["exact_prepared_hit_rate"] == 0.5


def test_response_mode_background_recovery_policy_targets_zero_hit_modes_only():
    replay_summary = {
        "average_quality_score": 0.974,
        "quality_ready_rate": 0.546,
        "first_speech_hit_rate": 0.217,
        "segments": {
            "expected_response_mode": {
                "ask_followup": {
                    "prepared_hit_rate": 0.833,
                    "quality_ready_rate": 0.833,
                },
                "disclose": {
                    "prepared_hit_rate": 0.0,
                    "quality_ready_rate": 0.0,
                },
                "inform": {
                    "prepared_hit_rate": 0.0,
                    "quality_ready_rate": 0.0,
                },
                "other": {
                    "prepared_hit_rate": 0.0,
                    "quality_ready_rate": 0.0,
                },
                "reassure": {
                    "prepared_hit_rate": 1.0,
                    "quality_ready_rate": 0.993,
                },
                "suggest": {
                    "prepared_hit_rate": 0.651,
                    "quality_ready_rate": 0.651,
                },
                "validate": {
                    "prepared_hit_rate": 1.0,
                    "quality_ready_rate": 0.811,
                },
            }
        },
    }

    policy = response_mode_background_recovery_policy(replay_summary)

    assert policy["target_modes"] == ["disclose", "inform", "other"]
    assert policy["preparation_role"] == "background_recovery"
    assert policy["first_speech_locked"] is True
    assert policy["quality_floor"] == 0.974


def test_response_mode_background_recovery_policy_uses_best_top_3_variants():
    replay_summary = {
        "average_quality_score": 0.974,
        "quality_ready_rate": 0.546,
        "first_speech_hit_rate": 0.217,
        "segments": {
            "expected_response_mode": {
                "disclose": {"prepared_hit_rate": 0.0},
                "inform": {"prepared_hit_rate": 0.0},
                "reassure": {"prepared_hit_rate": 1.0},
            }
        },
    }
    coverage_projection = {
        "expected_response_mode": {
            "disclose": {
                "best_top_3_variant": "learned_response_mode",
                "best_top_3_gain": 0.449,
            },
            "inform": {
                "best_top_3_variant": "balanced_response_mode_50",
                "best_top_3_gain": 0.739,
            },
            "reassure": {
                "best_top_3_variant": "calibrated_minority_specialist_coverage",
                "best_top_3_gain": 0.305,
            },
        }
    }

    policy = response_mode_background_recovery_policy(
        replay_summary,
        coverage_projection=coverage_projection,
    )

    assert policy["target_modes"] == ["disclose", "inform"]
    assert policy["mode_variants"] == {
        "disclose": "learned_response_mode",
        "inform": "balanced_response_mode_50",
    }


def test_response_mode_background_recovery_evaluation_blocks_quality_drop():
    baseline = {
        "prepared_hit_rate": 0.577,
        "first_speech_hit_rate": 0.217,
        "average_quality_score": 0.974,
        "segments": {
            "expected_response_mode": {
                "disclose": {"prepared_hit_rate": 0.0},
                "inform": {"prepared_hit_rate": 0.0},
            }
        },
    }
    candidate = {
        "prepared_hit_rate": 0.843,
        "first_speech_hit_rate": 0.217,
        "average_quality_score": 0.955,
        "segments": {
            "expected_response_mode": {
                "disclose": {"prepared_hit_rate": 0.858},
                "inform": {"prepared_hit_rate": 0.812},
            }
        },
    }
    policy = {
        "target_modes": ["disclose", "inform"],
        "quality_floor": 0.974,
        "first_speech_locked": True,
    }

    evaluation = response_mode_background_recovery_evaluation(
        baseline,
        candidate,
        policy,
    )

    assert evaluation["promoted"] is False
    assert evaluation["quality_floor_met"] is False
    assert evaluation["target_modes_improved"] is True
    assert evaluation["first_speech_preserved"] is True


def test_response_mode_background_recovery_evaluation_blocks_raw_prepared_drop():
    baseline = {
        "prepared_hit_rate": 0.536,
        "quality_ready_rate": 0.536,
        "first_speech_hit_rate": 0.225,
        "average_quality_score": 1.0,
        "segments": {
            "expected_response_mode": {
                "inform": {"prepared_hit_rate": 0.0},
            }
        },
    }
    candidate = {
        "prepared_hit_rate": 0.576,
        "quality_ready_rate": 0.576,
        "first_speech_hit_rate": 0.225,
        "average_quality_score": 1.0,
        "segments": {
            "expected_response_mode": {
                "inform": {"prepared_hit_rate": 0.609},
            }
        },
    }
    policy = {
        "target_modes": ["inform"],
        "quality_floor": 1.0,
        "prepared_hit_floor": 0.582,
        "first_speech_locked": True,
    }

    evaluation = response_mode_background_recovery_evaluation(
        baseline,
        candidate,
        policy,
    )

    assert evaluation["promoted"] is False
    assert evaluation["prepared_hit_floor_met"] is False
    assert evaluation["quality_ready_gain"] == 0.04
    assert evaluation["raw_prepared_hit_gain"] == -0.006


def test_response_mode_background_recovery_policy_candidates_include_mode_subsets():
    policy = {
        "target_modes": ["disclose", "inform", "other"],
        "mode_variants": {
            "disclose": "balanced_prior_top_3",
            "inform": "specialist_rescue_top_3",
            "other": "balanced_prior_top_3",
        },
        "preparation_role": "background_recovery",
        "first_speech_locked": True,
        "quality_floor": 0.974,
    }

    candidates = response_mode_background_recovery_policy_candidates(policy)

    candidate_names = {str(candidate["name"]) for candidate in candidates}
    assert "recover_disclose_inform_other" in candidate_names
    assert "recover_inform_other" in candidate_names
    assert "recover_inform" in candidate_names
    inform_policy = next(
        candidate for candidate in candidates if candidate["name"] == "recover_inform"
    )
    assert inform_policy["target_modes"] == ["inform"]
    assert inform_policy["mode_variants"] == {"inform": "specialist_rescue_top_3"}
    assert inform_policy["quality_floor"] == 0.974


def test_select_response_mode_background_recovery_candidate_prefers_quality_safe_subset():
    baseline = {
        "prepared_hit_rate": 0.577,
        "quality_ready_rate": 0.546,
        "first_speech_hit_rate": 0.217,
        "average_quality_score": 0.974,
        "segments": {
            "expected_response_mode": {
                "disclose": {"prepared_hit_rate": 0.0},
                "inform": {"prepared_hit_rate": 0.0},
            }
        },
    }
    broad_policy = {
        "name": "recover_disclose_inform",
        "target_modes": ["disclose", "inform"],
        "quality_floor": 0.974,
    }
    safe_policy = {
        "name": "recover_inform",
        "target_modes": ["inform"],
        "quality_floor": 0.974,
    }
    candidate_replays = {
        "recover_disclose_inform": {
            "prepared_hit_rate": 0.843,
            "quality_ready_rate": 0.765,
            "first_speech_hit_rate": 0.217,
            "average_quality_score": 0.955,
            "segments": {
                "expected_response_mode": {
                    "disclose": {"prepared_hit_rate": 0.858},
                    "inform": {"prepared_hit_rate": 0.812},
                }
            },
        },
        "recover_inform": {
            "prepared_hit_rate": 0.701,
            "quality_ready_rate": 0.681,
            "first_speech_hit_rate": 0.217,
            "average_quality_score": 0.976,
            "segments": {
                "expected_response_mode": {
                    "disclose": {"prepared_hit_rate": 0.0},
                    "inform": {"prepared_hit_rate": 0.812},
                }
            },
        },
    }

    selection = select_response_mode_background_recovery_candidate(
        baseline,
        candidate_replays=candidate_replays,
        candidate_policies=(broad_policy, safe_policy),
    )

    assert selection["promoted"] is True
    assert selection["selected_policy"]["name"] == "recover_inform"
    assert selection["selected_evaluation"]["quality_floor_met"] is True
    assert (
        selection["candidate_evaluations"]["recover_disclose_inform"][
            "quality_floor_met"
        ]
        is False
    )


def test_response_mode_recovery_policy_for_replay_uses_local_baseline_quality_floor():
    selected_policy = {
        "name": "recover_other",
        "target_modes": ["other"],
        "mode_variants": {"other": "protected_minority_specialist_coverage_low_margin"},
        "quality_floor": 0.98,
        "first_speech_locked": True,
    }
    test_baseline = {
        "average_quality_score": 0.974,
    }

    adjusted_policy = response_mode_recovery_policy_for_replay(
        selected_policy,
        test_baseline,
    )

    assert adjusted_policy["name"] == "recover_other"
    assert adjusted_policy["target_modes"] == ["other"]
    assert adjusted_policy["quality_floor"] == 0.974
    assert selected_policy["quality_floor"] == 0.98


def test_balanced_response_mode_brancher_adds_minority_mode_to_top_three():
    train_turns = (
        conversation_turn(
            "train-disclose-1",
            "storycue similar experience",
            "inform",
            expected_response_mode="disclose",
        ),
        conversation_turn(
            "train-disclose-2",
            "storycue shared experience",
            "inform",
            expected_response_mode="disclose",
        ),
        conversation_turn(
            "train-ask",
            "unclear details",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    ranker = train_response_mode_ranker(train_turns)
    turn = conversation_turn(
        "test-disclose",
        "storycue relatable moment",
        "inform",
        expected_response_mode="disclose",
    )

    branches = generate_response_mode_branches(
        turn,
        top_k=3,
        ranker=ranker,
        learned_weight=0.5,
        coverage_modes=("disclose",),
        coverage_min_score=0.0,
        scoring_variant="balanced_response_mode_50",
    )

    assert "disclose" in [branch["response_mode"] for branch in branches]


def test_balanced_response_mode_brancher_preserves_strong_top_mode():
    train_turns = (
        conversation_turn(
            "train-disclose",
            "storycue similar experience",
            "inform",
            expected_response_mode="disclose",
        ),
        conversation_turn(
            "train-ask-1",
            "unclear details?",
            "question",
            expected_response_mode="ask_followup",
        ),
        conversation_turn(
            "train-ask-2",
            "what happened next?",
            "question",
            expected_response_mode="ask_followup",
        ),
    )
    ranker = train_response_mode_ranker(train_turns)
    turn = conversation_turn(
        "test-ask",
        "what happened next?",
        "question",
        expected_response_mode="ask_followup",
    )

    branches = generate_response_mode_branches(
        turn,
        top_k=3,
        ranker=ranker,
        learned_weight=0.5,
        coverage_modes=("disclose",),
        coverage_min_score=0.0,
        scoring_variant="balanced_response_mode_50",
    )

    assert branches[0]["response_mode"] == "ask_followup"


def test_response_mode_bakeoff_variants_include_balanced_coverage():
    variants = {
        str(variant["name"]): variant
        for variant in response_mode_bakeoff_variants()
    }

    assert variants["balanced_response_mode_50"]["coverage_modes"] == (
        "disclose",
        "inform",
        "other",
        "reassure",
    )
    assert variants["balanced_response_mode_50"]["coverage_min_score"] > 0
    assert variants["balanced_prior_response_mode_75"]["class_balanced_prior"] is True


def test_build_probability_pack_prepares_speakable_tts_drafts():
    turns = load_conversation_turns(Path("data/human_conversation_sample.jsonl"))

    pack = build_probability_pack(turns[0], top_k=3)

    assert pack.turn_id == "hc-001"
    assert len(pack.top_branches) == 3
    assert pack.top_branches[0]["act"] == "question"
    assert pack.prepared_drafts[0]["voice_ready"] is True
    assert pack.prepared_drafts[0]["tts_text"]
    assert pack.confirmation_mode == "wait_for_observed_next_move"
    assert pack.expires_after_ms == 2500


def test_conversation_probability_loop_improves_held_out_readiness():
    turns = load_conversation_turns(Path("data/human_conversation_sample.jsonl"))

    report = run_conversation_probability_loop(turns, iterations=3, top_k=3)

    assert report["summary"]["total_turns"] == len(turns)
    assert len(report["iterations"]) == 3
    assert report["iterations"][0]["metrics"]["p_at_1"] < report["iterations"][-1]["metrics"]["p_at_1"]
    assert report["iterations"][-1]["metrics"]["p_at_1"] >= 0.75
    assert report["iterations"][-1]["metrics"]["top_3_recall"] == 1.0
    assert report["iterations"][-1]["metrics"]["tts_readiness_rate"] >= 0.75
    assert "directive" in report["final_guidance"]["act_keywords"]
    learned_tokens = {
        token
        for tokens in report["final_guidance"]["act_keywords"].values()
        for token in tokens
    }
    assert "speaker" not in learned_tokens
    assert "that" not in learned_tokens


def test_conversation_probability_loop_rejects_regressing_guidance(tmp_path):
    input_path = tmp_path / "conversation.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "turn_id": f"ambiguous-{index}",
                        "conversation": [
                            {
                                "role": "speaker_a",
                                "content": f"agenda topic {index} repeats the shared cue",
                            }
                        ],
                        "next_speaker": "speaker_b",
                        "actual_next_utterance": "Tell me more.",
                        "expected_act": "question" if index == 6 else "inform",
                        "expected_emotion": "no_emotion",
                        "latency_budget_ms": 650,
                    }
                )
                for index in range(1, 7)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    turns = load_conversation_turns(input_path)

    report = run_conversation_probability_loop(turns, iterations=3, top_k=3)
    metrics = [iteration["metrics"]["p_at_1"] for iteration in report["iterations"]]

    assert metrics == sorted(metrics)
    assert report["iterations"][0]["guidance_promoted"] is False


def test_act_specific_learning_ignores_shared_tokens():
    turns = (
        conversation_turn(
            "question-miss",
            "sharedcue lanterncue mysterycue",
            expected_act="question",
        ),
        conversation_turn(
            "inform-context-1",
            "sharedcue ledgercue finished",
            expected_act="inform",
        ),
        conversation_turn(
            "inform-context-2",
            "sharedcue plancue happy",
            expected_act="inform",
        ),
    )
    rows = (
        {
            "turn_id": "question-miss",
            "rank_1_act": "inform",
            "expected_act": "question",
        },
    )

    guidance = learn_conversation_guidance(
        turns=turns,
        rows=rows,
        prior=ConversationGuidance(act_keywords={}),
    )

    assert "lanterncue" in guidance.keywords_for("question")
    assert "sharedcue" not in guidance.keywords_for("question")


def test_learned_act_ranker_predicts_repeated_question_cue():
    train_turns = (
        conversation_turn("q1", "lanterncue mystery details", "question"),
        conversation_turn("q2", "lanterncue unclear plan", "question"),
        conversation_turn("i1", "finished report happy", "inform"),
        conversation_turn("i2", "finished ledger summary", "inform"),
    )
    ranker = train_conversation_act_ranker(train_turns)
    test_turn = conversation_turn("test-question", "lanterncue unclear topic", "question")

    branches = generate_conversation_branches(
        test_turn,
        top_k=3,
        act_ranker=ranker,
        learned_weight=1.0,
    )

    assert branches[0]["act"] == "question"
    assert branches[0]["scoring_variant"] == "learned"


def test_conversation_act_ranker_bakeoff_selects_on_dev_and_scores_test():
    train_turns = (
        conversation_turn("train-q1", "lanterncue mystery details", "question"),
        conversation_turn("train-q2", "lanterncue unclear plan", "question"),
        conversation_turn("train-i1", "finished report happy", "inform"),
        conversation_turn("train-i2", "finished ledger summary", "inform"),
    )
    dev_turns = (
        conversation_turn("dev-q", "lanterncue unclear topic", "question"),
        conversation_turn("dev-i", "finished report today", "inform"),
    )
    test_turns = (
        conversation_turn("test-q", "lanterncue mystery topic", "question"),
        conversation_turn("test-i", "finished ledger today", "inform"),
    )

    report = run_conversation_act_ranker_bakeoff(
        train_turns=train_turns,
        dev_turns=dev_turns,
        test_turns=test_turns,
        top_k=3,
    )

    assert report["summary"]["train_turns"] == 4
    assert "heuristic" in report["variants"]
    assert "learned" in report["variants"]
    assert "contextual_inform_overlay" in report["variants"]
    assert report["selected_variant"]["name"] in report["variants"]
    assert report["selected_variant"]["dev"]["p_at_1"] == 1.0
    assert report["selected_variant"]["test"]["p_at_1"] == 1.0
    assert "cross_validation" in report["selected_variant"]
    assert report["test"]["guided"]["p_at_1"] > report["test"]["baseline"]["p_at_1"]
    assert report["guidance_delta"]["regressed_turns"] == []
    assert "expected_act" in report["analytics"]["test_segments"]


def test_conversation_act_ranker_bakeoff_handles_single_train_turn():
    report = run_conversation_act_ranker_bakeoff(
        train_turns=(conversation_turn("train-q", "lanterncue details", "question"),),
        dev_turns=(conversation_turn("dev-q", "lanterncue topic", "question"),),
        test_turns=(conversation_turn("test-q", "lanterncue mystery", "question"),),
        top_k=3,
    )

    assert report["selected_variant"]["cross_validation"]["fold_count"] == 0


def test_transition_ranker_uses_observed_act_history():
    train_turns = (
        conversation_turn("train-q-i-1", "Anything else?", "inform", observed_acts=("question",)),
        conversation_turn("train-q-i-2", "Where is it?", "inform", observed_acts=("question",)),
        conversation_turn("train-i-q-1", "The package arrived.", "question", observed_acts=("inform",)),
        conversation_turn("train-i-q-2", "The report is done.", "question", observed_acts=("inform",)),
    )
    ranker = train_conversation_transition_ranker(train_turns)
    turn = conversation_turn("test", "Could you clarify?", "inform", observed_acts=("question",))

    branches = generate_conversation_branches(
        turn,
        top_k=3,
        transition_ranker=ranker,
        transition_weight=1.0,
    )

    assert branches[0]["act"] == "inform"
    assert branches[0]["scoring_variant"] == "contextual_transition"


def test_guarded_transition_preserves_protected_heuristic_act():
    train_turns = (
        conversation_turn("train-q-i-1", "Anything else?", "inform", observed_acts=("question",)),
        conversation_turn("train-q-i-2", "Where is it?", "inform", observed_acts=("question",)),
        conversation_turn("train-q-i-3", "Can you explain?", "inform", observed_acts=("question",)),
        conversation_turn("train-i-q-1", "The package arrived.", "question", observed_acts=("inform",)),
    )
    ranker = train_conversation_transition_ranker(train_turns)
    turn = conversation_turn(
        "test-directive",
        "What should we do next?",
        "directive",
        observed_acts=("question",),
    )

    unguarded = generate_conversation_branches(
        turn,
        top_k=3,
        transition_ranker=ranker,
        transition_weight=1.0,
    )
    guarded = generate_conversation_branches(
        turn,
        top_k=3,
        transition_ranker=ranker,
        transition_weight=1.0,
        transition_protected_acts=("directive", "question"),
        scoring_variant="guarded_contextual_transition",
    )

    assert unguarded[0]["act"] == "inform"
    assert guarded[0]["act"] == "directive"
    assert guarded[0]["scoring_variant"] == "guarded_contextual_transition"


def test_history_ranker_promotes_repeated_act_rhythm():
    train_turns = (
        conversation_turn(
            "train-history-q-1",
            "The agenda is ready.",
            "question",
            observed_acts=("inform", "question", "inform", "directive"),
        ),
        conversation_turn(
            "train-history-q-2",
            "The checklist is ready.",
            "question",
            observed_acts=("inform", "question", "inform", "directive"),
        ),
        conversation_turn(
            "train-prev-directive-i-1",
            "Anything else?",
            "inform",
            observed_acts=("commissive", "inform", "directive"),
        ),
        conversation_turn(
            "train-prev-directive-i-2",
            "Can you continue?",
            "inform",
            observed_acts=("question", "inform", "directive"),
        ),
    )
    transition_ranker = train_conversation_transition_ranker(train_turns)
    history_ranker = train_conversation_history_ranker(train_turns, window_size=4)
    turn = conversation_turn(
        "test-history-question",
        "The agenda is ready.",
        "question",
        observed_acts=("inform", "question", "inform", "directive"),
    )

    transition = generate_conversation_branches(
        turn,
        top_k=3,
        transition_ranker=transition_ranker,
        transition_weight=1.0,
    )
    history = generate_conversation_branches(
        turn,
        top_k=3,
        transition_ranker=transition_ranker,
        transition_weight=1.0,
        history_ranker=history_ranker,
        history_margin=0.25,
        scoring_variant="act_rhythm_contextual",
    )

    assert transition[0]["act"] == "inform"
    assert history[0]["act"] == "question"
    assert history[0]["scoring_variant"] == "act_rhythm_contextual"


def test_history_ranker_only_promotes_allowed_specialist_acts():
    train_turns = (
        conversation_turn(
            "train-history-inform-1",
            "Anything else?",
            "inform",
            observed_acts=("question", "directive"),
        ),
        conversation_turn(
            "train-history-inform-2",
            "Where next?",
            "inform",
            observed_acts=("question", "directive"),
        ),
        conversation_turn(
            "train-history-question",
            "The route changed.",
            "question",
            observed_acts=("inform", "commissive"),
        ),
    )
    history_ranker = train_conversation_history_ranker(train_turns, window_size=2)
    turn = conversation_turn(
        "test-history-guard",
        "Why did anyone move?",
        "question",
        observed_acts=("question", "directive"),
    )

    branches = generate_conversation_branches(
        turn,
        top_k=3,
        history_ranker=history_ranker,
        history_margin=0.0,
        history_overlay_acts=("directive", "question"),
        scoring_variant="protected_history",
    )

    assert branches[0]["act"] == "question"


def test_history_ranker_promotes_allowed_question_specialist():
    train_turns = (
        conversation_turn(
            "train-history-question-1",
            "The package arrived.",
            "question",
            observed_acts=("inform", "directive"),
        ),
        conversation_turn(
            "train-history-question-2",
            "The route changed.",
            "question",
            observed_acts=("inform", "directive"),
        ),
        conversation_turn(
            "train-history-inform",
            "Where next?",
            "inform",
            observed_acts=("question", "commissive"),
        ),
    )
    history_ranker = train_conversation_history_ranker(train_turns, window_size=2)
    turn = conversation_turn(
        "test-history-question",
        "The plan is ready.",
        "question",
        observed_acts=("inform", "directive"),
    )

    branches = generate_conversation_branches(
        turn,
        top_k=3,
        history_ranker=history_ranker,
        history_margin=0.0,
        history_overlay_acts=("question",),
        scoring_variant="question_history_specialist",
    )

    assert branches[0]["act"] == "question"


def test_question_specialist_preserves_current_directive_read():
    train_turns = (
        conversation_turn(
            "train-history-question-1",
            "The package arrived.",
            "question",
            observed_acts=("inform", "directive"),
        ),
        conversation_turn(
            "train-history-question-2",
            "The route changed.",
            "question",
            observed_acts=("inform", "directive"),
        ),
        conversation_turn(
            "train-history-inform",
            "Anything else?",
            "inform",
            observed_acts=("question", "commissive"),
        ),
    )
    history_ranker = train_conversation_history_ranker(train_turns, window_size=2)
    turn = conversation_turn(
        "test-question-preserve-directive",
        "What should we do next?",
        "directive",
        observed_acts=("inform", "directive"),
    )

    branches = generate_conversation_branches(
        turn,
        top_k=3,
        history_ranker=history_ranker,
        history_margin=0.0,
        history_overlay_acts=("question",),
        history_preserved_acts=("directive",),
        scoring_variant="safe_question_history_specialist",
    )

    assert branches[0]["act"] == "directive"


def test_question_evidence_ranker_promotes_question_language_beyond_history():
    train_turns = (
        conversation_turn(
            "train-question-1",
            "The curiouscue detail is unresolved.",
            "question",
            observed_acts=("directive",),
        ),
        conversation_turn(
            "train-question-2",
            "That curiouscue plan still feels unclear.",
            "question",
            observed_acts=("directive",),
        ),
        conversation_turn(
            "train-inform-1",
            "Anything else?",
            "inform",
            observed_acts=("directive",),
        ),
        conversation_turn(
            "train-inform-2",
            "Where next?",
            "inform",
            observed_acts=("directive",),
        ),
    )
    transition_ranker = train_conversation_transition_ranker(train_turns)
    question_evidence_ranker = train_conversation_question_evidence_ranker(train_turns)
    turn = conversation_turn(
        "test-question-evidence",
        "The curiouscue result is still unclear.",
        "question",
        observed_acts=("directive",),
    )

    without_evidence = generate_conversation_branches(
        turn,
        top_k=3,
        transition_ranker=transition_ranker,
        transition_weight=1.0,
    )
    with_evidence = generate_conversation_branches(
        turn,
        top_k=3,
        transition_ranker=transition_ranker,
        transition_weight=1.0,
        question_evidence_ranker=question_evidence_ranker,
        question_evidence_margin=0.0,
        scoring_variant="question_evidence_contextual",
    )

    assert without_evidence[0]["act"] == "inform"
    assert with_evidence[0]["act"] == "question"
    assert with_evidence[0]["scoring_variant"] == "question_evidence_contextual"


def test_question_evidence_overlay_preserves_current_directive_read():
    train_turns = (
        conversation_turn("train-question-1", "curiouscue unresolved", "question"),
        conversation_turn("train-question-2", "curiouscue unclear", "question"),
        conversation_turn("train-inform", "finished report", "inform"),
    )
    question_evidence_ranker = train_conversation_question_evidence_ranker(train_turns)
    turn = conversation_turn(
        "test-directive-preserved",
        "What should we do about curiouscue?",
        "directive",
    )

    branches = generate_conversation_branches(
        turn,
        top_k=3,
        question_evidence_ranker=question_evidence_ranker,
        question_evidence_margin=0.0,
        question_evidence_preserved_acts=("directive",),
        scoring_variant="safe_question_evidence",
    )

    assert branches[0]["act"] == "directive"


def test_bakeoff_variants_include_protected_act_specialists():
    variants = {
        str(variant["name"]): variant
        for variant in conversation_bakeoff_variants()
    }

    assert variants["protected_act_rhythm_contextual"]["history_overlay_acts"] == (
        "directive",
        "question",
    )
    assert variants["question_act_rhythm_contextual"]["history_overlay_acts"] == (
        "question",
    )
    assert variants["directive_act_rhythm_contextual"]["history_overlay_acts"] == (
        "directive",
    )
    assert variants["safe_question_act_rhythm_contextual"]["history_overlay_acts"] == (
        "question",
    )
    assert variants["safe_question_act_rhythm_contextual"]["history_preserved_acts"] == (
        "directive",
    )
    assert variants["safe_question_evidence_act_rhythm_contextual"][
        "use_question_evidence_ranker"
    ] is True
    assert variants["safe_question_evidence_act_rhythm_contextual"][
        "question_evidence_preserved_acts"
    ] == ("directive",)
    assert variants["deep_protected_act_rhythm_contextual"]["history_window_size"] == 8


def test_variant_fold_scoring_uses_deeper_history_window():
    turns = (
        conversation_turn(
            "validation-question",
            "The plain status changed.",
            "question",
            observed_acts=("commissive", "inform", "question", "inform", "directive"),
        ),
        conversation_turn(
            "train-question",
            "The plain status changed.",
            "question",
            observed_acts=("commissive", "inform", "question", "inform", "directive"),
        ),
        conversation_turn(
            "validation-inform",
            "The plain status changed.",
            "inform",
            observed_acts=("directive", "inform", "question", "inform", "directive"),
        ),
        conversation_turn(
            "train-inform",
            "The plain status changed.",
            "inform",
            observed_acts=("directive", "inform", "question", "inform", "directive"),
        ),
    )
    variant = {
        "name": "deep_question_window",
        "learned_weight": 0.0,
        "transition_weight": 0.0,
        "use_history_ranker": True,
        "history_window_size": 5,
        "history_margin": 0.0,
        "history_overlay_acts": ("question",),
    }

    fold = score_conversation_variant_fold(
        turns=turns,
        variant=variant,
        fold_index=0,
        fold_count=2,
        top_k=3,
    )

    assert fold["candidate"]["p_at_1"] > fold["baseline"]["p_at_1"]


def test_conversation_act_ranker_bakeoff_includes_contextual_transition_variant():
    train_turns = (
        conversation_turn("train-q-i-1", "Anything else?", "inform", observed_acts=("question",)),
        conversation_turn("train-q-i-2", "Where is it?", "inform", observed_acts=("question",)),
        conversation_turn("train-i-q-1", "The package arrived.", "question", observed_acts=("inform",)),
        conversation_turn("train-i-q-2", "The report is done.", "question", observed_acts=("inform",)),
    )
    dev_turns = (
        conversation_turn("dev-q-i", "Could you clarify?", "inform", observed_acts=("question",)),
        conversation_turn("dev-i-q", "The invoice is ready.", "question", observed_acts=("inform",)),
    )
    test_turns = (
        conversation_turn("test-q-i", "Can you explain?", "inform", observed_acts=("question",)),
        conversation_turn("test-i-q", "The route changed.", "question", observed_acts=("inform",)),
    )

    report = run_conversation_act_ranker_bakeoff(
        train_turns=train_turns,
        dev_turns=dev_turns,
        test_turns=test_turns,
        top_k=3,
    )

    assert "contextual_transition" in report["variants"]
    assert "guarded_contextual_transition" in report["variants"]
    assert "act_rhythm_contextual" in report["variants"]
    assert "contextual_inform_overlay" in report["variants"]
    assert report["selected_variant"]["name"] in report["variants"]
    assert report["selected_variant"]["test"]["p_at_1"] == 1.0


def test_bakeoff_selection_rejects_large_train_dev_gap():
    variants = {
        "overfit_dev_winner": {
            "learned_weight": 0.0,
            "train": {"p_at_1": 0.9, "top_3_recall": 1.0},
            "dev": {"p_at_1": 0.6, "top_3_recall": 1.0},
            "dev_segment_regressions": [],
        },
        "robust_candidate": {
            "learned_weight": 0.0,
            "train": {"p_at_1": 0.58, "top_3_recall": 1.0},
            "dev": {"p_at_1": 0.56, "top_3_recall": 1.0},
            "dev_segment_regressions": [],
        },
    }

    selected = select_conversation_bakeoff_variant(variants)

    assert selected == "robust_candidate"


def test_cross_validate_conversation_variant_reports_fold_stability():
    turns = (
        conversation_turn(
            "fold-a-1",
            "The agenda is ready.",
            "question",
            observed_acts=("inform", "question", "inform", "directive"),
        ),
        conversation_turn(
            "fold-a-2",
            "The checklist is ready.",
            "question",
            observed_acts=("inform", "question", "inform", "directive"),
        ),
        conversation_turn(
            "fold-b-1",
            "Anything else?",
            "inform",
            observed_acts=("commissive", "inform", "directive"),
        ),
        conversation_turn(
            "fold-b-2",
            "Can you continue?",
            "inform",
            observed_acts=("question", "inform", "directive"),
        ),
        conversation_turn(
            "fold-c-1",
            "The package arrived.",
            "question",
            observed_acts=("inform",),
        ),
        conversation_turn(
            "fold-c-2",
            "The package arrived.",
            "question",
            observed_acts=("inform",),
        ),
    )
    variant = next(
        row
        for row in conversation_bakeoff_variants()
        if row["name"] == "act_rhythm_contextual_strict"
    )

    report = cross_validate_conversation_variant(
        turns,
        variant,
        fold_count=3,
        top_k=3,
    )

    assert report["fold_count"] == 3
    assert len(report["folds"]) == 3
    assert "mean_p_at_1_gain" in report
    assert "min_p_at_1_gain" in report
    assert "segment_regression_count" in report


def test_bakeoff_selection_prefers_cross_validated_stability():
    variants = {
        "unstable_dev_winner": {
            "learned_weight": 0.0,
            "train": {"p_at_1": 0.58, "top_3_recall": 1.0},
            "dev": {"p_at_1": 0.6, "top_3_recall": 1.0},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.04,
                "min_p_at_1_gain": -0.1,
                "segment_regression_count": 0,
            },
        },
        "stable_candidate": {
            "learned_weight": 0.0,
            "train": {"p_at_1": 0.57, "top_3_recall": 1.0},
            "dev": {"p_at_1": 0.56, "top_3_recall": 1.0},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.02,
                "min_p_at_1_gain": 0.0,
                "segment_regression_count": 0,
            },
        },
    }

    selected = select_conversation_bakeoff_variant(variants)

    assert selected == "stable_candidate"


def test_bakeoff_selection_allows_small_gap_for_cross_validated_specialist():
    variants = {
        "guarded_candidate": {
            "learned_weight": 0.0,
            "train": {"p_at_1": 0.66, "top_3_recall": 1.0},
            "dev": {"p_at_1": 0.55, "top_3_recall": 1.0},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.04,
                "min_p_at_1_gain": 0.02,
                "segment_regression_count": 0,
            },
        },
        "stable_specialist": {
            "learned_weight": 0.0,
            "history_overlay_acts": ("directive",),
            "train": {"p_at_1": 0.684, "top_3_recall": 1.0},
            "dev": {"p_at_1": 0.56, "top_3_recall": 1.0},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.05,
                "min_p_at_1_gain": 0.02,
                "segment_regression_count": 0,
            },
        },
        "unstable_specialist": {
            "learned_weight": 0.0,
            "history_overlay_acts": ("question",),
            "train": {"p_at_1": 0.69, "top_3_recall": 1.0},
            "dev": {"p_at_1": 0.57, "top_3_recall": 1.0},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.05,
                "min_p_at_1_gain": 0.02,
                "segment_regression_count": 1,
            },
        },
    }

    selected = select_conversation_bakeoff_variant(variants)

    assert selected == "stable_specialist"


def test_bakeoff_selection_prefers_preserved_specialist_inside_dev_tie():
    variants = {
        "question_specialist": {
            "learned_weight": 0.0,
            "history_overlay_acts": ("question",),
            "train": {"p_at_1": 0.60, "top_3_recall": 0.99},
            "dev": {"p_at_1": 0.490, "top_3_recall": 0.973},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.094,
                "min_p_at_1_gain": 0.078,
                "segment_regression_count": 0,
            },
        },
        "safe_question_specialist": {
            "learned_weight": 0.0,
            "history_overlay_acts": ("question",),
            "history_preserved_acts": ("directive",),
            "train": {"p_at_1": 0.598, "top_3_recall": 0.99},
            "dev": {"p_at_1": 0.488, "top_3_recall": 0.973},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.092,
                "min_p_at_1_gain": 0.078,
                "segment_regression_count": 0,
            },
        },
    }

    selected = select_conversation_bakeoff_variant(variants)

    assert selected == "safe_question_specialist"


def test_bakeoff_selection_keeps_stronger_stable_protected_specialist():
    variants = {
        "protected_act_specialist": {
            "learned_weight": 0.0,
            "history_overlay_acts": ("directive", "question"),
            "train": {"p_at_1": 0.61, "top_3_recall": 0.99},
            "dev": {"p_at_1": 0.501, "top_3_recall": 0.977},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.108,
                "min_p_at_1_gain": 0.099,
                "segment_regression_count": 0,
            },
        },
        "safe_question_specialist": {
            "learned_weight": 0.0,
            "history_overlay_acts": ("question",),
            "history_preserved_acts": ("directive",),
            "train": {"p_at_1": 0.60, "top_3_recall": 0.99},
            "dev": {"p_at_1": 0.498, "top_3_recall": 0.977},
            "dev_segment_regressions": [],
            "cross_validation": {
                "mean_p_at_1_gain": 0.107,
                "min_p_at_1_gain": 0.098,
                "segment_regression_count": 0,
            },
        },
    }

    selected = select_conversation_bakeoff_variant(variants)

    assert selected == "protected_act_specialist"


def test_conversation_train_dev_test_loop_blocks_act_segment_regression():
    train_turns = (
        conversation_turn(
            "train-question",
            "lanterncue mysterycue aftercue",
            expected_act="question",
        ),
    )
    dev_turns = (
        conversation_turn(
            "dev-question-1",
            "lanterncue mysterycue aftercue",
            expected_act="question",
        ),
        conversation_turn(
            "dev-question-2",
            "lanterncue mysterycue aftercue",
            expected_act="question",
        ),
        conversation_turn(
            "dev-commissive",
            "please help with lanterncue mysterycue aftercue",
            expected_act="commissive",
        ),
    )
    test_turns = (
        conversation_turn(
            "test-question",
            "lanterncue mysterycue aftercue",
            expected_act="question",
        ),
    )

    report = run_conversation_train_dev_test_loop(
        train_turns=train_turns,
        dev_turns=dev_turns,
        test_turns=test_turns,
        iterations=2,
        top_k=3,
    )

    iteration = report["iterations"][0]
    assert iteration["dev"]["candidate"]["p_at_1"] > iteration["dev"]["selected"]["p_at_1"]
    assert iteration["dev_promote_guidance"] is False
    assert iteration["dev_segment_regressions"] == [
        {
            "segment": "expected_act",
            "name": "commissive",
            "baseline_p_at_1": 1.0,
            "candidate_p_at_1": 0.0,
            "p_at_1_delta": -1.0,
        }
    ]
    assert report["test"]["guided"] == report["test"]["baseline"]


def test_conversation_train_dev_test_loop_promotes_on_dev_and_scores_untouched_test():
    train_turns = (
        conversation_turn(
            "train-question",
            "The harbor lantern clue feels unresolved.",
            expected_act="question",
        ),
        conversation_turn(
            "train-inform",
            "I finally finished the report.",
            expected_act="inform",
        ),
    )
    dev_turns = (
        conversation_turn(
            "dev-question",
            "That harbor lantern clue is still unclear.",
            expected_act="question",
        ),
        conversation_turn(
            "dev-inform",
            "I am happy with the plan.",
            expected_act="inform",
        ),
    )
    test_turns = (
        conversation_turn(
            "test-question",
            "The harbor lantern detail needs another look.",
            expected_act="question",
        ),
        conversation_turn(
            "test-directive",
            "We should meet after lunch.",
            expected_act="directive",
        ),
    )

    report = run_conversation_train_dev_test_loop(
        train_turns=train_turns,
        dev_turns=dev_turns,
        test_turns=test_turns,
        iterations=2,
        top_k=3,
    )

    assert report["summary"]["train_turns"] == 2
    assert report["summary"]["dev_turns"] == 2
    assert report["summary"]["test_turns"] == 2
    assert report["iterations"][0]["dev_promote_guidance"] is True
    assert report["test"]["guided"]["p_at_1"] > report["test"]["baseline"]["p_at_1"]
    assert report["efficacy"]["test_p_at_1_gain"] > 0
    assert report["guidance_delta"]["improved_turns"] == ["test-question"]
    assert report["guidance_delta"]["regressed_turns"] == []
    assert report["analytics"]["test_segments"]["expected_act"]["question"]["guided"]["p_at_1"] == 1.0


def test_conversation_train_dev_test_loop_rejects_validation_regression():
    train_turns = (
        conversation_turn(
            "train-question",
            "The harbor lantern clue feels unresolved.",
            expected_act="question",
        ),
    )
    dev_turns = (
        conversation_turn(
            "dev-inform",
            "The harbor lantern clue was documented yesterday.",
            expected_act="inform",
        ),
    )
    test_turns = (
        conversation_turn(
            "test-inform",
            "The harbor lantern clue is already filed.",
            expected_act="inform",
        ),
    )

    report = run_conversation_train_dev_test_loop(
        train_turns=train_turns,
        dev_turns=dev_turns,
        test_turns=test_turns,
        iterations=2,
        top_k=3,
    )

    assert report["iterations"][0]["dev_promote_guidance"] is False
    assert report["test"]["guided"] == report["test"]["baseline"]
    assert report["efficacy"]["test_p_at_1_gain"] == 0
    assert report["guidance_delta"]["regressed_turns"] == []


def test_conversation_train_dev_test_loop_rejects_validation_noop():
    train_turns = (
        conversation_turn(
            "train-question",
            "The harbor lantern clue feels unresolved.",
            expected_act="question",
        ),
        conversation_turn(
            "train-inform-1",
            "The harbor lantern clue was documented yesterday.",
            expected_act="inform",
        ),
        conversation_turn(
            "train-inform-2",
            "The harbor lantern clue is already filed.",
            expected_act="inform",
        ),
    )
    dev_turns = (
        conversation_turn(
            "dev-question",
            "That harbor lantern clue is still unclear.",
            expected_act="question",
        ),
    )
    test_turns = (
        conversation_turn(
            "test-question",
            "The harbor lantern detail needs another look.",
            expected_act="question",
        ),
    )

    report = run_conversation_train_dev_test_loop(
        train_turns=train_turns,
        dev_turns=dev_turns,
        test_turns=test_turns,
        iterations=2,
        top_k=3,
    )

    assert report["iterations"][0]["dev"]["candidate"]["p_at_1"] == report["iterations"][0]["dev"]["selected"]["p_at_1"]
    assert report["iterations"][0]["dev_promote_guidance"] is False
    assert report["efficacy"]["test_p_at_1_gain"] == 0


def test_cli_runs_conversation_probability_loop(tmp_path):
    output = tmp_path / "conversation-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--conversation-input",
            "data/human_conversation_sample.jsonl",
            "--iterations",
            "3",
            "--conversation-report",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == report
    assert report["iterations"][-1]["metrics"]["p_at_1"] >= 0.75


def test_cli_runs_conversation_train_dev_test_loop(tmp_path):
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    test_path = tmp_path / "test.jsonl"
    output = tmp_path / "conversation-heldout-report.json"
    write_conversation_turns(
        (
            conversation_turn(
                "train-question",
                "The harbor lantern clue feels unresolved.",
                expected_act="question",
            ),
        ),
        train_path,
    )
    write_conversation_turns(
        (
            conversation_turn(
                "dev-question",
                "That harbor lantern clue is still unclear.",
                expected_act="question",
            ),
        ),
        dev_path,
    )
    write_conversation_turns(
        (
            conversation_turn(
                "test-question",
                "The harbor lantern detail needs another look.",
                expected_act="question",
            ),
        ),
        test_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--conversation-train-input",
            str(train_path),
            "--conversation-dev-input",
            str(dev_path),
            "--conversation-test-input",
            str(test_path),
            "--iterations",
            "2",
            "--conversation-report",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == report
    assert report["efficacy"]["test_p_at_1_gain"] > 0


def test_cli_runs_conversation_act_ranker_bakeoff(tmp_path):
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    test_path = tmp_path / "test.jsonl"
    output = tmp_path / "conversation-bakeoff-report.json"
    write_conversation_turns(
        (
            conversation_turn("train-q1", "lanterncue mystery details", "question"),
            conversation_turn("train-q2", "lanterncue unclear plan", "question"),
            conversation_turn("train-i1", "finished report happy", "inform"),
        ),
        train_path,
    )
    write_conversation_turns(
        (
            conversation_turn("dev-q", "lanterncue unclear topic", "question"),
            conversation_turn("dev-i", "finished report today", "inform"),
        ),
        dev_path,
    )
    write_conversation_turns(
        (
            conversation_turn("test-q", "lanterncue mystery topic", "question"),
            conversation_turn("test-i", "finished ledger today", "inform"),
        ),
        test_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--conversation-train-input",
            str(train_path),
            "--conversation-dev-input",
            str(dev_path),
            "--conversation-test-input",
            str(test_path),
            "--conversation-bakeoff-report",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == report
    assert report["selected_variant"]["test"]["p_at_1"] == 1.0


def test_cli_exports_dailydialog_sample(tmp_path):
    split_dir = tmp_path / "validation"
    split_dir.mkdir()
    (split_dir / "dialogues.txt").write_text(
        "I cannot find my keys.__eou__ Where did you last see them?__eou__ Check your coat pocket first.__eou__\n",
        encoding="utf-8",
    )
    (split_dir / "dialogues_act.txt").write_text("1 2 3\n", encoding="utf-8")
    (split_dir / "dialogues_emotion.txt").write_text("0 0 0\n", encoding="utf-8")
    output = tmp_path / "dailydialog-sample.jsonl"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "foresight_harness.cli",
            "--dailydialog-dir",
            str(split_dir),
            "--conversation-output",
            str(output),
            "--conversation-limit",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    exported = load_conversation_turns(output)

    assert len(exported) == 2
    assert exported[0].expected_act == "question"
    assert exported[1].expected_act == "directive"


def conversation_turn(
    turn_id: str,
    context: str,
    expected_act: str,
    expected_emotion: str = "no_emotion",
    observed_acts: tuple[str, ...] = (),
    expected_response_mode: str = "inform",
    observed_response_modes: tuple[str, ...] = (),
    source_metadata: tuple[tuple[str, str], ...] = (),
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        conversation=(Message(role="speaker_a", content=context),),
        next_speaker="speaker_b",
        actual_next_utterance="Okay.",
        expected_act=expected_act,
        expected_emotion=expected_emotion,
        expected_response_mode=expected_response_mode,
        observed_acts=observed_acts,
        observed_response_modes=observed_response_modes,
        source_metadata=source_metadata,
    )


def response_mode_branch(
    mode: str,
    scoring_variant: str,
    probability: float,
) -> dict[str, object]:
    templates = {
        "ask_followup": "I can ask one warm follow-up question.",
        "validate": "I can reflect the feeling back clearly and gently.",
        "reassure": "I can offer grounded reassurance without overpromising.",
        "disclose": "I can share a brief relatable disclosure when it is useful.",
        "suggest": "I can offer one practical suggestion.",
        "inform": "I can provide relevant information.",
        "other": "I can keep a neutral supportive response ready.",
    }
    return {
        "branch_id": f"branch-{mode}",
        "rank": 1,
        "response_mode": mode,
        "tts_text": templates.get(mode, f"Prepared {mode} response."),
        "probability": probability,
        "trigger_cues": [],
        "scoring_variant": scoring_variant,
    }
