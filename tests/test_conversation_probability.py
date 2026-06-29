import json
import subprocess
import sys
from pathlib import Path

from foresight_harness.conversation_probability import (
    build_probability_pack,
    ConversationGuidance,
    ConversationTurn,
    conversation_bakeoff_variants,
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
    run_conversation_act_ranker_bakeoff,
    run_conversation_probability_loop,
    run_conversation_train_dev_test_loop,
    run_response_mode_ranker_bakeoff,
    score_conversation_variant_fold,
    select_conversation_bakeoff_variant,
    train_conversation_act_ranker,
    train_conversation_history_ranker,
    train_conversation_question_evidence_ranker,
    train_conversation_transition_ranker,
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
    )
