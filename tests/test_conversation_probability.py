import json
import subprocess
import sys
from pathlib import Path

from foresight_harness.conversation_probability import (
    build_probability_pack,
    ConversationGuidance,
    ConversationTurn,
    generate_conversation_branches,
    learn_conversation_guidance,
    load_dailydialog_split,
    load_conversation_turns,
    load_dailydialog_export,
    run_conversation_act_ranker_bakeoff,
    run_conversation_probability_loop,
    run_conversation_train_dev_test_loop,
    train_conversation_act_ranker,
    train_conversation_transition_ranker,
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
    assert report["test"]["guided"]["p_at_1"] > report["test"]["baseline"]["p_at_1"]
    assert report["guidance_delta"]["regressed_turns"] == []
    assert "expected_act" in report["analytics"]["test_segments"]


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
    assert "contextual_inform_overlay" in report["variants"]
    assert report["selected_variant"]["name"] == "contextual_transition"
    assert report["selected_variant"]["test"]["p_at_1"] == 1.0


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
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        conversation=(Message(role="speaker_a", content=context),),
        next_speaker="speaker_b",
        actual_next_utterance="Okay.",
        expected_act=expected_act,
        expected_emotion=expected_emotion,
        observed_acts=observed_acts,
    )
