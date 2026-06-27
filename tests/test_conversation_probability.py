import json
import subprocess
import sys
from pathlib import Path

from foresight_harness.conversation_probability import (
    build_probability_pack,
    load_conversation_turns,
    load_dailydialog_export,
    run_conversation_probability_loop,
)


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
    assert turns[1].expected_act == "inform"
    assert turns[1].expected_emotion == "happiness"


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
