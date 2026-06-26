from pathlib import Path

from foresight_harness.branching import generate_branches
from foresight_harness.evaluator import load_replay_turns
from foresight_harness.guidance import Guidance, run_guidance_loop


def test_guidance_keywords_promote_expected_intent():
    turn = load_replay_turns(Path("data/queueahead_challenge.jsonl"))[0]

    baseline = generate_branches(turn, top_k=3)
    guided = generate_branches(
        turn,
        top_k=3,
        guidance=Guidance(intent_keywords={"escalation_request": ("bounced", "owns")}),
    )

    assert baseline[0].intent != "escalation_request"
    assert guided[0].intent == "escalation_request"


def test_guidance_loop_improves_p_at_1_on_challenge_split():
    turns = load_replay_turns(Path("data/queueahead_challenge.jsonl"))

    report = run_guidance_loop(turns, iterations=3, top_k=3)

    assert len(report["iterations"]) == 3
    assert report["iterations"][0]["report"]["harness"]["p_at_1"] < report["iterations"][-1]["report"]["harness"]["p_at_1"]
    assert report["iterations"][-1]["report"]["harness"]["p_at_1"] == 1.0
    assert report["iterations"][-1]["report"]["harness"]["usefulness_rate"] == 1.0
    assert report["iterations"][-1]["assessment"]["p_at_1_delta"] == 0.0
    assert report["iterations"][1]["assessment"]["p_at_1_delta"] > 0.0
    assert report["iterations"][1]["assessment"]["preparedness_delta"] > 0.0
    assert report["iterations"][-1]["assessment"]["unprepared_turns"] == []
    assert "billing_refund_timing" in report["final_guidance"]["intent_keywords"]
    assert "escalation_request" in report["final_guidance"]["intent_keywords"]
    assert report["guidance_markdown"].startswith("# Premonition Guidance")


def test_guidance_loop_filters_low_signal_guidance_tokens():
    turns = load_replay_turns(Path("data/queueahead_challenge.jsonl"))

    report = run_guidance_loop(turns, iterations=2, top_k=3)
    learned_keywords = {
        keyword
        for keywords in report["final_guidance"]["intent_keywords"].values()
        for keyword in keywords
    }

    assert "agent" not in learned_keywords
    assert "i" not in learned_keywords
    assert "be" not in learned_keywords
    assert "bounced" in report["final_guidance"]["intent_keywords"]["escalation_request"]
