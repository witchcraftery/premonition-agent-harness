from __future__ import annotations

from foresight_harness.analytics import summarize_segments
from foresight_harness.evaluator import run_replay, run_replay_turn_log
from foresight_harness.guidance import Guidance, render_guidance_markdown, run_guidance_loop
from foresight_harness.models import ReplayTurn


def run_split_benchmark(
    train_turns: tuple[ReplayTurn, ...],
    test_turns: tuple[ReplayTurn, ...],
    iterations: int,
    top_k: int = 3,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    train_loop = run_guidance_loop(train_turns, iterations=iterations, top_k=top_k)
    final_guidance = guidance_from_dict(train_loop["final_guidance"])

    train_baseline = train_loop["iterations"][0]["report"]
    train_guided = run_replay(train_turns, top_k=top_k, guidance=final_guidance)
    test_baseline = run_replay(test_turns, top_k=top_k, guidance=Guidance())
    test_guided = run_replay(test_turns, top_k=top_k, guidance=final_guidance)
    train_baseline_rows = run_replay_turn_log(train_turns, top_k=top_k, guidance=Guidance())
    train_guided_rows = run_replay_turn_log(
        train_turns,
        top_k=top_k,
        guidance=final_guidance,
    )
    test_baseline_rows = run_replay_turn_log(test_turns, top_k=top_k, guidance=Guidance())
    test_guided_rows = run_replay_turn_log(
        test_turns,
        top_k=top_k,
        guidance=final_guidance,
    )

    generalization = compare_generalization(
        train_baseline=train_baseline,
        train_guided=train_guided,
        test_baseline=test_baseline,
        test_guided=test_guided,
    )

    return {
        "train": {
            "baseline": train_baseline,
            "guided": train_guided,
        },
        "test": {
            "baseline": test_baseline,
            "guided": test_guided,
        },
        "generalization": generalization,
        "analytics": {
            "train_segments": summarize_segments(train_baseline_rows, train_guided_rows),
            "test_segments": summarize_segments(test_baseline_rows, test_guided_rows),
            "focus_areas": summarize_segments(test_baseline_rows, test_guided_rows)["focus_areas"],
        },
        "promote_guidance": should_promote(test_guided, generalization),
        "final_guidance": final_guidance.to_dict(),
        "guidance_markdown": render_guidance_markdown(final_guidance),
        "train_loop": train_loop,
    }


def guidance_from_dict(row: object) -> Guidance:
    if not isinstance(row, dict):
        return Guidance()
    intent_keywords = row.get("intent_keywords", {})
    if not isinstance(intent_keywords, dict):
        return Guidance()
    return Guidance(
        intent_keywords={
            str(intent): tuple(str(keyword) for keyword in keywords)
            for intent, keywords in intent_keywords.items()
            if isinstance(keywords, list)
        }
    )


def compare_generalization(
    train_baseline: dict[str, dict[str, float | int]],
    train_guided: dict[str, dict[str, float | int]],
    test_baseline: dict[str, dict[str, float | int]],
    test_guided: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    train_p_gain = metric_gain(train_baseline, train_guided, "p_at_1")
    test_p_gain = metric_gain(test_baseline, test_guided, "p_at_1")
    train_usefulness_gain = metric_gain(train_baseline, train_guided, "usefulness_rate")
    test_usefulness_gain = metric_gain(test_baseline, test_guided, "usefulness_rate")

    return {
        "train_p_at_1_gain": train_p_gain,
        "test_p_at_1_gain": test_p_gain,
        "overfit_gap": round(train_p_gain - test_p_gain, 3),
        "train_usefulness_gain": train_usefulness_gain,
        "test_usefulness_gain": test_usefulness_gain,
    }


def metric_gain(
    baseline: dict[str, dict[str, float | int]],
    guided: dict[str, dict[str, float | int]],
    metric: str,
) -> float:
    return round(float(guided["harness"][metric]) - float(baseline["harness"][metric]), 3)


def should_promote(
    test_guided: dict[str, dict[str, float | int]],
    generalization: dict[str, float],
) -> bool:
    return (
        generalization["test_p_at_1_gain"] > 0
        and generalization["test_usefulness_gain"] >= 0
        and float(test_guided["harness"]["unsafe_leak_rate"]) == 0.0
    )
