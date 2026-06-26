from __future__ import annotations

from statistics import mean

from foresight_harness.analytics import summarize_segments
from foresight_harness.evaluator import run_replay, run_replay_turn_log
from foresight_harness.guidance import Guidance, render_guidance_markdown, run_guidance_loop
from foresight_harness.models import ReplayTurn
from foresight_harness.split_benchmark import (
    compare_generalization,
    guidance_from_dict,
)


def run_cross_fold_benchmark(
    turns: tuple[ReplayTurn, ...],
    fold_count: int,
    iterations: int,
    top_k: int = 3,
) -> dict[str, object]:
    if fold_count < 3:
        raise ValueError("fold_count must be at least 3")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if len(turns) < fold_count:
        raise ValueError("turn count must be at least fold_count")

    fold_reports = [
        run_train_dev_test_fold(
            turns=turns,
            fold_index=fold_index,
            fold_count=fold_count,
            iterations=iterations,
            top_k=top_k,
        )
        for fold_index in range(fold_count)
    ]

    return {
        "summary": {
            "total_turns": len(turns),
            "fold_count": fold_count,
            "iterations": iterations,
            "top_k": top_k,
        },
        "folds": fold_reports,
        "aggregates": aggregate_folds(fold_reports),
        "weak_segments": weak_segments(fold_reports),
    }


def run_train_dev_test_fold(
    turns: tuple[ReplayTurn, ...],
    fold_index: int,
    fold_count: int,
    iterations: int,
    top_k: int,
) -> dict[str, object]:
    train_turns, dev_turns, test_turns = split_train_dev_test(
        turns=turns,
        fold_index=fold_index,
        fold_count=fold_count,
    )
    train_loop = run_guidance_loop(train_turns, iterations=iterations, top_k=top_k)
    candidate_guidance = guidance_from_dict(train_loop["final_guidance"])

    train_baseline = train_loop["iterations"][0]["report"]
    train_guided = run_replay(train_turns, top_k=top_k, guidance=candidate_guidance)
    dev_baseline = run_replay(dev_turns, top_k=top_k, guidance=Guidance())
    dev_guided = run_replay(dev_turns, top_k=top_k, guidance=candidate_guidance)
    dev_generalization = compare_generalization(
        train_baseline=train_baseline,
        train_guided=train_guided,
        test_baseline=dev_baseline,
        test_guided=dev_guided,
    )
    dev_promote_guidance = should_promote_from_dev(dev_guided, dev_generalization)
    selected_guidance = candidate_guidance if dev_promote_guidance else Guidance()

    test_baseline = run_replay(test_turns, top_k=top_k, guidance=Guidance())
    test_guided = run_replay(test_turns, top_k=top_k, guidance=selected_guidance)
    test_generalization = compare_generalization(
        train_baseline=train_baseline,
        train_guided=train_guided,
        test_baseline=test_baseline,
        test_guided=test_guided,
    )

    dev_baseline_rows = run_replay_turn_log(dev_turns, top_k=top_k, guidance=Guidance())
    dev_guided_rows = run_replay_turn_log(
        dev_turns,
        top_k=top_k,
        guidance=selected_guidance,
    )
    test_baseline_rows = run_replay_turn_log(test_turns, top_k=top_k, guidance=Guidance())
    test_guided_rows = run_replay_turn_log(
        test_turns,
        top_k=top_k,
        guidance=selected_guidance,
    )

    return {
        "fold": fold_index + 1,
        "turn_ids": {
            "train": [turn.turn_id for turn in train_turns],
            "dev": [turn.turn_id for turn in dev_turns],
            "test": [turn.turn_id for turn in test_turns],
        },
        "train": {
            "baseline": train_baseline,
            "guided": train_guided,
        },
        "dev": {
            "baseline": dev_baseline,
            "guided": dev_guided,
        },
        "test": {
            "baseline": test_baseline,
            "guided": test_guided,
        },
        "dev_generalization": dev_generalization,
        "test_generalization": test_generalization,
        "dev_promote_guidance": dev_promote_guidance,
        "selected_guidance": selected_guidance.to_dict(),
        "guidance_markdown": render_guidance_markdown(selected_guidance),
        "analytics": {
            "dev_segments": summarize_segments(dev_baseline_rows, dev_guided_rows),
            "test_segments": summarize_segments(test_baseline_rows, test_guided_rows),
        },
        "train_loop": train_loop,
    }


def split_train_dev_test(
    turns: tuple[ReplayTurn, ...],
    fold_index: int,
    fold_count: int,
) -> tuple[tuple[ReplayTurn, ...], tuple[ReplayTurn, ...], tuple[ReplayTurn, ...]]:
    test_slot = fold_index
    dev_slot = (fold_index + 1) % fold_count
    train: list[ReplayTurn] = []
    dev: list[ReplayTurn] = []
    test: list[ReplayTurn] = []

    for index, turn in enumerate(turns):
        slot = index % fold_count
        if slot == test_slot:
            test.append(turn)
        elif slot == dev_slot:
            dev.append(turn)
        else:
            train.append(turn)

    return tuple(train), tuple(dev), tuple(test)


def should_promote_from_dev(
    dev_guided: dict[str, dict[str, float | int]],
    dev_generalization: dict[str, float],
) -> bool:
    return (
        dev_generalization["train_p_at_1_gain"] > 0
        and dev_generalization["test_p_at_1_gain"] >= 0
        and dev_generalization["test_usefulness_gain"] >= 0
        and float(dev_guided["harness"]["unsafe_leak_rate"]) == 0.0
    )


def aggregate_folds(folds: list[dict[str, object]]) -> dict[str, object]:
    return {
        "dev": aggregate_split_metrics(folds, "dev"),
        "test": aggregate_split_metrics(folds, "test"),
        "dev_segments": aggregate_segment_metrics(folds, "dev_segments"),
        "test_segments": aggregate_segment_metrics(folds, "test_segments"),
        "promotion_rate": round(
            sum(bool(fold["dev_promote_guidance"]) for fold in folds) / len(folds),
            3,
        ),
    }


def aggregate_split_metrics(folds: list[dict[str, object]], split: str) -> dict[str, object]:
    return {
        "harness": {
            metric: aggregate_metric_values(
                folds=folds,
                split=split,
                metric=metric,
            )
            for metric in ("p_at_1", "usefulness_rate", "cache_hit_rate")
        }
    }


def aggregate_metric_values(
    folds: list[dict[str, object]],
    split: str,
    metric: str,
) -> dict[str, float]:
    baseline_values = [
        float(fold[split]["baseline"]["harness"][metric])
        for fold in folds
    ]
    guided_values = [
        float(fold[split]["guided"]["harness"][metric])
        for fold in folds
    ]
    gains = [
        round(guided - baseline, 3)
        for baseline, guided in zip(baseline_values, guided_values)
    ]
    return {
        "baseline_mean": round(mean(baseline_values), 3),
        "guided_mean": round(mean(guided_values), 3),
        "gain_mean": round(mean(gains), 3),
        "min_guided": round(min(guided_values), 3),
        "max_guided": round(max(guided_values), 3),
    }


def aggregate_segment_metrics(
    folds: list[dict[str, object]],
    segment_key: str,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    dimensions: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for fold in folds:
        segments = fold["analytics"][segment_key]
        if not isinstance(segments, dict):
            continue
        for dimension, dimension_rows in segments.items():
            if dimension == "focus_areas" or not isinstance(dimension_rows, dict):
                continue
            dimensions.setdefault(dimension, {})
            for name, summary in dimension_rows.items():
                if not isinstance(summary, dict):
                    continue
                dimensions[dimension].setdefault(str(name), [])
                dimensions[dimension][str(name)].append(
                    (
                        float(summary["baseline"]["p_at_1"]),
                        float(summary["guided"]["p_at_1"]),
                    )
                )

    return {
        dimension: {
            name: {
                "p_at_1": aggregate_baseline_guided_pairs(values),
                "usefulness_rate": aggregate_segment_usefulness(
                    folds,
                    segment_key,
                    dimension,
                    name,
                ),
            }
            for name, values in sorted(names.items())
        }
        for dimension, names in sorted(dimensions.items())
    }


def aggregate_baseline_guided_pairs(values: list[tuple[float, float]]) -> dict[str, float]:
    baseline_values = [baseline for baseline, _guided in values]
    guided_values = [guided for _baseline, guided in values]
    return {
        "baseline_mean": round(mean(baseline_values), 3),
        "guided_mean": round(mean(guided_values), 3),
        "gain_mean": round(
            mean(guided - baseline for baseline, guided in values),
            3,
        ),
        "min_guided": round(min(guided_values), 3),
        "max_guided": round(max(guided_values), 3),
    }


def aggregate_segment_usefulness(
    folds: list[dict[str, object]],
    segment_key: str,
    dimension: str,
    name: str,
) -> dict[str, float]:
    values: list[tuple[float, float]] = []
    for fold in folds:
        segment = fold["analytics"][segment_key][dimension].get(name)
        if segment:
            values.append(
                (
                    float(segment["baseline"]["usefulness_rate"]),
                    float(segment["guided"]["usefulness_rate"]),
                )
            )
    return aggregate_baseline_guided_pairs(values)


def weak_segments(folds: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate = aggregate_segment_metrics(folds, "test_segments")
    rows: list[dict[str, object]] = []
    for dimension, segments in aggregate.items():
        for name, metrics in segments.items():
            rows.append(
                {
                    "segment": dimension,
                    "name": name,
                    "guided_p_at_1": metrics["p_at_1"]["guided_mean"],
                    "p_at_1_gain": metrics["p_at_1"]["gain_mean"],
                    "guided_usefulness_rate": metrics["usefulness_rate"]["guided_mean"],
                    "usefulness_gain": metrics["usefulness_rate"]["gain_mean"],
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            float(row["guided_p_at_1"]),
            float(row["guided_usefulness_rate"]),
            float(row["p_at_1_gain"]),
        ),
    )[:10]
