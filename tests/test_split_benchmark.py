from pathlib import Path

from foresight_harness.evaluator import load_replay_turns
from foresight_harness.experiments import load_trial_config
from foresight_harness.split_benchmark import run_split_benchmark


def test_split_benchmark_learns_on_train_and_improves_test():
    train_turns = load_replay_turns(Path("data/queueahead_challenge.jsonl"))
    test_turns = load_replay_turns(Path("data/queueahead_challenge_test.jsonl"))

    result = run_split_benchmark(
        train_turns=train_turns,
        test_turns=test_turns,
        iterations=3,
        top_k=3,
    )

    assert result["train"]["baseline"]["harness"]["p_at_1"] == 0.4
    assert result["train"]["guided"]["harness"]["p_at_1"] == 1.0
    assert result["test"]["guided"]["harness"]["p_at_1"] > result["test"]["baseline"]["harness"]["p_at_1"]
    assert result["generalization"]["test_p_at_1_gain"] > 0
    assert result["generalization"]["overfit_gap"] >= 0
    assert result["promote_guidance"] is True
    assert result["analytics"]["test_segments"]["by_topic"]["address_change"]["guided"]["p_at_1"] == 1.0
    assert result["analytics"]["test_segments"]["by_actor"]["user"]["delta"]["p_at_1"] > 0
    assert result["analytics"]["test_segments"]["by_actor"]["environment"]["guided"]["p_at_1"] == 1.0
    assert result["analytics"]["test_segments"]["by_event_type"]["fulfillment"]["guided"]["usefulness_rate"] == 1.0
    assert result["analytics"]["focus_areas"]
    assert "guidance_markdown" in result


def test_split_benchmark_loads_configs():
    train_config = load_trial_config(Path("experiments/queueahead_challenge_train.json"))
    test_config = load_trial_config(Path("experiments/queueahead_challenge_test.json"))

    assert train_config.split == "train"
    assert test_config.split == "test"
    assert train_config.input_path.exists()
    assert test_config.input_path.exists()
