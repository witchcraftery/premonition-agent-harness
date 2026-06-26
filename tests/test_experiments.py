from pathlib import Path

from foresight_harness.experiments import load_trial_config


def test_load_trial_config_resolves_relative_input_path():
    config = load_trial_config(Path("experiments/queueahead_v1.json"))

    assert config.name == "queueahead_v1_sample"
    assert config.top_k == 3
    assert config.input_path.name == "queueahead_sample.jsonl"
    assert config.input_path.exists()
