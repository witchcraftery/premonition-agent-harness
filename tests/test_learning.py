from foresight_harness.learning import analyze_harness_misses


def test_analyze_harness_misses_counts_reasons():
    report = analyze_harness_misses(
        [
            {
                "variant": "harness",
                "selected_artifact_id": "artifact-1",
                "unsafe_leak": False,
                "branches": [{"match_grade": "exact_intent"}],
            },
            {
                "variant": "harness",
                "selected_artifact_id": None,
                "unsafe_leak": False,
                "branches": [{"match_grade": "miss"}],
            },
            {
                "variant": "live_agent",
                "selected_artifact_id": None,
                "unsafe_leak": False,
                "branches": [],
            },
        ]
    )

    assert report["harness_turns"] == 2
    assert report["reason_counts"] == {
        "exact_hit": 1,
        "no_prepared_artifact": 1,
    }
    assert report["recommendations"]
