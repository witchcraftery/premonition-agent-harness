from __future__ import annotations

from html import escape
from pathlib import Path


def render_benchmark_dashboard(report: dict[str, object]) -> str:
    if "probability_pack_replay" in report and "background_recovery_calibration" in report:
        return render_response_mode_dashboard(report)

    summary = report["summary"]
    aggregates = report["aggregates"]
    test_harness = aggregates["test"]["harness"]
    actor_segments = aggregates["test_segments"]["by_actor"]
    profile_segments = aggregates["test_segments"].get("by_profile", {})
    weak_segments = report["weak_segments"]
    folds = report["folds"]
    guidance_delta = report.get("guidance_delta_summary", {})

    overall = metric_card("Overall p@1", test_harness["p_at_1"])
    environment = metric_card("Environment p@1", actor_segments["environment"]["p_at_1"])
    user = metric_card("User p@1", actor_segments["user"]["p_at_1"])
    promotion = simple_card("Promotion rate", str(aggregates["promotion_rate"]))
    improved = simple_card(
        "Improved turns",
        str(guidance_delta.get("improved_turn_count", 0)),
    )

    actor_bars = "\n".join(
        comparison_bar(
            label=str(label),
            baseline=float(metrics["p_at_1"]["baseline_mean"]),
            guided=float(metrics["p_at_1"]["guided_mean"]),
        )
        for label, metrics in actor_segments.items()
    )
    profile_bars = "\n".join(
        comparison_bar(
            label=str(label),
            baseline=float(metrics["p_at_1"]["baseline_mean"]),
            guided=float(metrics["p_at_1"]["guided_mean"]),
        )
        for label, metrics in list(profile_segments.items())[:6]
    )
    weak_rows = "\n".join(
        f"""
        <li>
          <span>{escape(str(row["name"]))}</span>
          <strong>{float(row["guided_p_at_1"]):.3f}</strong>
          <em>gain {float(row["p_at_1_gain"]):+.3f}</em>
        </li>
        """
        for row in weak_segments[:8]
    )
    fold_rows = "\n".join(
        fold_row(fold)
        for fold in folds
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Premonition Benchmark</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8f4;
      --ink: #15201e;
      --muted: #68736f;
      --line: #dce1d8;
      --panel: #ffffff;
      --teal: #0f766e;
      --teal-soft: #d8eeea;
      --amber: #b7791f;
      --amber-soft: #f5e6c8;
      --shadow: 0 18px 55px rgba(21, 32, 30, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{
      width: min(1180px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 36px 0 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0;
      font-size: 40px;
      line-height: 1.05;
      font-weight: 760;
    }}
    .status {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
      text-align: right;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .panel, .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .metric {{
      padding: 18px;
      min-height: 124px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 18px;
    }}
    .metric strong {{
      display: block;
      font-size: 26px;
      line-height: 1;
    }}
    .metric em {{
      display: block;
      margin-top: 10px;
      color: var(--teal);
      font-style: normal;
      font-size: 13px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.35fr 0.9fr;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      padding: 22px;
    }}
    h2 {{
      margin: 0 0 18px;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 760;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 170px 1fr 70px;
      align-items: center;
      gap: 12px;
      padding: 10px 0;
      border-top: 1px solid var(--line);
    }}
    .bar-row:first-of-type {{ border-top: 0; }}
    .bar-label {{
      color: var(--ink);
      font-size: 13px;
      font-weight: 650;
    }}
    .track {{
      position: relative;
      height: 18px;
      background: #edf0eb;
      border-radius: 999px;
      overflow: hidden;
    }}
    .baseline, .guided {{
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      border-radius: 999px;
    }}
    .baseline {{ background: var(--amber-soft); }}
    .guided {{ background: var(--teal); opacity: 0.88; }}
    .bar-value {{
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .weak-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .weak-list li {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 12px;
      align-items: center;
      padding: 13px 0;
      border-top: 1px solid var(--line);
      font-size: 14px;
    }}
    .weak-list li:first-child {{ border-top: 0; }}
    .weak-list strong {{
      font-variant-numeric: tabular-nums;
    }}
    .weak-list em {{
      color: var(--teal);
      font-style: normal;
      font-size: 12px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-top: 1px solid var(--line);
      padding: 12px 8px;
      font-variant-numeric: tabular-nums;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
    }}
    .dot.base {{ background: var(--amber-soft); border: 1px solid #e4c98f; }}
    .dot.guided {{ background: var(--teal); }}
    @media (max-width: 900px) {{
      header {{ align-items: start; flex-direction: column; }}
      .status {{ text-align: left; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
      .bar-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Premonition Benchmark</h1>
      </div>
      <div class="status">
        5-fold train/dev/test loop<br>
        {summary["total_turns"]} turns · {summary["iterations"]} guidance iterations · top-{summary["top_k"]} branches
      </div>
    </header>
    <section class="metrics">
      {overall}
      {environment}
      {user}
      {promotion}
      {improved}
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Actor Performance</h2>
        <div class="legend"><span><i class="dot base"></i>Baseline</span><span><i class="dot guided"></i>Guided</span></div>
        {actor_bars}
      </div>
      <div class="panel">
        <h2>Weakest Segments</h2>
        <ol class="weak-list">{weak_rows}</ol>
      </div>
      <div class="panel">
        <h2>Profile Performance</h2>
        <div class="legend"><span><i class="dot base"></i>Baseline</span><span><i class="dot guided"></i>Guided</span></div>
        {profile_bars}
      </div>
      <div class="panel">
        <h2>Guidance Delta</h2>
        <p>{guidance_delta.get("improved_turn_count", 0)} held-out turns improved and {guidance_delta.get("regressed_turn_count", 0)} regressed across folds.</p>
      </div>
      <div class="panel wide">
        <h2>Fold Results</h2>
        <table>
          <thead><tr><th>Fold</th><th>Promoted</th><th>Baseline p@1</th><th>Guided p@1</th><th>Improved</th><th>Regressed</th></tr></thead>
          <tbody>{fold_rows}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def write_benchmark_dashboard(report: dict[str, object], output_path: Path) -> None:
    output_path.write_text(render_benchmark_dashboard(report), encoding="utf-8")


def render_premonition_outcome_dashboard(
    bakeoff_report: dict[str, object],
    stress_report: dict[str, object],
) -> str:
    summary = dict(bakeoff_report["summary"])
    baseline = dict(bakeoff_report["probability_pack_replay_baseline"])
    current = dict(bakeoff_report["probability_pack_replay"])
    evaluation = dict(bakeoff_report["background_recovery_evaluation"])
    stress_summary = dict(stress_report["summary"])
    stress_aggregates = dict(stress_report["aggregates"])
    stress_quality_gain = dict(stress_aggregates["quality_ready_gain"])
    stress_prepared_gain = dict(stress_aggregates["prepared_hit_gain"])
    policy_counts = dict(stress_aggregates.get("selected_policy_counts", {}))
    promotion_rate = float(stress_aggregates["promotion_rate"])
    stress_run_count = int(stress_summary["run_count"])
    promoted_count = round(promotion_rate * stress_run_count)

    first_speech_rate = float(baseline["first_speech_hit_rate"])
    baseline_prepared = float(baseline["prepared_hit_rate"])
    baseline_quality_ready = float(baseline["quality_ready_rate"])
    current_quality_ready = float(current["quality_ready_rate"])
    current_prepared = float(current["prepared_hit_rate"])
    background_recovery = float(current.get("background_recovery_hit_rate", 0.0))
    latency_saved = int(current.get("median_latency_saved_ms", 0))

    headline_cards = "\n".join(
        card.strip()
        for card in (
            outcome_metric_card(
                "Readiness Reach",
                f"{first_speech_rate:.3f} -> {current_quality_ready:.3f}",
                f"{current_quality_ready - first_speech_rate:+.3f} from first-speech base",
            ),
            outcome_metric_card(
                "Quality-Ready Lift",
                f"{baseline_quality_ready:.3f} -> {current_quality_ready:.3f}",
                f"{current_quality_ready - baseline_quality_ready:+.3f} over pack baseline",
            ),
            outcome_metric_card(
                "Stress Promotion",
                f"{promoted_count} / {stress_run_count}",
                f"{promotion_rate:.3f} gated pass rate",
            ),
            outcome_metric_card(
                "Raw Floor",
                f"{float(stress_prepared_gain['min']):+.3f}",
                "minimum raw prepared-hit gain",
            ),
        )
    )
    ladder = "\n".join(
        (
            outcome_stage(
                "Base State",
                "Single confirmed next response.",
                first_speech_rate,
                "Only the first-speech branch is ready for immediate delivery.",
            ),
            outcome_stage(
                "Probability Pack Baseline",
                "Background branches are prepared, but some are semantic-only.",
                baseline_quality_ready,
                f"Raw prepared coverage is {baseline_prepared:.3f}; voice-ready coverage is {baseline_quality_ready:.3f}.",
            ),
            outcome_stage(
                "Current Guarded Swarm",
                "Background recovery adds protected drafts behind the voice.",
                current_quality_ready,
                f"Prepared coverage is {current_prepared:.3f}; recovery adds {background_recovery:.3f} behind the first response.",
            ),
        )
    )
    mode_rows = "\n".join(
        outcome_mode_row(mode, dict(result))
        for mode, result in sorted(dict(evaluation["target_mode_results"]).items())
    )
    policy_rows = "\n".join(
        f"<li><span>{escape(str(policy))}</span><strong>{int(count)}</strong></li>"
        for policy, count in sorted(policy_counts.items(), key=lambda item: str(item[0]))
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Premonition Swarm Outcome</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7f4;
      --ink: #17211e;
      --muted: #66746f;
      --line: #d9e0da;
      --panel: #ffffff;
      --green: #176b5b;
      --blue: #315f86;
      --amber: #b7791f;
      --soft-green: #dbeee9;
      --soft-blue: #dbe8f2;
      --soft-amber: #f2e4c8;
      --track: #edf1ed;
      --shadow: 0 18px 55px rgba(23, 33, 30, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{
      width: min(1180px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 34px 0 48px;
    }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: end;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: 38px;
      line-height: 1.05;
    }}
    .subtitle {{
      color: var(--muted);
      margin: 10px 0 0;
      max-width: 760px;
      line-height: 1.5;
    }}
    .status {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      text-align: right;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .metric {{
      padding: 18px;
      min-height: 116px;
    }}
    .metric span, .kicker {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 14px;
    }}
    .metric strong {{
      display: block;
      font-size: 25px;
      line-height: 1;
      font-variant-numeric: tabular-nums;
    }}
    .metric em {{
      display: block;
      margin-top: 10px;
      color: var(--green);
      font-style: normal;
      font-size: 13px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.25fr 0.85fr;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      padding: 22px;
    }}
    .wide {{ grid-column: 1 / -1; }}
    h2 {{
      margin: 0 0 16px;
      font-size: 18px;
    }}
    .ladder {{
      display: grid;
      gap: 14px;
    }}
    .stage {{
      display: grid;
      grid-template-columns: 190px 1fr 76px;
      gap: 16px;
      align-items: center;
      padding: 14px 0;
      border-top: 1px solid var(--line);
    }}
    .stage:first-child {{ border-top: 0; }}
    .stage h3 {{
      margin: 0 0 5px;
      font-size: 15px;
    }}
    .stage p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .track {{
      height: 18px;
      background: var(--track);
      border-radius: 999px;
      overflow: hidden;
    }}
    .fill {{
      display: block;
      height: 100%;
      background: var(--green);
      border-radius: 999px;
    }}
    .stage-value {{
      text-align: right;
      font-weight: 760;
      font-variant-numeric: tabular-nums;
    }}
    .note {{
      color: var(--muted);
      margin: 0;
      line-height: 1.55;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 12px 8px;
      text-align: left;
      font-variant-numeric: tabular-nums;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
    }}
    .policy-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .policy-list li {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 0;
      border-top: 1px solid var(--line);
    }}
    .policy-list li:first-child {{ border-top: 0; }}
    .assessment {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .assessment div {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fbfcfa;
    }}
    .assessment strong {{
      display: block;
      margin-bottom: 8px;
    }}
    @media (max-width: 900px) {{
      header {{ grid-template-columns: 1fr; }}
      .status {{ text-align: left; }}
      .metrics, .grid, .assessment {{ grid-template-columns: 1fr; }}
      .stage {{ grid-template-columns: 1fr; }}
      .stage-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Premonition Swarm Outcome</h1>
        <p class="subtitle">Not prophecy. This is measured preparedness: how often the backend can have useful, voice-ready branches waiting before the conversational frontend confirms the next move.</p>
      </div>
      <div class="status">
        ESConv response-mode benchmark<br>
        {summary.get("test_turns", 0)} held-out turns · {stress_summary.get("seed_count", 0)} seeds x {stress_summary.get("fold_count", 0)} folds
      </div>
    </header>
    <section class="metrics">
{headline_cards}
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Preparedness Maturity</h2>
        <div class="ladder">{ladder}</div>
      </div>
      <div class="panel">
        <h2>Stress Gate</h2>
        <p class="note">{promoted_count} of {stress_run_count} shuffled folds promoted under both the quality-aware gate and raw prepared-hit floor. Mean quality-ready gain was {float(stress_quality_gain["mean"]):+.3f}; mean background recovery was {float(dict(stress_aggregates["background_recovery_hit_rate"])["mean"]):.3f}.</p>
        <ol class="policy-list">{policy_rows}</ol>
      </div>
      <div class="panel">
        <h2>Recovered Response Mechanisms</h2>
        <table>
          <thead><tr><th>Mode</th><th>Prepared Gain</th></tr></thead>
          <tbody>{mode_rows}</tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Voice-Agent Meaning</h2>
        <p class="note">The backend is now useful as a guarded prewarming layer: it can prepare likely response mechanisms behind the voice while preserving first-speech behavior, draft quality, and the raw prepared baseline. Median saved latency remains {latency_saved} ms when a prepared branch is selected.</p>
      </div>
      <div class="panel wide">
        <h2>Assessment</h2>
        <div class="assessment">
          <div><strong>What works</strong><span>Quality-ready preparedness rises from {baseline_quality_ready:.3f} to {current_quality_ready:.3f}, and target recovery adds {background_recovery:.3f} behind the confirmed first response.</span></div>
          <div><strong>What changed from base</strong><span>The base state had {first_speech_rate:.3f} immediate first-speech readiness. The current swarm keeps that path while preparing broader background branches.</span></div>
          <div><strong>What remains</strong><span>The `recover_inform` slice still has two held-back stress folds. The next improvement should strengthen that candidate, not relax the gates.</span></div>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


def write_premonition_outcome_dashboard(
    bakeoff_report: dict[str, object],
    stress_report: dict[str, object],
    output_path: Path,
) -> None:
    output_path.write_text(
        render_premonition_outcome_dashboard(bakeoff_report, stress_report),
        encoding="utf-8",
    )


def outcome_metric_card(label: str, value: str, note: str) -> str:
    return f"""
    <article class="metric">
      <span>{escape(label)}</span>
      <strong>{value}</strong>
      <em>{escape(note)}</em>
    </article>
    """


def outcome_stage(title: str, subtitle: str, value: float, detail: str) -> str:
    width = max(0.0, min(100.0, value * 100))
    return f"""
    <section class="stage">
      <div>
        <h3>{escape(title)}</h3>
        <p>{escape(subtitle)}</p>
      </div>
      <div>
        <div class="track"><span class="fill" style="width: {width:.1f}%"></span></div>
        <p>{escape(detail)}</p>
      </div>
      <div class="stage-value">{value:.3f}</div>
    </section>
    """


def outcome_mode_row(mode: str, result: dict[str, object]) -> str:
    return f"""
    <tr>
      <td>{escape(mode)}</td>
      <td>{float(result["prepared_hit_gain"]):+.3f}</td>
    </tr>
    """


def render_response_mode_dashboard(report: dict[str, object]) -> str:
    summary = dict(report["summary"])
    baseline = dict(report["probability_pack_replay_baseline"])
    quality_aware_baseline = dict(
        report.get("probability_pack_replay_baseline_quality_aware", baseline)
    )
    active = dict(report["probability_pack_replay"])
    calibration = dict(report["background_recovery_calibration"])
    evaluation = dict(report["background_recovery_evaluation"])
    selected_policy = calibration.get("selected_policy") or {}
    selected_policy_name = str(dict(selected_policy).get("name", "none"))
    min_quality_score = float(calibration.get("min_quality_score", 0.0))
    mode_rows = response_mode_rows(baseline, active)

    prepared = comparison_metric_card(
        "Prepared Hit Rate",
        float(baseline["prepared_hit_rate"]),
        float(active["prepared_hit_rate"]),
    )
    quality_ready = comparison_metric_card(
        "Quality-Ready Recovery",
        float(baseline["quality_ready_rate"]),
        float(active["quality_ready_rate"]),
    )
    quality_aware_gate = comparison_metric_card(
        "Quality-Aware Gate",
        float(quality_aware_baseline["quality_ready_rate"]),
        float(active["quality_ready_rate"]),
    )
    raw_semantic = comparison_metric_card(
        "Raw Semantic Coverage",
        float(baseline["semantic_prepared_hit_rate"]),
        float(active["semantic_prepared_hit_rate"]),
    )
    background = comparison_metric_card(
        "Background Recovery",
        float(baseline["background_recovery_hit_rate"]),
        float(active["background_recovery_hit_rate"]),
    )
    promoted = simple_card(
        "Held-Out Promoted",
        "yes" if bool(evaluation.get("promoted", False)) else "no",
    )
    metric_cards = "\n".join(
        card.strip()
        for card in (
            prepared,
            quality_ready,
            quality_aware_gate,
            raw_semantic,
            background,
            promoted,
        )
    )
    rows = "\n".join(response_mode_dashboard_row(row) for row in mode_rows)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Premonition Response-Mode Recovery</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f5;
      --ink: #17211e;
      --muted: #69746f;
      --line: #dce2dc;
      --panel: #ffffff;
      --green: #176b5b;
      --blue: #315f86;
      --shadow: 0 18px 55px rgba(23, 33, 30, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{
      width: min(1120px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 36px 0 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0; font-size: 34px; line-height: 1.08; }}
    .status {{ color: var(--muted); font-size: 14px; line-height: 1.45; text-align: right; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .metric, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }}
    .metric {{ padding: 18px; min-height: 118px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; font-weight: 650; margin-bottom: 16px; }}
    .metric strong {{ display: block; font-size: 24px; line-height: 1; }}
    .metric em {{ display: block; margin-top: 10px; color: var(--green); font-style: normal; font-size: 13px; font-weight: 700; }}
    .panel {{ padding: 22px; }}
    h2 {{ margin: 0 0 16px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-top: 1px solid var(--line); text-align: left; padding: 12px 8px; font-variant-numeric: tabular-nums; }}
    th {{ color: var(--muted); font-size: 12px; }}
    .mode {{ font-weight: 750; }}
    .gain {{ color: var(--green); font-weight: 750; }}
    .note {{ color: var(--muted); margin: 0; line-height: 1.55; max-width: 760px; }}
    @media (max-width: 900px) {{
      header {{ align-items: start; flex-direction: column; }}
      .status {{ text-align: left; }}
      .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Premonition Response-Mode Recovery</h1>
      </div>
      <div class="status">
        {summary.get("test_turns", 0)} held-out turns · top-{summary.get("top_k", 3)} branches<br>
        selected: {escape(selected_policy_name)} · quality floor {min_quality_score:.2f}
      </div>
    </header>
    <section class="metrics">
{metric_cards}
    </section>
    <section class="panel">
      <h2>Quality-Ready Recovery By Mode</h2>
      <p class="note">Raw semantic coverage can look useful before it is voice-ready. This view contrasts baseline prepared coverage with the active quality-filtered recovery pack.</p>
      <table>
        <thead>
          <tr><th>Mode</th><th>Baseline Prepared</th><th>Active Prepared</th><th>Quality-Ready</th><th>Recovery Hit</th><th>Gain</th></tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def response_mode_rows(
    baseline: dict[str, object],
    active: dict[str, object],
) -> list[dict[str, float | str]]:
    baseline_modes = dict(dict(baseline["segments"])["expected_response_mode"])  # type: ignore[index]
    active_modes = dict(dict(active["segments"])["expected_response_mode"])  # type: ignore[index]
    rows = []
    for mode in sorted(set(baseline_modes) | set(active_modes)):
        baseline_segment = dict(baseline_modes.get(mode, {}))
        active_segment = dict(active_modes.get(mode, {}))
        baseline_prepared = float(baseline_segment.get("prepared_hit_rate", 0.0))
        active_prepared = float(active_segment.get("prepared_hit_rate", 0.0))
        rows.append(
            {
                "mode": mode,
                "baseline_prepared": baseline_prepared,
                "active_prepared": active_prepared,
                "quality_ready": float(active_segment.get("quality_ready_rate", 0.0)),
                "background_recovery": float(
                    active_segment.get("background_recovery_hit_rate", 0.0)
                ),
                "gain": round(active_prepared - baseline_prepared, 3),
            }
        )
    return sorted(rows, key=lambda row: (float(row["gain"]), str(row["mode"])), reverse=True)


def response_mode_dashboard_row(row: dict[str, float | str]) -> str:
    return f"""
    <tr>
      <td class="mode">{escape(str(row["mode"]))}</td>
      <td>{float(row["baseline_prepared"]):.3f}</td>
      <td>{float(row["active_prepared"]):.3f}</td>
      <td>{float(row["quality_ready"]):.3f}</td>
      <td>{float(row["background_recovery"]):.3f}</td>
      <td class="gain">{float(row["gain"]):+.3f}</td>
    </tr>
    """


def comparison_metric_card(label: str, baseline: float, active: float) -> str:
    return f"""
    <article class="metric">
      <span>{escape(label)}</span>
      <strong>{baseline:.3f} -> {active:.3f}</strong>
      <em>{active - baseline:+.3f} gain</em>
    </article>
    """


def metric_card(label: str, metric: dict[str, float]) -> str:
    baseline = float(metric["baseline_mean"])
    guided = float(metric["guided_mean"])
    gain = float(metric["gain_mean"])
    return f"""
    <article class="metric">
      <span>{escape(label)}</span>
      <strong>{baseline:.3f} -> {guided:.3f}</strong>
      <em>{gain:+.3f} gain</em>
    </article>
    """


def simple_card(label: str, value: str) -> str:
    return f"""
    <article class="metric">
      <span>{escape(label)}</span>
      <strong>{escape(value)}</strong>
      <em>current run</em>
    </article>
    """


def comparison_bar(label: str, baseline: float, guided: float) -> str:
    baseline_width = max(0.0, min(100.0, baseline * 100))
    guided_width = max(0.0, min(100.0, guided * 100))
    return f"""
    <div class="bar-row">
      <div class="bar-label">{escape(label)}</div>
      <div class="track">
        <div class="baseline" style="width: {baseline_width:.1f}%"></div>
        <div class="guided" style="width: {guided_width:.1f}%"></div>
      </div>
      <div class="bar-value">{baseline:.3f} -> {guided:.3f}</div>
    </div>
    """


def fold_row(fold: dict[str, object]) -> str:
    baseline = float(fold["test"]["baseline"]["harness"]["p_at_1"])
    guided = float(fold["test"]["guided"]["harness"]["p_at_1"])
    delta = fold["guidance_delta"]
    promoted = "yes" if fold["dev_promote_guidance"] else "no"
    return f"""
    <tr>
      <td>Fold {fold["fold"]}</td>
      <td>{promoted}</td>
      <td>{baseline:.3f}</td>
      <td>{guided:.3f}</td>
      <td>{delta["improved_turn_count"]}</td>
      <td>{delta["regressed_turn_count"]}</td>
    </tr>
    """
