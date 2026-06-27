from __future__ import annotations

from html import escape
from pathlib import Path


def render_benchmark_dashboard(report: dict[str, object]) -> str:
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
