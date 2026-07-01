from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from foresight_harness.live_shadow import (
    LiveShadowSession,
    handle_live_shadow_api_request,
)


def render_live_shadow_lab_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Premonition Live Shadow Lab</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8f7;
      --panel: #ffffff;
      --panel-2: #f0f5f3;
      --ink: #192321;
      --muted: #65726e;
      --line: #d9e2de;
      --teal: #0d7c70;
      --teal-dark: #07564f;
      --amber: #b87514;
      --red: #b24236;
      --shadow: 0 16px 40px rgba(25, 35, 33, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(180deg, rgba(13, 124, 112, 0.06), transparent 34rem),
        repeating-linear-gradient(90deg, rgba(25, 35, 33, 0.035) 0 1px, transparent 1px 72px),
        var(--bg);
      color: var(--ink);
    }

    button, input, textarea, select {
      font: inherit;
    }

    .topbar {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      background: #17211f;
      color: #f7fbfa;
      border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .mark {
      width: 30px;
      height: 30px;
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 6px;
      display: grid;
      place-items: center;
      color: #8de0d4;
      font-weight: 800;
      font-size: 14px;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.1;
      font-weight: 720;
      letter-spacing: 0;
    }

    .topmeta {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 12px;
      color: rgba(247, 251, 250, 0.74);
      white-space: nowrap;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 99px;
      background: #8de0d4;
      box-shadow: 0 0 0 4px rgba(141, 224, 212, 0.16);
    }

    main {
      height: calc(100vh - 64px);
      display: grid;
      grid-template-rows: minmax(0, 1fr) 112px;
      gap: 14px;
      padding: 14px;
    }

    .lanes {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(280px, 0.95fr) minmax(360px, 1.28fr) minmax(300px, 1fr);
      gap: 14px;
    }

    .panel {
      min-height: 0;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .panel header {
      padding: 14px 16px 10px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }

    .panel h2 {
      margin: 0 0 4px;
      font-size: 14px;
      font-weight: 760;
    }

    .hint {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .body {
      min-height: 0;
      overflow: auto;
      padding: 14px;
    }

    .transcript {
      display: grid;
      gap: 10px;
    }

    .message {
      padding: 10px 11px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
    }

    .message strong {
      display: block;
      color: var(--teal-dark);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: 5px;
    }

    .message p {
      margin: 0;
      font-size: 13px;
      line-height: 1.4;
    }

    form {
      padding: 12px;
      border-top: 1px solid var(--line);
      background: var(--panel-2);
      display: grid;
      gap: 9px;
    }

    textarea, select, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      font-size: 13px;
      line-height: 1.35;
    }

    textarea {
      min-height: 72px;
      resize: vertical;
    }

    .row {
      display: grid;
      grid-template-columns: 128px 1fr;
      gap: 8px;
    }

    .button-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    button {
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 9px 11px;
      background: var(--teal);
      color: #fff;
      font-size: 12px;
      font-weight: 720;
      cursor: pointer;
    }

    button.secondary {
      color: var(--teal-dark);
      background: #e4f2ef;
      border-color: #bddbd5;
    }

    button.warning {
      color: #fff;
      background: var(--amber);
    }

    .drafts {
      display: grid;
      gap: 10px;
    }

    .draft {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }

    .draft-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 11px;
      border-bottom: 1px solid var(--line);
      background: #fbfdfc;
    }

    .mode {
      color: var(--teal-dark);
      font-size: 12px;
      font-weight: 760;
    }

    .rank {
      color: var(--muted);
      font-size: 12px;
    }

    .draft p {
      margin: 0;
      padding: 10px 11px;
      font-size: 13px;
      line-height: 1.42;
    }

    .meter {
      height: 7px;
      margin: 0 11px 11px;
      background: #e6ecea;
      border-radius: 99px;
      overflow: hidden;
    }

    .meter span {
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--teal), #7abfb5);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 11px;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 5px;
    }

    .metric strong {
      font-size: 22px;
      line-height: 1;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }

    th, td {
      text-align: left;
      padding: 9px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }

    th {
      color: var(--muted);
      font-size: 11px;
      font-weight: 720;
      background: #fbfdfc;
    }

    .timeline {
      background: #17211f;
      color: #f7fbfa;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      box-shadow: var(--shadow);
      padding: 14px 16px;
      overflow: hidden;
    }

    .timeline h2 {
      margin: 0 0 12px;
      font-size: 13px;
    }

    .events {
      display: flex;
      gap: 10px;
      overflow-x: auto;
      padding-bottom: 4px;
    }

    .event {
      min-width: 150px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 6px;
      padding: 9px;
      background: rgba(255, 255, 255, 0.05);
    }

    .event strong {
      display: block;
      color: #8de0d4;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: 5px;
    }

    .event span {
      display: block;
      color: rgba(247, 251, 250, 0.8);
      font-size: 12px;
      line-height: 1.35;
    }

    .empty {
      border: 1px dashed #bcc9c5;
      border-radius: 6px;
      padding: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      background: rgba(255, 255, 255, 0.65);
    }

    @media (max-width: 980px) {
      main { height: auto; }
      .lanes { grid-template-columns: 1fr; }
      .panel { min-height: 420px; }
      .topmeta { display: none; }
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <div class="mark">P</div>
      <h1>Premonition Live Shadow Lab</h1>
    </div>
    <div class="topmeta"><span class="status-dot"></span><span>Shadow mode: drafts are scored before speech release</span></div>
  </div>
  <main>
    <section class="lanes">
      <article class="panel">
        <header>
          <div>
            <h2>Live Conversation</h2>
            <p class="hint">Feed transcript turns from a human, a voice model, or a manual test.</p>
          </div>
        </header>
        <div class="body transcript" id="messages"></div>
        <form id="observe-form">
          <div class="row">
            <select id="observe-role">
              <option value="user">user</option>
              <option value="assistant">assistant</option>
              <option value="environment">environment</option>
              <option value="system">system</option>
            </select>
            <button type="submit">Observe + Draft</button>
          </div>
          <textarea id="observe-content" placeholder="Add the latest transcript turn..."></textarea>
        </form>
      </article>

      <article class="panel">
        <header>
          <div>
            <h2>Premonition Drafts</h2>
            <p class="hint">The backend prepares likely next response modes while reality keeps moving.</p>
          </div>
          <button class="secondary" id="refresh-button" type="button">Refresh</button>
        </header>
        <div class="body drafts" id="drafts"></div>
      </article>

      <article class="panel">
        <header>
          <div>
            <h2>Reality Grading</h2>
            <p class="hint">Capture what actually happened next, then score preparedness.</p>
          </div>
        </header>
        <div class="body">
          <div class="metrics" id="metrics"></div>
          <table>
            <thead><tr><th>Actual</th><th>Prepared</th><th>Grade</th><th>Saved</th></tr></thead>
            <tbody id="grades"></tbody>
          </table>
        </div>
        <form id="grade-form">
          <div class="row">
            <select id="actual-mode">
              <option value="">infer mode</option>
              <option>ask_followup</option>
              <option>validate</option>
              <option>reassure</option>
              <option>disclose</option>
              <option>suggest</option>
              <option>encourage</option>
              <option>inform</option>
              <option>commit</option>
              <option>apologize</option>
              <option>redirect</option>
              <option>other</option>
            </select>
            <button type="submit">Grade Reality</button>
          </div>
          <textarea id="actual-content" placeholder="Paste the actual next utterance..."></textarea>
          <div class="button-row">
            <button class="secondary" id="export-button" type="button">Export JSONL</button>
            <button class="warning" id="reset-button" type="button">Reset Session</button>
          </div>
        </form>
      </article>
    </section>
    <section class="timeline">
      <h2>Experiment Timeline</h2>
      <div class="events" id="timeline"></div>
    </section>
  </main>
  <script>
    let state = null;

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function pct(value) {
      return `${Math.round(Number(value || 0) * 100)}%`;
    }

    function renderMessages() {
      const root = document.getElementById("messages");
      if (!state.messages.length) {
        root.innerHTML = `<div class="empty">No transcript yet. Add a turn to start the shadow lane.</div>`;
        return;
      }
      root.innerHTML = state.messages.map(message => `
        <div class="message">
          <strong>${message.role}</strong>
          <p>${escapeHtml(message.content)}</p>
        </div>
      `).join("");
    }

    function renderDrafts() {
      const root = document.getElementById("drafts");
      const pack = state.active_pack;
      if (!pack) {
        root.innerHTML = `<div class="empty">Drafts appear after the first observed turn.</div>`;
        return;
      }
      root.innerHTML = pack.prepared_drafts.map((draft, index) => `
        <div class="draft">
          <div class="draft-top">
            <span class="mode">${draft.response_mode}</span>
            <span class="rank">rank ${index + 1} · ${draft.preparation_role}</span>
          </div>
          <p>${escapeHtml(draft.tts_text)}</p>
          <div class="meter"><span style="width: ${pct(draft.readiness_score)}"></span></div>
        </div>
      `).join("");
    }

    function renderMetrics() {
      const metrics = state.metrics;
      document.getElementById("metrics").innerHTML = `
        <div class="metric"><span>Prepared Hit</span><strong>${pct(metrics.prepared_hit_rate)}</strong></div>
        <div class="metric"><span>Exact Hit</span><strong>${pct(metrics.exact_hit_rate)}</strong></div>
        <div class="metric"><span>Quality Ready</span><strong>${pct(metrics.quality_ready_rate)}</strong></div>
        <div class="metric"><span>Latency Saved</span><strong>${metrics.median_latency_saved_ms}ms</strong></div>
      `;
    }

    function renderGrades() {
      const root = document.getElementById("grades");
      if (!state.grades.length) {
        root.innerHTML = `<tr><td colspan="4">No graded turns yet.</td></tr>`;
        return;
      }
      root.innerHTML = state.grades.slice().reverse().map(grade => `
        <tr>
          <td>${grade.actual_response_mode}</td>
          <td>${grade.prepared_response_mode || "none"}</td>
          <td>${grade.match_grade}</td>
          <td>${grade.latency_saved_ms}ms</td>
        </tr>
      `).join("");
    }

    function renderTimeline() {
      const root = document.getElementById("timeline");
      if (!state.timeline.length) {
        root.innerHTML = `<div class="event"><strong>ready</strong><span>Waiting for the first observed turn.</span></div>`;
        return;
      }
      root.innerHTML = state.timeline.map(event => `
        <div class="event"><strong>${event.event_type}</strong><span>${escapeHtml(event.label)}</span></div>
      `).join("");
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[character]));
    }

    function render() {
      renderMessages();
      renderDrafts();
      renderMetrics();
      renderGrades();
      renderTimeline();
    }

    async function load() {
      state = await api("/api/session");
      render();
    }

    document.getElementById("observe-form").addEventListener("submit", async event => {
      event.preventDefault();
      state = await api("/api/observe", {
        method: "POST",
        body: JSON.stringify({
          role: document.getElementById("observe-role").value,
          content: document.getElementById("observe-content").value
        })
      });
      document.getElementById("observe-content").value = "";
      render();
    });

    document.getElementById("grade-form").addEventListener("submit", async event => {
      event.preventDefault();
      state = await api("/api/grade", {
        method: "POST",
        body: JSON.stringify({
          actual_response_mode: document.getElementById("actual-mode").value,
          content: document.getElementById("actual-content").value
        })
      });
      document.getElementById("actual-content").value = "";
      render();
    });

    document.getElementById("refresh-button").addEventListener("click", load);
    document.getElementById("reset-button").addEventListener("click", async () => {
      state = await api("/api/reset", { method: "POST", body: "{}" });
      render();
    });
    document.getElementById("export-button").addEventListener("click", async () => {
      const exportData = await api("/api/export");
      const blob = new Blob([exportData.jsonl], { type: "application/jsonl" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "premonition-live-shadow.jsonl";
      link.click();
      URL.revokeObjectURL(url);
    });

    load();
  </script>
</body>
</html>"""


class LiveShadowHTTPRequestHandler(BaseHTTPRequestHandler):
    session = LiveShadowSession()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send_html(render_live_shadow_lab_html())
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status, payload = handle_live_shadow_api_request(
            self.session,
            "GET",
            self.path,
            None,
        )
        self._send_json(status, payload)

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        payload: dict[str, Any] = json.loads(raw_body or "{}")
        status, response = handle_live_shadow_api_request(
            self.session,
            "POST",
            self.path,
            payload,
        )
        self._send_json(status, response)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_live_shadow_app(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = ThreadingHTTPServer((host, port), LiveShadowHTTPRequestHandler)
    print(f"Premonition Live Shadow Lab running at http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
