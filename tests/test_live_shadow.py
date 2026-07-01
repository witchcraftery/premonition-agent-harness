from __future__ import annotations

import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from foresight_harness.live_shadow import (
    LiveShadowSession,
    handle_live_shadow_api_request,
    infer_response_mode_from_text,
)
from foresight_harness.live_shadow_app import (
    LiveShadowHTTPRequestHandler,
    render_live_shadow_lab_html,
)


def test_live_shadow_session_observes_and_drafts_probability_pack():
    session = LiveShadowSession(session_id="demo")

    state = session.observe("user", "I am overwhelmed and could use a little reassurance.")

    assert state["session_id"] == "demo"
    assert state["messages"][-1]["content"].startswith("I am overwhelmed")
    assert state["active_pack"]["confirmation_mode"] == "wait_for_reality_grade"
    assert len(state["active_pack"]["prepared_drafts"]) == 3
    assert "reassure" in {
        str(draft["response_mode"]) for draft in state["active_pack"]["prepared_drafts"]
    }
    assert [event["event_type"] for event in state["timeline"]] == ["observed", "drafted"]


def test_live_shadow_session_grades_latest_pack_against_reality():
    session = LiveShadowSession(session_id="demo")
    session.observe("user", "I am overwhelmed and could use a little reassurance.")

    state = session.grade_reality(
        "That sounds really heavy. You are not alone in it.",
        actual_response_mode="reassure",
    )

    grade = state["grades"][-1]
    assert grade["actual_response_mode"] == "reassure"
    assert grade["match_grade"] == "exact"
    assert grade["quality_ready"] is True
    assert grade["latency_saved_ms"] > 0
    assert state["metrics"]["graded_turns"] == 1
    assert state["metrics"]["exact_hit_rate"] == 1.0
    assert state["timeline"][-1]["event_type"] == "graded"


def test_live_shadow_api_request_round_trips_state_and_jsonl_export():
    session = LiveShadowSession(session_id="demo")

    status, observed = handle_live_shadow_api_request(
        session,
        "POST",
        "/api/observe",
        {"role": "user", "content": "Could you help me decide what to do next?"},
    )
    assert status == 200
    assert observed["active_pack"]["prepared_drafts"]

    status, graded = handle_live_shadow_api_request(
        session,
        "POST",
        "/api/grade",
        {
            "content": "One good next step is to write down the two options.",
            "actual_response_mode": "suggest",
        },
    )
    assert status == 200
    assert graded["grades"][-1]["actual_response_mode"] == "suggest"

    status, export = handle_live_shadow_api_request(session, "GET", "/api/export", None)
    assert status == 200
    rows = [json.loads(line) for line in str(export["jsonl"]).splitlines()]
    assert rows[0]["session_id"] == "demo"
    assert rows[0]["actual_response_mode"] == "suggest"
    assert "prepared_drafts" in rows[0]


def test_live_shadow_infers_response_mode_when_label_is_not_provided():
    assert infer_response_mode_from_text("I am sorry that happened.") == "apologize"
    assert infer_response_mode_from_text("Can you tell me more about that?") == "ask_followup"
    assert infer_response_mode_from_text("I will take care of that now.") == "commit"


def test_live_shadow_lab_html_exposes_experiment_lanes_and_api_hooks():
    html = render_live_shadow_lab_html()

    assert "Premonition Live Shadow Lab" in html
    assert "Live Conversation" in html
    assert "Premonition Drafts" in html
    assert "Reality Grading" in html
    assert "/api/observe" in html
    assert "/api/grade" in html
    assert "/api/export" in html


def test_live_shadow_app_serves_empty_favicon_without_console_404():
    server = ThreadingHTTPServer(("127.0.0.1", 0), LiveShadowHTTPRequestHandler)
    try:
        host, port = server.server_address
        import threading

        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = HTTPConnection(host, port)
        connection.request("GET", "/favicon.ico")
        response = connection.getresponse()

        assert response.status == 204
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
