"""HTTP API: Vercel talks to this; the graph stays on the server."""

from fastapi.testclient import TestClient

from support_reply.api import app, reset_store
from support_reply.mail import SendResult

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"
RAW_PAN = "ABCDE1234F"


class _FakeMcp:
    def initialize(self) -> dict:
        return {}

    def close(self) -> None:
        return None

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name == "reflect_content":
            return {"success": True, "document_id": "d-1"}
        if name == "send_email":
            return {"success": True, "message_id": "m-1"}
        raise AssertionError(name)


def setup_function() -> None:
    reset_store()
    app.state.mail_client = _FakeMcp()


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_create_session_runs_the_graph_and_hides_raw_pii(tmp_path, monkeypatch):
    monkeypatch.setenv("OUT_DIR", str(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions",
            json={"complaint": f"{EXIT_LOAD} PAN {RAW_PAN}"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert payload["draft"]["reply"]
    assert payload["review"]["status"] in {"pass", "fail", "blocked"}
    assert RAW_PAN not in response.text
    assert "[PAN]" in payload["redacted_text"]
    assert payload["approved"] is False


def test_approve_then_preview_then_send(tmp_path, monkeypatch):
    monkeypatch.setenv("OUT_DIR", str(tmp_path))
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"complaint": EXIT_LOAD}).json()
        sid = created["session_id"]
        approved = client.post(f"/v1/sessions/{sid}/approve")
        assert approved.status_code == 200
        assert approved.json()["approved"] is True
        preview = client.post(
            f"/v1/sessions/{sid}/mail/preview",
            json={"to": "ops@example.com"},
        )
        assert preview.status_code == 200
        body = preview.json()["preview"]
        assert body["to_address"] == "ops@example.com"
        assert body["body"] == created["draft"]["reply"]
        sent = client.post(
            f"/v1/sessions/{sid}/mail/send",
            json={"to": "ops@example.com"},
        )
    assert sent.status_code == 200
    assert sent.json()["sent"]["message_id"] == "m-1"


def test_send_before_approve_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OUT_DIR", str(tmp_path))
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"complaint": EXIT_LOAD}).json()
        sid = created["session_id"]
        response = client.post(
            f"/v1/sessions/{sid}/mail/send",
            json={"to": "ops@example.com"},
        )
    assert response.status_code == 409
