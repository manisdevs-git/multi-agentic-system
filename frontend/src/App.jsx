import { useMemo, useState } from "react";
import { approveSession, createSession, previewMail, sendMail } from "./api.js";

const SAMPLE = "Why was I charged an exit load on HDFC Large Cap Fund?";

export default function App() {
  const [complaint, setComplaint] = useState(SAMPLE);
  const [state, setState] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [mailOpen, setMailOpen] = useState(false);
  const [to, setTo] = useState("");
  const [preview, setPreview] = useState(null);

  const draft = state?.draft;
  const note = draft?.routing_note;
  const pattern = state?.pattern;
  const review = state?.review;
  const passed = review?.status === "pass";
  const blocked = review?.status === "blocked";
  const openTicket = draft && ["escalate", "disagree"].includes(draft.disposition);
  const words = useMemo(
    () => (complaint.trim() ? complaint.trim().split(/\s+/).length : 0),
    [complaint]
  );

  async function run() {
    setError("");
    if (!complaint.trim()) {
      setError("Paste a complaint first.");
      return;
    }
    setBusy(true);
    try {
      const next = await createSession(complaint.trim());
      setState(next);
      setTo(next.default_to || "");
      setPreview(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function onApprove() {
    setBusy(true);
    setError("");
    try {
      setState(await approveSession(state.session_id));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function openPreview() {
    setBusy(true);
    setError("");
    try {
      const next = await previewMail(state.session_id, to);
      setState(next);
      setPreview(next.preview);
      setMailOpen(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function onSend() {
    setBusy(true);
    setError("");
    try {
      const next = await sendMail(state.session_id, to);
      setState(next);
      setPreview(next.preview);
      setMailOpen(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <div className="mark">↗</div>
          <span className="title">Support reply</span>
          <span className="pill">MF OPS</span>
        </div>
        {state ? (
          <button className="secondary" type="button" onClick={() => setState(null)}>
            Clear
          </button>
        ) : null}
      </header>

      {!state ? (
        <>
          <h1>Paste a complaint, then run.</h1>
          <p className="lede">
            Direct complaint ingestion for mutual fund operations. The graph runs on the API, not in the browser.
          </p>
          <label htmlFor="complaint">Customer message</label>
          <textarea
            id="complaint"
            value={complaint}
            onChange={(event) => setComplaint(event.target.value)}
          />
          <div className="row">
            <p className="hint">
              PAN and phone numbers are masked before agents run. · {words} words · {complaint.length} characters
            </p>
            <button className="primary" type="button" onClick={run} disabled={busy}>
              {busy ? "Running…" : "Run"}
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="badge">Reviewer {review?.status || "—"}</div>
          <div className="grid">
            <section>
              <h2>Reply</h2>
              <p className="kicker">Draft message to customer</p>
              <div className="card">
                <p className="reply">{draft?.reply}</p>
                {draft?.citation ? (
                  <p>
                    Primary source:{" "}
                    <a href={draft.citation} target="_blank" rel="noreferrer">
                      {draft.citation}
                    </a>
                  </p>
                ) : null}
                <p className="hint">
                  {passed ? "SEBI scheme disclosure validation: Compliant" : `Review: ${review?.status}`}
                </p>
              </div>
            </section>
            <section>
              <h2>Internal</h2>
              <p className="kicker">Operational routing metadata</p>
              <div className="card">
                <div className="meta">
                  <div>
                    <span>Team</span>
                    <strong>{note?.team || "—"}</strong>
                  </div>
                  <div>
                    <span>Priority</span>
                    <strong>{note?.priority || "—"}</strong>
                  </div>
                  <div>
                    <span>Theme</span>
                    {pattern ? `${pattern.theme_name} · ${pattern.count} reviews` : "—"}
                  </div>
                  <div>
                    <span>Status line</span>
                    <strong>{openTicket ? "Ticket stays open" : "May close after reply"}</strong>
                  </div>
                </div>
                <p className="hint">Facts · Pattern · Compliance · 3 specialists executed</p>
                <p className="hint">Internal is not emailed.</p>
              </div>
            </section>
          </div>
          <div className="row">
            <div>
              {blocked ? (
                <p className="error">Blocked. Approve is not available.</p>
              ) : state.approved ? (
                <p className="ok">Approved. Draft saved. Does not send until Preview mail → Send.</p>
              ) : (
                <p className="hint">Saves a draft. Does not send.</p>
              )}
              {state.sent ? (
                <p className="ok">
                  Sent. message_id={state.sent.message_id || "—"} document_id={state.sent.document_id || "—"}
                </p>
              ) : null}
            </div>
            <div className="actions">
              <button
                className="secondary"
                type="button"
                disabled={!state.approved || busy}
                onClick={openPreview}
              >
                Preview mail
              </button>
              <button
                className="primary"
                type="button"
                disabled={blocked || state.approved || !state.approvable || busy}
                onClick={onApprove}
              >
                {state.approved ? "Approved" : "Approve"}
              </button>
            </div>
          </div>
          <details>
            <summary>Trace</summary>
            <pre>{state.trace || "—"}</pre>
          </details>
        </>
      )}

      {error ? <p className="error">{error}</p> : null}

      {mailOpen && preview ? (
        <div className="modal-back" role="dialog">
          <div className="modal">
            <h2>Mail preview</h2>
            <label htmlFor="to">To</label>
            <input id="to" value={to} onChange={(event) => setTo(event.target.value)} />
            <label htmlFor="subject">Subject</label>
            <input id="subject" value={preview.subject} readOnly />
            <label htmlFor="body">Body</label>
            <textarea id="body" value={preview.body} readOnly />
            <p className="hint">Saves nothing extra until Send. Sends from the MCP Gmail workspace account.</p>
            <div className="actions">
              <button className="secondary" type="button" onClick={() => setMailOpen(false)}>
                Cancel
              </button>
              <button className="primary" type="button" onClick={onSend} disabled={busy}>
                Send
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
