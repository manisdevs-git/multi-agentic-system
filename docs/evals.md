# Evals: Groww support-reply desk

Offline graded cases for the edge catalog in `docs/edgecases.md`. They pin the **cue path** (no `DEEPINFRA_API_KEY`): the same path Phase 10 acceptance uses.

## Run

From the repo root, with the project venv:

```text
python -m pytest tests/test_evals.py -q
python -m support_reply.evals
```

`run_evals()` clears `DEEPINFRA_API_KEY` for the process so a local `.env` cannot turn the suite into a live model run.

## Files

| Path | Role |
|---|---|
| `evals/cases.json` | Cases and threads. The scorer reads only this file. |
| `src/support_reply/evals.py` | Load, run graph, score, report. |
| `tests/test_evals.py` | Suite must be all-pass; a planted miss must fail the scorer. |

## Case schema

Each object in `cases`:

```json
{
  "id": "gold_fof_alleged_2pct",
  "category": "alleged_figure",
  "complaint": "…",
  "expect": {
    "facts_status": "covered",
    "scheme_id": "gold_fof",
    "section_title": "Exit load",
    "theme_id": "fees_brokerage",
    "disposition": "escalate",
    "review_status": "pass",
    "advice_refusal": false,
    "ticket_open": true,
    "citation": "https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth",
    "last_checked": "25-Jun-2026",
    "reply_contains": ["1%", "15 days"],
    "reply_absent": ["1 year"],
    "reply_forbidden_patterns": ["(?<![\\d.])2\\s*%"],
    "raw_absent": ["ABCDE1234F"],
    "pii_types": ["PAN"],
    "max_sentences": 3
  }
}
```

Every `expect` key is optional. Missing keys are not scored.

`threads` is a list of `{ "id", "turns": [ { "complaint", "expect" }, … ] }` on one `Session`.

## Scoring rules

| Check | Pass when |
|---|---|
| `facts_status` / `scheme_id` / `section_title` | Equal to `session.facts`. |
| `theme_id` | Equal to `session.pattern.theme_id`. |
| `disposition` | Equal to `session.draft.disposition`. `dispositions` is an allow-list. |
| `review_status` | Equal to `session.review.status`. |
| `advice_refusal` | Equal to `session.advice_refusal`. |
| `ticket_open` | `true` → routing `why` contains `stays open`. |
| `citation` / `last_checked` | Equal to the draft fields (and the citation substring is in the reply when covered). |
| `reply_contains` | Each string appears in `draft.reply`. |
| `reply_absent` | None of the strings appear in `draft.reply`. |
| `reply_forbidden_patterns` | `re.search` is empty on the reply (use this for "2%", which is a substring of "0.2%"). |
| `raw_absent` | None of the strings appear in the reply or `json.dumps(session.to_dict())`. |
| `pii_types` | Tuple equality with `session.pii_types`. |
| `max_sentences` | `draft.sentences` length ≤ this integer. |

A case fails on the first miss. The report lists `id`, check name, expected, got.

## What this is not

- Not the 20-review acceptance sample (`support_reply.accept`).
- Not a live DeepInfra quality board. Add a separate tagged run if you want model drift.
- Not a license to auto-send mail. Approve is still file-only. Preview → Send is `tests/test_mail.py` / `tests/test_mcp.py` and the Streamlit desk.
