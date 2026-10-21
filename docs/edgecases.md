# Edge cases: Groww support-reply desk

The desk answers from five HDFC scheme pages and a frozen Play Store theme index. It does **not** explain a user's ledger. These cases are the ones that look like a fact question and are not.

Machine-readable twins live in `evals/cases.json`. Scoring is in `docs/evals.md`.

## What "a useful reply" means

A customer-facing draft is useful when it:

1. Copies a **corpus sentence** (or says the corpus does not cover the question).
2. Carries **one** Groww `source_url` and `Last checked 25-Jun-2026` when it states a scheme rule.
3. Does **not** confirm a number the user alleged unless that exact figure is in the cited block.
4. Does **not** advise (redeem / hold / "good fund" / expected returns).
5. Leaves the ticket **open** when Pattern says the theme is recurring or rising.

The Gold FoF "why 2% exit load?" thread is the type specimen: the page says **1% if redeemed within 15 days**. The 2% is a dispute, not a fact the desk may invent.

## Corpus traps

| Case | Input shape | Correct behavior |
|---|---|---|
| Alleged fee ≠ SID | Gold/Silver FoF + "2% exit load" | Quote `Exit load of 1%, if redeemed within 15 days.` Do not write 2% as the rule. Escalate (`fees_brokerage` is recurring). |
| Equity window mix-up | Gold FoF + "1 year" | Still the 15-day sentence. Never Large/Mid/Small Cap's "within 1 year". |
| Equity alleged 2% | Large Cap + "2% exit load" | Quote `Exit load of 1% if redeemed within 1 year`. |
| Two HDFC schemes | Gold FoF **and** Large Cap | `not_covered`, handoff, no exit-load figure. |
| Other AMC | Nippon / SBI / ICICI … even if HDFC is also named | `not_covered`. Do not cite HDFC as the answer to a comparison. |
| No scheme | "Why 2% exit load?" | `not_covered`, handoff, no 1% and no 2%. |
| Wrong section | "NAV history", CAGR, "is it good" | Not the Key fund facts NAV paragraph. History → uncover. Advice/returns → **refuse**. |
| SIP vs NAV | "minimum SIP" on a named scheme | Minimum investments, not the NAV paragraph. |
| Tax vs exit load | "how is Gold FoF taxed" | Stamp duty and tax block (may contain 12.5%). That is a cited tax rule, not a performance claim. |
| Expense vs 2% | "2% expense ratio" on Gold FoF | Key fund facts; expense on the page is **0.2%**. Do not confirm 2%. |

## Pattern and routing

| Case | Correct behavior |
|---|---|
| Exit load / charged / brokerage | `fees_brokerage` even if the same sentence mentions a crash or an order (cue order). Recurring → **escalate**, P2, ticket stays open. |
| Withdrawal / UPI | `money_movement`. Uncovered scheme pages → handoff, not an invented NAV. |
| Theme `other` + covered fact | Team `Mutual fund support ({scheme_id})`. |
| Rising (≥40 recent or 2× weekly rate) | **disagree**, P1, recent count in the routing note. |

A covered fact plus a recurring fee theme is **not** a close. The reply states the rule; ops still owns the debit.

## Compliance and PII

| Case | Correct behavior |
|---|---|
| "Should I redeem …" / "is it good" / "what returns" | `refuse`. Sticky for the rest of the session, including a later clean exit-load ask. |
| "I was charged" / "why 2% charged" | **Not** advice. Ledger complaint. |
| PAN, phone, email, Aadhaar, OTP, account | Ingress tokens only. Reviewer `pass` on the redacted draft. Approve writes tokens, never the raw values. |
| Empty / whitespace complaint | `not_covered` + **handoff**. Not a refusal. |

## Reviewer, gate, refinement

| Case | Correct behavior |
|---|---|
| Uncited figure or advice in a draft | Round 1 `fail` → rewrite once (orchestrator only) → round 2 `pass` or `blocked`. |
| `blocked` | Not approvable. No `records.jsonl` line, no email file. |
| Sixth pushback | `RefinementRefused`. Last reviewed draft stands. |
| "They're asking about Large Cap" | Facts re-runs; Pattern/Compliance reused. |
| Approve | Human flag required. No SMTP from `approve()`. Streamlit then **Preview mail** → **Send** (MCP). |

## Live model vs cues

DeepInfra may flag a fee complaint as `advice`. The **eval suite is cue-path / no key**: a fee question must not refuse. If a live run refuses "why 2% charged for exit load", that is a model miss, not the documented desk policy.
