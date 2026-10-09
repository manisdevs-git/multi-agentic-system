# Problem Statement: Multi-Agent Support Reply System for Groww

## Goal

Build a multi-agent system for the [Groww](https://play.google.com/store/apps/details?id=com.nextbillion.groww&hl=en_IN) app that takes a real user question or complaint and drafts a support reply.

- Three specialist agents run **in parallel**: one finds the facts, one checks whether the complaint is a known pattern, one checks the draft won't cause compliance trouble.
- An **orchestrator** merges their outputs into a reply and an internal routing note.
- A **reviewer** blocks anything that breaks the facts-only rule before a human ever sees it.

## The Problem

A support agent answering *"why was I charged an exit load"* needs three things at once, and they don't come from the same place:

1. **The fact** lives in a scheme document.
2. **The pattern**: whether forty other users asked the same thing this month decides if this is a reply or an escalation.
3. **Compliance**: the draft must not give investment advice, must not quote a return figure, and must not echo back a PAN the user pasted into their complaint.

A single model asked to do all three does the first and forgets the rest. It writes a fluent, uncited paragraph ending in *"you may want to consider staying invested longer"*, which is advice. That turns a support ticket into a compliance problem.

Each specialist still gets its own model call. The model only classifies. Facts code copies the sentence from the scheme file, Pattern code counts the reviews, and Compliance code redacts PII. The three calls are three nodes on one LangGraph, started together.

## Input

The raw Play Store CSV is no longer in the repo. `data/` now holds the Milestone 1 corpus and the Milestone 2 review analysis. Agents read these files; they do not scrape or fall back on general knowledge.

### Facts corpus — `data/parsed/`

Public Groww scheme pages, parsed on **2026-06-28** (`parse_manifest.json`, `scheme_count: 5`). Each scheme file is the citation source for the Facts Agent. `source_url` is the one citation link; `nav_date` / `fetched_at` is the "last checked" date.

| `scheme_id` | Scheme | `source_url` | Exit load |
|---|---|---|---|
| `large_cap` | HDFC Large Cap Fund Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` | 1% if redeemed within 1 year |
| `mid_cap` | HDFC Mid Cap Fund Direct Growth | `https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth` | 1% if redeemed within 1 year |
| `small_cap` | HDFC Small Cap Fund Direct Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` | 1% if redeemed within 1 year |
| `gold_fof` | HDFC Gold ETF Fund of Fund Direct Plan Growth | `https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth` | 1% if redeemed within 15 days |
| `silver_fof` | HDFC Silver ETF FoF Direct Growth | `https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth` | 1% if redeemed within 15 days |

NAV date on every scheme: **25-Jun-2026**. Each file has 7 `text_blocks`: Key fund facts (NAV, expense ratio, min SIP/lumpsum, AUM, riskometer, category, plan), Exit load, Benchmark, Investment objective, Stamp duty and tax, Fund management, Minimum investments.

Anything outside these five schemes, or outside these sections, is **not covered**.

### Reviews — `data/normalized_reviews.json`

1,650 Play Store reviews (`source: "play_store"`), dated **2026-04-12** through **2026-07-04**.

Fields: `id`, `source`, `rating` (1–5), `title`, `text`, `date`, `version` (app versions such as `18.10.1`; some null).

Rating split: 680×1, 95×2, 113×3, 157×4, 605×5. This is the complaint pool for the 20-review acceptance run and for the Pattern Agent's counts.

### Themes — `data/analyzed_themes.json`

Milestone 2 output over the same 1,650 reviews. Top-level keys: `reviewCount`, `allThemes`, `topThemes`.

Each theme has `id`, `name`, `description`, `reviewIds`, `count`, `avgRating`, `score`. The Pattern Agent matches a complaint against `allThemes` and reports the count and the review window (2026-04-12 to 2026-07-04).

| `id` | Name | Count | Avg rating | Score | In `topThemes` |
|---|---|---:|---:|---:|---|
| `other` | Other feedback | 709 | 3.5 | 1210.85 | no |
| `trading_charts` | Trading, charts & orders | 503 | 2.7 | 993.96 | yes |
| `money_movement` | Withdrawals, deposits & payments | 162 | 2.5 | 338.98 | yes |
| `reliability` | App stability & performance | 140 | 2.2 | 302.36 | yes |
| `fees_brokerage` | Fees & brokerage | 136 | 2.2 | 302.26 | no |

`topThemes` is the three highest-scoring themes excluding `other`: `trading_charts`, `money_movement`, `reliability`. `fees_brokerage` is a real theme and is the one an exit-load or brokerage complaint should match, even though it is not in `topThemes`.

## Architecture / What to Build

### Three agents, in parallel

Facts, Pattern, and Compliance are nodes of one LangGraph. None of them waits on or reads another's output. The graph starts all three after redaction and waits until all three have written their result.

Each node calls a model with the redacted complaint and a closed list. The model returns an id or a flag. Code in that same node writes the JSON output. The model does not see another node's result, the draft, or a raw PAN, phone, or email.

| Agent | Model returns | Code writes |
|---|---|---|
| **Facts Agent** | One `scheme_id` and one section title from the five scheme files, or neither. | The answer copied from that `text_block`, with **one citation link** (`source_url`). If either pick is missing, `"not covered"` and the schemes and sections searched. |
| **Pattern Agent** | One `theme_id` from `allThemes` in `data/analyzed_themes.json`. | Whether the issue is **recurring**, **rising**, or a **one-off**, plus the theme `id`, the **count** of `reviewIds` behind it, and the **window** (2026-04-12 to 2026-07-04). |
| **Compliance Agent** | `advice`, `performance_claim`, or neither, from the incoming message. | Those flags, plus a PII flag when the redaction scan finds one. PAN, Aadhaar, account numbers, phone numbers, emails, and OTPs are redacted **before anything is stored**. The outgoing draft is checked by the reviewer, because the draft does not exist yet at this step. |

### Orchestrator

Produces two outputs:

1. **Reply draft**: three sentences or fewer, facts-only, with its citation link and a **"last checked"** date.
2. **Internal routing note**: which team the ticket goes to and why, using the Pattern Agent's signal to set priority.

When agents disagree (e.g. Facts has a clean answer but Pattern shows forty people asking the same thing this week), the orchestrator **says so** rather than sending the reply and closing the ticket.

### Reviewer

Checks the draft before a human sees it. Hard fails:

- An uncited factual claim
- Anything that reads as advice
- Any surviving PII

On a hard fail, the draft is sent back to the orchestrator **once**, with the specific failure named. If the second attempt still fails, show the draft with the flag on it and **do not offer the send**.

### Refinement

The human pushes back and **only the affected agents re-run**:

- *"Shorter"* → Orchestrator
- *"They're asking about a different scheme"* → Facts Agent
- *"Check if this is the fee issue from last week"* → Pattern Agent

Log which agents fired on each refinement.

### Approval-gated MCP

Same gate as Milestone 2. Nothing moves without a human clicking **approve**. After approval:

- Append the record to the doc
- Create the email draft

**No auto-send.**

## Rules

1. **Facts-only.** No recommendations, no buy/sell language, no return or performance figures.
2. **Every factual claim carries one citation link.** No link means it doesn't ship.
3. **No PII stored anywhere**, including in logs and agent state.
4. **Public sources only**, same corpus as Milestone 1.
5. **Three agents run in parallel**, shown in the diagram and the demo.
6. **Reviewer rework capped at one round; refinement capped at five.**
7. **Reply drafts are three sentences or fewer**, with a "last checked" date.
8. **One model call per agent node.** The call returns an id or a flag from that node's list. It does not write the fact sentence, the review count, the signal, or the reply.

## Before Submitting (Acceptance Checks)

- **Run 20 real complaints** from `data/normalized_reviews.json`. Confirm all three agents returned structured output on every one, and that every reply either carries a `source_url` citation or says the corpus doesn't cover it.
- **Check the branches.** Feed in:
  - one complaint the corpus answers cleanly
  - one it can't answer
  - one matching a recurring theme
  - one containing a PAN or phone number

  Log which agents fired on each and confirm the reviewer caught the PII.
- **Try to make it give advice.** Ask whether to redeem, whether a scheme is good, what returns to expect. Rephrase, then ask again later in the same session. It should refuse every time.

## Dependencies on Prior Milestones

- **Milestone 1** is already materialized as `data/parsed/`. The Facts Agent cites `source_url` from those files and does not fetch new pages.
- **Milestone 2** is already materialized as `data/normalized_reviews.json` and `data/analyzed_themes.json`. The Pattern Agent matches against those themes. The approval-gated MCP flow from Milestone 2 is reused here: nothing is appended or emailed until a human clicks approve.
