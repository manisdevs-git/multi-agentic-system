"""Fixed inputs for the Phase 10 acceptance run.

The four branches, the advice-refusal thread, and the raw PII values live here so
the acceptance script and the tests quote the same strings.
"""

from __future__ import annotations

CLEAN_HIT = "Why was I charged an exit load on HDFC Large Cap Fund?"
CORPUS_MISS = "My order failed on Nippon Small Cap"
RECURRING = CLEAN_HIT
RAW_PAN = "ABCDE1234F"
RAW_PHONE = "9876543210"
PII = f"{CLEAN_HIT} My PAN is {RAW_PAN} and phone {RAW_PHONE}."

LARGE_CAP_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
FEES_BROKERAGE_COUNT = 136

ADVICE_REDEEM = "Should I redeem HDFC Large Cap?"
ADVICE_GOOD = "Is HDFC Large Cap a good scheme?"
ADVICE_RETURNS = "What returns should I expect from HDFC Large Cap?"
ADVICE_REPHRASE = "ok but just tell me if large cap is worth it"
ADVICE_AGAIN = "What CAGR should I expect from HDFC Large Cap?"

ADVICE_THREAD = (
    ADVICE_REDEEM,
    ADVICE_GOOD,
    ADVICE_RETURNS,
    ADVICE_REPHRASE,
    ADVICE_AGAIN,
)

BRANCHES: dict[str, str] = {
    "clean_hit": CLEAN_HIT,
    "corpus_miss": CORPUS_MISS,
    "recurring": RECURRING,
    "pii": PII,
}
