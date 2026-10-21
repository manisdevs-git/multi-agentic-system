"""Phase 10 exit: 20 reviews, four branches, advice thread, parallel proof, caps.

The acceptance module is the thing under test. It must pass with no network.
"""

import json

import pytest

from tests.fixtures import (
    ADVICE_THREAD,
    BRANCHES,
    CLEAN_HIT,
    CORPUS_MISS,
    LARGE_CAP_URL,
    PII,
    RAW_PAN,
    RAW_PHONE,
)


def test_fixtures_name_the_four_branches_and_the_advice_thread():
    assert BRANCHES["clean_hit"] == CLEAN_HIT
    assert BRANCHES["corpus_miss"] == CORPUS_MISS
    assert "Nippon" in CORPUS_MISS
    assert RAW_PAN in PII and RAW_PHONE in PII
    assert LARGE_CAP_URL.startswith("https://groww.in/")
    assert len(ADVICE_THREAD) == 5


def test_acceptance_run_passes_offline_and_writes_only_approved_records(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    from support_reply.accept import run_acceptance

    report = run_acceptance(out_dir=tmp_path)

    assert report.ok is True
    assert len(report.review_ids) == 20
    assert len(set(report.review_ids)) == 20
    assert set(report.branches) == set(BRANCHES)
    records_path = tmp_path / "records.jsonl"
    assert records_path.is_file()
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert records
    dumped = records_path.read_text(encoding="utf-8")
    assert RAW_PAN not in dumped and RAW_PHONE not in dumped
    assert "[PAN]" in dumped
    email_dir = tmp_path / "email_drafts"
    assert email_dir.is_dir()
    for path in email_dir.iterdir():
        body = path.read_text(encoding="utf-8")
        assert RAW_PAN not in body and RAW_PHONE not in body
