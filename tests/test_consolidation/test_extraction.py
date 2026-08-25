"""Tests for rule-based extraction."""

from awhm.consolidation.extraction import extract_from_log_entries
from awhm.raw_log.models import LogEntry


def test_extract_corrections():
    entries = [
        LogEntry(timestamp="2024-01-01T00:00:00Z", session_id="s1", role="user",
                 content="Actually, the port is 8080"),
    ]
    extractions = extract_from_log_entries(entries)
    assert any(e.type == "correction" for e in extractions)


def test_extract_preferences():
    entries = [
        LogEntry(timestamp="2024-01-01T00:00:00Z", session_id="s1", role="user",
                 content="I prefer TypeScript over JavaScript"),
    ]
    extractions = extract_from_log_entries(entries)
    assert any(e.type == "preference" for e in extractions)


def test_extract_deduplicates():
    entries = [
        LogEntry(timestamp="2024-01-01T00:00:00Z", session_id="s1", role="user",
                 content="Actually, the port is 8080"),
        LogEntry(timestamp="2024-01-01T00:01:00Z", session_id="s1", role="user",
                 content="Actually, the port is 8080"),
    ]
    extractions = extract_from_log_entries(entries)
    # Should deduplicate
    correction_count = sum(1 for e in extractions if e.type == "correction")
    assert correction_count == 1
