"""Tests for SessionBuffer."""

from awhm.session_buffer.buffer import SessionBuffer
from awhm.types import BufferEntryType


def test_process_message_captures_correction():
    buf = SessionBuffer()
    entries = buf.process_message("Actually, the server is on port 3000", "2024-01-01T00:00:00Z", 0)
    assert len(entries) >= 1
    assert any(e.type == BufferEntryType.CORRECTION.value for e in entries)


def test_process_message_no_match():
    buf = SessionBuffer()
    entries = buf.process_message("What is the weather today?", "2024-01-01T00:00:00Z", 0)
    assert len(entries) == 0


def test_search():
    buf = SessionBuffer()
    buf.process_message("I prefer dark mode", "2024-01-01T00:00:00Z", 0)
    buf.process_message("The database is PostgreSQL", "2024-01-01T00:01:00Z", 1)

    results = buf.search("dark mode")
    assert len(results) >= 1


def test_get_by_type():
    buf = SessionBuffer()
    buf.process_message("I prefer dark mode", "2024-01-01T00:00:00Z", 0)
    buf.process_message("Actually, it should be port 8080", "2024-01-01T00:01:00Z", 1)

    prefs = buf.get_by_type(BufferEntryType.PREFERENCE)
    assert len(prefs) >= 1

    corrections = buf.get_by_type(BufferEntryType.CORRECTION)
    assert len(corrections) >= 1


def test_clear():
    buf = SessionBuffer()
    buf.process_message("I prefer dark mode", "2024-01-01T00:00:00Z", 0)
    assert len(buf.entries) > 0
    buf.clear()
    assert len(buf.entries) == 0
