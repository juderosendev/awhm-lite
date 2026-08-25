"""Tests for WAL manager."""

from awhm.session_buffer.buffer import SessionBuffer
from awhm.session_buffer.wal import WALManager


def test_flush_and_recover(config):
    buf = SessionBuffer()
    buf.process_message("I prefer dark mode", "2024-01-01T00:00:00Z", 0)
    buf.process_message("Actually, it's port 3000", "2024-01-01T00:01:00Z", 1)

    wal = WALManager(config, buf, session_id="s1")
    wal.flush()

    assert config.wal_path_for_session("s1").exists()

    # Recover into a new buffer
    buf2 = SessionBuffer()
    wal2 = WALManager(config, buf2, session_id="s1")
    count = wal2.recover()
    assert count >= 2
    assert len(buf2.entries) >= 2


def test_clear_wal(config):
    buf = SessionBuffer()
    wal = WALManager(config, buf, session_id="s1")
    wal.flush()
    assert config.wal_path_for_session("s1").exists()
    wal.clear_wal()
    assert not config.wal_path_for_session("s1").exists()


def test_session_scoped_recovery(config):
    buf1 = SessionBuffer()
    buf1.process_message("I prefer dark mode", "2024-01-01T00:00:00Z", 0)
    wal1 = WALManager(config, buf1, session_id="s1")
    wal1.flush()

    buf2 = SessionBuffer()
    buf2.process_message("Actually, use port 8080", "2024-01-01T00:01:00Z", 0)
    wal2 = WALManager(config, buf2, session_id="s2")
    wal2.flush()

    recovered_s1 = SessionBuffer()
    count_s1 = WALManager(config, recovered_s1, session_id="s1").recover()
    assert count_s1 == 1
    assert len(recovered_s1.entries) == 1
    assert "dark mode" in recovered_s1.entries[0].content.lower()


def test_flush_skips_when_unchanged(config):
    buf = SessionBuffer()
    buf.process_message("I prefer dark mode", "2024-01-01T00:00:00Z", 0)
    wal = WALManager(config, buf, session_id="s1")
    assert wal.flush() is True
    assert wal.flush() is False
    buf.process_message("Actually, use port 8080", "2024-01-01T00:01:00Z", 1)
    assert wal.flush() is True
    assert wal.flush(force=True) is True
