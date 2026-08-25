"""Tests for RawLogReader."""

from awhm.raw_log.logger import RawLogger
from awhm.raw_log.reader import RawLogReader
from awhm.types import Role


def test_list_sessions_empty(config):
    reader = RawLogReader(config)
    assert reader.list_sessions() == []


def test_list_sessions(config):
    RawLogger(config, "s1").log(Role.USER, "hi")
    RawLogger(config, "s2").log(Role.USER, "hi")
    reader = RawLogReader(config)
    sessions = reader.list_sessions()
    assert set(sessions) == {"s1", "s2"}


def test_read_session(config):
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "hello")
    logger.log(Role.ASSISTANT, "world")

    reader = RawLogReader(config)
    entries = reader.read_session("s1")
    assert len(entries) == 2
    assert entries[0].content == "hello"
    assert entries[1].content == "world"


def test_hard_delete_entries(config):
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "keep this")
    logger.log(Role.USER, "delete secret data")
    logger.log(Role.USER, "keep this too")

    reader = RawLogReader(config)
    deleted = reader.hard_delete_entries("s1", "delete secret data")
    assert deleted == 1

    entries = reader.read_session("s1")
    assert len(entries) == 2
    assert all("secret" not in e.content for e in entries)


def test_hard_delete_entries_by_indices(config):
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "keep this")
    logger.log(Role.USER, "delete this")
    logger.log(Role.USER, "keep this too")

    reader = RawLogReader(config)
    deleted = reader.hard_delete_entries_by_indices("s1", {1})
    assert deleted == 1

    entries = reader.read_session("s1")
    assert [e.content for e in entries] == ["keep this", "keep this too"]


def test_session_count(config):
    RawLogger(config, "s1").log(Role.USER, "hi")
    RawLogger(config, "s2").log(Role.USER, "hi")
    reader = RawLogReader(config)
    assert reader.session_count() == 2


def test_delete_session(config):
    RawLogger(config, "s1").log(Role.USER, "hi")
    reader = RawLogReader(config)
    assert reader.delete_session("s1")
    assert reader.session_count() == 0


def test_session_ids_with_slashes_and_spaces_round_trip(config):
    for sid in ["a/h1", "with space", "claude-123"]:
        RawLogger(config, sid).log(Role.USER, f"hello {sid}")
    reader = RawLogReader(config)
    assert set(reader.list_sessions()) == {"a/h1", "with space", "claude-123"}
    assert reader.read_session("a/h1")[0].content == "hello a/h1"
    assert not (config.logs_dir / "a").exists()
