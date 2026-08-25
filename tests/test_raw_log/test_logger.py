"""Tests for RawLogger."""

import json
from awhm.raw_log.logger import RawLogger
from awhm.types import Role


def test_log_creates_file(config):
    logger = RawLogger(config, "test-session")
    logger.log(Role.USER, "Hello world")
    assert logger.path.exists()


def test_log_appends_jsonl(config):
    logger = RawLogger(config, "test-session")
    logger.log(Role.USER, "First message")
    logger.log(Role.ASSISTANT, "Second message")

    with open(logger.path) as f:
        lines = f.readlines()
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["role"] == "user"
    assert entry1["content"] == "First message"

    entry2 = json.loads(lines[1])
    assert entry2["role"] == "assistant"


def test_log_increments_index(config):
    logger = RawLogger(config, "test-session")
    assert logger.msg_index == 0
    logger.log(Role.USER, "msg")
    assert logger.msg_index == 1
    logger.log(Role.ASSISTANT, "msg")
    assert logger.msg_index == 2


def test_log_metadata(config):
    logger = RawLogger(config, "test-session")
    logger.log(Role.TOOL_CALL, "run test", metadata={"tool_name": "pytest"})

    with open(logger.path) as f:
        entry = json.loads(f.readline())
    assert entry["metadata"]["tool_name"] == "pytest"
