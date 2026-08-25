"""Tests for the Claude Code hook commands."""

import io
import json

from awhm.config import AWHMConfig
from awhm.hooks import cmd_prompt, cmd_session_end, cmd_stop, run, settings_snippet
from awhm.raw_log.reader import RawLogReader


def _payload(session="abc", **extra):
    base = {"session_id": session, "transcript_path": "/tmp/t.jsonl", "cwd": "/tmp"}
    base.update(extra)
    return base


def test_prompt_logs_and_injects_across_processes(tmp_path):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"))

    first = cmd_prompt(_payload(user_prompt="My name is Alice and I prefer dark mode"), config, use_mock=True)
    # Nothing to recall yet beyond the prompt itself (which is deduplicated away)
    assert first is None or "Alice" in json.dumps(first)

    cmd_stop(_payload(last_assistant_message="Nice to meet you, Alice."), config, use_mock=True)

    second = cmd_prompt(_payload(user_prompt="What theme do I prefer?"), config, use_mock=True)
    assert second is not None
    ctx = second["hookSpecificOutput"]["additionalContext"]
    assert second["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "dark mode" in ctx

    entries = RawLogReader(config).read_session("claude-abc")
    roles = [e.role for e in entries]
    assert roles == ["user", "assistant", "user"]


def test_stop_ignores_duplicate_fires(tmp_path):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"))
    cmd_stop(_payload(last_assistant_message="Done."), config, use_mock=True)
    cmd_stop(_payload(last_assistant_message="Done."), config, use_mock=True)
    entries = RawLogReader(config).read_session("claude-abc")
    assert [e.content for e in entries] == ["Done."]


def test_session_end_consolidates(tmp_path):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"))
    cmd_prompt(_payload(user_prompt="The API endpoint is https://api.example.com"), config, use_mock=True)
    count = cmd_session_end(_payload(), config, use_mock=True)
    assert count >= 1
    assert not config.wal_path_for_session("claude-abc").exists()


def test_run_prompt_writes_hook_json(tmp_path, monkeypatch):
    monkeypatch.setenv("AWHM_DATA_DIR", str(tmp_path / "awhm"))
    monkeypatch.setenv("AWHM_MOCK_EMBEDDINGS", "1")
    run(["prompt"], stdin=io.StringIO(json.dumps(_payload(user_prompt="I prefer tabs"))), stdout=io.StringIO())
    out = io.StringIO()
    code = run(["prompt"], stdin=io.StringIO(json.dumps(_payload(user_prompt="tabs or spaces?"))), stdout=out)
    assert code == 0
    data = json.loads(out.getvalue())
    assert "tabs" in data["hookSpecificOutput"]["additionalContext"]


def test_run_never_fails_on_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv("AWHM_DATA_DIR", str(tmp_path / "awhm"))
    assert run(["stop"], stdin=io.StringIO("not json"), stdout=io.StringIO()) == 0


def test_settings_snippet_shape():
    snippet = settings_snippet("/usr/local/bin/awhm")
    hooks = snippet["hooks"]
    assert set(hooks) == {"UserPromptSubmit", "Stop", "SessionEnd"}
    assert hooks["UserPromptSubmit"][0]["hooks"][0]["command"] == "/usr/local/bin/awhm hook prompt"
    out = io.StringIO()
    assert run(["settings", "--command", "awhm"], stdout=out) == 0
    assert json.loads(out.getvalue())["hooks"]["Stop"][0]["hooks"][0]["command"] == "awhm hook stop"
