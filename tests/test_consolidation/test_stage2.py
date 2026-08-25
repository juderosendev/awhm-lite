"""Tests for Stage 2 LLM refinement (with a mock client)."""

import json
import subprocess
import sys

import pytest

from awhm import AWHMSession
from awhm.config import AWHMConfig
from awhm.consolidation.stage2 import (
    MEMORY_SCHEMA,
    AnthropicClient,
    MockLLMClient,
    Stage2Refiner,
    parse_memories,
)
from awhm.raw_log.models import LogEntry
from awhm.types import Role

MEMORY_SCHEMA_STUB = MEMORY_SCHEMA


def _entries(*texts):
    return [
        LogEntry(timestamp="2026-03-01T00:00:00+00:00", session_id="s1", role="user", content=t)
        for t in texts
    ]


def _memory(**kw):
    base = {
        "content": "The user's preferred editor is Neovim",
        "type": "semantic",
        "slot": "user preferred editor",
        "valid_from": None,
        "valid_to": None,
        "confidence": 0.9,
        "sources": [0],
        "supersedes_previous": False,
    }
    base.update(kw)
    return base


def test_parse_memories_tolerates_prose_and_garbage():
    payload = json.dumps({"memories": [_memory()]})
    assert len(parse_memories(payload)) == 1
    assert len(parse_memories("Here you go:\n" + payload + "\nDone.")) == 1
    assert parse_memories("not json at all") == []
    assert parse_memories(json.dumps({"memories": "nope"})) == []


def test_refiner_validates_proposals(config):
    config.stage2_enabled = True
    proposals = {
        "memories": [
            _memory(),
            _memory(content="Invented fact", sources=[99]),          # out-of-range provenance
            _memory(content="Low confidence guess", confidence=0.2),  # below threshold
            _memory(content="I prefer Rust", slot=None),              # already captured by stage 1
            _memory(content="", sources=[1]),                          # empty
            _memory(content="Valid window", slot=None, sources=[1],
                    valid_from="2026-03-01", valid_to="not a date"),
        ]
    }
    client = MockLLMClient([json.dumps(proposals)])
    refiner = Stage2Refiner(config, client)
    entries = _entries("Let's go with Neovim then, it's what I use", "I prefer Rust")
    cands = refiner.refine("s1", entries, 0, known_statements={"i prefer rust"})

    contents = [c.content for c in cands]
    assert contents == ["The user's preferred editor is Neovim", "Valid window"]
    first = cands[0]
    assert first.source == "stage2"
    assert first.key_override == "fact:user preferred editor"
    assert first.refs == [{"session_id": "s1", "message_index": 0}]
    assert cands[1].valid_from.startswith("2026-03-01") and cands[1].valid_to is None
    system, user = client.requests[0]
    assert "Already captured" in user and "[1] user: I prefer Rust" in user


def test_refiner_survives_client_errors_and_chunks(config):
    config.stage2_enabled = True
    config.stage2_max_messages = 2

    def respond(system, user):
        if "[0]" in user:
            raise RuntimeError("boom")
        return json.dumps({"memories": [_memory(content="From chunk two", slot=None, sources=[2])]})

    refiner = Stage2Refiner(config, MockLLMClient(respond))
    cands = refiner.refine("s1", _entries("a", "b", "c"), 0, set())
    assert [c.content for c in cands] == ["From chunk two"]


def test_pipeline_commits_stage2_memories_with_slot_supersession(tmp_path):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"), cold_start_session_count=0, stage2_enabled=True)
    responses = [
        json.dumps({"memories": [_memory(content="The user's preferred editor is Neovim", sources=[0])]}),
        json.dumps({"memories": [_memory(content="The user's preferred editor is Helix", sources=[1],
                                          supersedes_previous=True)]}),
    ]
    client = MockLLMClient(responses)
    with AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True, llm_client=client) as s:
        s.log_message(Role.USER, "Let's go with Neovim then, it's what I use")
        assert s.consolidate_current() >= 1
        s.log_message(Role.USER, "Scratch that, switched to Helix last week")
        assert s.consolidate_current() >= 1

        by_content = {n.content: n for n in s.graph.all_nodes() if n.source_refs and n.canonical_key}
        assert by_content["The user's preferred editor is Neovim"].status == "superseded"
        assert by_content["The user's preferred editor is Helix"].status == "active"
        assert by_content["The user's preferred editor is Helix"].canonical_key == "fact:user preferred editor"

        current = " ".join(r.content for r in s.query("preferred editor"))
        assert "Helix" in current and "Neovim" not in current


def test_stage2_disabled_never_calls_client(tmp_path):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"), cold_start_session_count=0)
    client = MockLLMClient([json.dumps({"memories": [_memory()]})])
    with AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True, llm_client=client) as s:
        s.log_message(Role.USER, "I prefer Rust")
        s.consolidate_current()
    assert client.requests == []


def test_anthropic_client_requires_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(RuntimeError, match=r"\[anthropic\]"):
        AnthropicClient()


def test_anthropic_client_request_shape():
    class FakeResponse:
        stop_reason = "end_turn"

        class _Block:
            type = "text"
            text = json.dumps({"memories": []})

        content = [_Block()]

    class FakeBeta:
        def __init__(self, log):
            self.messages = self
            self.log = log

        def create(self, **kwargs):
            self.log.append(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.beta = FakeBeta(self.calls)
            self.messages = FakeBeta(self.calls)

    fake = FakeClient()
    client = AnthropicClient(model="claude-opus-5", client=fake)
    out = client.complete_json("sys", "user text", {"type": "object"})
    assert json.loads(out) == {"memories": []}
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["messages"] == [{"role": "user", "content": "user text"}]

    fake2 = FakeClient()
    AnthropicClient(client=fake2, fallbacks=False).complete_json("s", "u", {"type": "object"})
    assert "fallbacks" not in fake2.calls[0] and "betas" not in fake2.calls[0]


def _fake_runner(envelope, returncode=0, stderr=""):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, returncode, stdout=json.dumps(envelope), stderr=stderr)

    run.calls = calls
    return run


def test_claude_code_client_uses_structured_output(monkeypatch):
    from awhm.consolidation.stage2 import HOOK_GUARD_ENV, ClaudeCodeClient

    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    runner = _fake_runner({"is_error": False, "structured_output": {"memories": [_memory()]}, "result": "ignored"})
    client = ClaudeCodeClient(model="sonnet", runner=runner)
    out = json.loads(client.complete_json("sys", "user text", MEMORY_SCHEMA_STUB))
    assert out["memories"][0]["content"].startswith("The user's preferred editor")

    cmd, kwargs = runner.calls[0]
    assert cmd[:3] == ["claude", "-p", "user text"]
    assert "--json-schema" in cmd and "--tools" in cmd and cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert "--no-session-persistence" in cmd and "--max-turns" in cmd
    env = kwargs["env"]
    assert env[HOOK_GUARD_ENV] == "1"
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_ENTRYPOINT" not in env
    assert kwargs["stdin"] is subprocess.DEVNULL


def test_claude_code_client_falls_back_to_result_text():
    from awhm.consolidation.stage2 import ClaudeCodeClient

    runner = _fake_runner({"is_error": False, "structured_output": None, "result": '{"memories": []}'})
    assert json.loads(ClaudeCodeClient(runner=runner).complete_json("s", "u", {})) == {"memories": []}


def test_claude_code_client_surfaces_errors():
    from awhm.consolidation.stage2 import ClaudeCodeClient

    with pytest.raises(RuntimeError, match="Not logged in"):
        ClaudeCodeClient(runner=_fake_runner({"is_error": True, "result": "Not logged in · Please run /login"})).complete_json("s", "u", {})
    with pytest.raises(RuntimeError, match="exited 1"):
        ClaudeCodeClient(runner=_fake_runner({}, returncode=1, stderr="boom")).complete_json("s", "u", {})


def test_make_client_defaults_to_claude_code(monkeypatch):
    from awhm.consolidation import stage2 as s2

    monkeypatch.setattr(s2.shutil, "which", lambda name: "/usr/local/bin/claude")
    assert isinstance(s2.make_client(), s2.ClaudeCodeClient)
    assert isinstance(s2.make_client("claude-code", "sonnet"), s2.ClaudeCodeClient)
    with pytest.raises(ValueError):
        s2.make_client("openai")
    monkeypatch.setattr(s2.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        s2.make_client()


def test_session_builds_default_client_when_stage2_enabled(tmp_path, monkeypatch):
    from awhm.consolidation import stage2 as s2

    monkeypatch.setattr(s2.shutil, "which", lambda name: "/usr/local/bin/claude")
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"), stage2_enabled=True, stage2_model="sonnet")
    with AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True) as s:
        assert isinstance(s.llm_client, s2.ClaudeCodeClient)
        assert s.llm_client.model == "sonnet"
