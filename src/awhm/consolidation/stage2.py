"""Stage 2: LLM refinement of what the rules missed.

Stage 1 is symbolic and cheap, and it has a hard ceiling: it catches "I
prefer Rust" and misses "let's go with Rust then". Stage 2 runs offline,
after Stage 1, and asks an LLM to read the new messages and propose the
memories the rules did not find. The LLM only *proposes*: deterministic code
validates every proposal (schema, provenance, confidence), drops anything
already captured, and commits through the same slot-key and supersession
rules as everything else. Retrieval stays zero-LLM.

No API key is required. The default client shells out to the Claude Code
CLI (``claude -p``), which uses whatever login the CLI already has. An
Anthropic SDK client exists as an explicitly optional alternative.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from typing import Any, Protocol

from ..config import AWHMConfig
from ..raw_log.models import LogEntry
from ..timeutil import parse_timestamp
from ..types import NodeType
from .canonical import normalize_text
from .deduplication import statement_key

logger = logging.getLogger("awhm.stage2")

MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string", "enum": ["semantic", "procedural", "episodic"]},
                    "slot": {"type": ["string", "null"]},
                    "valid_from": {"type": ["string", "null"]},
                    "valid_to": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "sources": {"type": "array", "items": {"type": "integer"}},
                    "supersedes_previous": {"type": "boolean"},
                },
                "required": [
                    "content", "type", "slot", "valid_from", "valid_to",
                    "confidence", "sources", "supersedes_previous",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You extract long-term memories from a transcript between a user and an AI assistant.

Return only memories worth keeping across future sessions: facts about the user, their preferences and standing rules, project details, decisions, corrections, and outcomes of things that were tried.

Rules:
- Each memory is one self-contained statement, written in the third person ("The user prefers ...", "The project's API endpoint is ..."). Preserve qualifiers: time, negation, uncertainty, scope.
- Never invent. Every memory cites the numbers of the messages it comes from in "sources".
- "type": semantic for facts, procedural for preferences and rules, episodic for events and outcomes.
- "slot": for facts with a single current value, name the slot in a few lowercase words (e.g. "user preferred language", "api endpoint", "user home city"). Two memories with the same slot mean the value changed. Otherwise null.
- "supersedes_previous": true only when the message explicitly corrects earlier information ("actually", "no, it's", "changed to").
- "valid_from" / "valid_to": ISO 8601 when the transcript states when something became or stops being true, else null.
- "confidence": 0 to 1, how certain you are the memory is correct and durable.
- Skip greetings, transient task chatter, and anything listed under "Already captured".
"""


class LLMClient(Protocol):
    """Anything that can turn (system, user, schema) into a JSON document."""

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str: ...


class AnthropicClient:
    """LLMClient backed by the Anthropic SDK (optional extra ``[anthropic]``).

    Only for people who prefer API billing; the default ``ClaudeCodeClient``
    needs no key.

    Uses structured outputs so the response is schema-valid JSON, and the
    server-side refusal fallback so a policy decline on the primary model
    re-runs on a fallback model inside the same call.
    """

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        max_tokens: int = 16000,
        fallbacks: bool = True,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "The Anthropic SDK client needs: pip install 'awhm-lite[anthropic]' "
                    "(or use the default ClaudeCodeClient, which needs no API key)"
                ) from exc
            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if self.fallbacks:
            response = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request,
            )
        else:
            response = self._client.messages.create(**request)
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("Stage 2 request was refused by the model")
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise RuntimeError("Stage 2 response contained no text block")


HOOK_GUARD_ENV = "AWHM_HOOK_ACTIVE"
_NESTED_SESSION_VARS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class ClaudeCodeClient:
    """LLMClient that runs ``claude -p`` (the Claude Code CLI). No API key.

    Uses the CLI's structured output (``--json-schema``) so the reply is
    schema-valid JSON, disables tools, and marks the subprocess with
    ``AWHM_HOOK_ACTIVE`` so the memory hooks do not fire recursively when
    Stage 2 itself runs from a hook.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        executable: str = "claude",
        timeout: float = 180.0,
        runner: Runner | None = None,
    ) -> None:
        self.model = model
        self.executable = executable
        self.timeout = timeout
        self._runner = runner or subprocess.run
        if runner is None and shutil.which(executable) is None:
            raise RuntimeError(
                f"Claude Code CLI '{executable}' not found on PATH; install Claude Code "
                "or pass a different LLM client."
            )

    def command(self, system: str, user: str, schema: dict[str, Any]) -> list[str]:
        cmd = [
            self.executable, "-p", user,
            "--system-prompt", system,
            "--output-format", "json",
            "--json-schema", json.dumps(schema),
            "--tools", "",
            "--no-session-persistence",
            "--max-turns", "1",
        ]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def environment(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _NESTED_SESSION_VARS}
        env[HOOK_GUARD_ENV] = "1"
        return env

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        proc = self._runner(
            self.command(system, user, schema),
            capture_output=True,
            text=True,
            env=self.environment(),
            stdin=subprocess.DEVNULL,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude -p returned non-JSON output: {proc.stdout[:200]!r}") from exc
        if envelope.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(envelope.get('result'))[:300]}")
        structured = envelope.get("structured_output")
        if structured is not None:
            return json.dumps(structured)
        result = envelope.get("result")
        if isinstance(result, str) and result.strip():
            return result
        raise RuntimeError("claude -p returned no result")


def make_client(kind: str = "claude-code", model: str | None = None) -> LLMClient:
    """Build the Stage 2 client named by ``config.stage2_client``."""
    kind = (kind or "claude-code").lower()
    if kind in ("claude-code", "claude", "cli"):
        return ClaudeCodeClient(model=model)
    if kind == "anthropic":
        return AnthropicClient(model=model or "claude-opus-5")
    raise ValueError(f"Unknown stage2_client {kind!r}; use 'claude-code' or 'anthropic'")


class MockLLMClient:
    """Deterministic client for tests: returns canned JSON per call."""

    def __init__(self, responses: list[str] | Callable[[str, str], str]) -> None:
        self._responses = responses
        self._calls = 0
        self.requests: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        self.requests.append((system, user))
        if callable(self._responses):
            return self._responses(system, user)
        if self._calls >= len(self._responses):
            return json.dumps({"memories": []})
        response = self._responses[self._calls]
        self._calls += 1
        return response


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_memories(text: str) -> list[dict[str, Any]]:
    """Parse the model's JSON; tolerate prose around the object. Empty list on failure."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    memories = data.get("memories") if isinstance(data, dict) else None
    return [m for m in memories if isinstance(m, dict)] if isinstance(memories, list) else []


def _iso_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return parse_timestamp(value).isoformat()
    except ValueError:
        return None


class Stage2Refiner:
    """Turn LLM proposals into validated candidates for the commit step."""

    def __init__(self, config: AWHMConfig, client: LLMClient) -> None:
        self.config = config
        self.client = client

    def refine(
        self,
        session_id: str,
        entries: list[LogEntry],
        start_idx: int,
        known_statements: set[str],
    ) -> list[Any]:
        from .pipeline import Candidate  # local import: pipeline imports this module

        candidates: list[Candidate] = []
        seen = set(known_statements)
        size = max(int(self.config.stage2_max_messages), 1)
        for offset in range(0, len(entries), size):
            chunk = entries[offset:offset + size]
            first = start_idx + offset
            last = first + len(chunk) - 1
            user = self._render(chunk, first, seen)
            try:
                raw = self.client.complete_json(SYSTEM_PROMPT, user, MEMORY_SCHEMA)
            except Exception as exc:
                logger.warning("stage 2 call failed for %s: %s", session_id, exc)
                continue
            for item in parse_memories(raw):
                cand = self._to_candidate(item, session_id, first, last)
                if cand is None:
                    continue
                key = statement_key(cand.content)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(cand)
        return candidates

    @staticmethod
    def _render(chunk: list[LogEntry], first: int, known: set[str]) -> str:
        lines: list[str] = []
        if known:
            lines.append("Already captured (do not repeat):")
            lines.extend(f"- {k}" for k in sorted(known)[:100])
            lines.append("")
        lines.append("Transcript:")
        for i, entry in enumerate(chunk):
            lines.append(f"[{first + i}] {entry.role}: {entry.content}")
        return "\n".join(lines)

    def _to_candidate(self, item: dict[str, Any], session_id: str, first: int, last: int):
        from .pipeline import Candidate

        content = str(item.get("content") or "").strip()
        if not content:
            return None
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if confidence < self.config.stage2_min_confidence:
            return None
        sources = [
            int(s) for s in item.get("sources") or []
            if isinstance(s, int | float) and first <= int(s) <= last
        ]
        if not sources:
            return None  # no verifiable provenance, no memory
        node_type = str(item.get("type") or "semantic")
        if node_type not in {t.value for t in NodeType}:
            node_type = NodeType.SEMANTIC.value
        slot = item.get("slot")
        key_override = f"fact:{normalize_text(str(slot))}" if isinstance(slot, str) and slot.strip() else None
        return Candidate(
            content=content,
            node_type=node_type,
            source="stage2",
            confidence=min(confidence, 1.0),
            message_index=min(sources),
            session_id=session_id,
            is_correction=bool(item.get("supersedes_previous")),
            refs=[{"session_id": session_id, "message_index": s} for s in sorted(set(sources))],
            valid_from=_iso_or_none(item.get("valid_from")),
            valid_to=_iso_or_none(item.get("valid_to")),
            key_override=key_override,
        )
