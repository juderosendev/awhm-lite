"""Claude Code hooks: memory that works without the model calling a tool.

Three commands, each run as a separate process by Claude Code:

* ``awhm hook prompt``      (UserPromptSubmit) logs the prompt, retrieves the
  most relevant memories and returns them as hidden context.
* ``awhm hook stop``        (Stop) logs the assistant's reply.
* ``awhm hook session-end`` (SessionEnd) consolidates the session into the
  graph.

``awhm hook settings`` prints the ``settings.json`` snippet that wires them
up. Hooks never block the user: every failure is logged to stderr and the
process exits 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from typing import IO, Any

from .config import AWHMConfig
from .types import Role

logger = logging.getLogger("awhm.hooks")

SESSION_PREFIX = "claude-"
DEFAULT_K = 5
CONTEXT_HEADER = "Relevant memories from previous sessions (from AWHM Lite):"


def _config_from_env() -> AWHMConfig:
    return AWHMConfig(data_dir=os.environ.get("AWHM_DATA_DIR", "~/.awhm"))


def _use_mock() -> bool:
    return os.environ.get("AWHM_MOCK_EMBEDDINGS", "").lower() in ("1", "true")


def _session_id(payload: dict[str, Any]) -> str:
    raw = str(payload.get("session_id") or "unknown")
    return raw if raw.startswith(SESSION_PREFIX) else SESSION_PREFIX + raw


def _read_payload(stdin: IO[str]) -> dict[str, Any]:
    raw = stdin.read()
    if not raw.strip():
        return {}
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def format_context(results: list[Any]) -> str:
    """Render retrieval results as the hidden context block."""
    lines = [CONTEXT_HEADER]
    lines.extend(f"- {r.content}" for r in results)
    return "\n".join(lines)


# ── Commands ───────────────────────────────────────────────────


def cmd_prompt(
    payload: dict[str, Any],
    config: AWHMConfig,
    *,
    k: int = DEFAULT_K,
    semantic: bool = False,
    use_mock: bool = False,
) -> dict[str, Any] | None:
    """Log the user prompt, retrieve memories, return the hook output (or None)."""
    from . import AWHMSession

    prompt = str(payload.get("user_prompt") or payload.get("prompt") or "").strip()
    if not prompt:
        return None
    session = AWHMSession.start_session(
        config, session_id=_session_id(payload), use_mock_embeddings=use_mock,
    )
    try:
        session.log_message(Role.USER, prompt)
        results = session.query(prompt, k=k, semantic=semantic)
    finally:
        session.suspend()
    if not results:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": format_context(results),
        }
    }


def cmd_stop(payload: dict[str, Any], config: AWHMConfig, *, use_mock: bool = False) -> None:
    """Log the assistant's final message for the turn."""
    from . import AWHMSession

    text = str(payload.get("last_assistant_message") or "").strip()
    if not text:
        return
    session = AWHMSession.start_session(
        config, session_id=_session_id(payload), use_mock_embeddings=use_mock,
    )
    try:
        entries = session.log_reader.read_session(session.session_id)
        if entries and entries[-1].role == Role.ASSISTANT.value and entries[-1].content == text:
            return  # hook fired twice for the same turn
        session.log_message(Role.ASSISTANT, text)
    finally:
        session.suspend()


def cmd_session_end(payload: dict[str, Any], config: AWHMConfig, *, use_mock: bool = False) -> int:
    """Consolidate the session into the graph. Returns new node count."""
    from . import AWHMSession

    session = AWHMSession.start_session(
        config, session_id=_session_id(payload), use_mock_embeddings=use_mock,
    )
    try:
        return session.consolidate_current()
    finally:
        session.end_session()


def settings_snippet(command: str | None = None) -> dict[str, Any]:
    """The ``hooks`` block to merge into ``~/.claude/settings.json``."""
    exe = command or shutil.which("awhm") or "awhm"
    return {
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": f"{exe} hook prompt", "timeout": 20}]}
            ],
            "Stop": [
                {"hooks": [{"type": "command", "command": f"{exe} hook stop", "timeout": 20}]}
            ],
            "SessionEnd": [
                {"hooks": [{"type": "command", "command": f"{exe} hook session-end", "timeout": 60}]}
            ],
        }
    }


# ── Entrypoint ─────────────────────────────────────────────────


def run(argv: list[str], stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> int:
    parser = argparse.ArgumentParser(prog="awhm hook", description="Claude Code hook commands")
    sub = parser.add_subparsers(dest="event", required=True)
    p_prompt = sub.add_parser("prompt", help="UserPromptSubmit: log prompt, inject memories")
    p_prompt.add_argument("-k", type=int, default=DEFAULT_K)
    p_prompt.add_argument("--semantic", action="store_true", help="Also use embeddings (slower start)")
    sub.add_parser("stop", help="Stop: log the assistant reply")
    sub.add_parser("session-end", help="SessionEnd: consolidate the session")
    p_settings = sub.add_parser("settings", help="Print the settings.json hooks block")
    p_settings.add_argument("--command", default=None, help="Executable path to use in the snippet")
    args = parser.parse_args(argv)

    if args.event == "settings":
        stdout.write(json.dumps(settings_snippet(args.command), indent=2) + "\n")
        return 0

    config = _config_from_env()
    use_mock = _use_mock()
    try:
        payload = _read_payload(stdin)
        if args.event == "prompt":
            output = cmd_prompt(payload, config, k=args.k, semantic=args.semantic, use_mock=use_mock)
            if output is not None:
                stdout.write(json.dumps(output) + "\n")
        elif args.event == "stop":
            cmd_stop(payload, config, use_mock=use_mock)
        elif args.event == "session-end":
            count = cmd_session_end(payload, config, use_mock=use_mock)
            sys.stderr.write(f"awhm: consolidated {count} new memor{'y' if count == 1 else 'ies'}\n")
    except Exception as exc:  # never block the user's session
        logger.exception("hook failed")
        sys.stderr.write(f"awhm hook error: {exc}\n")
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))
