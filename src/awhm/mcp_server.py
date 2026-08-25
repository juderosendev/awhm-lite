"""AWHM Lite MCP server: exposes the memory system as tools for Claude Code."""

from __future__ import annotations

import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from awhm import AWHMSession
from awhm.config import AWHMConfig
from awhm.types import Role

# Logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("awhm-mcp")

mcp = FastMCP("awhm-lite")

# ── Global session state ──────────────────────────────────────

_session: AWHMSession | None = None


def _get_session() -> AWHMSession:
    """Get or create the global AWHM session."""
    global _session
    if _session is None:
        data_dir = os.environ.get("AWHM_DATA_DIR", "~/.awhm")
        use_mock = os.environ.get("AWHM_MOCK_EMBEDDINGS", "").lower() in ("1", "true")
        config = AWHMConfig(data_dir=data_dir)
        _session = AWHMSession.start_session(
            config, use_mock_embeddings=use_mock,
        )
        logger.info(f"AWHM session started: {_session.session_id}")
    return _session


# ── Tools ─────────────────────────────────────────────────────


@mcp.tool()
async def memory_query(
    query: str,
    k: int = 10,
    include_history: bool = False,
    with_trace: bool = False,
    as_of: str | None = None,
) -> str:
    """Search long-term memory for information relevant to the query.

    Call this at the start of a conversation or when the user asks about
    something that might have come up in past sessions (preferences,
    facts, corrections, project details, names, etc.).

    Args:
        query: Natural language search query.
        k: Maximum number of results to return (default 10).
        include_history: Include superseded/retracted memories.
        with_trace: Include ranking feature trace details.
        as_of: ISO-8601 timestamp; answer as of that moment (time travel).
    """
    session = _get_session()
    results = session.query(
        query,
        k=k,
        include_history=include_history,
        with_trace=with_trace,
        as_of=as_of,
    )
    if not results:
        return "No memories found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r.source}] (score: {r.score:.3f}) {r.content}")
        if with_trace and r.trace:
            lines.append(f"   trace={json.dumps(r.trace, sort_keys=True)}")
    return "\n".join(lines)


@mcp.tool()
async def memory_log(role: str, content: str) -> str:
    """Log a message to the raw conversation log.

    Call this to record important user messages or assistant responses
    so they can be consolidated into long-term memory later.

    Args:
        role: One of "user", "assistant", "tool_call", "tool_result".
        content: The message content to log.
    """
    try:
        role_enum = Role(role.lower().strip())
    except ValueError:
        valid = ", ".join(r.value for r in Role)
        return f"Invalid role {role!r}. Expected one of: {valid}."
    session = _get_session()
    session.log_message(role_enum, content)
    return f"Logged. Buffer now has {len(session.buffer)} entries."


@mcp.tool()
async def memory_consolidate() -> str:
    """Run Stage 1 consolidation on all pending sessions.

    This extracts entities, facts, preferences, and corrections from
    raw logs and adds them to the memory graph. Run this at the end
    of a conversation or periodically.
    """
    session = _get_session()
    results = session.consolidate()
    total = sum(results.values())
    if not results:
        return "No pending sessions to consolidate."
    lines = [f"Consolidated {len(results)} session(s), {total} new node(s):"]
    for sid, count in results.items():
        lines.append(f"  {sid}: {count} nodes")
    return "\n".join(lines)


@mcp.tool()
async def memory_status() -> str:
    """Show the current state of the memory system.

    Returns node count, edge count, buffer entries, session count, etc.
    """
    session = _get_session()
    info = session.status()
    return json.dumps(info, indent=2)


@mcp.tool()
async def memory_snapshot_create() -> str:
    """Create a snapshot of the current memory graph.

    Use this before making destructive changes or as a periodic backup.
    """
    session = _get_session()
    path = session.create_snapshot()
    return f"Snapshot created: {path}"


@mcp.tool()
async def memory_delete_node(node_id: str) -> str:
    """Hard-delete a memory node for privacy compliance.

    Removes the node, its edges, related raw log entries, and buffer entries.

    Args:
        node_id: The UUID of the node to delete.
    """
    session = _get_session()
    result = session.delete_node(node_id)
    if result.node_deleted:
        affected = (
            f" Affected sessions: {', '.join(result.affected_sessions)}."
            if result.affected_sessions
            else ""
        )
        return (
            f"Deleted node {node_id}. "
            f"Edges removed: {result.edges_removed}, "
            f"Log entries removed: {result.log_entries_removed}, "
            f"Buffer entries removed: {result.buffer_entries_removed}, "
            f"match strategy: {result.match_strategy}, "
            f"snapshots touched: {result.snapshots_touched}."
            f"{affected}"
        )
    return f"Node {node_id} not found."


# ── Entrypoint ────────────────────────────────────────────────


def main():
    logger.info("Starting AWHM Lite MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
