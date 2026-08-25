"""RawLogReader: read sessions, list sessions, hard-delete entries."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import AWHMConfig
from .models import LogEntry


class RawLogReader:
    """Read and search raw log files."""

    def __init__(self, config: AWHMConfig) -> None:
        self.config = config

    def list_sessions(self) -> list[str]:
        """Return all session IDs (sorted by file modification time)."""
        logs_dir = self.config.logs_dir
        if not logs_dir.exists():
            return []
        files = sorted(logs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        return [f.stem for f in files]

    def session_count(self) -> int:
        return len(self.list_sessions())

    def read_session(self, session_id: str) -> list[LogEntry]:
        """Read all entries from a session log file."""
        path = self.config.logs_dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        entries: list[LogEntry] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(LogEntry.from_dict(json.loads(line)))
        return entries

    def read_all_sessions(self) -> dict[str, list[LogEntry]]:
        """Read all sessions. Returns {session_id: [entries]}."""
        result: dict[str, list[LogEntry]] = {}
        for sid in self.list_sessions():
            result[sid] = self.read_session(sid)
        return result

    def hard_delete_entries(self, session_id: str, content_match: str) -> int:
        """Delete entries matching content from a session log. Returns count deleted.

        Rewrites the file without matching entries. This is the only
        mutation allowed on raw logs (privacy compliance).
        """
        return self.hard_delete_entries_exact(session_id, content_match)

    def hard_delete_entries_exact(self, session_id: str, content_match: str) -> int:
        """Delete entries whose normalized content exactly matches content_match."""
        path = self.config.logs_dir / f"{session_id}.jsonl"
        if not path.exists():
            return 0
        entries = self.read_session(session_id)
        original_count = len(entries)
        target_norm = self._normalize_text(content_match)
        kept = [
            e for e in entries
            if self._normalize_text(e.content) != target_norm
        ]
        deleted_count = original_count - len(kept)
        if deleted_count > 0:
            self._rewrite_session(path, kept)
        return deleted_count

    def hard_delete_entries_by_indices(
        self,
        session_id: str,
        message_indices: set[int],
    ) -> int:
        """Delete entries by 0-based message index within a session log."""
        if not message_indices:
            return 0
        path = self.config.logs_dir / f"{session_id}.jsonl"
        if not path.exists():
            return 0
        entries = self.read_session(session_id)
        original_count = len(entries)
        kept = [
            entry for i, entry in enumerate(entries)
            if i not in message_indices
        ]
        deleted_count = original_count - len(kept)
        if deleted_count > 0:
            self._rewrite_session(path, kept)
        return deleted_count

    def delete_session(self, session_id: str) -> bool:
        """Delete an entire session log file."""
        path = self.config.logs_dir / f"{session_id}.jsonl"
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    def _rewrite_session(self, path: Path, entries: list[LogEntry]) -> None:
        """Atomically rewrite a session log with provided entries."""
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
        tmp_path.replace(path)
