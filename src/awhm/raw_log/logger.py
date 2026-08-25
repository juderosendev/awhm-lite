"""RawLogger: append-only JSONL writer, one file per session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import AWHMConfig
from ..types import Role
from .models import LogEntry


class RawLogger:
    """Append-only JSONL logger. One file per session."""

    def __init__(self, config: AWHMConfig, session_id: str) -> None:
        self.config = config
        self.session_id = session_id
        self._path = config.logs_dir / f"{session_id}.jsonl"
        self._msg_index = 0
        config.ensure_dirs()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def msg_index(self) -> int:
        return self._msg_index

    def log(
        self,
        role: Role | str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Append a single log entry. Returns the entry."""
        role_val = role.value if isinstance(role, Role) else role
        entry = LogEntry(
            timestamp=LogEntry.now_iso(),
            session_id=self.session_id,
            role=role_val,
            content=content,
            metadata=metadata or {},
        )
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        self._msg_index += 1
        return entry
