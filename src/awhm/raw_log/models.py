"""Raw log data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LogEntry:
    timestamp: str
    session_id: str
    role: str  # Role enum value
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogEntry:
        return cls(
            timestamp=d["timestamp"],
            session_id=d["session_id"],
            role=d["role"],
            content=d["content"],
            metadata=d.get("metadata", {}),
        )
