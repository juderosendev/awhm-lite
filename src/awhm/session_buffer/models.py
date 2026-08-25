"""Session buffer data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BufferEntry:
    type: str  # BufferEntryType enum value
    content: str
    timestamp: str
    source_msg: int  # Index into raw log

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
            "source_msg": self.source_msg,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BufferEntry:
        return cls(
            type=d["type"],
            content=d["content"],
            timestamp=d["timestamp"],
            source_msg=d["source_msg"],
        )
