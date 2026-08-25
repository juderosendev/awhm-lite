"""SessionBuffer: in-memory store with pattern matching on each message."""

from __future__ import annotations

from ..raw_log.models import LogEntry
from ..types import BufferEntryType
from .models import BufferEntry
from .patterns import match_message


class SessionBuffer:
    """In-memory buffer for intra-session continuity."""

    def __init__(self) -> None:
        self._entries: list[BufferEntry] = []

    @property
    def entries(self) -> list[BufferEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def process_message(self, content: str, timestamp: str, msg_index: int) -> list[BufferEntry]:
        """Run pattern matching on a message and store any matches.

        Returns the new entries added.
        """
        matches = match_message(content)
        new_entries: list[BufferEntry] = []
        for m in matches:
            entry = BufferEntry(
                type=m.type.value,
                content=m.content,
                timestamp=timestamp,
                source_msg=msg_index,
            )
            self._entries.append(entry)
            new_entries.append(entry)
        return new_entries

    def search(self, query: str) -> list[BufferEntry]:
        """Simple substring search over buffer entries."""
        query_lower = query.lower()
        return [e for e in self._entries if query_lower in e.content.lower()]

    def get_by_type(self, entry_type: BufferEntryType | str) -> list[BufferEntry]:
        """Get all entries of a specific type."""
        type_val = entry_type.value if isinstance(entry_type, BufferEntryType) else entry_type
        return [e for e in self._entries if e.type == type_val]

    def add_entry(self, entry: BufferEntry) -> None:
        """Add a pre-built entry (used during WAL recovery)."""
        self._entries.append(entry)

    def to_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]
