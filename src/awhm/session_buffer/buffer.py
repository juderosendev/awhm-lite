"""SessionBuffer: in-memory store with pattern matching on each message."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..types import BufferEntryType
from .models import BufferEntry
from .patterns import match_message


class SessionBuffer:
    """In-memory buffer for intra-session continuity.

    ``version`` increments on every mutation so the WAL can skip flushes when
    nothing changed.
    """

    def __init__(self) -> None:
        self._entries: list[BufferEntry] = []
        self.version = 0

    @property
    def entries(self) -> list[BufferEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        if self._entries:
            self._entries.clear()
            self.version += 1

    def process_message(self, content: str, timestamp: str, msg_index: int) -> list[BufferEntry]:
        """Run pattern matching on a message and store any matches.

        Returns the new entries added.
        """
        new_entries = [
            BufferEntry(
                type=m.type.value,
                content=m.content,
                timestamp=timestamp,
                source_msg=msg_index,
            )
            for m in match_message(content)
        ]
        if new_entries:
            self._entries.extend(new_entries)
            self.version += 1
        return new_entries

    def search(self, query: str) -> list[BufferEntry]:
        """Case-insensitive substring search over buffer entries."""
        query_lower = query.lower()
        return [e for e in self._entries if query_lower in e.content.lower()]

    def get_by_type(self, entry_type: BufferEntryType | str) -> list[BufferEntry]:
        """Get all entries of a specific type."""
        type_val = entry_type.value if isinstance(entry_type, BufferEntryType) else entry_type
        return [e for e in self._entries if e.type == type_val]

    def add_entry(self, entry: BufferEntry) -> None:
        """Add a pre-built entry (used during WAL and snapshot recovery)."""
        self._entries.append(entry)
        self.version += 1

    def remove_where(self, predicate: Callable[[BufferEntry], bool]) -> int:
        """Remove entries matching ``predicate``. Returns the number removed."""
        kept = [e for e in self._entries if not predicate(e)]
        removed = len(self._entries) - len(kept)
        if removed:
            self._entries = kept
            self.version += 1
        return removed

    def to_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]
