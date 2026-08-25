"""Rule-based extraction: reuses buffer patterns on full session log."""

from __future__ import annotations

from dataclasses import dataclass

from ..raw_log.models import LogEntry
from ..session_buffer.patterns import match_message


@dataclass
class Extraction:
    type: str  # BufferEntryType value
    content: str
    source_message: str
    message_index: int


def extract_from_log_entries(
    entries: list[LogEntry],
    message_offset: int = 0,
) -> list[Extraction]:
    """Run pattern matching over all log entries.

    Same patterns as session buffer, applied for completeness
    (catches anything the real-time buffer might have missed).
    """
    extractions: list[Extraction] = []
    seen: set[str] = set()

    for i, entry in enumerate(entries):
        matches = match_message(entry.content)
        for m in matches:
            key = f"{m.type.value}:{m.content[:100]}"
            if key not in seen:
                seen.add(key)
                extractions.append(Extraction(
                    type=m.type.value,
                    content=m.content,
                    source_message=entry.content,
                    message_index=message_offset + i,
                ))
    return extractions
