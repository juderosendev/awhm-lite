"""Date/time resolution via dateparser for Stage 1 consolidation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ResolvedDate:
    original: str
    resolved: datetime
    iso: str
    message_index: int | None = None


# Common date-like patterns to extract candidates
DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b(?:yesterday|today|tomorrow|last (?:week|month|year)|next (?:week|month|year))\b", re.IGNORECASE),
    re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE),
    re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,?\s*\d{4})?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?\b", re.IGNORECASE),
]


class TemporalParser:
    """Extract and resolve date/time references."""

    def extract_dates(
        self,
        text: str,
        message_index: int | None = None,
    ) -> list[ResolvedDate]:
        """Find and resolve date references in text."""
        import dateparser

        candidates: list[str] = []
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(text):
                candidates.append(match.group(0))

        results: list[ResolvedDate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.lower() in seen:
                continue
            seen.add(candidate.lower())
            parsed = dateparser.parse(candidate)
            if parsed:
                results.append(ResolvedDate(
                    original=candidate,
                    resolved=parsed,
                    iso=parsed.isoformat(),
                    message_index=message_index,
                ))
        return results

    def extract_from_messages(
        self,
        messages: list[str],
        message_offset: int = 0,
    ) -> list[ResolvedDate]:
        """Extract dates from multiple messages."""
        all_dates: list[ResolvedDate] = []
        seen_isos: set[str] = set()
        for i, msg in enumerate(messages):
            msg_index = message_offset + i
            for d in self.extract_dates(msg, message_index=msg_index):
                if d.iso not in seen_isos:
                    seen_isos.add(d.iso)
                    all_dates.append(d)
        return all_dates
