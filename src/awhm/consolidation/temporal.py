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
    start: int = 0  # character span within the source message
    end: int = 0


# Common date-like patterns to extract candidates
DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b(?:yesterday|today|tomorrow|last (?:week|month|year)|next (?:week|month|year))\b", re.IGNORECASE),
    re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE),
    re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,?\s*\d{4})?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?\b", re.IGNORECASE),
]


_FROM_MARKERS = re.compile(
    r"(?i)\b(?:from|since|starting|as of|effective|beginning)\s*(?:on|in)?\s*$"
)
_UNTIL_MARKERS = re.compile(r"(?i)\b(?:until|till|through|up to|ending)\s*(?:on|in)?\s*$")


def validity_from_context(text: str, dates: list[ResolvedDate]) -> tuple[str | None, str | None]:
    """Infer a validity window from how dates are introduced in ``text``.

    "from March 5" / "since yesterday" set ``valid_from``; "until Friday" sets
    ``valid_to``. Only the few words before each date are inspected, so a
    date mentioned in passing leaves the window open.
    """
    valid_from: str | None = None
    valid_to: str | None = None
    for d in dates:
        prefix = text[max(0, d.start - 24):d.start]
        if _FROM_MARKERS.search(prefix):
            valid_from = d.iso
        elif _UNTIL_MARKERS.search(prefix):
            valid_to = d.iso
    return valid_from, valid_to


class TemporalParser:
    """Extract and resolve date/time references."""

    def extract_dates(
        self,
        text: str,
        message_index: int | None = None,
    ) -> list[ResolvedDate]:
        """Find and resolve date references in text."""
        import dateparser

        candidates: list[tuple[str, int, int]] = []
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(text):
                candidates.append((match.group(0), match.start(), match.end()))

        results: list[ResolvedDate] = []
        seen: set[str] = set()
        for candidate, start, end in candidates:
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
                    start=start,
                    end=end,
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
