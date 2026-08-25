"""Compiled regex patterns for session buffer pattern matching."""

from __future__ import annotations

import re
from typing import NamedTuple

from ..types import BufferEntryType


class PatternMatch(NamedTuple):
    type: BufferEntryType
    content: str


# Correction patterns: "actually, X is Y", "no, it's X", "that's wrong, X"
CORRECTION_PATTERNS = [
    re.compile(r"(?i)\bactually[,:]?\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\bno[,:]?\s+(?:it(?:'s| is)|that(?:'s| is))\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\bthat(?:'s| is) (?:wrong|incorrect|not right)[,.]?\s*(.+)", re.DOTALL),
    re.compile(r"(?i)\bcorrection[:\s]+(.+)", re.DOTALL),
    re.compile(r"(?i)\bI meant\s+(.+)", re.DOTALL),
]

# Preference patterns: "I prefer X", "always use X", "don't do X", "never use X"
PREFERENCE_PATTERNS = [
    re.compile(r"(?i)\bI prefer\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\balways (?:use|do|make|keep|write)\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\bnever (?:use|do|make|keep|write)\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\bdon'?t (?:ever )?(?:use|do|make|keep|write)\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\bplease (?:always|never)\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\bI (?:like|want|need) (?:to use |to have |)\s*(.+)", re.DOTALL),
]

# Fact patterns: "the endpoint is X", "my name is X", "the password is X"
FACT_PATTERNS = [
    re.compile(r"(?i)\bthe (?:api |)(?:endpoint|url|path|port|host|server|database|repo|repository) (?:is|runs? (?:on|at))\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\b(?:the|my|our)\s+(?:\w+\s+){0,3}is\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\b(?:it|this) (?:is|was) (?:called|named|located at)\s+(.+)", re.DOTALL),
    re.compile(r"(?i)\bI (?:am|work at|work on|use|live in)\s+(.+)", re.DOTALL),
]

# Outcome patterns: detect tool success/failure sequences
OUTCOME_PATTERNS = [
    re.compile(r"(?i)\b(?:that|it) (?:worked|succeeded|passed|fixed it)", re.DOTALL),
    re.compile(r"(?i)\b(?:that|it) (?:failed|didn'?t work|broke|errored)", re.DOTALL),
    re.compile(r"(?i)\b(?:error|exception|traceback|failure)[:\s]+(.+)", re.DOTALL),
    re.compile(r"(?i)\bsuccess(?:fully)?[:\s]+(.+)", re.DOTALL),
]

ALL_PATTERNS: list[tuple[BufferEntryType, list[re.Pattern[str]]]] = [
    (BufferEntryType.CORRECTION, CORRECTION_PATTERNS),
    (BufferEntryType.PREFERENCE, PREFERENCE_PATTERNS),
    (BufferEntryType.FACT, FACT_PATTERNS),
    (BufferEntryType.OUTCOME, OUTCOME_PATTERNS),
]


def match_message(content: str) -> list[PatternMatch]:
    """Match a message against all patterns. Returns all matches found."""
    matches: list[PatternMatch] = []
    for entry_type, patterns in ALL_PATTERNS:
        for pattern in patterns:
            m = pattern.search(content)
            if m:
                matched_content = m.group(0).strip()
                matches.append(PatternMatch(type=entry_type, content=matched_content))
                break  # One match per category is enough
    return matches
