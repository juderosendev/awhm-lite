"""Canonical keys and correction rules for contradiction-aware memory.

A canonical key names the *slot* a memory fills ("fact:my name",
"preference:python", "policy:use:tabs"). Two active memories with the same
key contradict each other, so the newer one supersedes the older one.

The rules are deliberately conservative. Without an LLM in the loop we can
only recognise slots from surface form, so anything we cannot key with
confidence gets ``None`` and stays additive. See ``docs/`` for the ceiling
this implies.
"""

from __future__ import annotations

import re

from ..retrieval.bm25 import tokenize

_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\s,.:;!?-]+|[\s,.:;!?-]+$")

_CORRECTION_PREFIX_RE = re.compile(
    r"(?i)^(?:actually|correction|i meant|"
    r"no[,:]?\s+(?:it(?:'s| is)|that(?:'s| is))|"
    r"that(?:'s| is)\s+(?:wrong|incorrect|not right))\s*[,:\-]?\s*"
)

_IS_SLOT_RE = re.compile(r"(?i)^(?P<left>.+?)\s+(?:is|was|are|were)\s+(?P<right>.+)$")
_LIVE_IN_RE = re.compile(r"(?i)^i (?:live|am based|'m based) in\s+(.+)$")
_WORK_AT_RE = re.compile(r"(?i)^i work (?:at|for)\s+(.+)$")
_PREF_RE = re.compile(r"(?i)^(?:i|we)\s+prefer\s+(?:to\s+(?:use|have)\s+)?(?:using\s+)?(.+)$")
_POLICY_RE = re.compile(
    r"(?i)^(?:please\s+)?(?P<mode>always|never|don'?t(?:\s+ever)?|do not)\s+"
    r"(?P<verb>use|do|make|keep|write)\s+(?P<object>.+)$"
)

# Slots whose key already encodes the subject. Corrections only supersede
# these on an exact key match, never by proximity.
_SUBJECT_FAMILIES = frozenset({"fact"})
# Slots where the subject is implicit ("I prefer X"). An explicit correction
# close in the conversation supersedes the previous statement of the family.
_IMPLICIT_FAMILIES = frozenset({"preference", "policy"})

MAX_SUBJECT_WORDS = 8


def normalize_text(text: str) -> str:
    text = _WS_RE.sub(" ", text.strip().lower())
    return _PUNCT_EDGE_RE.sub("", text)


def strip_correction_prefix(text: str) -> tuple[str, bool]:
    """Remove a leading correction marker. Returns ``(text, was_correction)``."""
    stripped = _CORRECTION_PREFIX_RE.sub("", text, count=1).strip()
    return stripped, stripped != text.strip()


def is_correction(text: str) -> bool:
    return strip_correction_prefix(text)[1]


def _head_token(phrase: str) -> str | None:
    tokens = tokenize(phrase)
    return tokens[0] if tokens else None


def canonical_key_for_content(content: str, memory_type: str | None = None) -> str | None:
    """Derive the slot key for ``content``, or ``None`` when no slot is recognised.

    ``memory_type`` is accepted for backward compatibility; keys depend only
    on the surface form of the statement.
    """
    if not content or not content.strip():
        return None

    cleaned, _ = strip_correction_prefix(normalize_text(content))
    cleaned = normalize_text(cleaned)
    if not cleaned:
        return None

    match = _LIVE_IN_RE.match(cleaned)
    if match:
        return "fact:i live in"

    match = _WORK_AT_RE.match(cleaned)
    if match:
        return "fact:i work at"

    match = _PREF_RE.match(cleaned)
    if match:
        head = _head_token(match.group(1))
        return f"preference:{head}" if head else None

    match = _POLICY_RE.match(cleaned)
    if match:
        head = _head_token(match.group("object"))
        if head:
            return f"policy:{match.group('verb').lower()}:{head}"
        return None

    match = _IS_SLOT_RE.match(cleaned)
    if match:
        left = normalize_text(match.group("left"))
        if left and len(left.split()) <= MAX_SUBJECT_WORDS:
            return f"fact:{left}"

    return None


def key_family(key: str | None) -> str | None:
    """Return the family prefix of a canonical key ("fact", "preference", ...)."""
    if not key or ":" not in key:
        return None
    return key.split(":", 1)[0]


def correction_supersedes(
    new_key: str | None,
    new_is_correction: bool,
    old_key: str | None,
    message_distance: int | None,
    window: int,
) -> bool:
    """Decide whether a new statement supersedes an older one.

    * Same key: always (the slot is being restated, so the newer value wins).
    * Implicit-subject families (preference, policy): an explicit correction
      within ``window`` messages of the older statement supersedes it.
    * Subject-bearing families (fact): only on an exact key match.
    """
    if new_key is None or old_key is None:
        return False
    if new_key == old_key:
        return True
    if not new_is_correction:
        return False
    family = key_family(new_key)
    if family != key_family(old_key) or family not in _IMPLICIT_FAMILIES:
        return False
    if message_distance is None:
        return False
    return 0 < message_distance <= window
