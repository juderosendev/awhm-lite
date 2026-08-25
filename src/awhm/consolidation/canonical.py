"""Canonical-key helpers for contradiction-aware memory updates."""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\s,.:;!?-]+|[\s,.:;!?-]+$")

_CORRECTION_PREFIX_RE = re.compile(
    r"(?i)^(?:actually|correction|i meant|"
    r"no[,:]?\s+(?:it(?:'s| is)|that(?:'s| is))|"
    r"that(?:'s| is)\s+(?:wrong|incorrect|not right))\s*[,:\-]?\s*"
)

_IS_SLOT_RE = re.compile(r"(?i)^(?P<left>.+?)\s+(?:is|was|are|were)\s+(?P<right>.+)$")
_PREF_RE = re.compile(r"(?i)^(?:i|we)\s+prefer\s+(.+)$")
_POLICY_RE = re.compile(r"(?i)^(always|never|don'?t|do not)\s+(use|do|make|keep|write)\b")


def normalize_text(text: str) -> str:
    text = _WS_RE.sub(" ", text.strip().lower())
    return _PUNCT_EDGE_RE.sub("", text)


def canonical_key_for_content(content: str, memory_type: str) -> str | None:
    """Derive a canonical key used to supersede stale memories.

    This is intentionally conservative and deterministic; unknown forms
    fall back to None so no supersession is triggered.
    """
    if not content or not content.strip():
        return None

    cleaned = normalize_text(content)
    cleaned = _CORRECTION_PREFIX_RE.sub("", cleaned).strip()
    if not cleaned:
        return None

    # Common factual slot style: "<subject> is <value>"
    is_match = _IS_SLOT_RE.match(cleaned)
    if is_match:
        left = normalize_text(is_match.group("left"))
        if left:
            return f"fact:{left}"

    # Preference statements collapse to a stable intent key.
    if _PREF_RE.match(cleaned):
        return "preference:i_prefer"

    policy_match = _POLICY_RE.match(cleaned)
    if policy_match:
        return f"policy:{policy_match.group(1).lower()}_{policy_match.group(2).lower()}"

    # NER/temporal/system content are often additive and should not supersede by default.
    if memory_type in ("semantic", "procedural"):
        parts = cleaned.split()
        if len(parts) >= 4:
            return f"{memory_type}:{' '.join(parts[:4])}"
    return None
