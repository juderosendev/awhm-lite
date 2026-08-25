"""Timestamp helpers shared across the package."""

from __future__ import annotations

from datetime import date, datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: str | datetime | date) -> datetime:
    """Parse an ISO-8601 string (or pass through a datetime) as an aware UTC datetime."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def valid_at(valid_from: str | None, valid_to: str | None, moment: datetime) -> bool:
    """True when ``moment`` falls inside the half-open window ``[valid_from, valid_to)``."""
    if valid_from and parse_timestamp(valid_from) > moment:
        return False
    return not (valid_to and parse_timestamp(valid_to) <= moment)
