"""Memory graph data models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class StrengthScore:
    recency: float = 1.0
    frequency: int = 1
    composite: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recency": self.recency,
            "frequency": self.frequency,
            "composite": self.composite,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrengthScore:
        return cls(
            recency=d["recency"],
            frequency=d["frequency"],
            composite=d["composite"],
        )


@dataclass
class MemoryNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "semantic"  # NodeType enum value
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    embed_model: str = "all-MiniLM-L6-v2"
    strength: StrengthScore = field(default_factory=StrengthScore)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_sessions: list[str] = field(default_factory=list)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    access_count: int = 1
    canonical_key: str | None = None
    status: str = "active"
    supersedes: list[str] = field(default_factory=list)
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = 0.6
    entity_type: str | None = None
    aliases: list[str] = field(default_factory=list)
    mentioned_dates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "embedding": self.embedding,
            "embed_model": self.embed_model,
            "strength": self.strength.to_dict(),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "source_sessions": self.source_sessions,
            "source_refs": self.source_refs,
            "access_count": self.access_count,
            "canonical_key": self.canonical_key,
            "status": self.status,
            "supersedes": self.supersedes,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "confidence": self.confidence,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "mentioned_dates": self.mentioned_dates,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryNode:
        created_at = d["created_at"]
        return cls(
            id=d["id"],
            type=d["type"],
            content=d["content"],
            embedding=d.get("embedding", []),
            embed_model=d.get("embed_model", "all-MiniLM-L6-v2"),
            strength=StrengthScore.from_dict(d["strength"]),
            created_at=created_at,
            last_accessed=d["last_accessed"],
            source_sessions=d.get("source_sessions", []),
            source_refs=d.get("source_refs", []),
            access_count=d.get("access_count", 1),
            canonical_key=d.get("canonical_key"),
            status=d.get("status", "active"),
            supersedes=d.get("supersedes", []),
            valid_from=d.get("valid_from", created_at),
            valid_to=d.get("valid_to"),
            confidence=float(d.get("confidence", 0.6)),
            entity_type=d.get("entity_type"),
            aliases=list(d.get("aliases", [])),
            mentioned_dates=list(d.get("mentioned_dates", [])),
        )


@dataclass
class MemoryEdge:
    source: str
    target: str
    type: str = "association"  # EdgeType enum value
    weight: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEdge:
        return cls(
            source=d["source"],
            target=d["target"],
            type=d["type"],
            weight=d["weight"],
            created_at=d["created_at"],
        )
