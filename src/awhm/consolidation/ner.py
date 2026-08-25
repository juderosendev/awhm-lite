"""Named-entity recognition via spaCy for Stage 1 consolidation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SPACY_MODEL = "en_core_web_sm"


@dataclass
class ExtractedEntity:
    text: str
    label: str  # PERSON, ORG, GPE, PRODUCT, ...
    start: int
    end: int
    message_index: int | None = None


class NERExtractor:
    """Named entity recognition using spaCy.

    The pipeline is loaded on first use. Missing spaCy or a missing model
    raise ``RuntimeError`` so the consolidation pipeline can degrade
    gracefully (it then simply skips entity extraction).
    """

    def __init__(
        self,
        model: str = SPACY_MODEL,
        labels: Iterable[str] | None = None,
    ) -> None:
        self._model_name = model
        self._labels = {label.upper() for label in labels} if labels is not None else None
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:
                raise RuntimeError(
                    "spaCy is not installed; entity extraction is unavailable."
                ) from exc
            try:
                self._nlp = spacy.load(self._model_name)
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy model '{self._model_name}' not found. Install it with: "
                    f"python -m spacy download {self._model_name}"
                ) from exc
        return self._nlp

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract named entities from ``text`` (restricted to the configured labels)."""
        nlp = self._load()
        allowed = self._labels
        entities: list[ExtractedEntity] = []
        seen: set[str] = set()
        for ent in nlp(text).ents:
            if allowed is not None and ent.label_ not in allowed:
                continue
            key = f"{ent.text.lower()}:{ent.label_}"
            if key in seen:
                continue
            seen.add(key)
            entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            ))
        return entities

    def extract_from_messages(
        self,
        messages: list[str],
        message_offset: int = 0,
    ) -> list[ExtractedEntity]:
        """Extract entities from several messages, deduplicating across them."""
        all_entities: list[ExtractedEntity] = []
        seen: set[str] = set()
        for i, msg in enumerate(messages):
            for ent in self.extract(msg):
                key = f"{ent.text.lower()}:{ent.label}"
                if key in seen:
                    continue
                seen.add(key)
                ent.message_index = message_offset + i
                all_entities.append(ent)
        return all_entities
