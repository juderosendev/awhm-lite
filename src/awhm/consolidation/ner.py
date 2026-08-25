"""NER via spaCy (en_core_web_sm) for Stage 1 consolidation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractedEntity:
    text: str
    label: str  # PERSON, ORG, GPE, PRODUCT, etc.
    start: int
    end: int
    message_index: int | None = None


class NERExtractor:
    """Named entity recognition using spaCy."""

    def __init__(self) -> None:
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract named entities from text."""
        self._load()
        doc = self._nlp(text)
        entities: list[ExtractedEntity] = []
        seen: set[str] = set()
        for ent in doc.ents:
            key = f"{ent.text.lower()}:{ent.label_}"
            if key not in seen:
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
        """Extract entities from multiple messages, deduplicating."""
        all_entities: list[ExtractedEntity] = []
        seen: set[str] = set()
        for i, msg in enumerate(messages):
            msg_index = message_offset + i
            for ent in self.extract(msg):
                key = f"{ent.text.lower()}:{ent.label}"
                if key not in seen:
                    seen.add(key)
                    all_entities.append(ExtractedEntity(
                        text=ent.text,
                        label=ent.label,
                        start=ent.start,
                        end=ent.end,
                        message_index=msg_index,
                    ))
        return all_entities
