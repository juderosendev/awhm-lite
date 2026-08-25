"""Entity linking: match extracted entities to existing graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from ..graph.memory_graph import MemoryGraph
from ..retrieval.embedding import EmbeddingService, pairwise_cosine

STRING_PREFILTER = 0.3


@dataclass
class LinkCandidate:
    entity_text: str
    matched_node_id: str
    matched_content: str
    string_sim: float
    embedding_sim: float


def string_similarity(a: str, b: str) -> float:
    """Normalized string similarity using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def strip_entity_label(content: str) -> str:
    """Turn ``"ORG: Acme"`` into ``"Acme"``; leave other content untouched."""
    if ": " in content:
        label, rest = content.split(": ", 1)
        if label.isalpha() and label.isupper():
            return rest
    return content


def find_links(
    entity_texts: list[str],
    graph: MemoryGraph,
    embedding_service: EmbeddingService,
    threshold: float = 0.85,
    entity_labels: list[str] | None = None,
) -> list[LinkCandidate]:
    """Match entity texts to existing graph nodes.

    Embedding cosine (computed for all pairs in one product) decides the
    match; entity-type agreement and a cheap string-similarity prefilter
    guard against linking unrelated nodes that happen to embed close.
    """
    if not entity_texts or not graph.nodes:
        return []

    emb_matrix, node_ids = graph.get_embedding_matrix()
    if emb_matrix.size == 0:
        return []

    entity_embeddings = embedding_service.encode(entity_texts)
    sims = pairwise_cosine(np.asarray(entity_embeddings, dtype=np.float32), emb_matrix)

    links: list[LinkCandidate] = []
    for i, entity_text in enumerate(entity_texts):
        label = entity_labels[i] if entity_labels is not None and i < len(entity_labels) else None
        candidates = np.flatnonzero(sims[i] >= threshold)
        if candidates.size == 0:
            continue
        # Best first, so the first candidate that survives the guards wins.
        candidates = candidates[np.argsort(-sims[i][candidates])]

        for j in candidates:
            node = graph.get_node(node_ids[int(j)])
            if node is None:
                continue
            if label and node.entity_type and label != node.entity_type:
                continue
            str_sim = string_similarity(entity_text, strip_entity_label(node.content))
            if str_sim < STRING_PREFILTER:
                continue
            links.append(LinkCandidate(
                entity_text=entity_text,
                matched_node_id=node.id,
                matched_content=node.content,
                string_sim=str_sim,
                embedding_sim=float(sims[i][j]),
            ))
            break

    return links
