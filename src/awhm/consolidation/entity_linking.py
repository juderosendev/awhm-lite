"""Entity linking: string similarity + embedding cosine > threshold."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from ..graph.memory_graph import MemoryGraph
from ..graph.models import MemoryNode
from ..retrieval.embedding import EmbeddingService, cosine_similarity


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


def find_links(
    entity_texts: list[str],
    graph: MemoryGraph,
    embedding_service: EmbeddingService,
    threshold: float = 0.85,
    entity_labels: list[str] | None = None,
) -> list[LinkCandidate]:
    """Match entity texts to existing graph nodes.

    Uses string similarity as pre-filter, then embedding cosine for final decision.
    """
    if not entity_texts or not graph.nodes:
        return []

    # Encode entities
    entity_embeddings = embedding_service.encode(entity_texts)

    # Get graph embeddings
    emb_matrix, node_ids = graph.get_embedding_matrix()

    links: list[LinkCandidate] = []

    for i, entity_text in enumerate(entity_texts):
        entity_label = None
        if entity_labels is not None and i < len(entity_labels):
            entity_label = entity_labels[i]
        best_link: LinkCandidate | None = None
        best_score = 0.0

        for j, nid in enumerate(node_ids):
            node = graph.get_node(nid)
            if node is None:
                continue

            if entity_label and node.entity_type and entity_label != node.entity_type:
                continue

            node_text = node.content
            if ": " in node_text:
                maybe_label, maybe_text = node_text.split(": ", 1)
                if maybe_label.isalpha() and maybe_label.upper() == maybe_label:
                    node_text = maybe_text

            # String similarity as quick filter
            str_sim = string_similarity(entity_text, node_text)
            if str_sim < 0.3:  # Skip clearly unrelated
                continue

            # Embedding similarity
            if emb_matrix.size > 0 and j < emb_matrix.shape[0]:
                emb_sim = cosine_similarity(entity_embeddings[i], emb_matrix[j])
            else:
                emb_sim = 0.0

            # Combined score (embedding takes precedence)
            combined = emb_sim
            if combined >= threshold and combined > best_score:
                best_score = combined
                best_link = LinkCandidate(
                    entity_text=entity_text,
                    matched_node_id=nid,
                    matched_content=node.content,
                    string_sim=str_sim,
                    embedding_sim=emb_sim,
                )

        if best_link is not None:
            links.append(best_link)

    return links
