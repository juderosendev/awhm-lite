"""Cosine > 0.92 near-duplicate detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph.memory_graph import MemoryGraph
from ..graph.models import MemoryNode
from ..retrieval.embedding import EmbeddingService, cosine_similarity


@dataclass
class DuplicateCandidate:
    new_content: str
    new_embedding: np.ndarray
    existing_node_id: str
    existing_content: str
    similarity: float


def find_duplicates(
    new_contents: list[str],
    new_embeddings: np.ndarray,
    graph: MemoryGraph,
    threshold: float = 0.92,
) -> list[DuplicateCandidate]:
    """Find near-duplicates between new content and existing graph nodes."""
    if new_embeddings.size == 0 or not graph.nodes:
        return []

    emb_matrix, node_ids = graph.get_embedding_matrix()
    if emb_matrix.size == 0:
        return []

    duplicates: list[DuplicateCandidate] = []

    for i, content in enumerate(new_contents):
        new_emb = new_embeddings[i]
        # Compare against all existing nodes
        new_norm = new_emb / (np.linalg.norm(new_emb) + 1e-8)
        m_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        m_norms = np.maximum(m_norms, 1e-8)
        sims = (emb_matrix / m_norms) @ new_norm

        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])

        if max_sim >= threshold:
            existing_node = graph.get_node(node_ids[max_idx])
            if existing_node:
                duplicates.append(DuplicateCandidate(
                    new_content=content,
                    new_embedding=new_emb,
                    existing_node_id=existing_node.id,
                    existing_content=existing_node.content,
                    similarity=max_sim,
                ))

    return duplicates
