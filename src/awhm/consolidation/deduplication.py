"""Near-duplicate detection between new candidates and the existing graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph.memory_graph import MemoryGraph
from ..retrieval.embedding import pairwise_cosine
from .canonical import normalize_text, strip_correction_prefix


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
    """Find near-duplicates of ``new_contents`` among existing graph nodes.

    One matrix product scores every candidate against every stored
    embedding; a candidate is a duplicate when its best match reaches
    ``threshold``.
    """
    if new_embeddings.size == 0 or not graph.nodes:
        return []

    emb_matrix, node_ids = graph.get_embedding_matrix()
    if emb_matrix.size == 0:
        return []

    sims = pairwise_cosine(np.asarray(new_embeddings, dtype=np.float32), emb_matrix)
    best_idx = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(sims.shape[0]), best_idx]

    duplicates: list[DuplicateCandidate] = []
    for i, content in enumerate(new_contents):
        similarity = float(best_sim[i])
        if similarity < threshold:
            continue
        existing = graph.get_node(node_ids[int(best_idx[i])])
        if existing is None:
            continue
        duplicates.append(DuplicateCandidate(
            new_content=content,
            new_embedding=new_embeddings[i],
            existing_node_id=existing.id,
            existing_content=existing.content,
            similarity=similarity,
        ))
    return duplicates


def statement_key(content: str) -> str:
    """Normalized content with any correction prefix removed.

    "Actually, I prefer Rust" and "I prefer Rust" are the same statement; one
    message must not produce two nodes just because it matched two patterns.
    """
    stripped, _ = strip_correction_prefix(normalize_text(content))
    return normalize_text(stripped)


def first_occurrences(contents: list[str]) -> list[int]:
    """Indices of the first occurrence of each distinct statement."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, content in enumerate(contents):
        key = statement_key(content)
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep
