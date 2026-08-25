"""Entity resolution: one node per real-world entity, however it is written.

"Acme", "Acme Holdings Ltd" and "acme.com" should be the same node. The
resolver normalises surface forms (case, punctuation, corporate suffixes,
domains), then matches in three increasingly permissive passes:

1. exact match on a known alias,
2. token containment ("Acme" inside "Acme Holdings") when it is unambiguous,
3. embedding similarity with the same entity type, guarded by a string
   similarity prefilter.

Every surface form that resolves is recorded as an alias on the node, so
resolution gets cheaper and more exact over time.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..config import AWHMConfig
from ..graph.memory_graph import MemoryGraph
from ..graph.models import MemoryNode
from ..retrieval.bm25 import tokenize
from ..retrieval.embedding import EmbeddingService
from .entity_linking import LinkCandidate, find_links, strip_entity_label

CORPORATE_SUFFIXES = frozenset({
    "inc", "incorporated", "ltd", "limited", "llc", "plc", "corp", "corporation",
    "co", "company", "holdings", "group", "gmbh", "ag", "sa", "bv", "pty", "llp",
})
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)(?:\.[a-z0-9-]+)+(?:/.*)?$")
_POSSESSIVE_RE = re.compile(r"'s\b")
_PUNCT_RE = re.compile(r"[^a-z0-9\s-]")

LinkFn = Callable[..., list[LinkCandidate]]


def normalize_entity(text: str) -> str:
    """Canonical surface form: lowercase, no possessive, domain -> label, no suffixes."""
    lowered = text.strip().lower()
    domain = _DOMAIN_RE.match(lowered)
    if domain and " " not in lowered:
        return domain.group(1).replace("-", " ")
    lowered = _POSSESSIVE_RE.sub("", lowered)
    lowered = _PUNCT_RE.sub(" ", lowered)
    tokens = [t for t in lowered.split() if t not in CORPORATE_SUFFIXES]
    return " ".join(tokens) if tokens else lowered.strip()


def _is_entity_node(node: MemoryNode) -> bool:
    if node.entity_type:
        return True
    return strip_entity_label(node.content) != node.content


def _label_of(node: MemoryNode) -> str | None:
    if node.entity_type:
        return node.entity_type
    if ": " in node.content:
        label = node.content.split(": ", 1)[0]
        if label.isalpha() and label.isupper():
            return label
    return None


def _surface_forms(node: MemoryNode) -> list[str]:
    forms = [strip_entity_label(node.content)]
    forms.extend(a for a in node.aliases if a)
    return forms


class EntityResolver:
    """Resolve entity mentions against the graph's entity nodes."""

    def __init__(
        self,
        graph: MemoryGraph,
        embedding: EmbeddingService,
        config: AWHMConfig,
        link_fn: LinkFn = find_links,
    ) -> None:
        self.graph = graph
        self.embedding = embedding
        self.config = config
        self._link_fn = link_fn
        self._alias_index: dict[str, set[str]] = {}
        self._index_version: int | None = None

    # ── Index ──────────────────────────────────────────────────

    def _index(self) -> dict[str, set[str]]:
        if self._index_version != self.graph.version:
            index: dict[str, set[str]] = {}
            for node in self.graph.nodes.values():
                if not _is_entity_node(node):
                    continue
                for form in _surface_forms(node):
                    key = normalize_entity(form)
                    if key:
                        index.setdefault(key, set()).add(node.id)
            self._alias_index = index
            self._index_version = self.graph.version
        return self._alias_index

    def _compatible(self, node: MemoryNode, label: str | None) -> bool:
        node_label = _label_of(node)
        return not label or not node_label or node_label == label

    # ── Resolution ─────────────────────────────────────────────

    def resolve(self, text: str, label: str | None = None) -> MemoryNode | None:
        """Return the existing entity node ``text`` refers to, or ``None``."""
        key = normalize_entity(text)
        if not key:
            return None
        index = self._index()

        # 1. Exact alias match
        for nid in index.get(key, ()):
            node = self.graph.get_node(nid)
            if node is not None and self._compatible(node, label):
                return node

        # 2. Unambiguous token containment with a shared first token
        tokens = key.split()
        matches: set[str] = set()
        for alias, ids in index.items():
            alias_tokens = alias.split()
            if alias_tokens[0] != tokens[0]:
                continue
            small, big = (tokens, alias_tokens) if len(tokens) <= len(alias_tokens) else (alias_tokens, tokens)
            if set(small) <= set(big) and len(big) - len(small) <= 2:
                matches |= {
                    nid for nid in ids
                    if (n := self.graph.get_node(nid)) is not None and self._compatible(n, label)
                }
        if len(matches) == 1:
            return self.graph.get_node(next(iter(matches)))

        # 3. Embedding similarity, same entity type, string-similarity guarded
        links = self._link_fn(
            [text], self.graph, self.embedding,
            threshold=self.config.entity_link_threshold,
            entity_labels=[label or ""],
        )
        for link in links:
            node = self.graph.get_node(link.matched_node_id)
            if node is not None and _is_entity_node(node) and self._compatible(node, label):
                return node
        return None

    def register_alias(self, node: MemoryNode, text: str) -> bool:
        """Record ``text`` as a surface form of ``node``. Returns True if new."""
        known = {normalize_entity(f) for f in _surface_forms(node)}
        if normalize_entity(text) in known and text.strip() in _surface_forms(node):
            return False
        if text.strip() not in node.aliases and text.strip() != strip_entity_label(node.content):
            node.aliases.append(text.strip())
            self.graph.mark_dirty(node.id)
            self._index_version = None
            return True
        return False

    # ── Mentions ───────────────────────────────────────────────

    def mentions(self, content: str, candidates: list[MemoryNode] | None = None) -> list[MemoryNode]:
        """Entity nodes whose alias appears as a phrase in ``content``."""
        nodes = candidates if candidates is not None else [
            n for n in self.graph.nodes.values() if _is_entity_node(n)
        ]
        content_tokens = tokenize(content)
        if not content_tokens:
            return []
        joined = " " + " ".join(content_tokens) + " "
        found: list[MemoryNode] = []
        for node in nodes:
            for form in _surface_forms(node):
                phrase = tokenize(normalize_entity(form))
                if not phrase or (len(phrase) == 1 and len(phrase[0]) < 3):
                    continue
                if " " + " ".join(phrase) + " " in joined:
                    found.append(node)
                    break
        return found
