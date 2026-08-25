"""Tests for as-of queries, lexical-only mode and neighbour expansion."""

from awhm.graph.memory_graph import MemoryGraph
from awhm.graph.models import MemoryEdge, MemoryNode
from awhm.raw_log.reader import RawLogReader
from awhm.retrieval.engine import RetrievalEngine
from awhm.session_buffer.buffer import SessionBuffer
from awhm.timeutil import valid_at
from awhm.types import EdgeType, NodeType


def _graph_with_history(mock_embedding):
    g = MemoryGraph()
    old_emb, new_emb = mock_embedding.encode(["The API endpoint is v1", "The API endpoint is v2"])
    g.add_node(MemoryNode(
        id="old", type=NodeType.SEMANTIC.value, content="The API endpoint is v1",
        embedding=old_emb.tolist(), canonical_key="fact:the api endpoint",
        status="superseded", valid_from="2026-01-01T00:00:00+00:00",
        valid_to="2026-03-01T00:00:00+00:00", created_at="2026-01-01T00:00:00+00:00",
        last_accessed="2026-01-01T00:00:00+00:00",
    ))
    g.add_node(MemoryNode(
        id="new", type=NodeType.SEMANTIC.value, content="The API endpoint is v2",
        embedding=new_emb.tolist(), canonical_key="fact:the api endpoint",
        status="active", valid_from="2026-03-01T00:00:00+00:00", valid_to=None,
        created_at="2026-03-01T00:00:00+00:00", last_accessed="2026-03-01T00:00:00+00:00",
        supersedes=["old"],
    ))
    return g


def test_valid_at_window():
    assert valid_at("2026-01-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00", __import__("awhm.timeutil").timeutil.parse_timestamp("2026-02-01"))
    assert not valid_at("2026-01-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00", __import__("awhm.timeutil").timeutil.parse_timestamp("2026-03-01"))
    assert valid_at(None, None, __import__("awhm.timeutil").timeutil.parse_timestamp("1999-01-01"))


def test_as_of_returns_what_was_true_then(config, mock_embedding):
    config.cold_start_session_count = 0
    engine = RetrievalEngine(config, _graph_with_history(mock_embedding), SessionBuffer(), mock_embedding, RawLogReader(config))

    now = [r.content for r in engine.query("API endpoint")]
    assert now == ["The API endpoint is v2"]

    february = [r.content for r in engine.query("API endpoint", as_of="2026-02-01")]
    assert february == ["The API endpoint is v1"]

    before_anything = engine.query("API endpoint", as_of="2025-12-01")
    assert before_anything == []


def test_lexical_only_mode_skips_embeddings(config, sample_graph):
    class ExplodingEmbeddings:
        dim = 384

        def encode(self, texts):
            raise AssertionError("embeddings must not be used")

        def encode_single(self, text):
            raise AssertionError("embeddings must not be used")

    config.cold_start_session_count = 0
    engine = RetrievalEngine(config, sample_graph, SessionBuffer(), ExplodingEmbeddings(), RawLogReader(config))
    results = engine.query("Python programming", semantic=False, with_trace=True)
    assert results and results[0].content == "Python programming language"
    assert results[0].trace["semantic_sim"] == 0.0


def test_neighbour_expansion_pulls_in_linked_nodes(config, mock_embedding):
    config.cold_start_session_count = 0
    g = MemoryGraph()
    embs = mock_embedding.encode(["ORG: Acme", "Scope v4 approved"])
    g.add_node(MemoryNode(id="acme", content="ORG: Acme", embedding=embs[0].tolist(), entity_type="ORG"))
    g.add_node(MemoryNode(id="scope", content="Scope v4 approved", embedding=embs[1].tolist()))
    g.add_edge(MemoryEdge(source="scope", target="acme", type=EdgeType.ASSOCIATION.value, weight=0.9))

    engine = RetrievalEngine(config, g, SessionBuffer(), mock_embedding, RawLogReader(config))
    results = engine.query("Acme", with_trace=True)
    by_id = {r.node_id: r for r in results}
    assert "acme" in by_id and "scope" in by_id
    assert by_id["scope"].trace["association"] > 0
    assert by_id["acme"].score > by_id["scope"].score

    config.neighbor_expansion = False
    engine = RetrievalEngine(config, g, SessionBuffer(), mock_embedding, RawLogReader(config))
    assert {r.node_id for r in engine.query("Acme")} == {"acme"}
