"""Tests for RetrievalEngine."""

from awhm.raw_log.logger import RawLogger
from awhm.raw_log.reader import RawLogReader
from awhm.retrieval.engine import RetrievalEngine
from awhm.session_buffer.buffer import SessionBuffer
from awhm.types import Role


def test_query_returns_results(config, sample_graph, mock_embedding):
    buffer = SessionBuffer()
    reader = RawLogReader(config)
    engine = RetrievalEngine(config, sample_graph, buffer, mock_embedding, reader)
    results = engine.query("Python programming")
    # Should find the Python node
    assert len(results) > 0


def test_buffer_hits_take_priority(config, sample_graph, mock_embedding):
    buffer = SessionBuffer()
    buffer.process_message("I prefer using Python 3.11", "2024-01-01T00:00:00Z", 0)

    reader = RawLogReader(config)
    engine = RetrievalEngine(config, sample_graph, buffer, mock_embedding, reader)
    results = engine.query("Python")
    # Buffer hit should be first
    assert len(results) > 0
    assert results[0].source == "buffer"


def test_cold_start_fallback(config, mock_embedding):
    from awhm.graph.memory_graph import MemoryGraph

    # Create raw logs but empty graph
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "Tell me about Python programming language features")
    logger.log(Role.ASSISTANT, "Python is a versatile programming language with many features")

    graph = MemoryGraph()
    buffer = SessionBuffer()
    reader = RawLogReader(config)
    engine = RetrievalEngine(config, graph, buffer, mock_embedding, reader)

    results = engine.query("Python programming language")
    # Should get raw log fallback results
    assert any(r.source == "raw_log" for r in results)


def test_query_trace_fields(config, sample_graph, mock_embedding):
    buffer = SessionBuffer()
    reader = RawLogReader(config)
    engine = RetrievalEngine(config, sample_graph, buffer, mock_embedding, reader)
    results = engine.query("Python programming", with_trace=True)
    graph_results = [r for r in results if r.source == "graph"]
    assert graph_results
    assert graph_results[0].trace is not None
    assert "semantic_sim" in graph_results[0].trace


def test_raw_log_fallback_scores_are_bounded(config, mock_embedding):
    from awhm.graph.memory_graph import MemoryGraph

    logger = RawLogger(config, "s1")
    for _ in range(5):
        logger.log(Role.USER, "Python programming Python programming Python")

    engine = RetrievalEngine(config, MemoryGraph(), SessionBuffer(), mock_embedding, RawLogReader(config))
    results = engine.query("Python programming")
    assert results
    assert all(r.source == "raw_log" for r in results)
    assert all(0.0 < r.score <= config.raw_log_score_scale for r in results)


def test_graph_hits_outrank_raw_log_fallback(config, sample_graph, mock_embedding):
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "Python programming language Python programming language")

    engine = RetrievalEngine(config, sample_graph, SessionBuffer(), mock_embedding, RawLogReader(config))
    results = engine.query("Python programming language")
    sources = [r.source for r in results]
    assert "graph" in sources and "raw_log" in sources
    assert sources.index("graph") < sources.index("raw_log")


def test_bm25_index_is_cached_until_nodes_change(config, sample_graph, mock_embedding):
    from awhm.graph.models import MemoryNode

    engine = RetrievalEngine(config, sample_graph, SessionBuffer(), mock_embedding, RawLogReader(config))
    engine.query("Python")
    first = engine._bm25
    engine.query("dark mode")
    assert engine._bm25 is first

    sample_graph.add_node(MemoryNode(id="node-4", content="Rust systems programming"))
    engine.query("Rust")
    assert engine._bm25 is not first
    assert engine._bm25.document_count == 4


def test_buffer_correction_supersedes_recent_preference(config, sample_graph, mock_embedding):
    buffer = SessionBuffer()
    buffer.process_message("I prefer Python", "2024-01-01T00:00:00Z", 0)
    buffer.process_message("Actually, I prefer Rust", "2024-01-01T00:01:00Z", 1)

    engine = RetrievalEngine(config, sample_graph, buffer, mock_embedding, RawLogReader(config))
    current = [r.content.lower() for r in engine.query("prefer") if r.source == "buffer"]
    assert any("rust" in c for c in current)
    assert not any("python" in c for c in current)

    history = [r.content.lower() for r in engine.query("prefer", include_history=True) if r.source == "buffer"]
    assert any("python" in c for c in history)
