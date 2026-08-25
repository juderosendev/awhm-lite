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
