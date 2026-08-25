"""Tests for hard deletion."""

from awhm.deletion.hard_delete import hard_delete
from awhm.graph.memory_graph import MemoryGraph
from awhm.graph.models import MemoryNode
from awhm.graph.serialization import save_graph
from awhm.raw_log.logger import RawLogger
from awhm.raw_log.reader import RawLogReader
from awhm.session_buffer.buffer import SessionBuffer
from awhm.snapshots.manager import SnapshotManager
from awhm.types import NodeType, Role


def test_delete_node(config, sample_graph):
    save_graph(sample_graph, config)
    result = hard_delete("node-1", sample_graph, config)
    assert result.node_deleted
    assert result.edges_removed == 1
    assert result.tombstone_id is not None
    assert result.ledger_record_id is not None
    assert sample_graph.get_node("node-1") is None


def test_delete_nonexistent(config, sample_graph):
    result = hard_delete("nonexistent", sample_graph, config)
    assert not result.node_deleted


def test_delete_removes_from_logs(config, sample_graph):
    # Add raw log entry matching node content
    logger = RawLogger(config, "session-1")
    logger.log(Role.USER, "Python programming language")

    save_graph(sample_graph, config)
    result = hard_delete("node-1", sample_graph, config)
    assert result.node_deleted
    assert result.log_entries_removed > 0


def test_delete_does_not_substring_match_logs(config, sample_graph):
    logger = RawLogger(config, "session-1")
    logger.log(Role.USER, "Python programming language is great")

    save_graph(sample_graph, config)
    result = hard_delete("node-1", sample_graph, config)
    assert result.node_deleted
    assert result.log_entries_removed == 0


def test_delete_uses_source_refs_for_exact_entries(config):
    logger = RawLogger(config, "session-1")
    logger.log(Role.USER, "keep this")
    logger.log(Role.USER, "delete this exact message")
    logger.log(Role.USER, "keep this too")

    graph = MemoryGraph()
    graph.add_node(
        MemoryNode(
            id="n1",
            type=NodeType.SEMANTIC.value,
            content="Derived semantic memory",
            source_sessions=["session-1"],
            source_refs=[{"session_id": "session-1", "message_index": 1}],
        ),
    )

    result = hard_delete("n1", graph, config)
    assert result.node_deleted
    assert result.match_strategy == "source_refs"
    assert result.log_entries_removed == 1
    entries = RawLogReader(config).read_session("session-1")
    assert len(entries) == 2


def test_delete_removes_from_buffer(config, sample_graph):
    buffer = SessionBuffer()
    buffer.process_message("I prefer Python programming language", "2024-01-01T00:00:00Z", 0)

    save_graph(sample_graph, config)
    result = hard_delete("node-1", sample_graph, config, buffer)
    assert result.node_deleted


def test_delete_scrubs_snapshots(config):
    logger = RawLogger(config, "session-1")
    logger.log(Role.USER, "My token is alpha-123-secret")

    graph = MemoryGraph()
    graph.add_node(
        MemoryNode(
            id="n-secret",
            type=NodeType.SEMANTIC.value,
            content="My token is alpha-123-secret",
            source_sessions=["session-1"],
            source_refs=[{"session_id": "session-1", "message_index": 0}],
        ),
    )

    snapshot_manager = SnapshotManager(config)
    path = snapshot_manager.create(graph, [])

    result = hard_delete("n-secret", graph, config)
    assert result.node_deleted
    assert result.snapshots_touched >= 1

    restored_graph, _ = snapshot_manager.restore(path)
    assert restored_graph.get_node("n-secret") is None
