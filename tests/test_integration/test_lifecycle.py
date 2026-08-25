"""Integration test: full session lifecycle."""

from awhm import AWHMSession
from awhm.config import AWHMConfig
from awhm.types import Role


def test_full_lifecycle(tmp_path):
    """Log -> buffer -> consolidate -> retrieve."""
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"))

    # Session 1: log some messages
    s1 = AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True)
    s1.log_message(Role.USER, "My name is Alice")
    s1.log_message(Role.ASSISTANT, "Hello Alice!")
    s1.log_message(Role.USER, "I prefer Python over JavaScript")
    s1.log_message(Role.USER, "The API endpoint is https://api.example.com/v2")
    s1.log_message(Role.USER, "Always use black for formatting")

    # Check buffer captured some patterns
    assert len(s1.buffer.entries) > 0

    # Check status
    status = s1.status()
    assert status["session_id"] == "s1"
    assert status["log_messages"] == 5

    s1.end_session()

    # Session 2: consolidate and query
    s2 = AWHMSession.start_session(config, session_id="s2", use_mock_embeddings=True)

    # Consolidate session 1
    results = s2.consolidate()
    assert "s1" in results
    assert results["s1"] > 0

    # Graph should now have nodes
    assert s2.graph.node_count() > 0

    # Query should return results
    query_results = s2.query("Python")
    assert len(query_results) > 0

    s2.end_session()


def test_snapshot_restore(tmp_path):
    """Create snapshot, modify, restore."""
    config = AWHMConfig(
        data_dir=str(tmp_path / "awhm"),
        cold_start_session_count=0,
    )

    s1 = AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True)
    s1.log_message(Role.USER, "I prefer dark mode")
    s1.consolidate_current()

    original_count = s1.graph.node_count()
    snap_path = s1.create_snapshot()

    # Add more nodes
    s1.log_message(Role.USER, "The server port is 8080")
    s1.consolidate_current()

    # Restore
    s1.restore_snapshot(snap_path)
    assert s1.graph.node_count() == original_count
    restore_results = s1.query("server port 8080")
    assert all("8080" not in r.content for r in restore_results)

    s1.end_session()


def test_cold_start_fallback(tmp_path):
    """Verify BM25 raw log fallback with <10 sessions."""
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"))

    # Create a few sessions with content but no consolidation
    for i in range(3):
        s = AWHMSession.start_session(config, session_id=f"s{i}", use_mock_embeddings=True)
        s.log_message(Role.USER, f"Session {i}: Python programming is great")
        s.log_message(Role.ASSISTANT, f"Indeed, Python is widely used")
        s.end_session()

    # Query without consolidation — should use cold-start fallback
    s = AWHMSession.start_session(config, session_id="query-session", use_mock_embeddings=True)
    results = s.query("Python programming")
    assert any(r.source == "raw_log" for r in results)
    s.end_session()


def test_incremental_consolidation_same_session(tmp_path):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"))

    s = AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True)
    s.log_message(Role.USER, "I prefer Python")
    first = s.consolidate_current()
    assert first > 0

    s.log_message(Role.USER, "Actually, I prefer Rust")
    second = s.consolidate_current()
    assert second > 0
    s.end_session()


def test_correction_supersedes_old_memory(tmp_path):
    config = AWHMConfig(
        data_dir=str(tmp_path / "awhm"),
        cold_start_session_count=0,
    )

    s = AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True)
    s.log_message(Role.USER, "My preferred language is Python")
    s.consolidate_current()

    s.log_message(Role.USER, "Actually, my preferred language is Rust")
    s.consolidate_current()

    current_results = s.query("preferred language", include_history=False, k=10)
    current_text = " ".join(r.content.lower() for r in current_results)
    assert "rust" in current_text
    assert "python" not in current_text

    historical_results = s.query("preferred language", include_history=True, k=10)
    historical_text = " ".join(r.content.lower() for r in historical_results)
    assert "rust" in historical_text
    assert "python" in historical_text

    statuses = [
        n.status for n in s.graph.all_nodes()
        if n.canonical_key == "fact:my preferred language"
    ]
    assert "superseded" in statuses
    assert "active" in statuses
    s.end_session()
