"""Tests for snapshot manager."""

from awhm.snapshots.manager import SnapshotManager


def test_create_snapshot(config, sample_graph):
    mgr = SnapshotManager(config)
    path = mgr.create(sample_graph)
    assert path.exists()


def test_restore_snapshot(config, sample_graph):
    mgr = SnapshotManager(config)
    path = mgr.create(sample_graph, wal_state=[{"type": "fact", "content": "test", "timestamp": "now", "source_msg": 0}])

    graph, wal = mgr.restore(path)
    assert graph.node_count() == sample_graph.node_count()
    assert len(wal) == 1


def test_list_snapshots(config, sample_graph):
    import time
    mgr = SnapshotManager(config)
    mgr.create(sample_graph)
    time.sleep(1.1)  # Ensure different timestamp
    mgr.create(sample_graph)

    snapshots = mgr.list_snapshots()
    assert len(snapshots) >= 2


def test_latest(config, sample_graph):
    mgr = SnapshotManager(config)
    mgr.create(sample_graph)
    assert mgr.latest() is not None


def test_latest_empty(config):
    mgr = SnapshotManager(config)
    assert mgr.latest() is None
