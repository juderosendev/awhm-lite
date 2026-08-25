"""Create/restore/list timestamped JSON snapshots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import AWHMConfig
from ..graph.memory_graph import MemoryGraph
from ..graph.serialization import GRAPH_VERSION


class SnapshotManager:
    """Serialize full graph + WAL state to timestamped JSON snapshots."""

    def __init__(self, config: AWHMConfig) -> None:
        self.config = config
        config.ensure_dirs()

    def create(self, graph: MemoryGraph, wal_state: list[dict] | None = None) -> Path:
        """Create a snapshot. Returns the snapshot file path."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{ts}.json"
        path = self.config.snapshots_dir / filename

        data: dict[str, Any] = {
            "version": GRAPH_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "graph": graph.to_dict(),
            "wal_state": wal_state or [],
        }

        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

        return path

    def restore(self, snapshot_path: Path | str) -> tuple[MemoryGraph, list[dict]]:
        """Restore graph and WAL state from a snapshot."""
        path = Path(snapshot_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        graph = MemoryGraph.from_dict(data["graph"])
        wal_state = data.get("wal_state", [])

        return graph, wal_state

    def list_snapshots(self) -> list[Path]:
        """List all snapshots sorted by creation time (newest first)."""
        if not self.config.snapshots_dir.exists():
            return []
        files = sorted(
            self.config.snapshots_dir.glob("snapshot_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files

    def latest(self) -> Path | None:
        """Return the most recent snapshot path, or None."""
        snapshots = self.list_snapshots()
        return snapshots[0] if snapshots else None
