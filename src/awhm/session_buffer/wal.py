"""WALManager: 30s flush via background thread, crash recovery."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ..config import AWHMConfig
from .buffer import SessionBuffer
from .models import BufferEntry


class WALManager:
    """Write-ahead log for session buffer. Full-state overwrite every flush_interval."""

    def __init__(
        self,
        config: AWHMConfig,
        buffer: SessionBuffer,
        session_id: str,
    ) -> None:
        self.config = config
        self.buffer = buffer
        self.session_id = session_id
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._running = False
        config.ensure_dirs()

    @property
    def wal_path(self) -> Path:
        return self.config.wal_path_for_session(self.session_id)

    def start(self) -> None:
        """Start periodic WAL flushing."""
        self._running = True
        self._schedule_flush()

    def stop(self) -> None:
        """Stop periodic flushing and do a final flush."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.flush()

    def flush(self) -> None:
        """Write current buffer state to WAL (atomic)."""
        with self._lock:
            data = self.buffer.to_dicts()
            tmp_path = self.wal_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp_path.replace(self.wal_path)

    def recover(self) -> int:
        """Recover buffer entries from WAL file. Returns count recovered."""
        if not self.wal_path.exists():
            return 0
        with open(self.wal_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for d in data:
            self.buffer.add_entry(BufferEntry.from_dict(d))
            count += 1
        return count

    def clear_wal(self) -> None:
        """Remove the WAL file."""
        if self.wal_path.exists():
            self.wal_path.unlink()

    def _schedule_flush(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self.config.buffer_flush_interval, self._periodic_flush)
        self._timer.daemon = True
        self._timer.start()

    def _periodic_flush(self) -> None:
        self.flush()
        self._schedule_flush()
