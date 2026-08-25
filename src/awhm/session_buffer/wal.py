"""WALManager: periodic buffer flush on a background timer, plus crash recovery."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ..config import AWHMConfig
from .buffer import SessionBuffer
from .models import BufferEntry


class WALManager:
    """Write-ahead log for the session buffer.

    Each flush atomically overwrites the session's WAL file with the full
    buffer state. Flushes are skipped when the buffer has not changed since
    the last write.
    """

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
        self._last_flushed_version: int | None = None
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
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self.flush()

    def flush(self, force: bool = False) -> bool:
        """Write the buffer to the WAL. Returns True if a write happened."""
        with self._lock:
            version = self.buffer.version
            if not force and version == self._last_flushed_version:
                return False
            data = self.buffer.to_dicts()
            tmp_path = self.wal_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp_path.replace(self.wal_path)
            self._last_flushed_version = version
            return True

    def recover(self) -> int:
        """Load buffer entries from an existing WAL file. Returns count recovered."""
        if not self.wal_path.exists():
            return 0
        with open(self.wal_path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            self.buffer.add_entry(BufferEntry.from_dict(d))
        return len(data)

    def clear_wal(self) -> None:
        """Remove the WAL file."""
        if self.wal_path.exists():
            self.wal_path.unlink()
        self._last_flushed_version = None

    def _schedule_flush(self) -> None:
        if not self._running:
            return
        timer = threading.Timer(self.config.buffer_flush_interval, self._periodic_flush)
        timer.daemon = True
        with self._lock:
            self._timer = timer
        timer.start()

    def _periodic_flush(self) -> None:
        self.flush()
        self._schedule_flush()
