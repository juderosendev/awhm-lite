# Changelog

## 0.2.0 (August 2026)

Cleanup and hardening pass. Same architecture, same on-disk formats; existing
graphs and logs load unchanged.

### Fixed
- `MockEmbeddingService` now seeds from SHA-256 of the text. It used Python's
  salted `hash()`, so "deterministic" embeddings differed between processes and
  a graph built with `--mock` in one run could not be queried in the next.
- `AWHMConfig` expands `~` and environment variables in `data_dir`. The CLI's
  default `~/.awhm` was previously created literally as a directory named `~`.
- Cold-start raw-log hits are scaled into `[0, raw_log_score_scale]`; raw BM25
  scores could exceed 1.0 and outrank genuine graph matches.
- BM25 no longer returns documents that contain none of the query terms on tiny
  corpora (the negative-IDF shift is gone; see below).
- One message matching two extraction patterns ("Actually, I prefer Rust" is
  both a correction and a preference) produced two nodes, the second of which
  superseded the first. Candidates are now deduplicated on their statement.
- Hard delete no longer reaches into the buffer's private list; it uses
  `SessionBuffer.remove_where`.
- Missing spaCy (not just a missing model) degrades gracefully with one warning
  instead of crashing consolidation.
- `memory_log` (MCP) validates the role instead of writing arbitrary strings.

### Changed
- Contradiction rules are slot-based and conservative. `preference:i_prefer`
  collapsed every preference into one slot, so "I prefer dark mode" superseded
  "I prefer Python". The 4-word fallback key superseded unrelated statements
  that shared an opening. Both are gone. See README, "Contradictions".
- An explicit correction within `correction_window_messages` (default 3) of a
  preference or policy statement supersedes it. Facts only supersede on an
  exact slot match.
- NER output is filtered to entity labels worth remembering (`ner_labels`).
  CARDINAL, MONEY, DATE and friends no longer become nodes.
- BM25 is implemented in-package with Lucene-style IDF; `rank-bm25` dropped.
- `sentence-transformers` is an optional extra (`[embeddings]`) and is loaded
  lazily, so sessions start fast and the core install has no PyTorch.
- Retrieval caches the BM25 index until nodes change and recomputes strength
  only for ranked candidates, instead of rebuilding and rescoring the whole
  graph on every query.
- Entity linking and deduplication use a single cosine matrix product instead
  of per-pair Python loops.
- `MemoryGraph` keeps an adjacency index (edge lookups no longer scan every
  edge) and exposes `version` / `content_version` counters for caching.
- WAL flushes are skipped when the buffer has not changed.
- `Stage1Pipeline.consolidate_session` is split into candidate gathering and
  commit, with a `Candidate` dataclass replacing parallel lists.
- `AWHMSession` is a context manager and `end_session` is idempotent.

### Added
- `ruff` configuration, GitHub Actions CI (Python 3.11, 3.12, 3.13), 26 new tests
  covering every item above.

## 0.1.0 (February 2026)

Initial prototype: raw logs, session buffer, memory graph with contradiction
lifecycle, symbolic consolidation, BM25 + embedding retrieval, snapshots,
privacy hard-delete, eval harness, CLI and MCP server.
