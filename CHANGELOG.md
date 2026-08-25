# Changelog

## 0.3.0 (August 2026)

The seven highest-leverage improvements identified after the 0.2.0 cleanup.
Same architecture; the graph gains two fields (`aliases`, `mentioned_dates`)
and old files load unchanged.

### Added
- **Claude Code hooks** (`awhm hook prompt|stop|session-end|settings`):
  memory on every turn without the model calling a tool. Sessions resume
  across hook processes via the write-ahead log; `RawLogger` continues
  message numbering from the existing file. `AWHMSession.suspend()`.
- **Stage 2 LLM refinement**: an LLM proposes the memories the rules missed;
  code validates provenance, confidence and duplicates and commits through
  the normal slot rules. Default client is the Claude Code CLI
  (`ClaudeCodeClient`, `claude -p` with structured output: no API key);
  `AnthropicClient` is an optional alternative (`[anthropic]` extra).
  `MockLLMClient`, `LLMClient` protocol, `awhm consolidate --stage2`,
  `awhm hook session-end --stage2`. Retrieval stays zero-LLM.
- **Real-corpus evaluation**: `awhm eval --corpus` replays sessions and
  scores questions (Recall@k, nDCG@k, contradiction rate, latency,
  per-category); `--longmemeval` loads LongMemEval with per-instance isolation.
- **Entity resolution**: normalised surface forms, alias matching,
  unambiguous containment, embedding fallback; aliases accumulate on the node;
  statements link to the entities they mention.
- **Time travel**: `query(as_of=...)`, `awhm query --as-of`, MCP `as_of`.
  Dates introduced with from/since/until set the fact's validity window and
  are kept as `mentioned_dates`; "Date reference" nodes are gone.
- **Neighbour expansion**: anchors pull in one-hop graph neighbours with a
  decayed edge weight (`neighbor_expansion`, `neighbor_decay`,
  `w_association`). Only current anchors expand.
- **SQLite storage** (`storage_backend="sqlite"`): one row per node, saves
  write only what changed; imports an existing JSON graph on first load.
- `query(semantic=False)` for embedding-free retrieval.

### Changed
- Lexical anchors use a relative threshold (`bm25_anchor_ratio`) instead of
  an absolute BM25 floor that tiny corpora could never reach.
- `bm25_threshold` removed.
- The sentence-transformers model is shared per process.

### Measured
- LongMemEval oracle split, Stage 1 only: Recall@5 0.196 (docs/benchmarks.md).

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
