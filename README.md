# AWHM Lite

[![CI](https://github.com/juderosendev/awhm-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/juderosendev/awhm-lite/actions/workflows/ci.yml)

External long-term memory for LLM agents. No cloud, no API keys, runs entirely local.

AWHM Lite gives any LLM persistent memory across conversations through append-only logging, regex-based pattern matching, a contradiction-aware memory graph, symbolic consolidation (zero LLM calls), and retrieval via lexical + semantic feature fusion.

**Status:** research prototype. Built February 2026, published August 2026, cleaned up and hardened in v0.2.0. Working and tested (113 tests, CI on Python 3.11 to 3.13); not under active development.

## Project docs

- `docs/awhm-whitepaper.md`: the full AWHM architecture paper this is a subset of
- `docs/awhm-whitepaper-vs-lite.md`: what Lite keeps and what it leaves out
- `docs/Future Plans.md`: the planned next step (silent per-turn middleware)

## How this was built

The architecture and the ideas behind it are mine. The code was written entirely by AI coding agents (mainly Claude Code) under my direction: I set the design, scoped the tasks, reviewed the output and steered. The whitepaper was produced the same way.

```
INTERACTION TIME                        OFFLINE (SESSION END)
────────────────────                    ─────────────────────
┌──────────────────┐   real-time log    ┌──────────────────────┐
│  PRIMARY AGENT   │──────────────────► │  STAGE 1 CONSOLIDATION│
│  (user-facing)   │   (middleware,     │  (symbolic only,      │
└──────┬───────────┘    no LLM)         │   zero LLM calls)     │
       │                                └──────────┬───────────┘
       │ queries                                   │ writes
       ▼                                           ▼
┌──────────────┐    ┌──────────┐    ┌──────────────────────┐
│  RETRIEVAL   │◄───│ SESSION  │    │    FLAT MEMORY GRAPH  │
│  ENGINE      │    │ BUFFER   │    │                      │
│              │◄───┤(checked  │    │  nodes: episodic,    │
│ BM25 +       │    │ first)   │    │  semantic, procedural│
│ embedding    │    └──────────┘    │                      │
│ similarity   │◄───────────────────│  edges: typed        │
│              │                    │  strength: rec + freq│
└──────────────┘                    └──────────────────────┘
       ▲
       │ fallback (first ~10 sessions)
┌──────┴───────┐
│   RAW LOGS   │
│ (append-only)│
└──────────────┘
```

## Install

```bash
# Core (numpy, spaCy, dateparser) plus the sentence-transformers embedding model
pip install -e ".[embeddings]"

# spaCy NER model (used in consolidation; without it, entity extraction is skipped)
python -m spacy download en_core_web_sm

# Claude Code MCP integration
pip install -e ".[mcp]"
```

`sentence-transformers` is optional because it pulls in PyTorch. Without it, start
sessions with `use_mock_embeddings=True` (deterministic hash-based vectors, fine for
tests and for trying the CLI). The real model (`all-MiniLM-L6-v2`, 22 MB) downloads
on first use.

## Quick start

### Python API

```python
from awhm import AWHMSession
from awhm.types import Role

# Start a session (also usable as a context manager: `with AWHMSession.start_session() as session:`)
session = AWHMSession.start_session()

# Log messages
session.log_message(Role.USER, "My name is Alice")
session.log_message(Role.ASSISTANT, "Hello Alice!")
session.log_message(Role.USER, "I prefer Python over JavaScript")
session.log_message(Role.USER, "The API endpoint is https://api.example.com/v2")

# Query memory (works immediately via session buffer)
results = session.query("What language does the user prefer?")
for r in results:
    print(f"[{r.source}] {r.content}")

# Consolidate into long-term memory graph
session.consolidate_current()

# End session (flushes WAL, saves graph)
session.end_session()
```

### Integrating with an LLM

AWHM sits as middleware. It doesn't call any LLM — you wire it into whatever you use:

```python
from awhm import AWHMSession
from awhm.types import Role

session = AWHMSession.start_session()

def handle_message(user_text):
    session.log_message(Role.USER, user_text)

    # Retrieve relevant memories
    memories = session.query(user_text, k=5)
    memory_context = "\n".join(f"- {m.content}" for m in memories)

    # Inject into system prompt
    system = f"Memories from past conversations:\n{memory_context}"
    response = your_llm_call(system_prompt=system, user_message=user_text)

    session.log_message(Role.ASSISTANT, response)
    return response

# At end of conversation:
session.consolidate_current()
session.end_session()
```

### CLI

```bash
awhm status                        # Show system stats
awhm query "Python preferences"    # Search memory
awhm query "API endpoint" --include-history --trace
awhm consolidate                   # Run Stage 1 on pending sessions
awhm snapshot create               # Backup current graph
awhm snapshot list                 # List snapshots
awhm snapshot restore --path FILE  # Restore from snapshot
awhm delete NODE_ID                # Hard-delete a node (privacy)
awhm eval --json                   # Run built-in benchmark report
```

## Claude Code integration (MCP)

AWHM Lite ships as an MCP server so Claude Code can use it as a tool.

### Setup

```bash
# Install with MCP support
cd awhm-lite
pip install -e ".[mcp]"

# Register with Claude Code
claude mcp add --transport stdio awhm-lite -- awhm-mcp
```

Or manually add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "awhm-lite": {
      "type": "stdio",
      "command": "awhm-mcp",
      "env": {
        "AWHM_DATA_DIR": "~/.awhm"
      }
    }
  }
}
```

### Available MCP tools

| Tool | Description |
|------|-------------|
| `memory_query` | Search memory for a natural language query (`include_history`, `with_trace` optional) |
| `memory_log` | Log a message to the raw conversation log |
| `memory_consolidate` | Extract memories from pending sessions into the graph |
| `memory_status` | Show node count, edge count, session count |
| `memory_snapshot_create` | Create a backup snapshot |
| `memory_delete_node` | Hard-delete a node + scrub matching snapshot data |

Once connected, Claude Code will automatically have access to these tools and can query/store memories across conversations.

## How it works

### Raw logs
Every message is appended to a JSONL file (one per session). Append-only, never modified except for privacy hard-deletes. This is the ground truth.

### Session buffer
A regex-based pattern matcher runs on every user message in real time, catching:
- **Corrections**: "actually, X is Y", "no, it's X"
- **Preferences**: "I prefer X", "always use X", "never do X"
- **Facts**: "the endpoint is X", "my name is X"
- **Outcomes**: "that worked", "that failed"

Captures ~60-70% of explicit signals with zero LLM calls. The buffer is checked first during retrieval for instant intra-session continuity. During default retrieval, buffer entries that a later statement supersedes (same slot, or an explicit correction a few messages later) are hidden, so corrections win. Persisted via per-session write-ahead logs (30s flush interval, skipped when nothing changed).

### Memory graph
A flat directed graph with three node types (**episodic**, **semantic**, **procedural**) and three edge types (**temporal**, **abstraction**, **association**).  

Each node now carries contradiction-lifecycle metadata:
- `canonical_key` (slot-style identity, e.g. `fact:my preferred language`)
- `status` (`active`, `superseded`, `retracted`)
- `supersedes` (older node IDs replaced by this node)
- `valid_from` / `valid_to`
- `confidence`

Stored as JSON, loaded into memory.

Backward compatibility: older graph files (without lifecycle fields) are auto-migrated in-memory on load.

### Strength scoring
Each node has a composite strength score:

```
S(v) = 0.4 * recency + 0.6 * frequency
```

Recency uses power-law decay: `s_rec = (1 + 0.1 * hours)^(-0.3)` — roughly 0.71 at 24h, 0.40 at 7 days, 0.27 at 30 days. Frequency is access count normalized against the 90th percentile.

### Consolidation (Stage 1)
Runs at session end, zero LLM calls:
1. **NER** via spaCy: people, orgs, places, products. Numeric and time-like labels (CARDINAL, MONEY, DATE, ...) are filtered out; they made noise nodes. Configurable via `ner_labels`.
2. **Temporal parsing** via dateparser: resolve "yesterday", "March 5" to ISO timestamps
3. **Rule-based extraction**: same regex patterns as the session buffer, over the new messages
4. **Entity linking**: match entities to existing nodes (one cosine matrix product, then entity-type agreement and a string-similarity guard)
5. **Deduplication**: identical statements within the batch are collapsed; near-duplicates of existing nodes (cosine > 0.92) reinforce that node instead of creating a new one
6. **Commit**: assign canonical keys, supersede contradicted memories, add nodes and edges, refresh strength scores

### Contradictions: canonical keys
A canonical key names the *slot* a statement fills. Two active memories with the same key contradict each other, so the newer one supersedes the older (`status=superseded`, `valid_to` set, `supersedes` link on the new node).

| Statement | Key |
|-----------|-----|
| "My preferred language is Python" | `fact:my preferred language` |
| "I live in Cape Town" | `fact:i live in` |
| "I prefer dark mode" | `preference:dark` |
| "Never use tabs for indentation" | `policy:use:tabs` |
| "I use Python for scripting" | none (additive) |

Rules, deliberately conservative because there is no LLM to judge intent:
- **Same key**: always supersedes (the slot is restated with a new value).
- **Preference / policy families**: an explicit correction ("Actually, I prefer Rust") supersedes the previous statement of the same family if it comes within `correction_window_messages` (default 3) of it in the same session. Without a correction marker, preferences are additive: "I prefer tabs" and "I prefer dark mode" both stay active.
- **Fact family**: only an exact key match supersedes, so a correction about the API endpoint can never clobber your name.
- Anything not recognised gets no key and never supersedes.

### Retrieval
Zero LLM calls. Feature-based fusion:

1. **Buffer check**: search the session buffer first (instant hits, always ranked above graph results)
2. **Anchor identification**: BM25 term overlap plus embedding cosine similarity (union). The BM25 index is built in-process (Lucene-style IDF, so tiny corpora still score sensibly) and cached until nodes change.
3. **History filter**: by default, only `status=active` graph nodes are eligible
4. **Feature scoring**: semantic similarity + lexical score + strength + confidence, minus a contradiction penalty. Strength is recomputed only for the candidates being ranked.
5. Return top-k (default 10)
6. **Cold-start fallback**: for the first ~10 sessions, also runs BM25 over raw logs. Those hits are scaled into `[0, raw_log_score_scale]` so they never outrank a real graph match.

Set `include_history=True` to surface superseded/retracted memories.

Use `with_trace=True` to return per-result ranking feature traces.

### Built-in benchmark
Run the built-in replay benchmark to measure:
- `Recall@k`
- `nDCG@k`
- contradiction error rate
- p50/p95 query latency
- deletion audit pass/fail (graph/log/snapshot/query checks)

```bash
awhm eval
awhm eval --json
```

## Configuration

All parameters are configurable via `AWHMConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.3 | Decay rate (power-law exponent) |
| `beta` | 0.1 | Decay scaling constant |
| `w_rec` | 0.4 | Recency weight in strength score |
| `w_freq` | 0.6 | Frequency weight in strength score |
| `retrieval_profile` | `"balanced"` | Retrieval weighting profile |
| `w_semantic` | 0.55 | Semantic similarity weight |
| `w_lexical` | 0.20 | BM25 lexical weight |
| `w_strength` | 0.15 | Node strength weight |
| `w_confidence` | 0.10 | Consolidation confidence weight |
| `contradiction_penalty` | 0.35 | Penalty for non-active memories |
| `include_history_by_default` | `False` | Include superseded/retracted memories by default |
| `trace_retrieval` | `False` | Emit ranking traces by default |
| `k` | 10 | Top-k retrieval count |
| `entity_link_threshold` | 0.85 | Cosine threshold for entity linking |
| `dedup_threshold` | 0.92 | Cosine threshold for deduplication |
| `bm25_threshold` | 1.0 | Minimum BM25 score for anchor set |
| `embed_threshold` | 0.3 | Minimum cosine sim for anchor set |
| `raw_log_score_scale` | 0.5 | Upper bound for cold-start raw-log hit scores |
| `correction_window_messages` | 3 | How close an explicit correction must be to supersede a preference/policy |
| `ner_labels` | PERSON, ORG, GPE, ... | spaCy entity labels that become nodes |
| `buffer_flush_interval` | 30s | WAL persistence interval |
| `stage2_enabled` | `False` | Reserved flag for optional Stage-2 refinement |
| `ann_index_type` | `"none"` | Reserved ANN index mode |
| `delete_snapshots_on_hard_delete` | `True` | Scrub matching snapshot memory on hard delete |

```python
from awhm.config import AWHMConfig

config = AWHMConfig(
    data_dir="~/.my-project-memory",
    k=20,
    w_rec=0.5,
    w_freq=0.5,
)
```

## Data directory

```
~/.awhm/
├── logs/                          # Raw JSONL logs (one per session)
│   ├── {session_id}.jsonl
│   └── ...
├── graph/
│   └── memory_graph.json          # The memory graph
├── snapshots/
│   └── snapshot_{timestamp}.json  # Manual backups
├── wal/
│   └── {session_id}.wal           # Per-session write-ahead logs
└── meta/
    ├── consolidated_sessions.json # Tracks which sessions have been processed
    ├── deletion_tombstones.jsonl  # Deletion tombstones
    └── deletion_ledger.jsonl      # Deletion audit ledger
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All tests use `MockEmbeddingService` (deterministic across processes, no model download). `ruff check .` runs the linter; CI runs both on Python 3.11, 3.12 and 3.13.

## Dependencies

| Package | Size | Purpose |
|---------|------|---------|
| numpy | ~29 MB | Vector math |
| spacy + en_core_web_sm | ~35 MB | NER |
| dateparser | ~2 MB | Date parsing |
| sentence-transformers (optional, `[embeddings]`) | ~3 MB (+PyTorch ~350 MB) | Embedding model |
| mcp (optional, `[mcp]`) | ~1 MB | Claude Code integration |

BM25 is implemented in-package (about 60 lines), so there is no ranking dependency.

The embedding model (`all-MiniLM-L6-v2`, 22 MB) downloads on first use to `~/.cache/huggingface`.

## Project structure

```
src/awhm/
├── __init__.py            # AWHMSession facade (top-level API)
├── config.py              # All parameters + path helpers
├── types.py               # Enums: Role, NodeType, NodeStatus, EdgeType, BufferEntryType
├── mcp_server.py          # MCP server for Claude Code
├── eval/                  # Built-in benchmark harness
├── raw_log/               # Append-only JSONL logging
├── session_buffer/        # Regex pattern matching + WAL
├── graph/                 # Memory graph, strength scoring, serialization
├── consolidation/         # NER, temporal, extraction, linking, dedup, pipeline
├── retrieval/             # Embedding, BM25, ranking, retrieval engine
├── snapshots/             # Snapshot create/restore/list
├── deletion/              # Hard-delete cascade
└── cli/                   # argparse CLI
```
