# AWHM Lite

External long-term memory for LLMs. No cloud, no API keys, runs entirely local.

AWHM Lite gives any LLM persistent memory across conversations through append-only logging, regex-based pattern matching, a contradiction-aware memory graph, symbolic consolidation (zero LLM calls), and retrieval via lexical + semantic feature fusion.

**Status:** research prototype. Built February 2026, published August 2026. Working and tested (88 tests), not under active development.

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
pip install -e .

# For spaCy NER (used in consolidation):
python -m spacy download en_core_web_sm

# For Claude Code MCP integration:
pip install -e ".[mcp]"
```

The embedding model (`all-MiniLM-L6-v2`, 22 MB) downloads automatically on first use.

## Quick start

### Python API

```python
from awhm import AWHMSession
from awhm.types import Role

# Start a session
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

Captures ~60-70% of explicit signals with zero LLM calls. The buffer is checked first during retrieval for instant intra-session continuity. During default retrieval, contradictory buffer entries are collapsed by canonical key so newer corrections win. Persisted via per-session write-ahead logs (30s flush interval).

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
1. **NER** via spaCy — people, orgs, tools, URLs
2. **Temporal parsing** via dateparser — resolve "yesterday", "March 5" to ISO timestamps
3. **Rule-based extraction** — same regex patterns as session buffer, over the full log
4. **Entity linking** — match to existing nodes (string sim + cosine > 0.85)
5. **Deduplication** — cosine > 0.92 near-duplicate detection
6. **Commit** — assign canonical keys, supersede stale active memories, add new nodes/edges

### Retrieval
Zero LLM calls. Feature-based fusion:

1. **Buffer check** — search session buffer first (instant hits)
2. **Anchor identification** — BM25 term overlap + embedding cosine similarity (union)
3. **History filter** — by default, only `status=active` graph nodes are eligible
4. **Feature scoring** — semantic similarity + lexical score + strength + confidence − contradiction penalty
5. Return top-k (default 10)
6. **Cold-start fallback** — for the first ~10 sessions, also runs BM25 over raw logs

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

All tests use `MockEmbeddingService` (deterministic, no model download).

## Dependencies

| Package | Size | Purpose |
|---------|------|---------|
| sentence-transformers | ~3 MB (+PyTorch ~350 MB) | Embedding model |
| numpy | ~29 MB | Vector math |
| spacy + en_core_web_sm | ~35 MB | NER |
| rank-bm25 | <1 MB | BM25 scoring |
| dateparser | ~2 MB | Date parsing |
| mcp (optional) | ~1 MB | Claude Code integration |

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
