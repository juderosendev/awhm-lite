# Benchmarks

## LongMemEval (oracle split), Stage 1 only

Run on 2026-08-25 with `awhm eval --corpus longmemeval_oracle.json --longmemeval -k 5`,
real embeddings (`all-MiniLM-L6-v2`), spaCy `en_core_web_sm`, **no Stage 2**.
The oracle split contains only the evidence sessions for each question, so
this measures extraction and ranking, not needle-in-haystack retrieval.

| | |
|---|---|
| Questions | 500 |
| Sessions replayed | 948 (10,960 messages) |
| Nodes created | 28,688 |
| Ingest time | 753 s (one process, CPU) |
| **Recall@5** | **0.196** |
| nDCG@5 | 0.148 |
| Query latency p50 / p95 | 4.4 ms / 5.8 ms |
| Questions with no result at all | 24 |

| Category | Recall@5 | n |
|---|---|---|
| single-session-user | 0.400 | 70 |
| knowledge-update | 0.295 | 78 |
| multi-session | 0.165 | 133 |
| temporal-reasoning | 0.135 | 133 |
| single-session-assistant | 0.125 | 56 |
| single-session-preference | 0.000 | 30 |

## Reading the numbers

- **This is the Stage 1 ceiling, measured.** Extraction is regex plus NER.
  It stores whatever matches "my X is Y", "I prefer X", "actually, X" and
  named entities, and stores it as the raw message fragment, not as an atomic
  statement. 2.6 nodes per message says the patterns fire often and coarsely.
  The preference category scoring zero is the clearest signal: LongMemEval's
  preference questions ask for recommendations consistent with what the user
  said, which no substring of a raw fragment will answer.
- **Substring matching cuts both ways.** Paraphrased answers are not counted
  (a lower bound), but short answers ("bike", "car") can match a fragment that
  merely mentions the word (an upper bound). Treat the headline as
  approximate; the per-category ordering is the reliable part.
- **Temporal reasoning is handicapped by the replay, not only by the model.**
  Sessions were logged at ingestion time; the benchmark's session dates were
  not replayed, so date arithmetic questions cannot be answered from validity
  windows yet.
- **Latency is the good news:** 4 ms per query over ~60 nodes per instance,
  zero LLM calls.

## What should move these numbers

1. Stage 2 (`--stage2`): an LLM proposes atomic, third-person memories with
   slots and validity dates. This is the intended fix for the extraction
   ceiling and should lift every category, preference most of all.
2. Replaying session dates as log timestamps, so `as_of` queries and validity
   windows apply to temporal questions.
3. Answer matching by an LLM judge (the benchmark's own protocol) instead of
   substrings, once Stage 2 produces statements rather than fragments.

## Reproduce

```bash
pip install -e ".[embeddings]" && python -m spacy download en_core_web_sm
curl -L -o longmemeval_oracle.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
awhm eval --corpus longmemeval_oracle.json --longmemeval -k 5          # Stage 1
awhm eval --corpus longmemeval_oracle.json --longmemeval -k 5 --limit 100 --json > stage1.json
```

Stage 2 on a subset (uses the Claude Code CLI login, no API key):

```python
from awhm.config import AWHMConfig
from awhm.eval import load_longmemeval, run_replay, summarize

corpus = load_longmemeval("longmemeval_oracle.json", limit=100)
config = AWHMConfig(stage2_enabled=True, stage2_model="sonnet")
print(summarize(run_replay(corpus, k=5, config=config)))
```
