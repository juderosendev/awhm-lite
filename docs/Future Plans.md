# Future Plans

## Done (v0.3.0, August 2026)

The original plan below (silent per-turn middleware) shipped as Claude Code
hooks: `awhm hook prompt` / `stop` / `session-end`. Also shipped: Stage 2 LLM
refinement, entity resolution with aliases, time-travel queries, neighbour
expansion, SQLite storage and real-corpus evaluation. See CHANGELOG.md.

## Next

- Run LongMemEval end to end with real embeddings and Stage 2, and publish
  the numbers (recall by category, contradiction rate).
- Salience at formation: corrections and explicit rules should start with
  higher strength instead of earning it through access counts.
- Spreading activation beyond one hop, with the whitepaper's lateral
  inhibition, once the graph is dense enough to need it.
- Permissions: per-source visibility so a memory derived from one context
  is not surfaced in another.

## Original plan (February 2026)

Goal: make memory work automatically in the background for Claude Code and
Codex, without visible tool-call noise.

Per-turn flow: log the user message verbatim, query memory for the top
memories, inject them as hidden context, let the model respond, log the
response verbatim, consolidate in the background.

Design rules: verbatim logging is mandatory; atomic memories are derived
artifacts, not replacements for raw logs; preserve qualifiers (time,
negation, uncertainty, scope) during extraction; hide superseded memories by
default when injecting.
