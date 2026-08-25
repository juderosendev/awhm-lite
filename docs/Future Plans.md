# Future Plans

## Target Date
Sunday, February 22, 2026

## Goal
Make memory work automatically in the background for Claude Code and Codex, without visible tool-call noise.

## Why This Direction
- Better reliability: no dependence on the model remembering to call `memory_log`.
- Better accuracy: keep verbatim raw logs as ground truth.
- Better UX: memory happens silently behind the scenes.
- Better auditing: derived memories can be traced back to source text.

## Simple Per-Turn Flow
1. User sends a message.
2. Middleware logs the user message verbatim.
3. Middleware queries memory for the top 5 most relevant memories.
4. Middleware injects those memories as hidden context.
5. Model generates a response.
6. Middleware logs the assistant response verbatim.
7. Background consolidation extracts compact atomic memories from raw logs.

## Design Rules
- Verbatim logging is mandatory.
- Atomic memories are derived artifacts, not replacements for raw logs.
- Preserve qualifiers (time, negation, uncertainty, scope) during extraction.
- Use `include_history=false` by default when injecting memory context.

## Implementation Plan (Tomorrow)
1. Add a middleware wrapper that always executes `log -> query -> hidden inject -> respond -> log`.
2. Enforce top-5 memory injection with filtering (score threshold + dedupe).
3. Add periodic consolidation (every N turns) and final consolidation at session end.
4. Add tracing/debug mode to inspect what was injected each turn.
5. Add tests for:
   - verbatim logging always happening,
   - top-5 hidden injection behavior,
   - qualifier-preserving atomic extraction,
   - no paraphrase-only storage as source of truth.

## Open Decisions
- Where to host middleware (MCP server layer vs separate backend service).
- Score threshold and filtering rules for top-5 selection.
- Consolidation cadence (`every N turns` vs timed async worker).

