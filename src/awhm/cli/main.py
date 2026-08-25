"""CLI for AWHM Lite: status, query, consolidate, snapshot, delete, eval."""

from __future__ import annotations

import argparse
import json
import sys

from ..config import AWHMConfig


def get_session(args):
    """Create an AWHMSession from CLI args."""
    from .. import AWHMSession

    config = AWHMConfig(data_dir=args.data_dir)
    llm_client = None
    if getattr(args, "stage2", False):
        from ..consolidation.stage2 import AnthropicClient

        config.stage2_enabled = True
        llm_client = AnthropicClient(model=args.stage2_model or config.stage2_model)
    return AWHMSession.start_session(config, use_mock_embeddings=args.mock, llm_client=llm_client)


def cmd_status(args):
    session = get_session(args)
    try:
        info = session.status()
        print(json.dumps(info, indent=2))
    finally:
        session.end_session()


def cmd_query(args):
    session = get_session(args)
    try:
        results = session.query(
            args.query_text,
            k=args.k,
            include_history=args.include_history,
            with_trace=args.trace,
            as_of=args.as_of,
        )
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r.score:.4f}, source: {r.source}) ---")
            print(r.content[:200])
            if args.trace and r.trace:
                print("trace:", json.dumps(r.trace, sort_keys=True))
        if not results:
            print("No results found.")
    finally:
        session.end_session()


def cmd_consolidate(args):
    session = get_session(args)
    try:
        results = session.consolidate()
        total = sum(results.values())
        print(f"Consolidated {len(results)} session(s), {total} new node(s):")
        for sid, count in results.items():
            print(f"  {sid}: {count} nodes")
        if not results:
            print("No pending sessions to consolidate.")
    finally:
        session.end_session()


def cmd_snapshot(args):
    session = get_session(args)
    try:
        if args.action == "create":
            path = session.create_snapshot()
            print(f"Snapshot created: {path}")
        elif args.action == "list":
            snapshots = session.snapshots.list_snapshots()
            for s in snapshots:
                print(s)
            if not snapshots:
                print("No snapshots found.")
        elif args.action == "restore":
            if not args.path:
                print("Error: --path required for restore", file=sys.stderr)
                sys.exit(1)
            session.restore_snapshot(args.path)
            print(f"Restored from: {args.path}")
    finally:
        session.end_session()


def cmd_delete(args):
    session = get_session(args)
    try:
        result = session.delete_node(args.node_id)
        if result.node_deleted:
            print(f"Deleted node {args.node_id}")
            print(f"  Edges removed: {result.edges_removed}")
            print(f"  Log entries removed: {result.log_entries_removed}")
            print(f"  Buffer entries removed: {result.buffer_entries_removed}")
            print(f"  Match strategy: {result.match_strategy}")
            print(f"  Snapshots touched: {result.snapshots_touched}")
            if result.tombstone_id:
                print(f"  Tombstone ID: {result.tombstone_id}")
            if result.ledger_record_id:
                print(f"  Ledger record ID: {result.ledger_record_id}")
            if result.affected_sessions:
                print("  Affected sessions:")
                for sid in result.affected_sessions:
                    print(f"    - {sid}")
        else:
            print(f"Node {args.node_id} not found.")
    finally:
        session.end_session()


def cmd_eval(args):
    from ..eval import load_corpus, load_longmemeval, run_builtin_benchmark, run_replay, summarize

    if args.corpus:
        corpus = (
            load_longmemeval(args.corpus, limit=args.limit)
            if args.longmemeval
            else load_corpus(args.corpus)
        )
        report = run_replay(corpus, k=args.k, use_mock_embeddings=args.mock)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(summarize(report))
        return

    report = run_builtin_benchmark(
        k=args.k,
        use_mock_embeddings=args.mock,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("AWHM benchmark report")
    print(f"  Recall@{report['k']}: {report['recall_at_k']:.3f}")
    print(f"  nDCG@{report['k']}: {report['ndcg_at_k']:.3f}")
    print(f"  Contradiction error rate: {report['contradiction_error_rate']:.3f}")
    print(f"  Latency p50 (ms): {report['latency_ms']['p50']:.2f}")
    print(f"  Latency p95 (ms): {report['latency_ms']['p95']:.2f}")
    print(f"  Deletion audit passed: {report['deletion_audit']['passed']}")


def main():
    parser = argparse.ArgumentParser(
        prog="awhm",
        description="AWHM Lite: external memory for LLM agents",
    )
    parser.add_argument(
        "--data-dir", default="~/.awhm",
        help="Data directory (default: ~/.awhm)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock embeddings (for testing)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show system status")

    # query
    p_query = sub.add_parser("query", help="Query the memory system")
    p_query.add_argument("query_text", help="Query string")
    p_query.add_argument("-k", type=int, default=10, help="Top-k results")
    p_query.add_argument(
        "--include-history",
        action="store_true",
        help="Include superseded/retracted memories in results",
    )
    p_query.add_argument(
        "--trace",
        action="store_true",
        help="Show per-result ranking feature traces",
    )
    p_query.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        help="Answer as of this ISO-8601 moment (e.g. 2026-03-01)",
    )

    # consolidate
    p_cons = sub.add_parser("consolidate", help="Consolidate pending sessions into the graph")
    p_cons.add_argument(
        "--stage2",
        action="store_true",
        help="Also run Stage 2 LLM refinement (needs the [stage2] extra and Anthropic credentials)",
    )
    p_cons.add_argument("--stage2-model", dest="stage2_model", default=None, help="Model for Stage 2")

    # snapshot
    p_snap = sub.add_parser("snapshot", help="Manage snapshots")
    p_snap.add_argument("action", choices=["create", "list", "restore"])
    p_snap.add_argument("--path", help="Snapshot path (for restore)")

    # delete
    p_del = sub.add_parser("delete", help="Hard-delete a node")
    p_del.add_argument("node_id", help="Node ID to delete")

    # eval
    p_eval = sub.add_parser("eval", help="Run built-in memory quality benchmark")
    p_eval.add_argument("-k", type=int, default=5, help="Top-k for eval metrics")
    p_eval.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report instead of summary",
    )
    p_eval.add_argument("--corpus", default=None, help="Replay a corpus file instead of the built-in benchmark")
    p_eval.add_argument("--longmemeval", action="store_true", help="Corpus is in LongMemEval format")
    p_eval.add_argument("--limit", type=int, default=None, help="Only the first N LongMemEval instances")

    # hook: delegate everything after "hook" to the hooks module
    sub.add_parser("hook", help="Claude Code hook commands (see `awhm hook --help`)", add_help=False)

    if len(sys.argv) > 1 and sys.argv[1] == "hook":
        from ..hooks import run as run_hook

        sys.exit(run_hook(sys.argv[2:]))

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "query": cmd_query,
        "consolidate": cmd_consolidate,
        "snapshot": cmd_snapshot,
        "delete": cmd_delete,
        "eval": cmd_eval,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
