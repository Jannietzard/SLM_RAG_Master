#!/usr/bin/env python3
"""
Standalone graph-quality baseline diagnostic.

Opens an existing KuzuDB graph store, computes the baseline metrics defined
in `src.data_layer.graph_quality`, prints a human-readable report, and
optionally enforces the default invariants. No mutation of the graph.

Usage:
    python -X utf8 diagnose_graph_baseline.py --dataset hotpotqa
    python -X utf8 diagnose_graph_baseline.py --graph-path ./data/hotpotqa/graph
    python -X utf8 diagnose_graph_baseline.py --dataset hotpotqa --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve_graph_path(args: argparse.Namespace) -> Path:
    if args.graph_path:
        return Path(args.graph_path)
    if args.dataset:
        return Path(f"./data/{args.dataset}/graph")
    raise SystemExit("error: either --dataset or --graph-path is required")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute and print the graph-quality baseline for a KuzuDB store.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name; resolves to ./data/<dataset>/graph",
    )
    parser.add_argument(
        "--graph-path",
        default=None,
        help="Explicit path to a KuzuDB directory (overrides --dataset)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of the human-readable report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any invariant is violated",
    )
    args = parser.parse_args()

    graph_path = _resolve_graph_path(args)
    if not graph_path.exists():
        print(f"error: graph not found: {graph_path}", file=sys.stderr)
        return 1

    try:
        from src.data_layer.storage import KuzuGraphStore
        from src.data_layer.graph_quality import (
            assert_graph_invariants,
            compute_graph_baseline,
            format_baseline_report,
        )
    except ImportError as exc:
        print(f"error: import failed: {exc}", file=sys.stderr)
        return 1

    store = KuzuGraphStore(str(graph_path))
    metrics = compute_graph_baseline(store)

    if args.json:
        print(json.dumps(metrics, indent=2, default=str))
    else:
        print()
        print("=" * 70)
        print(f"  GRAPH QUALITY BASELINE  -  {graph_path}")
        print("=" * 70)
        print(format_baseline_report(metrics))

    violations = assert_graph_invariants(metrics, strict=False)
    if violations:
        print()
        print("  Invariant violations:")
        for v in violations:
            print(f"    - {v}")
        if args.strict:
            return 2
    else:
        print()
        print("  All invariants OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
