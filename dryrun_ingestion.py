#!/usr/bin/env python3
"""
DRY-RUN INGESTION  -  small-slice sanity check before full re-ingestion.

Takes the first N chunks from chunks_export.json (default N=5) and the
matching N entries from extraction_results.json, runs the FULL Phase-3
pipeline (entity insertion + REBEL relations + co-occurrence + cleanup +
baseline + invariants) into a disposable KuzuDB at ./data/_dryrun/graph,
then dumps every node and every edge for visual inspection.

This exercises the same code paths as `local_importingestion.py` but on a
slice small enough to read end-to-end. If the dry-run graph looks correct,
the full ingestion will produce the same shape, just bigger.

Usage:
    python -X utf8 dryrun_ingestion.py
    python -X utf8 dryrun_ingestion.py --dataset hotpotqa --n-chunks 5
    python -X utf8 dryrun_ingestion.py --n-chunks 10 --keep   # don't tear down
    python -X utf8 dryrun_ingestion.py --no-cleanup           # skip cleanup pass
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_slice(
    chunks_path: Path,
    extractions_path: Path,
    n_chunks: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load the first n chunks and their matching extraction records."""
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)
    with open(extractions_path, encoding="utf-8") as f:
        ext_data = json.load(f)
    all_results = ext_data.get("results", [])

    # Match by chunk_id (the Phase-1 metadata.chunk_id and Phase-2 chunk_id
    # are both string-coerced integer counters, so we can join on them).
    chunks = all_chunks[:n_chunks]
    wanted_ids = {str(c.get("metadata", {}).get("chunk_id")) for c in chunks}
    results = [r for r in all_results if str(r.get("chunk_id")) in wanted_ids]

    if len(results) != len(chunks):
        logger.warning(
            "Only %d of %d chunks have a matching extraction record",
            len(results), len(chunks),
        )
    return chunks, results


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY-PRINTERS
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def print_input_slice(chunks: List[Dict], results: List[Dict]) -> None:
    print()
    print("=" * 78)
    print("  INPUT  -  what Phase 2 produced for this slice")
    print("=" * 78)
    by_id = {str(r["chunk_id"]): r for r in results}

    for c in chunks:
        cid = str(c.get("metadata", {}).get("chunk_id"))
        title = c.get("metadata", {}).get("article_title", "?")
        text = _truncate(c.get("text", ""), 90)
        result = by_id.get(cid, {"entities": [], "relations": []})
        ents = result.get("entities", [])
        rels = result.get("relations", [])

        print()
        print(f"  Chunk[{cid}]  source='{title}'")
        print(f"    text:  {text}")
        print(f"    entities ({len(ents)}):")
        for e in ents[:12]:
            print(
                f"      - {e.get('name'):<35}  "
                f"type={e.get('type', e.get('entity_type', '?')):<14} "
                f"conf={e.get('confidence', 0):.2f}"
            )
        if len(ents) > 12:
            print(f"      ... +{len(ents) - 12} more")
        print(f"    relations ({len(rels)}):")
        for r in rels[:10]:
            subj = r.get("subject_entity") or r.get("subject")
            obj = r.get("object_entity") or r.get("object")
            rtype = r.get("relation_type") or r.get("relation")
            print(f"      - {subj}  -[{rtype}]->  {obj}")
        if len(rels) > 10:
            print(f"      ... +{len(rels) - 10} more")


def print_graph_state(graph_store) -> None:
    """Dump every node and every edge in the dry-run graph."""

    def fetch(query: str, params: Optional[Dict] = None) -> List[Tuple]:
        try:
            r = graph_store.conn.execute(query, params or {})
        except Exception as exc:
            logger.warning("query failed (%s): %s", query.split()[0], exc)
            return []
        out = []
        while r.has_next():
            out.append(tuple(r.get_next()))
        return out

    print()
    print("=" * 78)
    print("  FINAL GRAPH STATE")
    print("=" * 78)

    # ---- DocumentChunk nodes -------------------------------------------
    chunks = fetch(
        "MATCH (c:DocumentChunk) "
        "RETURN c.chunk_id, c.source_file, c.chunk_index, c.text "
        "ORDER BY c.chunk_id"
    )
    print()
    print(f"  DocumentChunk nodes: {len(chunks)}")
    for cid, src, idx, txt in chunks:
        print(
            f"    [{cid:>4}] idx={idx:<3} src='{_truncate(src, 32)}'  "
            f"text='{_truncate(txt or '', 50)}'"
        )

    # ---- Entity nodes (with mention count) -----------------------------
    entities = fetch(
        """
        MATCH (e:Entity)
        OPTIONAL MATCH (c:DocumentChunk)-[:MENTIONS]->(e)
        WITH e.entity_id AS eid, e.name AS name, e.type AS etype,
             e.confidence AS conf, count(c) AS mc
        RETURN eid, name, etype, conf, mc
        ORDER BY mc DESC, name
        """
    )
    print()
    print(f"  Entity nodes: {len(entities)}")
    for eid, name, etype, conf, mc in entities:
        print(
            f"    {etype or '?':<14}  {name:<40}  "
            f"mentions={int(mc or 0):<3} conf={float(conf or 0):.2f}  "
            f"[{eid[:12]}…]"
        )

    # ---- MENTIONS edges -------------------------------------------------
    mentions = fetch(
        """
        MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
        RETURN c.chunk_id, e.name, e.type
        ORDER BY c.chunk_id, e.name
        """
    )
    print()
    print(f"  MENTIONS edges: {len(mentions)}")
    by_chunk: Dict[str, List[Tuple[str, str]]] = {}
    for cid, name, etype in mentions:
        by_chunk.setdefault(str(cid), []).append((name, etype or "?"))
    for cid in sorted(by_chunk.keys()):
        names = ", ".join(f"{n}({t})" for n, t in by_chunk[cid][:8])
        more = "" if len(by_chunk[cid]) <= 8 else f" +{len(by_chunk[cid]) - 8}"
        print(f"    Chunk[{cid}]  ->  {names}{more}")

    # ---- RELATED_TO edges (with confidence + source_chunks) -------------
    relations = fetch(
        """
        MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
        RETURN a.name, b.name, r.relation_type, r.confidence, r.source_chunks
        ORDER BY r.relation_type, a.name, b.name
        """
    )
    print()
    print(f"  RELATED_TO edges: {len(relations)}")
    rebel_count = sum(1 for *_, t, _, _ in relations if t != "cooccurs")
    cooc_count = sum(1 for *_, t, _, _ in relations if t == "cooccurs")
    print(f"    breakdown:  REBEL = {rebel_count}   |   cooccurs = {cooc_count}")

    # Show all REBEL relations (usually few enough), and a sample of cooccurs
    rebel_edges = [r for r in relations if r[2] != "cooccurs"]
    cooc_edges = [r for r in relations if r[2] == "cooccurs"]

    if rebel_edges:
        print()
        print("    REBEL relations (all):")
        for a, b, rt, conf, src in rebel_edges:
            src_str = src or ""
            print(
                f"      {a:<28} -[{rt:<14} conf={float(conf or 0):.2f}, "
                f"src={src_str}]-> {b}"
            )
    if cooc_edges:
        print()
        print(f"    Co-occurrence edges (showing 15 of {len(cooc_edges)}):")
        for a, b, rt, conf, src in cooc_edges[:15]:
            src_str = src or ""
            print(
                f"      {a:<28} -[{rt} conf={float(conf or 0):.2f}, "
                f"src={src_str}]-> {b}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run Phase-3 ingestion on a small slice of the corpus."
    )
    parser.add_argument(
        "--dataset",
        default="hotpotqa",
        help="Dataset name; resolves to ./data/<dataset>/",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=5,
        help="Number of chunks to ingest (default: 5)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip the cleanup pass (orphans, stop-list, hubs, duplicates)",
    )
    parser.add_argument(
        "--no-cooccurrence",
        action="store_true",
        help="Skip co-occurrence edge construction",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the dry-run graph at ./data/_dryrun/graph/ for further inspection",
    )
    parser.add_argument(
        "--cooccurrence-min-confidence",
        type=float,
        default=0.5,
        help="Minimum NER confidence for co-occurrence (default: 0.5)",
    )
    parser.add_argument(
        "--entity-confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum NER confidence for entity insertion (default: 0.5)",
    )
    args = parser.parse_args()

    base = Path("./data") / args.dataset
    chunks_path = base / "chunks_export.json"
    extractions_path = base / "graph" / "extraction_results.json"
    if not chunks_path.exists():
        print(f"error: chunks file not found: {chunks_path}", file=sys.stderr)
        return 1
    if not extractions_path.exists():
        print(f"error: extractions file not found: {extractions_path}", file=sys.stderr)
        return 1

    # Disposable graph location. Use a unique per-run subdirectory so a
    # leftover KuzuDB file lock from a previous crashed run does not block
    # us. The whole `./data/_dryrun/` tree is best-effort cleaned up on
    # exit.
    import time as _time
    dryrun_root = Path("./data/_dryrun")
    dryrun_graph = dryrun_root / f"run_{int(_time.time())}" / "graph"
    if dryrun_root.exists():
        try:
            shutil.rmtree(dryrun_root)
        except PermissionError:
            logger.warning(
                "Could not remove %s (stale Kuzu lock from earlier run). "
                "Continuing into a new sub-directory.", dryrun_root,
            )
    dryrun_graph.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"  DRY-RUN INGESTION  -  {args.n_chunks} chunks from {args.dataset}")
    print("=" * 78)
    print(f"  Source chunks:        {chunks_path}")
    print(f"  Source extractions:   {extractions_path}")
    print(f"  Dry-run graph at:     {dryrun_graph}")
    print(f"  Cleanup:              {'OFF' if args.no_cleanup else 'ON'}")
    print(f"  Co-occurrence:        {'OFF' if args.no_cooccurrence else 'ON'}")

    chunks, results = load_slice(chunks_path, extractions_path, args.n_chunks)
    print_input_slice(chunks, results)

    # Bring in the production code paths so the dry-run exercises EXACTLY
    # the same logic as the full ingestion.
    try:
        from src.data_layer import KuzuGraphStore
        from src.data_layer.graph_quality import (
            assert_graph_invariants,
            build_cooccurrence_edges,
            cleanup_graph,
            compute_graph_baseline,
            format_baseline_report,
        )
    except ImportError as exc:
        print(f"error: import failed: {exc}", file=sys.stderr)
        return 1

    # We import the Phase-3 ingestion function and the canonical-id helper
    # so the dry-run produces the same graph shape as the real pipeline.
    sys.path.insert(0, str(Path(".").resolve()))
    from local_importingestion import (  # noqa: E402
        ingest_knowledge_graph,
        chunks_to_documents,
    )

    # Build LangChain Documents (matches what the full ingestion does).
    documents = chunks_to_documents(chunks)

    print()
    print("─" * 78)
    print("  PHASE 3b: ENTITY + REBEL RELATION INSERTION")
    print("─" * 78)
    graph_store, stats = ingest_knowledge_graph(
        documents=documents,
        extraction_results=results,
        graph_path=dryrun_graph,
        dataset_name=f"_dryrun_{args.dataset}",
        entity_confidence_threshold=args.entity_confidence_threshold,
    )
    if graph_store is None:
        print("error: ingest_knowledge_graph returned no graph_store", file=sys.stderr)
        return 1

    name_to_id = stats.pop("_name_to_id", {})
    print(f"  unique_entities:  {stats.get('unique_entities', 0)}")
    print(f"  total_mentions:   {stats.get('mentions', 0)}")
    print(f"  REBEL relations:  {stats.get('relations', 0)}")

    # Phase 3c: co-occurrence
    cooc_edges = 0
    if not args.no_cooccurrence:
        print()
        print("─" * 78)
        print("  PHASE 3c: CO-OCCURRENCE EDGES")
        print("─" * 78)
        cooc_edges = build_cooccurrence_edges(
            graph_store=graph_store,
            extraction_results=results,
            name_to_id=name_to_id,
            min_confidence=args.cooccurrence_min_confidence,
            relation_type="cooccurs",
        )
        print(f"  cooccurrence_edges: {cooc_edges}")

    # Phase 3d: cleanup
    cleanup_ops = {
        "stoplist_dropped": 0,
        "orphans_dropped": 0,
        "hubs_dropped": 0,
        "duplicates_merged": 0,
    }
    if not args.no_cleanup:
        print()
        print("─" * 78)
        print("  PHASE 3d: CLEANUP")
        print("─" * 78)
        # On 5 chunks, hub_threshold_min=50 dominates → no hubs dropped.
        # That's intentional for the dry-run: hub filtering needs a corpus.
        cleanup_ops = cleanup_graph(
            graph_store=graph_store,
            drop_orphans=True,
            hub_threshold_ratio=0.03,
            merge_duplicates=True,
        )
        for k, v in cleanup_ops.items():
            print(f"  {k:<22}: {v}")

    # Phase 3e: baseline + invariants
    print()
    print("─" * 78)
    print("  PHASE 3e: BASELINE + INVARIANTS")
    print("─" * 78)
    metrics = compute_graph_baseline(graph_store)
    print(format_baseline_report(metrics))

    violations = assert_graph_invariants(metrics, strict=False)
    if violations:
        print()
        print("  Invariant violations (expected on a 5-chunk slice — relation density")
        print("  threshold is calibrated for full corpus, not for tiny samples):")
        for v in violations:
            print(f"    - {v}")

    # Full graph dump
    print_graph_state(graph_store)

    # Summary line
    totals = metrics["totals"]
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(
        f"  chunks={totals['chunks']}  "
        f"entities={totals['entities']}  "
        f"mentions={totals['mentions']}  "
        f"relations={totals['relations']}  "
        f"cooccurrence={cooc_edges}"
    )
    print(
        f"  isolated_rate={metrics['isolated']['rate']:.1%}  "
        f"duplicate_rate={metrics['duplicates']['rate']:.1%}  "
        f"relations_per_chunk={metrics['densities']['relations_per_chunk']:.2f}"
    )

    # Tear down
    try:
        graph_store.close()
    except Exception:
        pass
    if not args.keep and dryrun_root.exists():
        try:
            shutil.rmtree(dryrun_root)
            print()
            print(f"  Dry-run graph removed.  Use --keep to inspect at {dryrun_graph}.")
        except PermissionError:
            print()
            print(f"  Dry-run graph at {dryrun_graph} could not be auto-removed "
                  "(KuzuDB file handle still held); will be cleaned on next run.")
    else:
        print()
        print(f"  Dry-run graph kept at: {dryrun_graph}")
        print("  Inspect via: python -X utf8 diagnose_graph_baseline.py "
              f"--graph-path {dryrun_graph}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
