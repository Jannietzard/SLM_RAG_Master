"""
Linking-threshold probe — dry-run analysis of Phase 3d.5 thresholds.

Why this exists
---------------
Phase 3d.5 (embedding-based entity linking) at threshold 0.92 with
nomic-embed-text merged 87 % of PERSON, 77 % of ORG, 67 % of LOCATION
entities on a previous ingest — almost certainly far too aggressive,
because nomic-embed-text compresses scores into a narrow high band.

This script answers the question "what threshold should I pick?"
**without** running a full re-ingest. For each candidate threshold it
reports, per type bucket:

  - how many entities would be merged
  - how many clusters would form
  - the merge rate (merged / total)
  - the largest cluster's size (an early-warning sign of over-merging)
  - up to 10 example clusters (size 2+ only)

The script is **read-only on the graph**. It loads entity names + types
via Cypher, embeds them once, computes pairwise cosine similarity in
memory per type bucket, and runs the same union-find logic
`link_entities_by_embedding` uses — but only counts; it does not write.

Usage
-----
    python -X utf8 probe_linking_threshold.py --dataset hotpotqa `
        --thresholds 0.92,0.95,0.97,0.98,0.99 `
        --type-filter PERSON,ORG,LOCATION

If no --type-filter is given, all type buckets are probed.

Interpretation guide
--------------------
At the chosen threshold the merge rate per bucket should be a small
fraction (single-digit percent) unless your corpus genuinely contains
many name variants. Cluster sizes > 5 are usually a smell — real-world
aliases rarely come in groups of 6+. If even at threshold 0.99 the
merge rate is double-digit percent on PERSON, the embedding model is
simply not discriminative on person names and you should disable
embedding-based linking for that type and rely on `canonical_form`
exact-match deduplication only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = (0.92, 0.95, 0.97, 0.98, 0.99)
LENGTH_RATIO_FLOOR = 0.4   # matches link_entities_by_embedding


def _load_entities_by_type(
    graph_path: Path,
) -> Dict[str, List[Tuple[str, str, int]]]:
    """Return {type: [(entity_id, name, mention_count), ...]}.

    Mirrors the SELECT used inside `link_entities_by_embedding` so the
    probe sees exactly the same input.
    """
    from src.data_layer.storage import KuzuGraphStore

    store = KuzuGraphStore(graph_path)
    by_type: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    res = store.conn.execute(
        """
        MATCH (e:Entity)
        OPTIONAL MATCH (c:DocumentChunk)-[:MENTIONS]->(e)
        WITH e.entity_id AS eid, e.name AS name, e.type AS etype, count(c) AS mc
        RETURN eid, name, etype, mc
        """,
    )
    while res.has_next():
        eid, name, etype, mc = res.get_next()
        if not name:
            continue
        by_type[etype or "unknown"].append((eid, name, int(mc or 0)))
    store.close()
    return dict(by_type)


def _build_embedder(cfg: dict, dataset: str):
    """Construct the same BatchedOllamaEmbeddings instance the ingest uses.

    `cache_path` points at the SAME SQLite the ingest populated
    (`./cache/{dataset}_embeddings.db`) so the probe reuses the entity-name
    embeddings already paid for — no fresh Ollama calls, no chance of a
    cold-cache discrepancy that makes thresholds look harmless when in
    practice they merged 87 % of PERSON.
    """
    from src.data_layer.embeddings import BatchedOllamaEmbeddings
    emb_cfg  = cfg.get("embeddings", {})
    perf_cfg = cfg.get("performance", {})
    return BatchedOllamaEmbeddings(
        model_name=emb_cfg.get("model_name", "nomic-embed-text"),
        base_url=emb_cfg.get("base_url", "http://localhost:11434"),
        batch_size=perf_cfg.get("batch_size", 64),
        cache_path=Path(f"./cache/{dataset}_embeddings.db"),
    )


def _probe_one_threshold(
    embeds: np.ndarray,
    members: List[Tuple[str, str, int]],
    threshold: float,
) -> Tuple[int, int, int, List[List[str]]]:
    """Run the union-find merge logic for one threshold.

    Returns (n_merged, n_clusters_size_ge_2, largest_cluster_size,
    example_clusters_top10).

    `n_merged` = entities absorbed into another cluster (the count
    `link_entities_by_embedding` reports as 'merged' for that type).
    """
    n = len(members)
    if n < 2:
        return 0, 0, 0, []

    # Pairwise cosine similarity. Self-similarity ignored (diagonal=0).
    sim = embeds @ embeds.T
    np.fill_diagonal(sim, 0.0)

    # Union-find with mention-count as the "win" criterion (cluster root
    # is the entity with the most mentions — same rule as the real linker).
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi == pj:
            return
        if members[pi][2] >= members[pj][2]:
            parent[pj] = pi
        else:
            parent[pi] = pj

    for i in range(n):
        name_i = members[i][1]
        for j in range(i + 1, n):
            if sim[i, j] < threshold:
                continue
            name_j = members[j][1]
            short, long_ = sorted((len(name_i), len(name_j)))
            if long_ > 0 and short / long_ < LENGTH_RATIO_FLOOR:
                continue
            union(i, j)

    # Group by root.
    clusters: Dict[int, List[str]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(members[i][1])

    n_merged = sum(1 for i in range(n) if find(i) != i)
    multi = [c for c in clusters.values() if len(c) >= 2]
    largest = max((len(c) for c in multi), default=0)
    # Sort example clusters by size descending; show up to 10.
    examples = sorted(multi, key=lambda c: -len(c))[:10]
    return n_merged, len(multi), largest, examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="Comma-separated thresholds (e.g. '0.92,0.95,0.97,0.98,0.99').",
    )
    parser.add_argument(
        "--type-filter", default=None,
        help="Comma-separated type subset (e.g. 'PERSON,ORG'). "
             "Default: probe every type bucket.",
    )
    parser.add_argument(
        "--min-bucket-size", type=int, default=50,
        help="Skip buckets smaller than this (defaults to 50 — no point "
             "running pairwise sim on tiny buckets).",
    )
    parser.add_argument(
        "--max-bucket-size", type=int, default=20000,
        help="Skip buckets larger than this for memory safety.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=0,
        help="If > 0 and the bucket has more entities than this, take a "
             "uniform random sample of this size instead of skipping. "
             "Lets you peek at huge buckets (PERSON ~10k) in seconds "
             "without an O(n^2) explosion. Sample is mention-count-weighted "
             "if --weighted-sample is also passed.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for --sample-size (default 42, deterministic).",
    )
    args = parser.parse_args()

    # Parse thresholds.
    try:
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")
                      if t.strip()]
    except ValueError as exc:
        raise SystemExit(f"Bad --thresholds: {exc}")
    if not thresholds:
        raise SystemExit("Need at least one threshold.")
    type_filter = (
        {t.strip().upper() for t in args.type_filter.split(",")}
        if args.type_filter else None
    )

    # Load settings (for the embedder factory).
    import yaml
    cfg_path = _PROJECT_ROOT / "config" / "settings.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    graph_path = _PROJECT_ROOT / "data" / args.dataset / "graph"
    if not graph_path.exists():
        raise SystemExit(f"No graph at {graph_path}. Ingest first.")

    logger.info("Loading entities from %s …", graph_path)
    by_type = _load_entities_by_type(graph_path)
    logger.info("Loaded %d type buckets, %d entities total",
                len(by_type), sum(len(v) for v in by_type.values()))

    embedder = _build_embedder(cfg, args.dataset)

    # Single RNG for reproducible sampling across buckets.
    rng = np.random.default_rng(args.seed)

    # ── Per-bucket probe ────────────────────────────────────────────────
    for etype, members in sorted(by_type.items(),
                                 key=lambda kv: -len(kv[1])):
        if type_filter is not None and etype.upper() not in type_filter:
            continue
        if len(members) < args.min_bucket_size:
            continue
        if len(members) > args.max_bucket_size:
            # If --sample-size is set, fall back to a sample instead of
            # silently skipping. Lets you keep --max-bucket-size as a true
            # hard ceiling for memory while still peeking at huge buckets.
            if args.sample_size and args.sample_size < len(members):
                idx = rng.choice(
                    len(members), size=args.sample_size, replace=False,
                )
                sampled = [members[i] for i in idx]
                logger.info(
                    "Sampling %s: %d -> %d entities (--sample-size, seed=%d)",
                    etype, len(members), args.sample_size, args.seed,
                )
                members = sampled
            else:
                logger.info("Skipping %s (%d entities > --max-bucket-size %d)",
                            etype, len(members), args.max_bucket_size)
                continue
        elif args.sample_size and args.sample_size < len(members):
            # Bucket is within max-bucket-size but caller still wants a
            # sample (e.g. to keep all four type buckets at the same n for
            # a clean cross-type comparison).
            idx = rng.choice(
                len(members), size=args.sample_size, replace=False,
            )
            sampled = [members[i] for i in idx]
            logger.info(
                "Sampling %s: %d -> %d entities (--sample-size, seed=%d)",
                etype, len(members), args.sample_size, args.seed,
            )
            members = sampled

        print()
        print("=" * 74)
        print(f"  TYPE: {etype:<14}  (n={len(members)} entities)")
        print("=" * 74)

        names = [name for _, name, _ in members]
        try:
            raw = embedder.embed_documents(names)
        except Exception as exc:
            print(f"  Embedding failed: {exc}")
            continue
        embeds = np.asarray(raw, dtype=np.float32)
        # L2-normalise so dot = cosine.
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / np.where(norms > 0, norms, 1.0)

        print(f"  {'threshold':>10}  {'merged':>8}  {'rate':>7}  "
              f"{'clusters':>9}  {'largest':>8}")
        print("  " + "-" * 50)
        for t in thresholds:
            n_merged, n_clusters, largest, examples = _probe_one_threshold(
                embeds, members, t,
            )
            rate = n_merged / len(members) * 100.0 if members else 0.0
            print(f"  {t:>10.3f}  {n_merged:>8d}  {rate:>6.1f}%  "
                  f"{n_clusters:>9d}  {largest:>8d}")
            # Show example clusters at the lowest threshold only — at
            # higher thresholds the clusters shrink to subsets of these.
            if t == thresholds[0] and examples:
                print(f"\n  Largest clusters at threshold {t} "
                      f"(name → merge target):")
                for cluster in examples[:5]:
                    head = cluster[0]
                    rest = cluster[1:7]   # cap display
                    suffix = (f" +{len(cluster) - 7} more"
                              if len(cluster) > 7 else "")
                    print(f"    [{len(cluster):>3}x] {head!r}: "
                          + ", ".join(f"{n!r}" for n in rest)
                          + suffix)
                print()

    print()
    print("Interpretation:")
    print("  - 'rate' is the fraction of entities that would be absorbed.")
    print("    A healthy linker produces single-digit percent on PERSON / ORG.")
    print("  - 'largest' is the biggest cluster's size. > 5 is usually a")
    print("    smell — real aliases rarely come in groups of 6+.")
    print("  - If every threshold gives a high merge rate, the embedding")
    print("    model is not discriminative on that type; disable linking")
    print("    for that type via --skip-types in the ingest, or rely on")
    print("    canonical_form exact-match deduplication only.")


if __name__ == "__main__":
    main()
