"""
════════════════════════════════════════════════════════════════════════
  INGESTION DIAGNOSTIC — chunks → LanceDB → KuzuDB → Retrieval Rank
════════════════════════════════════════════════════════════════════════

For each question index, this script traces the full path from the
source article through ingestion into the stores, then checks whether
the required chunks are actually retrieved and at what rank.

Usage:
    python -X utf8 diagnose_ingestion.py --indices 11,12
    python -X utf8 diagnose_ingestion.py --indices 0-19
    python -X utf8 diagnose_ingestion.py --indices 11,12 --dataset hotpotqa
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── ANSI colours (same helpers as diagnose_verbose.py) ───────────────────────
def _esc(code: str) -> str:
    return f"\033[{code}m"

RESET  = _esc("0")
BOLD   = _esc("1")
DIM    = _esc("2")
GREEN  = _esc("32")
RED    = _esc("31")
YELLOW = _esc("33")
CYAN   = _esc("36")
BLUE   = _esc("34")

def bold(s):   return f"{BOLD}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"

def header(title: str) -> None:
    print()
    print("═" * 72)
    print(f"  {bold(title)}")
    print("═" * 72)

def section(title: str) -> None:
    print()
    print(f"  {bold(CYAN + title + RESET)}")
    print("  " + "─" * 68)

def ok(msg):    print(f"  {green('✓')} {msg}")
def warn(msg):  print(f"  {yellow('⚠')} {msg}")
def fail(msg):  print(f"  {red('✗')} {msg}")
def info(msg):  print(f"  {dim('·')} {msg}")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_questions(dataset: str) -> List[Dict]:
    path = Path(f"data/{dataset}/questions.json")
    if not path.exists():
        print(red(f"questions.json not found: {path}"))
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks_index(dataset: str) -> Dict[str, Dict]:
    """Returns {chunk_id_str: chunk_dict} and {source_file: [chunks]}."""
    path = Path(f"data/{dataset}/chunks_export.json")
    if not path.exists():
        print(red(f"chunks_export.json not found: {path}"))
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    by_id: Dict[str, Dict] = {}
    by_source: Dict[str, List[Dict]] = {}
    for c in chunks:
        cid = str(c["metadata"].get("chunk_id", ""))
        by_id[cid] = c
        src = c["metadata"].get("source_file", "")
        by_source.setdefault(src, []).append(c)
    return by_id, by_source, chunks


def load_extractions_index(dataset: str) -> Dict[str, Dict]:
    path = Path(f"data/{dataset}/graph/extraction_results.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(r["chunk_id"]): r for r in data.get("results", [])}


# ── Store checks ─────────────────────────────────────────────────────────────

def check_lancedb(dataset: str, source_file: str) -> Tuple[bool, List[Dict]]:
    """Returns (found, rows) for the given source_file in LanceDB."""
    try:
        import lancedb
        db = lancedb.connect(f"data/{dataset}/vector")
        tables = db.table_names()
        if not tables:
            return False, []
        tbl = db.open_table(tables[0])
        df = tbl.to_pandas()
        rows = df[df["source_file"] == source_file]
        return len(rows) > 0, rows.to_dict("records")
    except Exception as e:
        warn(f"LanceDB check failed: {e}")
        return False, []


def check_kuzu(dataset: str, source_file: str, graph_store) -> Tuple[bool, List[str]]:
    """Returns (found, chunk_ids_in_graph) for the source_file."""
    try:
        res = graph_store.conn.execute(
            "MATCH (c:DocumentChunk {source_file: $sf}) RETURN c.chunk_id",
            {"sf": source_file},
        )
        cids = []
        while res.has_next():
            cids.append(str(res.get_next()[0]))
        return len(cids) > 0, cids
    except Exception as e:
        warn(f"KuzuDB check failed: {e}")
        return False, []


def check_graph_entities(dataset: str, chunk_id: str, extraction_index: Dict) -> List[str]:
    """Return entity names stored for a chunk in extraction_results."""
    ext = extraction_index.get(str(chunk_id), {})
    return [e["name"] for e in ext.get("entities", [])]


# ── Retrieval rank check ──────────────────────────────────────────────────────

def check_vector_rank(dataset: str, query: str, target_source: str) -> Optional[int]:
    """
    Run a vector search for query and return the rank (1-based) of the first
    chunk from target_source, or None if not found in top-20.
    """
    try:
        import sys, os
        sys.path.insert(0, ".")
        from src.data_layer.storage import VectorStoreAdapter
        from src.data_layer.embeddings import BatchedOllamaEmbeddings

        embeddings = BatchedOllamaEmbeddings(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            cache_path=Path(f"cache/{dataset}_embeddings.db"),
        )
        vec = embeddings.embed_query(query)
        store = VectorStoreAdapter(
            db_path=f"data/{dataset}/vector",
            embedding_dim=768,
        )
        results = store.vector_search(vec, top_k=20, threshold=0.0)
        for rank, r in enumerate(results, 1):
            src = r.get("metadata", {}).get("source_file", r.get("source_file", ""))
            if src == target_source:
                return rank
        return None
    except Exception as e:
        warn(f"Vector rank check failed: {e}")
        return None


def check_graph_rank(graph_store, entity_name: str, target_source: str) -> Optional[int]:
    """
    Run a graph search for entity_name and return rank of first result from
    target_source, or None if not found in top-10.
    """
    try:
        results = graph_store.find_chunks_by_entity_multihop(entity_name, max_results=10)
        for rank, r in enumerate(results, 1):
            if r.get("source_file", "") == target_source:
                return rank
        return None
    except Exception as e:
        warn(f"Graph rank check failed: {e}")
        return None


# ── Main diagnostic ───────────────────────────────────────────────────────────

def diagnose_question(
    q_idx: int,
    question: Dict,
    by_id: Dict,
    by_source: Dict,
    all_chunks: List,
    extraction_index: Dict,
    graph_store,
    dataset: str,
    run_vector: bool,
) -> None:
    """Full ingestion → retrieval trace for one question."""

    q_text = question["question"]
    gold   = question["answer"]
    q_type = question.get("question_type", "?")
    facts  = question.get("supporting_facts", [])  # list of [article_title, sentence_idx]

    header(f"idx={q_idx}  [{q_type}]")
    print(f"  {bold('Frage:')}  {q_text}")
    print(f"  {bold('Gold:')}   {gold}")
    print(f"  {bold('Supporting facts:')} {facts}")

    # ── 1. Find answer-bearing chunks (text contains gold answer) ─────────────
    section("1. Answer-bearing chunks (text CONTAINS gold answer)")
    gold_lower = gold.lower()
    answer_chunks = [
        c for c in all_chunks
        if gold_lower in c["text"].lower()
    ]
    if answer_chunks:
        ok(f"{len(answer_chunks)} chunk(s) contain the gold answer")
        for c in answer_chunks[:5]:
            src = c["metadata"].get("source_file", "?")
            cid = c["metadata"].get("chunk_id", "?")
            info(f"chunk_id={cid}  source={src}")
            info(f"  text: {c['text'][:160].replace(chr(10), ' ')}")
    else:
        fail(f"Gold answer '{gold}' not found in any chunk text")

    # ── 2. Supporting fact articles ───────────────────────────────────────────
    section("2. Supporting fact articles → chunks in stores")

    supporting_sources = set()
    for fact in facts:
        article_title = fact[0] if isinstance(fact, list) else fact.get("title", "")
        source_key = f"{dataset}_{article_title}"
        supporting_sources.add(source_key)

    for source in sorted(supporting_sources):
        print(f"\n  {bold(source)}")

        # 2a. In chunks_export.json?
        src_chunks = by_source.get(source, [])
        if src_chunks:
            ok(f"chunks_export.json: {len(src_chunks)} chunk(s)")
            for c in src_chunks[:3]:
                cid = c["metadata"].get("chunk_id", "?")
                ents = check_graph_entities(dataset, cid, extraction_index)
                info(f"  chunk_id={cid}  entities: {ents[:8]}")
        else:
            fail(f"NOT in chunks_export.json — article was not ingested!")
            continue

        # 2b. In LanceDB?
        found_vec, vec_rows = check_lancedb(dataset, source)
        if found_vec:
            ok(f"LanceDB: {len(vec_rows)} row(s)")
        else:
            fail("NOT in LanceDB — vector search cannot find this article")

        # 2c. In KuzuDB?
        found_graph, graph_cids = check_kuzu(dataset, source, graph_store)
        if found_graph:
            ok(f"KuzuDB: {len(graph_cids)} DocumentChunk node(s): {graph_cids[:5]}")
        else:
            fail("NOT in KuzuDB — graph search cannot find this article")

    # ── 3. Retrieval rank for supporting sources ──────────────────────────────
    section("3. Retrieval rank — where do supporting chunks appear?")

    # Graph search for each query entity
    # Extract entities simply: capitalized multi-word phrases + quoted strings
    entity_candidates = re.findall(r'"([^"]+)"', q_text)  # quoted
    entity_candidates += re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', q_text)
    entity_candidates = list(dict.fromkeys(entity_candidates))[:5]  # dedup, max 5

    print(f"\n  {bold('Query entities (heuristic):')} {entity_candidates}")

    for source in sorted(supporting_sources):
        print(f"\n  {bold('Target:')} {source}")

        # Graph rank per entity
        for ent in entity_candidates:
            rank = check_graph_rank(graph_store, ent, source)
            if rank is not None:
                ok(f"Graph  entity={ent!r:30s} → rank #{rank}")
            else:
                fail(f"Graph  entity={ent!r:30s} → NOT in top-10")

        # Vector rank
        if run_vector:
            rank = check_vector_rank(dataset, q_text, source)
            if rank is not None:
                ok(f"Vector query  → rank #{rank} (out of 20)")
            else:
                fail(f"Vector query  → NOT in top-20")
        else:
            info("Vector rank check skipped (use --vector to enable)")

    # ── 4. Crowd-out analysis: how many chunks compete per entity? ────────────
    section("4. Crowd-out analysis — how many chunks compete for each entity?")

    for ent in entity_candidates:
        try:
            results = graph_store.find_chunks_by_entity_multihop(ent, max_results=20)
            sources = [r.get("source_file", "?") for r in results]
            print(f"\n  {bold(f'Entity: {ent!r}')}")
            info(f"  {len(results)} graph results total")
            from collections import Counter
            src_counts = Counter(sources)
            for src, cnt in src_counts.most_common(8):
                marker = green("← TARGET") if src in supporting_sources else ""
                print(f"    {cnt}×  {src}  {marker}")
        except Exception as e:
            warn(f"  crowd-out check failed for {ent!r}: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_indices(s: str) -> List[int]:
    """Parse '11,12' or '0-19' or '5' into a list of ints."""
    indices = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion diagnostic: trace chunks through stores to retrieval rank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--indices", "-i", required=True,
                        help="Question indices, e.g. '11,12' or '0-19'")
    parser.add_argument("--dataset", "-d", default="hotpotqa",
                        help="Dataset name (default: hotpotqa)")
    parser.add_argument("--vector", action="store_true",
                        help="Also run vector search rank check (slow — needs Ollama)")
    args = parser.parse_args()

    indices = parse_indices(args.indices)
    dataset = args.dataset

    print(bold(f"\nIngestion Diagnostic — dataset={dataset}  indices={indices}"))
    print(dim("Loading data..."))

    questions = load_questions(dataset)
    by_id, by_source, all_chunks = load_chunks_index(dataset)
    extraction_index = load_extractions_index(dataset)

    print(dim(f"  {len(questions)} questions, {len(all_chunks)} chunks loaded"))

    # Load graph store once
    try:
        from src.data_layer.storage import KuzuGraphStore
        graph_store = KuzuGraphStore(f"data/{dataset}/graph")
        ok("KuzuDB connected")
    except Exception as e:
        fail(f"KuzuDB connection failed: {e}")
        graph_store = None

    for idx in indices:
        if idx >= len(questions):
            warn(f"idx={idx} out of range (max {len(questions)-1})")
            continue
        diagnose_question(
            q_idx=idx,
            question=questions[idx],
            by_id=by_id,
            by_source=by_source,
            all_chunks=all_chunks,
            extraction_index=extraction_index,
            graph_store=graph_store,
            dataset=dataset,
            run_vector=args.vector,
        )

    print()
    print("═" * 72)
    print(bold("  DIAGNOSTIC COMPLETE"))
    print("═" * 72)
    print()


if __name__ == "__main__":
    main()
