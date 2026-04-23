"""
diagnose.py - Layer-by-Layer Pipeline Diagnostik mit echten HotpotQA Daten

Testet jeden Stack-Layer einzeln:
  Layer 1: Embedding (nomic-embed-text via Ollama)
  Layer 2: Vector Search (LanceDB)
  Layer 3: Graph Search (KuzuDB)
  Layer 4: HybridRetriever (RRF Fusion)
  Layer 5: Planner / S_P (Query Decomposition)
  Layer 6: Navigator / S_N (Retrieval + Filtering)
  Layer 7: Verifier / S_V (Antwortgenerierung mit Kontext)
  Layer 8: Full Pipeline (End-to-End)

Usage:
    python test_system/diagnose.py                        # Test alle Layer (Frage idx=0)
    python test_system/diagnose.py --idx 5                # Frage Nr. 5 aus questions.json
    python test_system/diagnose.py --multi 20             # Vector-Scores für 20 Fragen (kein LLM)
    python test_system/diagnose.py --layer embedding      # Nur Embedding
    python test_system/diagnose.py --layer retrieval      # Nur Vector + Graph
    python test_system/diagnose.py --layer pipeline       # Nur Full Pipeline
    python test_system/diagnose.py --question "Who is..."  # Eigene Frage
    python test_system/diagnose.py --skip-llm             # LLM-Calls überspringen (Verifier)
"""

import json
import sys
import time
import argparse
from pathlib import Path

# Projektverzeichnis zu sys.path hinzufügen (damit src.* Imports funktionieren)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Farben für Terminal-Output ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg):  print(f"  {BLUE}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{'─'*70}\n  {msg}\n{'─'*70}{RESET}")

# ─── Konfiguration ─────────────────────────────────────────────────────────────
DATASET   = "hotpotqa"
DATA_DIR  = PROJECT_ROOT / "data" / "hotpotqa"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

import yaml
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)


def load_sample_question(idx: int = 0) -> dict:
    """Lade eine echte HotpotQA Frage."""
    q_path = DATA_DIR / "questions.json"
    if not q_path.exists():
        return None
    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)
    return questions[idx] if idx < len(questions) else questions[0]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def test_embedding(query: str):
    header("LAYER 1 — Embedding (nomic-embed-text via Ollama)")

    from src.data_layer import BatchedOllamaEmbeddings

    emb_cfg = config.get("embeddings", {})
    cache_path = PROJECT_ROOT / "cache" / f"{DATASET}_embeddings.db"

    try:
        t0 = time.time()
        embeddings = BatchedOllamaEmbeddings(
            model_name=emb_cfg.get("model_name", "nomic-embed-text"),
            base_url=emb_cfg.get("base_url", "http://localhost:11434"),
            batch_size=1,
            cache_path=cache_path,
        )
        init_time = (time.time() - t0) * 1000
        ok(f"Ollama erreichbar ({init_time:.0f}ms)")

        t0 = time.time()
        vec = embeddings.embed_query(query)
        embed_time = (time.time() - t0) * 1000

        if vec and len(vec) > 0:
            ok(f"Embedding erzeugt: dim={len(vec)}, Zeit={embed_time:.0f}ms")
            info(f"Erste 5 Werte: {[round(v, 4) for v in vec[:5]]}")
            if len(vec) != 768:
                warn(f"Erwartet dim=768, bekommen dim={len(vec)}")
        else:
            fail("Leerer Embedding-Vektor zurückgegeben")
            return None

        return embeddings

    except Exception as e:
        fail(f"Embedding fehlgeschlagen: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: VECTOR SEARCH (LanceDB direkt)
# ══════════════════════════════════════════════════════════════════════════════

def test_vector_search(query: str, embeddings):
    header("LAYER 2 — Vector Search (LanceDB direkt)")

    from src.data_layer import HybridStore, StorageConfig

    emb_cfg  = config.get("embeddings", {})
    vec_cfg  = config.get("vector_store", {})

    storage_config = StorageConfig(
        vector_db_path=DATA_DIR / "vector",
        graph_db_path=DATA_DIR / "graph",
        embedding_dim=emb_cfg.get("embedding_dim", 768),
        similarity_threshold=vec_cfg.get("similarity_threshold", 0.3),
        normalize_embeddings=vec_cfg.get("normalize_embeddings", True),
        distance_metric=vec_cfg.get("distance_metric", "cosine"),
    )

    try:
        t0 = time.time()
        store = HybridStore(config=storage_config, embeddings=embeddings)
        init_time = (time.time() - t0) * 1000
        ok(f"HybridStore geladen ({init_time:.0f}ms)")

        info("Berechne Query-Embedding...")
        query_embedding = embeddings.embed_query(query)

        t0 = time.time()
        results = store.vector_search(query_embedding, top_k=5)
        search_time = (time.time() - t0) * 1000

        if results:
            ok(f"Vector Search: {len(results)} Treffer in {search_time:.0f}ms")
            for i, r in enumerate(results[:3], 1):
                score = r.get("similarity", r.get("score", "?"))
                text  = r.get("text", "")[:80]
                info(f"  #{i} score={score:.3f}: \"{text}...\"")
        else:
            fail(f"Vector Search: 0 Treffer (Zeit={search_time:.0f}ms)")
            warn("Mögliche Ursachen:")
            warn("  - similarity_threshold zu hoch (aktuell: "
                 f"{vec_cfg.get('similarity_threshold', 0.3)})")
            warn("  - Embedding-Dimension stimmt nicht")
            warn("  - LanceDB leer oder korruptes Index")

        return store

    except Exception as e:
        fail(f"Vector Search fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: GRAPH SEARCH (KuzuDB direkt)
# ══════════════════════════════════════════════════════════════════════════════

def test_graph_search(query: str, store):
    header("LAYER 3 — Graph Search (KuzuDB direkt)")

    import re
    entities = [w for w in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)]
    entities = list(dict.fromkeys(entities))
    info(f"Extrahierte Entitäten: {entities}")

    try:
        t0 = time.time()
        results = store.graph_search(entities=entities if entities else ["test"], top_k=5)
        search_time = (time.time() - t0) * 1000

        if results:
            ok(f"Graph Search: {len(results)} Treffer in {search_time:.0f}ms")
            for i, r in enumerate(results[:3], 1):
                entity = r.get("matched_entity", "?")
                hops   = r.get("hops", "?")
                text   = r.get("text", "")[:80]
                info(f"  #{i} entity='{entity}' hops={hops}: \"{text}...\"")
        else:
            warn(f"Graph Search: 0 Treffer (Zeit={search_time:.0f}ms)")
            warn("  - Entitätsextraktion findet keine Keywords in der Query")
            warn("  - Knowledge Graph hat keine passenden Nodes")
            info("  → Ist normal wenn keine Named Entities in der Query")

    except Exception as e:
        fail(f"Graph Search fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4: HYBRID RETRIEVER (RRF Fusion)
# ══════════════════════════════════════════════════════════════════════════════

def test_hybrid_retriever(query: str, store, embeddings):
    header("LAYER 4 — HybridRetriever (RRF Fusion)")

    from src.data_layer import HybridRetriever, RetrievalConfig, RetrievalMode

    rag_cfg = config.get("rag", {})
    vec_cfg = config.get("vector_store", {})

    retrieval_config = RetrievalConfig(
        mode=RetrievalMode.HYBRID,
        vector_weight=rag_cfg.get("vector_weight", 0.7),
        graph_weight=rag_cfg.get("graph_weight", 0.3),
        similarity_threshold=vec_cfg.get("similarity_threshold", 0.3),
    )

    try:
        retriever = HybridRetriever(
            hybrid_store=store,
            embeddings=embeddings,
            config=retrieval_config,
        )
        ok("HybridRetriever initialisiert")

        t0 = time.time()
        results, metrics = retriever.retrieve(query)
        retrieval_time = (time.time() - t0) * 1000

        if results:
            ok(f"HybridRetriever: {len(results)} Treffer in {retrieval_time:.0f}ms")
            info(f"Metriken: vector={metrics.vector_results}, "
                 f"graph={metrics.graph_results}, "
                 f"fused={metrics.final_results}")
            for i, r in enumerate(results[:3], 1):
                score = getattr(r, "rrf_score", getattr(r, "score", "?"))
                text  = getattr(r, "text", str(r))[:80]
                info(f"  #{i} rrf_score={score:.4f}: \"{text}...\"")
        else:
            fail(f"HybridRetriever: 0 Treffer (Zeit={retrieval_time:.0f}ms)")
            info(f"Metriken: {metrics}")

        return retriever

    except Exception as e:
        fail(f"HybridRetriever fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5: PLANNER / S_P
# ══════════════════════════════════════════════════════════════════════════════

def test_planner(query: str):
    header("LAYER 5 — Planner / S_P (Query Decomposition, kein LLM)")

    from src.logic_layer import create_planner

    try:
        t0 = time.time()
        planner = create_planner(config)
        plan = planner.plan(query)
        plan_time = (time.time() - t0) * 1000

        ok(f"Planner ausgeführt ({plan_time:.0f}ms)")
        info(f"  Query-Typ:  {plan.query_type.value}")
        info(f"  Strategie:  {plan.strategy.value}")
        info(f"  Konfidenz:  {plan.confidence:.2f}")

        if hasattr(plan, 'hop_sequence') and plan.hop_sequence:
            info(f"  Sub-Queries ({len(plan.hop_sequence)}):")
            for hop in plan.hop_sequence:
                info(f"    - \"{hop.sub_query}\"")
        else:
            warn("  Keine Hop-Sequenz erzeugt → nur Original-Query wird verwendet")

        return plan

    except Exception as e:
        fail(f"Planner fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6: NAVIGATOR / S_N
# ══════════════════════════════════════════════════════════════════════════════

def test_navigator(query: str, plan, retriever):
    header("LAYER 6 — Navigator / S_N (Retrieval + Pre-Gen Filtering)")

    from src.logic_layer import Navigator, ControllerConfig

    if retriever is None:
        fail("Kein HybridRetriever verfügbar — Layer 4 muss zuerst erfolgreich sein")
        return None

    try:
        nav_config = ControllerConfig()
        navigator = Navigator(nav_config)
        navigator.set_retriever(retriever)
        ok("Navigator initialisiert, Retriever gesetzt")

        sub_queries = (
            [h.sub_query for h in plan.hop_sequence]
            if (plan and hasattr(plan, 'hop_sequence') and plan.hop_sequence)
            else [query]
        )
        info(f"Sub-Queries: {sub_queries}")

        t0 = time.time()
        nav_result = navigator.navigate(plan, sub_queries)
        nav_time = (time.time() - t0) * 1000

        raw_count = len(nav_result.raw_context) if nav_result.raw_context else 0
        filtered_count = len(nav_result.filtered_context) if nav_result.filtered_context else 0

        if filtered_count > 0:
            ok(f"Navigator: {raw_count} raw → {filtered_count} gefilterte Chunks ({nav_time:.0f}ms)")
            for i, chunk in enumerate(nav_result.filtered_context[:3], 1):
                info(f"  Chunk #{i}: \"{chunk[:100]}...\"")
        else:
            fail(f"Navigator: {raw_count} raw → {filtered_count} Chunks ({nav_time:.0f}ms)")
            meta = nav_result.metadata or {}
            info(f"  Pre-Filter:       {meta.get('pre_filter_count', '?')}")
            info(f"  Nach Relevanz:    {meta.get('after_relevance_filter', '?')}")
            info(f"  Nach Redundanz:   {meta.get('after_redundancy_filter', '?')}")
            if meta.get("retrieval_errors"):
                for err in meta["retrieval_errors"]:
                    warn(f"  Retrieval-Fehler: {err}")

        return nav_result

    except Exception as e:
        fail(f"Navigator fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7: VERIFIER / S_V
# ══════════════════════════════════════════════════════════════════════════════

def test_verifier(query: str, gold_answer: str, context: list, skip_llm: bool):
    header("LAYER 7 — Verifier / S_V (Antwortgenerierung)")

    if skip_llm:
        warn("--skip-llm aktiv: Verifier-Test übersprungen")
        return

    from src.logic_layer import Verifier, VerifierConfig

    if not context:
        warn("Kein Context von Navigator → nutze Dummy-Kontext")
        context = [
            f"[DUMMY] Dieser Text ist ein Platzhalter weil der Navigator keinen Kontext gefunden hat.",
            f"[DUMMY] Die Frage lautet: {query}"
        ]

    try:
        verifier_config = VerifierConfig(
            max_tokens=100,
            timeout=120,
        )
        verifier = Verifier(config=verifier_config)
        ok("Verifier initialisiert")

        info(f"Kontext-Chunks: {len(context)}")
        info(f"Gold-Antwort:   \"{gold_answer}\"")

        t0 = time.time()
        gen_result = verifier.generate_and_verify(query=query, context=context)
        gen_time = (time.time() - t0) * 1000

        ok(f"Verifier abgeschlossen ({gen_time:.0f}ms)")
        info(f"  Antwort:     \"{gen_result.answer}\"")
        info(f"  Konfidenz:   {gen_result.confidence.value}")

        import re
        def norm(t):
            t = t.lower().strip()
            t = re.sub(r'\b(a|an|the)\b', ' ', t)
            t = re.sub(r'[^\w\s]', '', t)
            return ' '.join(t.split())

        pred_n = norm(gen_result.answer)
        gold_n = norm(gold_answer)

        em = gold_n in pred_n or pred_n == gold_n
        pred_tok = pred_n.split()
        gold_tok = gold_n.split()
        common = sum(min(pred_tok.count(w), gold_tok.count(w)) for w in set(gold_tok))
        p = common / len(pred_tok) if pred_tok else 0
        r = common / len(gold_tok) if gold_tok else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0

        if em:
            ok(f"  EM = True  | F1 = {f1:.3f}")
        else:
            warn(f"  EM = False | F1 = {f1:.3f}")
            info(f"  Pred (norm): \"{pred_n[:80]}\"")
            info(f"  Gold (norm): \"{gold_n}\"")

    except Exception as e:
        fail(f"Verifier fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8: FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline(query: str, gold_answer: str, skip_llm: bool,
                       store=None, embeddings=None):
    header("LAYER 8 — Full Pipeline (End-to-End)")

    if skip_llm:
        warn("--skip-llm aktiv: Full Pipeline übersprungen")
        return

    from src.pipeline import create_full_pipeline
    from src.data_layer import HybridStore, StorageConfig
    from src.data_layer import BatchedOllamaEmbeddings
    from src.data_layer import HybridRetriever, RetrievalConfig, RetrievalMode

    emb_cfg = config.get("embeddings", {})
    vec_cfg = config.get("vector_store", {})
    rag_cfg = config.get("rag", {})

    if embeddings is None:
        cache_path = PROJECT_ROOT / "cache" / f"{DATASET}_embeddings.db"
        embeddings = BatchedOllamaEmbeddings(
            model_name=emb_cfg.get("model_name", "nomic-embed-text"),
            base_url=emb_cfg.get("base_url", "http://localhost:11434"),
            batch_size=1,
            cache_path=cache_path,
        )

    if store is None:
        storage_config = StorageConfig(
            vector_db_path=DATA_DIR / "vector",
            graph_db_path=DATA_DIR / "graph",
            embedding_dim=emb_cfg.get("embedding_dim", 768),
            similarity_threshold=vec_cfg.get("similarity_threshold", 0.3),
            normalize_embeddings=vec_cfg.get("normalize_embeddings", True),
            distance_metric=vec_cfg.get("distance_metric", "cosine"),
        )
        store = HybridStore(config=storage_config, embeddings=embeddings)
    else:
        ok("Store aus früherem Layer wiederverwendet (kein KuzuDB-Lock-Konflikt)")

    retrieval_config = RetrievalConfig(
        mode=RetrievalMode.HYBRID,
        vector_weight=rag_cfg.get("vector_weight", 0.7),
        graph_weight=rag_cfg.get("graph_weight", 0.3),
    )
    retriever = HybridRetriever(hybrid_store=store, embeddings=embeddings,
                                config=retrieval_config)

    try:
        pipeline = create_full_pipeline(
            hybrid_retriever=retriever,
            graph_store=store.graph_store,
            config=config,
        )
        ok("Pipeline erstellt")

        t0 = time.time()
        result = pipeline.process(query)
        total_time = (time.time() - t0) * 1000

        ok(f"Pipeline abgeschlossen ({total_time:.0f}ms)")
        info(f"  Antwort:   \"{result.answer}\"")
        info(f"  Gold:      \"{gold_answer}\"")
        info(f"  Konfidenz: {result.confidence}")
        info(f"  S_P: {result.planner_time_ms:.0f}ms | "
             f"S_N: {result.navigator_time_ms:.0f}ms | "
             f"S_V: {result.verifier_time_ms:.0f}ms")

        nav = result.navigator_result or {}
        fc = nav.get("filtered_context", [])
        info(f"  Kontext-Chunks: {len(fc)}")
        if len(fc) == 0:
            warn("  → Coverage=0%! Navigator hat keinen Kontext gefunden.")
            info(f"  Navigator-Metadata: {nav.get('metadata', {})}")

    except Exception as e:
        fail(f"Pipeline fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-QUESTION VECTOR SCORE ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

def test_multi_vector(n_questions: int):
    header(f"MULTI-VECTOR ANALYSE — {n_questions} Fragen (kein LLM)")

    q_path = DATA_DIR / "questions.json"
    if not q_path.exists():
        fail(f"questions.json nicht gefunden: {q_path}")
        return

    with open(q_path, encoding="utf-8") as f:
        all_qs = json.load(f)

    n_questions = min(n_questions, len(all_qs))
    info(f"Verfügbare Fragen: {len(all_qs)} — teste {n_questions}")

    from src.data_layer import BatchedOllamaEmbeddings
    from src.data_layer import HybridStore, StorageConfig

    emb_cfg = config.get("embeddings", {})
    vec_cfg = config.get("vector_store", {})
    cache_path = PROJECT_ROOT / "cache" / f"{DATASET}_embeddings.db"

    try:
        embeddings = BatchedOllamaEmbeddings(
            model_name=emb_cfg.get("model_name", "nomic-embed-text"),
            base_url=emb_cfg.get("base_url", "http://localhost:11434"),
            batch_size=1,
            cache_path=cache_path,
        )
        storage_config = StorageConfig(
            vector_db_path=DATA_DIR / "vector",
            graph_db_path=DATA_DIR / "graph",
            embedding_dim=emb_cfg.get("embedding_dim", 768),
            similarity_threshold=0.0,
            normalize_embeddings=vec_cfg.get("normalize_embeddings", True),
            distance_metric=vec_cfg.get("distance_metric", "cosine"),
        )
        store = HybridStore(config=storage_config, embeddings=embeddings)
    except Exception as e:
        fail(f"Init fehlgeschlagen: {e}")
        return

    def categorize(q: str) -> str:
        q_lower = q.lower()
        if q_lower.startswith(("were ", "was ", "is ", "are ", "did ", "do ", "does ", "has ", "have ")):
            return "yes/no"
        elif "same" in q_lower or "both" in q_lower or "also" in q_lower:
            return "compare"
        elif q_lower.startswith(("who ", "what ", "which ")):
            return "entity"
        elif q_lower.startswith(("when ", "where ", "how ")):
            return "factual"
        return "other"

    rows = []
    bad_idx = []

    print(f"\n  {'#':>3}  {'Top-1 Score':>11}  {'Top-3 Ø':>8}  {'Typ':>8}  Frage (gekürzt)")
    print(f"  {'─'*3}  {'─'*11}  {'─'*8}  {'─'*8}  {'─'*50}")

    for i in range(n_questions):
        sample = all_qs[i]
        query  = sample.get("question", sample.get("q", ""))
        answer = sample.get("answer", sample.get("a", ""))
        qtype  = categorize(query)

        try:
            qvec    = embeddings.embed_query(query)
            results = store.vector_search(qvec, top_k=3)

            if results:
                scores  = [r.get("similarity", r.get("score", 0)) for r in results]
                top1    = scores[0]
                top3avg = sum(scores) / len(scores)
            else:
                top1 = top3avg = 0.0

            if top1 >= 0.70:
                color = GREEN
            elif top1 >= 0.50:
                color = YELLOW
            else:
                color = RED
                bad_idx.append(i)

            rows.append((i, top1, top3avg, qtype, query, answer))
            q_short = query[:50].ljust(50)
            print(f"  {i:>3}  {color}{top1:>11.4f}{RESET}  {top3avg:>8.4f}  {qtype:>8}  {q_short}")

        except Exception as e:
            rows.append((i, -1, -1, qtype, query, answer))
            print(f"  {i:>3}  {RED}  ERROR{RESET}  {'?':>8}  {'?':>8}  {query[:50]}")

    valid_rows = [(i, s1, s3, t, q, a) for (i, s1, s3, t, q, a) in rows if s1 >= 0]
    if valid_rows:
        avg_top1 = sum(r[1] for r in valid_rows) / len(valid_rows)
        good     = sum(1 for r in valid_rows if r[1] >= 0.70)
        ok_count = sum(1 for r in valid_rows if 0.50 <= r[1] < 0.70)
        bad      = sum(1 for r in valid_rows if r[1] < 0.50)

        print(f"\n  Zusammenfassung ({len(valid_rows)} Fragen):")
        ok(f"Gut   (≥0.70): {good:>3} ({100*good/len(valid_rows):.0f}%)")
        warn(f"OK    (≥0.50): {ok_count:>3} ({100*ok_count/len(valid_rows):.0f}%)")
        fail(f"Schlecht (<0.50): {bad:>3} ({100*bad/len(valid_rows):.0f}%)")
        info(f"Ø Top-1 Score: {avg_top1:.4f}")

        if bad_idx:
            print(f"\n  Schlechteste Fragen (idx für --idx nutzen):")
            for idx in bad_idx[:5]:
                r = next(r for r in valid_rows if r[0] == idx)
                info(f"  [{idx}] score={r[1]:.3f} typ={r[3]} | \"{r[4][:70]}\"")
                info(f"       Gold: \"{r[5]}\"")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Diagnose Pipeline Layer-by-Layer")
    parser.add_argument("--layer", choices=[
        "embedding", "retrieval", "planner", "navigator", "verifier", "pipeline", "all"
    ], default="all")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--multi", type=int, default=0)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    if args.multi > 0:
        print(f"\n{BOLD}{'═'*70}\n  EDGE-RAG DIAGNOSE — MULTI-VECTOR MODUS\n{'═'*70}{RESET}")
        test_multi_vector(args.multi)
        print(f"\n{BOLD}{'═'*70}\n  DIAGNOSE ABGESCHLOSSEN\n{'═'*70}{RESET}\n")
        return

    sample = load_sample_question(args.idx)
    if sample is None:
        print(f"{RED}FEHLER: data/hotpotqa/questions.json nicht gefunden!{RESET}")
        print("Zuerst ingestieren: python benchmark_datasets.py ingest --dataset hotpotqa --samples 50")
        sys.exit(1)

    query       = args.question or sample.get("question", sample.get("q", ""))
    gold_answer = sample.get("answer", sample.get("a", ""))

    print(f"\n{BOLD}{'═'*70}")
    print(f"  EDGE-RAG DIAGNOSE")
    print(f"{'═'*70}{RESET}")
    info(f"Frage ({args.idx}): \"{query}\"")
    info(f"Gold-Antwort:   \"{gold_answer}\"")
    print()

    run_all = args.layer == "all"

    embeddings = None
    if run_all or args.layer == "embedding":
        embeddings = test_embedding(query)

    store = None
    if run_all or args.layer == "retrieval":
        if embeddings is None:
            embeddings = test_embedding(query)
        if embeddings:
            store = test_vector_search(query, embeddings)
            if store:
                test_graph_search(query, store)

    retriever = None
    if run_all or args.layer in ("retrieval", "navigator"):
        if embeddings is None:
            embeddings = test_embedding(query)
        if store is None and embeddings:
            store = test_vector_search(query, embeddings)
        if store and embeddings:
            retriever = test_hybrid_retriever(query, store, embeddings)

    plan = None
    if run_all or args.layer == "planner":
        plan = test_planner(query)

    nav_result = None
    if run_all or args.layer == "navigator":
        if plan is None:
            plan = test_planner(query)
        if retriever is None:
            if embeddings is None:
                embeddings = test_embedding(query)
            if store is None and embeddings:
                store = test_vector_search(query, embeddings)
            if store and embeddings:
                retriever = test_hybrid_retriever(query, store, embeddings)
        nav_result = test_navigator(query, plan, retriever)

    if run_all or args.layer == "verifier":
        context = nav_result.filtered_context if nav_result else []
        test_verifier(query, gold_answer, context, args.skip_llm)

    if run_all or args.layer == "pipeline":
        test_full_pipeline(query, gold_answer, args.skip_llm,
                           store=store, embeddings=embeddings)

    print(f"\n{BOLD}{'═'*70}\n  DIAGNOSE ABGESCHLOSSEN\n{'═'*70}{RESET}\n")


if __name__ == "__main__":
    main()
