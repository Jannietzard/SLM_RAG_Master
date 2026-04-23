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
    python diagnose.py                        # Test alle Layer (Frage idx=0)
    python diagnose.py --idx 5                # Frage Nr. 5 aus questions.json
    python diagnose.py --multi 20             # Vector-Scores für 20 Fragen (kein LLM)
    python diagnose.py --layer embedding      # Nur Embedding
    python diagnose.py --layer retrieval      # Nur Vector + Graph
    python diagnose.py --layer pipeline       # Nur Full Pipeline
    python diagnose.py --question "Who is..."  # Eigene Frage
    python diagnose.py --skip-llm             # LLM-Calls überspringen (Verifier)
"""

import json
import sys
import time
import argparse
from pathlib import Path

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
DATA_DIR  = Path("./data/hotpotqa")
CONFIG_PATH = Path("./config/settings.yaml")

# ─── Setup ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

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
    cache_path = Path(f"./cache/{DATASET}_embeddings.db")

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

        # Embedding des Query-Textes berechnen
        info("Berechne Query-Embedding...")
        query_embedding = embeddings.embed_query(query)

        # Test: direkter Vektorabruf (erwartet vorberechneten Vektor)
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

    # GLiNER-Entitätsextraktion (konsistent mit Ingestion)
    from src.data_layer import ImprovedQueryEntityExtractor
    _qee = ImprovedQueryEntityExtractor()
    entities = _qee.extract(query)
    if not entities:
        # Fallback: großgeschriebene Wörter falls GLiNER leer
        import re
        entities = list(dict.fromkeys(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)))
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
        nav_cfg = config.get("navigator", {})
        llm_cfg = config.get("llm", {})
        agent_cfg = config.get("agent", {})
        nav_config = ControllerConfig(
            model_name=llm_cfg.get("model_name", "phi3"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=llm_cfg.get("temperature", 0.1),
            max_verification_iterations=agent_cfg.get("max_verification_iterations", 1),
            relevance_threshold_factor=nav_cfg.get("relevance_threshold_factor", 0.6),
            redundancy_threshold=nav_cfg.get("redundancy_threshold", 0.8),
            max_context_chunks=nav_cfg.get("max_context_chunks", 10),
            rrf_k=nav_cfg.get("rrf_k", 60),
            top_k_per_subquery=nav_cfg.get("top_k_per_subquery", 10),
            max_chars_per_doc=llm_cfg.get("max_chars_per_doc", 300),
        )
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
# LAYER 7: VERIFIER / S_V (mit bekanntem Kontext)
# ══════════════════════════════════════════════════════════════════════════════

def test_verifier(query: str, gold_answer: str, context: list, skip_llm: bool):
    header("LAYER 7 — Verifier / S_V (Antwortgenerierung)")

    if skip_llm:
        warn("--skip-llm aktiv: Verifier-Test übersprungen")
        return

    from src.logic_layer import create_verifier

    # Wenn kein echtes Context vorhanden, nutze Dummy
    if not context:
        warn("Kein Context von Navigator → nutze Dummy-Kontext")
        context = [
            f"[DUMMY] Dieser Text ist ein Platzhalter weil der Navigator keinen Kontext gefunden hat.",
            f"[DUMMY] Die Frage lautet: {query}"
        ]

    try:
        # Lädt alle Werte aus settings.yaml (max_docs, max_context_chars, timeout, …)
        verifier = create_verifier(cfg=config)
        ok("Verifier initialisiert")

        info(f"Kontext-Chunks: {len(context)}")
        info(f"Gold-Antwort:   \"{gold_answer}\"")

        t0 = time.time()
        gen_result = verifier.generate_and_verify(query=query, context=context)
        gen_time = (time.time() - t0) * 1000

        ok(f"Verifier abgeschlossen ({gen_time:.0f}ms)")
        info(f"  Antwort:     \"{gen_result.answer}\"")
        info(f"  Konfidenz:   {gen_result.confidence.value}")

        # Manuelle EM + F1 Berechnung
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

    # Wiederverwendung aus früheren Layern verhindert KuzuDB-Lock-Konflikt
    if embeddings is None:
        cache_path = Path(f"./cache/{DATASET}_embeddings.db")
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

        # Coverage aus navigator_result
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
# GRAPH QUALITY ANALYSE (--graph-quality N)
# ══════════════════════════════════════════════════════════════════════════════

def test_graph_quality(n_questions: int):
    """
    Misst wie gut der Graph für N echte HotpotQA-Fragen arbeitet.

    Erweiterte Diagnose zeigt pro Frage:
      - Welche Entität wurde tatsächlich gematcht? (Hub-Kontamination?)
      - Enthält ein Graph-Chunk die Gold-Antwort? (Answer Coverage)
      - Vergleich mit Vector-Search Answer Coverage

    Erkennt das "False-Positive-Hit" Problem: Keyword trifft Hub-Entitäten
    wie "He", "American", "United States" die für fast jede Frage matchen,
    aber keine relevanten Chunks zurückgeben.
    """
    header(f"GRAPH QUALITY ANALYSE — {n_questions} Fragen")

    q_path = DATA_DIR / "questions.json"
    if not q_path.exists():
        fail(f"questions.json nicht gefunden: {q_path}")
        return

    with open(q_path, encoding="utf-8") as f:
        all_qs = json.load(f)

    n_questions = min(n_questions, len(all_qs))
    info(f"Verfügbare Fragen: {len(all_qs)} — analysiere {n_questions}")

    # Graph-Metadata anzeigen
    meta_path = DATA_DIR / "graph" / "extraction_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        stats = meta.get("graph_stats", {})
        ok(f"Graph-Stats: {stats.get('unique_entities', '?')} unique entities, "
           f"{stats.get('relations', '?')} relations, "
           f"{stats.get('document_chunks', '?')} chunks")
        gliner_model = meta.get("gliner_model", "?")
        ok(f"Ingestion-Methode: GLiNER ({gliner_model})")
    else:
        warn("Keine extraction_metadata.json — Graph-Stats unbekannt")

    from src.data_layer import HybridStore, StorageConfig
    from src.data_layer import BatchedOllamaEmbeddings

    emb_cfg = config.get("embeddings", {})
    vec_cfg = config.get("vector_store", {})
    cache_path = Path(f"./cache/{DATASET}_embeddings.db")

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
            normalize_embeddings=vec_cfg.get("normalize_embeddings", True),
            distance_metric=vec_cfg.get("distance_metric", "cosine"),
            similarity_threshold=0.0,   # Kein Filter – alle Treffer sehen
        )
        store = HybridStore(config=storage_config, embeddings=embeddings)
    except Exception as e:
        fail(f"Store-Init fehlgeschlagen: {e}")
        return

    # GLiNER laden (optional)
    gliner = None
    try:
        from gliner import GLiNER
        gliner = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        ok("GLiNER geladen (urchade/gliner_small-v2.1)")
    except Exception as e:
        warn(f"GLiNER nicht ladbar: {e} — nur Keyword-Extraktion")

    # ─── Hub-Erkennung ────────────────────────────────────────────────────────
    # Entitäten die in fast jeder englischen Antwort vorkommen → Rauschen statt Signal
    HUB_WORDS = {
        # Pronomen (häufigster Fehler bei GLiNER/NER)
        "i", "he", "she", "it", "they", "we", "you",
        "his", "her", "their", "him", "them", "who", "that",
        "this", "these", "those", "its",
        # Generische Terme (treffen zu viele Chunks)
        "american", "united states", "us", "uk", "british",
        "country", "film", "movie", "people", "world", "city",
        "government", "man", "woman", "year", "time", "place",
        "football", "song", "album", "book", "company", "university",
        "music", "television", "television series", "series",
    }

    def is_hub(entity_name: str) -> bool:
        """True wenn Entity ein Hub ist (nicht-spezifisch / zu generisch)."""
        if not entity_name:
            return False
        e = entity_name.strip().lower()
        if len(e) <= 2:              # "I", "He", "US", Einzelbuchstaben
            return True
        if e in HUB_WORDS:
            return True
        return False

    # ─── Entity-Extraktion ───────────────────────────────────────────────────
    import re

    def extract_keyword(query: str):
        ents = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        return list(dict.fromkeys(ents))

    GLINER_TYPES = ["person", "organization", "city", "country", "film", "movie", "event"]

    def extract_gliner(query: str):
        if gliner is None:
            return []
        try:
            preds = gliner.predict_entities(query, GLINER_TYPES, threshold=0.3)
            return list(dict.fromkeys(p["text"] for p in preds))
        except Exception:
            return []

    # ─── Answer Coverage Prüfung ─────────────────────────────────────────────
    def norm_ans(t: str) -> str:
        t = t.lower().strip()
        t = re.sub(r'\b(a|an|the)\b', ' ', t)
        t = re.sub(r'[^\w\s]', '', t)
        return ' '.join(t.split())

    def answer_in_chunks(gold: str, chunks: list) -> bool:
        """True wenn normalisierte Gold-Antwort in irgendeinem Chunk vorkommt."""
        gold_n = norm_ans(gold)
        if not gold_n or len(gold_n) < 2:
            return False
        for c in chunks:
            text = c.get("text", "") if isinstance(c, dict) else str(c)
            if gold_n in norm_ans(text):
                return True
        return False

    # ─── Tabellen-Header ─────────────────────────────────────────────────────
    print(f"\n  {'#':>3}  {'KW-H':>4}  {'Top-Entity (gematcht)':>22}  {'Hub':>3}  "
          f"{'GrAns':>5}  {'VecH':>4}  {'VecAns':>6}  Frage")
    print(f"  {'─'*3}  {'─'*4}  {'─'*22}  {'─'*3}  {'─'*5}  {'─'*4}  {'─'*6}  {'─'*40}")

    # ─── Zähler ──────────────────────────────────────────────────────────────
    kw_hits = kw_miss = 0
    hub_contaminated   = 0   # Fragen wo alle KW-Treffer via Hub-Entity kamen
    any_hub_hit        = 0   # Fragen wo mindestens ein Treffer via Hub war
    graph_answer_hits  = 0   # Fragen wo Gold-Antwort in Graph-Chunks vorkommt
    vec_answer_hits    = 0   # Fragen wo Gold-Antwort in Vector-Chunks vorkommt

    gl_hits = gl_graph_answer_hits = 0  # GLiNER-Zähler (nur wenn verfügbar)

    for i in range(n_questions):
        sample = all_qs[i]
        query  = sample.get("question", "")
        gold   = sample.get("answer", "")

        ents_kw = extract_keyword(query)

        try:
            res_kw = store.graph_search(entities=ents_kw if ents_kw else ["___"], top_k=5)
        except Exception as e:
            warn(f"  [{i}] graph_search Fehler: {e}")
            continue

        n_kw = len(res_kw)
        if n_kw > 0:
            kw_hits += 1
        else:
            kw_miss += 1

        # ─── Hub-Kontaminations-Diagnose ──────────────────────────────────
        matched_entities = [r.get("matched_entity", "") for r in res_kw]
        top_entity = matched_entities[0] if matched_entities else "NONE"
        top_is_hub = is_hub(top_entity)

        if res_kw and all(is_hub(e) for e in matched_entities):
            hub_contaminated += 1
        if res_kw and any(is_hub(e) for e in matched_entities):
            any_hub_hit += 1

        # ─── Graph Answer Coverage ────────────────────────────────────────
        g_in_graph = answer_in_chunks(gold, res_kw) if gold else False
        if g_in_graph:
            graph_answer_hits += 1

        # ─── Vector Search für Vergleich ──────────────────────────────────
        vec_ans = False
        n_vec = 0
        try:
            qvec    = embeddings.embed_query(query)
            res_vec = store.vector_search(qvec, top_k=5)
            n_vec   = len(res_vec)
            vec_ans = answer_in_chunks(gold, res_vec)
            if vec_ans:
                vec_answer_hits += 1
        except Exception:
            pass

        # ─── Zeilenausgabe ────────────────────────────────────────────────
        kw_c   = GREEN if n_kw > 0 else RED
        hub_c  = RED   if top_is_hub else GREEN
        grn_c  = GREEN if g_in_graph else RED
        vec_c  = GREEN if vec_ans else RED

        top_ent_disp = (top_entity[:20] + "..") if len(top_entity) > 22 else top_entity.ljust(22)
        hub_flag = f"{RED}HUB{RESET}" if top_is_hub else f"{GREEN} ok{RESET}"
        gr_flag  = f"{GREEN}  ✓  {RESET}" if g_in_graph else f"{RED}  ✗  {RESET}"
        v_flag   = f"{GREEN}  ✓   {RESET}" if vec_ans else f"{RED}  ✗   {RESET}"

        print(
            f"  {i:>3}  {kw_c}{n_kw:>4}{RESET}  {hub_c}{top_ent_disp}{RESET}  "
            f"{hub_flag}  {gr_flag}  {n_vec:>4}  {v_flag}  {query[:40]}"
        )

    total = kw_hits + kw_miss
    print(f"\n  {'─'*80}")
    print(f"\n  {BOLD}Zusammenfassung ({total} Fragen):{RESET}")

    # Graph-Hits
    ok(f"Graph-Hits (KW):          {kw_hits:>3}/{total} ({100*kw_hits//max(total,1):>3}%)")

    # Hub-Kontamination
    hub_all_rate = 100 * hub_contaminated // max(kw_hits, 1)
    hub_any_rate = 100 * any_hub_hit      // max(kw_hits, 1)
    if hub_all_rate >= 50:
        fail(f"Hub-Kontamination (alle): {hub_contaminated:>3}/{kw_hits} Hits ({hub_all_rate:>3}%) NUR via Hub-Entity")
    else:
        warn(f"Hub-Kontamination (alle): {hub_contaminated:>3}/{kw_hits} Hits ({hub_all_rate:>3}%) NUR via Hub-Entity")
    warn(f"Hub-Kontamination (mind):{any_hub_hit:>4}/{kw_hits} Hits ({hub_any_rate:>3}%) mit ≥1 Hub-Entity")

    # Answer Coverage
    ans_rate = 100 * graph_answer_hits // max(total, 1)
    vec_rate = 100 * vec_answer_hits   // max(total, 1)
    if ans_rate >= 50:
        ok(f"Graph Answer Coverage:    {graph_answer_hits:>3}/{total} ({ans_rate:>3}%) — Gold in Graph-Chunk")
    else:
        fail(f"Graph Answer Coverage:    {graph_answer_hits:>3}/{total} ({ans_rate:>3}%) — Gold in Graph-Chunk")
    if vec_rate >= 50:
        ok(f"Vector Answer Coverage:   {vec_answer_hits:>3}/{total} ({vec_rate:>3}%) — Gold in Vector-Chunk")
    else:
        warn(f"Vector Answer Coverage:   {vec_answer_hits:>3}/{total} ({vec_rate:>3}%) — Gold in Vector-Chunk")

    # Effektiver Graph-Mehrwert
    graph_only_answers = sum(
        1 for i in range(n_questions)
        # Nicht messbar ohne separate Liste, Hinweis im Fazit
    )
    print()
    info("Diagnose:")
    if hub_all_rate >= 60:
        fail(f"  PROBLEM: {hub_all_rate}% der Graph-Hits kommen NUR via Hub-Entitäten")
        fail(f"  → Die '100% Hit-Rate' ist ein False-Positive-Artefakt!")
        fail(f"  → Keyword findet 'American', 'He', 'United States' in fast jeder Frage")
        warn(f"  → Graph Answer Coverage ({ans_rate}%) = echter Mehrwert")
        info(f"  → Das erklärt warum graph_only ≈ vector_only in Ablation")
        print()
        info(f"  Lösungsansätze für Masterarbeit:")
        info(f"  1. Hub-Filter in graph_search: Entitäten mit >200 MENTIONS überspringen")
        info(f"  2. Min-specificity Score: nur Named Entities mit ≥2 Wörtern oder")
        info(f"     bekannte Proper Nouns (Kapitalisierung + nicht in HUB_WORDS)")
        info(f"  3. Als Limitation dokumentieren: Graph-Mehrwert nur bei bridge-type")
        info(f"     Fragen mit klar benannten Entitäten in der Query messbar")
    elif ans_rate < vec_rate - 10:
        warn(f"  Graph Coverage ({ans_rate}%) < Vector Coverage ({vec_rate}%) um >10pp")
        warn(f"  → Graph fügt keinen verlässlichen Mehrwert hinzu")
        info(f"  → Hybrid-Fusion verwässert ggf. die Vector-Ergebnisse")
    elif ans_rate >= vec_rate:
        ok(f"  Graph Coverage ({ans_rate}%) ≥ Vector Coverage ({vec_rate}%) → Graph hilft!")
    else:
        info(f"  Gemischtes Bild: Graph={ans_rate}%, Vector={vec_rate}%")

    # GLiNER Vergleich (wenn verfügbar)
    if gliner:
        print()
        info("GLiNER Answer Coverage Vergleich:")
        gl_ans_hits = 0
        for i in range(min(n_questions, len(all_qs))):
            sample  = all_qs[i]
            query   = sample.get("question", "")
            gold    = sample.get("answer", "")
            ents_gl = extract_gliner(query)
            try:
                res_gl = store.graph_search(entities=ents_gl if ents_gl else ["___"], top_k=5)
                if answer_in_chunks(gold, res_gl):
                    gl_ans_hits += 1
            except Exception:
                pass
        gl_ans_rate = 100 * gl_ans_hits // max(n_questions, 1)
        if gl_ans_rate > ans_rate:
            ok(f"  GLiNER Graph Coverage: {gl_ans_hits}/{n_questions} ({gl_ans_rate}%) — besser als Keyword ({ans_rate}%)")
            ok(f"  → Re-Ingest mit GLiNER-Extraction würde Graph-Mehrwert verbessern!")
        else:
            info(f"  GLiNER Graph Coverage: {gl_ans_hits}/{n_questions} ({gl_ans_rate}%) vs Keyword ({ans_rate}%)")
            info(f"  → GLiNER bringt keinen messbaren Graph-Coverage-Vorteil")
    else:
        warn("GLiNER nicht verfügbar — GL-Vergleich nicht möglich")


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-QUESTION VECTOR SCORE ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

def test_multi_vector(n_questions: int):
    """
    Testet Vector Search für N Fragen und zeigt Score-Verteilung.
    Hilft schlechte Fragen (zu geringe Ähnlichkeit) von guten zu trennen.
    Kein LLM nötig.
    """
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
    cache_path = Path(f"./cache/{DATASET}_embeddings.db")

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
            similarity_threshold=0.0,          # Kein Filter – wir wollen alle Scores sehen
            normalize_embeddings=vec_cfg.get("normalize_embeddings", True),
            distance_metric=vec_cfg.get("distance_metric", "cosine"),
        )
        store = HybridStore(config=storage_config, embeddings=embeddings)
    except Exception as e:
        fail(f"Init fehlgeschlagen: {e}")
        return

    # Frage-Typ-Kategorisierung (grob)
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

            # Farbcodierung
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

    # Zusammenfassung
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
    parser.add_argument("--question", type=str, default=None,
                        help="Eigene Frage statt HotpotQA Sample")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index der HotpotQA Frage (default: 0)")
    parser.add_argument("--multi", type=int, default=0,
                        help="Vector-Score-Analyse für N Fragen (kein LLM, z.B. --multi 30)")
    parser.add_argument("--graph-quality", type=int, default=0, metavar="N",
                        help="Graph-Qualitäts-Analyse: Hub-Kontamination, Answer Coverage, Vector-Vergleich")
    parser.add_argument("--skip-llm", action="store_true",
                        help="LLM-Calls überspringen (Verifier + Pipeline)")
    args = parser.parse_args()

    # ─── Graph-Quality Modus ─────────────────────────────────────────────────
    if args.graph_quality > 0:
        print(f"\n{BOLD}{'═'*70}\n  EDGE-RAG DIAGNOSE — GRAPH QUALITY MODUS\n{'═'*70}{RESET}")
        test_graph_quality(args.graph_quality)
        print(f"\n{BOLD}{'═'*70}\n  DIAGNOSE ABGESCHLOSSEN\n{'═'*70}{RESET}\n")
        return

    # ─── Multi-Vector Modus ──────────────────────────────────────────────────
    if args.multi > 0:
        print(f"\n{BOLD}{'═'*70}\n  EDGE-RAG DIAGNOSE — MULTI-VECTOR MODUS\n{'═'*70}{RESET}")
        test_multi_vector(args.multi)
        print(f"\n{BOLD}{'═'*70}\n  DIAGNOSE ABGESCHLOSSEN\n{'═'*70}{RESET}\n")
        return

    # Lade Sample-Frage
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

    # ─── Layer 1: Embedding ───────────────────────────────────────────────────
    embeddings = None
    if run_all or args.layer == "embedding":
        embeddings = test_embedding(query)

    # ─── Layer 2+3: Vector + Graph Search ────────────────────────────────────
    store = None
    if run_all or args.layer == "retrieval":
        if embeddings is None:
            embeddings = test_embedding(query)
        if embeddings:
            store = test_vector_search(query, embeddings)
            if store:
                test_graph_search(query, store)

    # ─── Layer 4: HybridRetriever ─────────────────────────────────────────────
    retriever = None
    if run_all or args.layer in ("retrieval", "navigator"):
        if embeddings is None:
            embeddings = test_embedding(query)
        if store is None and embeddings:
            store = test_vector_search(query, embeddings)
        if store and embeddings:
            retriever = test_hybrid_retriever(query, store, embeddings)

    # ─── Layer 5: Planner ─────────────────────────────────────────────────────
    plan = None
    if run_all or args.layer == "planner":
        plan = test_planner(query)

    # ─── Layer 6: Navigator ───────────────────────────────────────────────────
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

    # ─── Layer 7: Verifier ────────────────────────────────────────────────────
    if run_all or args.layer == "verifier":
        context = nav_result.filtered_context if nav_result else []
        test_verifier(query, gold_answer, context, args.skip_llm)

    # ─── Layer 8: Full Pipeline ───────────────────────────────────────────────
    if run_all or args.layer == "pipeline":
        # store + embeddings weitergeben → verhindert KuzuDB-Lock-Konflikt
        test_full_pipeline(query, gold_answer, args.skip_llm,
                           store=store, embeddings=embeddings)

    print(f"\n{BOLD}{'═'*70}\n  DIAGNOSE ABGESCHLOSSEN\n{'═'*70}{RESET}\n")


if __name__ == "__main__":
    main()
