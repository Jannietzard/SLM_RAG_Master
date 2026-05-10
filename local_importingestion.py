"""
═══════════════════════════════════════════════════════════════════════════════
LOCAL IMPORT INGESTION - Phase 3 of the decoupled ingestion pipeline
═══════════════════════════════════════════════════════════════════════════════

Imports the Colab extraction results into the local stores (LanceDB + KuzuDB).

Input:
    - chunks_export.json                            (Phase 1: chunks from benchmark_datasets.py)
    - data/<dataset>/graph/extraction_results.json  (Phase 2: entities + relations from Colab)

Output:
    - data/<dataset>/vector/       (LanceDB)
    - data/<dataset>/graph/        (KuzuDB)

Usage:
    python local_importingestion.py \\
        --chunks data/hotpotqa/chunks_export.json \\
        --extractions data/hotpotqa/graph/extraction_results.json \\
        --dataset hotpotqa

    # With an explicit config file
    python local_importingestion.py \\
        --chunks data/hotpotqa/chunks_export.json \\
        --extractions data/hotpotqa/graph/extraction_results.json \\
        --dataset hotpotqa \\
        --config config/settings.yaml

    # Graph only (vector store already exists)
    python local_importingestion.py \\
        --chunks data/hotpotqa/chunks_export.json \\
        --extractions data/hotpotqa/graph/extraction_results.json \\
        --dataset hotpotqa \\
        --graph-only

Graph-quality post-processing (always applied unless --no-cleanup):
    1. Canonical entity IDs           — deduplicate at MERGE time using
                                        canonical surface form (parentheticals,
                                        honorifics, suffixes, possessives).
    2. Co-occurrence edges            — every pair of entities co-mentioned
                                        in the same chunk gets a RELATED_TO
                                        edge (relation_type='cooccurs').
    3. Cleanup pass                   — drop orphan entities, drop hubs that
                                        exceed the mention-count threshold,
                                        merge duplicates sharing canonical form.
    4. Baseline + invariant assertion — print metrics and warn on threshold
                                        violations; never aborts the import.

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import yaml

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Quiet sub-modules
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ============================================================================
# IMPORTS MIT FALLBACK
# ============================================================================

try:
    from langchain.schema import Document
except ImportError:
    @dataclass
    class Document:
        page_content: str
        metadata: dict

try:
    from src.data_layer import HybridStore, StorageConfig, KuzuGraphStore
    from src.data_layer import BatchedOllamaEmbeddings
    from src.data_layer.graph_quality import (
        assert_graph_invariants,
        build_cooccurrence_edges,
        canonical_form,
        cleanup_graph,
        compute_graph_baseline,
        format_baseline_report,
        link_entities_by_embedding,
    )
    from src.data_layer.svo_extraction import (
        extract_svo_relations,
        is_available as svo_available,
    )
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.error(
        "Storage modules not found! "
        "Make sure you are in the project root and "
        "src/data_layer/storage.py exists."
    )


# ============================================================================
# CANONICAL ENTITY IDENTIFIERS
# ============================================================================

def _canonical_entity_id(name: str, entity_type: str) -> str:
    """
    Deterministic entity identifier based on the canonical surface form.

    Differs from `entity_extraction._generate_entity_id` only in the
    normalisation applied to the name: this version uses
    `graph_quality.canonical_form`, which strips parentheticals,
    honorifics, name suffixes, possessives, and applies NFKC + casefold.

    Two surface forms that collapse under `canonical_form` (e.g. "Ed Wood",
    "Ed Wood (filmmaker)", "ED WOOD ") produce the same entity_id, so the
    KuzuDB MERGE on entity_id deduplicates them at insert time.
    """
    canon = canonical_form(name)
    combined = f"{canon}:{entity_type or 'UNKNOWN'}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:24]

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable
        def __iter__(self):
            return iter(self._iterable) if self._iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kwargs):
            pass


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load config from YAML or fall back to defaults."""
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # Try the default path
    default_path = Path("./config/settings.yaml")
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    logger.warning("Keine Config gefunden — verwende Defaults")
    return {
        "embeddings": {
            "model_name": "nomic-embed-text",
            "base_url": "http://localhost:11434",
            "embedding_dim": 768,
        },
        "vector_store": {
            "similarity_threshold": 0.3,
            "distance_metric": "cosine",
            "normalize_embeddings": True,
        },
        "performance": {
            "batch_size": 32,
            "device": "cpu",
        },
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunks(chunks_path: Path) -> List[Dict]:
    """Lade chunks_export.json."""
    logger.info(f"Lade Chunks: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"  {len(chunks)} Chunks geladen")
    return chunks


def load_extractions(extractions_path: Path) -> Dict:
    """Lade extraction_results.json."""
    logger.info(f"Loading extraction results: {extractions_path}")
    with open(extractions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    results = data.get("results", [])

    logger.info(f"  {meta.get('total_chunks', '?')} Chunks verarbeitet")
    logger.info(f"  {meta.get('total_entities', '?')} Entities extrahiert")
    logger.info(f"  {meta.get('unique_entities', '?')} unique Entities")
    logger.info(f"  {meta.get('total_relations', '?')} Relationen extrahiert")
    logger.info(f"  Device: {meta.get('device', '?')}")
    logger.info(f"  NER: {meta.get('ner_time_seconds', '?')}s, RE: {meta.get('re_time_seconds', '?')}s")

    return data


def chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    """Konvertiere chunks_export.json-Format zu LangChain Documents."""
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        documents.append(doc)
    return documents


# ============================================================================
# PHASE 3a: VECTOR STORE INGESTION (Embeddings via Ollama)
# ============================================================================

def ingest_vector_store(
    documents: List[Document],
    vector_path: Path,
    config: Dict,
    dataset_name: str,
) -> None:
    """
    Ingest chunks into the LanceDB vector store.

    Verwendet BatchedOllamaEmbeddings für Embedding-Generierung.
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage-Module nicht verfügbar!")
        return

    logger.info(f"\n{'─'*70}")
    logger.info(f"PHASE 3a: VECTOR STORE INGESTION ({len(documents)} Chunks)")
    logger.info(f"{'─'*70}")

    embedding_config = config.get("embeddings", {})
    perf_config = config.get("performance", {})

    # Embedding-Cache
    cache_path = Path(f"./cache/{dataset_name}_embeddings.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=perf_config.get("batch_size", 32),
        cache_path=cache_path,
        device=perf_config.get("device", "cpu"),
    )

    # StorageConfig OHNE Entity Extraction (wir machen das separat)
    vector_config = config.get("vector_store", {})
    storage_config = StorageConfig(
        vector_db_path=vector_path,
        graph_db_path=vector_path.parent / "graph",  # wird hier nicht genutzt
        embedding_dim=embedding_config.get("embedding_dim", 768),
        similarity_threshold=vector_config.get("similarity_threshold", 0.3),
        normalize_embeddings=vector_config.get("normalize_embeddings", True),
        distance_metric=vector_config.get("distance_metric", "cosine"),
        enable_entity_extraction=False,  # DEAKTIVIERT — kommt aus Colab
    )

    # Nur VectorStoreAdapter verwenden
    from src.data_layer import VectorStoreAdapter
    vector_store = VectorStoreAdapter(
        db_path=str(vector_path),
        embedding_dim=storage_config.embedding_dim,
        distance_metric=storage_config.distance_metric,
    )

    # Batch-Ingestion
    start_time = time.time()
    batch_size = 100

    for i in tqdm(
        range(0, len(documents), batch_size),
        desc="Vector Store",
        unit="batch",
    ):
        batch = documents[i : i + batch_size]
        vector_store.add_documents_with_embeddings(batch, embeddings)

    elapsed = time.time() - start_time
    logger.info(f"  ✓ Vector Store: {len(documents)} Chunks in {elapsed:.1f}s")
    logger.info(f"    Pfad: {vector_path}")


# ============================================================================
# PHASE 3b: KNOWLEDGE GRAPH INGESTION (KuzuDB)
# ============================================================================

def ingest_knowledge_graph(
    documents: List[Document],
    extraction_results: List[Dict],
    graph_path: Path,
    dataset_name: str,
    entity_confidence_threshold: float = 0.5,
) -> Tuple[Optional["KuzuGraphStore"], Dict[str, Any]]:
    """
    Importiere Entities und Relationen in KuzuDB Knowledge Graph.

    Steps:
        1. DocumentChunk-Nodes erstellen
        2. SourceDocument-Nodes erstellen
        3. FROM_SOURCE + NEXT_CHUNK Relationen
        4. Entity nodes from extraction results
        5. MENTIONS Relationen (Chunk → Entity)
        6. RELATED_TO Relationen (Entity → Entity)
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage modules not available!")
        return None, {}

    logger.info(f"\n{'─'*70}")
    logger.info("PHASE 3b: KNOWLEDGE GRAPH INGESTION")
    logger.info(f"{'─'*70}")

    # KuzuDB Graph Store initialisieren
    graph_store = KuzuGraphStore(str(graph_path))

    stats = {
        "document_chunks": 0,
        "source_documents": 0,
        "from_source": 0,
        "next_chunk": 0,
        "entities": 0,
        "unique_entities": 0,
        "mentions": 0,
        "relations": 0,
    }

    # Index: chunk_id → extraction result
    extraction_by_chunk = {}
    for result in extraction_results:
        extraction_by_chunk[str(result["chunk_id"])] = result

    # ── Step 1-3: Document Structure ─────────────────────────────────────

    logger.info("  Steps 1-3: building document structure...")
    seen_sources = set()
    prev_chunk_id = None

    def _to_int(value, default: int = 0) -> int:
        """Coerce a metadata value (int OR str, since Phase 1 stringifies)."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    for doc in tqdm(documents, desc="Graph Nodes", unit="doc"):
        chunk_id = str(doc.metadata.get("chunk_id", "unknown"))
        source_file = doc.metadata.get("source_file", "unknown")

        # Phase 1 writes the chunk's order-within-document under "position";
        # the storage schema calls the same field "chunk_index". Map either
        # name to the schema column. All Phase-1 metadata values are strings
        # (see chunks_export serialiser), so coerce.
        chunk_index = _to_int(
            doc.metadata.get("chunk_index", doc.metadata.get("position", 0))
        )
        page_number = _to_int(doc.metadata.get("page_number", 0))

        # DocumentChunk Node — pass the FULL text; the storage layer's
        # max_text_chars=500 default applies the truncation.
        try:
            graph_store.add_document_chunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                page_number=page_number,
                chunk_index=chunk_index,
                source_file=source_file,
            )
            stats["document_chunks"] += 1
        except Exception as e:
            logger.debug(f"    Chunk {chunk_id}: {e}")

        # SourceDocument Node (nur einmal pro source)
        if source_file not in seen_sources:
            try:
                graph_store.add_source_document(
                    doc_id=source_file,
                    filename=source_file,
                    total_pages=int(doc.metadata.get("total_pages", 0)),
                )
                seen_sources.add(source_file)
                stats["source_documents"] += 1
            except Exception as e:
                logger.debug(f"    Source {source_file}: {e}")

        # FROM_SOURCE Relation
        try:
            graph_store.add_from_source_relation(chunk_id, source_file)
            stats["from_source"] += 1
        except Exception as e:
            logger.debug(f"    FROM_SOURCE: {e}")

        # NEXT_CHUNK Relation (nur innerhalb desselben Source-Dokuments)
        if prev_chunk_id is not None:
            prev_doc = documents[stats["document_chunks"] - 2] if stats["document_chunks"] > 1 else None
            if prev_doc and prev_doc.metadata.get("source_file") == source_file:
                try:
                    graph_store.add_next_chunk_relation(prev_chunk_id, chunk_id)
                    stats["next_chunk"] += 1
                except Exception:
                    pass

        prev_chunk_id = chunk_id

    logger.info(f"    ✓ {stats['document_chunks']} chunks, {stats['source_documents']} sources")

    # ── Step 4-6: Entities & Relations ───────────────────────────────────

    logger.info("  Steps 4-6: importing entities and relations...")

    seen_entities: set = set()
    # Map: canonical_form(name) -> canonical entity_id (used for relation
    # resolution AND co-occurrence edge construction). One key per canonical
    # surface form; collisions are intentional and produce the deduplication.
    name_to_id: Dict[str, str] = {}

    # Build a lookup: chunk_id -> raw text (used for SVO extraction below).
    chunk_id_to_text: Dict[str, str] = {
        str(doc.metadata.get("chunk_id", f"chunk_{i}")): doc.page_content
        for i, doc in enumerate(documents)
    }

    for result in tqdm(extraction_results, desc="Entities & Relations", unit="chunk"):
        chunk_id = str(result["chunk_id"])

        # Step 4: Entity Nodes
        for ent in result.get("entities", []):
            entity_name = (ent.get("name") or "").strip()
            entity_type = ent.get("entity_type") or ent.get("type", "UNKNOWN")
            confidence = ent.get("confidence", 0.5)

            # Skip low-confidence entities (recall-vs-noise trade-off).
            if confidence < entity_confidence_threshold or not entity_name:
                continue

            # Canonical entity_id: identical canonical surface forms produce
            # the same id, so KuzuDB MERGE deduplicates at insert time.
            entity_id = _canonical_entity_id(entity_name, entity_type)
            canon_key = canonical_form(entity_name)
            name_to_id[canon_key] = entity_id

            if entity_id not in seen_entities:
                try:
                    graph_store.add_entity(
                        entity_id=entity_id,
                        name=entity_name,
                        entity_type=entity_type,
                        confidence=confidence,
                    )
                    seen_entities.add(entity_id)
                    stats["unique_entities"] += 1
                except Exception as e:
                    logger.debug(f"    Entity {entity_id}: {e}")

            stats["entities"] += 1

            # Step 5: MENTIONS Relation (Chunk -> Entity)
            try:
                graph_store.add_mentions_relation(
                    chunk_id=chunk_id,
                    entity_id=entity_id,
                )
                stats["mentions"] += 1
            except Exception as e:
                logger.debug(f"    MENTIONS {chunk_id}->{entity_id}: {e}")

        # Step 6: RELATED_TO Relations (Entity -> Entity)
        # Lookup is strict on canonical_form; the previous substring fallback
        # was lossy ("Local H" -> "Localeur") and is intentionally removed.
        # Unresolved relations are dropped with a debug log.
        # REBEL confidence and source chunks are persisted on the edge so
        # downstream retrieval can filter low-confidence relations.
        for rel in result.get("relations", []):
            subject = (rel.get("subject_entity") or rel.get("subject") or "").strip()
            obj = (rel.get("object_entity") or rel.get("object") or "").strip()
            rel_type = rel.get("relation_type") or rel.get("relation") or "related_to"
            rel_conf = float(rel.get("confidence", 0.5))
            rel_sources = rel.get("source_chunk_ids") or rel.get("source_chunks") or [chunk_id]
            if isinstance(rel_sources, str):
                rel_sources = [rel_sources]
            if not subject or not obj:
                continue

            subject_id = name_to_id.get(canonical_form(subject))
            object_id = name_to_id.get(canonical_form(obj))

            if subject_id and object_id and subject_id != object_id:
                try:
                    graph_store.add_related_to_relation(
                        entity1_id=subject_id,
                        entity2_id=object_id,
                        relation_type=rel_type,
                        confidence=rel_conf,
                        source_chunks=[str(c) for c in rel_sources],
                    )
                    stats["relations"] += 1
                except Exception as e:
                    logger.debug(f"    RELATED_TO: {e}")
            else:
                logger.debug(
                    "    RELATED_TO unresolved: subject=%r object=%r",
                    subject, obj,
                )

    logger.info(
        "    OK %d entities, %d mentions, %d REBEL relations",
        stats["unique_entities"], stats["mentions"], stats["relations"],
    )

    # ── SVO extraction: narrative relations from the SpaCy dependency parse ──
    # Adds RELATED_TO edges with relation_type=verb_lemma and confidence=0.7.
    # Both subject and object must resolve to a known entity in name_to_id;
    # unmatched triples are dropped silently.
    stats["svo_relations"] = 0
    if svo_available():
        svo_added = 0
        for result in tqdm(extraction_results, desc="SVO relations", unit="chunk"):
            chunk_id = str(result.get("chunk_id", ""))
            text = chunk_id_to_text.get(chunk_id, "")
            if not text:
                continue
            triples = extract_svo_relations(
                text=text,
                name_to_id=name_to_id,
                chunk_id=chunk_id,
                canonical_form_fn=canonical_form,
            )
            for subj_id, verb_lemma, obj_id, conf in triples:
                try:
                    graph_store.add_related_to_relation(
                        entity1_id=subj_id,
                        entity2_id=obj_id,
                        relation_type=verb_lemma,
                        confidence=conf,
                        source_chunks=[chunk_id],
                    )
                    svo_added += 1
                except Exception as exc:
                    logger.debug("SVO RELATED_TO failed: %s", exc)
        stats["svo_relations"] = svo_added
        logger.info("    OK %d SVO narrative relations", svo_added)
    else:
        logger.info("    SVO extraction unavailable (spaCy missing); skipped")

    # Expose the canonical-id map to the caller so co-occurrence edges and
    # cleanup can use the same lookup convention.
    stats["_name_to_id"] = name_to_id
    return graph_store, stats


# ============================================================================
# FULL IMPORT PIPELINE
# ============================================================================

def run_full_import(
    chunks_path: Path,
    extractions_path: Path,
    dataset_name: str,
    config: Dict,
    graph_only: bool = False,
    clear: bool = False,
    skip_cleanup: bool = False,
    skip_cooccurrence: bool = False,
    cleanup_dry_run: bool = False,
    cooccurrence_min_confidence: float = 0.5,
    hub_threshold_ratio: float = 0.03,
    enable_entity_linking: bool = True,
    linking_threshold: float = 0.92,
) -> None:
    """
    Run the full Phase-3 import (vector store + knowledge graph + cleanup).

    Args:
        chunks_path:                  Path to chunks_export.json (Phase 1).
        extractions_path:             Path to extraction_results.json (Phase 2).
        dataset_name:                 Dataset name (e.g. "hotpotqa").
        config:                       Settings dict (from settings.yaml or defaults).
        graph_only:                   Skip the vector-store ingestion.
        clear:                        Delete existing stores before import.
        skip_cleanup:                 Disable the post-ingestion cleanup pass.
        skip_cooccurrence:            Disable co-occurrence edge construction.
        cleanup_dry_run:              Cleanup pass counts operations only.
        cooccurrence_min_confidence:  Minimum NER confidence required for an
                                      entity to participate in co-occurrence.
        hub_threshold_ratio:          Drop entities mentioned in more than
                                      ratio * total_chunks chunks.
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage modules not available - aborting")
        sys.exit(1)

    total_start = time.time()

    print()
    print("═" * 70)
    print("DECOUPLED INGESTION  -  PHASE 3: LOCAL IMPORT")
    print("═" * 70)
    print(f"  Dataset:      {dataset_name}")
    print(f"  Chunks:       {chunks_path}")
    print(f"  Extractions:  {extractions_path}")
    print(f"  Graph only:   {graph_only}")
    print(f"  Cleanup:      {'off' if skip_cleanup else ('dry-run' if cleanup_dry_run else 'on')}")
    print(f"  Co-occur:     {'off' if skip_cooccurrence else 'on'} "
          f"(conf >= {cooccurrence_min_confidence})")
    print("═" * 70)

    # Paths
    base_path = Path("./data") / dataset_name
    vector_path = base_path / "vector"
    graph_path = base_path / "graph"

    # Clear if requested.
    # IMPORTANT: extraction_results.json and chunks_export.json are NEVER
    # deleted - they are source artifacts (Phase 1 / Colab output) and must
    # be preserved. Only the derived stores (vector, KuzuDB graph files) are
    # removed.
    if clear:
        import shutil
        if graph_only:
            targets = [graph_path]
        else:
            targets = [vector_path, graph_path]
        for target in targets:
            if not target.exists():
                continue
            # ── SAFETY: rescue every .json file before deletion ───────────────
            # JSON files are source artifacts (extraction_results.json,
            # chunks_export.json) or derived metadata - never regenerated by
            # external GPU jobs. Delete only database files, never JSON.
            rescued: dict[str, bytes] = {}
            for json_file in target.rglob("*.json"):
                rescued[json_file.name] = json_file.read_bytes()
                logger.info(f"  Protected before --clear: {json_file.name}")
            # ── DELETE database directory ─────────────────────────────────────
            shutil.rmtree(target)
            logger.info(f"  Cleared: {target}")
            # ── RESTORE rescued JSON files ────────────────────────────────────
            if rescued:
                target.mkdir(parents=True, exist_ok=True)
                for fname, data in rescued.items():
                    restored = target / fname
                    restored.write_bytes(data)
                    logger.info(f"  Restored: {restored}")

    base_path.mkdir(parents=True, exist_ok=True)

    # Load source data.
    chunks = load_chunks(chunks_path)
    extraction_data = load_extractions(extractions_path)
    extraction_results = extraction_data.get("results", [])

    # Validation: chunk count should match extraction count.
    if len(chunks) != len(extraction_results):
        logger.warning(
            "  WARNING: chunk count mismatch (chunks=%d, extractions=%d)",
            len(chunks), len(extraction_results),
        )
        logger.warning("  Continuing anyway (using chunk_id intersection).")

    documents = chunks_to_documents(chunks)

    # Phase 3a: Vector Store
    if not graph_only:
        try:
            ingest_vector_store(documents, vector_path, config, dataset_name)
        except Exception as e:
            logger.error(f"Vector store ingestion failed: {e}")
            logger.error("Use --graph-only when the vector store is built separately.")
            raise
    else:
        logger.info("  Vector store skipped (--graph-only)")

    # Phase 3b: Knowledge Graph
    try:
        entity_conf = (
            config.get("entity_extraction", {})
                  .get("gliner", {})
                  .get("confidence_threshold", 0.5)
        )
        graph_store, stats = ingest_knowledge_graph(
            documents, extraction_results, graph_path, dataset_name,
            entity_confidence_threshold=entity_conf,
        )
    except Exception as e:
        logger.error(f"Knowledge graph ingestion failed: {e}")
        raise

    name_to_id = stats.pop("_name_to_id", {})

    # Phase 3c: Co-occurrence edges (every pair of entities co-mentioned in
    # the same chunk gets a RELATED_TO {relation_type='cooccurs'} edge).
    cooccurrence_edges = 0
    if graph_store is not None and not skip_cooccurrence:
        logger.info(f"\n{'─'*70}")
        logger.info("PHASE 3c: CO-OCCURRENCE EDGES")
        logger.info(f"{'─'*70}")
        try:
            cooccurrence_edges = build_cooccurrence_edges(
                graph_store=graph_store,
                extraction_results=extraction_results,
                name_to_id=name_to_id,
                min_confidence=cooccurrence_min_confidence,
                relation_type="cooccurs",
            )
            logger.info("  OK %d co-occurrence edges added", cooccurrence_edges)
        except Exception as e:
            logger.error("Co-occurrence edge construction failed: %s", e)

    # Phase 3d: Cleanup pass (orphans + hubs + duplicate merge).
    cleanup_ops = {
        "orphans_dropped": 0,
        "hubs_dropped": 0,
        "duplicates_merged": 0,
        "stoplist_dropped": 0,
    }
    if graph_store is not None and not skip_cleanup:
        logger.info(f"\n{'─'*70}")
        logger.info(
            "PHASE 3d: GRAPH CLEANUP%s",
            " (DRY RUN)" if cleanup_dry_run else "",
        )
        logger.info(f"{'─'*70}")
        try:
            cleanup_ops = cleanup_graph(
                graph_store=graph_store,
                drop_orphans=True,
                hub_threshold_ratio=hub_threshold_ratio,
                merge_duplicates=True,
                dry_run=cleanup_dry_run,
            )
            logger.info(
                "  Stop-list dropped:  %d (pronouns, nationality adjectives)",
                cleanup_ops["stoplist_dropped"],
            )
            logger.info(
                "  Orphans dropped:    %d",
                cleanup_ops["orphans_dropped"],
            )
            logger.info(
                "  Hubs dropped:       %d (threshold ratio = %.1f%% of chunks)",
                cleanup_ops["hubs_dropped"], hub_threshold_ratio * 100,
            )
            logger.info(
                "  Duplicates merged:  %d",
                cleanup_ops["duplicates_merged"],
            )
        except Exception as e:
            logger.error("Cleanup pass failed: %s", e)

    # Phase 3d.5: Embedding-based entity linking (alias resolution beyond
    # canonical_form). Merges "Ed Wood" with "Edward Davis Wood Jr." etc.
    # by clustering name embeddings within each canonical type bucket.
    linked_count = 0
    if graph_store is not None and enable_entity_linking and not skip_cleanup:
        logger.info(f"\n{'─'*70}")
        logger.info("PHASE 3d.5: EMBEDDING-BASED ENTITY LINKING")
        logger.info(f"{'─'*70}")
        try:
            embedding_config = config.get("embeddings", {})
            perf_config = config.get("performance", {})
            cache_path = Path(f"./cache/{dataset_name}_embeddings.db")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            embedder = BatchedOllamaEmbeddings(
                model_name=embedding_config.get("model_name", "nomic-embed-text"),
                base_url=embedding_config.get("base_url", "http://localhost:11434"),
                batch_size=perf_config.get("batch_size", 64),
                cache_path=cache_path,
                device=perf_config.get("device", "cpu"),
            )
            linked_count = link_entities_by_embedding(
                graph_store=graph_store,
                embedder=embedder,
                similarity_threshold=linking_threshold,
                dry_run=cleanup_dry_run,
            )
            logger.info(
                "  Embedding-linked entities merged: %d (threshold=%.2f)",
                linked_count, linking_threshold,
            )
        except Exception as exc:
            logger.error("Entity linking failed: %s", exc)

    # Phase 3e: Baseline metrics + invariant assertions.
    baseline: Dict[str, Any] = {}
    violations: List[str] = []
    if graph_store is not None:
        try:
            baseline = compute_graph_baseline(graph_store)
            print()
            print(format_baseline_report(baseline))
            violations = assert_graph_invariants(baseline, strict=False)
            if violations:
                print()
                print("  INVARIANT VIOLATIONS (warning, not fatal):")
                for v in violations:
                    print(f"    - {v}")
        except Exception as e:
            logger.error("Baseline computation failed: %s", e)

    # Summary
    total_elapsed = time.time() - total_start

    print()
    print("═" * 70)
    print("IMPORT COMPLETE")
    print("═" * 70)
    print(f"  Total time:       {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Vector store:     {vector_path}")
    print(f"  Knowledge graph:  {graph_path}")
    print()
    print("  Ingestion stats:")
    for key, val in stats.items():
        print(f"    {key:<22}: {val:>10,}")
    print(f"    {'cooccurrence_edges':<22}: {cooccurrence_edges:>10,}")
    print(f"    {'svo_relations':<22}: {stats.get('svo_relations', 0):>10,}")
    print(f"    {'stoplist_dropped':<22}: {cleanup_ops['stoplist_dropped']:>10,}")
    print(f"    {'orphans_dropped':<22}: {cleanup_ops['orphans_dropped']:>10,}")
    print(f"    {'hubs_dropped':<22}: {cleanup_ops['hubs_dropped']:>10,}")
    print(f"    {'duplicates_merged':<22}: {cleanup_ops['duplicates_merged']:>10,}")
    print(f"    {'embedding_linked':<22}: {linked_count:>10,}")
    print()
    print("═" * 70)
    print("  Next steps:")
    print(f"    python benchmark_datasets.py evaluate --dataset {dataset_name} --samples 100")
    print(f"    python benchmark_datasets.py ablation --dataset {dataset_name} --samples 100")
    print(f"    python -X utf8 diagnose_graph_baseline.py --dataset {dataset_name}")
    print("═" * 70)

    # Persist extraction + import metadata.
    meta_path = base_path / "graph" / "extraction_metadata.json"
    meta = extraction_data.get("metadata", {})
    meta["import_time_seconds"] = round(total_elapsed, 1)
    meta["graph_stats"] = stats
    meta["cleanup_ops"] = cleanup_ops
    meta["cooccurrence_edges"] = cooccurrence_edges
    if baseline:
        # Strip top_clusters because they hold raw entity_id/name tuples that
        # bloat the metadata file. Keep the summary numbers.
        meta["graph_baseline"] = {
            "totals": baseline["totals"],
            "densities": baseline["densities"],
            "isolated": baseline["isolated"],
            "duplicates": {
                k: v for k, v in baseline["duplicates"].items()
                if k != "top_clusters"
            },
            "violations": violations,
        }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"  Metadata written: {meta_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decoupled ingestion - Phase 3: local import",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard import (vector store + knowledge graph + cleanup)
  python local_importingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions data/hotpotqa/graph/extraction_results.json \\
      --dataset hotpotqa

  # Knowledge graph only (vector store already exists)
  python local_importingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions data/hotpotqa/graph/extraction_results.json \\
      --dataset hotpotqa \\
      --graph-only

  # With explicit YAML config
  python local_importingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions data/hotpotqa/graph/extraction_results.json \\
      --dataset hotpotqa \\
      --config config/settings.yaml

  # Preview cleanup without mutating the graph
  python local_importingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions data/hotpotqa/graph/extraction_results.json \\
      --dataset hotpotqa \\
      --cleanup-dry-run
        """,
    )

    parser.add_argument(
        "--chunks", "-c",
        type=Path,
        required=True,
        help="Path to chunks_export.json (Phase 1 output)",
    )
    parser.add_argument(
        "--extractions", "-e",
        type=Path,
        required=True,
        help="Path to extraction_results.json (Phase 2 / Colab output)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Dataset name (e.g. hotpotqa, 2wikimultihop)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to settings.yaml (optional)",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Import only the knowledge graph; skip the vector store",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing stores before import",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable the post-ingestion cleanup pass (orphans, hubs, duplicate merge)",
    )
    parser.add_argument(
        "--no-cooccurrence",
        action="store_true",
        help="Disable co-occurrence edge construction (RELATED_TO {cooccurs})",
    )
    parser.add_argument(
        "--cleanup-dry-run",
        action="store_true",
        help="Run the cleanup pass in dry-run mode (count operations, no mutation)",
    )
    parser.add_argument(
        "--cooccurrence-min-confidence",
        type=float,
        default=0.5,
        help="Minimum NER confidence required for an entity to participate in "
             "co-occurrence edges (default: 0.5)",
    )
    parser.add_argument(
        "--hub-threshold-ratio",
        type=float,
        default=0.03,
        help="Drop entities mentioned in more than ratio*total_chunks chunks "
             "(default: 0.03 = 3 %% of the corpus). Lower than the literature "
             "default because GLiNER misclassifications such as 'He' (4.1 %%) "
             "and 'England' (1.7 %%) need to be reachable.",
    )
    parser.add_argument(
        "--no-entity-linking",
        action="store_true",
        help="Disable embedding-based entity linking (alias resolution beyond "
             "canonical_form, e.g. 'Ed Wood' <-> 'Edward Davis Wood Jr.')",
    )
    parser.add_argument(
        "--linking-threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold for embedding-based entity linking "
             "(default: 0.92). Higher = stricter merging. Range [0.85, 0.97].",
    )

    args = parser.parse_args()

    # Validation
    if not args.chunks.exists():
        logger.error(f"Chunks file not found: {args.chunks}")
        sys.exit(1)
    if not args.extractions.exists():
        logger.error(f"Extractions file not found: {args.extractions}")
        sys.exit(1)

    config = load_config(args.config)

    run_full_import(
        chunks_path=args.chunks,
        extractions_path=args.extractions,
        dataset_name=args.dataset,
        config=config,
        graph_only=args.graph_only,
        clear=args.clear,
        skip_cleanup=args.no_cleanup,
        skip_cooccurrence=args.no_cooccurrence,
        cleanup_dry_run=args.cleanup_dry_run,
        cooccurrence_min_confidence=args.cooccurrence_min_confidence,
        hub_threshold_ratio=args.hub_threshold_ratio,
        enable_entity_linking=not args.no_entity_linking,
        linking_threshold=args.linking_threshold,
    )


if __name__ == "__main__":
    main()