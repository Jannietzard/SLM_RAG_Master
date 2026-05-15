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

    # Resume after crash — skip phases already done in checkpoint
    python local_importingestion.py \\
        --chunks data/hotpotqa/chunks_export.json \\
        --extractions data/hotpotqa/graph/extraction_results.json \\
        --dataset hotpotqa \\
        --graph-only --resume

    # Re-run only Phase 3d.5 (entity linking) after clearing the 3d5 checkpoint entry,
    # with a raised max-type-size to process large buckets (e.g. PERSON):
    python local_importingestion.py \\
        --chunks data/hotpotqa/chunks_export.json \\
        --extractions data/hotpotqa/graph/extraction_results.json \\
        --dataset hotpotqa \\
        --graph-only --resume --no-cooccurrence --no-cleanup \\
        --linking-max-type-size 20000

Available flags:
    REQUIRED
      --chunks PATH               Path to chunks_export.json (Phase 1 output)
      --extractions PATH          Path to extraction_results.json (Phase 2 output)
      --dataset / -d NAME         Dataset name (e.g. hotpotqa)

    OPTIONAL
      --config PATH               Path to settings.yaml (default: config/settings.yaml)
      --graph-only                Skip vector store (LanceDB); import KG only
      --clear                     Delete existing stores before import (full re-run)
      --resume                    Skip phases already marked done in checkpoint;
                                  mutually exclusive with --clear

    CLEANUP / CO-OCCURRENCE
      --no-cleanup                Disable orphan/hub/duplicate cleanup pass (Phase 3d)
      --no-cooccurrence           Disable RELATED_TO co-occurrence edges (Phase 3c)
      --no-subsumptive-cleanup    Disable Phase 3c.5 — keep cooccurs edges even
                                  when a semantic edge already covers the pair
      --no-isolated-drop          Disable Phase 3f — keep entities with zero
                                  RELATED_TO edges after Phase 3d.5 linking
      --cleanup-dry-run           Dry-run cleanup — print mutations but do not apply
      --cooccurrence-min-confidence FLOAT
                                  Minimum confidence for co-occurrence edges (default: 0.5)
      --hub-threshold-ratio FLOAT Hub suppression ratio (default: 0.03)

    ENTITY LINKING (Phase 3d.5)
      --no-entity-linking         Disable embedding-based alias resolution entirely
      --linking-threshold FLOAT   Cosine similarity for alias merging (default: 0.92)
      --linking-max-type-size INT Max entity-bucket size before the bucket is skipped
                                  to avoid OOM (default: 8000; raise to 20000+ for
                                  large types like PERSON — requires ~1.6 GB RAM)

    EMBEDDINGS
      --embeddings-backend {ollama,huggingface}
                                  Embedding backend (default: ollama)

    MISC
      --attribute-objects         Capture REBEL attribute values as CONCEPT nodes
                                  (default off — keeps OntoNotes-5 taxonomy)

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
from typing import List, Dict, Any, Optional, Set, Tuple
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
    from sentence_transformers import SentenceTransformer as _ST
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    from src.data_layer import HybridStore, StorageConfig, KuzuGraphStore
    from src.data_layer import BatchedOllamaEmbeddings
    from src.data_layer.graph_quality import (
        assert_graph_invariants,
        build_cooccurrence_edges,
        canonical_form,
        cleanup_graph,
        compute_graph_baseline,
        drop_isolated_entities,
        drop_subsumed_cooccurrence_edges,
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
# PHASE CHECKPOINTING
# ============================================================================
# After each phase completes successfully, we write a small JSON file to disk.
# On re-run with --resume, phases that already have "done": true are skipped.
# This means a crash (power cut, Ctrl-C, etc.) after Phase 3b doesn't require
# re-running the full 45-minute entity import — you just resume from 3c.
#
# Checkpoint file location:  data/<dataset>/graph/.import_checkpoint.json
# name_to_id cache:          data/<dataset>/graph/.name_to_id.json
#
# The checkpoint is deleted by --clear.

_CP_DONE = "done"
_CP_TS   = "ts"

def _checkpoint_path(graph_path: Path) -> Path:
    return graph_path / ".import_checkpoint.json"

def _name_to_id_path(graph_path: Path) -> Path:
    return graph_path / ".name_to_id.json"

def _load_checkpoint(graph_path: Path) -> Dict[str, Any]:
    """Load existing checkpoint or return empty dict."""
    p = _checkpoint_path(graph_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_checkpoint(graph_path: Path, phase: str, data: Dict[str, Any]) -> None:
    """Mark a phase as done and persist phase-level stats to the checkpoint."""
    cp = _load_checkpoint(graph_path)
    cp[phase] = {_CP_DONE: True, _CP_TS: time.time(), **data}
    _checkpoint_path(graph_path).write_text(
        json.dumps(cp, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("  Checkpoint saved: phase=%s", phase)

def _phase_done(checkpoint: Dict[str, Any], phase: str) -> bool:
    return checkpoint.get(phase, {}).get(_CP_DONE, False)

def _phase_eta(label: str, elapsed_s: float) -> str:
    """Format a phase timing line for the console."""
    m, s = divmod(int(elapsed_s), 60)
    return f"  [{label}]  elapsed: {m}m {s:02d}s"


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


# Relation-object surface forms that are *never* concept nodes even when they
# fail to resolve to a named entity — single letters / digits ("R" for
# Republican, "D" for Democrat), bare punctuation, the empty string. Keeping
# these out of the graph prevents a "CONCEPT: r" hub that links every American
# politician chunk. This is a structural filter (length / character class),
# NOT a curated stop-list of seen values.
def _is_plausible_concept(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return False
    # Must contain at least one alphabetic character.
    if not any(ch.isalpha() for ch in t):
        return False
    return True


# Synthetic, lower-cased "concept" type for REBEL relation objects that are
# attribute values rather than named entities ("genre -> punk rock",
# "sport -> ice hockey", "occupation -> DJ"). Tagged distinctly from the
# GLiNER entity types (PERSON, GPE, ...) so graph cleanup and retrieval can
# treat them differently if needed. They still receive MENTIONS edges from
# the chunk that produced the relation, so they are reachable.
_CONCEPT_ENTITY_TYPE = "CONCEPT"

class _HFEmbeddings:
    """Thin wrapper so SentenceTransformer has the same interface as BatchedOllamaEmbeddings."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1", batch_size: int = 64):
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info("  Loading HuggingFace model: %s", model_name)
        self._model = _ST(model_name, trust_remote_code=True)
        self._batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self._model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0].tolist()


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
    embeddings=None,
) -> None:
    """
    Ingest chunks into the LanceDB vector store.

    Uses BatchedOllamaEmbeddings by default; pass a pre-built embeddings
    object (e.g. _HFEmbeddings) to switch backends.
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage-Module nicht verfügbar!")
        return

    logger.info(f"\n{'─'*70}")
    logger.info(f"PHASE 3a: VECTOR STORE INGESTION ({len(documents)} Chunks)")
    logger.info(f"{'─'*70}")

    embedding_config = config.get("embeddings", {})
    if embeddings is None:
        perf_config = config.get("performance", {})
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
    capture_attribute_objects: bool = False,
) -> Tuple[Optional["KuzuGraphStore"], Dict[str, Any]]:
    """
    Importiere Entities und Relationen in KuzuDB Knowledge Graph.

    Steps:
        1. DocumentChunk-Nodes erstellen
        2. SourceDocument-Nodes erstellen
        3. FROM_SOURCE + NEXT_CHUNK Relationen
        4. Entity nodes from extraction results
        5. MENTIONS Relationen (Chunk → Entity)
        6. RELATED_TO Relationen (Entity → Entity)  — second pass, after the
           full `name_to_id` map is built, so a relation in chunk N can be
           resolved against an entity first seen in chunk N+k.

    Args:
        capture_attribute_objects: When True, a REBEL relation whose object is
            a free-text attribute value rather than a named entity
            ("genre -> punk rock", "sport -> ice hockey") still produces a
            graph edge — the object is materialised as a synthetic CONCEPT-typed
            Entity node. Default False: such relations are dropped, keeping the
            graph within the OntoNotes-5 entity-type taxonomy (PERSON, GPE,
            ORGANIZATION, LOCATION, DATE, EVENT, WORK_OF_ART, PRODUCT) used by
            GLiNER. Enable with --attribute-objects only after considering the
            impact on co-occurrence edge quality and hub detection.
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
        "rebel_duplicates_skipped": 0,
        "rebel_unresolved_dropped": 0,
        "attribute_relations": 0,
        "concept_nodes": 0,
        "rebel_confidence_is_constant": False,
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

    # ── Batch-transaction constants for Phase 3b ─────────────────────────────
    # KuzuDB auto-commits every conn.execute() call individually (one fsync
    # per statement). For ~94 000 entity inserts + 94 000 MENTIONS + 12 000
    # REBEL relations = ~200 000 individual fsyncs → 45 min on Windows.
    # Grouping 200 chunks per transaction (typically ~2 000 inserts) cuts
    # the fsync count by 200× and brings Phase 3b down to ~3-5 min.
    _PHASE3B_BATCH = 200   # chunks per transaction commit

    # ── Pass A: Entity nodes + MENTIONS edges ────────────────────────────────
    # Build the COMPLETE name_to_id map before touching relations. Relations
    # are resolved in Pass B, so a relation in chunk N can reference an entity
    # first extracted in chunk N+k (forward reference) — previously those were
    # silently dropped because the map was only partially populated.
    for _batch_i, result in enumerate(
        tqdm(extraction_results, desc="Entities & MENTIONS", unit="chunk")
    ):
        if _batch_i % _PHASE3B_BATCH == 0:
            if _batch_i > 0:
                try:
                    graph_store.batch_commit()
                except Exception:
                    pass
            graph_store.batch_begin()

        chunk_id = str(result["chunk_id"])

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

            try:
                graph_store.add_mentions_relation(
                    chunk_id=chunk_id,
                    entity_id=entity_id,
                )
                stats["mentions"] += 1
            except Exception as e:
                logger.debug(f"    MENTIONS {chunk_id}->{entity_id}: {e}")

    try:
        graph_store.batch_commit()
    except Exception:
        pass

    # ── Pass B: RELATED_TO edges (REBEL relations) ───────────────────────────
    # Resolution is strict on canonical_form (no lossy substring fallback).
    # Two-pass means `name_to_id` is now complete, so both endpoints can be a
    # named entity from ANY chunk. A relation whose object resolves to no named
    # entity is treated as a free-text attribute value and (when
    # capture_attribute_objects=True) materialised as a CONCEPT node, so the
    # genre/sport/occupation knowledge is not lost. Within-run triple
    # deduplication keeps the relation count honest (REBEL beam search emits
    # the same triple multiple times — KuzuDB MERGE collapses them, but the
    # counter must not double-count).
    seen_triples: Set[Tuple[str, str, str]] = set()
    _rel_confs: Set[float] = set()
    _concept_ids: Dict[str, str] = {}  # canonical_form(value) -> concept entity_id
    _concept_mentions: Set[Tuple[str, str]] = set()  # (chunk_id, concept_id) already linked

    def _ensure_concept_node(value: str, src_chunk: str) -> Optional[str]:
        """MERGE a CONCEPT entity for a free-text relation object; link it from src_chunk."""
        if not _is_plausible_concept(value):
            return None
        key = canonical_form(value)
        cid = _concept_ids.get(key)
        if cid is None:
            cid = _canonical_entity_id(value, _CONCEPT_ENTITY_TYPE)
            _concept_ids[key] = cid
            try:
                graph_store.add_entity(
                    entity_id=cid,
                    name=value,
                    entity_type=_CONCEPT_ENTITY_TYPE,
                    confidence=0.5,
                )
                stats["concept_nodes"] += 1
            except Exception as e:
                logger.debug(f"    CONCEPT node {value!r}: {e}")
                return None
        # Make the concept reachable: MENTIONS edge from the originating chunk
        # (once per chunk/concept pair — KuzuDB MERGE is idempotent but the
        # counter must not double-count repeated references in the same chunk).
        mkey = (src_chunk, cid)
        if mkey not in _concept_mentions:
            _concept_mentions.add(mkey)
            try:
                graph_store.add_mentions_relation(chunk_id=src_chunk, entity_id=cid)
                stats["mentions"] += 1
            except Exception:
                pass
        # Keep it resolvable by later relations referencing the same value.
        name_to_id.setdefault(key, cid)
        return cid

    _batch_i = 0
    for result in tqdm(extraction_results, desc="RELATED_TO (REBEL)", unit="chunk"):
        if _batch_i % _PHASE3B_BATCH == 0:
            if _batch_i > 0:
                try:
                    graph_store.batch_commit()
                except Exception:
                    pass
            graph_store.batch_begin()
        _batch_i += 1

        chunk_id = str(result["chunk_id"])

        for rel in result.get("relations", []):
            subject = (rel.get("subject_entity") or rel.get("subject") or "").strip()
            obj = (rel.get("object_entity") or rel.get("object") or "").strip()
            rel_type = rel.get("relation_type") or rel.get("relation") or "related_to"
            rel_conf = float(rel.get("confidence", 0.5))
            _rel_confs.add(rel_conf)
            rel_sources = rel.get("source_chunk_ids") or rel.get("source_chunks") or [chunk_id]
            if isinstance(rel_sources, str):
                rel_sources = [rel_sources]
            if not subject or not obj:
                continue

            subject_id = name_to_id.get(canonical_form(subject))
            if not subject_id:
                # A relation whose SUBJECT is not a named entity is almost
                # always a parse artefact ("biographical -> ..."); drop it.
                stats["rebel_unresolved_dropped"] += 1
                continue

            object_id = name_to_id.get(canonical_form(obj))
            is_attribute = False
            if not object_id:
                if capture_attribute_objects:
                    object_id = _ensure_concept_node(obj, chunk_id)
                    is_attribute = object_id is not None
                if not object_id:
                    stats["rebel_unresolved_dropped"] += 1
                    continue

            if subject_id == object_id:
                continue

            triple = (subject_id, rel_type, object_id)
            if triple in seen_triples:
                stats["rebel_duplicates_skipped"] += 1
                continue
            seen_triples.add(triple)

            try:
                graph_store.add_related_to_relation(
                    entity1_id=subject_id,
                    entity2_id=object_id,
                    relation_type=rel_type,
                    confidence=rel_conf,
                    source_chunks=[str(c) for c in rel_sources],
                )
                stats["relations"] += 1
                if is_attribute:
                    stats["attribute_relations"] += 1
            except Exception as e:
                logger.debug(f"    RELATED_TO: {e}")

    try:
        graph_store.batch_commit()
    except Exception:
        pass

    # REBEL emits a constant 0.5 confidence for every triple (no per-triplet
    # score from the seq2seq decoder). Surface this so downstream consumers
    # don't mistake the value for a real ranking signal.
    stats["rebel_confidence_is_constant"] = len(_rel_confs) <= 1
    if stats["rebel_confidence_is_constant"] and _rel_confs:
        logger.warning(
            "    REBEL relation confidence is CONSTANT (%.2f for all %d edges) — "
            "do not filter graph relations on this value; it carries no signal.",
            next(iter(_rel_confs)), stats["relations"],
        )

    logger.info(
        "    OK %d REBEL relations  (+%d attribute/CONCEPT, %d dup skipped, "
        "%d unresolved dropped, %d concept nodes)",
        stats["relations"], stats["attribute_relations"],
        stats["rebel_duplicates_skipped"], stats["rebel_unresolved_dropped"],
        stats["concept_nodes"],
    )
    logger.info(
        "    OK %d entities, %d mentions",
        stats["unique_entities"], stats["mentions"],
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
    skip_subsumptive_cleanup: bool = False,
    skip_isolated_drop: bool = False,
    cleanup_dry_run: bool = False,
    cooccurrence_min_confidence: float = 0.5,
    hub_threshold_ratio: float = 0.03,
    enable_entity_linking: bool = True,
    linking_threshold: float = 0.92,
    linking_max_type_size: int = 8000,
    resume: bool = False,
    embeddings_backend: str = "ollama",
    capture_attribute_objects: bool = False,
) -> None:
    """
    Run the full Phase-3 import (vector store + knowledge graph + cleanup).

    ═══════════════════════════════════════════════════════════════════════
    WHAT DOES THIS FUNCTION DO?
    ═══════════════════════════════════════════════════════════════════════

    This is the main entry point for Phase 3 of the three-phase ingestion
    pipeline.  Phases 1 and 2 ran earlier:

        Phase 1 (benchmark_datasets.py)
            Splits raw Wikipedia/HotpotQA text into overlapping chunks and
            writes chunks_export.json.  Runs on CPU, fast (~1 min).

        Phase 2 (Colab notebook)
            Runs GLiNER (NER) and REBEL (relation extraction) on every chunk
            using a GPU.  Writes extraction_results.json (~2 h on Colab T4).

        Phase 3 (this file) — runs on your local machine
            Imports the GPU outputs into the local databases:

            3a  Vector store   → LanceDB (ANN index for dense retrieval).
                                 Skipped with --graph-only.
            3b  Knowledge graph → KuzuDB.
                                 Sub-steps:
                                   1. DocumentChunk nodes (one per chunk)
                                   2. SourceDocument nodes (one per source file)
                                   3. FROM_SOURCE + NEXT_CHUNK relations
                                   4. Entity nodes from extraction_results.json
                                   5. MENTIONS edges (chunk → entity)
                                   6. RELATED_TO edges from REBEL extractions
                                   7. SVO (Subject-Verb-Object) narrative edges
            3c  Co-occurrence edges
                                 Every pair of entities that appear in the SAME
                                 chunk gets a RELATED_TO(cooccurs) edge.  This
                                 dramatically increases graph density (~171 000
                                 additional edges for HotpotQA).
            3d  Graph cleanup   Drop orphan entities, hub entities, and merge
                                 surface-form duplicates (canonical_form).
            3d.5 Entity linking  Embed all entity names via nomic-embed-text and
                                 merge near-duplicates within each type bucket
                                 (cosine >= 0.92). Handles "VCU" ↔ "Virginia
                                 Commonwealth University" style aliases.
            3e  Baseline metrics Print graph health statistics and warn on
                                 threshold violations.

    ═══════════════════════════════════════════════════════════════════════
    CHECKPOINTING (--resume)
    ═══════════════════════════════════════════════════════════════════════

    After each phase completes, a checkpoint is saved to:
        data/<dataset>/graph/.import_checkpoint.json

    If a phase crashes (Ctrl-C, power cut, KuzuDB lock error) you can
    restart with --resume and skip all already-completed phases.  The
    name_to_id mapping needed by Phase 3c is also persisted to:
        data/<dataset>/graph/.name_to_id.json

    --clear deletes the checkpoint, forcing a full fresh import.
    --resume and --clear are mutually exclusive.

    ═══════════════════════════════════════════════════════════════════════

    Args:
        chunks_path:                  Path to chunks_export.json (Phase 1).
        extractions_path:             Path to extraction_results.json (Phase 2).
        dataset_name:                 Dataset name (e.g. "hotpotqa").
        config:                       Settings dict (from settings.yaml or defaults).
        graph_only:                   Skip the vector-store ingestion (Phase 3a).
        clear:                        Delete existing stores before import.
        skip_cleanup:                 Disable Phase 3d cleanup pass.
        skip_cooccurrence:            Disable Phase 3c co-occurrence edges.
        skip_subsumptive_cleanup:     Disable Phase 3c.5 (delete cooccurs
                                      edges where a semantic edge already
                                      covers the same pair).
        skip_isolated_drop:           Disable Phase 3f (drop entities with
                                      zero RELATED_TO edges after linking).
        cleanup_dry_run:              Cleanup pass counts only, no DB writes.
        cooccurrence_min_confidence:  Min NER confidence for co-occurrence.
        hub_threshold_ratio:          Hub cutoff: ratio × total_chunks.
        enable_entity_linking:        Run Phase 3d.5 embedding-based linking.
        linking_threshold:            Cosine threshold for entity merging.
        resume:                       Skip phases recorded as done in checkpoint.
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage modules not available - aborting")
        sys.exit(1)

    if resume and clear:
        logger.error("--resume and --clear are mutually exclusive. Use one or the other.")
        sys.exit(1)

    total_start = time.time()

    # Paths (defined early so --clear and checkpoint logic can use them)
    base_path = Path("./data") / dataset_name
    vector_path = base_path / "vector"
    graph_path  = base_path / "graph"

    # ── Load checkpoint (--resume) ────────────────────────────────────────────
    checkpoint = _load_checkpoint(graph_path) if resume else {}
    if resume and checkpoint:
        done = [p for p, v in checkpoint.items() if v.get(_CP_DONE)]
        logger.info("  Resuming: phases already done: %s", done)

    print()
    print("═" * 70)
    print("DECOUPLED INGESTION  -  PHASE 3: LOCAL IMPORT")
    print("═" * 70)
    print(f"  Dataset:      {dataset_name}")
    print(f"  Chunks:       {chunks_path}")
    print(f"  Extractions:  {extractions_path}")
    print(f"  Graph only:   {graph_only}")
    print(f"  Resume:       {'on' if resume else 'off'}")
    print(f"  Cleanup:      {'off' if skip_cleanup else ('dry-run' if cleanup_dry_run else 'on')}")
    print(f"  Co-occur:     {'off' if skip_cooccurrence else 'on'} "
          f"(conf >= {cooccurrence_min_confidence})")
    print(f"  Attr objects: {'CONCEPT nodes (--attribute-objects)' if capture_attribute_objects else 'dropped (default — OntoNotes-5 taxonomy)'}")
    print(f"  Embeddings:   {embeddings_backend}")
    print("═" * 70)

    # Build the shared embeddings object once; both Phase 3a and 3d.5 use it.
    if embeddings_backend == "huggingface":
        embedding_config = config.get("embeddings", {})
        perf_config = config.get("performance", {})
        _hf_model = embedding_config.get("hf_model_name", "nomic-ai/nomic-embed-text-v1")
        _shared_embeddings = _HFEmbeddings(
            model_name=_hf_model,
            batch_size=perf_config.get("batch_size", 64),
        )
        logger.info("  HuggingFace embeddings ready (%s)", _hf_model)
    else:
        _shared_embeddings = None  # Phase 3a/3d.5 build BatchedOllamaEmbeddings lazily

    # Clear if requested.
    # IMPORTANT: extraction_results.json and chunks_export.json are NEVER
    # deleted - they are source artifacts (Phase 1 / Colab output) and must
    # be preserved. Only the derived stores (vector, KuzuDB graph files) are
    # removed.
    if clear:
        import shutil, os, stat

        # Windows / KuzuDB compatibility: KuzuDB holds an OS-level lock on
        # its .lock and shadow files until the holding process exits.  A
        # bare shutil.rmtree raises PermissionError [WinError 5] on any
        # locked file.  This handler chmod+retries each failed entry and
        # skips the ones that remain locked, so a partial clean still
        # succeeds and the next run can recreate what was missed.
        def _rm_retry(func, path):
            try:
                os.chmod(path, stat.S_IWRITE)
            except OSError:
                pass
            for _ in range(3):
                try:
                    func(path)
                    return
                except PermissionError:
                    time.sleep(0.4)
                except FileNotFoundError:
                    return
            logger.warning(
                "  Skipped (still locked, close Python/IDE holders and retry): %s",
                path,
            )

        # Python 3.12 deprecates onerror in favour of onexc; support both.
        if sys.version_info >= (3, 12):
            def _rm_handler(func, path, exc):  # onexc signature
                _rm_retry(func, path)
            _rmtree_kwargs = {"onexc": _rm_handler}
        else:
            def _rm_handler(func, path, exc_info):  # onerror signature
                _rm_retry(func, path)
            _rmtree_kwargs = {"onerror": _rm_handler}

        if graph_only:
            targets = [graph_path]
        else:
            targets = [vector_path, graph_path]
        for target in targets:
            if not target.exists():
                continue
            # ── SAFETY: filesystem sidecar for every .json file ───────────────
            # JSON files are SOURCE ARTIFACTS (extraction_results.json,
            # chunks_export.json) that take ~30 min of GPU time to regenerate
            # and MUST survive --clear.
            #
            # CRITICAL: rescue must be filesystem-based, not in-memory.  If
            # rmtree raises mid-way (Windows DB lock, Ctrl-C, OOM, segfault
            # in a C extension, etc.) the in-memory dict is lost with the
            # process while the on-disk JSON is already deleted.  Sidecar
            # the files OUT of the target first, then delete, then move
            # them back — at every point the data lives on disk somewhere.
            sidecar = target.parent / f".{target.name}_rescue"
            if sidecar.exists():
                # Leftover from a prior crashed run — restore those first
                # before we touch anything else, so we never overwrite a
                # rescued file with a fresh one.
                logger.warning(
                    "  Found prior rescue dir %s; restoring before --clear.",
                    sidecar,
                )
                target.mkdir(parents=True, exist_ok=True)
                for f in sidecar.iterdir():
                    dst = target / f.name
                    if not dst.exists():
                        f.replace(dst)
                        logger.info(f"  Recovered from prior rescue: {dst}")
                # Remove leftover empty sidecar; ignore if anything remains.
                try:
                    sidecar.rmdir()
                except OSError:
                    pass

            sidecar.mkdir(parents=True, exist_ok=True)
            json_files = list(target.rglob("*.json"))
            for json_file in json_files:
                dst = sidecar / json_file.name
                # os.replace is atomic on a single filesystem
                os.replace(json_file, dst)
                logger.info(f"  Sidecarred before --clear: {json_file.name}")

            # ── DELETE database directory (try/finally so restore ALWAYS
            #    runs, even if rmtree raises) ─────────────────────────────────
            rmtree_error: Optional[BaseException] = None
            try:
                shutil.rmtree(target, **_rmtree_kwargs)
                logger.info(f"  Cleared: {target}")
            except BaseException as exc:   # noqa: BLE001 — must restore on ANY failure
                rmtree_error = exc
                logger.error(
                    "  rmtree failed (%s); will restore JSON sidecar before re-raising.",
                    exc,
                )
            finally:
                # ── RESTORE rescued JSON files ────────────────────────────
                target.mkdir(parents=True, exist_ok=True)
                for f in sidecar.iterdir():
                    dst = target / f.name
                    os.replace(f, dst)
                    logger.info(f"  Restored: {dst}")
                try:
                    sidecar.rmdir()
                except OSError:
                    logger.warning(
                        "  Sidecar dir %s not empty after restore — inspect manually.",
                        sidecar,
                    )

            if rmtree_error is not None:
                raise rmtree_error

    # --clear also invalidates the checkpoint so the next run starts fresh.
    if clear:
        for _cp in [_checkpoint_path(graph_path), _name_to_id_path(graph_path)]:
            try:
                _cp.unlink(missing_ok=True)
            except OSError:
                pass

    base_path.mkdir(parents=True, exist_ok=True)

    # Load source data (needed by all phases).
    chunks = load_chunks(chunks_path)
    extraction_data = load_extractions(extractions_path)
    extraction_results = extraction_data.get("results", [])

    if len(chunks) != len(extraction_results):
        logger.warning(
            "  WARNING: chunk count mismatch (chunks=%d, extractions=%d)",
            len(chunks), len(extraction_results),
        )
        logger.warning("  Continuing anyway (using chunk_id intersection).")

    documents = chunks_to_documents(chunks)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3a — VECTOR STORE  (skipped with --graph-only)
    # ══════════════════════════════════════════════════════════════════════
    # Builds a LanceDB IVF-Flat index from the chunk texts.
    # This index is used for dense (embedding) retrieval at query time.
    # Typical time: ~2-5 min for 9 000 chunks on CPU.
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    if not graph_only:
        if _phase_done(checkpoint, "3a"):
            logger.info("  Phase 3a (vector store): SKIPPED — already in checkpoint")
        else:
            try:
                ingest_vector_store(documents, vector_path, config, dataset_name,
                                    embeddings=_shared_embeddings)
                _save_checkpoint(graph_path, "3a", {"chunks": len(documents)})
                logger.info(_phase_eta("3a vector store", time.time() - _t0))
            except Exception as e:
                logger.error(f"Vector store ingestion failed: {e}")
                logger.error("Use --graph-only when the vector store is built separately.")
                raise
    else:
        logger.info("  Phase 3a (vector store): SKIPPED — --graph-only")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3b — KNOWLEDGE GRAPH  (KuzuDB)
    # ══════════════════════════════════════════════════════════════════════
    # Imports ALL entities, relations, and chunk-graph structure into KuzuDB.
    # Sub-steps:
    #   1-3  DocumentChunk + SourceDocument nodes + FROM_SOURCE / NEXT_CHUNK
    #   4    Entity nodes (deduplicated via canonical_form + SHA-256 id)
    #   5    MENTIONS edges: chunk → entity
    #   6    RELATED_TO edges from REBEL relation extraction
    #   7    SVO narrative edges from SpaCy dependency parse
    #
    # Typical time (9 000 chunks, Windows, SSD):
    #   Without batch transactions:  ~45 min  (one fsync per statement)
    #   With batch transactions:     ~5-8 min (one fsync per 200 chunks)
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    graph_store = None
    stats: Dict[str, Any] = {}
    name_to_id: Dict[str, str] = {}

    if _phase_done(checkpoint, "3b"):
        logger.info("  Phase 3b (knowledge graph): SKIPPED — already in checkpoint")
        # Re-open the existing graph store so downstream phases can use it.
        # KuzuGraphStore takes the *container* directory and appends
        # KUZU_DIR_NAME itself — pass graph_path, not graph_path/"graph_KuzuDB",
        # otherwise the path is doubled (.../graph_KuzuDB/graph_KuzuDB).
        graph_store = KuzuGraphStore(str(graph_path))
        stats = checkpoint["3b"].get("stats", {})
        # Reload name_to_id from the sidecar file saved by the previous run.
        n2i_path = _name_to_id_path(graph_path)
        if n2i_path.exists():
            name_to_id = json.loads(n2i_path.read_text(encoding="utf-8"))
            logger.info("  Loaded name_to_id: %d entries", len(name_to_id))
        else:
            logger.warning(
                "  name_to_id cache not found — Phase 3c (co-occurrence) "
                "will produce 0 edges. Re-run without --resume to rebuild."
            )
    else:
        try:
            entity_conf = (
                config.get("entity_extraction", {})
                      .get("gliner", {})
                      .get("confidence_threshold", 0.5)
            )
            graph_store, stats = ingest_knowledge_graph(
                documents, extraction_results, graph_path, dataset_name,
                entity_confidence_threshold=entity_conf,
                capture_attribute_objects=capture_attribute_objects,
            )
            name_to_id = stats.pop("_name_to_id", {})
            # Persist name_to_id so --resume can reload it for Phase 3c.
            _name_to_id_path(graph_path).write_text(
                json.dumps(name_to_id, ensure_ascii=False), encoding="utf-8"
            )
            _save_checkpoint(graph_path, "3b", {"stats": stats})
            logger.info(_phase_eta("3b knowledge graph", time.time() - _t0))
        except Exception as e:
            logger.error(f"Knowledge graph ingestion failed: {e}")
            raise

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3c — CO-OCCURRENCE EDGES
    # ══════════════════════════════════════════════════════════════════════
    # Every pair of entities that appear in the SAME chunk gets a
    # RELATED_TO(cooccurs) edge.  This is the primary mechanism for
    # increasing graph density from ~25% to ~95%+ chunk coverage.
    #
    # Example: chunk mentions ["Tim Burton", "Johnny Depp", "Ed Wood"]
    #   → 3 new edges: Tim Burton↔Johnny Depp, Tim Burton↔Ed Wood,
    #                  Johnny Depp↔Ed Wood
    #
    # Typical result: ~171 000 unique edges for 9 412 HotpotQA chunks.
    # Typical time:   ~3-5 min (bulk transactional writes, 500/batch).
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    cooccurrence_edges = 0
    if graph_store is not None and not skip_cooccurrence:
        if _phase_done(checkpoint, "3c"):
            logger.info("  Phase 3c (co-occurrence): SKIPPED — already in checkpoint")
            cooccurrence_edges = checkpoint["3c"].get("edges", 0)
        else:
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
                _save_checkpoint(graph_path, "3c", {"edges": cooccurrence_edges})
                logger.info(_phase_eta("3c co-occurrence", time.time() - _t0))
            except Exception as e:
                logger.error("Co-occurrence edge construction failed: %s", e)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3c.5 — SUBSUMPTIVE CO-OCCURRENCE CLEANUP (semantic wins)
    # ══════════════════════════════════════════════════════════════════════
    # For every entity-pair that has BOTH a REBEL/SVO semantic edge AND a
    # cooccurs edge, delete the cooccurs edge. The semantic relation already
    # entails co-occurrence, so the cooccurs row is redundant — keeping it
    # only inflates edge counts (~8:1 cooccurs:semantic on HotpotQA) and
    # pollutes visualisation and ablation metrics. Pairs whose ONLY signal
    # is co-occurrence are kept (still the only edge we have for them).
    # See §12.36 retrieval-time weighting (kept as a belt-and-braces).
    #
    # Typical removal: ~30 000 of the ~290 000 cooccurs on HotpotQA. The
    # remaining cooccurs edges are between pairs that REBEL never connected
    # — exactly where co-occurrence is the only available bridge signal.
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    subsumed_dropped = 0
    if (
        graph_store is not None
        and not skip_cooccurrence
        and not skip_subsumptive_cleanup
    ):
        if _phase_done(checkpoint, "3c5"):
            logger.info("  Phase 3c.5 (subsumptive cleanup): SKIPPED — already in checkpoint")
            subsumed_dropped = checkpoint["3c5"].get("dropped", 0)
        else:
            logger.info(f"\n{'─'*70}")
            logger.info(
                "PHASE 3c.5: SUBSUMPTIVE CO-OCCURRENCE CLEANUP%s",
                " (DRY RUN)" if cleanup_dry_run else "",
            )
            logger.info(f"{'─'*70}")
            try:
                subsumed_dropped = drop_subsumed_cooccurrence_edges(
                    graph_store=graph_store,
                    cooccurs_relation_type="cooccurs",
                    dry_run=cleanup_dry_run,
                )
                logger.info(
                    "  Subsumed cooccurs deleted: %d (semantic relation already exists)",
                    subsumed_dropped,
                )
                _save_checkpoint(graph_path, "3c5", {"dropped": subsumed_dropped})
                logger.info(_phase_eta("3c.5 subsumptive cleanup", time.time() - _t0))
            except Exception as e:
                logger.error("Subsumptive cleanup failed: %s", e)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3d — GRAPH CLEANUP
    # ══════════════════════════════════════════════════════════════════════
    # Four-pass cleanup to improve graph quality:
    #   Pass 1  Stop-list:   Drop entities matching DEFAULT_STOPLIST
    #           (pronouns "He", "She", nationality adjectives "American", etc.)
    #   Pass 2  Orphans:     Drop entities with 0 MENTIONS edges.
    #           (entities that GLiNER extracted but no chunk references)
    #   Pass 3  Hubs:        Drop entities mentioned in > 3% of all chunks.
    #           (overly generic nodes like "United States" that link
    #            unrelated chunks and reduce retrieval precision)
    #   Pass 4  Duplicates:  Merge entities sharing the same canonical_form
    #           within a type bucket ("Ed Wood" + "ed wood" → one node).
    # Typical time: < 1 min.
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    cleanup_ops = {"orphans_dropped": 0, "hubs_dropped": 0,
                   "duplicates_merged": 0, "stoplist_dropped": 0}
    if graph_store is not None and not skip_cleanup:
        if _phase_done(checkpoint, "3d"):
            logger.info("  Phase 3d (cleanup): SKIPPED — already in checkpoint")
            cleanup_ops = checkpoint["3d"].get("ops", cleanup_ops)
        else:
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
                logger.info("  Orphans dropped:    %d", cleanup_ops["orphans_dropped"])
                logger.info(
                    "  Hubs dropped:       %d (threshold ratio = %.1f%% of chunks)",
                    cleanup_ops["hubs_dropped"], hub_threshold_ratio * 100,
                )
                logger.info("  Duplicates merged:  %d", cleanup_ops["duplicates_merged"])
                _save_checkpoint(graph_path, "3d", {"ops": cleanup_ops})
                logger.info(_phase_eta("3d cleanup", time.time() - _t0))
            except Exception as e:
                logger.error("Cleanup pass failed: %s", e)

    # Phase 3d.5: Embedding-based entity linking (alias resolution beyond
    # canonical_form). Merges "Ed Wood" with "Edward Davis Wood Jr." etc.
    # by clustering name embeddings within each canonical type bucket.
    #
    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3d.5 — EMBEDDING-BASED ENTITY LINKING
    # ══════════════════════════════════════════════════════════════════════
    # After canonical_form deduplication, some aliases still differ:
    #   "VCU" ≠ "Virginia Commonwealth University"
    #   "Ed Wood" ≠ "Edward Davis Wood Jr."  (after removing Jr. suffix)
    # This phase embeds every entity name using nomic-embed-text (via Ollama)
    # and merges pairs with cosine similarity >= 0.92 within the same type.
    # Typical time: ~10 min for ~5 000 entities (Ollama on CPU).
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    linked_count = 0
    if graph_store is not None and enable_entity_linking:
        if _phase_done(checkpoint, "3d5"):
            prev_3d5 = checkpoint["3d5"]
            prev_note = prev_3d5.get("note", "")
            prev_max_type_size = prev_3d5.get("max_type_size", 0)
            logger.info("  Phase 3d.5 (entity linking): SKIPPED — already in checkpoint")
            linked_count = prev_3d5.get("linked", 0)
            if "partial" in prev_note.lower() or (
                prev_max_type_size and linking_max_type_size > prev_max_type_size
            ):
                logger.warning(
                    "  Phase 3d.5 prior run was PARTIAL (note='%s', max_type_size=%d) "
                    "but current --linking-max-type-size=%d is larger. "
                    "Delete the '3d5' entry from .import_checkpoint.json and re-run "
                    "with --resume to process the previously-skipped type buckets "
                    "(typically PERSON, ORG).",
                    prev_note, prev_max_type_size, linking_max_type_size,
                )
        else:
            logger.info(f"\n{'─'*70}")
            logger.info("PHASE 3d.5: EMBEDDING-BASED ENTITY LINKING")
            logger.info(f"{'─'*70}")
            try:
                if _shared_embeddings is not None:
                    embedder = _shared_embeddings
                else:
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
                    max_type_size=linking_max_type_size,
                    dry_run=cleanup_dry_run,
                )
                logger.info(
                    "  Embedding-linked entities merged: %d (threshold=%.2f)",
                    linked_count, linking_threshold,
                )
                _save_checkpoint(
                    graph_path,
                    "3d5",
                    {
                        "linked": linked_count,
                        "max_type_size": linking_max_type_size,
                        "threshold": linking_threshold,
                    },
                )
                logger.info(_phase_eta("3d.5 entity linking", time.time() - _t0))
            except Exception as exc:
                logger.error("Entity linking failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3f — POST-LINK ISOLATED-ENTITY DROP
    # ══════════════════════════════════════════════════════════════════════
    # After Phase 3d.5 (alias resolution) some entities end up with MENTIONS
    # edges but ZERO RELATED_TO edges (their cluster-mates absorbed the
    # connectivity during the merge, or REBEL/SVO never produced a triple
    # for them). These dead-leaf nodes inflate the graph and cause the
    # baseline `isolated_rate` invariant to fail (~27% on HotpotQA before
    # this phase). They contribute nothing to graph traversal — every
    # multi-hop search rooted on them returns the empty set.
    #
    # We drop them here so the §3e baseline reports the correctly-pruned
    # state and the published "we cleaned the graph" claim is truthful.
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    isolated_dropped = 0
    if (
        graph_store is not None
        and not skip_isolated_drop
    ):
        if _phase_done(checkpoint, "3f"):
            logger.info("  Phase 3f (drop-isolated): SKIPPED — already in checkpoint")
            isolated_dropped = checkpoint["3f"].get("dropped", 0)
        else:
            logger.info(f"\n{'─'*70}")
            logger.info(
                "PHASE 3f: POST-LINK ISOLATED-ENTITY DROP%s",
                " (DRY RUN)" if cleanup_dry_run else "",
            )
            logger.info(f"{'─'*70}")
            try:
                isolated_dropped = drop_isolated_entities(
                    graph_store=graph_store,
                    dry_run=cleanup_dry_run,
                )
                logger.info(
                    "  Isolated entities dropped: %d (zero RELATED_TO edges)",
                    isolated_dropped,
                )
                _save_checkpoint(graph_path, "3f", {"dropped": isolated_dropped})
                logger.info(_phase_eta("3f drop-isolated", time.time() - _t0))
            except Exception as exc:
                logger.error("Drop-isolated failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3e — BASELINE METRICS
    # ══════════════════════════════════════════════════════════════════════
    # Computes graph health statistics and checks invariants:
    #   - total nodes / edges / densities
    #   - isolated entity rate (should be < 5% after co-occurrence)
    #   - duplicate cluster rate (should be < 2%)
    #   - relations per chunk (should be >= 5.0)
    # Warnings are printed but never abort the import.
    # Typical time: < 30 seconds.
    # ══════════════════════════════════════════════════════════════════════
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
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        print(f"    {key:<26}: {val:>10,}")
    print(f"    {'cooccurrence_edges':<26}: {cooccurrence_edges:>10,}")
    print(f"    {'cooccurs_subsumed':<26}: {subsumed_dropped:>10,}")
    print(f"    {'svo_relations':<26}: {stats.get('svo_relations', 0):>10,}")
    print(f"    {'stoplist_dropped':<26}: {cleanup_ops['stoplist_dropped']:>10,}")
    print(f"    {'orphans_dropped':<26}: {cleanup_ops['orphans_dropped']:>10,}")
    print(f"    {'hubs_dropped':<26}: {cleanup_ops['hubs_dropped']:>10,}")
    print(f"    {'duplicates_merged':<26}: {cleanup_ops['duplicates_merged']:>10,}")
    print(f"    {'embedding_linked':<26}: {linked_count:>10,}")
    print(f"    {'post_link_isolated':<26}: {isolated_dropped:>10,}")
    if stats.get("rebel_confidence_is_constant"):
        print()
        print("    NOTE: REBEL relation confidence is constant (0.5) for all edges —")
        print("          it is a decoder sentinel, not a ranking signal. Graph")
        print("          retrieval must not filter RELATED_TO edges on confidence.")
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
    meta["cooccurs_subsumed_dropped"] = subsumed_dropped
    meta["post_link_isolated_dropped"] = isolated_dropped
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
        "--no-subsumptive-cleanup",
        action="store_true",
        help="Disable Phase 3c.5 — keep cooccurs edges even when a semantic "
             "edge already covers the same entity-pair. Default is ON: any "
             "cooccurs edge that has a paired semantic edge in either "
             "direction is deleted as redundant.",
    )
    parser.add_argument(
        "--no-isolated-drop",
        action="store_true",
        help="Disable Phase 3f — keep entities with zero RELATED_TO edges "
             "after entity linking. Default is ON: dead-leaf entities "
             "(MENTIONS present but RELATED_TO empty in both directions) "
             "are removed so the §3e baseline reports the correctly-"
             "pruned state.",
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
    parser.add_argument(
        "--linking-max-type-size",
        type=int,
        default=8000,
        help="Maximum entity-bucket size for embedding-based linking (default: 8000). "
             "Buckets larger than this are skipped to avoid OOM (8000 entities = ~256 MB "
             "float32 matrix). Raise to 50000+ only if you have sufficient RAM/VRAM.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases already recorded as complete in the checkpoint file "
             "(data/<dataset>/graph/.import_checkpoint.json). "
             "Mutually exclusive with --clear.",
    )
    parser.add_argument(
        "--embeddings-backend",
        choices=["ollama", "huggingface"],
        default="ollama",
        help="Embedding backend: 'ollama' (default, requires Ollama running locally) "
             "or 'huggingface' (uses sentence-transformers, no Ollama needed — "
             "ideal for Colab/GPU environments).",
    )
    parser.add_argument(
        "--attribute-objects",
        action="store_true",
        help="Opt-in: capture REBEL relations whose object is a free-text "
             "attribute value ('genre -> punk rock', 'sport -> ice hockey') "
             "as synthetic CONCEPT entity nodes. Default OFF — keeps the "
             "graph within the OntoNotes-5 GLiNER taxonomy. Enable only for "
             "exploratory runs or ablation studies.",
    )

    args = parser.parse_args()

    # Validation
    if args.resume and args.clear:
        logger.error("--resume and --clear are mutually exclusive.")
        sys.exit(1)
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
        skip_subsumptive_cleanup=args.no_subsumptive_cleanup,
        skip_isolated_drop=args.no_isolated_drop,
        cleanup_dry_run=args.cleanup_dry_run,
        cooccurrence_min_confidence=args.cooccurrence_min_confidence,
        hub_threshold_ratio=args.hub_threshold_ratio,
        enable_entity_linking=not args.no_entity_linking,
        linking_threshold=args.linking_threshold,
        linking_max_type_size=args.linking_max_type_size,
        resume=args.resume,
        embeddings_backend=args.embeddings_backend,
        capture_attribute_objects=args.attribute_objects,
    )


if __name__ == "__main__":
    main()