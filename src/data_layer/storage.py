"""
Hybrid Storage Module: Vector Store (LanceDB) + Knowledge Graph (KuzuDB)

Version: 4.0.0
Author: Jan Nietzard
Last Modified: 2026-04-24

===============================================================================
OVERVIEW
===============================================================================

This module implements the dual-storage persistence layer of the Edge-RAG
system.  Two embedded databases are combined under a single `HybridStore`
facade:

  - **LanceDB** (vector store): approximate nearest-neighbour search over
    dense embeddings produced by nomic-embed-text via Ollama.
    Reference: Malkov & Yashunin (2018). "Efficient and robust approximate
    nearest neighbor search using hierarchical navigable small world graphs."
    IEEE Transactions on Pattern Analysis and Machine Intelligence.

  - **KuzuDB** (knowledge graph): native Cypher-based multi-hop traversal
    over a DocumentChunk–Entity graph constructed during ingestion.
    Reference: Feng et al. (2023). "Kùzu Graph Database Management System."
    CIDR 2023.

The combination supports the hybrid retrieval strategy described in thesis
section 2.4: dense vector recall is augmented with graph-based entity
expansion, enabling bridge-entity reasoning across documents without loading
full document text into memory (< 16 GB RAM constraint).

NetworkX is retained as a fallback for environments where KuzuDB is
unavailable, but multi-hop Cypher retrieval is only available under KuzuDB.

===============================================================================
GRAPH SCHEMA
===============================================================================

Node Tables:
    DocumentChunk(chunk_id PK, text, page_number, chunk_index, source_file)
    SourceDocument(doc_id PK, filename, total_pages)
    Entity(entity_id PK, name, type, confidence)

Relationship Tables:
    FROM_SOURCE:  DocumentChunk → SourceDocument
    NEXT_CHUNK:   DocumentChunk → DocumentChunk  (sequential ordering)
    MENTIONS:     DocumentChunk → Entity
    RELATED_TO:   Entity → Entity                (with relation_type, confidence)

===============================================================================
USAGE
===============================================================================

    from storage import HybridStore, StorageConfig
    from langchain_core.embeddings import Embeddings

    config = StorageConfig(
        vector_db_path=Path("./data/vector"),
        graph_db_path=Path("./data/graph"),
    )
    store = HybridStore(config, embeddings)
    store.add_documents(documents)
    results = store.vector_search(query_embedding, top_k=5)

===============================================================================
REVIEW HISTORY
===============================================================================

    Last Reviewed:  2026-04-19
    Review Result:  0 CRITICAL, 4 IMPORTANT, 3 RECOMMENDED
    Reviewer:       Code Review Prompt v2.1
    Next Review:    After schema changes or graph backend refactoring

===============================================================================
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from dataclasses import dataclass

if TYPE_CHECKING:
    from .entity_extraction import EntityExtractionPipeline

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import lancedb

# KuzuDB for graph storage
try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

logger = logging.getLogger(__name__)

# Expected storage-layer exception types.
#
# Used in all except-clauses throughout this module instead of bare
# `except Exception` so that programming bugs (AttributeError, NameError,
# IndexError) are NOT silently swallowed alongside legitimate DB/IO errors.
#
# SystemExit, KeyboardInterrupt, and MemoryError are already excluded by
# `Exception`, but this alias further excludes AssertionError and the broad
# catch-all for library-internal exceptions that should propagate.
_STORAGE_ERRORS: tuple = (
    OSError,        # file-system / IPC failures
    IOError,        # alias for OSError, explicit for clarity
    RuntimeError,   # KuzuDB / LanceDB internal errors
    ValueError,     # type mismatch, out-of-range inputs
    TypeError,      # wrong argument type from DB layer
    KeyError,       # missing column / field in result row
)
if KUZU_AVAILABLE and hasattr(kuzu, "Error"):
    _STORAGE_ERRORS = (*_STORAGE_ERRORS, kuzu.Error)

if not KUZU_AVAILABLE:
    logger.warning("KuzuDB not available. Install with: pip install kuzu")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class StorageConfig:
    """
    Configuration for the Hybrid Storage System.

    Frozen dataclass: fields are immutable after construction, preventing
    accidental mutation of shared config objects (e.g. in HybridStore.__init__).

    All configurable thresholds should be read from config/settings.yaml
    via a factory (e.g. create_hybrid_retriever in hybrid_retriever.py).
    Defaults here serve as documented emergency fallbacks only.

    Attributes:
        vector_db_path: Directory path for LanceDB.
        graph_db_path: Container directory for KuzuDB
                       (actual DB stored as graph_KuzuDB/ inside it).
        embedding_dim: Embedding vector dimensionality (None = auto-detect).
        similarity_threshold: Minimum cosine similarity for vector results.
        normalize_embeddings: L2-normalise vectors before storage and search.
        distance_metric: LanceDB distance metric ("cosine" or "l2").
        graph_backend: Storage backend — only "kuzu" is supported.
        overfetch_factor: ANN over-fetch multiplier. LanceDB retrieves
            top_k * overfetch_factor candidates; Python then re-ranks and
            filters by similarity_threshold. Factor 3 balances recall vs
            latency for typical top_k=5–10 queries (empirically validated).
        graph_text_max_chars: Maximum characters stored in graph node text
            field. Full text lives in the vector store; the graph field is
            used only for lightweight context display.
        enable_entity_extraction: Enable GLiNER + REBEL entity extraction
            during ingestion (opt-in; can be injected via entity_pipeline).
        entity_cache_path: Path to entity SQLite cache (None = auto-generate).
    """
    vector_db_path: Path
    graph_db_path: Path
    embedding_dim: Optional[int] = None
    similarity_threshold: float = 0.3
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"
    graph_backend: str = "kuzu"
    overfetch_factor: int = 3
    graph_text_max_chars: int = 500
    enable_entity_extraction: bool = False
    entity_cache_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive: %d" % self.embedding_dim)

        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                "similarity_threshold must be in [0,1]: %f" % self.similarity_threshold
            )

        if self.distance_metric not in ("cosine", "l2"):
            raise ValueError(
                "distance_metric must be 'cosine' or 'l2': %s" % self.distance_metric
            )

        if self.graph_backend != "kuzu":
            raise ValueError(
                "graph_backend must be 'kuzu' (NetworkX fallback removed): %s" % self.graph_backend
            )

        if self.overfetch_factor < 1:
            raise ValueError(
                "overfetch_factor must be >= 1: %d" % self.overfetch_factor
            )


# ============================================================================
# VECTOR STORE ADAPTER (LanceDB)
# ============================================================================

class VectorStoreAdapter:
    """
    LanceDB Vector Store Adapter with distance-to-similarity conversion.

    LanceDB returns raw distances (lower = more similar for cosine/L2).
    This adapter converts them to similarity scores in [0, 1] so that
    downstream components can apply a uniform threshold.

    The ANN search over-fetches by `overfetch_factor` to allow threshold
    filtering after similarity conversion without losing top candidates.
    """

    SCHEMA_VERSION = "3.1.0"
    TABLE_NAME = "documents"

    def __init__(
        self,
        db_path: Path,
        embedding_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        distance_metric: str = "cosine",
        overfetch_factor: int = 3,
    ) -> None:
        if distance_metric not in ("cosine", "l2"):
            raise ValueError("Unsupported distance metric: %s" % distance_metric)

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(db_path))
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        self.distance_metric = distance_metric
        self.overfetch_factor = overfetch_factor
        self.table = None

        logger.info(
            "VectorStoreAdapter initialised: path=%s, metric=%s",
            db_path,
            distance_metric,
        )

        self._load_metadata()

        try:
            table_names = self.db.table_names()
            if self.TABLE_NAME in table_names:
                self.table = self.db.open_table(self.TABLE_NAME)
                logger.info(
                    "Opened existing table '%s' with %d rows",
                    self.TABLE_NAME,
                    len(self.table),
                )
        except (RuntimeError, OSError) as exc:
            logger.warning("Could not open existing table: %s", exc)

    def _get_metadata_path(self) -> Path:
        return self.db_path / "vector_store_metadata.json"

    def _load_metadata(self) -> None:
        metadata_path = self._get_metadata_path()
        if not metadata_path.exists():
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            stored_dim = metadata.get("embedding_dim")
            if stored_dim and self.embedding_dim is None:
                self.embedding_dim = stored_dim
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Could not load metadata: %s", exc)

    def _save_metadata(self) -> None:
        if self.embedding_dim is None:
            return
        metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "embedding_dim": self.embedding_dim,
            "distance_metric": self.distance_metric,
            "normalize_embeddings": self.normalize_embeddings,
            "num_documents": len(self.table) if self.table else 0,
            "timestamp": time.time(),
        }
        metadata_path = self._get_metadata_path()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Guard against zero-norm vectors (e.g. all-zero padding)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _validate_embedding_dimension(self, embeddings: List[List[float]]) -> None:
        if not embeddings:
            return
        actual_dim = len(embeddings[0])
        if self.embedding_dim is None:
            self.embedding_dim = actual_dim
            self._save_metadata()
            return
        if actual_dim != self.embedding_dim:
            raise ValueError(
                "EMBEDDING DIMENSION MISMATCH: expected %d, got %d"
                % (self.embedding_dim, actual_dim)
            )

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert a raw LanceDB distance to a similarity score in [0, 1].

        Cosine: similarity = 1 - distance  (distance ∈ [0, 2] in unnormalised
                space; ∈ [0, 1] after L2 normalisation).
        L2:     similarity = 1 / (1 + distance).
        """
        if self.distance_metric == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        elif self.distance_metric == "l2":
            return max(0.0, min(1.0, 1.0 / (1.0 + distance)))
        else:
            raise ValueError("Unknown distance metric: %s" % self.distance_metric)

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: Embeddings,
    ) -> None:
        """Embed documents and insert them into the LanceDB table."""
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        logger.info("Generating embeddings for %d documents...", len(texts))

        t0 = time.time()
        embeddings_list = embeddings.embed_documents(texts)
        logger.info("Embeddings generated in %.2fs", time.time() - t0)

        self._validate_embedding_dimension(embeddings_list)

        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        if self.normalize_embeddings:
            embeddings_array = self._normalize_vectors(embeddings_array)
        embeddings_list = embeddings_array.tolist()

        data = []
        for doc, emb in zip(documents, embeddings_list):
            data.append({
                "document_id": str(doc.metadata.get("chunk_id", "unknown")),
                "text": doc.page_content,
                "vector": emb,
                "metadata": json.dumps(doc.metadata),
                "source_file": doc.metadata.get("source_file", "unknown"),
            })

        try:
            if self.table is None:
                self.table = self.db.create_table(
                    self.TABLE_NAME, data=data, mode="overwrite"
                )
            else:
                self.table.add(data)
            self._save_metadata()
        except _STORAGE_ERRORS as exc:
            logger.error("Failed to insert documents: %s", exc)
            raise

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Perform ANN vector search with similarity threshold filtering.

        Over-fetches by `overfetch_factor` before threshold filtering to
        avoid losing top-k candidates when many results fall below threshold.
        """
        if self.table is None:
            return []

        try:
            if self.normalize_embeddings:
                query_array = np.array([query_embedding], dtype=np.float32)
                query_array = self._normalize_vectors(query_array)
                query_embedding = query_array[0].tolist()

            raw_results = (
                self.table
                .search(query_embedding)
                .metric(self.distance_metric)
                .limit(top_k * self.overfetch_factor)
                .to_list()
            )

            filtered_results = []
            for result in raw_results:
                distance = result.get("_distance", 0.0)
                similarity = self._distance_to_similarity(distance)

                if similarity >= threshold:
                    try:
                        metadata = json.loads(result.get("metadata", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                    filtered_results.append({
                        "text": result.get("text", ""),
                        "similarity": similarity,
                        "document_id": result.get("document_id", "unknown"),
                        "metadata": metadata,
                    })

            filtered_results.sort(key=lambda x: x["similarity"], reverse=True)
            return filtered_results[:top_k]

        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Vector search failed: %s", exc)
            return []


def _entity_name_variants(name: str) -> List[str]:
    """
    Build a prioritised list of name variants for alias-tolerant entity lookup.

    Handles common patterns without a full alias table:
      "Ed Wood"          → also try "Wood" (last-name-only for short first names)
      "Scott Derrickson" → also try "Derrickson" (multi-token fallback)

    Used by KuzuGraphStore.find_chunks_by_entity_multihop.
    """
    variants = [name]
    tokens = name.split()
    if len(tokens) == 2 and len(tokens[0]) <= 3:
        variants.append(tokens[-1])
    if len(tokens) > 1:
        for tok in tokens:
            if len(tok) >= 4 and tok not in variants:
                variants.append(tok)
    return variants


# ============================================================================
# KUZU GRAPH STORE
# ============================================================================

class KuzuGraphStore:
    """
    KuzuDB-based Knowledge Graph Store.

    Provides Cypher-based multi-hop traversal over a DocumentChunk–Entity
    graph.  Advantages over the NetworkX fallback:

    - Native Cypher support for complex path queries.
    - Columnar, vectorised query execution (10–100x faster for large graphs).
    - ACID transactions and crash recovery.
    - Out-of-core processing via memory-mapped files (handles graphs > RAM).

    Reference: Feng et al. (2023). "Kùzu Graph Database Management System."
               CIDR 2023.

    Schema:
        Nodes:  DocumentChunk, SourceDocument, Entity
        Edges:  FROM_SOURCE, NEXT_CHUNK, MENTIONS, RELATED_TO
    """

    SCHEMA_VERSION = "3.1.0"
    # Subdirectory inside graph_db_path where KuzuDB stores its files.
    # Keeping it in a subdirectory allows sibling files (e.g. entity_cache.db)
    # to share the same parent without confusing KuzuDB.
    KUZU_DIR_NAME = "graph_KuzuDB"

    def __init__(self, db_path: Path) -> None:
        """
        Initialise KuzuDB graph store.

        Args:
            db_path: Container directory.  KuzuDB files are stored at
                     db_path / KUZU_DIR_NAME.
        """
        self.db_path = Path(db_path)

        if not KUZU_AVAILABLE:
            raise ImportError("KuzuDB not installed. Install with: pip install kuzu")

        self.db_path.mkdir(parents=True, exist_ok=True)
        kuzu_file = self.db_path / self.KUZU_DIR_NAME

        self.db = kuzu.Database(str(kuzu_file))
        self.conn = kuzu.Connection(self.db)

        self._init_schema()
        logger.info("KuzuGraphStore initialised: %s", kuzu_file)

    def _init_schema(self) -> None:
        """
        Create node and relationship tables if they do not yet exist.

        KuzuDB requires explicit schema definition before INSERT/MERGE.
        IF NOT EXISTS prevents errors on repeated initialisation; any
        exception here is therefore unexpected and logged as a warning.
        """
        try:
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS DocumentChunk(
                    chunk_id STRING,
                    text STRING,
                    page_number INT64,
                    chunk_index INT64,
                    source_file STRING,
                    PRIMARY KEY (chunk_id)
                )
            """)
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS SourceDocument(
                    doc_id STRING,
                    filename STRING,
                    total_pages INT64,
                    PRIMARY KEY (doc_id)
                )
            """)
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    entity_id STRING,
                    name STRING,
                    type STRING,
                    confidence DOUBLE,
                    PRIMARY KEY (entity_id)
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS FROM_SOURCE(
                    FROM DocumentChunk TO SourceDocument
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS NEXT_CHUNK(
                    FROM DocumentChunk TO DocumentChunk
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MENTIONS(
                    FROM DocumentChunk TO Entity
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATED_TO(
                    FROM Entity TO Entity,
                    relation_type STRING,
                    confidence DOUBLE,
                    source_chunks STRING
                )
            """)
            logger.debug("Graph schema initialised")

        except _STORAGE_ERRORS as exc:
            # IF NOT EXISTS makes duplicate-table errors impossible;
            # any exception here signals an unexpected problem.
            logger.warning("Unexpected error during schema init: %s", exc)

    # ========================================================================
    # NODE WRITERS
    # ========================================================================

    def add_document_chunk(
        self,
        chunk_id: str,
        text: str,
        page_number: int,
        chunk_index: int,
        source_file: str,
        max_text_chars: int = 500,
    ) -> None:
        """
        MERGE a DocumentChunk node.

        Full text is stored in the vector store; `text` here is a truncated
        preview used for lightweight graph-side context display only.
        """
        truncated = text[:max_text_chars] if len(text) > max_text_chars else text
        try:
            self.conn.execute(
                """
                MERGE (c:DocumentChunk {chunk_id: $chunk_id})
                SET c.text = $text,
                    c.page_number = $page_number,
                    c.chunk_index = $chunk_index,
                    c.source_file = $source_file
                """,
                {
                    "chunk_id": chunk_id,
                    "text": truncated,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "source_file": source_file,
                },
            )
        except _STORAGE_ERRORS as exc:
            logger.error("Failed to add chunk %s: %s", chunk_id, exc)
            raise

    def add_source_document(
        self,
        doc_id: str,
        filename: str,
        total_pages: int = 0,
    ) -> None:
        """MERGE a SourceDocument node."""
        try:
            self.conn.execute(
                """
                MERGE (d:SourceDocument {doc_id: $doc_id})
                SET d.filename = $filename,
                    d.total_pages = $total_pages
                """,
                {"doc_id": doc_id, "filename": filename, "total_pages": total_pages},
            )
        except _STORAGE_ERRORS as exc:
            logger.error("Failed to add source doc %s: %s", doc_id, exc)
            raise

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "unknown",
        confidence: float = 0.0,
    ) -> None:
        """MERGE an Entity node."""
        try:
            self.conn.execute(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.confidence = $confidence
                """,
                {
                    "entity_id": entity_id,
                    "name": name,
                    "entity_type": entity_type,
                    "confidence": confidence,
                },
            )
        except _STORAGE_ERRORS as exc:
            logger.error("Failed to add entity %s: %s", entity_id, exc)
            raise

    def add_entity_from_metadata(
        self,
        entity_id: str,
        entity_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Compatibility wrapper: routes entity_type string to the typed
        add_document_chunk or add_source_document method.
        """
        if entity_type == "document_chunk":
            self.add_document_chunk(
                chunk_id=entity_id,
                text=metadata.get("text", ""),
                page_number=metadata.get("page_number", 0),
                chunk_index=metadata.get("chunk_index", 0),
                source_file=metadata.get("source_file", ""),
            )
        elif entity_type == "source_document":
            self.add_source_document(
                doc_id=entity_id,
                filename=metadata.get("filename", entity_id),
                total_pages=metadata.get("total_pages", 0),
            )
        else:
            self.add_entity(
                entity_id=entity_id,
                name=metadata.get("name", entity_id),
                entity_type=entity_type,
            )

    # ========================================================================
    # EDGE WRITERS
    # ========================================================================

    def add_from_source_relation(self, chunk_id: str, doc_id: str) -> None:
        """MERGE a FROM_SOURCE edge: DocumentChunk → SourceDocument."""
        try:
            self.conn.execute(
                """
                MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                MATCH (d:SourceDocument {doc_id: $doc_id})
                MERGE (c)-[:FROM_SOURCE]->(d)
                """,
                {"chunk_id": chunk_id, "doc_id": doc_id},
            )
        except _STORAGE_ERRORS as exc:
            logger.warning("FROM_SOURCE relation failed (%s → %s): %s", chunk_id, doc_id, exc)

    def add_next_chunk_relation(self, chunk_id: str, next_chunk_id: str) -> None:
        """MERGE a NEXT_CHUNK edge for sequential ordering."""
        try:
            self.conn.execute(
                """
                MATCH (c1:DocumentChunk {chunk_id: $chunk_id})
                MATCH (c2:DocumentChunk {chunk_id: $next_chunk_id})
                MERGE (c1)-[:NEXT_CHUNK]->(c2)
                """,
                {"chunk_id": chunk_id, "next_chunk_id": next_chunk_id},
            )
        except _STORAGE_ERRORS as exc:
            logger.warning(
                "NEXT_CHUNK relation failed (%s → %s): %s", chunk_id, next_chunk_id, exc
            )

    def add_mentions_relation(self, chunk_id: str, entity_id: str) -> None:
        """MERGE a MENTIONS edge: DocumentChunk → Entity."""
        try:
            self.conn.execute(
                """
                MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                MATCH (e:Entity {entity_id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {"chunk_id": chunk_id, "entity_id": entity_id},
            )
        except _STORAGE_ERRORS as exc:
            logger.warning(
                "MENTIONS relation failed (%s → %s): %s", chunk_id, entity_id, exc
            )

    def add_related_to_relation(
        self,
        entity1_id: str,
        entity2_id: str,
        relation_type: str = "related",
    ) -> None:
        """MERGE a RELATED_TO edge: Entity → Entity."""
        try:
            self.conn.execute(
                """
                MATCH (e1:Entity {entity_id: $entity1_id})
                MATCH (e2:Entity {entity_id: $entity2_id})
                MERGE (e1)-[:RELATED_TO {relation_type: $relation_type}]->(e2)
                """,
                {
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "relation_type": relation_type,
                },
            )
        except _STORAGE_ERRORS as exc:
            logger.warning(
                "RELATED_TO relation failed (%s → %s): %s", entity1_id, entity2_id, exc
            )

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Compatibility method: routes relation_type to the appropriate
        typed add_*_relation method.
        """
        if relation_type == "from_source":
            self.add_from_source_relation(source_id, target_id)
        elif relation_type == "next_chunk":
            self.add_next_chunk_relation(source_id, target_id)
        elif relation_type == "mentions":
            self.add_mentions_relation(source_id, target_id)
        else:
            self.add_related_to_relation(source_id, target_id, relation_type)

    # ========================================================================
    # GRAPH TRAVERSAL (Cypher)
    # ========================================================================

    def graph_traversal(
        self,
        start_entity: str,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> Dict[str, int]:
        """
        Multi-hop graph traversal via Cypher.

        Returns a mapping of node_id → hop_distance for all nodes reachable
        from start_entity within max_hops steps.  start_entity is included
        only if it exists as a DocumentChunk or Entity node.

        Args:
            start_entity: Starting node ID (chunk_id or entity_id).
            relation_types: Reserved for future filtering; not yet applied
                            in the Cypher query.
            max_hops: Maximum traversal depth.

        Returns:
            Dict mapping node_id -> hop_distance.
        """
        visited: Dict[str, int] = {}

        # Attempt 1: start_entity as DocumentChunk.
        # If the query executes but returns 0 rows (e.g. start_entity is an
        # entity ID, not a chunk ID), visited remains empty and Attempt 2
        # runs unconditionally below.
        try:
            result = self.conn.execute(
                f"""
                MATCH (start:DocumentChunk {{chunk_id: $start_id}})
                MATCH path = (start)-[*1..{max_hops}]->(connected:DocumentChunk)
                RETURN DISTINCT
                    connected.chunk_id AS node_id,
                    length(path) AS hops
                """,
                {"start_id": start_entity},
            )
            # Include start node only if it actually exists in the graph
            start_exists = False
            while result.has_next():
                row = result.get_next()
                node_id, hops = row[0], row[1]
                if node_id:
                    start_exists = True
                    if node_id not in visited or visited[node_id] > hops:
                        visited[node_id] = hops
            if start_exists:
                visited[start_entity] = 0

        except _STORAGE_ERRORS as exc:
            logger.debug("Traversal as DocumentChunk failed: %s", exc)

        # Attempt 2: start_entity as Entity.
        # Runs whenever Attempt 1 found nothing (empty result OR exception),
        # so entity-based traversal is not gated on a DocumentChunk error.
        if not visited:
            try:
                result = self.conn.execute(
                    f"""
                    MATCH (start:Entity {{entity_id: $start_id}})
                    MATCH path = (start)-[*1..{max_hops}]-(connected)
                    RETURN DISTINCT
                        CASE
                            WHEN connected:DocumentChunk THEN connected.chunk_id
                            WHEN connected:Entity THEN connected.entity_id
                        END AS node_id,
                        length(path) AS hops
                    """,
                    {"start_id": start_entity},
                )
                start_exists = False
                while result.has_next():
                    row = result.get_next()
                    node_id, hops = row[0], row[1]
                    if node_id:
                        start_exists = True
                        if node_id not in visited or visited[node_id] > hops:
                            visited[node_id] = hops
                if start_exists:
                    visited[start_entity] = 0

            except _STORAGE_ERRORS as exc2:
                logger.debug("Traversal as Entity failed: %s", exc2)

        return visited

    def find_related_chunks(
        self,
        chunk_id: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks reachable from chunk_id through any graph path.

        Useful for manual context expansion in diagnostic tools.

        Returns:
            List of dicts with chunk_id, text, source_file, hops.
        """
        related: List[Dict[str, Any]] = []
        try:
            result = self.conn.execute(
                f"""
                MATCH (start:DocumentChunk {{chunk_id: $chunk_id}})
                MATCH path = (start)-[*1..{max_hops}]-(related:DocumentChunk)
                WHERE related.chunk_id <> $chunk_id
                RETURN DISTINCT
                    related.chunk_id AS chunk_id,
                    related.text AS text,
                    related.source_file AS source_file,
                    min(length(path)) AS hops
                ORDER BY hops ASC
                """,
                {"chunk_id": chunk_id},
            )
            while result.has_next():
                row = result.get_next()
                related.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "source_file": row[2],
                    "hops": row[3],
                })
        except _STORAGE_ERRORS as exc:
            logger.error("find_related_chunks failed: %s", exc)
        return related

    def get_context_chunks(self, chunk_id: str, window: int = 2) -> List[str]:
        """
        Return chunk IDs within ±window positions via NEXT_CHUNK edges.

        Args:
            chunk_id: Centre chunk ID.
            window: Number of neighbours before and after to include.

        Returns:
            List of chunk IDs in document order (prev … centre … next).
        """
        context_chunks = [chunk_id]
        try:
            # Forward: follow NEXT_CHUNK edges
            result = self.conn.execute(
                f"""
                MATCH path = (start:DocumentChunk {{chunk_id: $chunk_id}})
                    -[:NEXT_CHUNK*1..{window}]->(next:DocumentChunk)
                RETURN next.chunk_id AS chunk_id, length(path) AS distance
                ORDER BY distance ASC
                """,
                {"chunk_id": chunk_id},
            )
            while result.has_next():
                row = result.get_next()
                if row[0] and row[0] not in context_chunks:
                    context_chunks.append(row[0])

            # Backward: reverse NEXT_CHUNK edges
            result = self.conn.execute(
                f"""
                MATCH path = (prev:DocumentChunk)
                    -[:NEXT_CHUNK*1..{window}]->(end:DocumentChunk {{chunk_id: $chunk_id}})
                RETURN prev.chunk_id AS chunk_id, length(path) AS distance
                ORDER BY distance DESC
                """,
                {"chunk_id": chunk_id},
            )
            backward: List[str] = []
            while result.has_next():
                row = result.get_next()
                if row[0] and row[0] not in context_chunks:
                    backward.append(row[0])
            context_chunks = backward + context_chunks

        except _STORAGE_ERRORS as exc:
            logger.debug("get_context_chunks note: %s", exc)

        return context_chunks

    def find_chunks_by_entity_multihop(
        self,
        entity_name: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        3-hop entity search: returns chunks related to entity_name through
        up to 3 hops of RELATED_TO edges.

        Hop=0 (direct):   DocumentChunk -[MENTIONS]-> Entity (name CONTAINS query)
        Hop=2 (1 bridge): e1 -[RELATED_TO]- e2 <-[MENTIONS]- Chunk
        Hop=3 (2 bridges): e1 -[REL]-> e2 -[REL]-> e3 <-[MENTIONS]- Chunk

        This design enables bridge-entity reasoning as described in thesis
        section 2.4 (original architectural contribution): a query about
        "Scott Derrickson" can reach chunks about "Ed Wood" via an
        intermediate director-relation hop, without loading all document
        text into memory (< 16 GB RAM constraint).

        Args:
            entity_name: Entity name substring to match.
            max_results: Maximum total results across all hops.

        Returns:
            List of dicts: chunk_id, text, source_file, matched_entity,
            hops, bridge_entity, relation_type.
        """
        if not entity_name.strip():
            return []

        chunks: List[Dict[str, Any]] = []
        seen: set[str] = set()

        name_variants = _entity_name_variants(entity_name)

        try:
            # Hop 0: direct MENTIONS — try name variants until we get a hit
            effective_name = entity_name  # will be updated on first hit
            for candidate in name_variants:
                res = self.conn.execute(
                    """
                    MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
                    WHERE e.name CONTAINS $entity_name
                    RETURN c.chunk_id, c.text, c.source_file, e.name
                    LIMIT $limit
                    """,
                    {"entity_name": candidate, "limit": max_results},
                )
                hop0_rows = []
                while res.has_next():
                    hop0_rows.append(res.get_next())
                if hop0_rows:
                    effective_name = candidate
                    if candidate != entity_name:
                        logger.info(
                            "Entity alias resolved: %r → %r", entity_name, candidate
                        )
                    for row in hop0_rows:
                        cid = row[0]
                        if cid and cid not in seen:
                            seen.add(cid)
                            chunks.append({
                                "chunk_id": cid,
                                "text": row[1],
                                "source_file": row[2],
                                "matched_entity": row[3],
                                "hops": 0,
                                "bridge_entity": None,
                                "relation_type": None,
                            })
                    break  # stop trying variants once we have results

            # Use effective_name (the variant that matched) for hop 2/3 queries
            entity_name = effective_name  # noqa: PLW2901 — intentional rebinding for hop queries

            # Hop 2: one RELATED_TO bridge (bidirectional)
            remaining = max_results - len(chunks)
            if remaining > 0:
                res = self.conn.execute(
                    """
                    MATCH (e1:Entity)-[r:RELATED_TO]-(e2:Entity)
                    WHERE e1.name CONTAINS $entity_name
                    WITH e2, r.relation_type AS rel
                    MATCH (c:DocumentChunk)-[:MENTIONS]->(e2)
                    RETURN c.chunk_id, c.text, c.source_file,
                           e2.name AS bridge, rel
                    LIMIT $limit
                    """,
                    {"entity_name": entity_name, "limit": remaining},
                )
                while res.has_next():
                    row = res.get_next()
                    cid = row[0]
                    if cid and cid not in seen:
                        seen.add(cid)
                        chunks.append({
                            "chunk_id": cid,
                            "text": row[1],
                            "source_file": row[2],
                            "matched_entity": entity_name,
                            "hops": 2,
                            "bridge_entity": row[3],
                            "relation_type": row[4],
                        })

            # Hop 3: two RELATED_TO bridges
            remaining = max_results - len(chunks)
            if remaining > 0:
                res = self.conn.execute(
                    """
                    MATCH (e1:Entity)-[:RELATED_TO]-(e2:Entity)-[r2:RELATED_TO]-(e3:Entity)
                    WHERE e1.name CONTAINS $entity_name
                      AND e3.name <> e1.name
                    WITH e3, e2.name AS mid_entity, r2.relation_type AS rel
                    MATCH (c:DocumentChunk)-[:MENTIONS]->(e3)
                    RETURN c.chunk_id, c.text, c.source_file,
                           e3.name AS bridge, mid_entity, rel
                    LIMIT $limit
                    """,
                    {"entity_name": entity_name, "limit": remaining},
                )
                while res.has_next():
                    row = res.get_next()
                    cid = row[0]
                    if cid and cid not in seen:
                        seen.add(cid)
                        chunks.append({
                            "chunk_id": cid,
                            "text": row[1],
                            "source_file": row[2],
                            "matched_entity": entity_name,
                            "hops": 3,
                            "bridge_entity": row[3],
                            # row[4] is the intermediate entity (mid_entity);
                            # row[5] is the relation type of the second edge.
                            "relation_type": row[5],
                        })

        except _STORAGE_ERRORS as exc:
            logger.error("find_chunks_by_entity_multihop failed: %s", exc)

        return chunks

    def get_document_structure(self, source_file: str) -> List[Dict[str, Any]]:
        """
        Return ordered chunks for a document (useful for diagnostic tools).

        Args:
            source_file: Source document filename.

        Returns:
            Ordered list of chunk dicts (chunk_id, text, page_number, chunk_index).
        """
        chunks: List[Dict[str, Any]] = []
        try:
            result = self.conn.execute(
                """
                MATCH (c:DocumentChunk {source_file: $source_file})
                RETURN c.chunk_id, c.text, c.page_number, c.chunk_index
                ORDER BY c.chunk_index ASC
                """,
                {"source_file": source_file},
            )
            while result.has_next():
                row = result.get_next()
                chunks.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "page_number": row[2],
                    "chunk_index": row[3],
                })
        except _STORAGE_ERRORS as exc:
            logger.error("get_document_structure failed: %s", exc)
        return chunks

    # ========================================================================
    # STATISTICS AND UTILITIES
    # ========================================================================

    def get_statistics(self) -> Dict[str, int]:
        """Return node and edge counts for all table types."""
        stats: Dict[str, int] = {
            "document_chunks": 0,
            "source_documents": 0,
            "entities": 0,
            "from_source_edges": 0,
            "next_chunk_edges": 0,
            "mentions_edges": 0,
            "related_to_edges": 0,
        }

        try:
            for label, key in [
                ("DocumentChunk", "document_chunks"),
                ("SourceDocument", "source_documents"),
                ("Entity", "entities"),
            ]:
                result = self.conn.execute(
                    "MATCH (n:%s) RETURN count(n)" % label
                )
                if result.has_next():
                    stats[key] = result.get_next()[0]

            for rel_type, key in [
                ("FROM_SOURCE", "from_source_edges"),
                ("NEXT_CHUNK", "next_chunk_edges"),
                ("MENTIONS", "mentions_edges"),
                ("RELATED_TO", "related_to_edges"),
            ]:
                try:
                    result = self.conn.execute(
                        "MATCH ()-[r:%s]->() RETURN count(r)" % rel_type
                    )
                    if result.has_next():
                        stats[key] = result.get_next()[0]
                except _STORAGE_ERRORS as exc:
                    logger.debug("Edge count for %s failed: %s", rel_type, exc)

        except _STORAGE_ERRORS as exc:
            logger.error("get_statistics failed: %s", exc)

        return stats

    def clear(self) -> None:
        """Delete all nodes and edges from the graph."""
        for rel_type in ["FROM_SOURCE", "NEXT_CHUNK", "MENTIONS", "RELATED_TO"]:
            try:
                self.conn.execute(
                    "MATCH ()-[r:%s]->() DELETE r" % rel_type
                )
            except _STORAGE_ERRORS as exc:
                logger.warning("Failed to clear edges %s: %s", rel_type, exc)

        for label in ["DocumentChunk", "SourceDocument", "Entity"]:
            try:
                self.conn.execute("MATCH (n:%s) DELETE n" % label)
            except _STORAGE_ERRORS as exc:
                logger.warning("Failed to clear nodes %s: %s", label, exc)

        logger.info("Graph cleared")

    def save(self) -> None:
        """No-op: KuzuDB auto-persists after every transaction."""
        logger.debug("KuzuDB auto-persists; no explicit save required")


# ============================================================================
# HYBRID STORE (Facade)
# ============================================================================

class HybridStore:
    """
    Unified facade over VectorStoreAdapter (LanceDB) and KuzuDB.

    Design pattern: Facade — hides storage heterogeneity from callers.
    Callers interact exclusively with `add_documents`, `vector_search`, and
    `graph_search`.

    KuzuDB is the only supported graph backend. If KuzuDB is not installed,
    HybridStore raises RuntimeError immediately rather than falling back to an
    alternative.  This is intentional: the thesis evaluates a specific system
    configuration; silent fallbacks would invalidate the evaluation results.
    """

    def __init__(
        self,
        config: StorageConfig,
        embeddings: Embeddings,
        entity_pipeline: "Optional[EntityExtractionPipeline]" = None,
    ) -> None:
        """
        Initialise the hybrid store.

        Args:
            config: StorageConfig instance.
            embeddings: Embeddings model used for both ingestion and
                        dimension auto-detection.
            entity_pipeline: Optional pre-constructed EntityExtractionPipeline.
                             If None and config.enable_entity_extraction is True,
                             the pipeline is constructed internally.
        """
        self.config = config
        self.embeddings = embeddings

        # Auto-detect embedding dimension from the live model if not set.
        # Use a local variable to avoid mutating the caller's StorageConfig
        # object — a side effect that could confuse shared config instances.
        embedding_dim = config.embedding_dim
        if embedding_dim is None:
            test_emb = embeddings.embed_query("dimension test")
            embedding_dim = len(test_emb)
            logger.info("Detected embedding dim: %d", embedding_dim)
        self.embedding_dim: int = embedding_dim

        # Vector store
        self.vector_store = VectorStoreAdapter(
            db_path=config.vector_db_path,
            embedding_dim=embedding_dim,
            normalize_embeddings=config.normalize_embeddings,
            distance_metric=config.distance_metric,
            overfetch_factor=config.overfetch_factor,
        )

        # Graph store — KuzuDB only. No fallback by design.
        if not KUZU_AVAILABLE:
            raise RuntimeError(
                "KuzuDB not available. Install with: pip install kuzu. "
                "No fallback is provided — the thesis evaluation requires KuzuDB."
            )
        self.graph_store: KuzuGraphStore = KuzuGraphStore(config.graph_db_path)
        logger.info("Using KuzuDB for graph storage")

        # Entity extraction pipeline (optional)
        self.entity_pipeline: Optional["EntityExtractionPipeline"] = entity_pipeline
        if self.entity_pipeline is None and config.enable_entity_extraction:
            self.entity_pipeline = self._init_entity_pipeline(config)

        logger.info("HybridStore initialised: dim=%d", embedding_dim)

    def _init_entity_pipeline(self, config: StorageConfig) -> "Optional[EntityExtractionPipeline]":
        """
        Construct EntityExtractionPipeline from settings when not injected.

        Separated from __init__ to keep the constructor readable and to make
        the pipeline injectable for testing (Dependency Inversion Principle).
        """
        try:
            from .entity_extraction import EntityExtractionPipeline, ExtractionConfig

            cache_path = config.entity_cache_path
            if cache_path is None:
                cache_path = Path(config.graph_db_path).parent / "entity_cache.db"

            extraction_config = ExtractionConfig(
                cache_enabled=True,
                cache_path=str(cache_path),
                # All NER/RE parameters sourced from settings.yaml via ExtractionConfig
                # defaults; override by passing a pre-configured pipeline instead.
            )
            pipeline = EntityExtractionPipeline(extraction_config)
            logger.info("Entity extraction pipeline initialised (cache: %s)", cache_path)
            return pipeline

        except ImportError as exc:
            logger.warning(
                "Entity extraction not available: %s. "
                "Install with: pip install gliner transformers",
                exc,
            )
        except _STORAGE_ERRORS as exc:
            logger.warning("Failed to initialise entity pipeline: %s", exc)

        return None

    def add_documents(self, documents: List[Document]) -> None:
        """
        Ingest documents into both the vector store and the knowledge graph.

        Ingests into both the vector store (LanceDB) and the knowledge graph
        (KuzuDB): creates DocumentChunk, SourceDocument nodes, FROM_SOURCE
        and NEXT_CHUNK edges, and optionally MENTIONS / RELATED_TO edges via
        the entity extraction pipeline.
        """
        if not documents:
            return

        # Embed and insert into vector store
        self.vector_store.add_documents_with_embeddings(documents, self.embeddings)

        max_chars = self.config.graph_text_max_chars
        prev_chunk_id: Optional[str] = None

        for doc in documents:
            chunk_id = str(doc.metadata.get("chunk_id", "unknown"))
            source_file = doc.metadata.get("source_file", "unknown")

            self.graph_store.add_document_chunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                page_number=doc.metadata.get("page_number", 0),
                chunk_index=doc.metadata.get("chunk_index", 0),
                source_file=source_file,
                max_text_chars=max_chars,
            )
            self.graph_store.add_source_document(
                doc_id=source_file,
                filename=source_file,
                total_pages=doc.metadata.get("total_pages", 0),
            )
            self.graph_store.add_from_source_relation(chunk_id, source_file)
            if prev_chunk_id:
                self.graph_store.add_next_chunk_relation(prev_chunk_id, chunk_id)

            prev_chunk_id = chunk_id

        # Entity extraction and graph integration
        if self.entity_pipeline:
            try:
                entity_stats = self._integrate_entities(documents)
                logger.info(
                    "Entity integration: %d entities, %d mentions, %d relations",
                    entity_stats.get("unique_entities", 0),
                    entity_stats.get("total_mentions", 0),
                    entity_stats.get("total_relations", 0),
                )
            except _STORAGE_ERRORS as exc:
                logger.warning("Entity integration failed: %s", exc)

        logger.info("Added %d documents to hybrid store", len(documents))

    def _integrate_entities(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Extract entities and relations and integrate them into the graph.

        Pipeline (thesis section 2.5):
        1. Batch entity extraction with GLiNER (configured batch size).
        2. Add unique Entity nodes (deduplicated by entity_id).
        3. Create MENTIONS edges: DocumentChunk → Entity.
        4. Create RELATED_TO edges from REBEL relation extraction.

        Args:
            documents: Documents already present in vector and graph stores.

        Returns:
            Statistics dict with extraction metrics.
        """
        if not self.entity_pipeline:
            return {}

        stats: Dict[str, Any] = {
            "total_chunks": len(documents),
            "chunks_with_entities": 0,
            "total_entities": 0,
            "unique_entities": 0,
            "total_mentions": 0,
            "total_relations": 0,
        }

        if not documents:
            return stats

        texts = [doc.page_content for doc in documents]
        chunk_ids = [
            str(doc.metadata.get("chunk_id", "chunk_%d" % i))
            for i, doc in enumerate(documents)
        ]

        logger.debug("Extracting entities from %d chunks...", len(documents))
        extraction_results = self.entity_pipeline.process_chunks_batch(texts, chunk_ids)

        seen_entities: set = set()
        entity_name_to_id: Dict[str, str] = {}

        # Add entity nodes and MENTIONS edges
        for result in extraction_results:
            if not result.entities:
                continue

            stats["chunks_with_entities"] += 1

            for entity in result.entities:
                stats["total_entities"] += 1
                entity_name_to_id[entity.name.lower()] = entity.entity_id

                if entity.entity_id not in seen_entities:
                    try:
                        self.graph_store.add_entity(
                            entity_id=entity.entity_id,
                            name=entity.name,
                            entity_type=entity.entity_type,
                            confidence=entity.confidence,
                        )
                        seen_entities.add(entity.entity_id)
                        stats["unique_entities"] += 1
                    except _STORAGE_ERRORS as exc:
                        logger.debug(
                            "Entity add failed (may exist): %s — %s",
                            entity.entity_id,
                            exc,
                        )

                try:
                    self.graph_store.add_mentions_relation(
                        chunk_id=result.chunk_id,
                        entity_id=entity.entity_id,
                    )
                    stats["total_mentions"] += 1
                except _STORAGE_ERRORS as exc:
                    logger.debug("MENTIONS relation failed: %s", exc)

        # Add RELATED_TO edges from REBEL relation extraction.
        # NOTE (known limitation): entity_name_to_id is populated only from
        # the current batch.  If a relation's subject or object was extracted
        # in a previous ingestion batch, its entity_id will not be found here
        # and the relation is silently skipped.  For full cross-document
        # relation coverage, a single-batch ingestion (or a post-hoc graph
        # repair pass) is required.  This limitation is documented in thesis
        # section 2.5.
        for result in extraction_results:
            for relation in result.relations:
                try:
                    subject_id = entity_name_to_id.get(relation.subject_entity.lower())
                    object_id = entity_name_to_id.get(relation.object_entity.lower())
                    if subject_id and object_id:
                        self.graph_store.add_related_to_relation(
                            entity1_id=subject_id,
                            entity2_id=object_id,
                            relation_type=relation.relation_type,
                        )
                        stats["total_relations"] += 1
                except _STORAGE_ERRORS as exc:
                    logger.debug("RELATED_TO relation failed: %s", exc)

        return stats

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Delegate to VectorStoreAdapter.vector_search.

        Args:
            query_embedding: Dense query vector (same dim as stored embeddings).
            top_k: Maximum results to return.
            threshold: Minimum similarity score.

        Returns:
            List of dicts with text, similarity, document_id, metadata.
        """
        return self.vector_store.vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

    def graph_search(
        self,
        entities: List[str],
        max_hops: int = 2,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Entity-driven graph retrieval.

        Uses KuzuDB find_chunks_by_entity_multihop (up to 3 hops).
        Results are sorted by hop distance (0 = direct mention = best).

        Args:
            entities: Entity name strings extracted from the query.
            max_hops: Unused (KuzuDB multihop is fixed at 3 hops internally).
            top_k: Maximum results to return.

        Returns:
            List of dicts: chunk_id, text, source_file, matched_entity,
            hops, bridge_entity, relation_type.
        """
        results: List[Dict[str, Any]] = []
        seen_chunks: set = set()

        for entity_name in entities:
            entity_chunks = self.graph_store.find_chunks_by_entity_multihop(
                entity_name=entity_name,
                max_results=top_k,
            )

            for chunk in entity_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    results.append({
                        "chunk_id": chunk_id,
                        "text": chunk.get("text", ""),
                        "source_file": chunk.get("source_file", ""),
                        "matched_entity": chunk.get("matched_entity", entity_name),
                        "hops": chunk.get("hops", 0),
                        "bridge_entity": chunk.get("bridge_entity"),
                        "relation_type": chunk.get("relation_type"),
                    })

        results.sort(key=lambda x: x.get("hops", 999))
        return results[:top_k]

    def save(self) -> None:
        """Persist the graph store (no-op for KuzuDB — data is written on each operation)."""
        self.graph_store.save()

    def reset_vector_store(self) -> None:
        """Wipe and re-create the vector store (used for ablation studies)."""
        if self.config.vector_db_path.exists():
            shutil.rmtree(self.config.vector_db_path)
        self.vector_store = VectorStoreAdapter(
            self.config.vector_db_path,
            self.embedding_dim,
            self.config.normalize_embeddings,
            self.config.distance_metric,
            self.config.overfetch_factor,
        )
        logger.info("Vector store reset")

    def reset_graph_store(self) -> None:
        """Clear all graph data (used for ablation studies)."""
        self.graph_store.clear()
        logger.info("Graph store reset")

    def reset_all(self) -> None:
        """Reset both vector and graph stores."""
        self.reset_vector_store()
        self.reset_graph_store()


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics(config: StorageConfig, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Run diagnostic checks on storage configuration.

    Args:
        config: StorageConfig to validate.
        embeddings: Embeddings model to probe for dimension detection.

    Returns:
        Dict with embedding_dim, graph_backend, availability flags, issues list.
    """
    results: Dict[str, Any] = {
        "embedding_dim": None,
        "kuzu_available": KUZU_AVAILABLE,
        "issues": [],
    }

    try:
        test_emb = embeddings.embed_query("diagnostic test")
        results["embedding_dim"] = len(test_emb)
    except Exception as exc:
        results["issues"].append("Embedding failed: %s" % exc)

    if not KUZU_AVAILABLE:
        results["issues"].append("KuzuDB not installed — install with: pip install kuzu")

    return results


# ============================================================================
# FACTORY
# ============================================================================

def create_storage_config(
    cfg: Optional[Dict[str, Any]] = None,
    dataset: str = "default",
) -> "StorageConfig":
    """
    Build a StorageConfig from a settings.yaml configuration dictionary.

    This is the canonical way to construct StorageConfig in production code.
    It keeps storage.py decoupled from YAML-loading and ensures all paths and
    thresholds are sourced from config/settings.yaml rather than scattered
    across caller code.

    Parameter mapping from settings.yaml:
        vector_store.db_path          → vector_db_path  (with dataset sub-dir)
        graph.graph_path              → graph_db_path   (with dataset sub-dir)
        embeddings.embedding_dim      → embedding_dim
        vector_store.similarity_threshold  → similarity_threshold
        vector_store.normalize_embeddings  → normalize_embeddings
        vector_store.distance_metric       → distance_metric
        graph.backend                      → graph_backend
        ingestion.extract_entities         → enable_entity_extraction

    Args:
        cfg:     Full settings dict as loaded from config/settings.yaml.
                 Pass None or {} to fall back to class-level defaults.
                 Paths will then resolve to ./data/default/vector_db and
                 ./data/default/knowledge_graph.
        dataset: Dataset sub-directory name (e.g. "hotpotqa"). Appended to
                 the base paths from settings.yaml so each dataset gets an
                 isolated store, preventing cross-dataset data leakage.

    Returns:
        StorageConfig (frozen dataclass).
    """
    cfg = cfg or {}
    vs_cfg = cfg.get("vector_store", {})
    gr_cfg = cfg.get("graph", {})
    emb_cfg = cfg.get("embeddings", {})
    ing_cfg = cfg.get("ingestion", {})

    base_vector = Path(vs_cfg.get("db_path", "./data/vector_db"))
    base_graph = Path(gr_cfg.get("graph_path", "./data/knowledge_graph"))

    # Append dataset sub-directory when a named dataset is specified
    if dataset and dataset != "default":
        vector_db_path = base_vector.parent / dataset / base_vector.name
        graph_db_path = base_graph.parent / dataset / base_graph.name
    else:
        vector_db_path = base_vector
        graph_db_path = base_graph

    return StorageConfig(
        vector_db_path=vector_db_path,
        graph_db_path=graph_db_path,
        embedding_dim=emb_cfg.get("embedding_dim", None),
        similarity_threshold=vs_cfg.get("similarity_threshold", 0.3),
        normalize_embeddings=vs_cfg.get("normalize_embeddings", True),
        distance_metric=vs_cfg.get("distance_metric", "cosine"),
        graph_backend=gr_cfg.get("backend", "kuzu"),
        enable_entity_extraction=ing_cfg.get("extract_entities", False),
    )


# ============================================================================
# SELF-VERIFICATION
# ============================================================================

def _main() -> None:
    """Smoke demo and test runner for direct module invocation."""
    import sys
    import subprocess
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --- smoke demo -----------------------------------------------------------
    class _MockEmbeddings:
        """Deterministic mock embeddings for smoke testing."""
        def __init__(self, dim: int = 64) -> None:
            self.dim = dim

        def embed_documents(self, texts):
            results = []
            for text in texts:
                # Content-based seed: deterministic across processes, no global RNG mutation.
                seed = sum(ord(c) for c in text[:64]) % 10000
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self.dim).astype(np.float32)
                vec /= np.linalg.norm(vec) + 1e-8
                results.append(vec.tolist())
            return results

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(
            vector_db_path=Path(tmpdir) / "vector",
            graph_db_path=Path(tmpdir) / "graph",
            embedding_dim=64,
        )
        store = HybridStore(config, _MockEmbeddings(dim=64))

        docs = [
            Document(
                page_content="Albert Einstein developed the theory of relativity.",
                metadata={"chunk_id": "c1", "source_file": "physics.pdf",
                          "chunk_index": 0, "page_number": 1},
            ),
            Document(
                page_content="He received the Nobel Prize in Physics in 1921.",
                metadata={"chunk_id": "c2", "source_file": "physics.pdf",
                          "chunk_index": 1, "page_number": 1},
            ),
        ]
        store.add_documents(docs)

        query_emb = _MockEmbeddings(dim=64).embed_query("relativity")
        # threshold=0.0: smoke test verifies plumbing, not semantic quality.
        results = store.vector_search(query_emb, top_k=2, threshold=0.0)
        assert results, "Expected at least one vector result"
        logger.info("Smoke demo: vector_search returned %d results", len(results))

        if KUZU_AVAILABLE:
            graph_results = store.graph_search(["Einstein"], top_k=5)
            logger.info(
                "Smoke demo: graph_search returned %d results", len(graph_results)
            )

    logger.info("Smoke demo passed.")

    # --- pytest ---------------------------------------------------------------
    test_file = Path(__file__).parent / "test_data_layer.py"
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", str(test_file),
         "-v", "-k", "storage or Storage or VectorStore or HybridStore"],
        check=False,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    _main()
