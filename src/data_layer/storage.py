"""
Hybrid Storage Module: Vector Store (LanceDB) + Knowledge Graph (KuzuDB)

Version: 3.0.0 - KUZU INTEGRATION
Author: Edge-RAG Research Project
Last Modified: 2026-01-25

===============================================================================
MAJOR CHANGE: NetworkX → KuzuDB
===============================================================================

Previous Implementation (v2.1.0):
    - NetworkX for in-memory graph
    - GraphML file persistence
    - Python-based traversal (slow for large graphs)
    
New Implementation (v3.0.0):
    - KuzuDB embedded graph database
    - Cypher query language
    - Native multi-hop traversal (10-100x faster)
    - ACID transactions
    - Columnar storage (memory efficient)

SCIENTIFIC RATIONALE:

KuzuDB advantages for Edge-RAG:
1. Embedded: No server process (like LanceDB)
2. Cypher: Standard graph query language
3. Performance: Columnar storage, vectorized execution
4. Multi-hop: Native path queries (vs BFS in Python)

Reference: Feng et al. (2023). "Kùzu Graph Database Management System."
CIDR 2023.

===============================================================================
GRAPH SCHEMA
===============================================================================

Node Types:
    - DocumentChunk: Text chunks from ingested documents
    - SourceDocument: Original PDF/document files
    - Entity: Extracted named entities (optional)
    - Concept: Domain concepts (optional)

Edge Types:
    - FROM_SOURCE: DocumentChunk -> SourceDocument
    - NEXT_CHUNK: DocumentChunk -> DocumentChunk (sequential)
    - MENTIONS: DocumentChunk -> Entity
    - RELATED_TO: Entity -> Entity
    - SIMILAR_TO: DocumentChunk -> DocumentChunk (semantic similarity)

Cypher Schema:
    CREATE NODE TABLE DocumentChunk(
        chunk_id STRING PRIMARY KEY,
        text STRING,
        page_number INT64,
        chunk_index INT64,
        embedding_id STRING
    )
    
    CREATE NODE TABLE SourceDocument(
        doc_id STRING PRIMARY KEY,
        filename STRING,
        total_pages INT64,
        ingestion_timestamp TIMESTAMP
    )
    
    CREATE REL TABLE FROM_SOURCE(
        FROM DocumentChunk TO SourceDocument
    )

===============================================================================
"""

import logging
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings

# LanceDB for vector storage
import lancedb

# KuzuDB for graph storage
try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False
    logging.warning(
        "KuzuDB not available. Install with: pip install kuzu"
    )

# Fallback to NetworkX if Kuzu not available
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class StorageConfig:
    """
    Configuration for Hybrid Storage System.
    
    Attributes:
        vector_db_path: Directory path for LanceDB vector database
        graph_db_path: Directory path for KuzuDB graph database
        embedding_dim: Dimensionality of embedding vectors
        similarity_threshold: Minimum similarity score for retrieval
        normalize_embeddings: Whether to L2-normalize vectors
        distance_metric: Distance metric for vector search
        graph_backend: "kuzu" or "networkx" (fallback)
    """
    vector_db_path: Path
    graph_db_path: Path
    embedding_dim: Optional[int] = None
    similarity_threshold: float = 0.3
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"
    graph_backend: str = "kuzu"  # NEW: "kuzu" or "networkx"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive: {self.embedding_dim}")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be in [0,1]: {self.similarity_threshold}")
        
        if self.distance_metric not in ("cosine", "l2"):
            raise ValueError(f"distance_metric must be 'cosine' or 'l2': {self.distance_metric}")
        
        if self.graph_backend not in ("kuzu", "networkx"):
            raise ValueError(f"graph_backend must be 'kuzu' or 'networkx': {self.graph_backend}")


# ============================================================================
# VECTOR STORE ADAPTER (LanceDB) - UNCHANGED FROM v2.1.0
# ============================================================================

class VectorStoreAdapter:
    """
    LanceDB Vector Store Adapter with correct distance metric handling.
    [Same as v2.1.0 - no changes needed]
    """

    SCHEMA_VERSION = "3.0.0"
    TABLE_NAME = "documents"
    
    def __init__(
        self, 
        db_path: Path, 
        embedding_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        distance_metric: str = "cosine"
    ):
        if distance_metric not in ("cosine", "l2"):
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db = lancedb.connect(str(db_path))
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        self.distance_metric = distance_metric
        self.table = None
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"VectorStoreAdapter initialized: "
            f"path={db_path}, metric={distance_metric}"
        )
        
        self._load_metadata()

    def _get_metadata_path(self) -> Path:
        return self.db_path.parent / "vector_store_metadata.json"
    
    def _load_metadata(self) -> None:
        metadata_path = self._get_metadata_path()
        if not metadata_path.exists():
            return
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            stored_dim = metadata.get("embedding_dim")
            if stored_dim and self.embedding_dim is None:
                self.embedding_dim = stored_dim
        except Exception as e:
            self.logger.debug(f"Could not load metadata: {e}")
    
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
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
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
                f"EMBEDDING DIMENSION MISMATCH: expected {self.embedding_dim}, got {actual_dim}"
            )

    def _distance_to_similarity(self, distance: float) -> float:
        if self.distance_metric == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        elif self.distance_metric == "l2":
            return 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: Embeddings,
    ) -> None:
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        self.logger.info(f"Generating embeddings for {len(texts)} documents...")
        
        start_time = time.time()
        embeddings_list = embeddings.embed_documents(texts)
        embed_time = time.time() - start_time
        
        self.logger.info(f"Embeddings generated in {embed_time:.2f}s")
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
                self.table = self.db.create_table(self.TABLE_NAME, data=data, mode="overwrite")
            else:
                self.table.add(data)
            self._save_metadata()
        except Exception as e:
            self.logger.error(f"Failed to insert documents: {e}")
            raise

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
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
                .limit(top_k * 3)
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

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []


# ============================================================================
# KUZU GRAPH STORE (NEW!)
# ============================================================================

class KuzuGraphStore:
    """
    KuzuDB-based Knowledge Graph Store.
    
    ADVANTAGES OVER NETWORKX:
    
    1. Native Cypher Support:
       - Standard graph query language
       - Complex path queries in single statement
       - Aggregations, filtering, projections
    
    2. Performance:
       - Columnar storage (cache-efficient)
       - Vectorized query execution
       - Native multi-hop traversal
       - 10-100x faster than Python BFS for large graphs
    
    3. Persistence:
       - ACID transactions
       - Crash recovery
       - No need for manual save/load
    
    4. Memory Efficiency:
       - Out-of-core processing
       - Memory-mapped files
       - Handles graphs larger than RAM
    
    SCHEMA:
    
    Node Tables:
        DocumentChunk(chunk_id, text, page_number, chunk_index, source_file)
        SourceDocument(doc_id, filename, total_pages)
        Entity(entity_id, name, entity_type)
    
    Relationship Tables:
        FROM_SOURCE(DocumentChunk -> SourceDocument)
        NEXT_CHUNK(DocumentChunk -> DocumentChunk)
        MENTIONS(DocumentChunk -> Entity)
        RELATED_TO(Entity -> Entity, relation_type)
    """
    
    SCHEMA_VERSION = "3.0.0"
    
    def __init__(self, db_path: Path):
        """
        Initialize KuzuDB graph store.
        
        Args:
            db_path: DIRECTORY path for KuzuDB database
        
        CRITICAL FIX for KuzuDB 0.11.3+:
            KuzuDB wants the PARENT directory to exist, but will create
            the database directory itself. DO NOT create db_path!
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        
        if not KUZU_AVAILABLE:
            raise ImportError("KuzuDB not installed. Install with: pip install kuzu")
        
        # ===================================================================
        # CRITICAL FIX: Create PARENT directory only, let KuzuDB create db_path
        # ===================================================================
        # KuzuDB 0.11.3+ will create the db_path directory itself
        # We just need to ensure the parent exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize KuzuDB - it will create db_path directory
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)
        
        # Initialize schema
        self._init_schema()
        
        self.logger.info(f"KuzuGraphStore initialized: {self.db_path}")
        
    def _init_schema(self) -> None:
        """
        Initialize graph schema (creates tables if not exist).
        
        KuzuDB requires explicit schema definition before inserting data.
        """
        try:
            # Node Tables
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
                    entity_type STRING,
                    PRIMARY KEY (entity_id)
                )
            """)
            
            # Relationship Tables
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
                    relation_type STRING
                )
            """)

            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    entity_id STRING,
                    name STRING,
                    type STRING,
                    mention_count INT64,
                    PRIMARY KEY (entity_id)
                )
            """)

            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MENTIONS(
                    FROM DocumentChunk TO Entity,
                    mention_span STRING,
                    confidence DOUBLE
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
            
            self.logger.debug("Graph schema initialized")
            
        except Exception as e:
            # Tables might already exist
            self.logger.debug(f"Schema init note: {e}")
    
    def add_document_chunk(
        self,
        chunk_id: str,
        text: str,
        page_number: int,
        chunk_index: int,
        source_file: str,
    ) -> None:
        """
        Add a document chunk node.
        
        Args:
            chunk_id: Unique identifier for chunk
            text: Chunk text content (truncated for storage)
            page_number: Source page number
            chunk_index: Sequential chunk index
            source_file: Source document filename
        """
        # Truncate text for graph storage (full text is in vector store)
        truncated_text = text[:500] if len(text) > 500 else text
        
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
                    "text": truncated_text,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "source_file": source_file,
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to add chunk {chunk_id}: {e}")
    
    def add_source_document(
        self,
        doc_id: str,
        filename: str,
        total_pages: int = 0,
    ) -> None:
        """Add a source document node."""
        try:
            self.conn.execute(
                """
                MERGE (d:SourceDocument {doc_id: $doc_id})
                SET d.filename = $filename,
                    d.total_pages = $total_pages
                """,
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "total_pages": total_pages,
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to add source doc {doc_id}: {e}")
    
    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "unknown",
    ) -> None:
        """Add an entity node."""
        try:
            self.conn.execute(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.name = $name,
                    e.entity_type = $entity_type
                """,
                {
                    "entity_id": entity_id,
                    "name": name,
                    "entity_type": entity_type,
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to add entity {entity_id}: {e}")
    
    def add_from_source_relation(self, chunk_id: str, doc_id: str) -> None:
        """Create FROM_SOURCE relationship between chunk and document."""
        try:
            self.conn.execute(
                """
                MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                MATCH (d:SourceDocument {doc_id: $doc_id})
                MERGE (c)-[:FROM_SOURCE]->(d)
                """,
                {"chunk_id": chunk_id, "doc_id": doc_id}
            )
        except Exception as e:
            self.logger.debug(f"FROM_SOURCE relation note: {e}")
    
    def add_next_chunk_relation(self, chunk_id: str, next_chunk_id: str) -> None:
        """Create NEXT_CHUNK relationship for sequential ordering."""
        try:
            self.conn.execute(
                """
                MATCH (c1:DocumentChunk {chunk_id: $chunk_id})
                MATCH (c2:DocumentChunk {chunk_id: $next_chunk_id})
                MERGE (c1)-[:NEXT_CHUNK]->(c2)
                """,
                {"chunk_id": chunk_id, "next_chunk_id": next_chunk_id}
            )
        except Exception as e:
            self.logger.debug(f"NEXT_CHUNK relation note: {e}")
    
    def add_mentions_relation(self, chunk_id: str, entity_id: str) -> None:
        """Create MENTIONS relationship between chunk and entity."""
        try:
            self.conn.execute(
                """
                MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                MATCH (e:Entity {entity_id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {"chunk_id": chunk_id, "entity_id": entity_id}
            )
        except Exception as e:
            self.logger.debug(f"MENTIONS relation note: {e}")
    
    def add_related_to_relation(
        self,
        entity1_id: str,
        entity2_id: str,
        relation_type: str = "related",
    ) -> None:
        """Create RELATED_TO relationship between entities."""
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
                }
            )
        except Exception as e:
            self.logger.debug(f"RELATED_TO relation note: {e}")
    
    # ========================================================================
    # GRAPH TRAVERSAL (Cypher-based)
    # ========================================================================
    
    def graph_traversal(
        self,
        start_entity: str,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> Dict[str, int]:
        """
        Perform multi-hop graph traversal using Cypher.
        
        Args:
            start_entity: Starting node ID (chunk_id or entity_id)
            relation_types: Filter by relation types (None = all)
            max_hops: Maximum traversal depth
            
        Returns:
            Dict mapping node_id -> hop_distance
        """
        visited = {start_entity: 0}
        
        try:
            # Try as DocumentChunk first - FIXED: Use undirected pattern
            result = self.conn.execute(
                f"""
                MATCH (start:DocumentChunk {{chunk_id: $start_id}})
                MATCH path = (start)-[*1..{max_hops}]-(connected:DocumentChunk)
                RETURN DISTINCT 
                    connected.chunk_id AS node_id,
                    length(path) AS hops
                """,
                {"start_id": start_entity}
            )
            
            while result.has_next():
                row = result.get_next()
                node_id, hops = row[0], row[1]
                if node_id and (node_id not in visited or visited[node_id] > hops):
                    visited[node_id] = hops
            
        except Exception as e:
            self.logger.debug(f"Traversal as DocumentChunk: {e}")
            
            # Try as Entity
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
                    {"start_id": start_entity}
                )
                
                while result.has_next():
                    row = result.get_next()
                    node_id, hops = row[0], row[1]
                    if node_id and (node_id not in visited or visited[node_id] > hops):
                        visited[node_id] = hops
                        
            except Exception as e2:
                self.logger.debug(f"Traversal as Entity: {e2}")
        
        return visited
    
    def find_related_chunks(
        self,
        chunk_id: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks related to a given chunk through any path.
        
        Useful for expanding retrieval context.
        
        Args:
            chunk_id: Starting chunk ID
            max_hops: Maximum path length
            
        Returns:
            List of related chunks with hop distance
        """
        related = []
        
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
                {"chunk_id": chunk_id}
            )
            
            while result.has_next():
                row = result.get_next()
                related.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "source_file": row[2],
                    "hops": row[3],
                })
                
        except Exception as e:
            self.logger.error(f"find_related_chunks failed: {e}")
        
        return related
    
    def find_chunks_by_entity(
        self,
        entity_name: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks that mention a specific entity.
        
        Args:
            entity_name: Entity name to search for
            max_results: Maximum results to return
            
        Returns:
            List of chunks mentioning the entity
        """
        chunks = []
        
        try:
            result = self.conn.execute(
                """
                MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
                WHERE e.name CONTAINS $entity_name
                RETURN c.chunk_id, c.text, c.source_file, e.name
                LIMIT $limit
                """,
                {"entity_name": entity_name, "limit": max_results}
            )
            
            while result.has_next():
                row = result.get_next()
                chunks.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "source_file": row[2],
                    "entity": row[3],
                })
                
        except Exception as e:
            self.logger.error(f"find_chunks_by_entity failed: {e}")
        
        return chunks
    
    def get_document_structure(self, source_file: str) -> List[Dict[str, Any]]:
        """
        Get ordered chunks for a document (for context expansion).
        
        Args:
            source_file: Source document filename
            
        Returns:
            Ordered list of chunks from the document
        """
        chunks = []
        
        try:
            result = self.conn.execute(
                """
                MATCH (c:DocumentChunk {source_file: $source_file})
                RETURN c.chunk_id, c.text, c.page_number, c.chunk_index
                ORDER BY c.chunk_index ASC
                """,
                {"source_file": source_file}
            )
            
            while result.has_next():
                row = result.get_next()
                chunks.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "page_number": row[2],
                    "chunk_index": row[3],
                })
                
        except Exception as e:
            self.logger.error(f"get_document_structure failed: {e}")
        
        return chunks
    
    # ========================================================================
    # STATISTICS AND UTILITIES
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, int]:
        """Get graph statistics."""
        stats = {
            "document_chunks": 0,
            "source_documents": 0,
            "entities": 0,
            "from_source_edges": 0,
            "next_chunk_edges": 0,
            "mentions_edges": 0,
            "related_to_edges": 0,
        }
        
        try:
            # Count nodes
            for label, key in [
                ("DocumentChunk", "document_chunks"),
                ("SourceDocument", "source_documents"),
                ("Entity", "entities"),
            ]:
                result = self.conn.execute(f"MATCH (n:{label}) RETURN count(n)")
                if result.has_next():
                    stats[key] = result.get_next()[0]
            
            # Count edges
            for rel_type, key in [
                ("FROM_SOURCE", "from_source_edges"),
                ("NEXT_CHUNK", "next_chunk_edges"),
                ("MENTIONS", "mentions_edges"),
                ("RELATED_TO", "related_to_edges"),
            ]:
                try:
                    result = self.conn.execute(f"MATCH ()-[r:{rel_type}]->() RETURN count(r)")
                    if result.has_next():
                        stats[key] = result.get_next()[0]
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"get_statistics failed: {e}")
        
        return stats
    
    def clear(self) -> None:
        """Clear all data from graph."""
        try:
            # Delete all relationships first
            for rel_type in ["FROM_SOURCE", "NEXT_CHUNK", "MENTIONS", "RELATED_TO"]:
                try:
                    self.conn.execute(f"MATCH ()-[r:{rel_type}]->() DELETE r")
                except:
                    pass
            
            # Delete all nodes
            for label in ["DocumentChunk", "SourceDocument", "Entity"]:
                try:
                    self.conn.execute(f"MATCH (n:{label}) DELETE n")
                except:
                    pass
            
            self.logger.info("Graph cleared")
        except Exception as e:
            self.logger.error(f"clear failed: {e}")
    
    # ========================================================================
    # COMPATIBILITY: NetworkX-like interface
    # ========================================================================
    
    @property
    def graph(self):
        """
        Compatibility property for code expecting NetworkX graph.
        
        Returns a minimal wrapper that provides basic graph interface.
        """
        return _KuzuGraphWrapper(self)
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Compatibility method for KnowledgeGraphStore interface.
        
        Maps to appropriate KuzuDB node type.
        """
        if entity_type == "document_chunk":
            self.add_document_chunk(
                chunk_id=entity_id,
                text=metadata.get("text", ""),
                page_number=metadata.get("page_number", 0),
                chunk_index=metadata.get("chunk_index", 0),
                source_file=metadata.get("source_file", "unknown"),
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
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Compatibility method for KnowledgeGraphStore interface.
        
        Maps relation_type to appropriate KuzuDB relationship.
        """
        if relation_type == "from_source":
            self.add_from_source_relation(source_id, target_id)
        elif relation_type == "next_chunk":
            self.add_next_chunk_relation(source_id, target_id)
        elif relation_type == "mentions":
            self.add_mentions_relation(source_id, target_id)
        else:
            self.add_related_to_relation(source_id, target_id, relation_type)
    
    def save(self) -> None:
        """
        Compatibility method - KuzuDB auto-persists.
        """
        self.logger.debug("KuzuDB auto-persists, no explicit save needed")


class _KuzuGraphWrapper:
    """
    Minimal wrapper for KuzuDB to provide NetworkX-like interface.
    
    Used for backwards compatibility with existing code.
    """
    
    def __init__(self, kuzu_store: KuzuGraphStore):
        self._store = kuzu_store
    
    def number_of_nodes(self) -> int:
        stats = self._store.get_statistics()
        return (
            stats.get("document_chunks", 0) +
            stats.get("source_documents", 0) +
            stats.get("entities", 0)
        )
    
    def number_of_edges(self) -> int:
        stats = self._store.get_statistics()
        return (
            stats.get("from_source_edges", 0) +
            stats.get("next_chunk_edges", 0) +
            stats.get("mentions_edges", 0) +
            stats.get("related_to_edges", 0)
        )
    
    def nodes(self, data: bool = False):
        """Get all node IDs."""
        nodes = []
        try:
            for label in ["DocumentChunk", "SourceDocument", "Entity"]:
                id_field = {
                    "DocumentChunk": "chunk_id",
                    "SourceDocument": "doc_id",
                    "Entity": "entity_id",
                }[label]
                
                result = self._store.conn.execute(
                    f"MATCH (n:{label}) RETURN n.{id_field}"
                )
                while result.has_next():
                    nodes.append(result.get_next()[0])
        except:
            pass
        return nodes


# ============================================================================
# NETWORKX FALLBACK (for systems without KuzuDB)
# ============================================================================

class NetworkXGraphStore:
    """
    NetworkX-based Knowledge Graph (fallback).
    
    Use this if KuzuDB is not available.
    Same interface as KuzuGraphStore for compatibility.
    """
    
    def __init__(self, graph_path: Path):
        self.graph_path = Path(graph_path)
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)
        
        if self.graph_path.exists():
            try:
                self.graph = nx.read_graphml(str(self.graph_path))
                self.logger.info(f"Loaded NetworkX graph: {self.graph.number_of_nodes()} nodes")
            except Exception as e:
                self.logger.error(f"Failed to load graph: {e}")
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        self.graph.add_node(entity_id, entity_type=entity_type, **metadata)
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        edge_data = {"relation_type": relation_type}
        if metadata:
            edge_data.update(metadata)
        self.graph.add_edge(source_id, target_id, **edge_data)
    
    def graph_traversal(
        self,
        start_entity: str,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> Dict[str, int]:
        if start_entity not in self.graph:
            return {}
        
        visited = {start_entity: 0}
        queue = [(start_entity, 0)]
        
        while queue:
            current, hops = queue.pop(0)
            if hops >= max_hops:
                continue
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    edge_data = self.graph.get_edge_data(current, neighbor)
                    rel_type = edge_data.get("relation_type")
                    
                    if relation_types is None or rel_type in relation_types:
                        visited[neighbor] = hops + 1
                        queue.append((neighbor, hops + 1))
        
        return visited
    
    def save(self) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, str(self.graph_path))
    
    def get_statistics(self) -> Dict[str, int]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }
    
    def clear(self) -> None:
        self.graph.clear()


# ============================================================================
# HYBRID STORE (Facade Pattern)
# ============================================================================

class HybridStore:
    """
    Unified interface for Vector Store and Knowledge Graph.
    
    Automatically selects KuzuDB or NetworkX based on availability
    and configuration.
    """
    
    def __init__(self, config: StorageConfig, embeddings: Embeddings):
        self.config = config
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect embedding dimension
        if config.embedding_dim is None:
            test_emb = embeddings.embed_query("dimension test")
            config.embedding_dim = len(test_emb)
            self.logger.info(f"Detected embedding dim: {config.embedding_dim}")
        
        # Initialize Vector Store
        self.vector_store = VectorStoreAdapter(
            db_path=config.vector_db_path,
            embedding_dim=config.embedding_dim,
            normalize_embeddings=config.normalize_embeddings,
            distance_metric=config.distance_metric,
        )
        
        # Initialize Graph Store (KuzuDB or NetworkX)
        if config.graph_backend == "kuzu" and KUZU_AVAILABLE:
            self.graph_store = KuzuGraphStore(config.graph_db_path)
            self.logger.info("Using KuzuDB for graph storage")
        elif NETWORKX_AVAILABLE:
            self.graph_store = NetworkXGraphStore(config.graph_db_path)
            self.logger.info("Using NetworkX for graph storage (fallback)")
        else:
            raise ImportError("No graph backend available!")
        
        self.logger.info(f"HybridStore initialized: dim={config.embedding_dim}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to both vector and graph stores."""
        if not documents:
            return
        
        # Add to vector store
        self.vector_store.add_documents_with_embeddings(documents, self.embeddings)
        
        # Add to graph store
        prev_chunk_id = None
        
        for doc in documents:
            chunk_id = str(doc.metadata.get("chunk_id", "unknown"))
            source_file = doc.metadata.get("source_file", "unknown")
            
            # Add document chunk node
            if isinstance(self.graph_store, KuzuGraphStore):
                self.graph_store.add_document_chunk(
                    chunk_id=chunk_id,
                    text=doc.page_content[:500],
                    page_number=doc.metadata.get("page_number", 0),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    source_file=source_file,
                )
                
                # Add source document node
                self.graph_store.add_source_document(
                    doc_id=source_file,
                    filename=source_file,
                    total_pages=doc.metadata.get("total_pages", 0),
                )
                
                # Add FROM_SOURCE relation
                self.graph_store.add_from_source_relation(chunk_id, source_file)
                
                # Add NEXT_CHUNK relation (sequential ordering)
                if prev_chunk_id:
                    self.graph_store.add_next_chunk_relation(prev_chunk_id, chunk_id)
                
            else:
                # NetworkX fallback
                self.graph_store.add_entity(chunk_id, "document_chunk", {"source_file": source_file})
                self.graph_store.add_entity(source_file, "source_document", {})
                self.graph_store.add_relation(chunk_id, source_file, "from_source")
            
            prev_chunk_id = chunk_id
        
        self.logger.info(f"Added {len(documents)} documents to hybrid store")
    
    def save(self) -> None:
        """Persist stores to disk."""
        self.graph_store.save()
    
    def reset_vector_store(self) -> None:
        """Reset vector store for ablation studies."""
        if self.config.vector_db_path.exists():
            shutil.rmtree(self.config.vector_db_path)
        
        self.vector_store = VectorStoreAdapter(
            self.config.vector_db_path,
            self.config.embedding_dim,
            self.config.normalize_embeddings,
            self.config.distance_metric,
        )
        self.logger.info("Vector store reset")
    
    def reset_graph_store(self) -> None:
        """Reset graph store for ablation studies."""
        self.graph_store.clear()
        self.logger.info("Graph store reset")
    
    def reset_all(self) -> None:
        """Reset both stores."""
        self.reset_vector_store()
        self.reset_graph_store()


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics(config: StorageConfig, embeddings: Embeddings) -> Dict[str, Any]:
    """Run diagnostic checks on storage configuration."""
    results = {
        "embedding_dim": None,
        "graph_backend": config.graph_backend,
        "kuzu_available": KUZU_AVAILABLE,
        "networkx_available": NETWORKX_AVAILABLE,
        "issues": [],
    }
    
    try:
        test_emb = embeddings.embed_query("diagnostic test")
        results["embedding_dim"] = len(test_emb)
    except Exception as e:
        results["issues"].append(f"Embedding failed: {e}")
    
    if config.graph_backend == "kuzu" and not KUZU_AVAILABLE:
        results["issues"].append("KuzuDB requested but not installed")
    
    return results

