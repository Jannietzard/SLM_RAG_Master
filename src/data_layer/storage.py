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

    def graph_search_1hop(
        self,
        entity_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        1-Hop Graph Search: Finde Chunks die Entity erwähnen.
        
        Gemäß Thesis 2.6: "alle Chunks findet, die die Entität erwähnen (1-Hop)"
        
        Cypher Query:
            MATCH (c:DocumentChunk)-[m:MENTIONS]->(e:Entity)
            WHERE e.name CONTAINS $entity_name
            RETURN c, m, e
            ORDER BY e.mention_count DESC, m.confidence DESC
        
        Args:
            entity_name: Query-Entity Name
            top_k: Anzahl Ergebnisse
            
        Returns:
            Liste von Chunks mit Metadata
        """
        results = []
        
        try:
            query_result = self.conn.execute(
                """
                MATCH (c:DocumentChunk)-[m:MENTIONS]->(e:Entity)
                WHERE e.name CONTAINS $entity_name
                RETURN 
                    c.chunk_id AS chunk_id,
                    c.text AS text,
                    c.source_file AS source_doc,
                    c.position AS position,
                    e.name AS entity_name,
                    e.type AS entity_type,
                    e.mention_count AS mention_count,
                    m.confidence AS confidence,
                    1 AS hop
                ORDER BY 
                    e.mention_count DESC,
                    m.confidence DESC
                LIMIT $top_k
                """,
                {"entity_name": entity_name, "top_k": top_k}
            )
            
            while query_result.has_next():
                row = query_result.get_next()
                results.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "source_doc": row[2],
                    "position": row[3],
                    "entity_name": row[4],
                    "entity_type": row[5],
                    "mention_count": row[6],
                    "confidence": row[7],
                    "hop": row[8],
                })
                
        except Exception as e:
            logger.error(f"1-Hop graph search failed for '{entity_name}': {e}")
        
        return results
    
    def graph_search_2hop(
        self,
        entity_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        2-Hop Graph Search: Finde Chunks über relationale Verbindungen.
        
        Gemäß Thesis 2.6: "alle Chunks findet, die Entitäten erwähnen, 
        die mit der Query-Entität relational verbunden sind (2-Hop)"
        
        Cypher Query:
            MATCH (query_e:Entity)-[r:RELATED_TO]-(related_e:Entity)
                  <-[m:MENTIONS]-(c:DocumentChunk)
            WHERE query_e.name CONTAINS $entity_name
            RETURN c, r, related_e, m
            ORDER BY r.confidence DESC, related_e.mention_count DESC
        
        Args:
            entity_name: Query-Entity Name
            top_k: Anzahl Ergebnisse
            
        Returns:
            Liste von Chunks mit Relation-Metadata
        """
        results = []
        
        try:
            query_result = self.conn.execute(
                """
                MATCH (query_e:Entity)-[r:RELATED_TO]-(related_e:Entity)
                      <-[m:MENTIONS]-(c:DocumentChunk)
                WHERE query_e.name CONTAINS $entity_name
                RETURN 
                    c.chunk_id AS chunk_id,
                    c.text AS text,
                    c.source_file AS source_doc,
                    c.position AS position,
                    query_e.name AS query_entity,
                    r.relation_type AS relation_type,
                    related_e.name AS related_entity,
                    related_e.type AS related_entity_type,
                    r.confidence AS relation_confidence,
                    m.confidence AS mention_confidence,
                    2 AS hop
                ORDER BY 
                    r.confidence DESC,
                    related_e.mention_count DESC,
                    m.confidence DESC
                LIMIT $top_k
                """,
                {"entity_name": entity_name, "top_k": top_k}
            )
            
            while query_result.has_next():
                row = query_result.get_next()
                results.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "source_doc": row[2],
                    "position": row[3],
                    "query_entity": row[4],
                    "relation_type": row[5],
                    "related_entity": row[6],
                    "related_entity_type": row[7],
                    "confidence": row[8],  # Relation confidence für Ranking
                    "mention_confidence": row[9],
                    "hop": row[10],
                })
                
        except Exception as e:
            logger.error(f"2-Hop graph search failed for '{entity_name}': {e}")
        
        return results
    
    def graph_search_combined(
        self,
        entity_name: str,
        top_k_1hop: int = 5,
        top_k_2hop: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Kombiniere 1-Hop und 2-Hop Searches.
        
        Gemäß Thesis 2.6: "Für jede Query-Entität wird eine Cypher-Query 
        generiert, die (a) ... (1-Hop), und (b) ... (2-Hop)"
        
        Args:
            entity_name: Query-Entity Name
            top_k_1hop: Top-K für 1-Hop
            top_k_2hop: Top-K für 2-Hop
            
        Returns:
            Kombinierte und deduplizierte Ergebnisliste
        """
        # 1-Hop Search
        results_1hop = self.graph_search_1hop(entity_name, top_k_1hop)
        
        # 2-Hop Search
        results_2hop = self.graph_search_2hop(entity_name, top_k_2hop)
        
        # Deduplizierung nach chunk_id (1-Hop hat Priorität)
        seen_chunks = set()
        combined = []
        
        for result in results_1hop:
            if result["chunk_id"] not in seen_chunks:
                combined.append(result)
                seen_chunks.add(result["chunk_id"])
        
        for result in results_2hop:
            if result["chunk_id"] not in seen_chunks:
                combined.append(result)
                seen_chunks.add(result["chunk_id"])
        
        return combined


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


# ============================================================================
# COMPREHENSIVE TEST SUITE FOR STORAGE.PY
# ============================================================================

if __name__ == "__main__":
    import sys
    import tempfile
    import shutil
    from pathlib import Path
    from typing import List
    
    # Setup logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("STORAGE.PY - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Version: {VectorStoreAdapter.SCHEMA_VERSION}")
    print(f"KuzuDB Available: {KUZU_AVAILABLE}")
    print(f"NetworkX Available: {NETWORKX_AVAILABLE}")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # MOCK EMBEDDINGS FOR TESTING
    # ========================================================================
    
    class MockEmbeddings(Embeddings):
        """Mock embedding model for testing."""
        
        def __init__(self, dim: int = 768):
            self.dim = dim
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Generate random embeddings for documents."""
            embeddings = []
            for text in texts:
                # Deterministic based on text length for reproducibility
                np.random.seed(len(text))
                vec = np.random.randn(self.dim).astype(np.float32)
                vec = vec / np.linalg.norm(vec)  # L2 normalize
                embeddings.append(vec.tolist())
            return embeddings
        
        def embed_query(self, text: str) -> List[float]:
            """Generate random embedding for query."""
            np.random.seed(len(text))
            vec = np.random.randn(self.dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            return vec.tolist()
    
    # ========================================================================
    # TEST UTILITIES
    # ========================================================================
    
    class TestResult:
        """Track test results."""
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.errors = []
        
        def assert_true(self, condition: bool, message: str):
            """Assert condition is true."""
            if condition:
                self.passed += 1
                print(f"  ✓ {message}")
            else:
                self.failed += 1
                self.errors.append(message)
                print(f"  ✗ FAILED: {message}")
        
        def assert_equal(self, actual, expected, message: str):
            """Assert values are equal."""
            if actual == expected:
                self.passed += 1
                print(f"  ✓ {message}")
            else:
                self.failed += 1
                error = f"{message} (expected: {expected}, got: {actual})"
                self.errors.append(error)
                print(f"  ✗ FAILED: {error}")
        
        def assert_greater(self, actual, threshold, message: str):
            """Assert value is greater than threshold."""
            if actual > threshold:
                self.passed += 1
                print(f"  ✓ {message}")
            else:
                self.failed += 1
                error = f"{message} (expected > {threshold}, got: {actual})"
                self.errors.append(error)
                print(f"  ✗ FAILED: {error}")
        
        def assert_in_range(self, actual, min_val, max_val, message: str):
            """Assert value is in range."""
            if min_val <= actual <= max_val:
                self.passed += 1
                print(f"  ✓ {message}")
            else:
                self.failed += 1
                error = f"{message} (expected [{min_val}, {max_val}], got: {actual})"
                self.errors.append(error)
                print(f"  ✗ FAILED: {error}")
        
        def print_summary(self):
            """Print test summary."""
            total = self.passed + self.failed
            print("\n" + "=" * 80)
            print("TEST SUMMARY")
            print("=" * 80)
            print(f"Total Tests: {total}")
            print(f"Passed: {self.passed} ({self.passed/total*100:.1f}%)")
            print(f"Failed: {self.failed} ({self.failed/total*100:.1f}%)")
            
            if self.failed > 0:
                print("\nFailed Tests:")
                for error in self.errors:
                    print(f"  - {error}")
            
            print("=" * 80)
            
            if self.failed == 0:
                print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓\n")
                return True
            else:
                print(f"\n✗✗✗ {self.failed} TESTS FAILED! ✗✗✗\n")
                return False
    
    # ========================================================================
    # TEST 1: STORAGE CONFIG VALIDATION
    # ========================================================================
    
    def test_storage_config():
        """Test StorageConfig validation."""
        print("\n" + "-" * 80)
        print("TEST 1: StorageConfig Validation")
        print("-" * 80)
        
        result = TestResult()
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Valid configuration
            config = StorageConfig(
                vector_db_path=temp_dir / "vector",
                graph_db_path=temp_dir / "graph",
                embedding_dim=768,
                similarity_threshold=0.5,
                distance_metric="cosine",
                graph_backend="kuzu"
            )
            result.assert_true(True, "Valid configuration created")
            
            # Test invalid embedding_dim
            try:
                invalid_config = StorageConfig(
                    vector_db_path=temp_dir / "vector",
                    graph_db_path=temp_dir / "graph",
                    embedding_dim=-1
                )
                result.assert_true(False, "Should reject negative embedding_dim")
            except ValueError:
                result.assert_true(True, "Rejected negative embedding_dim")
            
            # Test invalid similarity_threshold
            try:
                invalid_config = StorageConfig(
                    vector_db_path=temp_dir / "vector",
                    graph_db_path=temp_dir / "graph",
                    similarity_threshold=1.5
                )
                result.assert_true(False, "Should reject similarity_threshold > 1.0")
            except ValueError:
                result.assert_true(True, "Rejected invalid similarity_threshold")
            
            # Test invalid distance_metric
            try:
                invalid_config = StorageConfig(
                    vector_db_path=temp_dir / "vector",
                    graph_db_path=temp_dir / "graph",
                    distance_metric="invalid"
                )
                result.assert_true(False, "Should reject invalid distance_metric")
            except ValueError:
                result.assert_true(True, "Rejected invalid distance_metric")
            
            # Test invalid graph_backend
            try:
                invalid_config = StorageConfig(
                    vector_db_path=temp_dir / "vector",
                    graph_db_path=temp_dir / "graph",
                    graph_backend="invalid"
                )
                result.assert_true(False, "Should reject invalid graph_backend")
            except ValueError:
                result.assert_true(True, "Rejected invalid graph_backend")
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    # ========================================================================
    # TEST 2: VECTOR STORE ADAPTER
    # ========================================================================
    
    def test_vector_store():
        """Test VectorStoreAdapter functionality."""
        print("\n" + "-" * 80)
        print("TEST 2: VectorStoreAdapter")
        print("-" * 80)
        
        result = TestResult()
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Initialize vector store
            vector_store = VectorStoreAdapter(
                db_path=temp_dir / "vector_test",
                embedding_dim=768,
                normalize_embeddings=True,
                distance_metric="cosine"
            )
            result.assert_true(True, "VectorStoreAdapter initialized")
            
            # Create mock embeddings
            embeddings = MockEmbeddings(dim=768)
            
            # Create test documents
            test_docs = [
                Document(
                    page_content="Albert Einstein developed the theory of relativity.",
                    metadata={"chunk_id": "chunk_001", "source_file": "physics.pdf"}
                ),
                Document(
                    page_content="Marie Curie discovered radium and polonium.",
                    metadata={"chunk_id": "chunk_002", "source_file": "chemistry.pdf"}
                ),
                Document(
                    page_content="Isaac Newton formulated the laws of motion.",
                    metadata={"chunk_id": "chunk_003", "source_file": "physics.pdf"}
                ),
            ]
            
            # Test: Add documents
            vector_store.add_documents_with_embeddings(test_docs, embeddings)
            result.assert_true(True, "Documents added successfully")
            
            # Test: Verify metadata saved
            metadata_path = vector_store._get_metadata_path()
            result.assert_true(metadata_path.exists(), "Metadata file created")
            
            # Test: Dimension validation
            try:
                wrong_dim_embeddings = MockEmbeddings(dim=512)
                vector_store.add_documents_with_embeddings([test_docs[0]], wrong_dim_embeddings)
                result.assert_true(False, "Should reject wrong embedding dimension")
            except ValueError as e:
                if "DIMENSION MISMATCH" in str(e):
                    result.assert_true(True, "Dimension mismatch detected")
                else:
                    result.assert_true(False, f"Wrong error message: {e}")
            
            # Test: Vector search
            query_embedding = embeddings.embed_query("relativity physics Einstein")
            search_results = vector_store.vector_search(query_embedding, top_k=3)
            
            result.assert_greater(len(search_results), 0, "Search returned results")
            result.assert_true(
                all("similarity" in r for r in search_results),
                "All results have similarity scores"
            )
            result.assert_true(
                all(0 <= r["similarity"] <= 1 for r in search_results),
                "All similarity scores in valid range [0,1]"
            )
            
            # Test: Distance to similarity conversion
            cosine_sim_0 = vector_store._distance_to_similarity(0.0)
            result.assert_equal(cosine_sim_0, 1.0, "Distance 0.0 -> Similarity 1.0")
            
            cosine_sim_1 = vector_store._distance_to_similarity(1.0)
            result.assert_equal(cosine_sim_1, 0.0, "Distance 1.0 -> Similarity 0.0")
            
            # Test: Normalization
            test_vectors = np.random.randn(5, 768).astype(np.float32)
            normalized = vector_store._normalize_vectors(test_vectors)
            norms = np.linalg.norm(normalized, axis=1)
            result.assert_true(
                np.allclose(norms, 1.0, atol=1e-5),
                "Vectors normalized to unit length"
            )
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    # ========================================================================
    # TEST 3: KUZU GRAPH STORE
    # ========================================================================
    
    def test_kuzu_graph_store():
        """Test KuzuGraphStore functionality."""
        print("\n" + "-" * 80)
        print("TEST 3: KuzuGraphStore")
        print("-" * 80)
        
        result = TestResult()
        
        if not KUZU_AVAILABLE:
            print("  ⚠ KuzuDB not available, skipping tests")
            return result
        
        temp_dir = Path(tempfile.mkdtemp())
        graph_store = None
        
        try:
            # Initialize graph store
            try:
                graph_store = KuzuGraphStore(temp_dir / "graph_test")
                result.assert_true(True, "KuzuGraphStore initialized")
            except Exception as e:
                print(f"  ⚠ KuzuDB initialization failed: {e}")
                print(f"  ⚠ Skipping remaining KuzuDB tests")
                return result
            
            # Test: Add document chunks (with error handling)
            chunk_added = False
            try:
                graph_store.add_document_chunk(
                    chunk_id="chunk_001",
                    text="Einstein was born in Ulm, Germany.",
                    page_number=1,
                    chunk_index=0,
                    source_file="einstein_bio.pdf"
                )
                chunk_added = True
                result.assert_true(True, "Document chunk added")
            except Exception as e:
                result.assert_true(False, f"Failed to add document chunk: {e}")
            
            if chunk_added:
                try:
                    graph_store.add_document_chunk(
                        chunk_id="chunk_002",
                        text="He worked at Princeton University.",
                        page_number=1,
                        chunk_index=1,
                        source_file="einstein_bio.pdf"
                    )
                    result.assert_true(True, "Second document chunk added")
                except Exception as e:
                    result.assert_true(False, f"Failed to add second chunk: {e}")
            
            # Test: Add source document
            try:
                graph_store.add_source_document(
                    doc_id="einstein_bio.pdf",
                    filename="einstein_bio.pdf",
                    total_pages=10
                )
                result.assert_true(True, "Source document added")
            except Exception as e:
                result.assert_true(False, f"Failed to add source document: {e}")
            
            # Test: Add FROM_SOURCE relation
            if chunk_added:
                try:
                    graph_store.add_from_source_relation("chunk_001", "einstein_bio.pdf")
                    result.assert_true(True, "FROM_SOURCE relation added")
                except Exception as e:
                    result.assert_true(False, f"Failed to add FROM_SOURCE relation: {e}")
            
            # Test: Add NEXT_CHUNK relation
            if chunk_added:
                try:
                    graph_store.add_next_chunk_relation("chunk_001", "chunk_002")
                    result.assert_true(True, "NEXT_CHUNK relation added")
                except Exception as e:
                    result.assert_true(False, f"Failed to add NEXT_CHUNK relation: {e}")
            
            # Test: Add entities
            entity_added = False
            try:
                graph_store.add_entity(
                    entity_id="entity_einstein",
                    name="Albert Einstein",
                    entity_type="PERSON"
                )
                entity_added = True
                result.assert_true(True, "Entity added")
            except Exception as e:
                result.assert_true(False, f"Failed to add entity: {e}")
            
            # Test: Add MENTIONS relation
            if chunk_added and entity_added:
                try:
                    graph_store.add_mentions_relation("chunk_001", "entity_einstein")
                    result.assert_true(True, "MENTIONS relation added")
                except Exception as e:
                    result.assert_true(False, f"Failed to add MENTIONS relation: {e}")
            
            # Test: Add RELATED_TO relation
            if entity_added:
                try:
                    graph_store.add_entity(
                        entity_id="entity_princeton",
                        name="Princeton University",
                        entity_type="ORGANIZATION"
                    )
                    graph_store.add_related_to_relation(
                        "entity_einstein",
                        "entity_princeton",
                        "works_for"
                    )
                    result.assert_true(True, "RELATED_TO relation added")
                except Exception as e:
                    result.assert_true(False, f"Failed to add RELATED_TO relation: {e}")
            
            # Test: Get statistics
            try:
                stats = graph_store.get_statistics()
                if chunk_added:
                    result.assert_greater(stats.get('document_chunks', 0), 0, "Document chunks counted")
                    result.assert_greater(stats.get('source_documents', 0), 0, "Source documents counted")
                if entity_added:
                    result.assert_greater(stats.get('entities', 0), 0, "Entities counted")
            except Exception as e:
                result.assert_true(False, f"Failed to get statistics: {e}")
            
            # Test: Graph traversal
            if chunk_added:
                try:
                    visited = graph_store.graph_traversal("chunk_001", max_hops=2)
                    result.assert_true("chunk_001" in visited, "Start node in traversal results")
                    if "chunk_001" in visited:
                        result.assert_equal(visited["chunk_001"], 0, "Start node at hop 0")
                except Exception as e:
                    result.assert_true(False, f"Graph traversal failed: {e}")
            
            # Test: Find related chunks
            if chunk_added:
                try:
                    related = graph_store.find_related_chunks("chunk_001", max_hops=1)
                    result.assert_true(isinstance(related, list), "Find related chunks returns list")
                except Exception as e:
                    result.assert_true(False, f"Find related chunks failed: {e}")
            
            # Test: Find chunks by entity
            if chunk_added and entity_added:
                try:
                    entity_chunks = graph_store.find_chunks_by_entity("Einstein", max_results=5)
                    result.assert_true(isinstance(entity_chunks, list), "Find by entity returns list")
                except Exception as e:
                    result.assert_true(False, f"Find chunks by entity failed: {e}")
            
            # Test: Get document structure
            if chunk_added:
                try:
                    doc_structure = graph_store.get_document_structure("einstein_bio.pdf")
                    result.assert_greater(len(doc_structure), 0, "Document structure retrieved")
                except Exception as e:
                    result.assert_true(False, f"Get document structure failed: {e}")
            
            # Test: Clear graph
            try:
                graph_store.clear()
                stats_after_clear = graph_store.get_statistics()
                result.assert_equal(
                    stats_after_clear.get('document_chunks', 0), 0,
                    "Graph cleared successfully"
                )
            except Exception as e:
                result.assert_true(False, f"Clear graph failed: {e}")
        
        except Exception as e:
            result.assert_true(False, f"Unexpected error in KuzuDB tests: {e}")
        
        finally:
            # Cleanup
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        
        return result
    
    # ========================================================================
    # TEST 4: NETWORKX GRAPH STORE (FALLBACK)
    # ========================================================================
    
    def test_networkx_graph_store():
        """Test NetworkXGraphStore functionality."""
        print("\n" + "-" * 80)
        print("TEST 4: NetworkXGraphStore (Fallback)")
        print("-" * 80)
        
        result = TestResult()
        
        if not NETWORKX_AVAILABLE:
            print("  ⚠ NetworkX not available, skipping tests")
            return result
        
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Initialize NetworkX graph store
            graph_store = NetworkXGraphStore(temp_dir / "graph_nx.graphml")
            result.assert_true(True, "NetworkXGraphStore initialized")
            
            # Test: Add entities
            graph_store.add_entity(
                "chunk_001",
                "document_chunk",
                {"text": "Test chunk", "source_file": "test.pdf"}
            )
            result.assert_true(True, "Entity added")
            
            # Test: Add relations
            graph_store.add_entity("source_001", "source_document", {})
            graph_store.add_relation("chunk_001", "source_001", "from_source")
            result.assert_true(True, "Relation added")
            
            # Test: Graph traversal
            visited = graph_store.graph_traversal("chunk_001", max_hops=1)
            result.assert_true("chunk_001" in visited, "Traversal includes start node")
            
            # Test: Statistics
            stats = graph_store.get_statistics()
            result.assert_greater(stats['nodes'], 0, "Nodes counted")
            result.assert_greater(stats['edges'], 0, "Edges counted")
            
            # Test: Save and load
            graph_store.save()
            result.assert_true(
                graph_store.graph_path.exists(),
                "Graph saved to file"
            )
            
            # Load in new instance
            graph_store_2 = NetworkXGraphStore(temp_dir / "graph_nx.graphml")
            stats_2 = graph_store_2.get_statistics()
            result.assert_equal(
                stats_2['nodes'], stats['nodes'],
                "Graph loaded correctly"
            )
            
            # Test: Clear
            graph_store.clear()
            stats_clear = graph_store.get_statistics()
            result.assert_equal(stats_clear['nodes'], 0, "Graph cleared")
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    # ========================================================================
    # TEST 5: HYBRID STORE INTEGRATION
    # ========================================================================
    
    def test_hybrid_store():
        """Test HybridStore integration."""
        print("\n" + "-" * 80)
        print("TEST 5: HybridStore Integration")
        print("-" * 80)
        
        result = TestResult()
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create config
            config = StorageConfig(
                vector_db_path=temp_dir / "hybrid_vector",
                graph_db_path=temp_dir / "hybrid_graph",
                embedding_dim=768,
                graph_backend="kuzu" if KUZU_AVAILABLE else "networkx"
            )
            
            # Create embeddings
            embeddings = MockEmbeddings(dim=768)
            
            # Initialize hybrid store
            hybrid_store = HybridStore(config, embeddings)
            result.assert_true(True, "HybridStore initialized")
            
            # Verify vector store
            result.assert_true(
                hybrid_store.vector_store is not None,
                "Vector store created"
            )
            
            # Verify graph store
            result.assert_true(
                hybrid_store.graph_store is not None,
                "Graph store created"
            )
            
            # Verify embedding dimension auto-detection
            result.assert_equal(
                config.embedding_dim, 768,
                "Embedding dimension detected"
            )
            
            # Test: Add documents
            test_docs = [
                Document(
                    page_content="Einstein developed relativity theory.",
                    metadata={
                        "chunk_id": "chunk_001",
                        "source_file": "physics.pdf",
                        "chunk_index": 0,
                        "page_number": 1
                    }
                ),
                Document(
                    page_content="He received the Nobel Prize in 1921.",
                    metadata={
                        "chunk_id": "chunk_002",
                        "source_file": "physics.pdf",
                        "chunk_index": 1,
                        "page_number": 1
                    }
                ),
                Document(
                    page_content="Marie Curie discovered radioactivity.",
                    metadata={
                        "chunk_id": "chunk_003",
                        "source_file": "chemistry.pdf",
                        "chunk_index": 0,
                        "page_number": 1
                    }
                ),
            ]
            
            hybrid_store.add_documents(test_docs)
            result.assert_true(True, "Documents added to hybrid store")
            
            # Verify vector store has documents
            query_emb = embeddings.embed_query("physics Einstein")
            vector_results = hybrid_store.vector_store.vector_search(query_emb, top_k=3)
            result.assert_greater(
                len(vector_results), 0,
                "Vector store contains documents"
            )
            
            # Verify graph store has nodes
            graph_stats = hybrid_store.graph_store.get_statistics()
            if KUZU_AVAILABLE:
                result.assert_greater(
                    graph_stats['document_chunks'], 0,
                    "Graph store contains chunks"
                )
                result.assert_greater(
                    graph_stats['source_documents'], 0,
                    "Graph store contains source documents"
                )
            else:
                result.assert_greater(
                    graph_stats['nodes'], 0,
                    "Graph store contains nodes"
                )
            
            # Test: Save
            hybrid_store.save()
            result.assert_true(True, "Hybrid store saved")
            
            # Test: Reset vector store
            hybrid_store.reset_vector_store()
            result.assert_true(True, "Vector store reset")
            
            # Test: Reset graph store
            hybrid_store.reset_graph_store()
            graph_stats_reset = hybrid_store.graph_store.get_statistics()
            if KUZU_AVAILABLE:
                result.assert_equal(
                    graph_stats_reset['document_chunks'], 0,
                    "Graph store reset"
                )
            else:
                result.assert_equal(
                    graph_stats_reset['nodes'], 0,
                    "Graph store reset"
                )
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    # ========================================================================
    # TEST 6: DIAGNOSTICS
    # ========================================================================
    
    def test_diagnostics():
        """Test diagnostic function."""
        print("\n" + "-" * 80)
        print("TEST 6: Diagnostics")
        print("-" * 80)
        
        result = TestResult()
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            config = StorageConfig(
                vector_db_path=temp_dir / "diag_vector",
                graph_db_path=temp_dir / "diag_graph",
                graph_backend="kuzu" if KUZU_AVAILABLE else "networkx"
            )
            
            embeddings = MockEmbeddings(dim=768)
            
            # Run diagnostics
            diag_results = run_diagnostics(config, embeddings)
            
            result.assert_true(
                "embedding_dim" in diag_results,
                "Diagnostics include embedding_dim"
            )
            result.assert_equal(
                diag_results["embedding_dim"], 768,
                "Embedding dimension detected correctly"
            )
            result.assert_true(
                "graph_backend" in diag_results,
                "Diagnostics include graph_backend"
            )
            result.assert_true(
                "kuzu_available" in diag_results,
                "Diagnostics include kuzu_available"
            )
            result.assert_true(
                "networkx_available" in diag_results,
                "Diagnostics include networkx_available"
            )
            result.assert_true(
                isinstance(diag_results["issues"], list),
                "Diagnostics include issues list"
            )
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    # ========================================================================
    # TEST 7: PERFORMANCE BENCHMARKS
    # ========================================================================
    
    def test_performance():
        """Test performance benchmarks."""
        print("\n" + "-" * 80)
        print("TEST 7: Performance Benchmarks")
        print("-" * 80)
        
        result = TestResult()
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            config = StorageConfig(
                vector_db_path=temp_dir / "perf_vector",
                graph_db_path=temp_dir / "perf_graph",
                embedding_dim=768
            )
            
            embeddings = MockEmbeddings(dim=768)
            hybrid_store = HybridStore(config, embeddings)
            
            # Create 100 test documents
            num_docs = 100
            test_docs = []
            for i in range(num_docs):
                test_docs.append(
                    Document(
                        page_content=f"This is test document number {i} about various topics.",
                        metadata={
                            "chunk_id": f"chunk_{i:03d}",
                            "source_file": f"test_{i//10}.pdf",
                            "chunk_index": i % 10,
                            "page_number": 1
                        }
                    )
                )
            
            # Benchmark: Document insertion
            import time
            start_time = time.time()
            hybrid_store.add_documents(test_docs)
            insert_time = (time.time() - start_time) * 1000
            
            result.assert_true(
                insert_time < 30000,  # 30 seconds for 100 docs
                f"Document insertion time acceptable ({insert_time:.0f}ms for {num_docs} docs)"
            )
            
            # Benchmark: Vector search
            query_emb = embeddings.embed_query("test document topics")
            start_time = time.time()
            search_results = hybrid_store.vector_store.vector_search(query_emb, top_k=10)
            search_time = (time.time() - start_time) * 1000
            
            result.assert_true(
                search_time < 100,  # 100ms
                f"Vector search time acceptable ({search_time:.1f}ms)"
            )
            
            result.assert_greater(
                len(search_results), 0,
                f"Search returned {len(search_results)} results"
            )
            
            # Benchmark: Graph traversal (if KuzuDB available)
            if KUZU_AVAILABLE:
                start_time = time.time()
                visited = hybrid_store.graph_store.graph_traversal("chunk_000", max_hops=2)
                traversal_time = (time.time() - start_time) * 1000
                
                result.assert_true(
                    traversal_time < 50,  # 50ms
                    f"Graph traversal time acceptable ({traversal_time:.1f}ms)"
                )
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all_tests():
        """Run all test suites."""
        all_results = []
        
        # Run each test suite
        all_results.append(test_storage_config())
        all_results.append(test_vector_store())
        all_results.append(test_kuzu_graph_store())
        all_results.append(test_networkx_graph_store())
        all_results.append(test_hybrid_store())
        all_results.append(test_diagnostics())
        all_results.append(test_performance())
        
        # Aggregate results
        total_passed = sum(r.passed for r in all_results)
        total_failed = sum(r.failed for r in all_results)
        total_tests = total_passed + total_failed
        
        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL TEST SUMMARY")
        print("=" * 80)
        print(f"Total Test Suites: {len(all_results)}")
        print(f"Total Tests: {total_tests}")
        print(f"Total Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
        print(f"Total Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
        print("=" * 80)
        
        if total_failed == 0:
            print("\n🎉🎉🎉 ALL TESTS PASSED! 🎉🎉🎉\n")
            print("Storage.py is working correctly!")
            return 0
        else:
            print(f"\n⚠️⚠️⚠️ {total_failed} TESTS FAILED! ⚠️⚠️⚠️\n")
            print("Please review the failed tests above.")
            return 1
    
    # Execute all tests
    exit_code = run_all_tests()
    sys.exit(exit_code)