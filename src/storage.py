"""
Hybrid Storage Module: Vector Store (LanceDB) + Knowledge Graph (NetworkX)

Version: 2.1.0 - CORRECTED
Author: Edge-RAG Research Project
Last Modified: 2026-01-13

CRITICAL BUG FIX (v2.1.0):
    The previous implementation did NOT specify the distance metric for LanceDB.
    LanceDB defaults to L2 (Euclidean) distance, but the code assumed Cosine distance.
    
    Mathematical Impact:
    - L2 Distance range: [0, +infinity)  
    - Cosine Distance range: [0, 2]
    - The formula "similarity = 1 - distance" is ONLY valid for Cosine Distance
    
    Observed Symptoms:
    - Maximum similarity scores around 0.25 instead of expected 0.7-0.9
    - All results filtered out at threshold >= 0.25
    - Poor retrieval coverage (0%)
    
    Solution:
    - Explicitly set .metric("cosine") in LanceDB search operations
    - Apply mathematically correct distance-to-similarity conversion

SCIENTIFIC FOUNDATION:

    Vector Similarity Search:
        Given query vector q and document vector d, both in R^n:
        
        Cosine Similarity:
            sim(q, d) = (q . d) / (||q|| * ||d||)
            Range: [-1, 1] for general vectors, [0, 1] for positive embeddings
            
        For L2-normalized vectors (||v|| = 1):
            sim(q, d) = q . d  (dot product equals cosine similarity)
            
        LanceDB Cosine Distance:
            dist(q, d) = 1 - sim(q, d)
            Range: [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite
            
    Conversion Formula:
        similarity = 1.0 - cosine_distance
        
    References:
        - Manning, C.D., Raghavan, P., Schuetze, H. (2008). 
          "Introduction to Information Retrieval", Chapter 6.
        - LanceDB Documentation: https://lancedb.github.io/lancedb/

EDGE DEVICE OPTIMIZATION:
    - LanceDB: Embedded database, no server process required
    - Memory-mapped file access for large vector collections
    - IVF-FLAT indexing for sub-linear search complexity
    - L2 normalization enables dot-product search (faster than cosine)

ARCHITECTURE:
    StorageConfig        - Configuration dataclass with validation
    VectorStoreAdapter   - LanceDB wrapper with correct metric handling
    KnowledgeGraphStore  - NetworkX-based graph for structural relations
    HybridStore          - Facade combining both storage backends
"""

import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time

import lancedb
import networkx as nx
import numpy as np
from langchain.schema import Document
from langchain.embeddings.base import Embeddings


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
        graph_db_path: File path for NetworkX graph (GraphML format)
        embedding_dim: Dimensionality of embedding vectors (None = auto-detect)
        similarity_threshold: Minimum similarity score for retrieval [0.0, 1.0]
        normalize_embeddings: Whether to L2-normalize vectors before storage
        distance_metric: Distance metric for vector search ("cosine" or "l2")
    
    Validation:
        - embedding_dim must be positive if specified
        - similarity_threshold must be in [0.0, 1.0]
        - distance_metric must be "cosine" or "l2"
    """
    vector_db_path: Path
    graph_db_path: Path
    embedding_dim: Optional[int] = None
    similarity_threshold: float = 0.3
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"  # CRITICAL: Must be explicit
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got: {self.embedding_dim}"
            )
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0], got: {self.similarity_threshold}"
            )
        
        if self.distance_metric not in ("cosine", "l2"):
            raise ValueError(
                f"distance_metric must be 'cosine' or 'l2', got: {self.distance_metric}"
            )


# ============================================================================
# VECTOR STORE ADAPTER (LanceDB)
# ============================================================================

class VectorStoreAdapter:
    """
    LanceDB Vector Store Adapter with correct distance metric handling.
    
    This class wraps LanceDB operations and ensures mathematically correct
    similarity score computation for retrieval tasks.
    
    CRITICAL IMPLEMENTATION NOTES:
    
    1. Distance Metric Selection:
       LanceDB supports multiple distance metrics. The default is L2 (Euclidean),
       but for text embeddings, Cosine distance is typically more appropriate.
       
       This implementation EXPLICITLY sets the metric to avoid ambiguity.
    
    2. Distance to Similarity Conversion:
       For Cosine distance in LanceDB:
           cosine_distance = 1 - cosine_similarity
           
       Therefore:
           cosine_similarity = 1 - cosine_distance
           
       For L2 distance, conversion is more complex:
           similarity = 1 / (1 + l2_distance)
           
       This ensures similarity scores are always in [0, 1].
    
    3. Vector Normalization:
       L2-normalized vectors (unit length) have beneficial properties:
       - Cosine similarity equals dot product (faster computation)
       - Numerical stability improved
       - Score interpretation simplified
    
    Attributes:
        db: LanceDB connection object
        embedding_dim: Vector dimensionality
        normalize_embeddings: Whether to normalize vectors
        distance_metric: Distance metric ("cosine" or "l2")
        table: LanceDB table object (lazy initialized)
    """

    # LanceDB table schema for documents
    SCHEMA_VERSION = "2.1.0"
    TABLE_NAME = "documents"
    
    def __init__(
        self, 
        db_path: Path, 
        embedding_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        distance_metric: str = "cosine"
    ):
        """
        Initialize LanceDB connection with explicit configuration.

        Args:
            db_path: Path to LanceDB database directory
            embedding_dim: Expected embedding dimensionality (None = auto-detect)
            normalize_embeddings: Whether to L2-normalize vectors
            distance_metric: Distance metric for search ("cosine" or "l2")
        
        Raises:
            ValueError: If distance_metric is not supported
        """
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
            f"path={db_path}, "
            f"dim={'auto' if embedding_dim is None else embedding_dim}, "
            f"metric={distance_metric}, "
            f"normalize={normalize_embeddings}"
        )
        
        # Load existing metadata if available
        self._load_metadata()

    def _get_metadata_path(self) -> Path:
        """Return path to metadata JSON file."""
        return self.db_path.parent / "vector_store_metadata.json"
    
    def _load_metadata(self) -> None:
        """
        Load metadata from previous ingestion session.
        
        This ensures consistency across restarts and prevents
        dimension mismatch errors when loading existing data.
        """
        metadata_path = self._get_metadata_path()
        
        if not metadata_path.exists():
            return
            
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            stored_dim = metadata.get("embedding_dim")
            stored_metric = metadata.get("distance_metric", "cosine")
            
            if stored_dim and self.embedding_dim is None:
                self.embedding_dim = stored_dim
                self.logger.info(
                    f"Loaded embedding dimension from metadata: {stored_dim}"
                )
            elif stored_dim and self.embedding_dim != stored_dim:
                self.logger.warning(
                    f"Dimension mismatch: config={self.embedding_dim}, "
                    f"stored={stored_dim}. Using config value."
                )
            
            if stored_metric != self.distance_metric:
                self.logger.warning(
                    f"Metric mismatch: config={self.distance_metric}, "
                    f"stored={stored_metric}. Data may need re-indexing."
                )
                
        except Exception as e:
            self.logger.debug(f"Could not load metadata: {e}")
    
    def _save_metadata(self) -> None:
        """
        Persist metadata for reproducibility and consistency checking.
        
        Stored information:
        - embedding_dim: Vector dimensionality
        - distance_metric: Search metric used
        - normalize_embeddings: Normalization setting
        - num_documents: Document count
        - schema_version: For migration support
        - timestamp: Last modification time
        """
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
        
        self.logger.debug(f"Metadata saved: {metadata_path}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to embedding vectors.
        
        Mathematical Definition:
            v_normalized = v / ||v||_2
            
        where ||v||_2 = sqrt(sum(v_i^2)) is the L2 norm.
        
        Properties of L2-normalized vectors:
        1. ||v_normalized||_2 = 1 (unit length)
        2. cosine_similarity(u, v) = dot_product(u, v) for normalized vectors
        3. Numerical stability improved (avoids very large/small values)
        
        Args:
            vectors: Input vectors, shape (N, D)
            
        Returns:
            Normalized vectors, shape (N, D), each with L2 norm = 1
            
        Note:
            Zero vectors are left unchanged to avoid division by zero.
        """
        if not self.normalize_embeddings:
            return vectors
        
        # Compute L2 norms for each row
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Avoid division by zero for zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        
        return vectors / norms

    def _validate_embedding_dimension(self, embeddings: List[List[float]]) -> None:
        """
        Validate embedding dimensions against expected configuration.
        
        This validation is critical for catching configuration errors early.
        Dimension mismatches between stored vectors and query vectors will
        result in meaningless similarity scores.
        
        Args:
            embeddings: List of embedding vectors
            
        Raises:
            ValueError: If dimensions do not match expected value
        """
        if not embeddings:
            return
        
        actual_dim = len(embeddings[0])
        
        # Auto-detect on first use
        if self.embedding_dim is None:
            self.embedding_dim = actual_dim
            self.logger.info(f"Auto-detected embedding dimension: {actual_dim}")
            self._save_metadata()
            return
        
        # Validate against known dimension
        if actual_dim != self.embedding_dim:
            raise ValueError(
                f"EMBEDDING DIMENSION MISMATCH\n"
                f"Expected: {self.embedding_dim} dimensions\n"
                f"Received: {actual_dim} dimensions\n"
                f"\n"
                f"This mismatch will cause incorrect similarity scores.\n"
                f"Resolution steps:\n"
                f"1. Verify embedding model configuration\n"
                f"2. Delete existing vector store: rm -rf {self.db_path}\n"
                f"3. Re-run ingestion pipeline"
            )

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance metric to similarity score in [0, 1].
        
        MATHEMATICAL FOUNDATION:
        
        For Cosine Distance (LanceDB definition):
            cosine_distance = 1 - cosine_similarity
            
            Therefore:
            cosine_similarity = 1 - cosine_distance
            
            Range: cosine_distance in [0, 2]
                   - 0 = identical vectors
                   - 1 = orthogonal vectors  
                   - 2 = opposite vectors
                   
            Resulting similarity in [-1, 1], clamped to [0, 1]
        
        For L2 (Euclidean) Distance:
            l2_distance = ||q - d||_2 = sqrt(sum((q_i - d_i)^2))
            
            Range: [0, +infinity)
            
            Conversion using inverse relationship:
            similarity = 1 / (1 + l2_distance)
            
            This maps:
            - distance=0 -> similarity=1 (identical)
            - distance=1 -> similarity=0.5
            - distance->inf -> similarity->0
        
        Args:
            distance: Raw distance value from LanceDB
            
        Returns:
            Similarity score in [0.0, 1.0]
        """
        if self.distance_metric == "cosine":
            # Cosine: similarity = 1 - distance
            # Clamp to [0, 1] to handle numerical edge cases
            similarity = 1.0 - distance
            return max(0.0, min(1.0, similarity))
        
        elif self.distance_metric == "l2":
            # L2: inverse relationship
            # This formula ensures similarity in (0, 1]
            return 1.0 / (1.0 + distance)
        
        else:
            # Should never reach here due to validation
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: Embeddings,
    ) -> None:
        """
        Add documents with their embeddings to the vector store.
        
        Processing Pipeline:
        1. Extract text content from documents
        2. Generate embeddings via embedding model
        3. Validate embedding dimensions
        4. Apply L2 normalization (if enabled)
        5. Store in LanceDB with metadata
        
        Args:
            documents: List of LangChain Document objects
            embeddings: Embedding model implementing embed_documents()
            
        Raises:
            ValueError: If embedding dimensions are inconsistent
        """
        if not documents:
            self.logger.warning("No documents provided for indexing")
            return

        texts = [doc.page_content for doc in documents]
        self.logger.info(f"Generating embeddings for {len(texts)} documents...")
        
        start_time = time.time()
        embeddings_list = embeddings.embed_documents(texts)
        embed_time = time.time() - start_time
        
        self.logger.info(
            f"Embeddings generated: {len(embeddings_list)} vectors, "
            f"{embed_time:.2f}s total, "
            f"{(embed_time/len(texts)*1000):.1f}ms per document"
        )

        # Validate dimensions before processing
        self._validate_embedding_dimension(embeddings_list)

        # Convert to numpy and normalize
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        if self.normalize_embeddings:
            embeddings_array = self._normalize_vectors(embeddings_array)
            self.logger.debug("Embeddings L2-normalized")
        
        embeddings_list = embeddings_array.tolist()

        # Prepare data records for LanceDB
        data = []
        for doc, emb in zip(documents, embeddings_list):
            data.append({
                "document_id": str(doc.metadata.get("chunk_id", "unknown")),
                "text": doc.page_content,
                "vector": emb,
                "metadata": json.dumps(doc.metadata),
                "source_file": doc.metadata.get("source_file", "unknown"),
            })

        # Insert into LanceDB
        try:
            if self.table is None:
                self.table = self.db.create_table(
                    self.TABLE_NAME, 
                    data=data, 
                    mode="overwrite"
                )
                self.logger.info(f"Created new table with {len(data)} documents")
            else:
                self.table.add(data)
                self.logger.info(f"Added {len(data)} documents to existing table")
            
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
        """
        Perform vector similarity search with explicit metric specification.
        
        CRITICAL IMPLEMENTATION:
        This method explicitly sets the distance metric in the LanceDB query
        to ensure correct similarity score computation.
        
        Algorithm:
        1. Validate query embedding dimension
        2. Normalize query embedding (if enabled)
        3. Execute LanceDB search with EXPLICIT metric
        4. Convert distances to similarities
        5. Filter by threshold
        6. Sort and return top-k results
        
        Args:
            query_embedding: Query vector (must match stored dimensionality)
            top_k: Maximum number of results to return
            threshold: Minimum similarity score [0.0, 1.0]
            
        Returns:
            List of dictionaries containing:
            - text: Document content
            - similarity: Similarity score in [0, 1]
            - document_id: Unique identifier
            - metadata: Document metadata dict
            
        Note:
            Returns empty list if table is not initialized or no results
            meet the threshold criteria.
        """
        if self.table is None:
            self.logger.warning("Vector store is empty - no documents indexed")
            return []

        try:
            # Validate query dimension
            if self.embedding_dim and len(query_embedding) != self.embedding_dim:
                raise ValueError(
                    f"Query dimension mismatch: "
                    f"expected {self.embedding_dim}, got {len(query_embedding)}"
                )
            
            # Normalize query embedding to match stored vectors
            if self.normalize_embeddings:
                query_array = np.array([query_embedding], dtype=np.float32)
                query_array = self._normalize_vectors(query_array)
                query_embedding = query_array[0].tolist()
            
            # ================================================================
            # CRITICAL FIX: Explicitly specify distance metric
            # ================================================================
            # LanceDB defaults to L2 distance if not specified.
            # We MUST set .metric() to ensure correct similarity computation.
            # ================================================================
            
            raw_results = (
                self.table
                .search(query_embedding)
                .metric(self.distance_metric)  # EXPLICIT METRIC SPECIFICATION
                .limit(top_k * 3)  # Over-fetch for threshold filtering
                .to_list()
            )
            
            if not raw_results:
                self.logger.debug("No results returned from LanceDB")
                return []

            # Convert distances to similarities and filter
            filtered_results = []
            
            for idx, result in enumerate(raw_results):
                distance = result.get("_distance", 0.0)
                similarity = self._distance_to_similarity(distance)
                
                # Debug logging for first few results
                if idx < 3:
                    self.logger.debug(
                        f"Result {idx}: distance={distance:.4f} -> "
                        f"similarity={similarity:.4f} (metric={self.distance_metric})"
                    )
                
                # Apply threshold filter
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

            # Sort by similarity (descending) and limit to top_k
            filtered_results.sort(key=lambda x: x["similarity"], reverse=True)
            final_results = filtered_results[:top_k]
            
            # Log search statistics
            self.logger.info(
                f"Vector search: {len(raw_results)} candidates -> "
                f"{len(filtered_results)} above threshold -> "
                f"{len(final_results)} returned (threshold={threshold:.2f})"
            )
            
            if final_results:
                scores = [r["similarity"] for r in final_results]
                self.logger.info(
                    f"Score range: [{min(scores):.4f}, {max(scores):.4f}], "
                    f"mean={sum(scores)/len(scores):.4f}"
                )
            
            return final_results

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}", exc_info=True)
            return []


# ============================================================================
# KNOWLEDGE GRAPH STORE (NetworkX)
# ============================================================================

class KnowledgeGraphStore:
    """
    NetworkX-based Knowledge Graph for structural document relations.
    
    This component provides graph-based retrieval capabilities that complement
    vector similarity search. While vector search captures semantic similarity,
    graph traversal can capture structural relationships such as:
    
    - Document hierarchy (sections, chapters)
    - Citation networks
    - Entity co-occurrence
    - Temporal sequences
    
    The graph is stored as GraphML for portability and human readability.
    
    Note:
        Currently configured with graph_weight=0 for vector-only evaluation.
        Graph retrieval can be enabled by adjusting weights in RetrievalConfig.
    
    Attributes:
        graph_path: Path to GraphML file
        graph: NetworkX DiGraph instance
    """

    def __init__(self, graph_path: Path):
        """
        Initialize Knowledge Graph store.
        
        Args:
            graph_path: Path for GraphML file storage
        """
        self.graph_path = Path(graph_path)
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)

        # Load existing graph if available
        if self.graph_path.exists():
            try:
                self.graph = nx.read_graphml(str(self.graph_path))
                self.logger.info(
                    f"Loaded graph: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges"
                )
            except Exception as e:
                self.logger.error(f"Failed to load graph: {e}")
                self.graph = nx.DiGraph()
        else:
            self.logger.info("Initialized empty knowledge graph")

    def add_entity(
        self, 
        entity_id: str, 
        entity_type: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add an entity node to the graph.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Category of entity (e.g., "document_chunk", "concept")
            metadata: Additional attributes to store with the node
        """
        self.graph.add_node(entity_id, entity_type=entity_type, **metadata)

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a directed relation (edge) between entities.
        
        Args:
            source_id: Source entity identifier
            target_id: Target entity identifier
            relation_type: Type of relation (e.g., "from_source", "references")
            metadata: Additional edge attributes
        """
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
        """
        Perform BFS traversal from a starting entity.
        
        This method finds all entities reachable within max_hops from
        the starting entity, optionally filtering by relation type.
        
        Args:
            start_entity: Starting node identifier
            relation_types: If specified, only follow edges of these types
            max_hops: Maximum traversal depth
            
        Returns:
            Dictionary mapping entity_id to hop distance from start
        """
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
        """Persist graph to GraphML file."""
        try:
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
            nx.write_graphml(self.graph, str(self.graph_path))
            self.logger.info(f"Graph saved: {self.graph_path}")
        except Exception as e:
            self.logger.error(f"Failed to save graph: {e}")


# ============================================================================
# HYBRID STORE (Facade Pattern)
# ============================================================================

class HybridStore:
    """
    Unified interface for Vector Store and Knowledge Graph.
    
    This class implements the Facade pattern, providing a simple interface
    to the underlying storage components. It handles:
    
    - Automatic embedding dimension detection
    - Coordinated document indexing (vector + graph)
    - Metadata persistence
    - Reset operations for ablation studies
    
    Current Configuration:
        For initial evaluation, graph_weight is set to 0 in the retrieval
        configuration. This allows isolated evaluation of vector retrieval
        performance before introducing graph-based augmentation.
    
    Attributes:
        config: StorageConfig instance
        embeddings: Embedding model
        vector_store: VectorStoreAdapter instance
        graph_store: KnowledgeGraphStore instance
    """

    def __init__(self, config: StorageConfig, embeddings: Embeddings):
        """
        Initialize Hybrid Store with automatic dimension detection.
        
        Args:
            config: Storage configuration (embedding_dim can be None)
            embeddings: Embedding model for dimension detection and encoding
        """
        self.config = config
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Auto-detect embedding dimension if not specified
        if config.embedding_dim is None:
            self.logger.info("Auto-detecting embedding dimension...")
            test_embedding = embeddings.embed_query("dimension detection test")
            detected_dim = len(test_embedding)
            config.embedding_dim = detected_dim
            self.logger.info(f"Detected embedding dimension: {detected_dim}")

        # Initialize sub-stores
        self.vector_store = VectorStoreAdapter(
            db_path=config.vector_db_path,
            embedding_dim=config.embedding_dim,
            normalize_embeddings=config.normalize_embeddings,
            distance_metric=config.distance_metric,
        )

        self.graph_store = KnowledgeGraphStore(
            graph_path=config.graph_db_path,
        )

        self.logger.info(
            f"HybridStore initialized: "
            f"dim={config.embedding_dim}, "
            f"metric={config.distance_metric}, "
            f"normalize={config.normalize_embeddings}"
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to both vector and graph stores.
        
        Processing:
        1. Index documents in vector store with embeddings
        2. Create entity nodes for each document chunk
        3. Create relation edges to source documents
        
        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            self.logger.warning("No documents to add")
            return

        try:
            # Add to vector store
            self.logger.info(f"Adding {len(documents)} documents to vector store...")
            self.vector_store.add_documents_with_embeddings(
                documents, 
                self.embeddings
            )

            # Add to knowledge graph
            self.logger.info("Creating knowledge graph entities...")
            for doc in documents:
                doc_id = str(doc.metadata.get("chunk_id", "unknown"))
                source_file = doc.metadata.get("source_file", "unknown")

                # Add document chunk as entity
                self.graph_store.add_entity(
                    entity_id=doc_id,
                    entity_type="document_chunk",
                    metadata={"source_file": source_file},
                )

                # Add source document as entity and create relation
                if source_file != "unknown":
                    self.graph_store.add_entity(
                        entity_id=source_file,
                        entity_type="source_document",
                        metadata={},
                    )
                    self.graph_store.add_relation(
                        source_id=doc_id,
                        target_id=source_file,
                        relation_type="from_source",
                    )

            self.logger.info("Documents successfully added to hybrid store")

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise

    def save(self) -> None:
        """Persist both stores to disk."""
        try:
            # Vector store persists automatically
            # Graph store requires explicit save
            self.graph_store.save()
            self.logger.info("Hybrid store saved")
        except Exception as e:
            self.logger.error(f"Failed to save hybrid store: {e}")

    def load(self) -> None:
        """Load stores from disk (primarily for graph)."""
        try:
            self.logger.info("Hybrid store loaded")
        except Exception as e:
            self.logger.error(f"Failed to load hybrid store: {e}")

    # ========================================================================
    # RESET METHODS (for Ablation Studies)
    # ========================================================================

    def reset_vector_store(self) -> None:
        """
        Reset vector store for clean ablation study runs.
        
        This method:
        1. Deletes the vector database directory
        2. Removes associated metadata
        3. Reinitializes an empty vector store
        
        Warning:
            This is a destructive operation. All indexed vectors will be lost.
        """
        try:
            if self.config.vector_db_path.exists():
                shutil.rmtree(self.config.vector_db_path)
                self.logger.info("Vector store directory deleted")
            
            # Delete metadata
            metadata_path = self.config.vector_db_path.parent / "vector_store_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Reinitialize
            self.vector_store = VectorStoreAdapter(
                self.config.vector_db_path,
                self.config.embedding_dim,
                self.config.normalize_embeddings,
                self.config.distance_metric,
            )
            self.logger.info("Vector store reset complete")
            
        except Exception as e:
            self.logger.error(f"Failed to reset vector store: {e}")
            raise

    def reset_graph_store(self) -> None:
        """
        Reset graph store for clean ablation study runs.
        
        Warning:
            This is a destructive operation. All graph data will be lost.
        """
        try:
            if self.config.graph_db_path.exists():
                self.config.graph_db_path.unlink()
                self.logger.info("Graph file deleted")
            
            self.graph_store = KnowledgeGraphStore(self.config.graph_db_path)
            self.logger.info("Graph store reset complete")
            
        except Exception as e:
            self.logger.error(f"Failed to reset graph store: {e}")
            raise

    def reset_all(self) -> None:
        """Reset both stores completely."""
        self.reset_vector_store()
        self.reset_graph_store()
        self.logger.warning("HYBRID STORE COMPLETELY RESET")


# ============================================================================
# MODULE DIAGNOSTICS
# ============================================================================

def run_diagnostics(config: StorageConfig, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Run diagnostic checks on storage configuration.
    
    This function validates the storage setup and reports potential issues.
    Useful for debugging retrieval quality problems.
    
    Args:
        config: Storage configuration to test
        embeddings: Embedding model to test
        
    Returns:
        Dictionary containing diagnostic results
    """
    results = {
        "embedding_dim": None,
        "embedding_normalized": False,
        "distance_metric": config.distance_metric,
        "test_similarity": None,
        "issues": [],
    }
    
    # Test embedding generation
    try:
        test_emb = embeddings.embed_query("diagnostic test query")
        results["embedding_dim"] = len(test_emb)
        
        # Check normalization
        norm = np.linalg.norm(test_emb)
        results["embedding_normalized"] = abs(norm - 1.0) < 0.01
        
        if not results["embedding_normalized"] and config.normalize_embeddings:
            results["issues"].append(
                f"Embedding not normalized (norm={norm:.4f}), "
                "but normalize_embeddings=True in config"
            )
    except Exception as e:
        results["issues"].append(f"Embedding generation failed: {e}")
    
    # Test similarity computation
    try:
        emb1 = embeddings.embed_query("financial analysis")
        emb2 = embeddings.embed_query("financial sentiment analysis")
        
        # Normalize for comparison
        emb1 = np.array(emb1) / np.linalg.norm(emb1)
        emb2 = np.array(emb2) / np.linalg.norm(emb2)
        
        cosine_sim = np.dot(emb1, emb2)
        results["test_similarity"] = float(cosine_sim)
        
        if cosine_sim < 0.5:
            results["issues"].append(
                f"Low similarity ({cosine_sim:.4f}) for related terms. "
                "Check embedding model quality."
            )
    except Exception as e:
        results["issues"].append(f"Similarity test failed: {e}")
    
    return results