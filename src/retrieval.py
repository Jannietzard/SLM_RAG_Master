"""
Hybrid Retrieval Engine: Vector and Graph-based Query Processing

Version: 2.1.0
Author: Edge-RAG Research Project
Last Modified: 2026-01-13

===============================================================================
SCIENTIFIC FOUNDATION
===============================================================================

This module implements a hybrid retrieval system combining two complementary
approaches for document retrieval in RAG (Retrieval-Augmented Generation):

1. DENSE RETRIEVAL (Vector-based):
   - Uses embedding vectors for semantic similarity matching
   - Bi-Encoder architecture: Query and documents encoded separately
   - Enables sub-linear search via approximate nearest neighbor (ANN)
   - Primary retrieval method for semantic understanding
   
   Reference: Karpukhin et al. (2020). "Dense Passage Retrieval for 
   Open-Domain Question Answering." EMNLP 2020.

2. SPARSE/STRUCTURAL RETRIEVAL (Graph-based):
   - Uses knowledge graph for structural relationships
   - Enables multi-hop reasoning through entity relations
   - Captures document structure (sections, references, hierarchy)
   - Complementary to vector search for complex queries
   
   Reference: Yu et al. (2024). "Graph-RAG: Enhancing Retrieval-Augmented
   Generation with Knowledge Graphs."

3. HYBRID ENSEMBLE:
   - Weighted combination of vector and graph results
   - Configurable weights enable ablation studies
   - Reduces bias inherent in single-method retrieval
   
   Reference: Ma et al. (2021). "A Replication Study of Dense Passage
   Retriever." arXiv:2104.05740

===============================================================================
EDGE DEVICE OPTIMIZATION
===============================================================================

Design decisions for resource-constrained deployment:

1. Lazy Initialization: Components initialized only when needed
2. Configurable top_k: Limits memory usage during retrieval
3. No Cross-Encoder Reranking: Avoids additional model inference
4. Efficient Score Combination: Simple weighted average (O(n))

Memory Complexity: O(top_k * embedding_dim)
Time Complexity: O(log n) for vector search (with ANN index)

===============================================================================
ABLATION STUDY SUPPORT
===============================================================================

This implementation supports systematic ablation studies for thesis validation:

Configuration Examples:
- Vector-only:  vector_weight=1.0, graph_weight=0.0
- Graph-only:   vector_weight=0.0, graph_weight=1.0  
- Hybrid 70/30: vector_weight=0.7, graph_weight=0.3
- Hybrid 50/50: vector_weight=0.5, graph_weight=0.5

Metrics to collect for each configuration:
- Precision@k (k=1,3,5,10)
- Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Latency (ms per query)

===============================================================================
MODULE STRUCTURE
===============================================================================

Classes:
    RetrievalMode      - Enum for retrieval mode selection
    RetrievalConfig    - Configuration dataclass
    RetrievalResult    - Result container dataclass
    VectorRetriever    - Dense vector-based retrieval
    GraphRetriever     - Knowledge graph-based retrieval
    HybridRetriever    - Ensemble combining both methods

Design Patterns:
    - Strategy Pattern: Different retrieval strategies (vector/graph/hybrid)
    - Facade Pattern: HybridRetriever provides unified interface
    - Composition: HybridRetriever composes VectorRetriever and GraphRetriever
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain.embeddings.base import Embeddings

from src.storage import HybridStore


logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS AND CONFIGURATION
# ============================================================================

class RetrievalMode(str, Enum):
    """
    Enumeration of available retrieval modes.
    
    Used for ablation studies to isolate contribution of each retrieval method.
    
    Values:
        VECTOR: Dense retrieval using embedding similarity only
        GRAPH:  Sparse retrieval using knowledge graph traversal only
        HYBRID: Weighted ensemble of vector and graph retrieval
    """
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    """
    Configuration container for hybrid retrieval system.
    
    This dataclass encapsulates all parameters needed for retrieval,
    enabling reproducible experiments and systematic ablation studies.
    
    Attributes:
        mode: Retrieval mode (vector, graph, or hybrid)
        top_k_vector: Number of results from vector retrieval
        top_k_graph: Number of results from graph retrieval
        vector_weight: Weight for vector scores in ensemble [0.0, 1.0]
        graph_weight: Weight for graph scores in ensemble [0.0, 1.0]
        similarity_threshold: Minimum similarity score for filtering [0.0, 1.0]
    
    Weight Normalization:
        Final scores are computed as:
        score = (vector_score * vector_weight + graph_score * graph_weight) 
                / (vector_weight + graph_weight)
        
        This ensures scores remain in [0, 1] regardless of weight values.
    
    Example Configurations:
        Vector-only: RetrievalConfig(mode=VECTOR, vector_weight=1.0, graph_weight=0.0)
        Hybrid:      RetrievalConfig(mode=HYBRID, vector_weight=0.7, graph_weight=0.3)
    """
    mode: RetrievalMode
    top_k_vector: int
    top_k_graph: int
    vector_weight: float
    graph_weight: float
    similarity_threshold: float
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.top_k_vector < 1:
            raise ValueError(f"top_k_vector must be >= 1, got {self.top_k_vector}")
        
        if self.top_k_graph < 1:
            raise ValueError(f"top_k_graph must be >= 1, got {self.top_k_graph}")
        
        if not (0.0 <= self.vector_weight <= 1.0):
            raise ValueError(f"vector_weight must be in [0,1], got {self.vector_weight}")
        
        if not (0.0 <= self.graph_weight <= 1.0):
            raise ValueError(f"graph_weight must be in [0,1], got {self.graph_weight}")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0,1], got {self.similarity_threshold}"
            )
        
        # Warn if both weights are zero
        if self.vector_weight == 0.0 and self.graph_weight == 0.0:
            raise ValueError("At least one weight must be non-zero")


@dataclass
class RetrievalResult:
    """
    Container for individual retrieval results.
    
    This dataclass provides a standardized format for results from any
    retrieval method, enabling uniform processing in downstream components.
    
    Attributes:
        text: Retrieved document content (chunk text)
        relevance_score: Computed relevance score [0.0, 1.0]
        document_id: Unique identifier for the document chunk
        source_file: Original source file name
        retrieval_method: Method that produced this result ("vector"/"graph"/"hybrid")
        metadata: Additional metadata from document indexing
    
    Score Interpretation:
        0.0 - 0.3: Low relevance (likely noise)
        0.3 - 0.5: Moderate relevance (potentially useful)
        0.5 - 0.7: Good relevance (likely relevant)
        0.7 - 1.0: High relevance (strongly relevant)
    
    Note:
        Score ranges are empirical and may vary based on embedding model
        and document characteristics. Calibration recommended for each dataset.
    """
    text: str
    relevance_score: float
    document_id: str
    source_file: str
    retrieval_method: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "text": self.text,
            "relevance_score": self.relevance_score,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "retrieval_method": self.retrieval_method,
            "metadata": self.metadata,
        }


# ============================================================================
# VECTOR RETRIEVER
# ============================================================================

class VectorRetriever:
    """
    Dense vector-based retrieval using embedding similarity.
    
    SCIENTIFIC BACKGROUND:
    
    Dense retrieval encodes queries and documents into continuous vector
    representations (embeddings) and retrieves documents based on vector
    similarity. This approach captures semantic meaning beyond lexical overlap.
    
    Architecture: Bi-Encoder
        - Query encoder: f(q) -> R^d
        - Document encoder: g(d) -> R^d  
        - Similarity: sim(q, d) = cosine(f(q), g(d))
        
    In this implementation, both query and documents use the same encoder
    (symmetric bi-encoder), which is common for passage retrieval tasks.
    
    COMPLEXITY ANALYSIS:
    
    Let n = number of indexed documents, d = embedding dimension
    
    - Query encoding: O(d) - single forward pass
    - Vector search: O(log n) with ANN index, O(n) without
    - Result processing: O(k) where k = top_k
    - Total: O(d + log n + k)
    
    EDGE OPTIMIZATION:
    
    - Embeddings cached in SQLite (see embeddings.py)
    - LanceDB uses memory-mapped files for efficient access
    - No reranking step to minimize model inference
    
    Attributes:
        hybrid_store: Reference to storage backend
        embeddings: Embedding model for query encoding
        top_k: Maximum number of results to return
        threshold: Minimum similarity score for filtering
    """

    def __init__(
        self,
        hybrid_store: HybridStore,
        embeddings: Embeddings,
        top_k: int = 10,
        threshold: float = 0.3,
    ):
        """
        Initialize vector retriever with storage and embedding model.

        Args:
            hybrid_store: HybridStore instance containing vector index
            embeddings: Embedding model for query encoding
            top_k: Maximum number of results to return (default: 10)
            threshold: Minimum similarity threshold (default: 0.3)
        
        Note:
            The threshold should be calibrated based on the embedding model
            and expected similarity score distribution. See verify_storage_fix.py
            for diagnostic tools.
        """
        self.hybrid_store = hybrid_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"VectorRetriever initialized: top_k={top_k}, threshold={threshold}"
        )

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve documents relevant to the query using vector similarity.
        
        ALGORITHM:
        
        1. Encode query into embedding vector
        2. Search vector index for nearest neighbors
        3. Filter results by similarity threshold
        4. Convert to RetrievalResult objects
        5. Return sorted by relevance score (descending)
        
        Args:
            query: Natural language query string
            
        Returns:
            List of RetrievalResult objects, sorted by relevance_score descending.
            Empty list if no results meet the threshold or on error.
        
        Performance:
            Typical latency: 10-50ms (depends on index size and hardware)
            - Query embedding: 5-20ms
            - Vector search: 5-20ms  
            - Result processing: <5ms
        """
        start_time = time.time()
        
        try:
            # Step 1: Encode query into embedding vector
            # This uses the same encoder as document indexing (symmetric bi-encoder)
            query_embedding = self.embeddings.embed_query(query)
            
            embed_time = time.time() - start_time
            
            # Step 2-3: Search vector index with threshold filtering
            # The vector_search method handles:
            # - Cosine similarity computation
            # - Distance to similarity conversion
            # - Threshold filtering
            vector_results = self.hybrid_store.vector_store.vector_search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                threshold=self.threshold,
            )
            
            search_time = time.time() - start_time - embed_time
            
            # Step 4: Convert to RetrievalResult objects
            results = []
            for result in vector_results:
                results.append(
                    RetrievalResult(
                        text=result["text"],
                        relevance_score=result["similarity"],
                        document_id=result["document_id"],
                        source_file=result["metadata"].get("source_file", "unknown"),
                        retrieval_method="vector",
                        metadata=result["metadata"],
                    )
                )
            
            total_time = time.time() - start_time
            
            # Log performance metrics
            self.logger.debug(
                f"VectorRetriever.retrieve: "
                f"{len(results)} results in {total_time*1000:.1f}ms "
                f"(embed={embed_time*1000:.1f}ms, search={search_time*1000:.1f}ms)"
            )
            
            return results

        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {str(e)}", exc_info=True)
            return []


# ============================================================================
# GRAPH RETRIEVER
# ============================================================================

class GraphRetriever:
    """
    Knowledge graph-based retrieval using structural relationships.
    
    SCIENTIFIC BACKGROUND:
    
    Graph-based retrieval exploits structural relationships between documents
    and entities that are not captured by embedding similarity alone. This
    approach is particularly effective for:
    
    - Multi-hop reasoning questions
    - Questions about document structure
    - Queries requiring entity relationships
    
    Graph Structure:
        Nodes: Document chunks, source files, extracted entities
        Edges: Relations (from_source, references, mentions, etc.)
    
    Retrieval Algorithm:
        1. Extract entities from query
        2. Match entities to graph nodes
        3. Traverse graph using BFS up to max_hops
        4. Score nodes by hop distance (closer = higher score)
    
    COMPLEXITY ANALYSIS:
    
    Let V = number of nodes, E = number of edges, h = max_hops
    
    - Entity extraction: O(|query|)
    - Node matching: O(V * |entities|) 
    - BFS traversal: O(V + E) worst case, typically O(branching_factor^h)
    - Total: O(V + E) worst case
    
    LIMITATIONS:
    
    Current implementation has simplified entity extraction (keyword-based).
    For production use, consider:
    - Named Entity Recognition (NER) using spaCy or similar
    - Entity linking to knowledge bases
    - Query understanding using LLM
    
    Attributes:
        hybrid_store: Reference to storage backend
        embeddings: Embedding model (for future entity ranking)
        top_k: Maximum number of results to return
        max_hops: Maximum traversal depth in graph
    """
    
    # Stopwords for simple entity extraction
    # These are filtered out when extracting potential entities from queries
    STOPWORDS = frozenset({
        "what", "how", "why", "when", "where", "who", "which",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "as", "into",
        "this", "that", "these", "those", "it", "its",
        "can", "could", "will", "would", "should", "may", "might",
        "about", "between", "through", "during", "before", "after",
    })

    def __init__(
        self,
        hybrid_store: HybridStore,
        embeddings: Embeddings,
        top_k: int = 5,
        max_hops: int = 2,
    ):
        """
        Initialize graph retriever with storage backend.

        Args:
            hybrid_store: HybridStore instance containing knowledge graph
            embeddings: Embedding model (reserved for future entity ranking)
            top_k: Maximum number of results to return (default: 5)
            max_hops: Maximum graph traversal depth (default: 2)
        
        Note on max_hops:
            - 1 hop: Direct neighbors only
            - 2 hops: Neighbors of neighbors (recommended default)
            - >2 hops: Often introduces too much noise ("small world" effect)
        """
        self.hybrid_store = hybrid_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.max_hops = max_hops
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"GraphRetriever initialized: top_k={top_k}, max_hops={max_hops}"
        )

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract potential entity mentions from query string.
        
        ALGORITHM:
        
        This is a simplified keyword-based extraction:
        1. Tokenize query by whitespace
        2. Remove punctuation from tokens
        3. Filter out stopwords
        4. Filter out short tokens (< 4 characters)
        5. Return remaining tokens as potential entities
        
        LIMITATIONS:
        
        - No phrase detection (e.g., "machine learning" -> two tokens)
        - No entity type recognition
        - Language-dependent stopword list
        
        FUTURE IMPROVEMENTS:
        
        For production systems, consider:
        - spaCy NER for named entity recognition
        - Phrase detection using n-grams
        - Entity linking to external knowledge bases
        - LLM-based query understanding

        Args:
            query: Natural language query string
            
        Returns:
            List of potential entity strings (lowercase)
        """
        # Tokenize and clean
        tokens = query.lower().split()
        entities = []
        
        for token in tokens:
            # Remove common punctuation
            cleaned = token.strip('.,!?;:()[]{}"\'-')
            
            # Filter criteria
            if (cleaned 
                and cleaned not in self.STOPWORDS 
                and len(cleaned) >= 4):
                entities.append(cleaned)
        
        # Fallback: use entire query if no entities extracted
        if not entities:
            entities = [query.lower().strip()]
            self.logger.debug(
                f"No entities extracted, using full query: '{query}'"
            )
        
        self.logger.debug(f"Extracted entities: {entities}")
        return entities

    def _match_entity_to_node(self, entity: str) -> Optional[str]:
        """
        Find best matching node in graph for an entity string.
        
        Current implementation uses simple substring matching.
        Returns the first node whose ID contains the entity string.
        
        Args:
            entity: Entity string to match (lowercase)
            
        Returns:
            Node ID if match found, None otherwise
        """
        graph = self.hybrid_store.graph_store.graph
        
        for node_id in graph.nodes():
            if entity in str(node_id).lower():
                return node_id
        
        return None

    def _compute_hop_score(self, hops: int) -> float:
        """
        Compute relevance score based on hop distance.
        
        SCORING FORMULA:
        
        score = 1.0 - (hops / (max_hops + 1))
        
        This gives:
        - hops=0: score=1.0 (direct match)
        - hops=1: score=0.67 (1 hop away)
        - hops=2: score=0.33 (2 hops away)
        
        The +1 in denominator ensures scores are always positive.
        
        Args:
            hops: Number of hops from matched entity
            
        Returns:
            Relevance score in (0, 1]
        """
        return 1.0 - (hops / (self.max_hops + 1))

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve documents using knowledge graph traversal.
        
        ALGORITHM:
        
        1. Extract entities from query
        2. For each entity:
           a. Find matching node in graph
           b. Perform BFS traversal up to max_hops
           c. Collect visited nodes with hop distances
        3. Merge results from all entity traversals
        4. Score nodes by minimum hop distance
        5. Return top_k results sorted by score
        
        Args:
            query: Natural language query string
            
        Returns:
            List of RetrievalResult objects, sorted by relevance_score descending.
            Empty list if graph is empty or no entities matched.
        
        Note:
            Graph retrieval returns entity references, not document text.
            In hybrid mode, these are combined with vector results which
            contain actual document content.
        """
        start_time = time.time()
        
        try:
            # Check if graph has any nodes
            graph = self.hybrid_store.graph_store.graph
            if graph.number_of_nodes() == 0:
                self.logger.warning("Knowledge graph is empty, skipping graph retrieval")
                return []
            
            # Step 1: Extract entities from query
            entities = self._extract_entities_from_query(query)
            
            # Step 2-3: Match entities and traverse graph
            # Dictionary to track minimum hop distance to each node
            all_visited: Dict[str, int] = {}
            
            for entity in entities:
                matched_node = self._match_entity_to_node(entity)
                
                if matched_node:
                    # BFS traversal from matched node
                    visited = self.hybrid_store.graph_store.graph_traversal(
                        start_entity=matched_node,
                        relation_types=None,  # Follow all relation types
                        max_hops=self.max_hops,
                    )
                    
                    # Merge with existing results (keep minimum hop distance)
                    for node_id, hops in visited.items():
                        if node_id not in all_visited:
                            all_visited[node_id] = hops
                        else:
                            all_visited[node_id] = min(all_visited[node_id], hops)
            
            # Step 4-5: Score and sort results
            scored_results = []
            for node_id, hops in all_visited.items():
                score = self._compute_hop_score(hops)
                scored_results.append((node_id, hops, score))
            
            # Sort by score descending, take top_k
            scored_results.sort(key=lambda x: x[2], reverse=True)
            top_results = scored_results[:self.top_k]
            
            # Convert to RetrievalResult objects
            results = []
            for node_id, hops, score in top_results:
                # Get node attributes if available
                node_data = graph.nodes.get(node_id, {})
                entity_type = node_data.get("entity_type", "unknown")
                
                results.append(
                    RetrievalResult(
                        text=f"Entity: {node_id}",
                        relevance_score=score,
                        document_id=str(node_id),
                        source_file=node_data.get("source_file", "graph"),
                        retrieval_method="graph",
                        metadata={
                            "hops": hops,
                            "entity_type": entity_type,
                            "matched_entities": entities,
                        },
                    )
                )
            
            elapsed_time = time.time() - start_time
            self.logger.debug(
                f"GraphRetriever.retrieve: "
                f"{len(results)} results in {elapsed_time*1000:.1f}ms"
            )
            
            return results

        except Exception as e:
            self.logger.error(f"Graph retrieval failed: {str(e)}", exc_info=True)
            return []


# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """
    Ensemble retriever combining vector and graph-based retrieval.
    
    SCIENTIFIC BACKGROUND:
    
    Ensemble methods combine multiple retrieval approaches to leverage their
    complementary strengths and mitigate individual weaknesses:
    
    Vector Retrieval Strengths:
    - Captures semantic similarity beyond lexical overlap
    - Effective for paraphrase and conceptual matching
    - Fast with approximate nearest neighbor indices
    
    Vector Retrieval Weaknesses:
    - May miss structural relationships
    - Sensitive to embedding model quality
    - Can fail on out-of-distribution queries
    
    Graph Retrieval Strengths:
    - Captures explicit relationships between concepts
    - Supports multi-hop reasoning
    - Interpretable retrieval paths
    
    Graph Retrieval Weaknesses:
    - Requires knowledge graph construction
    - Limited to known relationships
    - Entity extraction quality dependent
    
    ENSEMBLE COMBINATION:
    
    This implementation uses weighted score fusion:
    
    final_score = (vector_score * w_v + graph_score * w_g) / (w_v + w_g)
    
    where w_v and w_g are configurable weights.
    
    Alternative fusion methods (not implemented):
    - Reciprocal Rank Fusion (RRF)
    - CombSUM, CombMNZ
    - Learning-to-rank
    
    ABLATION STUDY SUPPORT:
    
    The configurable weights enable systematic ablation studies:
    
    Experiment 1: Vector-only (w_v=1.0, w_g=0.0)
    Experiment 2: Graph-only (w_v=0.0, w_g=1.0)
    Experiment 3: Equal weights (w_v=0.5, w_g=0.5)
    Experiment 4: Vector-heavy (w_v=0.7, w_g=0.3)
    Experiment 5: Graph-heavy (w_v=0.3, w_g=0.7)
    
    Compare metrics: Precision@k, Recall@k, MRR, nDCG, Latency
    
    DESIGN PATTERNS:
    
    - Strategy Pattern: Different retrieval strategies selected by mode
    - Composition: HybridRetriever composes Vector and Graph retrievers
    - Facade: Provides unified interface regardless of mode
    
    Attributes:
        config: RetrievalConfig with mode and weights
        hybrid_store: Reference to storage backend
        embeddings: Embedding model for query encoding
        vector_retriever: VectorRetriever instance
        graph_retriever: GraphRetriever instance
    """

    def __init__(
        self,
        config: RetrievalConfig,
        hybrid_store: HybridStore,
        embeddings: Embeddings,
    ):
        """
        Initialize hybrid retriever with configuration.

        Args:
            config: RetrievalConfig specifying mode, weights, and parameters
            hybrid_store: HybridStore instance containing vector and graph indices
            embeddings: Embedding model for query encoding
        
        The retriever initializes both sub-retrievers regardless of mode,
        allowing runtime mode switching without re-initialization.
        """
        self.config = config
        self.hybrid_store = hybrid_store
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Initialize sub-retrievers
        self.vector_retriever = VectorRetriever(
            hybrid_store=hybrid_store,
            embeddings=embeddings,
            top_k=config.top_k_vector,
            threshold=config.similarity_threshold,
        )

        self.graph_retriever = GraphRetriever(
            hybrid_store=hybrid_store,
            embeddings=embeddings,
            top_k=config.top_k_graph,
            max_hops=2,  # Default, could be added to config
        )

        self.logger.info(
            f"HybridRetriever initialized: "
            f"mode={config.mode.value}, "
            f"vector_weight={config.vector_weight}, "
            f"graph_weight={config.graph_weight}"
        )

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve documents relevant to query using configured mode.
        
        MODES:
        
        VECTOR: Uses only vector retriever
            - Fastest option
            - Best for semantic similarity queries
            - Use for initial baseline evaluation
        
        GRAPH: Uses only graph retriever
            - For structural relationship queries
            - Use to evaluate graph contribution
            - May return entity references instead of text
        
        HYBRID: Combines both retrievers
            - Most comprehensive results
            - Configurable weights for ablation
            - Recommended for production use
        
        Args:
            query: Natural language query string
            
        Returns:
            List of RetrievalResult objects, sorted by relevance_score descending.
            
        Raises:
            ValueError: If config contains unknown retrieval mode
        """
        start_time = time.time()
        
        if self.config.mode == RetrievalMode.VECTOR:
            results = self.vector_retriever.retrieve(query)
            
        elif self.config.mode == RetrievalMode.GRAPH:
            results = self.graph_retriever.retrieve(query)
            
        elif self.config.mode == RetrievalMode.HYBRID:
            # Get results from both retrievers
            vector_results = self.vector_retriever.retrieve(query)
            graph_results = self.graph_retriever.retrieve(query)
            
            # Combine using weighted ensemble
            results = self._ensemble_combine(vector_results, graph_results)
            
        else:
            raise ValueError(f"Unknown retrieval mode: {self.config.mode}")
        
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"HybridRetriever.retrieve: "
            f"mode={self.config.mode.value}, "
            f"{len(results)} results in {elapsed_time*1000:.1f}ms"
        )
        
        return results

    def _ensemble_combine(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Combine vector and graph results using weighted score fusion.
        
        ALGORITHM:
        
        1. Create unified result set indexed by document_id
        2. For each document, store vector_score and graph_score
        3. Compute final score using weighted average:
           score = (v_score * w_v + g_score * w_g) / (w_v + w_g)
        4. Sort by final score descending
        
        SCORE NORMALIZATION:
        
        Both vector and graph scores are assumed to be in [0, 1].
        The weighted average preserves this range.
        
        HANDLING MISSING SCORES:
        
        If a document appears in only one result set:
        - Missing score defaults to 0.0
        - This naturally penalizes single-source results
        
        Alternative approaches (not implemented):
        - Impute missing scores using similarity to present results
        - Use rank-based fusion instead of score-based
        
        Args:
            vector_results: Results from vector retriever
            graph_results: Results from graph retriever
            
        Returns:
            Combined results sorted by weighted score
        """
        # Dictionary to accumulate scores by document_id
        # Structure: {doc_id: {"text": ..., "vector_score": ..., "graph_score": ...}}
        combined: Dict[str, Dict[str, Any]] = {}

        # Process vector results
        for result in vector_results:
            combined[result.document_id] = {
                "text": result.text,
                "source_file": result.source_file,
                "metadata": result.metadata,
                "vector_score": result.relevance_score,
                "graph_score": 0.0,  # Default if not in graph results
            }

        # Process graph results
        for result in graph_results:
            if result.document_id not in combined:
                # New document from graph only
                combined[result.document_id] = {
                    "text": result.text,
                    "source_file": result.source_file,
                    "metadata": result.metadata,
                    "vector_score": 0.0,  # Default if not in vector results
                    "graph_score": result.relevance_score,
                }
            else:
                # Update existing with graph score
                combined[result.document_id]["graph_score"] = result.relevance_score
                
                # Merge metadata from graph result
                combined[result.document_id]["metadata"].update(result.metadata)

        # Compute final scores and create result objects
        results = []
        weight_sum = self.config.vector_weight + self.config.graph_weight
        
        for doc_id, data in combined.items():
            # Weighted average score
            final_score = (
                data["vector_score"] * self.config.vector_weight +
                data["graph_score"] * self.config.graph_weight
            ) / weight_sum

            results.append(
                RetrievalResult(
                    text=data["text"],
                    relevance_score=final_score,
                    document_id=doc_id,
                    source_file=data["source_file"],
                    retrieval_method="hybrid",
                    metadata={
                        **data["metadata"],
                        "vector_score": data["vector_score"],
                        "graph_score": data["graph_score"],
                    },
                )
            )

        # Sort by final score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        self.logger.debug(
            f"Ensemble combine: "
            f"{len(vector_results)} vector + {len(graph_results)} graph "
            f"-> {len(results)} combined"
        )

        return results

    def get_retrieval_statistics(
        self, 
        results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        Compute statistics for retrieval results.
        
        Useful for evaluation and debugging.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Dictionary containing:
            - count: Number of results
            - score_min: Minimum score
            - score_max: Maximum score  
            - score_mean: Mean score
            - score_std: Standard deviation of scores
            - methods: Count of results by retrieval method
        """
        if not results:
            return {
                "count": 0,
                "score_min": 0.0,
                "score_max": 0.0,
                "score_mean": 0.0,
                "score_std": 0.0,
                "methods": {},
            }
        
        scores = [r.relevance_score for r in results]
        methods = {}
        for r in results:
            methods[r.retrieval_method] = methods.get(r.retrieval_method, 0) + 1
        
        import statistics
        
        return {
            "count": len(results),
            "score_min": min(scores),
            "score_max": max(scores),
            "score_mean": statistics.mean(scores),
            "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "methods": methods,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_retriever_from_config(
    config_dict: Dict[str, Any],
    hybrid_store: HybridStore,
    embeddings: Embeddings,
) -> HybridRetriever:
    """
    Factory function to create HybridRetriever from configuration dictionary.
    
    This function simplifies retriever creation from YAML configuration files.
    
    Args:
        config_dict: Dictionary with retrieval configuration
            Expected keys:
            - retrieval_mode: "vector", "graph", or "hybrid"
            - top_k_vectors: int
            - top_k_entities: int  
            - vector_weight: float
            - graph_weight: float
            - similarity_threshold: float
        hybrid_store: Initialized HybridStore instance
        embeddings: Initialized embedding model
        
    Returns:
        Configured HybridRetriever instance
    """
    retrieval_config = RetrievalConfig(
        mode=RetrievalMode(config_dict.get("retrieval_mode", "hybrid")),
        top_k_vector=config_dict.get("top_k_vectors", 10),
        top_k_graph=config_dict.get("top_k_entities", 5),
        vector_weight=config_dict.get("vector_weight", 1.0),
        graph_weight=config_dict.get("graph_weight", 0.0),
        similarity_threshold=config_dict.get("similarity_threshold", 0.3),
    )
    
    return HybridRetriever(
        config=retrieval_config,
        hybrid_store=hybrid_store,
        embeddings=embeddings,
    )