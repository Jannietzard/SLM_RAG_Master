"""
Hybrid Retrieval Engine: Vector and Graph-based Query Processing

Version: 3.1.0 - UNIFIED
Author: Edge-RAG Research Project
Last Modified: 2026-01-25

===============================================================================
CONSOLIDATION: retrieval.py + retrieval_kuzu.py → UNIFIED retrieval.py
===============================================================================

This module combines both NetworkX (v2.1.0) and KuzuDB (v3.0.0) retrieval
implementations into a single file with automatic backend detection.

The appropriate graph traversal method is selected based on the underlying
graph store type (KuzuGraphStore vs NetworkXGraphStore).

SCIENTIFIC FOUNDATION:

1. DENSE RETRIEVAL (Vector-based):
   - Embedding similarity via LanceDB
   - Reference: Karpukhin et al. (2020). "Dense Passage Retrieval"

2. SPARSE/STRUCTURAL RETRIEVAL (Graph-based):
   - KuzuDB: Native Cypher path queries (10-100x faster)
   - NetworkX: Python BFS (fallback)
   - Reference: Yu et al. (2024). "Graph-RAG"

3. HYBRID ENSEMBLE:
   - Weighted combination: score = (v_score * w_v + g_score * w_g) / (w_v + w_g)
   - Reference: Ma et al. (2021). "A Replication Study of Dense Passage Retriever"

ABLATION STUDY SUPPORT:
    - Vector-only:  vector_weight=1.0, graph_weight=0.0
    - Graph-only:   vector_weight=0.0, graph_weight=1.0
    - Hybrid 70/30: vector_weight=0.7, graph_weight=0.3

===============================================================================
"""

import logging
import time
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain.embeddings.base import Embeddings

from src.data_layer.storage import HybridStore, KuzuGraphStore, NetworkXGraphStore


logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS AND CONFIGURATION
# ============================================================================

class RetrievalMode(str, Enum):
    """Retrieval mode enumeration for ablation studies."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    """
    Configuration for hybrid retrieval system.
    
    Attributes:
        mode: Retrieval mode (vector, graph, or hybrid)
        top_k_vector: Number of results from vector retrieval
        top_k_graph: Number of results from graph retrieval
        vector_weight: Weight for vector scores in ensemble [0.0, 1.0]
        graph_weight: Weight for graph scores in ensemble [0.0, 1.0]
        similarity_threshold: Minimum similarity score for filtering
        max_hops: Maximum graph traversal depth (for graph retrieval)
        expand_context: Whether to include surrounding chunks (KuzuDB only)
    """
    mode: RetrievalMode
    top_k_vector: int
    top_k_graph: int
    vector_weight: float
    graph_weight: float
    similarity_threshold: float
    max_hops: int = 2
    expand_context: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.top_k_vector < 1:
            raise ValueError(f"top_k_vector must be >= 1: {self.top_k_vector}")
        if self.top_k_graph < 1:
            raise ValueError(f"top_k_graph must be >= 1: {self.top_k_graph}")
        if not (0.0 <= self.vector_weight <= 1.0):
            raise ValueError(f"vector_weight must be in [0,1]: {self.vector_weight}")
        if not (0.0 <= self.graph_weight <= 1.0):
            raise ValueError(f"graph_weight must be in [0,1]: {self.graph_weight}")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be in [0,1]: {self.similarity_threshold}")
        if self.vector_weight == 0.0 and self.graph_weight == 0.0:
            raise ValueError("At least one weight must be non-zero")


@dataclass
class RetrievalResult:
    """
    Container for individual retrieval results.
    
    Score Interpretation:
        0.0 - 0.3: Low relevance
        0.3 - 0.5: Moderate relevance
        0.5 - 0.7: Good relevance
        0.7 - 1.0: High relevance
    """
    text: str
    relevance_score: float
    document_id: str
    source_file: str
    retrieval_method: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
    
    Uses LanceDB for efficient approximate nearest neighbor search.
    """

    def __init__(
        self,
        hybrid_store: HybridStore,
        embeddings: Embeddings,
        top_k: int = 10,
        threshold: float = 0.3,
    ):
        self.hybrid_store = hybrid_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"VectorRetriever: top_k={top_k}, threshold={threshold}")

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve documents using vector similarity."""
        start_time = time.time()
        
        try:
            # Encode query
            query_embedding = self.embeddings.embed_query(query)
            embed_time = time.time() - start_time
            
            # Search vector store
            vector_results = self.hybrid_store.vector_store.vector_search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                threshold=self.threshold,
            )
            
            # Convert to RetrievalResult
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
            
            elapsed = (time.time() - start_time) * 1000
            self.logger.debug(
                f"VectorRetriever: {len(results)} results in {elapsed:.1f}ms "
                f"(embed={embed_time*1000:.1f}ms)"
            )
            
            return results

        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}", exc_info=True)
            return []


# ============================================================================
# GRAPH RETRIEVER (Unified: KuzuDB + NetworkX)
# ============================================================================

class GraphRetriever:
    """
    Knowledge graph-based retrieval with automatic backend detection.
    
    Uses Cypher queries for KuzuDB (fast) or Python BFS for NetworkX (fallback).
    """
    
    # Stopwords for entity extraction
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
        expand_context: bool = True,
    ):
        self.hybrid_store = hybrid_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.max_hops = max_hops
        self.expand_context = expand_context
        self.logger = logging.getLogger(__name__)
        
        # Detect backend type
        self.is_kuzu = isinstance(hybrid_store.graph_store, KuzuGraphStore)
        backend_name = "KuzuDB" if self.is_kuzu else "NetworkX"
        
        self.logger.info(
            f"GraphRetriever: backend={backend_name}, max_hops={max_hops}, top_k={top_k}"
        )

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract potential entity mentions from query string.
        
        Uses simple keyword-based extraction. For better results,
        consider integrating spaCy NER or GLiNER.
        """
        tokens = query.lower().split()
        entities = []
        
        for token in tokens:
            cleaned = token.strip('.,!?;:()[]{}"\'-')
            if cleaned and cleaned not in self.STOPWORDS and len(cleaned) >= 3:
                entities.append(cleaned)
        
        # Also try bigrams for multi-word entities
        words = [t.strip('.,!?;:()[]{}"\'-') for t in tokens]
        for i in range(len(words) - 1):
            if words[i] not in self.STOPWORDS and words[i+1] not in self.STOPWORDS:
                if len(words[i]) >= 2 and len(words[i+1]) >= 2:
                    bigram = f"{words[i]} {words[i+1]}"
                    entities.append(bigram)
        
        # Fallback: use entire query if no entities extracted
        if not entities:
            entities = [query.lower().strip()]
        
        self.logger.debug(f"Extracted entities: {entities}")
        return entities

    def _compute_hop_score(self, hops: int) -> float:
        """Compute relevance score based on hop distance."""
        return 1.0 - (hops / (self.max_hops + 1))

    def _search_kuzu(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Search using KuzuDB Cypher queries."""
        matches = []
        graph_store = self.hybrid_store.graph_store
        
        for entity in entities:
            try:
                # Text search in DocumentChunk nodes
                safe_entity = entity.replace("'", "''")
                query = f"""
                    MATCH (c:DocumentChunk)
                    WHERE c.text CONTAINS '{safe_entity}'
                    RETURN c.chunk_id, c.text, c.source_file
                    LIMIT 10
                """
                result = graph_store.conn.execute(query)
                while result.has_next():
                    row = result.get_next()
                    matches.append({
                        "chunk_id": row[0],
                        "text": row[1] or "",
                        "source_file": row[2] or "unknown",
                        "match_type": "direct",
                        "hops": 0,
                    })
            except Exception as e:
                self.logger.debug(f"Kuzu search warning: {e}")
        
        # Expand with graph traversal for context
        if self.expand_context and matches:
            for match in list(matches):
                try:
                    related = graph_store.find_related_chunks(
                        match["chunk_id"], 
                        max_hops=self.max_hops
                    )
                    for rel in related:
                        if rel["chunk_id"] not in [m["chunk_id"] for m in matches]:
                            rel["match_type"] = "context"
                            matches.append(rel)
                except Exception as e:
                    self.logger.debug(f"Context expansion warning: {e}")
        
        return matches

    def _search_networkx(self, entities: List[str]) -> Dict[str, Dict[str, Any]]:
        """Search using NetworkX graph traversal."""
        all_results = {}
        graph = self.hybrid_store.graph_store.graph
        
        for entity in entities:
            # Find matching nodes
            for node_id in graph.nodes():
                if entity in str(node_id).lower():
                    # BFS traversal
                    traversal = self.hybrid_store.graph_store.graph_traversal(
                        start_entity=node_id,
                        max_hops=self.max_hops,
                    )
                    
                    for visited_node, hops in traversal.items():
                        if visited_node not in all_results or all_results[visited_node]["hops"] > hops:
                            node_data = graph.nodes.get(visited_node, {})
                            all_results[visited_node] = {
                                "hops": hops,
                                "entity_type": node_data.get("entity_type", "unknown"),
                                "source_file": node_data.get("source_file", "unknown"),
                            }
        
        return all_results

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve documents using knowledge graph traversal."""
        start_time = time.time()
        
        try:
            # Check if graph is empty
            stats = self.hybrid_store.graph_store.get_statistics()
            node_count = stats.get("document_chunks", stats.get("nodes", 0))
            
            if node_count == 0:
                self.logger.warning("Graph is empty")
                return []
            
            # Extract entities from query
            entities = self._extract_entities(query)
            
            results = []
            
            if self.is_kuzu:
                # KuzuDB: Use Cypher text search + traversal
                matches = self._search_kuzu(entities)
                
                # Deduplicate and score
                seen_ids = set()
                for match in matches:
                    if match["chunk_id"] not in seen_ids:
                        seen_ids.add(match["chunk_id"])
                        
                        hops = match.get("hops", 0)
                        match_type = match.get("match_type", "direct")
                        
                        # Score based on hop distance and match type
                        base_score = self._compute_hop_score(hops)
                        if match_type == "direct":
                            base_score = min(1.0, base_score + 0.2)
                        elif match_type == "context":
                            base_score = max(0.1, base_score - 0.1)
                        
                        results.append(
                            RetrievalResult(
                                text=match.get("text", ""),
                                relevance_score=base_score,
                                document_id=match["chunk_id"],
                                source_file=match.get("source_file", "unknown"),
                                retrieval_method="graph",
                                metadata={
                                    "hops": hops,
                                    "match_type": match_type,
                                    "entities": entities,
                                },
                            )
                        )
            else:
                # NetworkX: Use Python BFS
                all_results = self._search_networkx(entities)
                
                for node_id, data in all_results.items():
                    hops = data.get("hops", 0)
                    score = self._compute_hop_score(hops)
                    
                    results.append(
                        RetrievalResult(
                            text=f"Entity: {node_id}",
                            relevance_score=score,
                            document_id=str(node_id),
                            source_file=data.get("source_file", "graph"),
                            retrieval_method="graph",
                            metadata={
                                "hops": hops,
                                "entity_type": data.get("entity_type", "unknown"),
                                "matched_entities": entities,
                            },
                        )
                    )
            
            # Sort and limit
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            results = results[:self.top_k]
            
            elapsed = (time.time() - start_time) * 1000
            self.logger.debug(f"GraphRetriever: {len(results)} results in {elapsed:.1f}ms")
            
            return results

        except Exception as e:
            self.logger.error(f"Graph retrieval failed: {e}", exc_info=True)
            return []


# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """
    Ensemble retriever combining vector and graph-based retrieval.
    
    Supports three modes for ablation studies:
    - VECTOR: Vector retrieval only
    - GRAPH: Graph retrieval only
    - HYBRID: Weighted ensemble of both
    """

    def __init__(
        self,
        config: RetrievalConfig,
        hybrid_store: HybridStore,
        embeddings: Embeddings,
    ):
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
            max_hops=config.max_hops,
            expand_context=config.expand_context,
        )

        self.logger.info(
            f"HybridRetriever: mode={config.mode.value}, "
            f"v_weight={config.vector_weight}, g_weight={config.graph_weight}"
        )

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve documents using configured mode."""
        start_time = time.time()
        
        if self.config.mode == RetrievalMode.VECTOR:
            results = self.vector_retriever.retrieve(query)
            
        elif self.config.mode == RetrievalMode.GRAPH:
            results = self.graph_retriever.retrieve(query)
            
        elif self.config.mode == RetrievalMode.HYBRID:
            vector_results = self.vector_retriever.retrieve(query)
            graph_results = self.graph_retriever.retrieve(query)
            results = self._ensemble_combine(vector_results, graph_results)
            
        else:
            raise ValueError(f"Unknown retrieval mode: {self.config.mode}")
        
        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"HybridRetriever: {len(results)} results in {elapsed:.1f}ms")
        
        return results

    def _ensemble_combine(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Combine vector and graph results using weighted score fusion."""
        combined: Dict[str, Dict[str, Any]] = {}

        # Add vector results
        for result in vector_results:
            combined[result.document_id] = {
                "text": result.text,
                "source_file": result.source_file,
                "metadata": result.metadata,
                "vector_score": result.relevance_score,
                "graph_score": 0.0,
            }

        # Add/merge graph results
        for result in graph_results:
            if result.document_id not in combined:
                combined[result.document_id] = {
                    "text": result.text,
                    "source_file": result.source_file,
                    "metadata": result.metadata,
                    "vector_score": 0.0,
                    "graph_score": result.relevance_score,
                }
            else:
                combined[result.document_id]["graph_score"] = result.relevance_score
                combined[result.document_id]["metadata"].update(result.metadata)

        # Compute final scores
        results = []
        weight_sum = self.config.vector_weight + self.config.graph_weight
        
        for doc_id, data in combined.items():
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

        # Sort by final score
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        self.logger.debug(
            f"Ensemble: {len(vector_results)} vector + {len(graph_results)} graph "
            f"-> {len(results)} combined"
        )

        return results

    def get_statistics(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Compute statistics for retrieval results."""
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
        
        return {
            "count": len(results),
            "score_min": min(scores),
            "score_max": max(scores),
            "score_mean": statistics.mean(scores),
            "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "methods": methods,
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_retriever_from_config(
    config_dict: Dict[str, Any],
    hybrid_store: HybridStore,
    embeddings: Embeddings,
) -> HybridRetriever:
    """
    Factory function to create HybridRetriever from configuration dictionary.
    
    Args:
        config_dict: Dictionary with retrieval configuration
        hybrid_store: Initialized HybridStore instance
        embeddings: Initialized embedding model
        
    Returns:
        Configured HybridRetriever instance
    """
    retrieval_config = RetrievalConfig(
        mode=RetrievalMode(config_dict.get("retrieval_mode", "hybrid")),
        top_k_vector=config_dict.get("top_k_vectors", 10),
        top_k_graph=config_dict.get("top_k_entities", 5),
        vector_weight=config_dict.get("vector_weight", 0.7),
        graph_weight=config_dict.get("graph_weight", 0.3),
        similarity_threshold=config_dict.get("similarity_threshold", 0.3),
        max_hops=config_dict.get("max_hops", 2),
        expand_context=config_dict.get("expand_context", True),
    )
    
    return HybridRetriever(
        config=retrieval_config,
        hybrid_store=hybrid_store,
        embeddings=embeddings,
    )