"""
Hybrid Retrieval Engine: Vector + Graph-basierte Anfrageverarbeitung.

Scientific Foundation:
- Dense Retrieval (Vectors): Sub-millisecond latency, semantic matching
- Sparse Retrieval (Graph): Strukturelle Relationen, Multi-Hop Reasoning
- Ensemble Approach: Gewichtete Kombination beider Modalitäten reduziert
  Fehler beider Systeme (vgl. Hybrid Retrieval, Ma et al., 2021)
- Reranking optional: Cross-Encoder für Fine-grained Relevanz
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain.embeddings.base import Embeddings
import networkx as nx

from src.storage import HybridStore


logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    """Retrieval-Modi."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    """Konfiguration für Hybrid Retrieval."""
    mode: RetrievalMode
    top_k_vector: int
    top_k_graph: int
    vector_weight: float
    graph_weight: float
    similarity_threshold: float


@dataclass
class RetrievalResult:
    """Struktur für Retrieval-Ergebnisse."""
    text: str
    relevance_score: float
    document_id: str
    source_file: str
    retrieval_method: str  # "vector", "graph", "hybrid"
    metadata: Dict[str, Any]


class VectorRetriever:
    """
    Vector-basierte Dichte-Retrieval.
    
    Scientific Rationale:
    Embedding-basierte Retrieval hat sich als State-of-the-Art
    für semantische Matching etabliert. Bi-Encoder Ansatz
    (Query + Document separate Encoding) ist praktisch überlegen
    zu Cross-Encodern für Skalierbarkeit auf Edge (vgl. Thakur et al.,
    BEIR Benchmark, 2021).
    """

    def __init__(
        self,
        hybrid_store: HybridStore,
        embeddings: Embeddings,
        top_k: int = 5,
        threshold: float = 0.5,
    ):
        """
        Initialisiere Vector Retriever.

        Args:
            hybrid_store: HybridStore Instance
            embeddings: Embedding-Modell
            top_k: Anzahl Top Results
            threshold: Similarity Threshold
        """
        self.hybrid_store = hybrid_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.threshold = threshold
        self.logger = logger

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Führe Vector-basierte Retrieval durch.

        Args:
            query: Nutzer-Anfrage

        Returns:
            Liste von RetrievalResult-Objekten
        """
        try:
            # Embedde Query
            query_embedding = self.embeddings.embed_query(query)

            # Vector Search im Store
            vector_results = self.hybrid_store.vector_store.vector_search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                threshold=self.threshold,
            )

            # Konvertiere zu RetrievalResult
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

            self.logger.debug(f"Vector Retrieval: {len(results)} Results für Query: '{query}'")
            return results

        except Exception as e:
            self.logger.error(f"Fehler in Vector Retrieval: {str(e)}")
            return []


class GraphRetriever:
    """
    Graph-basierte strukturelle Retrieval.
    
    Scientific Rationale:
    Graphen ermöglichen Multi-Hop Reasoning durch explizite
    semantische Relationen. Besonders effektiv für Fragen, die
    Verbindungen zwischen Konzepten erfordern (vgl. Graph-RAG,
    Yu et al., 2024; Knowledge Graph QA, Lan et al., 2021).
    """

    def __init__(
        self,
        hybrid_store: HybridStore,
        embeddings: Embeddings,
        top_k: int = 3,
        max_hops: int = 2,
    ):
        """
        Initialisiere Graph Retriever.

        Args:
            hybrid_store: HybridStore Instance
            embeddings: Embedding-Modell (für Entity Ranking)
            top_k: Anzahl Top Entities
            max_hops: Maximale Graphtravesal-Distanz
        """
        self.hybrid_store = hybrid_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.max_hops = max_hops
        self.logger = logger

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Einfache Entity Extraction aus Query.
        
        TODO: Ersetze durch NER-basierte Extraction (spaCy, Ollama)
        für Production-Qualität.

        Args:
            query: Query-String

        Returns:
            Liste potentieller Entity-Strings
        """
        """
        Improved entity extraction with keyword focus.
        """
        # Remove question words
        stopwords = {"what", "how", "why", "when", "where", "who", 
                    "is", "are", "the", "a", "an"}
        
        # Extract meaningful words
        words = query.lower().split()
        entities = []
        
        for word in words:
            cleaned = word.strip('.,!?')
            if cleaned and cleaned not in stopwords and len(cleaned) > 3:
                entities.append(cleaned)
        
        # Fallback: use full query as entity
        if not entities:
            entities = [query.lower()]
        
        return entities

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Führe Graph-basierte Retrieval durch.

        Args:
            query: Nutzer-Anfrage

        Returns:
            Liste von RetrievalResult-Objekten
        """
        try:
            entities = self._extract_entities_from_query(query)
            
            if not self.hybrid_store.graph_store.graph.number_of_nodes():
                self.logger.warning("Graph ist leer, keine Retrieval möglich")
                return []

            # Sammle Entities und ihre Nachbarn
            all_entities = {}
            for entity in entities:
                # Finde beste Match im Graph
                best_match = None
                for node in self.hybrid_store.graph_store.graph.nodes():
                    if entity.lower() in str(node).lower():
                        best_match = node
                        break

                if best_match:
                    # Traversiere Graph
                    neighbors = self.hybrid_store.graph_store.graph_traversal(
                        best_match, relation_types=None
                    )
                    all_entities.update(neighbors)

            # Konvertiere Entities zu Results (einfache Impl.)
            results = []
            for i, (entity_id, hops) in enumerate(sorted(
                all_entities.items(),
                key=lambda x: x[1]
            )[:self.top_k]):
                # Score: 1 - (hops / max_hops)
                relevance = 1.0 - (hops / (self.max_hops + 1))
                
                results.append(
                    RetrievalResult(
                        text=f"Entity: {entity_id}",
                        relevance_score=relevance,
                        document_id=str(entity_id),
                        source_file="graph",
                        retrieval_method="graph",
                        metadata={"hops": hops, "type": "entity"},
                    )
                )

            self.logger.debug(f"Graph Retrieval: {len(results)} Results")
            return results

        except Exception as e:
            self.logger.error(f"Fehler in Graph Retrieval: {str(e)}")
            return []


class HybridRetriever:
    """
    Ensemble Retriever kombiniert Vector + Graph.
    
    Design Pattern: Strategy Pattern + Composition
    
    Scientific Rationale:
    Ensemble Methods reduzieren Bias einzelner Retriever.
    Gewichtete Kombination ermöglicht ablation studies für
    Thesis-Validierung (vgl. Hybrid Retrieval, Khattab et al.,
    ColBERT 2020; DSP Framework).
    """

    def __init__(
        self,
        config: RetrievalConfig,
        hybrid_store: HybridStore,
        embeddings: Embeddings,
    ):
        """
        Initialisiere Hybrid Retriever.

        Args:
            config: RetrievalConfig
            hybrid_store: HybridStore Instance
            embeddings: Embedding-Modell
        """
        self.config = config
        self.hybrid_store = hybrid_store
        self.embeddings = embeddings

        # Initialisiere Sub-Retriever
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
        )

        self.logger = logger
        self.logger.info(f"HybridRetriever initialisiert: mode={config.mode}")

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Führe Hybrid Retrieval durch.

        Args:
            query: Nutzer-Anfrage

        Returns:
            Gekombinete und rerankte Results
        """
        if self.config.mode == RetrievalMode.VECTOR:
            return self.vector_retriever.retrieve(query)

        elif self.config.mode == RetrievalMode.GRAPH:
            return self.graph_retriever.retrieve(query)

        elif self.config.mode == RetrievalMode.HYBRID:
            # Ensemble
            vector_results = self.vector_retriever.retrieve(query)
            graph_results = self.graph_retriever.retrieve(query)

            # Kombiniere und reranke
            return self._ensemble_combine(vector_results, graph_results)

        else:
            raise ValueError(f"Unbekannter Retrieval Mode: {self.config.mode}")

    def _ensemble_combine(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Kombiniere Vector + Graph Results mit gewichteter Normalisierung.

        Scientific Rationale:
        Weighted Ensemble normalisiert Scores zu [0,1] und kombiniert
        mit konfigurierbaren Gewichten. Dies ermöglicht ablation
        studies: vector_weight=1, graph_weight=0 → nur Vector, etc.

        Args:
            vector_results: Vektorsuch-Resultate
            graph_results: Graph-Suche Resultate

        Returns:
            Kombinierte und rerankte Resultate
        """
        # Kombiniere in Dict nach document_id
        combined = {}

        # Vektor-Results
        for result in vector_results:
            combined[result.document_id] = {
                "text": result.text,
                "source_file": result.source_file,
                "metadata": result.metadata,
                "vector_score": result.relevance_score,
                "graph_score": 0.0,
            }

        # Graph-Results
        for result in graph_results:
            if result.document_id not in combined:
                combined[result.document_id] = {
                    "text": result.text,
                    "source_file": result.source_file,
                    "metadata": result.metadata,
                    "vector_score": 0.0,
                    "graph_score": 0.0,
                }
            combined[result.document_id]["graph_score"] = result.relevance_score

        # Berechne finale Scores
        results = []
        for doc_id, data in combined.items():
            # Gewichtete Kombination
            final_score = (
                data["vector_score"] * self.config.vector_weight +
                data["graph_score"] * self.config.graph_weight
            ) / (self.config.vector_weight + self.config.graph_weight)

            results.append(
                RetrievalResult(
                    text=data["text"],
                    relevance_score=final_score,
                    document_id=doc_id,
                    source_file=data["source_file"],
                    retrieval_method="hybrid",
                    metadata=data["metadata"],
                )
            )

        # Sortiere nach Score
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        self.logger.debug(
            f"Ensemble Combine: {len(vector_results)} vector + "
            f"{len(graph_results)} graph → {len(results)} combined"
        )

        return results