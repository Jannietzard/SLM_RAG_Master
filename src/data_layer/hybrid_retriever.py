"""
Hybrid Retriever mit Reciprocal Rank Fusion (RRF)

Version: 4.0.0 - MASTERTHESIS IMPLEMENTATION
Author: Edge-RAG Research Project

===============================================================================
IMPLEMENTATION GEMÄSS MASTERTHESIS ABSCHNITT 2.6
===============================================================================

Hybrid Retriever orchestriert parallele Suchen:
    1. Vector Retrieval (LanceDB): ANN-Suche, Top-K=20, 8-12ms
    2. Graph Retrieval (KuzuDB): Entity-basiert, 1-Hop + 2-Hop, Top-K=10

RRF-Fusion (Reciprocal Rank Fusion):
    - Formel: RRF(d) = Σ 1/(k + rank_i(d))
    - k = 60 (Standard-Konstante)
    - Cross-Source Boost für Chunks in beiden Pfaden

Vorteile von RRF vs. Weighted Fusion:
    - Rank-basiert statt Score-basiert (robuster)
    - Keine Score-Normalisierung nötig
    - Bessere Fusion heterogener Retrieval-Methoden

Referenz: Cormack, Clarke & Büttcher (2009). "Reciprocal Rank Fusion 
Outperforms Condorcet and Individual Rank Learning Methods"

===============================================================================
"""

import logging
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class RetrievalMode(str, Enum):
    """Retrieval-Modi für Ablation Studies."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    """Konfiguration für Hybrid Retriever."""
    
    # Mode
    mode: RetrievalMode = RetrievalMode.HYBRID
    
    # Vector Retrieval
    vector_top_k: int = 20
    
    # Graph Retrieval
    graph_top_k: int = 10
    max_hops: int = 2
    
    # RRF Parameters
    rrf_k: int = 60  # RRF-Konstante
    
    # Fusion
    final_top_k: int = 10
    cross_source_boost: float = 1.2  # Boost für Chunks in beiden Quellen
    
    # Thresholds
    similarity_threshold: float = 0.3
    
    # SpaCy für Query Entity Extraction
    spacy_model: str = "en_core_web_sm"
    
    # Alias parameters for API compatibility
    top_k_vector: int = None  # Alias for vector_top_k
    top_k_graph: int = None   # Alias for graph_top_k
    vector_weight: float = 0.7  # Weight for vector results in fusion
    graph_weight: float = 0.3   # Weight for graph results in fusion
    
    def __post_init__(self):
        """Handle alias parameters."""
        # If alias parameters provided, use them
        if self.top_k_vector is not None:
            self.vector_top_k = self.top_k_vector
        if self.top_k_graph is not None:
            self.graph_top_k = self.top_k_graph
        # Set aliases to actual values for consistency
        self.top_k_vector = self.vector_top_k
        self.top_k_graph = self.graph_top_k


@dataclass
class RetrievalResult:
    """Einzelnes Retrieval-Ergebnis."""
    chunk_id: str
    text: str
    source_doc: str
    position: int
    
    # Scores
    rrf_score: float
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    graph_score: Optional[float] = None
    graph_rank: Optional[int] = None
    
    # Metadata
    retrieval_method: str = "hybrid"  # "vector", "graph", "hybrid"
    hop_distance: Optional[int] = None
    matched_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_doc": self.source_doc,
            "position": self.position,
            "rrf_score": self.rrf_score,
            "vector_score": self.vector_score,
            "vector_rank": self.vector_rank,
            "graph_score": self.graph_score,
            "graph_rank": self.graph_rank,
            "retrieval_method": self.retrieval_method,
            "hop_distance": self.hop_distance,
            "matched_entities": self.matched_entities,
        }


@dataclass
class RetrievalMetrics:
    """Performance-Metriken für Retrieval."""
    total_time_ms: float
    vector_time_ms: float
    graph_time_ms: float
    fusion_time_ms: float
    vector_results: int
    graph_results: int
    final_results: int
    query_entities: List[str]


# ============================================================================
# QUERY ENTITY EXTRACTOR
# ============================================================================

class QueryEntityExtractor:
    """
    Extrahiert Entitäten aus Query für Graph Retrieval.
    
    Verwendet SpaCy NER mit Confidence-Threshold.
    Latenz: 3-5ms pro Query.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        self._load_model(spacy_model)
    
    def _load_model(self, model_name: str):
        """Lade SpaCy Modell."""
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            logger.info(f"SpaCy model loaded for query analysis: {model_name}")
        except Exception as e:
            logger.warning(f"SpaCy not available: {e}")
            self.nlp = None
    
    def extract(self, query: str, confidence_threshold: float = 0.7) -> List[str]:
        """
        Extrahiere Entitäten aus Query.
        
        Args:
            query: User Query
            confidence_threshold: Minimum Confidence
            
        Returns:
            Liste von Entity-Namen
        """
        if self.nlp is None:
            return self._fallback_extract(query)
        
        doc = self.nlp(query)
        entities = []
        
        for ent in doc.ents:
            # SpaCy gibt keine explizite Confidence, nutze Heuristiken
            # Längere Entities und bekannte Types sind typischerweise zuverlässiger
            if len(ent.text) > 2 and ent.label_ in [
                "PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "PRODUCT"
            ]:
                entities.append(ent.text)
        
        # Auch Proper Nouns ohne NER-Label
        for token in doc:
            if token.pos_ == "PROPN" and token.text not in entities:
                if len(token.text) > 2:
                    entities.append(token.text)
        
        return entities
    
    def _fallback_extract(self, query: str) -> List[str]:
        """Regex-basierte Fallback-Extraktion."""
        import re
        
        entities = []
        
        # Capitalized words/phrases
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query):
            entity = match.group(1)
            if len(entity) > 2:
                entities.append(entity)
        
        # Quoted strings
        for match in re.finditer(r'"([^"]+)"', query):
            entities.append(match.group(1))
        
        return entities


# ============================================================================
# RRF FUSION
# ============================================================================
    
class RRFFusion:
    def __init__(self, k: int = 60, cross_source_boost: float = 1.2):
        """
        Args:
            k: RRF-Konstante
            cross_source_boost: Additiver Bonus für Cross-Source Chunks
                               (interpretiert als zusätzliche RRF-Punkte)
        """
        self.k = k
        self.cross_source_boost = cross_source_boost
    
    def fuse(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        final_top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        RRF Fusion mit additiver Cross-Source Boost.
        
        FORMEL:
            RRF(d) = Σ 1/(k + rank_i(d)) + BONUS
            
            wobei BONUS = cross_source_boost, falls d in beiden Sources
        """
        rrf_scores = defaultdict(float)
        chunk_data = {}
        vector_ranks = {}
        graph_ranks = {}
        vector_scores = {}
        graph_scores = {}
        graph_metadata = {}
        
        # Vector Ranks
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.get("chunk_id")
            rrf_scores[chunk_id] += 1.0 / (self.k + rank)
            vector_ranks[chunk_id] = rank
            vector_scores[chunk_id] = result.get("relevance_score", 0)
            
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "source_doc": result.get("source_doc", "unknown"),
                    "position": result.get("position", 0),
                }
        
        # Graph Ranks
        for rank, result in enumerate(graph_results, start=1):
            chunk_id = result.get("chunk_id")
            rrf_scores[chunk_id] += 1.0 / (self.k + rank)
            graph_ranks[chunk_id] = rank
            graph_scores[chunk_id] = result.get("confidence", 0)
            
            graph_metadata[chunk_id] = {
                "hop_distance": result.get("hop", 1),
                "matched_entities": [
                    result.get("entity_name", ""),
                    result.get("entity_from", ""),
                    result.get("entity_to", ""),
                ],
            }
            
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "source_doc": result.get("source_doc", "unknown"),
                    "position": result.get("position", 0),
                }
        
        # ========================================================================
        # CORRECTED: Additiver Cross-Source Boost
        # ========================================================================
        in_both = set(vector_ranks.keys()) & set(graph_ranks.keys())
        for chunk_id in in_both:
            # ADDITIVE statt multiplikativ
            # Interpretiere cross_source_boost als zusätzliche RRF-Punkte
            # Typischer Wert: 0.02 - 0.05 (entspricht Rank-Verbesserung um ~3-5)
            bonus = self.cross_source_boost / (self.k + 1)
            rrf_scores[chunk_id] += bonus
        
        # Sort by RRF score
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_top_k]
        
        # Build results
        results = []
        for chunk_id, rrf_score in sorted_chunks:
            data = chunk_data.get(chunk_id, {})
            gm = graph_metadata.get(chunk_id, {})
            
            in_vector = chunk_id in vector_ranks
            in_graph = chunk_id in graph_ranks
            
            if in_vector and in_graph:
                method = "hybrid"
            elif in_vector:
                method = "vector"
            else:
                method = "graph"
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                text=data.get("text", ""),
                source_doc=data.get("source_doc", "unknown"),
                position=data.get("position", 0),
                rrf_score=rrf_score,
                vector_score=vector_scores.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
                graph_score=graph_scores.get(chunk_id),
                graph_rank=graph_ranks.get(chunk_id),
                retrieval_method=method,
                hop_distance=gm.get("hop_distance"),
                matched_entities=[e for e in gm.get("matched_entities", []) if e],
            )
            results.append(result)
        
        return results
#% ============================================================================
# IMPROVED QUERY ENTITY EXTRACTOR   
# ============================================================================
class ImprovedQueryEntityExtractor:
    """
    Query Entity Extraction mit GLiNER-Konsistenz.
    
    Nutzt dasselbe GLiNER-Modell wie für Chunk-Entity-Extraction,
    um Konsistenz zwischen Query-Entities und Graph-Entities zu gewährleisten.
    """
    
    def __init__(self, gliner_model=None, spacy_model: str = "en_core_web_sm"):
        self.gliner = gliner_model  # Shared GLiNER instance
        self.nlp = None
        self._load_spacy(spacy_model)
        
        # GLiNER Entity Types (konsistent mit Thesis 2.5)
        self.entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "CONCEPT"
        ]
    
    def _load_spacy(self, model_name: str):
        """Lade SpaCy als Fallback."""
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            logger.info(f"SpaCy loaded for query analysis: {model_name}")
        except Exception as e:
            logger.warning(f"SpaCy not available: {e}")
            self.nlp = None
    
    def extract(self, query: str, confidence_threshold: float = 0.5) -> List[str]:
        """
        Extrahiere Entitäten aus Query.
        
        Präferenz-Reihenfolge:
        1. GLiNER (wenn verfügbar, für Konsistenz mit Chunk-Extraction)
        2. SpaCy NER (Fallback)
        3. Regex (letzter Fallback)
        
        Args:
            query: User Query
            confidence_threshold: Minimum Confidence (0.5 wie in Thesis 2.5)
            
        Returns:
            Liste von Entity-Namen
        """
        # Methode 1: GLiNER (bevorzugt für Konsistenz)
        if self.gliner is not None:
            try:
                entities = self.gliner.predict_entities(
                    query,
                    self.entity_types,
                    threshold=confidence_threshold
                )
                return [ent["text"] for ent in entities]
            except Exception as e:
                logger.warning(f"GLiNER query extraction failed: {e}")
        
        # Methode 2: SpaCy NER (Fallback)
        if self.nlp is not None:
            return self._spacy_extract(query)
        
        # Methode 3: Regex (letzter Fallback)
        return self._fallback_extract(query)
    
    def _spacy_extract(self, query: str) -> List[str]:
        """SpaCy-basierte Extraktion mit GLiNER-Type-Mapping."""
        doc = self.nlp(query)
        entities = []
        
        # Type Mapping: SpaCy -> GLiNER (für Konsistenz)
        type_map = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "DATE": "DATE",
            "EVENT": "EVENT",
        }
        
        for ent in doc.ents:
            if ent.label_ in type_map and len(ent.text) > 2:
                entities.append(ent.text)
        
        # Auch Proper Nouns (als CONCEPT)
        for token in doc:
            if token.pos_ == "PROPN" and token.text not in entities:
                if len(token.text) > 2:
                    entities.append(token.text)
        
        return entities
    
    def _fallback_extract(self, query: str) -> List[str]:
        """Regex-basierte Fallback-Extraktion."""
        import re
        
        entities = []
        
        # Capitalized words/phrases
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query):
            entity = match.group(1)
            if len(entity) > 2:
                entities.append(entity)
        
        # Quoted strings
        for match in re.finditer(r'"([^"]+)"', query):
            entities.append(match.group(1))
        
        return entities

# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """
    Hybrid Retriever mit RRF-Fusion.
    
    Orchestriert:
        1. Vector Retrieval (LanceDB, ANN)
        2. Graph Retrieval (KuzuDB, Entity-basiert)
        3. RRF-Fusion
    
    Performance-Ziele:
        - Vector Retrieval: 20-40ms
        - Graph Retrieval: 10-30ms (KuzuDB)
        - Total: < 100ms
    """
    
    def __init__(
        self,
        hybrid_store,
        embeddings,
        config: RetrievalConfig = None
    ):
        """
        Args:
            hybrid_store: HybridStore Instance
            embeddings: Embedding-Modell (z.B. BatchedOllamaEmbeddings)
            config: RetrievalConfig
        """
        self.store = hybrid_store
        self.embeddings = embeddings
        self.config = config or RetrievalConfig()
        
        # Components
        self.entity_extractor = QueryEntityExtractor(self.config.spacy_model)
        self.rrf_fusion = RRFFusion(
            k=self.config.rrf_k,
            cross_source_boost=self.config.cross_source_boost
        )
        gliner_model = None
        if hasattr(hybrid_store, 'entity_pipeline') and hybrid_store.entity_pipeline is not None:
            gliner_model = hybrid_store.entity_pipeline.ner_extractor.model
        
        self.entity_extractor = ImprovedQueryEntityExtractor(
            gliner_model=gliner_model,
            spacy_model=self.config.spacy_model
        )
        logger.info(f"HybridRetriever initialized: mode={self.config.mode}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Führe Hybrid Retrieval durch.
        
        Args:
            query: User Query
            top_k: Anzahl Ergebnisse (override config)
            
        Returns:
            Tuple (Results, Metrics)
        """
        start_time = time.time()
        top_k = top_k or self.config.final_top_k
        
        # 1. Query Entity Extraction (3-5ms)
        query_entities = self.entity_extractor.extract(query)
        logger.debug(f"Query entities: {query_entities}")
        
        # 2. Vector Retrieval
        vector_start = time.time()
        vector_results = []
        
        if self.config.mode in [RetrievalMode.VECTOR, RetrievalMode.HYBRID]:
            query_embedding = self._embed_query(query)
            vector_results = self.store.vector_search(
                query_embedding,
                top_k=self.config.vector_top_k
            )
        
        vector_time = (time.time() - vector_start) * 1000
        
        # 3. Graph Retrieval
        graph_start = time.time()
        graph_results = []
        
        if self.config.mode in [RetrievalMode.GRAPH, RetrievalMode.HYBRID]:
            if query_entities:
                graph_results = self.store.graph_search(
                    entities=query_entities,
                    max_hops=self.config.max_hops,
                    top_k=self.config.graph_top_k
                )
        
        graph_time = (time.time() - graph_start) * 1000
        
        # 4. RRF Fusion
        fusion_start = time.time()
        
        if self.config.mode == RetrievalMode.VECTOR:
            results = self._vector_only_results(vector_results, top_k)
        elif self.config.mode == RetrievalMode.GRAPH:
            results = self._graph_only_results(graph_results, top_k)
        else:
            results = self.rrf_fusion.fuse(
                vector_results,
                graph_results,
                final_top_k=top_k
            )
        
        fusion_time = (time.time() - fusion_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        # Metrics
        metrics = RetrievalMetrics(
            total_time_ms=total_time,
            vector_time_ms=vector_time,
            graph_time_ms=graph_time,
            fusion_time_ms=fusion_time,
            vector_results=len(vector_results),
            graph_results=len(graph_results),
            final_results=len(results),
            query_entities=query_entities,
        )
        
        logger.info(
            f"Retrieval complete: {len(results)} results, "
            f"{total_time:.1f}ms total "
            f"(vector: {vector_time:.1f}ms, graph: {graph_time:.1f}ms)"
        )
        
        return results, metrics
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Generiere Query-Embedding."""
        if hasattr(self.embeddings, 'embed_query'):
            embedding = self.embeddings.embed_query(query)
        else:
            embedding = self.embeddings.embed_documents([query])[0]
        
        return np.array(embedding, dtype=np.float32)
    
    def _vector_only_results(
        self,
        vector_results: List[Dict],
        top_k: int
    ) -> List[RetrievalResult]:
        """Konvertiere Vector-Results zu RetrievalResult."""
        results = []
        for rank, r in enumerate(vector_results[:top_k], start=1):
            result = RetrievalResult(
                chunk_id=r.get("chunk_id"),
                text=r.get("text", ""),
                source_doc=r.get("source_doc", "unknown"),
                position=r.get("position", 0),
                rrf_score=r.get("relevance_score", 0),
                vector_score=r.get("relevance_score"),
                vector_rank=rank,
                retrieval_method="vector",
            )
            results.append(result)
        return results
    
    def _graph_only_results(
        self,
        graph_results: List[Dict],
        top_k: int
    ) -> List[RetrievalResult]:
        """Konvertiere Graph-Results zu RetrievalResult."""
        results = []
        for rank, r in enumerate(graph_results[:top_k], start=1):
            result = RetrievalResult(
                chunk_id=r.get("chunk_id"),
                text=r.get("text", ""),
                source_doc=r.get("source_doc", "unknown"),
                position=r.get("position", 0),
                rrf_score=r.get("confidence", 0.5),
                graph_score=r.get("confidence"),
                graph_rank=rank,
                retrieval_method="graph",
                hop_distance=r.get("hop", 1),
                matched_entities=[r.get("entity_name", "")],
            )
            results.append(result)
        return results


# ============================================================================
# PRE-GENERATIVE FILTERING
# ============================================================================

class PreGenerativeFilter:
    """
    Pre-Generative Filtering zur Reduktion von Halluzinationen.
    
    Gemäß Masterthesis Abschnitt 3.3:
        1. Relevance Filter: Dynamischer Threshold
        2. Redundancy Filter: Jaccard-Similarity
        3. Contradiction Filter: NLI (optional)
    
    Studien zeigen: 40-60% der Halluzinationen werden durch
    irrelevante/widersprüchliche Retrieval-Ergebnisse verursacht.
    """
    
    def __init__(
        self,
        relevance_threshold_factor: float = 0.6,
        jaccard_threshold: float = 0.8,
        enable_contradiction: bool = True,
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        nli_threshold: float = 0.7
    ):
        self.relevance_threshold_factor = relevance_threshold_factor
        self.jaccard_threshold = jaccard_threshold
        self.enable_contradiction = enable_contradiction
        self.nli_threshold = nli_threshold
        
        # NLI Model (lazy load)
        self.nli_model = None
        self.nli_model_name = nli_model
    
    def filter(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Wende alle Filter an.
        
        Args:
            results: Retrieval-Ergebnisse
            
        Returns:
            Gefilterte Ergebnisse
        """
        if not results:
            return results
        
        # 1. Relevance Filter
        results = self._relevance_filter(results)
        
        # 2. Redundancy Filter
        results = self._redundancy_filter(results)
        
        # 3. Contradiction Filter (optional)
        if self.enable_contradiction:
            results = self._contradiction_filter(results)
        
        return results
    
    def _relevance_filter(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Entferne Chunks mit niedrigem Score.
        
        Threshold = factor * max_score
        """
        if not results:
            return results
        
        max_score = max(r.rrf_score for r in results)
        threshold = self.relevance_threshold_factor * max_score
        
        filtered = [r for r in results if r.rrf_score >= threshold]
        
        removed = len(results) - len(filtered)
        if removed > 0:
            logger.debug(f"Relevance filter removed {removed} low-score chunks")
        
        return filtered
    
    def _redundancy_filter(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Entferne redundante Chunks (hohe lexikalische Überlappung).
        
        Nutzt Jaccard-Similarity auf Wort-Ebene.
        Behalte Chunk mit höherem Score.
        """
        if len(results) <= 1:
            return results
        
        def jaccard_similarity(text1: str, text2: str) -> float:
            """Jaccard-Similarity auf Wort-Ebene."""
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
        
        # Sortiere nach Score (absteigend)
        sorted_results = sorted(results, key=lambda r: r.rrf_score, reverse=True)
        
        # Greedy Deduplikation
        filtered = []
        for candidate in sorted_results:
            is_redundant = False
            
            for accepted in filtered:
                sim = jaccard_similarity(candidate.text, accepted.text)
                if sim >= self.jaccard_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                filtered.append(candidate)
        
        removed = len(results) - len(filtered)
        if removed > 0:
            logger.debug(f"Redundancy filter removed {removed} duplicate chunks")
        
        return filtered
    
    def _contradiction_filter(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Entferne widersprüchliche Chunks (NLI-basiert).
        
        Nutzt Natural Language Inference um Widersprüche zu erkennen.
        """
        if len(results) <= 1:
            return results
        
        # Lazy load NLI model
        if self.nli_model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.nli_model = CrossEncoder(self.nli_model_name)
                logger.info(f"NLI model loaded: {self.nli_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, skipping contradiction filter")
                return results
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {e}")
                return results
        
        # Paarweise NLI-Prüfung
        # Vereinfachte Version: Prüfe nur Top-Chunk gegen andere
        if not results:
            return results
        
        top_chunk = results[0]
        filtered = [top_chunk]
        
        for candidate in results[1:]:
            # NLI: Premise = Top Chunk, Hypothesis = Candidate
            try:
                scores = self.nli_model.predict([
                    (top_chunk.text, candidate.text)
                ])
                
                # CrossEncoder gibt [contradiction, entailment, neutral] zurück
                # oder nur einen Score (modellabhängig)
                if isinstance(scores[0], (list, np.ndarray)):
                    contradiction_score = scores[0][0]
                else:
                    # Einzelner Score: > 0.5 = entailment, < 0.5 = contradiction
                    contradiction_score = 1 - scores[0] if scores[0] < 0.5 else 0
                
                if contradiction_score < self.nli_threshold:
                    filtered.append(candidate)
                else:
                    logger.debug(
                        f"Contradiction filter removed chunk "
                        f"(score={contradiction_score:.2f})"
                    )
                    
            except Exception as e:
                logger.warning(f"NLI failed: {e}")
                filtered.append(candidate)
        
        return filtered


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_hybrid_retriever(
    hybrid_store,
    embeddings,
    mode: str = "hybrid",
    rrf_k: int = 60,
    **kwargs
) -> HybridRetriever:
    """Factory für HybridRetriever."""
    config = RetrievalConfig(
        mode=RetrievalMode(mode),
        rrf_k=rrf_k,
        **kwargs
    )
    return HybridRetriever(hybrid_store, embeddings, config)


# ============================================================================
# CLI / TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("RRF FUSION TEST")
    print("="*70)
    
    # Test RRF Fusion
    fusion = RRFFusion(k=60, cross_source_boost=1.2)
    
    # Simulated results
    vector_results = [
        {"chunk_id": "c1", "text": "Einstein born in Ulm", "relevance_score": 0.95, "source_doc": "a", "position": 0},
        {"chunk_id": "c2", "text": "Theory of relativity", "relevance_score": 0.85, "source_doc": "a", "position": 1},
        {"chunk_id": "c3", "text": "Nobel Prize 1921", "relevance_score": 0.75, "source_doc": "a", "position": 2},
        {"chunk_id": "c4", "text": "Princeton University", "relevance_score": 0.65, "source_doc": "b", "position": 0},
    ]
    
    graph_results = [
        {"chunk_id": "c1", "text": "Einstein born in Ulm", "confidence": 0.9, "hop": 1, "source_doc": "a", "position": 0, "entity_name": "Einstein"},
        {"chunk_id": "c4", "text": "Princeton University", "confidence": 0.8, "hop": 1, "source_doc": "b", "position": 0, "entity_name": "Princeton"},
        {"chunk_id": "c5", "text": "Worked with Oppenheimer", "confidence": 0.7, "hop": 2, "source_doc": "c", "position": 0, "entity_name": "Oppenheimer"},
    ]
    
    results = fusion.fuse(vector_results, graph_results, final_top_k=5)
    
    print("\n--- RRF Fusion Results ---")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.retrieval_method}] {r.chunk_id}: RRF={r.rrf_score:.4f}")
        print(f"   Vector rank: {r.vector_rank}, Graph rank: {r.graph_rank}")
        print(f"   Text: {r.text[:50]}...")
    
    # Test Pre-Generative Filter
    print("\n--- Pre-Generative Filter Test ---")
    pf = PreGenerativeFilter(
        relevance_threshold_factor=0.6,
        jaccard_threshold=0.8,
        enable_contradiction=False  # Skip NLI for test
    )
    
    filtered = pf.filter(results)
    print(f"Before filter: {len(results)}, After: {len(filtered)}")