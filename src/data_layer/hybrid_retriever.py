"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF)

Version: 4.1.0 - MASTER THESIS IMPLEMENTATION
Author: Edge-RAG Research Project

===============================================================================
ARCHITECTURE ROLE
===============================================================================

This module implements the Hybrid Retrieval component of Artefakt A (Data Layer).
It is consumed by:
  - src/logic_layer/navigator.py  (S_N agent, primary consumer)
  - src/evaluations/evaluate_hotpotqa.py  (evaluation harness)
  - src/pipeline/ingestion_pipeline.py  (end-to-end smoke tests)

The retriever orchestrates two parallel search paths:
    1. Vector Retrieval (LanceDB): ANN search, Top-K=20, ~8-12 ms
    2. Graph Retrieval (KuzuDB): entity-based, 1-hop + 2-hop, Top-K=10

===============================================================================
RRF FUSION
===============================================================================

Reciprocal Rank Fusion formula (Cormack et al., 2009):

    RRF(d) = Σ  1 / (k + rank_i(d))  +  BONUS

where
    k    = 60  (empirically optimal constant, cf. Cormack et al. 2009)
    BONUS = cross_source_boost / (k + 1)  when chunk appears in both paths

Advantages of RRF over weighted score fusion:
    - Rank-based, not score-based  → robust to score compression artefacts
    - No score normalisation required
    - Better fusion of heterogeneous retrieval methods

Reference:
    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
    Reciprocal Rank Fusion outperforms Condorcet and individual rank learning
    methods. In Proceedings of the 32nd ACM SIGIR Conference on Research and
    Development in Information Retrieval (SIGIR '09), pp. 758-759. ACM.
    https://doi.org/10.1145/1571941.1572114

===============================================================================
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity-name normalisation (consistent with GLiNERExtractor in entity_extraction.py)
# ---------------------------------------------------------------------------
_STRIP_ARTICLE_TYPES = frozenset({"GPE", "LOCATION", "EVENT"})

_QUERY_LABEL_MAP = {
    "person": "PERSON", "director": "PERSON", "actor": "PERSON",
    "organization": "ORGANIZATION", "company": "ORGANIZATION", "studio": "ORGANIZATION",
    "city": "GPE", "country": "GPE", "state": "GPE",
    "location": "LOCATION", "place": "LOCATION",
    "landmark": "LOCATION", "monument": "LOCATION", "building": "LOCATION",
    "film": "WORK_OF_ART", "movie": "WORK_OF_ART", "album": "WORK_OF_ART",
    "work of art": "WORK_OF_ART", "work_of_art": "WORK_OF_ART",
    "award": "WORK_OF_ART", "prize": "WORK_OF_ART",
    "event": "EVENT",
}


def _normalize_query_entity(text: str, label: str) -> str:
    """
    Normalise query entity names consistently with the ingestion pipeline.

    - Strip leading/trailing whitespace and trailing punctuation.
    - Strip leading articles ('The ', 'A ', 'An ') only for GPE/LOCATION/EVENT
      (e.g. 'The Cold War' -> 'Cold War', but 'The Beatles' stays 'The Beatles').
    """
    _ABBREV_SUFFIXES = (" Inc.", " Ltd.", " Bros.", " Corp.", " Co.", " Jr.", " Sr.", " Dr.")
    name = text.strip().rstrip(',;:')
    if name.endswith('.') and not any(name.endswith(s) for s in _ABBREV_SUFFIXES):
        name = name[:-1]
    canonical_type = _QUERY_LABEL_MAP.get(label.lower(), label.upper())
    if canonical_type in _STRIP_ARTICLE_TYPES:
        for article in ("The ", "A ", "An "):
            if name.startswith(article) and len(name) > len(article) + 1:
                name = name[len(article):]
                break
    return name


# ---------------------------------------------------------------------------
# Module-level GLiNER cache — loaded at most once per process
# ---------------------------------------------------------------------------
_GLINER_MODEL_CACHE = None
_GLINER_CACHE_LOCK = threading.Lock()


def _get_gliner_model(model_name: str = "urchade/gliner_small-v2.1"):
    """
    Load GLiNER once and cache it for the lifetime of the process.

    Uses double-checked locking so the model is loaded at most once even
    under concurrent first-call scenarios.

    Args:
        model_name: HuggingFace model identifier for GLiNER.

    Returns:
        Loaded GLiNER model, or None if loading fails.
    """
    global _GLINER_MODEL_CACHE
    if _GLINER_MODEL_CACHE is None:
        with _GLINER_CACHE_LOCK:
            if _GLINER_MODEL_CACHE is None:  # double-checked locking
                try:
                    from gliner import GLiNER
                    _GLINER_MODEL_CACHE = GLiNER.from_pretrained(model_name)
                    logger.info("GLiNER model loaded and cached: %s", model_name)
                except (ImportError, OSError, RuntimeError) as e:
                    logger.warning(
                        "FALLBACK ACTIVE: GLiNER could not be loaded (%s)"
                        " -> SpaCy/Regex extraction will be used for query entities.",
                        e,
                    )
    return _GLINER_MODEL_CACHE


# ============================================================================
# CONFIGURATION
# ============================================================================

class RetrievalMode(str, Enum):
    """Retrieval modes for ablation studies."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    """Configuration for the Hybrid Retriever."""

    # Retrieval mode
    mode: RetrievalMode = RetrievalMode.HYBRID

    # Vector retrieval
    vector_top_k: int = 10

    # Graph retrieval
    graph_top_k: int = 10
    max_hops: int = 2

    # RRF parameters
    rrf_k: int = 60  # standard RRF constant (Cormack et al. 2009)

    # Fusion
    final_top_k: int = 10
    cross_source_boost: float = 1.2  # extra RRF credit for chunks in both paths

    # Similarity threshold (applied before fusion)
    similarity_threshold: float = 0.3

    # SpaCy model used for query entity extraction fallback
    spacy_model: str = "en_core_web_sm"

    # Query-time NER settings (sourced from settings.yaml entity_extraction.gliner)
    query_ner_confidence: float = 0.15
    query_entity_types: Optional[List[str]] = None  # None -> use ExtractionConfig default

    # GLiNER model name (sourced from settings.yaml entity_extraction.gliner.model_name)
    gliner_model_name: str = "urchade/gliner_small-v2.1"

    # Reserved for future weighted-fusion ablation mode. NOT used by the current
    # RRF implementation.
    vector_weight: float = 0.7
    graph_weight: float = 0.3


@dataclass
class RetrievalResult:
    """A single retrieval result returned by HybridRetriever."""
    chunk_id: str
    text: str
    source_doc: str
    position: int

    # Scores
    rrf_score: float = 0.0
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    graph_score: Optional[float] = None
    graph_rank: Optional[int] = None

    # Metadata
    retrieval_method: str = "hybrid"  # "vector", "graph", or "hybrid"
    hop_distance: Optional[int] = None
    matched_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dictionary."""
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
    """Performance metrics for a single retrieval call."""
    total_time_ms: float
    vector_time_ms: float
    graph_time_ms: float
    fusion_time_ms: float
    vector_results: int
    graph_results: int
    final_results: int
    query_entities: List[str]


# ============================================================================
# RRF FUSION
# ============================================================================

class RRFFusion:
    """
    Reciprocal Rank Fusion of vector and graph result lists.

    Reference:
        Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
        Reciprocal Rank Fusion outperforms Condorcet and individual rank
        learning methods. SIGIR '09, pp. 758-759.
        https://doi.org/10.1145/1571941.1572114
    """

    def __init__(self, k: int = 60, cross_source_boost: float = 1.2) -> None:
        """
        Args:
            k: RRF constant (default 60, empirically optimal).
            cross_source_boost: Additional RRF credit for chunks found in
                both vector and graph paths (interpreted as
                cross_source_boost / (k + 1) additive bonus).
        """
        self.k = k
        self.cross_source_boost = cross_source_boost

    def fuse(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        final_top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Fuse vector and graph result lists using RRF with additive cross-source boost.

        Formula:
            RRF(d) = Σ  1 / (k + rank_i(d))  +  BONUS

        where BONUS = cross_source_boost / (k + 1) when d appears in both lists.

        Expected key names (from storage layer):
            vector_results items: "document_id", "text", "similarity",
                                  "metadata" -> {"source_file": ...}, "position"
            graph_results items:  "chunk_id", "text", "hops",
                                  "source_file", "matched_entity", "position"

        Reference:
            Cormack, Clarke & Buettcher (2009). SIGIR '09, pp. 758-759.
            https://doi.org/10.1145/1571941.1572114

        Args:
            vector_results: Ranked list from LanceDB vector search.
            graph_results:  Ranked list from KuzuDB entity-based graph search.
            final_top_k:    Maximum number of results to return.

        Returns:
            Sorted list of RetrievalResult objects (highest RRF score first).
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, Dict[str, Any]] = {}
        vector_ranks: Dict[str, int] = {}
        graph_ranks: Dict[str, int] = {}
        vector_scores: Dict[str, float] = {}
        graph_scores: Dict[str, float] = {}
        graph_metadata: Dict[str, Dict[str, Any]] = {}

        # ------------------------------------------------------------------
        # Deduplication: identical text fingerprints get one RRF slot only.
        # Problem: Ingestion duplicates or overlapping chunks can accumulate
        # disproportionate RRF credit if the same content appears multiple
        # times.  Fix: deduplicate on the first 80 characters of text before
        # rank assignment; only the earliest (best-ranked) copy is kept.
        # ------------------------------------------------------------------
        seen_fps: set = set()
        deduped_vector: List[Dict[str, Any]] = []
        for r in vector_results:
            fp = r.get("text", "")[:80]
            if fp not in seen_fps:
                seen_fps.add(fp)
                deduped_vector.append(r)
        vector_results = deduped_vector

        # Vector ranks
        # VectorStoreAdapter.vector_search() returns dicts with keys:
        #   "document_id", "text", "similarity", "metadata" -> {"source_file": ...}
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.get("document_id", "")
            rrf_scores[chunk_id] += 1.0 / (self.k + rank)
            vector_ranks[chunk_id] = rank
            vector_scores[chunk_id] = result.get("similarity", 0.0)

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "source_doc": result.get("metadata", {}).get("source_file", "unknown"),
                    "position": result.get("position", 0),
                }

        # Graph ranks
        # HybridStore.graph_search() returns dicts with keys:
        #   "chunk_id", "text", "hops", "source_file", "matched_entity", "position"
        for rank, result in enumerate(graph_results, start=1):
            chunk_id = result.get("chunk_id", "")
            rrf_scores[chunk_id] += 1.0 / (self.k + rank)
            graph_ranks[chunk_id] = rank
            hops = result.get("hops", 1)
            graph_scores[chunk_id] = 1.0 / (hops + 1)  # derive proxy score from hop distance

            graph_metadata[chunk_id] = {
                "hop_distance": hops,
                "matched_entities": [result.get("matched_entity", "")],
            }

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "source_doc": result.get("source_file", "unknown"),
                    "position": result.get("position", 0),
                }

        # Additive cross-source boost for chunks present in both result lists
        in_both = set(vector_ranks.keys()) & set(graph_ranks.keys())
        for chunk_id in in_both:
            bonus = self.cross_source_boost / (self.k + 1)
            rrf_scores[chunk_id] += bonus

        # Sort by RRF score descending
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:final_top_k]

        # Build RetrievalResult objects
        results: List[RetrievalResult] = []
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

            results.append(RetrievalResult(
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
            ))

        return results


# ============================================================================
# IMPROVED QUERY ENTITY EXTRACTOR
# ============================================================================

class ImprovedQueryEntityExtractor:
    """
    Query entity extraction with GLiNER consistency.

    Uses the same GLiNER model as chunk-level entity extraction so that
    query entities and graph entities share the same label space, improving
    graph lookup hit rates.

    Preference order:
        1. GLiNER (preferred — consistent with ingestion-time extraction)
        2. SpaCy NER (fallback when GLiNER unavailable)
        3. Regex (last-resort fallback)
    """

    def __init__(
        self,
        gliner_model: Optional[Any] = None,
        spacy_model: str = "en_core_web_sm",
        entity_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.15,
        gliner_model_name: str = "urchade/gliner_small-v2.1",
    ) -> None:
        """
        Args:
            gliner_model: Pre-loaded GLiNER model.  If None, the module-level
                cache is used (loading on demand).
            spacy_model: SpaCy model name for the fallback extractor.
            entity_types: List of GLiNER entity type labels.  Defaults to the
                standard thesis set sourced from settings.yaml.
            confidence_threshold: Minimum GLiNER confidence for an entity to
                be accepted.
            gliner_model_name: HuggingFace model ID passed to _get_gliner_model()
                when no pre-loaded model is supplied.
        """
        self.gliner = gliner_model
        self.nlp = None
        self.confidence_threshold = confidence_threshold
        self._gliner_model_name = gliner_model_name
        self._load_spacy(spacy_model)

        if self.gliner is None:
            self._load_gliner()

        # Entity types come from settings.yaml (via RetrievalConfig.query_entity_types)
        self.entity_types = entity_types or [
            "person", "organization", "city", "country",
            "state", "location", "film", "movie", "album",
            "work of art", "landmark", "event", "award",
        ]

    def _load_gliner(self) -> None:
        """Use the module-level cached GLiNER — loaded at most once per process."""
        self.gliner = _get_gliner_model(self._gliner_model_name)

    def _load_spacy(self, model_name: str) -> None:
        """Load SpaCy as fallback NER backend."""
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            logger.info("SpaCy loaded for query analysis: %s", model_name)
        except (ImportError, OSError) as e:
            logger.warning("SpaCy not available: %s", e)
            self.nlp = None

    def extract(self, query: str, confidence_threshold: Optional[float] = None) -> List[str]:
        """
        Extract named entities from a query string.

        Args:
            query: User query text.
            confidence_threshold: Override for the instance-level threshold
                (None = use self.confidence_threshold).

        Returns:
            List of normalised entity name strings.
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # Method 1: GLiNER (preferred for ingestion-query consistency)
        if self.gliner is not None:
            try:
                entities = self.gliner.predict_entities(
                    query,
                    self.entity_types,
                    threshold=threshold,
                )
                return [_normalize_query_entity(ent["text"], ent["label"]) for ent in entities]
            except (RuntimeError, ValueError) as e:
                logger.warning("GLiNER query extraction failed: %s", e)

        # Method 2: SpaCy NER (fallback — GLiNER unavailable or failed)
        if self.nlp is not None:
            logger.warning(
                "FALLBACK ACTIVE: GLiNER not available -> SpaCy extraction for query entities."
            )
            return self._spacy_extract(query)

        # Method 3: Regex (last resort — neither GLiNER nor SpaCy available)
        logger.warning(
            "FALLBACK ACTIVE: Neither GLiNER nor SpaCy available -> regex extraction."
            " Graph retrieval will be severely limited!"
        )
        return self._fallback_extract(query)

    def _spacy_extract(self, query: str) -> List[str]:
        """SpaCy-based extraction with GLiNER-compatible type mapping."""
        doc = self.nlp(query)
        entities: List[str] = []

        # Map SpaCy labels to the canonical label set used at ingestion
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

        # Also capture proper nouns not covered by NER
        for token in doc:
            if token.pos_ == "PROPN" and token.text not in entities:
                if len(token.text) > 2:
                    entities.append(token.text)

        return entities

    def _fallback_extract(self, query: str) -> List[str]:
        """Regex-based last-resort extraction."""
        import re

        entities: List[str] = []

        # Capitalised words/phrases (skip sentence-initial question words)
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
    Hybrid Retriever with RRF fusion.

    Orchestrates:
        1. Vector retrieval (LanceDB, ANN)
        2. Graph retrieval (KuzuDB, entity-based)
        3. RRF fusion

    Performance targets (thesis Abschnitt 2.6):
        - Vector retrieval: 20-40 ms
        - Graph retrieval:  10-30 ms
        - Total:            < 100 ms
    """

    def __init__(
        self,
        hybrid_store: Any,
        embeddings: Any,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        """
        Args:
            hybrid_store: HybridStore instance (LanceDB + KuzuDB).
            embeddings: Embedding model (e.g. BatchedOllamaEmbeddings).
            config: RetrievalConfig.  Defaults to RetrievalConfig() if None.
        """
        self.store = hybrid_store
        self.embeddings = embeddings
        self.config = config or RetrievalConfig()

        self.rrf_fusion = RRFFusion(
            k=self.config.rrf_k,
            cross_source_boost=self.config.cross_source_boost,
        )

        # Re-use GLiNER model from HybridStore's entity pipeline when available
        gliner_model = None
        if hasattr(hybrid_store, "entity_pipeline") and hybrid_store.entity_pipeline is not None:
            gliner_model = hybrid_store.entity_pipeline.ner_extractor.model

        self.entity_extractor = ImprovedQueryEntityExtractor(
            gliner_model=gliner_model,
            spacy_model=self.config.spacy_model,
            entity_types=self.config.query_entity_types,
            confidence_threshold=self.config.query_ner_confidence,
            gliner_model_name=self.config.gliner_model_name,
        )
        logger.info("HybridRetriever initialised: mode=%s", self.config.mode)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        entity_hints: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Execute hybrid retrieval for the given query.

        Args:
            query: User query string.
            top_k: Number of results to return (overrides config.final_top_k).
            entity_hints: Optional list of entity strings extracted upstream by
                the Planner (S_P).  When provided, these are used directly for
                graph search instead of re-running GLiNER on the (usually short)
                sub-query string.  GLiNER often fails on isolated 3–5 word
                sub-queries, so reusing the full-query entities from S_P
                significantly improves graph recall.

        Returns:
            Tuple of (results, metrics).
        """
        start_time = time.time()
        top_k = top_k if top_k is not None else self.config.final_top_k

        # 1. Query entity extraction (~3-5 ms)
        # Use caller-supplied hints when available; they come from Planner's
        # full-query analysis and are more reliable than re-running GLiNER on
        # a short sub-query fragment.
        if entity_hints:
            query_entities = entity_hints
            logger.debug("Using entity hints from S_P: %s", query_entities)
        else:
            query_entities = self.entity_extractor.extract(query)
            logger.debug("Query entities (GLiNER): %s", query_entities)

        # 2. Vector retrieval
        vector_start = time.time()
        vector_results: List[Dict[str, Any]] = []

        if self.config.mode in [RetrievalMode.VECTOR, RetrievalMode.HYBRID]:
            try:
                query_embedding = self._embed_query(query)
                vector_results = self.store.vector_search(
                    query_embedding,
                    top_k=self.config.vector_top_k,
                    threshold=self.config.similarity_threshold,
                )
            except Exception as e:
                logger.error("Vector retrieval failed: %s", e)
                vector_results = []

        vector_time = (time.time() - vector_start) * 1000

        # 3. Graph retrieval
        graph_start = time.time()
        graph_results: List[Dict[str, Any]] = []

        if self.config.mode in [RetrievalMode.GRAPH, RetrievalMode.HYBRID]:
            if query_entities:
                try:
                    graph_results = self.store.graph_search(
                        entities=query_entities,
                        max_hops=self.config.max_hops,
                        top_k=self.config.graph_top_k,
                    )
                except (ValueError, RuntimeError, OSError) as e:
                    logger.warning("Graph retrieval failed: %s", e)
                    graph_results = []

        graph_time = (time.time() - graph_start) * 1000

        # 4. Fusion
        fusion_start = time.time()

        if self.config.mode == RetrievalMode.VECTOR:
            results = self._vector_only_results(vector_results, top_k)
        elif self.config.mode == RetrievalMode.GRAPH:
            results = self._graph_only_results(graph_results, top_k)
        else:
            results = self.rrf_fusion.fuse(
                vector_results,
                graph_results,
                final_top_k=top_k,
            )

        fusion_time = (time.time() - fusion_start) * 1000
        total_time = (time.time() - start_time) * 1000

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
            "Retrieval complete: %d results, %.1f ms total"
            " (vector: %.1f ms, graph: %.1f ms)",
            len(results), total_time, vector_time, graph_time,
        )

        return results, metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> "np.ndarray":
        """Generate a query embedding vector."""
        if hasattr(self.embeddings, "embed_query"):
            embedding = self.embeddings.embed_query(query)
        else:
            embedding = self.embeddings.embed_documents([query])[0]
        return np.array(embedding, dtype=np.float32)

    def _vector_only_results(
        self,
        vector_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Convert raw vector search dicts to RetrievalResult list (VECTOR mode).

        Expected keys from VectorStoreAdapter:
            "document_id", "text", "similarity",
            "metadata" -> {"source_file": ...}, "position"
        """
        results: List[RetrievalResult] = []
        for rank, r in enumerate(vector_results[:top_k], start=1):
            results.append(RetrievalResult(
                chunk_id=r.get("document_id", ""),
                text=r.get("text", ""),
                source_doc=r.get("metadata", {}).get("source_file", "unknown"),
                position=r.get("position", 0),
                rrf_score=r.get("similarity", 0.0),
                vector_score=r.get("similarity"),
                vector_rank=rank,
                retrieval_method="vector",
            ))
        return results

    def _graph_only_results(
        self,
        graph_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Convert raw graph search dicts to RetrievalResult list (GRAPH mode).

        Expected keys from HybridStore.graph_search():
            "chunk_id", "text", "hops", "source_file",
            "matched_entity", "position"
        """
        results: List[RetrievalResult] = []
        for rank, r in enumerate(graph_results[:top_k], start=1):
            results.append(RetrievalResult(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", ""),
                source_doc=r.get("source_file", "unknown"),
                position=r.get("position", 0),
                rrf_score=r.get("confidence", 0.5),
                graph_score=r.get("confidence"),
                graph_rank=rank,
                retrieval_method="graph",
                hop_distance=r.get("hops", 1),
                matched_entities=[r.get("matched_entity", "")],
            ))
        return results


# ============================================================================
# PRE-GENERATIVE FILTERING
#
# NOTE: This class logically belongs in the logic layer (src/logic_layer/) and
# should be moved there in a future refactoring pass.  It lives here temporarily
# because it depends on RetrievalResult, which is defined in this module.
# HybridRetriever itself does NOT call PreGenerativeFilter in production; the
# filter is applied by the Navigator agent (S_N) in logic_layer/navigator.py.
# ============================================================================

class PreGenerativeFilter:
    """
    Pre-generative context filter to reduce hallucination risk.

    Implements the three-stage pipeline described in thesis Abschnitt 3.3:
        1. Relevance filter  — dynamic threshold based on max RRF score
        2. Redundancy filter — Jaccard similarity deduplication
           (Jaccard, P. (1901). Distribution de la flore alpine.
            Bulletin de la Societe Vaudoise des Sciences Naturelles, 37, 241-272.)
        3. Contradiction filter — NLI-based (optional, computationally expensive)

    Empirical motivation: 40-60% of LLM hallucinations are caused by irrelevant
    or contradictory retrieval results.
    """

    def __init__(
        self,
        relevance_threshold_factor: float = 0.6,
        jaccard_threshold: float = 0.8,
        enable_contradiction: bool = True,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
        nli_threshold: float = 0.7,
    ) -> None:
        """
        Args:
            relevance_threshold_factor: Relevance threshold = factor * max_score.
            jaccard_threshold: Word-level Jaccard similarity above which two
                chunks are considered redundant.
            enable_contradiction: Whether to run the NLI contradiction filter.
            nli_model_name: HuggingFace model ID for the CrossEncoder NLI model.
            nli_threshold: Contradiction score above which a chunk is removed.
        """
        self.relevance_threshold_factor = relevance_threshold_factor
        self.jaccard_threshold = jaccard_threshold
        self.enable_contradiction = enable_contradiction
        self.nli_threshold = nli_threshold

        # NLI model — lazy loaded on first contradiction filter call
        self._nli_model = None
        self._nli_model_name = nli_model_name

    def filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Apply all enabled filters in sequence.

        Args:
            results: Ranked list of retrieval results.

        Returns:
            Filtered list of retrieval results.
        """
        if not results:
            return results

        results = self._relevance_filter(results)
        results = self._redundancy_filter(results)

        if self.enable_contradiction:
            results = self._contradiction_filter(results)

        return results

    def _relevance_filter(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Remove chunks whose RRF score falls below a dynamic threshold.

        Threshold = relevance_threshold_factor * max(rrf_score).
        """
        if not results:
            return results

        max_score = max(r.rrf_score for r in results)
        threshold = self.relevance_threshold_factor * max_score
        filtered = [r for r in results if r.rrf_score >= threshold]

        removed = len(results) - len(filtered)
        if removed > 0:
            logger.debug("Relevance filter removed %d low-score chunks", removed)

        return filtered

    def _redundancy_filter(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Remove near-duplicate chunks using word-level Jaccard similarity.

        Reference:
            Jaccard, P. (1901). Distribution de la flore alpine dans le
            bassin des Dranses et dans quelques regions voisines.
            Bulletin de la Societe Vaudoise des Sciences Naturelles, 37, 241-272.

        Among a pair of near-duplicates the higher-scoring chunk is kept.
        """
        if len(results) <= 1:
            return results

        def jaccard_similarity(text1: str, text2: str) -> float:
            """Word-level Jaccard similarity."""
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

        sorted_results = sorted(results, key=lambda r: r.rrf_score, reverse=True)

        filtered: List[RetrievalResult] = []
        for candidate in sorted_results:
            is_redundant = any(
                jaccard_similarity(candidate.text, accepted.text) >= self.jaccard_threshold
                for accepted in filtered
            )
            if not is_redundant:
                filtered.append(candidate)

        removed = len(results) - len(filtered)
        if removed > 0:
            logger.debug("Redundancy filter removed %d duplicate chunks", removed)

        return filtered

    def _contradiction_filter(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Remove chunks that contradict the top-ranked chunk (NLI-based).

        Uses a CrossEncoder NLI model (lazy loaded).  The top chunk acts as
        the premise; all other chunks are treated as hypotheses.  Chunks with
        a contradiction score >= nli_threshold are discarded.
        """
        if len(results) <= 1:
            return results

        if self._nli_model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._nli_model = CrossEncoder(self._nli_model_name)
                logger.info("NLI model loaded: %s", self._nli_model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not available — skipping contradiction filter"
                )
                return results
            except (OSError, RuntimeError) as e:
                logger.warning("Failed to load NLI model: %s", e)
                return results

        top_chunk = results[0]
        filtered: List[RetrievalResult] = [top_chunk]

        for candidate in results[1:]:
            try:
                scores = self._nli_model.predict([(top_chunk.text, candidate.text)])

                if isinstance(scores[0], (list, np.ndarray)):
                    contradiction_score = scores[0][0]
                else:
                    contradiction_score = 1 - scores[0] if scores[0] < 0.5 else 0.0

                if contradiction_score < self.nli_threshold:
                    filtered.append(candidate)
                else:
                    logger.debug(
                        "Contradiction filter removed chunk (score=%.2f)",
                        contradiction_score,
                    )
            except (RuntimeError, ValueError) as e:
                logger.warning("NLI inference failed: %s", e)
                filtered.append(candidate)

        return filtered


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_hybrid_retriever(
    hybrid_store: Any,
    embeddings: Any,
    cfg: Optional[Dict[str, Any]] = None,
) -> "HybridRetriever":
    """
    Factory for HybridRetriever.  Reads all parameters from a settings dict
    (typically loaded from config/settings.yaml).

    Args:
        hybrid_store: Initialised HybridStore instance.
        embeddings: Embedding model.
        cfg: Full settings dictionary.  Keys used:
            rag.retrieval_mode, rag.rrf_k, rag.cross_source_boost,
            vector_store.top_k_vectors, vector_store.similarity_threshold,
            graph.top_k_entities, graph.max_hops,
            ingestion.spacy_model,
            entity_extraction.gliner.confidence_threshold,
            entity_extraction.gliner.entity_types,
            entity_extraction.gliner.model_name

    Returns:
        Configured HybridRetriever instance.
    """
    cfg = cfg or {}
    rag_cfg = cfg.get("rag", {})
    vs_cfg = cfg.get("vector_store", {})
    graph_cfg = cfg.get("graph", {})
    ingestion_cfg = cfg.get("ingestion", {})
    gliner_cfg = cfg.get("entity_extraction", {}).get("gliner", {})

    config = RetrievalConfig(
        mode=RetrievalMode(rag_cfg.get("retrieval_mode", "hybrid")),
        vector_top_k=vs_cfg.get("top_k_vectors", 10),
        graph_top_k=graph_cfg.get("top_k_entities", 10),
        max_hops=graph_cfg.get("max_hops", 2),
        rrf_k=rag_cfg.get("rrf_k", 60),
        final_top_k=vs_cfg.get("top_k_vectors", 10),
        cross_source_boost=rag_cfg.get("cross_source_boost", 1.2),
        similarity_threshold=vs_cfg.get("similarity_threshold", 0.3),
        spacy_model=ingestion_cfg.get("spacy_model", "en_core_web_sm"),
        query_ner_confidence=gliner_cfg.get("confidence_threshold", 0.15),
        query_entity_types=gliner_cfg.get("entity_types", None),
        gliner_model_name=gliner_cfg.get("model_name", "urchade/gliner_small-v2.1"),
    )
    return HybridRetriever(hybrid_store, embeddings, config)


# ============================================================================
# SMOKE DEMO / TEST RUNNER
# ============================================================================

def _main() -> None:
    """
    Smoke demo and test runner.

    Performs:
        1. RRF fusion smoke demo with correct storage-layer key names.
        2. PreGenerativeFilter smoke demo.
        3. pytest run targeting RRF / HybridRetriever / PreGenerativeFilter tests.
    """
    import subprocess
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    logger.info("=" * 70)
    logger.info("RRF FUSION SMOKE DEMO")
    logger.info("=" * 70)

    fusion = RRFFusion(k=60, cross_source_boost=1.2)

    # Vector results use storage key "document_id" (from VectorStoreAdapter)
    vector_results = [
        {
            "document_id": "c1",
            "text": "Einstein born in Ulm",
            "similarity": 0.95,
            "metadata": {"source_file": "physics.pdf"},
            "position": 0,
        },
        {
            "document_id": "c2",
            "text": "Theory of relativity",
            "similarity": 0.85,
            "metadata": {"source_file": "physics.pdf"},
            "position": 1,
        },
        {
            "document_id": "c3",
            "text": "Nobel Prize 1921",
            "similarity": 0.75,
            "metadata": {"source_file": "physics.pdf"},
            "position": 2,
        },
        {
            "document_id": "c4",
            "text": "Princeton University",
            "similarity": 0.65,
            "metadata": {"source_file": "biography.pdf"},
            "position": 0,
        },
    ]

    # Graph results use storage keys "chunk_id", "source_file", "hops", "matched_entity"
    graph_results = [
        {
            "chunk_id": "c1",
            "text": "Einstein born in Ulm",
            "hops": 1,
            "source_file": "physics.pdf",
            "matched_entity": "Einstein",
            "position": 0,
        },
        {
            "chunk_id": "c4",
            "text": "Princeton University",
            "hops": 1,
            "source_file": "biography.pdf",
            "matched_entity": "Princeton",
            "position": 0,
        },
        {
            "chunk_id": "c5",
            "text": "Worked with Oppenheimer",
            "hops": 2,
            "source_file": "history.pdf",
            "matched_entity": "Oppenheimer",
            "position": 0,
        },
    ]

    results = fusion.fuse(vector_results, graph_results, final_top_k=5)

    logger.info("--- RRF Fusion Results ---")
    for i, r in enumerate(results, 1):
        logger.info(
            "%d. [%s] %s: RRF=%.4f  vector_rank=%s  graph_rank=%s",
            i, r.retrieval_method, r.chunk_id, r.rrf_score,
            r.vector_rank, r.graph_rank,
        )

    logger.info("=" * 70)
    logger.info("PRE-GENERATIVE FILTER SMOKE DEMO")
    logger.info("=" * 70)

    pf = PreGenerativeFilter(
        relevance_threshold_factor=0.6,
        jaccard_threshold=0.8,
        enable_contradiction=False,  # skip NLI in smoke demo
    )
    filtered = pf.filter(results)
    logger.info(
        "Before filter: %d  After: %d", len(results), len(filtered)
    )

    logger.info("=" * 70)
    logger.info("RUNNING PYTEST")
    logger.info("=" * 70)

    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "src/data_layer/test_data_layer.py",
            "-k", "RRF or HybridRetriever or PreGenerativeFilter",
            "-v", "--tb=short",
        ],
        check=False,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    _main()
