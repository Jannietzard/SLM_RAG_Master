"""
===============================================================================
Agentic Controller - S_P → S_N → S_V Pipeline Orchestrierung
===============================================================================

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"
Artefakt B: Agent-Based Query Processing

===============================================================================
ÜBERBLICK
===============================================================================

Der Agentic Controller orchestriert die drei Agenten gemäß Masterarbeit:

1. S_P (Planner)
   - Query-Klassifikation (Single-Hop, Multi-Hop, Comparison, Temporal)
   - Entity & Bridge Detection
   - Retrieval-Plan Generierung

2. S_N (Navigator) - Implementiert in diesem Modul
   - Hybrid Retrieval Orchestrierung (Vector + Graph)
   - RRF-Fusion der Retrieval-Ergebnisse
   - Pre-Generative Filtering:
     * Relevance Filter (dynamischer Threshold: 0.6 × max_score)
     * Redundancy Filter (Jaccard-Similarity > 0.8)
     * [Optional] Contradiction Filter

3. S_V (Verifier)
   - Pre-Generation Validation
   - Answer Generation mit SLM
   - Self-Correction Loop

Die sequentielle Architektur (S_P → S_N → S_V) minimiert Kommunikations-Overhead
und ermöglicht Early-Exit-Optimierungen.

===============================================================================
ARCHITEKTUR
===============================================================================

    User Query
        │
        ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                    AGENTIC CONTROLLER                              │
    │                                                                    │
    │   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐│
    │   │     S_P     │────────▶│     S_N     │────────▶│     S_V     ││
    │   │   PLANNER   │         │  NAVIGATOR  │         │  VERIFIER   ││
    │   └─────────────┘         └─────────────┘         └─────────────┘│
    │         │                       │                       │         │
    │         │                       │                       │         │
    │    ┌────▼────┐            ┌────▼────┐            ┌────▼────┐    │
    │    │Query    │            │Hybrid   │            │Pre-     │    │
    │    │Analysis │            │Retrieval│            │Validation│    │
    │    │         │            │         │            │         │    │
    │    │Entity   │            │RRF      │            │Generation│    │
    │    │Extract  │            │Fusion   │            │         │    │
    │    │         │            │         │            │Self-    │    │
    │    │Plan Gen │            │Pre-Gen  │            │Correct  │    │
    │    └─────────┘            │Filter   │            └─────────┘    │
    │                           └─────────┘                           │
    │                                                                    │
    └───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            Final Answer + Metadata

===============================================================================
KOMMUNIKATION ZWISCHEN AGENTEN
===============================================================================

Die Agenten kommunizieren über strukturierte Nachrichten (JSON-Schema),
die Zwischenergebnisse und Metadaten (Confidence-Scores, Retrieval-Provenance)
kapseln.

S_P → S_N:
    - RetrievalPlan (Query-Typ, Strategie, Entities, Hop-Sequenz)

S_N → S_V:
    - Filtered Context (nach RRF-Fusion und Pre-Gen Filtering)
    - Retrieval Metadata (Scores, Provenance)

===============================================================================
"""

import logging
import time
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re

# LangGraph (optional)
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph nicht installiert: pip install langgraph")

# Lokale Imports - unterstützt sowohl direkten Aufruf als auch Modul-Import
try:
    # Wenn als Modul importiert (from src.logic_layer.agent import ...)
    from src.logic_layer.planner import (
        Planner, 
        create_planner, 
        RetrievalPlan,
        QueryType,
        RetrievalStrategy
    )
    from src.logic_layer.verifier import (
        Verifier, 
        create_verifier, 
        VerificationResult,
        PreValidationResult
    )
except ModuleNotFoundError:
    # Wenn direkt ausgeführt (python agent.py)
    from planner import (
        Planner, 
        create_planner, 
        RetrievalPlan,
        QueryType,
        RetrievalStrategy
    )
    from verifier import (
        Verifier, 
        create_verifier, 
        VerificationResult,
        PreValidationResult
    )

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    State für den Agentic Controller.
    
    Enthält alle Zwischenergebnisse der Pipeline-Stufen.
    
    Planner Output:
        query: Original User-Query
        retrieval_plan: Vollständiger RetrievalPlan von S_P
        sub_queries: Flache Liste der Sub-Queries
        entities: Extrahierte Entities
        query_type: Klassifizierter Query-Typ
    
    Navigator Output:
        raw_context: Ungefilterter Context aus Retrieval
        context: Gefilterter Context nach Pre-Gen Filtering
        retrieval_scores: RRF-Scores pro Chunk
        retrieval_metadata: Zusätzliche Metadaten
    
    Verifier Output:
        answer: Finale Antwort
        iterations: Anzahl Self-Correction Iterations
        verified_claims: Verifizierte Claims
        violated_claims: Nicht-verifizierte Claims
        all_verified: True wenn alle Claims verifiziert
        pre_validation: Pre-Generation Validation Result
    
    Metadata:
        total_time_ms: Gesamtzeit der Pipeline
        errors: Liste aufgetretener Fehler
        stage_timings: Timing pro Stage
    """
    # Input
    query: str
    
    # Planner Output
    retrieval_plan: Optional[Dict[str, Any]]
    sub_queries: List[str]
    entities: List[str]
    query_type: str
    
    # Navigator Output
    raw_context: List[str]
    context: List[str]
    retrieval_scores: List[float]
    retrieval_metadata: Dict[str, Any]
    
    # Verifier Output
    answer: str
    iterations: int
    verified_claims: List[str]
    violated_claims: List[str]
    all_verified: bool
    pre_validation: Optional[Dict[str, Any]]
    
    # Metadata
    total_time_ms: float
    errors: List[str]
    stage_timings: Dict[str, float]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ControllerConfig:
    """
    Konfiguration für Agentic Controller.
    
    LLM Settings:
        model_name: Ollama-Modell für S_V (z.B. "phi3")
        base_url: Ollama API URL
        temperature: Sampling Temperature
    
    Pipeline Settings:
        max_verification_iterations: Max Self-Correction Loops
        enable_early_exit: Triviale Queries direkt beantworten
        cache_enabled: Caching für wiederholte Queries
    
    Navigator Settings (Pre-Generative Filtering):
        relevance_threshold_factor: Faktor für dynamischen Threshold (0.6 × max)
        redundancy_threshold: Jaccard-Similarity für Deduplizierung (0.8)
        max_context_chunks: Maximale Chunks nach Filtering
    """
    # LLM Settings
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    
    # Pipeline Settings
    max_verification_iterations: int = 3
    enable_early_exit: bool = True
    cache_enabled: bool = False
    
    # Navigator Settings (gemäß Masterarbeit Abschnitt 3.3)
    relevance_threshold_factor: float = 0.6  # 0.6 × max_score
    redundancy_threshold: float = 0.8        # Jaccard > 0.8 = redundant
    max_context_chunks: int = 10             # Max Chunks nach Filtering


@dataclass
class NavigatorResult:
    """
    Ergebnis des Navigator (S_N).
    
    Attributes:
        filtered_context: Context nach Pre-Gen Filtering
        raw_context: Ungefilterter Context
        scores: RRF-Scores pro Chunk
        metadata: Zusätzliche Metadaten (Provenance, etc.)
    """
    filtered_context: List[str] = field(default_factory=list)
    raw_context: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# NAVIGATOR (S_N) IMPLEMENTATION
# =============================================================================

class Navigator:
    """
    S_N: Navigator mit Hybrid Retrieval und Pre-Generative Filtering.
    
    Der Navigator ist die zentrale Orchestrierungskomponente, die den
    Retrieval-Plan des Planners ausführt und hochqualitative Evidenz
    für die Generierung bereitstellt.
    
    Gemäß Masterarbeit Abschnitt 3.3 implementiert der Navigator:
    
    1. HYBRID RETRIEVAL ORCHESTRIERUNG
       - Vector Retrieval (Semantic Search)
       - Graph Retrieval (Relation-basiert)
       - Strategie basierend auf Query-Typ
    
    2. RRF-FUSION
       - Reciprocal Rank Fusion der Retrieval-Ergebnisse
       - Cross-Source Corroboration Boost
    
    3. PRE-GENERATIVE FILTERING
       a) Relevance Filter: Chunks unter dynamischem Threshold verwerfen
       b) Redundancy Filter: Ähnliche Chunks deduplizieren
       c) [Optional] Contradiction Filter
    """
    
    def __init__(self, config: ControllerConfig):
        """
        Initialisiere Navigator.
        
        Args:
            config: ControllerConfig mit Navigator-Settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Retriever wird später gesetzt (HybridRetriever aus retrieval.py)
        self.retriever = None
        self.documents = {}  # doc_id → text mapping
        
        self.logger.info(
            f"Navigator initialisiert: "
            f"relevance_factor={config.relevance_threshold_factor}, "
            f"redundancy_threshold={config.redundancy_threshold}"
        )
    
    def set_retriever(self, retriever, documents: Dict[str, str] = None) -> None:
        """
        Setze den HybridRetriever aus retrieval.py.
        
        Args:
            retriever: HybridRetriever Instance
            documents: Optional dict doc_id → text
        """
        self.retriever = retriever
        if documents:
            self.documents = documents
        self.logger.info("HybridRetriever verbunden")
    
    def navigate(
        self,
        retrieval_plan: RetrievalPlan,
        sub_queries: List[str]
    ) -> NavigatorResult:
        """
        Führe Hybrid Retrieval und Pre-Generative Filtering aus.
        
        Algorithmus:
        1. Führe Retrieval für alle Sub-Queries aus
        2. Fusioniere Ergebnisse mit RRF
        3. Wende Pre-Generative Filter an
        4. Returniere gefilterten Context
        
        Args:
            retrieval_plan: RetrievalPlan von S_P
            sub_queries: Liste der Sub-Queries
            
        Returns:
            NavigatorResult mit gefiltertem Context
        """
        start_time = time.time()
        
        result = NavigatorResult()
        result.metadata["retrieval_plan"] = retrieval_plan.to_dict() if retrieval_plan else None
        
        if self.retriever is None:
            self.logger.warning("Kein Retriever gesetzt!")
            return result
        
        # ─────────────────────────────────────────────────────────────────────
        # STUFE 1: HYBRID RETRIEVAL
        # ─────────────────────────────────────────────────────────────────────
        
        self.logger.info(f"[Navigator] Retrieval für {len(sub_queries)} Sub-Queries")
        
        all_results = []
        retrieval_scores = {}  # text → score mapping für Deduplizierung
        
        for sub_query in sub_queries:
            try:
                # Nutze HybridRetriever.retrieve() - returns (results, metrics) tuple
                results, _metrics = self.retriever.retrieve(sub_query)
                
                for res in results[:10]:  # Top-10 pro Sub-Query
                    text = res.text if hasattr(res, 'text') else str(res)
                    score = res.rrf_score if hasattr(res, 'rrf_score') else (res.score if hasattr(res, 'score') else 1.0)
                    
                    # Track höchsten Score pro Text
                    if text not in retrieval_scores or score > retrieval_scores[text]:
                        retrieval_scores[text] = score
                    
                    all_results.append({
                        "text": text,
                        "score": score,
                        "source": res.source_doc if hasattr(res, 'source_doc') else (res.source if hasattr(res, 'source') else "unknown"),
                        "sub_query": sub_query
                    })
                    
            except Exception as e:
                self.logger.error(f"[Navigator] Retrieval Error: {e}")
                result.metadata["retrieval_errors"] = result.metadata.get("retrieval_errors", []) + [str(e)]
        
        # ─────────────────────────────────────────────────────────────────────
        # STUFE 2: RRF-FUSION
        # ─────────────────────────────────────────────────────────────────────
        
        self.logger.info(f"[Navigator] RRF-Fusion von {len(all_results)} Ergebnissen")
        
        fused_results = self._rrf_fusion(all_results)
        
        result.raw_context = [r["text"] for r in fused_results]
        result.scores = [r["rrf_score"] for r in fused_results]
        
        result.metadata["pre_filter_count"] = len(fused_results)
        result.metadata["fusion_time_ms"] = (time.time() - start_time) * 1000
        
        # ─────────────────────────────────────────────────────────────────────
        # STUFE 3: PRE-GENERATIVE FILTERING
        # ─────────────────────────────────────────────────────────────────────
        
        self.logger.info("[Navigator] Pre-Generative Filtering")
        
        filter_start = time.time()
        
        # Filter 1: Relevance Filter
        relevance_filtered = self._relevance_filter(fused_results)
        result.metadata["after_relevance_filter"] = len(relevance_filtered)
        
        # Filter 2: Redundancy Filter
        redundancy_filtered = self._redundancy_filter(relevance_filtered)
        result.metadata["after_redundancy_filter"] = len(redundancy_filtered)
        
        # Limitiere auf max_context_chunks
        final_results = redundancy_filtered[:self.config.max_context_chunks]
        
        result.filtered_context = [r["text"] for r in final_results]
        result.scores = [r["rrf_score"] for r in final_results]
        
        result.metadata["filter_time_ms"] = (time.time() - filter_start) * 1000
        result.metadata["total_time_ms"] = (time.time() - start_time) * 1000
        
        self.logger.info(
            f"[Navigator] Ergebnis: {len(result.filtered_context)} Chunks "
            f"(von {len(all_results)} raw), "
            f"Zeit: {result.metadata['total_time_ms']:.0f}ms"
        )
        
        return result
    
    def _rrf_fusion(
        self,
        results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) der Retrieval-Ergebnisse.
        
        RRF Score = Σ 1 / (k + rank_i)
        
        wobei k eine Konstante ist (typisch 60) und rank_i der Rang
        in der i-ten Ergebnisliste.
        
        Cross-Source Corroboration: Chunks, die in mehreren Listen
        erscheinen, erhalten einen Boost.
        
        Args:
            results: Liste von Retrieval-Ergebnissen
            k: RRF-Konstante (default 60)
            
        Returns:
            Fusionierte und sortierte Ergebnisse
        """
        # Gruppiere nach Text
        text_groups = {}
        for r in results:
            text = r["text"]
            if text not in text_groups:
                text_groups[text] = {
                    "text": text,
                    "scores": [],
                    "sources": set(),
                    "sub_queries": set()
                }
            text_groups[text]["scores"].append(r["score"])
            text_groups[text]["sources"].add(r["source"])
            text_groups[text]["sub_queries"].add(r["sub_query"])
        
        # Berechne RRF-Score
        # Sortiere zunächst innerhalb jeder Sub-Query-Gruppe nach Score
        sub_query_rankings = {}
        for r in results:
            sq = r["sub_query"]
            if sq not in sub_query_rankings:
                sub_query_rankings[sq] = []
            sub_query_rankings[sq].append(r)
        
        for sq, sq_results in sub_query_rankings.items():
            sq_results.sort(key=lambda x: x["score"], reverse=True)
            for rank, r in enumerate(sq_results):
                text = r["text"]
                if "rrf_contributions" not in text_groups[text]:
                    text_groups[text]["rrf_contributions"] = []
                text_groups[text]["rrf_contributions"].append(1.0 / (k + rank))
        
        # Aggregiere RRF-Scores
        fused = []
        for text, group in text_groups.items():
            rrf_score = sum(group.get("rrf_contributions", [1.0 / k]))
            
            # Cross-Source Corroboration Boost
            # Erscheint in mehreren Sub-Queries oder Sources → Boost
            source_count = len(group["sources"])
            query_count = len(group["sub_queries"])
            corroboration_boost = 1.0 + 0.1 * (source_count - 1) + 0.05 * (query_count - 1)
            
            fused.append({
                "text": text,
                "rrf_score": rrf_score * corroboration_boost,
                "original_scores": group["scores"],
                "source_count": source_count,
                "query_count": query_count
            })
        
        # Sortiere nach RRF-Score
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        return fused
    
    def _relevance_filter(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Relevance Filter: Entferne Low-Confidence Kandidaten.
        
        Gemäß Masterarbeit Abschnitt 3.3:
        "Chunks mit RRF Scores unter einem dynamischen Threshold
        (berechnet als 0.6 × max_score) werden verworfen."
        
        Args:
            results: Fusionierte Ergebnisse
            
        Returns:
            Gefilterte Ergebnisse
        """
        if not results:
            return results
        
        max_score = max(r["rrf_score"] for r in results)
        threshold = self.config.relevance_threshold_factor * max_score
        
        filtered = [r for r in results if r["rrf_score"] >= threshold]
        
        self.logger.debug(
            f"[Navigator] Relevance Filter: "
            f"threshold={threshold:.4f}, "
            f"{len(filtered)}/{len(results)} behalten"
        )
        
        return filtered
    
    def _redundancy_filter(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Redundancy Filter: Dedupliziere ähnliche Chunks.
        
        Gemäß Masterarbeit Abschnitt 3.3:
        "Chunks mit hoher lexikalischer Überlappung (Jaccard-Similarity > 0.8)
        werden dedupliziert, wobei der Chunk mit höherem RRF-Score
        beibehalten wird."
        
        Args:
            results: Relevanz-gefilterte Ergebnisse
            
        Returns:
            Deduplizierte Ergebnisse
        """
        if not results:
            return results
        
        # Results sind bereits nach Score sortiert, daher behalten wir
        # bei Duplikaten automatisch den mit höherem Score
        
        filtered = []
        seen_texts = []  # Für Ähnlichkeitsvergleich
        
        for r in results:
            text = r["text"]
            is_duplicate = False
            
            # Vergleiche mit bereits akzeptierten Chunks
            for seen in seen_texts:
                similarity = self._jaccard_similarity(text, seen)
                if similarity > self.config.redundancy_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(r)
                seen_texts.append(text)
        
        self.logger.debug(
            f"[Navigator] Redundancy Filter: "
            f"{len(filtered)}/{len(results)} unique"
        )
        
        return filtered
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Berechne Jaccard-Similarity zwischen zwei Texten.
        
        Jaccard = |A ∩ B| / |A ∪ B|
        
        wobei A und B die Wort-Mengen sind.
        """
        # Tokenisierung: Einfaches Wort-Splitting
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


# =============================================================================
# AGENTIC CONTROLLER
# =============================================================================

class AgenticController:
    """
    Agentic Controller: S_P → S_N → S_V Pipeline.
    
    Orchestriert die drei Agenten und verwaltet den Pipeline-State.
    
    Verwendung:
        controller = create_controller()
        controller.set_retriever(hybrid_retriever)
        result = controller.run("Who directed Inception?")
        print(result["answer"])
    """
    
    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        planner: Optional[Planner] = None,
        verifier: Optional[Verifier] = None,
    ):
        """
        Initialisiere Agentic Controller.
        
        Args:
            config: ControllerConfig
            planner: Optional vorkonfigurierter Planner
            verifier: Optional vorkonfigurierter Verifier
        """
        self.config = config or ControllerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialisiere Komponenten
        self.planner = planner or create_planner()
        
        self.verifier = verifier or create_verifier(
            model_name=self.config.model_name,
            base_url=self.config.base_url,
            max_iterations=self.config.max_verification_iterations,
        )
        
        self.navigator = Navigator(self.config)
        
        # Build Workflow
        if LANGGRAPH_AVAILABLE:
            self.app = self._build_workflow()
            self.logger.info("AgenticController mit LangGraph initialisiert")
        else:
            self.app = None
            self.logger.info("AgenticController mit Simple Pipeline initialisiert")
    
    def set_retriever(self, retriever, documents: Dict[str, str] = None) -> None:
        """
        Setze den HybridRetriever aus retrieval.py.
        
        Args:
            retriever: HybridRetriever Instance
            documents: Optional dict doc_id → text
        """
        self.navigator.set_retriever(retriever, documents)
        self.logger.info("HybridRetriever mit Navigator verbunden")
    
    def set_graph_store(self, graph_store) -> None:
        """
        Setze KnowledgeGraphStore aus storage.py für Verification.
        
        Args:
            graph_store: KnowledgeGraphStore Instance
        """
        self.verifier.set_graph_store(graph_store)
        self.logger.info("GraphStore für Verification verbunden")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LANGGRAPH WORKFLOW
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_workflow(self):
        """Build LangGraph Workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("navigator", self._navigator_node)
        workflow.add_node("verifier", self._verifier_node)
        
        workflow.add_edge("planner", "navigator")
        workflow.add_edge("navigator", "verifier")
        workflow.add_edge("verifier", END)
        
        workflow.set_entry_point("planner")
        
        return workflow.compile()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE NODES
    # ─────────────────────────────────────────────────────────────────────────
    
    def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """
        S_P (Planner) Node: Query Analysis und Plan-Generierung.
        
        Input: query
        Output: retrieval_plan, sub_queries, entities, query_type
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("[S_P PLANNER] Query Analysis")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # Generiere Retrieval-Plan
            plan = self.planner.plan(state["query"])
            
            # Extrahiere Informationen für State
            sub_queries = plan.sub_queries
            entities = [e.text for e in plan.entities]
            query_type = plan.query_type.value
            
            self.logger.info(f"[S_P] Query-Typ: {query_type}")
            self.logger.info(f"[S_P] Strategie: {plan.strategy.value}")
            self.logger.info(f"[S_P] Entities: {entities}")
            self.logger.info(f"[S_P] Sub-Queries: {len(sub_queries)}")
            for i, sq in enumerate(sub_queries, 1):
                self.logger.info(f"      {i}. {sq}")
            
            elapsed = (time.time() - start_time) * 1000
            
            return {
                "retrieval_plan": plan.to_dict(),
                "sub_queries": sub_queries,
                "entities": entities,
                "query_type": query_type,
                "stage_timings": {"planner_ms": elapsed}
            }
            
        except Exception as e:
            self.logger.error(f"[S_P] Error: {e}")
            elapsed = (time.time() - start_time) * 1000
            return {
                "retrieval_plan": None,
                "sub_queries": [state["query"]],
                "entities": [],
                "query_type": "single_hop",
                "errors": [f"Planner Error: {str(e)}"],
                "stage_timings": {"planner_ms": elapsed}
            }
    
    def _navigator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        S_N (Navigator) Node: Hybrid Retrieval + Pre-Generative Filtering.
        
        Input: retrieval_plan, sub_queries
        Output: raw_context, context, retrieval_scores, retrieval_metadata
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("[S_N NAVIGATOR] Hybrid Retrieval + Filtering")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        if self.navigator.retriever is None:
            self.logger.warning("[S_N] Kein Retriever gesetzt!")
            return {
                "raw_context": [],
                "context": [],
                "retrieval_scores": [],
                "retrieval_metadata": {"error": "No retriever"},
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "navigator_ms": 0
                }
            }
        
        try:
            # Rekonstruiere RetrievalPlan (wenn vorhanden)
            plan_dict = state.get("retrieval_plan")
            if plan_dict:
                # Minimal-Rekonstruktion für Navigator
                plan = RetrievalPlan(
                    original_query=state["query"],
                    query_type=QueryType(plan_dict.get("query_type", "single_hop")),
                    strategy=RetrievalStrategy(plan_dict.get("strategy", "hybrid")),
                    sub_queries=state["sub_queries"],
                )
            else:
                plan = None
            
            # Führe Navigator aus
            nav_result = self.navigator.navigate(
                retrieval_plan=plan,
                sub_queries=state["sub_queries"]
            )
            
            self.logger.info(f"[S_N] Raw Context: {len(nav_result.raw_context)} Chunks")
            self.logger.info(f"[S_N] Filtered Context: {len(nav_result.filtered_context)} Chunks")
            
            elapsed = (time.time() - start_time) * 1000
            
            return {
                "raw_context": nav_result.raw_context,
                "context": nav_result.filtered_context,
                "retrieval_scores": nav_result.scores,
                "retrieval_metadata": nav_result.metadata,
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "navigator_ms": elapsed
                }
            }
            
        except Exception as e:
            self.logger.error(f"[S_N] Error: {e}")
            elapsed = (time.time() - start_time) * 1000
            return {
                "raw_context": [],
                "context": [],
                "retrieval_scores": [],
                "retrieval_metadata": {"error": str(e)},
                "errors": state.get("errors", []) + [f"Navigator Error: {str(e)}"],
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "navigator_ms": elapsed
                }
            }
    
    def _verifier_node(self, state: AgentState) -> Dict[str, Any]:
        """
        S_V (Verifier) Node: Pre-Validation + Generation + Self-Correction.
        
        Input: query, context, entities
        Output: answer, iterations, verified_claims, violated_claims, all_verified
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("[S_V VERIFIER] Pre-Validation + Generation")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # Extrahiere Hop-Sequenz für Pre-Validation
            plan_dict = state.get("retrieval_plan", {})
            hop_sequence = plan_dict.get("hop_sequence") if plan_dict else None
            
            # Führe Verifier aus
            result = self.verifier.generate_and_verify(
                query=state["query"],
                context=state["context"],
                entities=state.get("entities", []),
                hop_sequence=hop_sequence
            )
            
            self.logger.info(f"[S_V] Iterations: {result.iterations}")
            self.logger.info(f"[S_V] All Verified: {result.all_verified}")
            self.logger.info(f"[S_V] Verified Claims: {len(result.verified_claims)}")
            self.logger.info(f"[S_V] Violated Claims: {len(result.violated_claims)}")
            
            elapsed = (time.time() - start_time) * 1000
            
            # Pre-Validation Result für State
            pre_val_dict = None
            if result.pre_validation:
                pre_val_dict = {
                    "status": result.pre_validation.status.value,
                    "entity_path_valid": result.pre_validation.entity_path_valid,
                    "contradictions_count": len(result.pre_validation.contradictions),
                    "validation_time_ms": result.pre_validation.validation_time_ms
                }
            
            return {
                "answer": result.answer,
                "iterations": result.iterations,
                "verified_claims": result.verified_claims,
                "violated_claims": result.violated_claims,
                "all_verified": result.all_verified,
                "pre_validation": pre_val_dict,
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "verifier_ms": elapsed
                }
            }
            
        except Exception as e:
            self.logger.error(f"[S_V] Error: {e}")
            elapsed = (time.time() - start_time) * 1000
            return {
                "answer": f"[Error: {str(e)}]",
                "iterations": 0,
                "verified_claims": [],
                "violated_claims": [],
                "all_verified": False,
                "pre_validation": None,
                "errors": state.get("errors", []) + [f"Verifier Error: {str(e)}"],
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "verifier_ms": elapsed
                }
            }
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIMPLE PIPELINE (Fallback ohne LangGraph)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _run_simple_pipeline(self, query: str) -> AgentState:
        """Fallback: Simple Pipeline ohne LangGraph."""
        state: AgentState = {
            "query": query,
            "retrieval_plan": None,
            "sub_queries": [],
            "entities": [],
            "query_type": "single_hop",
            "raw_context": [],
            "context": [],
            "retrieval_scores": [],
            "retrieval_metadata": {},
            "answer": "",
            "iterations": 0,
            "verified_claims": [],
            "violated_claims": [],
            "all_verified": False,
            "pre_validation": None,
            "total_time_ms": 0,
            "errors": [],
            "stage_timings": {},
        }
        
        # S_P: Planner
        update = self._planner_node(state)
        state.update(update)
        
        # S_N: Navigator
        update = self._navigator_node(state)
        state.update(update)
        
        # S_V: Verifier
        update = self._verifier_node(state)
        state.update(update)
        
        return state
    
    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────────────────────
    
    def run(self, query: str) -> AgentState:
        """
        Führe Agentic Pipeline aus.
        
        Args:
            query: User Query
            
        Returns:
            AgentState mit Answer und vollständiger Metadata
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("AGENTIC CONTROLLER - Pipeline Start")
        self.logger.info("=" * 70)
        self.logger.info(f"Query: {query}")
        
        start_time = time.time()
        
        initial_state: AgentState = {
            "query": query,
            "retrieval_plan": None,
            "sub_queries": [],
            "entities": [],
            "query_type": "single_hop",
            "raw_context": [],
            "context": [],
            "retrieval_scores": [],
            "retrieval_metadata": {},
            "answer": "",
            "iterations": 0,
            "verified_claims": [],
            "violated_claims": [],
            "all_verified": False,
            "pre_validation": None,
            "total_time_ms": 0,
            "errors": [],
            "stage_timings": {},
        }
        
        if self.app is not None:
            final_state = self.app.invoke(initial_state)
        else:
            final_state = self._run_simple_pipeline(query)
        
        total_time = (time.time() - start_time) * 1000
        final_state["total_time_ms"] = total_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info(
            f"Zeit: {total_time:.0f}ms | "
            f"Context: {len(final_state['context'])} | "
            f"Iterations: {final_state['iterations']} | "
            f"Verified: {final_state['all_verified']}"
        )
        self.logger.info("=" * 70)
        
        return final_state
    
    def __call__(self, query: str) -> str:
        """Shortcut: Return nur Answer."""
        return self.run(query)["answer"]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_controller(
    model_name: str = "phi3",
    base_url: str = "http://localhost:11434",
    max_iterations: int = 3,
    relevance_threshold: float = 0.6,
    redundancy_threshold: float = 0.8,
) -> AgenticController:
    """
    Factory-Funktion für AgenticController.
    
    Args:
        model_name: Ollama-Modell für Verifier
        base_url: Ollama API URL
        max_iterations: Max Self-Correction Iterations
        relevance_threshold: Faktor für Relevance Filter (0.6 × max)
        redundancy_threshold: Jaccard-Threshold für Redundancy Filter
        
    Returns:
        Konfigurierte AgenticController-Instanz
    """
    config = ControllerConfig(
        model_name=model_name,
        base_url=base_url,
        max_verification_iterations=max_iterations,
        relevance_threshold_factor=relevance_threshold,
        redundancy_threshold=redundancy_threshold,
    )
    return AgenticController(config)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 70)
    print("AGENTIC CONTROLLER TEST")
    print(f"LangGraph verfügbar: {LANGGRAPH_AVAILABLE}")
    print("=" * 70)
    
    # Erstelle Controller
    controller = create_controller(
        max_iterations=3,
        relevance_threshold=0.6,
        redundancy_threshold=0.8,
    )
    
    # Ohne Retriever: Nur Planner testen
    print("\n--- Planner-Only Test (ohne Retriever) ---")
    
    test_queries = [
        "What is the capital of France?",
        "Who directed the movie that stars Tom Hanks in Forrest Gump?",
        "Is Berlin older than Munich?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Nur Planner
        plan = controller.planner.plan(query)
        print(f"  Type: {plan.query_type.value}")
        print(f"  Strategy: {plan.strategy.value}")
        print(f"  Entities: {[e.text for e in plan.entities]}")
        print(f"  Sub-Queries: {plan.sub_queries}")
    
    print("\n" + "=" * 70)
    print("Hinweis: Vollständiger Pipeline-Test benötigt:")
    print("  1. HybridRetriever (retrieval.py)")
    print("  2. Ollama mit phi3 Modell")
    print("=" * 70)