"""
Unified Agent Pipeline: S_P → S_N → S_V

Version: 4.0.0 - MASTERTHESIS IMPLEMENTATION
Author: Edge-RAG Research Project

===============================================================================
IMPLEMENTATION GEMÄSS MASTERTHESIS ABSCHNITT 3.1
===============================================================================

Die sequentielle Architektur (S_P → S_N → S_V) minimiert Kommunikations-Overhead
und ermöglicht Early-Exit-Optimierungen.

Pipeline-Architektur:
    1. S_P (Planner): Query-Analyse, Klassifikation, Plan-Generierung
    2. S_N (Navigator): Hybrid Retrieval, RRF-Fusion, Pre-Generative Filtering
    3. S_V (Verifier): Pre-Generation Validation, Answer Generation

Kommunikation:
    - Strukturierte JSON-Nachrichten zwischen Agenten
    - Metadaten (Confidence-Scores, Retrieval-Provenance) werden durchgereicht

Early-Exit-Optimierungen:
    - Triviale Single-Hop-Queries können S_V überspringen
    - Cached Results bei wiederholten Queries

Latenz-Budgets:
    - S_P: < 10ms (CPU-only, keine GPU)
    - S_N: < 100ms (Retrieval + Filtering)
    - S_V: < 500ms (Validation) + Generation-Zeit

===============================================================================
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# PIPELINE RESULT
# ============================================================================

@dataclass
class PipelineResult:
    """
    Unified Result der gesamten Agent-Pipeline.
    
    Enthält alle Zwischenergebnisse und Timing-Informationen.
    """
    # Final Output
    answer: str
    confidence: str
    
    # Input
    query: str
    
    # Stage Results
    planner_result: Dict[str, Any]
    navigator_result: Dict[str, Any]
    verifier_result: Dict[str, Any]
    
    # Timing (ms)
    planner_time_ms: float
    navigator_time_ms: float
    verifier_time_ms: float
    total_time_ms: float
    
    # Optimization flags
    early_exit_used: bool = False
    cached_result: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "query": self.query,
            "stages": {
                "planner": self.planner_result,
                "navigator": self.navigator_result,
                "verifier": self.verifier_result
            },
            "timing": {
                "planner_ms": self.planner_time_ms,
                "navigator_ms": self.navigator_time_ms,
                "verifier_ms": self.verifier_time_ms,
                "total_ms": self.total_time_ms
            },
            "optimization": {
                "early_exit": self.early_exit_used,
                "cached": self.cached_result
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# AGENT PIPELINE
# ============================================================================

class AgentPipeline:
    """
    Orchestrator für die Agent-Pipeline S_P → S_N → S_V.
    
    Features:
        - Sequentielle Ausführung der drei Agenten
        - Early-Exit für triviale Queries
        - Query-Caching für Performance
        - Comprehensive Timing und Logging
    
    Design Philosophy:
        Separation of Concerns: Jeder Agent ist für eine klar definierte
        Teilaufgabe optimiert. Die Pipeline koordiniert den Datenfluss
        und sammelt Metriken.
    """
    
    def __init__(
        self,
        planner=None,
        navigator=None,
        verifier=None,
        hybrid_retriever=None,
        graph_store=None,
        enable_early_exit: bool = True,
        enable_caching: bool = True,
        cache_max_size: int = 1000,
        config: Dict[str, Any] = None
    ):
        """
        Args:
            planner: QueryPlanner (S_P) Instanz
            navigator: Navigator (S_N) Instanz
            verifier: Verifier (S_V) Instanz
            hybrid_retriever: HybridRetriever für S_N
            graph_store: KnowledgeGraphStore für S_V
            enable_early_exit: Triviale Queries direkt beantworten
            enable_caching: Query-Result Caching
            cache_max_size: Maximale Cache-Größe
            config: Konfiguration aus settings.yaml
        """
        self.config = config or {}
        
        # Initialize agents
        self.planner = planner
        self.navigator = navigator
        self.verifier = verifier
        
        # Store dependencies
        self.hybrid_retriever = hybrid_retriever
        self.graph_store = graph_store
        
        # Optimization settings
        self.enable_early_exit = enable_early_exit
        self.enable_caching = enable_caching
        
        # Simple LRU cache
        self._cache: Dict[str, PipelineResult] = {}
        self._cache_max_size = cache_max_size
        
        # Statistics
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "early_exits": 0,
            "avg_latency_ms": 0.0
        }
        
        logger.info(
            f"AgentPipeline initialized: "
            f"early_exit={enable_early_exit}, caching={enable_caching}"
        )
    
    def _lazy_init_agents(self):
        """Lazy initialization der Agenten wenn noch nicht vorhanden."""
        if self.planner is None:
            from ..logic_layer.planner import QueryPlanner
            self.planner = QueryPlanner()
            logger.info("Planner (S_P) lazy-initialized")
        
        if self.navigator is None:
            from ..logic_layer.navigator import Navigator, StandaloneNavigator
            if self.hybrid_retriever is not None:
                self.navigator = Navigator(self.hybrid_retriever, self.config)
            else:
                self.navigator = StandaloneNavigator(self.config)
            logger.info("Navigator (S_N) lazy-initialized")
        
        if self.verifier is None:
            from ..logic_layer.verifier import Verifier
            self.verifier = Verifier(
                graph_store=self.graph_store,
                use_mock_generator=True,  # Default to mock for safety
                config=self.config
            )
            logger.info("Verifier (S_V) lazy-initialized")
    
    def process(self, query: str) -> PipelineResult:
        """
        Verarbeite Query durch die gesamte Pipeline.
        
        Pipeline: Query → S_P → S_N → S_V → Answer
        
        Args:
            query: User Query String
        
        Returns:
            PipelineResult mit Answer und Metadaten
        """
        start_time = time.time()
        self._stats["total_queries"] += 1
        
        # Cache check
        if self.enable_caching:
            cache_key = self._get_cache_key(query)
            if cache_key in self._cache:
                self._stats["cache_hits"] += 1
                cached = self._cache[cache_key]
                cached.cached_result = True
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached
        
        # Lazy init agents
        self._lazy_init_agents()
        
        # Stage 1: S_P (Planner)
        planner_start = time.time()
        plan = self.planner.plan(query)
        planner_time = (time.time() - planner_start) * 1000
        
        planner_result = plan.to_dict()
        logger.debug(
            f"S_P completed: {plan.query_type.value} / {plan.strategy.value} "
            f"({planner_time:.2f}ms)"
        )
        
        # Early Exit check
        if self.enable_early_exit and plan.is_trivial:
            self._stats["early_exits"] += 1
            logger.info(f"Early exit for trivial query: {query[:50]}...")
            
            result = PipelineResult(
                answer=plan.cached_answer or "This is a simple factual question that requires database lookup.",
                confidence="high" if plan.confidence > 0.9 else "medium",
                query=query,
                planner_result=planner_result,
                navigator_result={},
                verifier_result={},
                planner_time_ms=planner_time,
                navigator_time_ms=0,
                verifier_time_ms=0,
                total_time_ms=planner_time,
                early_exit_used=True
            )
            
            self._update_cache(query, result)
            return result
        
        # Stage 2: S_N (Navigator)
        navigator_start = time.time()
        
        # Navigator expects RetrievalPlan object
        nav_result = self.navigator.navigate(query, plan)
        navigator_time = (time.time() - navigator_start) * 1000
        
        navigator_result = nav_result.to_dict()
        logger.debug(
            f"S_N completed: {nav_result.final_count} chunks "
            f"({navigator_time:.2f}ms)"
        )
        
        # Stage 3: S_V (Verifier)
        verifier_start = time.time()
        
        # Build hop sequence for verifier
        hop_sequence = [
            {
                "step_number": h.step_number,
                "source_entity": h.source_entity,
                "target_entity": h.target_entity,
                "relation_hint": h.relation_hint
            }
            for h in plan.hop_sequence
        ]
        
        gen_result = self.verifier.verify_and_generate(
            navigator_result=nav_result,
            hop_sequence=hop_sequence,
            query_type=plan.query_type.value
        )
        verifier_time = (time.time() - verifier_start) * 1000
        
        verifier_result = gen_result.to_dict()
        logger.debug(
            f"S_V completed: confidence={gen_result.confidence.value} "
            f"({verifier_time:.2f}ms)"
        )
        
        total_time = (time.time() - start_time) * 1000
        
        # Update average latency
        n = self._stats["total_queries"]
        old_avg = self._stats["avg_latency_ms"]
        self._stats["avg_latency_ms"] = old_avg + (total_time - old_avg) / n
        
        result = PipelineResult(
            answer=gen_result.answer,
            confidence=gen_result.confidence.value,
            query=query,
            planner_result=planner_result,
            navigator_result=navigator_result,
            verifier_result=verifier_result,
            planner_time_ms=planner_time,
            navigator_time_ms=navigator_time,
            verifier_time_ms=verifier_time,
            total_time_ms=total_time
        )
        
        # Update cache
        self._update_cache(query, result)
        
        logger.info(
            f"Pipeline completed: {total_time:.2f}ms "
            f"(S_P: {planner_time:.1f}, S_N: {navigator_time:.1f}, S_V: {verifier_time:.1f})"
        )
        
        return result
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key für Query."""
        # Simple hash-based key
        import hashlib
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    
    def _update_cache(self, query: str, result: PipelineResult):
        """Update cache mit neuen Ergebnis."""
        if not self.enable_caching:
            return
        
        cache_key = self._get_cache_key(query)
        
        # Evict oldest if full (simple FIFO for now)
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Return Pipeline-Statistiken."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(self._stats["total_queries"], 1)
            ),
            "early_exit_rate": (
                self._stats["early_exits"] / max(self._stats["total_queries"], 1)
            )
        }
    
    def clear_cache(self):
        """Clear Query-Cache."""
        self._cache.clear()
        logger.info("Pipeline cache cleared")


# ============================================================================
# BATCH PROCESSING
# ============================================================================

class BatchProcessor:
    """
    Batch-Verarbeitung für mehrere Queries.
    
    Nützlich für:
        - Evaluation auf Datasets (HotpotQA, 2WikiMultiHopQA)
        - Parallelisierte Verarbeitung
        - Progress Tracking
    """
    
    def __init__(
        self,
        pipeline: AgentPipeline,
        batch_size: int = 10,
        show_progress: bool = True
    ):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.show_progress = show_progress
    
    def process_batch(
        self,
        queries: List[str],
        return_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Verarbeite Liste von Queries.
        
        Args:
            queries: Liste von Query-Strings
            return_details: Ob detaillierte Results zurückgegeben werden sollen
        
        Returns:
            Liste von Results (simplified oder full)
        """
        results = []
        total = len(queries)
        
        for i, query in enumerate(queries):
            try:
                result = self.pipeline.process(query)
                
                if return_details:
                    results.append(result.to_dict())
                else:
                    results.append({
                        "query": query,
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "latency_ms": result.total_time_ms
                    })
                
                if self.show_progress and (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{total} queries processed")
                    
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                results.append({
                    "query": query,
                    "error": str(e)
                })
        
        return results
    
    def evaluate(
        self,
        queries: List[str],
        ground_truths: List[str],
        metric_fn=None
    ) -> Dict[str, float]:
        """
        Evaluiere Pipeline auf Dataset mit Ground Truth.
        
        Args:
            queries: Liste von Queries
            ground_truths: Liste von korrekten Antworten
            metric_fn: Custom Metrik-Funktion (default: exact_match)
        
        Returns:
            Dict mit Evaluations-Metriken
        """
        if metric_fn is None:
            metric_fn = self._exact_match
        
        results = self.process_batch(queries)
        
        correct = 0
        total_latency = 0
        
        for i, (result, gt) in enumerate(zip(results, ground_truths)):
            if "error" in result:
                continue
            
            answer = result.get("answer", "")
            if metric_fn(answer, gt):
                correct += 1
            
            total_latency += result.get("latency_ms", 0)
        
        n = len(queries)
        return {
            "accuracy": correct / n if n > 0 else 0,
            "total_queries": n,
            "correct": correct,
            "avg_latency_ms": total_latency / n if n > 0 else 0,
            "pipeline_stats": self.pipeline.get_stats()
        }
    
    @staticmethod
    def _exact_match(prediction: str, ground_truth: str) -> bool:
        """Simple exact match (case-insensitive)."""
        return prediction.lower().strip() == ground_truth.lower().strip()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_pipeline(
    config: Dict[str, Any] = None,
    use_mock: bool = True
) -> AgentPipeline:
    """
    Factory für AgentPipeline.
    
    Args:
        config: Konfiguration aus settings.yaml
        use_mock: True für MockGenerator (Tests ohne GPU)
    
    Returns:
        Konfigurierte AgentPipeline
    """
    config = config or {}
    
    # Import agents
    from ..logic_layer.planner import QueryPlanner, create_planner
    from ..logic_layer.navigator import StandaloneNavigator
    from ..logic_layer.verifier import Verifier
    
    # Create agents
    planner = create_planner(config)
    navigator = StandaloneNavigator(config)
    verifier = Verifier(
        use_mock_generator=use_mock,
        config=config
    )
    
    return AgentPipeline(
        planner=planner,
        navigator=navigator,
        verifier=verifier,
        config=config
    )


def create_full_pipeline(
    hybrid_retriever,
    graph_store,
    config: Dict[str, Any] = None,
    use_mock_generator: bool = False
) -> AgentPipeline:
    """
    Factory für vollständige Pipeline mit echten Stores.
    
    Args:
        hybrid_retriever: HybridRetriever Instanz
        graph_store: KnowledgeGraphStore Instanz
        config: Konfiguration
        use_mock_generator: True für MockGenerator
    
    Returns:
        Vollständig konfigurierte Pipeline
    """
    config = config or {}
    
    from ..logic_layer.planner import create_planner
    from ..logic_layer.navigator import Navigator
    from ..logic_layer.verifier import Verifier
    
    planner = create_planner(config)
    navigator = Navigator(hybrid_retriever, config)
    verifier = Verifier(
        graph_store=graph_store,
        use_mock_generator=use_mock_generator,
        config=config
    )
    
    return AgentPipeline(
        planner=planner,
        navigator=navigator,
        verifier=verifier,
        hybrid_retriever=hybrid_retriever,
        graph_store=graph_store,
        config=config
    )


# ============================================================================
# CLI / TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("AGENT PIPELINE TEST")
    print("="*70)
    
    # Direkte Imports für Test
    from src.logic_layer.planner import QueryPlanner
    from src.logic_layer.navigator import StandaloneNavigator
    from src.logic_layer.verifier import Verifier
    
    # Create pipeline mit Mock-Komponenten
    pipeline = AgentPipeline(
        planner=QueryPlanner(),
        navigator=StandaloneNavigator(),
        verifier=Verifier(use_mock_generator=True),
        enable_early_exit=True,
        enable_caching=True
    )
    
    # Test Queries
    test_queries = [
        "What is the capital of France?",
        "Who founded the company that created the iPhone?",
        "How does Tesla compare to Ford in market capitalization?",
        "When was World War II?",
        "The director of Inception also directed what Batman movie?"
    ]
    
    print("\nProcessing test queries...\n")
    
    for query in test_queries:
        print("-" * 70)
        print(f"Query: {query}")
        print("-" * 70)
        
        result = pipeline.process(query)
        
        print(f"Answer: {result.answer[:100]}...")
        print(f"Confidence: {result.confidence}")
        print(f"Query Type: {result.planner_result.get('query_type', 'unknown')}")
        print(f"Strategy: {result.planner_result.get('strategy', 'unknown')}")
        print(f"Early Exit: {result.early_exit_used}")
        print(f"\nTiming:")
        print(f"  S_P (Planner):   {result.planner_time_ms:7.2f} ms")
        print(f"  S_N (Navigator): {result.navigator_time_ms:7.2f} ms")
        print(f"  S_V (Verifier):  {result.verifier_time_ms:7.2f} ms")
        print(f"  Total:           {result.total_time_ms:7.2f} ms")
        print()
    
    # Test cache
    print("\n" + "="*70)
    print("CACHE TEST")
    print("="*70)
    
    # Process same query twice
    query = "What is the capital of France?"
    
    print(f"\nFirst call: {query}")
    result1 = pipeline.process(query)
    print(f"  Cached: {result1.cached_result}, Time: {result1.total_time_ms:.2f}ms")
    
    print(f"Second call: {query}")
    result2 = pipeline.process(query)
    print(f"  Cached: {result2.cached_result}, Time: {result2.total_time_ms:.2f}ms")
    
    # Stats
    print("\n" + "="*70)
    print("PIPELINE STATISTICS")
    print("="*70)
    
    stats = pipeline.get_stats()
    print(f"\nTotal Queries: {stats['total_queries']}")
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
    print(f"Early Exits: {stats['early_exits']}")
    print(f"Early Exit Rate: {stats['early_exit_rate']:.2%}")
    print(f"Average Latency: {stats['avg_latency_ms']:.2f} ms")
    
    print("\n" + "="*70)
    print("PIPELINE TESTS COMPLETED")
    print("="*70)
