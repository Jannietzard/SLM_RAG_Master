"""
Ablation Study: Vergleiche Hybrid vs Vector vs Graph Retrieval.

Scientific Purpose:
Diese Skript ermöglicht systematische Evaluierung der
verschiedenen Retrieval-Modi für die Masterthesis.
Mess-Metriken:
- Retrieval Latency (ms)
- Relevance Score Distribution
- Coverage (wie viele Queries liefern Results)

Usage:
    python examples/ablation_study.py
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import yaml
from langchain_community.embeddings import OllamaEmbeddings

from src.ingestion import DocumentIngestionPipeline, load_ingestion_config
from src.storage import HybridStore, StorageConfig
from src.retrieval import HybridRetriever, RetrievalConfig, RetrievalMode


logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Durchführe Ablation Study für Retrieval-Modi.
    
    Experimentales Design:
    - Fixed: Dokumente, Embeddings, Modelle
    - Variable: Retrieval Mode (vector, graph, hybrid)
    - Messung: Latenz, Score Distribution, Coverage
    """

    def __init__(self, config_path: Path = Path("./config/settings.yaml")):
        """
        Initialisiere Ablation Study.

        Args:
            config_path: Pfad zur Config
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.embeddings = None
        self.hybrid_store = None
        self.results = {}

    def _load_config(self) -> dict:
        """Lade Config aus YAML."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_pipeline(self) -> None:
        """Initialisiere Pipeline (shared Components)."""
        # Embeddings mit Batching + Caching
        embedding_config = self.config.get("embeddings", {})
        perf_config = self.config.get("performance", {})
        
        self.embeddings = BatchedOllamaEmbeddings(
            model_name=embedding_config.get("model_name", "nomic-embed-text"),
            base_url=embedding_config.get("base_url", "http://localhost:11434"),
            batch_size=perf_config.get("batch_size", 32),
            cache_path=Path(self.config.get("paths", {}).get("cache", "./cache")) / "embeddings.db",
            device=perf_config.get("device", "cpu"),
        )

        # Hybrid Store
        storage_config = StorageConfig(
            vector_db_path=Path(self.config.get("paths", {}).get("vector_db", "./data/vector_db")),
            graph_db_path=Path(self.config.get("paths", {}).get("graph_db", "./data/knowledge_graph")),
            embedding_dim=embedding_config.get("embedding_dim", 384),
        )

        self.hybrid_store = HybridStore(config=storage_config, embeddings=self.embeddings)
        
        # Lade gespeicherte Stores
        self.hybrid_store.load()

        print("✓ Pipeline initialisiert")

    def run_retrieval_experiment(
        self,
        mode: RetrievalMode,
        queries: List[str],
    ) -> Dict[str, Any]:
        """
        Führe Retrieval-Experiment für einen Modus durch.

        Args:
            mode: RetrievalMode (VECTOR, GRAPH, HYBRID)
            queries: Liste von Test-Queries

        Returns:
            Experimentalresultate mit Metrics
        """
        # Konfiguriere Retriever für diesen Modus
        rag_config = self.config.get("rag", {})
        retrieval_config = RetrievalConfig(
            mode=mode,
            top_k_vector=rag_config.get("top_k_vectors", 5),
            top_k_graph=rag_config.get("top_k_entities", 3),
            vector_weight=1.0 if mode == RetrievalMode.VECTOR else rag_config.get("vector_weight", 0.6),
            graph_weight=1.0 if mode == RetrievalMode.GRAPH else rag_config.get("graph_weight", 0.4),
            similarity_threshold=rag_config.get("similarity_threshold", 0.5),
        )

        retriever = HybridRetriever(
            config=retrieval_config,
            hybrid_store=self.hybrid_store,
            embeddings=self.embeddings,
        )

        # Führe Queries durch
        latencies = []
        scores = []
        coverage = 0

        print(f"\n{'='*60}")
        print(f"RETRIEVAL MODE: {mode.value.upper()}")
        print(f"{'='*60}")

        for query in queries:
            start_time = time.time()
            results = retriever.retrieve(query)
            latency = (time.time() - start_time) * 1000  # ms

            latencies.append(latency)

            if results:
                coverage += 1
                avg_score = sum(r.relevance_score for r in results) / len(results)
                scores.append(avg_score)

                print(f"\nQuery: '{query}'")
                print(f"  Results: {len(results)}")
                print(f"  Avg Score: {avg_score:.4f}")
                print(f"  Latency: {latency:.2f} ms")
            else:
                print(f"\nQuery: '{query}'")
                print(f"  Results: 0 (NO COVERAGE)")

        # Berechne Aggregate Metrics
        return {
            "mode": mode.value,
            "num_queries": len(queries),
            "coverage": coverage / len(queries),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "avg_relevance_score": sum(scores) / len(scores) if scores else 0.0,
            "raw_latencies": latencies,
            "raw_scores": scores,
        }

    def print_summary(self) -> None:
        """Drucke Vergleichszusammenfassung aller Modi."""
        print(f"\n{'='*60}")
        print("ABLATION STUDY SUMMARY")
        print(f"{'='*60}")

        print(f"\n{'Mode':<15} {'Coverage':<12} {'Latency (ms)':<15} {'Relevance':<12}")
        print("-" * 60)

        for mode_name, metrics in self.results.items():
            print(
                f"{mode_name:<15} "
                f"{metrics['coverage']:<12.2%} "
                f"{metrics['avg_latency_ms']:<15.2f} "
                f"{metrics['avg_relevance_score']:<12.4f}"
            )

        # Ablation Insights
        print(f"\n{'='*60}")
        print("ABLATION INSIGHTS")
        print(f"{'='*60}")

        if "hybrid" in self.results and "vector" in self.results:
            hybrid_latency = self.results["hybrid"]["avg_latency_ms"]
            vector_latency = self.results["vector"]["avg_latency_ms"]
            latency_overhead = (hybrid_latency - vector_latency) / vector_latency * 100

            print(f"\nHybrid vs Vector:")
            print(f"  Latency Overhead: {latency_overhead:+.1f}%")
            print(f"  Coverage Delta: {self.results['hybrid']['coverage'] - self.results['vector']['coverage']:+.1%}")

    def run_full_study(self, queries: List[str]) -> None:
        """
        Führe vollständige Ablation Study durch.

        Args:
            queries: Liste von Test-Queries
        """
        print("EDGE-RAG ABLATION STUDY")
        print(f"Queries: {len(queries)}")
        print(f"Test Queries: {queries[:3]}...")

        # Setup
        self.setup_pipeline()

        # Experiment für jeden Modus
        for mode in [RetrievalMode.VECTOR, RetrievalMode.GRAPH, RetrievalMode.HYBRID]:
            try:
                # WICHTIG: Reset Vector Store vor jedem Experiment
                # (Graph bleibt gleich, da strukturelle Relationen unverändert)
                print(f"\n{'='*60}")
                print(f"Resetting Vector Store für sauberes Experiment: {mode.value}")
                print(f"{'='*60}")
                
                self.hybrid_store.reset_vector_store()
                
                # Re-populate Store für diesen Durchlauf
                # (In Production: Würde aus Cache stammen via Embedding Cache)
                
                metrics = self.run_retrieval_experiment(mode, queries)
                self.results[mode.value] = metrics
                
            except Exception as e:
                print(f"✗ Fehler bei {mode.value}: {str(e)}")

        # Summary
        self.print_summary()

        # Speichere Results
        self._save_results()
        
        # Print Embedding Cache Stats
        print(f"\n{'='*60}")
        print("EMBEDDING CACHE STATISTICS")
        print(f"{'='*60}")
        self.embeddings.print_metrics()

    def _save_results(self) -> None:
        """Speichere Ablation Results."""
        import json
        
        output_path = Path("./ablation_results.json")
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results gespeichert: {output_path}")


def main():
    """Main Entry Point für Ablation Study."""
    # Test Queries (für Thesis: erweitere mit Domain-spezifischen Queries)
    test_queries = [
        "What is the main concept of the paper?",
        "How does quantization affect performance?",
        "What are the edge device constraints?",
        "Describe the retrieval architecture.",
        "What is the relationship between vector and graph storage?",
    ]

    study = AblationStudy()
    study.run_full_study(test_queries)


if __name__ == "__main__":
    main()