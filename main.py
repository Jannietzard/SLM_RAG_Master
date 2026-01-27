"""
main_retrieval.py - Artifact A Testing: Hybrid Retrieval Pipeline

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

Purpose:
    Test and evaluate the Data Layer components (Artifact A):
    - ingestion.py → DocumentIngestionPipeline
    - embeddings.py → BatchedOllamaEmbeddings
    - storage.py → HybridStore (Vector + Graph)
    - retrieval.py → HybridRetriever

This script tests retrieval ONLY without LLM generation.
Use main_agentic.py for the full Agentic RAG pipeline (Artifact B).

Usage:
    python main_retrieval.py                    # Run full pipeline
    python main_retrieval.py --ingest-only      # Only ingest documents
    python main_retrieval.py --query "..."      # Single query
    python main_retrieval.py --interactive      # Interactive mode
    python main_retrieval.py --benchmark        # Run benchmark queries
"""

import logging
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import yaml

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configure logging with UTF-8 support (Windows compatible).
    
    Args:
        level: Logging level
        log_file: Optional path for file logging
        
    Returns:
        Configured logger instance
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    handlers = []
    
    # Console handler with UTF-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    
    # Windows UTF-8 fix
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: Path = Path("./config/settings.yaml")) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to settings.yaml
        
    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        logging.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # Merge with defaults
    default = get_default_config()
    for key, value in default.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "embeddings": {
            "model_name": "nomic-embed-text",
            "base_url": "http://localhost:11434",
            "embedding_dim": 768,
        },
        "chunking": {
            "mode": "standard",
            "chunk_size": 512,
            "chunk_overlap": 128,
        },
        "vector_store": {
            "db_path": "./data/vector_db",
            "similarity_threshold": 0.3,
            "distance_metric": "cosine",
            "normalize_embeddings": True,
            "top_k_vectors": 10,
        },
        "graph": {
            "graph_path": "./data/knowledge_graph",
            "max_hops": 2,
            "top_k_entities": 5,
        },
        "rag": {
            "retrieval_mode": "hybrid",
            "vector_weight": 0.7,
            "graph_weight": 0.3,
        },
        "paths": {
            "documents": "./data/documents",
            "vector_db": "./data/vector_db",
            "graph_db": "./data/knowledge_graph",
            "cache": "./cache",
            "logs": "./logs",
        },
        "performance": {
            "batch_size": 32,
            "device": "cpu",
            "cache_embeddings": True,
        },
    }


# ============================================================================
# IMPORTS - DATA LAYER MODULES (ARTIFACT A)
# ============================================================================

def import_modules():
    """
    Import Data Layer modules with error handling.
    
    Returns:
        Tuple of (modules_dict, success_bool)
    """
    modules = {}
    
    try:
        from src.data_layer.storage import HybridStore, StorageConfig
        modules['HybridStore'] = HybridStore
        modules['StorageConfig'] = StorageConfig
    except ImportError as e:
        print(f"[ERROR] Cannot import storage: {e}")
        return modules, False
    
    try:
        from src.data_layer.embeddings import BatchedOllamaEmbeddings
        modules['BatchedOllamaEmbeddings'] = BatchedOllamaEmbeddings
    except ImportError as e:
        print(f"[ERROR] Cannot import embeddings: {e}")
        return modules, False
    
    try:
        from src.data_layer.ingestion import DocumentIngestionPipeline, load_ingestion_config
        modules['DocumentIngestionPipeline'] = DocumentIngestionPipeline
        modules['load_ingestion_config'] = load_ingestion_config
    except ImportError as e:
        print(f"[ERROR] Cannot import ingestion: {e}")
        return modules, False
    
    try:
        from data_layer.hybrid_retriever import HybridRetriever, RetrievalConfig, RetrievalMode
        modules['HybridRetriever'] = HybridRetriever
        modules['RetrievalConfig'] = RetrievalConfig
        modules['RetrievalMode'] = RetrievalMode
    except ImportError as e:
        print(f"[ERROR] Cannot import retrieval: {e}")
        return modules, False
    
    return modules, True


# ============================================================================
# RETRIEVAL PIPELINE (ARTIFACT A)
# ============================================================================

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation."""
    query: str
    num_results: int
    avg_score: float
    max_score: float
    min_score: float
    retrieval_time_ms: float
    retrieval_mode: str


class RetrievalPipeline:
    """
    Artifact A: Hybrid Retrieval Pipeline.
    
    Components:
    - Embeddings (BatchedOllamaEmbeddings)
    - Ingestion (DocumentIngestionPipeline)
    - Storage (HybridStore: Vector + Graph)
    - Retrieval (HybridRetriever)
    
    NO LLM generation - retrieval only.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Components
        self.embeddings = None
        self.hybrid_store = None
        self.retriever = None
        self.ingestion_pipeline = None
        
        # Import modules
        self.modules, success = import_modules()
        if not success:
            raise ImportError("Failed to import required modules")
        
        self.logger.info("RetrievalPipeline initialized")
    
    def setup(self) -> None:
        """Initialize all components."""
        self.logger.info("=" * 70)
        self.logger.info("ARTIFACT A: RETRIEVAL PIPELINE SETUP")
        self.logger.info("=" * 70)
        
        self._init_embeddings()
        self._init_storage()
        self._init_retriever()
        
        self.logger.info("=" * 70)
        self.logger.info("[OK] RETRIEVAL PIPELINE READY")
        self.logger.info("=" * 70)
    
    def _init_embeddings(self) -> None:
        """Initialize embedding model."""
        self.logger.info("Initializing Embeddings...")
        
        embedding_config = self.config.get("embeddings", {})
        perf_config = self.config.get("performance", {})
        cache_path = Path(self.config["paths"].get("cache", "./cache")) / "embeddings.db"
        
        BatchedOllamaEmbeddings = self.modules['BatchedOllamaEmbeddings']
        
        self.embeddings = BatchedOllamaEmbeddings(
            model_name=embedding_config.get("model_name", "nomic-embed-text"),
            base_url=embedding_config.get("base_url", "http://localhost:11434"),
            batch_size=perf_config.get("batch_size", 32),
            cache_path=cache_path,
            device=perf_config.get("device", "cpu"),
        )
        
        self.logger.info(f"  Model: {embedding_config.get('model_name')}")
        self.logger.info(f"  Cache: {cache_path}")
        self.logger.info("[OK] Embeddings initialized")
    
    def _init_storage(self) -> None:
        """Initialize hybrid storage (Vector + Graph)."""
        self.logger.info("Initializing Hybrid Storage...")
        
        embedding_config = self.config.get("embeddings", {})
        vector_config = self.config.get("vector_store", {})
        
        StorageConfig = self.modules['StorageConfig']
        HybridStore = self.modules['HybridStore']
        
        storage_config = StorageConfig(
            vector_db_path=Path(self.config["paths"].get("vector_db", "./data/vector_db")),
            graph_db_path=Path(self.config["paths"].get("graph_db", "./data/knowledge_graph")),
            embedding_dim=embedding_config.get("embedding_dim", 768),
            similarity_threshold=vector_config.get("similarity_threshold", 0.3),
            normalize_embeddings=vector_config.get("normalize_embeddings", True),
            distance_metric=vector_config.get("distance_metric", "cosine"),
        )
        
        self.hybrid_store = HybridStore(config=storage_config, embeddings=self.embeddings)
        
        # Try to load existing data
        self._try_load_existing_store()
        
        self.logger.info(f"  Vector DB: {storage_config.vector_db_path}")
        self.logger.info(f"  Graph DB: {storage_config.graph_db_path}")
        self.logger.info(f"  Distance Metric: {storage_config.distance_metric}")
        self.logger.info("[OK] Hybrid Storage initialized")
    
    def _try_load_existing_store(self) -> None:
        """Try to load existing vector store."""
        try:
            vector_path = Path(self.config["paths"].get("vector_db", "./data/vector_db"))
            table_path = vector_path / "documents.lance"
            
            if table_path.exists():
                if hasattr(self.hybrid_store.vector_store, 'db'):
                    self.hybrid_store.vector_store.table = \
                        self.hybrid_store.vector_store.db.open_table("documents")
                    doc_count = len(self.hybrid_store.vector_store.table)
                    self.logger.info(f"  Loaded existing store: {doc_count} documents")
        except Exception as e:
            self.logger.debug(f"No existing store to load: {e}")
    
    def _init_retriever(self) -> None:
        """Initialize hybrid retriever."""
        self.logger.info("Initializing Retriever...")
        
        rag_config = self.config.get("rag", {})
        vector_config = self.config.get("vector_store", {})
        graph_config = self.config.get("graph", {})
        
        RetrievalConfig = self.modules['RetrievalConfig']
        RetrievalMode = self.modules['RetrievalMode']
        HybridRetriever = self.modules['HybridRetriever']
        
        retrieval_config = RetrievalConfig(
            mode=RetrievalMode(rag_config.get("retrieval_mode", "hybrid")),
            top_k_vector=vector_config.get("top_k_vectors", 10),
            top_k_graph=graph_config.get("top_k_entities", 5),
            vector_weight=rag_config.get("vector_weight", 0.7),
            graph_weight=rag_config.get("graph_weight", 0.3),
            similarity_threshold=vector_config.get("similarity_threshold", 0.3),
        )
        
        self.retriever = HybridRetriever(
            config=retrieval_config,
            hybrid_store=self.hybrid_store,
            embeddings=self.embeddings,
        )
        
        self.logger.info(f"  Mode: {rag_config.get('retrieval_mode', 'hybrid')}")
        self.logger.info(f"  Vector Weight: {rag_config.get('vector_weight', 0.7)}")
        self.logger.info(f"  Graph Weight: {rag_config.get('graph_weight', 0.3)}")
        self.logger.info("[OK] Retriever initialized")
    
    def ingest_documents(self, documents_path: Optional[Path] = None) -> int:
        """
        Run document ingestion pipeline.
        
        Args:
            documents_path: Path to documents folder
            
        Returns:
            Number of chunks processed
        """
        self.logger.info("=" * 70)
        self.logger.info("DOCUMENT INGESTION")
        self.logger.info("=" * 70)
        
        if documents_path is None:
            documents_path = Path(self.config["paths"].get("documents", "./data/documents"))
        
        if not documents_path.exists():
            self.logger.error(f"Documents path not found: {documents_path}")
            return 0
        
        # Load ingestion config
        load_ingestion_config = self.modules['load_ingestion_config']
        DocumentIngestionPipeline = self.modules['DocumentIngestionPipeline']
        
        chunking_config = load_ingestion_config(Path("./config/settings.yaml"))
        
        # Create ingestion pipeline
        self.ingestion_pipeline = DocumentIngestionPipeline(
            documents_path=documents_path,
            chunking_config=chunking_config,
        )
        
        # Process documents
        start_time = time.time()
        chunked_docs = self.ingestion_pipeline.process_documents()
        
        if not chunked_docs:
            self.logger.warning("No documents found to ingest")
            return 0
        
        # Add to storage
        self.hybrid_store.add_documents(chunked_docs)
        self.hybrid_store.save()
        
        elapsed = time.time() - start_time
        
        self.logger.info(f"[OK] Ingested {len(chunked_docs)} chunks in {elapsed:.2f}s")
        
        return len(chunked_docs)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Perform hybrid retrieval.
        
        Args:
            query: Search query
            top_k: Number of results (optional)
            
        Returns:
            List of RetrievalResult objects
        """
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Call setup() first.")
        
        start_time = time.time()
        results = self.retriever.retrieve(query)
        elapsed_ms = (time.time() - start_time) * 1000
        
        if top_k:
            results = results[:top_k]
        
        self.logger.debug(f"Retrieved {len(results)} results in {elapsed_ms:.1f}ms")
        
        return results
    
    def retrieve_with_metrics(self, query: str) -> tuple:
        """
        Retrieve with detailed metrics.
        
        Args:
            query: Search query
            
        Returns:
            Tuple of (results, metrics)
        """
        start_time = time.time()
        results = self.retrieve(query)
        elapsed_ms = (time.time() - start_time) * 1000
        
        if results:
            scores = [r.relevance_score for r in results]
            metrics = RetrievalMetrics(
                query=query,
                num_results=len(results),
                avg_score=sum(scores) / len(scores),
                max_score=max(scores),
                min_score=min(scores),
                retrieval_time_ms=elapsed_ms,
                retrieval_mode=self.config.get("rag", {}).get("retrieval_mode", "hybrid"),
            )
        else:
            metrics = RetrievalMetrics(
                query=query,
                num_results=0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                retrieval_time_ms=elapsed_ms,
                retrieval_mode=self.config.get("rag", {}).get("retrieval_mode", "hybrid"),
            )
        
        return results, metrics
    
    def print_results(self, results: List[Any], max_display: int = 5) -> None:
        """Pretty print retrieval results."""
        if not results:
            print("\n[!] No results found.\n")
            return
        
        print("\n" + "=" * 70)
        print(f"RETRIEVAL RESULTS ({len(results)} total)")
        print("=" * 70)
        
        for i, result in enumerate(results[:max_display], 1):
            print(f"\n--- Result {i} ---")
            print(f"Score: {result.relevance_score:.4f}")
            print(f"Method: {result.retrieval_method}")
            
            # Truncate text
            text = result.text[:300] + "..." if len(result.text) > 300 else result.text
            print(f"Text: {text}")
        
        if len(results) > max_display:
            print(f"\n... and {len(results) - max_display} more results")
        
        print("=" * 70 + "\n")
    
    def print_embedding_metrics(self) -> None:
        """Print embedding performance metrics."""
        if self.embeddings and hasattr(self.embeddings, 'print_metrics'):
            self.embeddings.print_metrics()


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def run_interactive(pipeline: RetrievalPipeline) -> None:
    """Run interactive query mode."""
    print("\n" + "=" * 70)
    print("ARTIFACT A: INTERACTIVE RETRIEVAL MODE")
    print("=" * 70)
    print("Commands:")
    print("  [query]     - Search for documents")
    print("  :metrics    - Show last query metrics")
    print("  :stats      - Show embedding stats")
    print("  :quit       - Exit")
    print("=" * 70 + "\n")
    
    last_metrics = None
    
    while True:
        try:
            query = input("Query> ").strip()
            
            if not query:
                continue
            
            if query.lower() in [':quit', ':exit', ':q']:
                print("Exiting...")
                break
            
            if query == ':metrics' and last_metrics:
                print(f"\nLast Query Metrics:")
                print(f"  Results: {last_metrics.num_results}")
                print(f"  Avg Score: {last_metrics.avg_score:.4f}")
                print(f"  Max Score: {last_metrics.max_score:.4f}")
                print(f"  Time: {last_metrics.retrieval_time_ms:.1f}ms")
                continue
            
            if query == ':stats':
                pipeline.print_embedding_metrics()
                continue
            
            # Perform retrieval
            results, last_metrics = pipeline.retrieve_with_metrics(query)
            pipeline.print_results(results)
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


# ============================================================================
# BENCHMARK MODE
# ============================================================================

BENCHMARK_QUERIES = [
    "What is financial sentiment analysis?",
    "How does knowledge management work?",
    "What is the model structure of MTMEC?",
    "Explain neural network architectures",
    "What are embedding models?",
    "How does retrieval augmented generation work?",
    "What is hybrid search?",
    "Explain vector databases",
]


def run_benchmark(pipeline: RetrievalPipeline) -> None:
    """Run benchmark queries."""
    print("\n" + "=" * 70)
    print("ARTIFACT A: RETRIEVAL BENCHMARK")
    print("=" * 70 + "\n")
    
    all_metrics = []
    
    for query in BENCHMARK_QUERIES:
        results, metrics = pipeline.retrieve_with_metrics(query)
        all_metrics.append(metrics)
        
        print(f"Query: {query[:50]}...")
        print(f"  Results: {metrics.num_results} | "
              f"Avg: {metrics.avg_score:.3f} | "
              f"Max: {metrics.max_score:.3f} | "
              f"Time: {metrics.retrieval_time_ms:.1f}ms")
    
    # Summary
    print("\n" + "-" * 70)
    print("BENCHMARK SUMMARY")
    print("-" * 70)
    
    total_results = sum(m.num_results for m in all_metrics)
    queries_with_results = sum(1 for m in all_metrics if m.num_results > 0)
    avg_time = sum(m.retrieval_time_ms for m in all_metrics) / len(all_metrics)
    
    print(f"  Queries: {len(BENCHMARK_QUERIES)}")
    print(f"  Queries with results: {queries_with_results}/{len(BENCHMARK_QUERIES)}")
    print(f"  Total results: {total_results}")
    print(f"  Avg retrieval time: {avg_time:.1f}ms")
    
    if queries_with_results > 0:
        avg_scores = [m.avg_score for m in all_metrics if m.num_results > 0]
        print(f"  Avg relevance score: {sum(avg_scores) / len(avg_scores):.4f}")
    
    print("=" * 70 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Artifact A: Hybrid Retrieval Pipeline Testing"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./config/settings.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only run document ingestion"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to execute"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark queries"
    )
    
    parser.add_argument(
        "--documents",
        type=Path,
        help="Path to documents folder for ingestion"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_file = Path("./logs/retrieval_pipeline.log")
    logger = setup_logging(level=args.log_level, log_file=log_file)
    
    logger.info("=" * 70)
    logger.info("ARTIFACT A: HYBRID RETRIEVAL PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize pipeline
        pipeline = RetrievalPipeline(config, logger)
        pipeline.setup()
        
        # Ingest documents if needed or requested
        if args.ingest_only:
            doc_path = args.documents or Path(config["paths"].get("documents"))
            num_chunks = pipeline.ingest_documents(doc_path)
            logger.info(f"Ingestion complete: {num_chunks} chunks")
            return 0
        
        # Check if we have documents
        try:
            vector_path = Path(config["paths"].get("vector_db", "./data/vector_db"))
            table_path = vector_path / "documents.lance"
            
            if not table_path.exists():
                logger.warning("No documents indexed. Running ingestion...")
                doc_path = args.documents or Path(config["paths"].get("documents"))
                pipeline.ingest_documents(doc_path)
        except Exception as e:
            logger.debug(f"Could not check vector store: {e}")
        
        # Execute based on mode
        if args.query:
            # Single query mode
            results, metrics = pipeline.retrieve_with_metrics(args.query)
            pipeline.print_results(results)
            logger.info(f"Query completed: {metrics.num_results} results in {metrics.retrieval_time_ms:.1f}ms")
        
        elif args.interactive:
            # Interactive mode
            run_interactive(pipeline)
        
        elif args.benchmark:
            # Benchmark mode
            run_benchmark(pipeline)
        
        else:
            # Default: run example query
            example_query = "What is knowledge management?"
            logger.info(f"Running example query: {example_query}")
            
            results, metrics = pipeline.retrieve_with_metrics(example_query)
            pipeline.print_results(results)
            
            # Print embedding metrics
            pipeline.print_embedding_metrics()
        
        logger.info("=" * 70)
        logger.info("ARTIFACT A: PIPELINE COMPLETED")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())