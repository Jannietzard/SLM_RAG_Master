"""
main_agentic.py - Artifact B Testing: Full Agentic RAG Pipeline

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

Purpose:
    Test and evaluate the complete Agentic RAG system (Artifact A + B):
    
    Data Layer (Artifact A):
    - ingestion.py → DocumentIngestionPipeline
    - embeddings.py → BatchedOllamaEmbeddings
    - storage.py → HybridStore (Vector + Graph)
    - retrieval.py → HybridRetriever
    
    Logic Layer (Artifact B):
    - planner.py → Query Decomposition
    - verifier.py → Answer Generation + Verification
    - agent.py → AgenticController (LangGraph DAG)

Pipeline Flow:
    Query → [Planner] → [Navigator (HybridRetriever)] → [Verifier] → Answer
                ↓              ↓                           ↓
         Sub-queries     Context Docs              Verified Answer

Usage:
    python main_agentic.py                          # Run demo query
    python main_agentic.py --query "..."            # Single query
    python main_agentic.py --interactive            # Interactive chat
    python main_agentic.py --ingest                 # Ingest documents first
    python main_agentic.py --benchmark              # Run benchmark
    python main_agentic.py --compare                # Compare with retrieval-only
"""

import logging
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

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
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    handlers = []
    
    # Console handler with UTF-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    
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
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
    
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: Path = Path("./config/settings.yaml")) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        logging.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
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
        "llm": {
            "model_name": "phi3",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 300,
        },
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
        "agent": {
            "max_verification_iterations": 3,
            "enable_verification": True,
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
# MODULE IMPORTS
# ============================================================================

def import_data_layer():
    """Import Data Layer modules (Artifact A)."""
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
        print(f"[WARNING] Ingestion module not available: {e}")
        # Ingestion ist optional für --interactive Mode
        modules['DocumentIngestionPipeline'] = None
        modules['load_ingestion_config'] = None
    
    try:
        from data_layer.hybrid_retriever import HybridRetriever, RetrievalConfig, RetrievalMode
        modules['HybridRetriever'] = HybridRetriever
        modules['RetrievalConfig'] = RetrievalConfig
        modules['RetrievalMode'] = RetrievalMode
    except ImportError as e:
        print(f"[ERROR] Cannot import retrieval: {e}")
        return modules, False
    
    return modules, True


def import_logic_layer():
    """Import Logic Layer modules (Artifact B)."""
    modules = {}
    
    try:
        from src.logic_layer.planner import Planner, create_planner, PlannerConfig
        modules['Planner'] = Planner
        modules['create_planner'] = create_planner
        modules['PlannerConfig'] = PlannerConfig
    except ImportError as e:
        print(f"[WARNING] Cannot import planner: {e}")
        return modules, False
    
    try:
        from src.logic_layer.verifier import Verifier, create_verifier, VerifierConfig, VerificationResult
        modules['Verifier'] = Verifier
        modules['create_verifier'] = create_verifier
        modules['VerifierConfig'] = VerifierConfig
        modules['VerificationResult'] = VerificationResult
    except ImportError as e:
        print(f"[WARNING] Cannot import verifier: {e}")
        return modules, False
    
    try:
        from src.logic_layer.Agent import AgenticController, create_controller, ControllerConfig
        modules['AgenticController'] = AgenticController
        modules['create_controller'] = create_controller
        modules['ControllerConfig'] = ControllerConfig
    except ImportError as e:
        print(f"[WARNING] Cannot import agent: {e}")
        return modules, False
    
    return modules, True


# ============================================================================
# RESULT DATA CLASSES
# ============================================================================

@dataclass
class AgenticResult:
    """Result from agentic pipeline."""
    query: str
    answer: str
    sub_queries: List[str] = field(default_factory=list)
    context_docs: List[str] = field(default_factory=list)
    iterations: int = 0
    verified_claims: List[str] = field(default_factory=list)
    violated_claims: List[str] = field(default_factory=list)
    all_verified: bool = False
    total_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Comparison between retrieval-only and agentic."""
    query: str
    retrieval_results: int
    retrieval_time_ms: float
    retrieval_top_score: float
    agentic_answer: str
    agentic_iterations: int
    agentic_verified: bool
    agentic_time_ms: float


# ============================================================================
# AGENTIC RAG PIPELINE (ARTIFACT A + B)
# ============================================================================

class AgenticRAGPipeline:
    """
    Complete Agentic RAG Pipeline.
    
    Integrates:
    - Artifact A: Data Layer (Embeddings, Storage, Retrieval)
    - Artifact B: Logic Layer (Planner, Verifier, AgenticController)
    
    Pipeline:
        Query → Planner → Navigator (HybridRetriever) → Verifier → Answer
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Data Layer (Artifact A)
        self.embeddings = None
        self.hybrid_store = None
        self.retriever = None
        
        # Logic Layer (Artifact B)
        self.planner = None
        self.verifier = None
        self.controller = None
        
        # Document cache
        self.documents = {}
        
        # Import modules
        self.data_modules, data_ok = import_data_layer()
        self.logic_modules, logic_ok = import_logic_layer()
        
        if not data_ok:
            raise ImportError("Failed to import Data Layer modules (Artifact A)")
        
        self.logic_layer_available = logic_ok
        
        self.logger.info("AgenticRAGPipeline initialized")
        if not logic_ok:
            self.logger.warning("Logic Layer (Artifact B) not available - running in retrieval-only mode")
    
    def setup(self) -> None:
        """Initialize all components."""
        self.logger.info("=" * 70)
        self.logger.info("ARTIFACT A + B: AGENTIC RAG PIPELINE SETUP")
        self.logger.info("=" * 70)
        
        # Initialize Data Layer (Artifact A)
        self._init_data_layer()
        
        # Initialize Logic Layer (Artifact B)
        if self.logic_layer_available:
            self._init_logic_layer()
        else:
            self.logger.warning("Skipping Logic Layer initialization")
        
        self.logger.info("=" * 70)
        self.logger.info("[OK] AGENTIC PIPELINE READY")
        self.logger.info("=" * 70)
    
    def _init_data_layer(self) -> None:
        """Initialize Data Layer components."""
        self.logger.info("\n--- DATA LAYER (ARTIFACT A) ---")
        
        # Embeddings
        self.logger.info("Initializing Embeddings...")
        embedding_config = self.config.get("embeddings", {})
        perf_config = self.config.get("performance", {})
        cache_path = Path(self.config["paths"].get("cache", "./cache")) / "embeddings.db"
        
        BatchedOllamaEmbeddings = self.data_modules['BatchedOllamaEmbeddings']
        self.embeddings = BatchedOllamaEmbeddings(
            model_name=embedding_config.get("model_name", "nomic-embed-text"),
            base_url=embedding_config.get("base_url", "http://localhost:11434"),
            batch_size=perf_config.get("batch_size", 32),
            cache_path=cache_path,
            device=perf_config.get("device", "cpu"),
        )
        self.logger.info(f"  [OK] Embeddings: {embedding_config.get('model_name')}")
        
        # Storage
        self.logger.info("Initializing Hybrid Storage...")
        vector_config = self.config.get("vector_store", {})
        
        StorageConfig = self.data_modules['StorageConfig']
        HybridStore = self.data_modules['HybridStore']
        
        storage_config = StorageConfig(
            vector_db_path=Path(self.config["paths"].get("vector_db", "./data/vector_db")),
            graph_db_path=Path(self.config["paths"].get("graph_db", "./data/knowledge_graph")),
            embedding_dim=embedding_config.get("embedding_dim", 768),
            similarity_threshold=vector_config.get("similarity_threshold", 0.3),
            normalize_embeddings=vector_config.get("normalize_embeddings", True),
            distance_metric=vector_config.get("distance_metric", "cosine"),
        )
        
        self.hybrid_store = HybridStore(config=storage_config, embeddings=self.embeddings)
        self._try_load_existing_store()
        self.logger.info("  [OK] Hybrid Storage initialized")
        
        # Retriever
        self.logger.info("Initializing Retriever...")
        rag_config = self.config.get("rag", {})
        graph_config = self.config.get("graph", {})
        
        RetrievalConfig = self.data_modules['RetrievalConfig']
        RetrievalMode = self.data_modules['RetrievalMode']
        HybridRetriever = self.data_modules['HybridRetriever']
        
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
        self.logger.info(f"  [OK] Retriever: mode={rag_config.get('retrieval_mode', 'hybrid')}")
    
    def _init_logic_layer(self) -> None:
        """Initialize Logic Layer components."""
        self.logger.info("\n--- LOGIC LAYER (ARTIFACT B) ---")
        
        llm_config = self.config.get("llm", {})
        agent_config = self.config.get("agent", {})
        
        # Planner
        self.logger.info("Initializing Planner...")
        try:
            create_planner = self.logic_modules['create_planner']
            self.planner = create_planner(
                model_name=llm_config.get("model_name", "phi3"),
                base_url=llm_config.get("base_url", "http://localhost:11434"),
            )
            self.logger.info(f"  [OK] Planner: model={llm_config.get('model_name')}")
        except Exception as e:
            self.logger.error(f"  [FAIL] Planner: {e}")
            self.planner = None
        
        # Verifier
        self.logger.info("Initializing Verifier...")
        try:
            create_verifier = self.logic_modules['create_verifier']
            self.verifier = create_verifier(
                model_name=llm_config.get("model_name", "phi3"),
                base_url=llm_config.get("base_url", "http://localhost:11434"),
                max_iterations=agent_config.get("max_verification_iterations", 3),
            )
            self.logger.info(f"  [OK] Verifier: max_iter={agent_config.get('max_verification_iterations', 3)}")
        except Exception as e:
            self.logger.error(f"  [FAIL] Verifier: {e}")
            self.verifier = None
        
        # Agentic Controller
        self.logger.info("Initializing AgenticController...")
        try:
            create_controller = self.logic_modules['create_controller']
            self.controller = create_controller(
                model_name=llm_config.get("model_name", "phi3"),
                base_url=llm_config.get("base_url", "http://localhost:11434"),
            )
            
            # Connect retriever
            self.controller.set_retriever(self.retriever, self.documents)
            
            # Connect graph store for verification
            if hasattr(self.hybrid_store, 'graph_store'):
                self.controller.set_graph_store(self.hybrid_store.graph_store)
            
            self.logger.info("  [OK] AgenticController connected")
        except Exception as e:
            self.logger.error(f"  [FAIL] AgenticController: {e}")
            self.controller = None
    
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
            self.logger.debug(f"No existing store: {e}")
    
    def ingest_documents(self, documents_path: Optional[Path] = None) -> int:
        """Run document ingestion pipeline."""
        self.logger.info("=" * 70)
        self.logger.info("DOCUMENT INGESTION")
        self.logger.info("=" * 70)
        
        if documents_path is None:
            documents_path = Path(self.config["paths"].get("documents", "./data/documents"))
        
        if not documents_path.exists():
            self.logger.error(f"Documents path not found: {documents_path}")
            return 0
        
        load_ingestion_config = self.data_modules['load_ingestion_config']
        DocumentIngestionPipeline = self.data_modules['DocumentIngestionPipeline']
        
        chunking_config = load_ingestion_config(Path("./config/settings.yaml"))
        
        pipeline = DocumentIngestionPipeline(
            documents_path=documents_path,
            chunking_config=chunking_config,
        )
        
        start_time = time.time()
        chunked_docs = pipeline.process_documents()
        
        if not chunked_docs:
            self.logger.warning("No documents found")
            return 0
        
        # Add to storage
        self.hybrid_store.add_documents(chunked_docs)
        self.hybrid_store.save()
        
        # Cache for navigator
        for doc in chunked_docs:
            doc_id = str(doc.metadata.get("chunk_id", len(self.documents)))
            self.documents[doc_id] = doc.page_content
        
        # Update controller if available
        if self.controller:
            self.controller.set_retriever(self.retriever, self.documents)
        
        elapsed = time.time() - start_time
        self.logger.info(f"[OK] Ingested {len(chunked_docs)} chunks in {elapsed:.2f}s")
        
        return len(chunked_docs)
    
    def query(self, question: str) -> AgenticResult:
        """
        Execute full agentic pipeline.
        
        Pipeline: Planner → Navigator → Verifier
        
        Args:
            question: User question
            
        Returns:
            AgenticResult with answer and metadata
        """
        start_time = time.time()
        
        # Use AgenticController if available
        if self.controller:
            self.logger.info(f"\n[AGENTIC QUERY] {question}")
            
            try:
                state = self.controller.run(question)
                elapsed_ms = (time.time() - start_time) * 1000
                
                return AgenticResult(
                    query=question,
                    answer=state.get("answer", ""),
                    sub_queries=state.get("sub_queries", []),
                    context_docs=state.get("context", []),
                    iterations=state.get("iterations", 0),
                    verified_claims=state.get("verified_claims", []),
                    violated_claims=state.get("violated_claims", []),
                    all_verified=state.get("all_verified", False),
                    total_time_ms=elapsed_ms,
                    errors=state.get("errors", []),
                )
            except Exception as e:
                self.logger.error(f"AgenticController failed: {e}")
                return self._fallback_query(question, start_time, str(e))
        else:
            return self._fallback_query(question, start_time, "Controller not available")
    
    def _fallback_query(
        self,
        question: str,
        start_time: float,
        error_msg: str
    ) -> AgenticResult:
        """Fallback to simple retrieval + LLM generation."""
        self.logger.warning(f"Using fallback query mode: {error_msg}")
        
        # Step 1: Retrieval
        results = self.retriever.retrieve(question)
        context = [r.text for r in results[:5]]
        
        # Step 2: Generate answer (if verifier available)
        answer = ""
        if self.verifier:
            try:
                verification_result = self.verifier.generate_and_verify(question, context)
                answer = verification_result.answer
            except Exception as e:
                answer = f"[Retrieval completed, but LLM generation failed: {e}]"
        else:
            answer = f"[Retrieved {len(results)} documents. LLM not available for answer generation.]"
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return AgenticResult(
            query=question,
            answer=answer,
            sub_queries=[question],  # No decomposition
            context_docs=context,
            iterations=1,
            total_time_ms=elapsed_ms,
            errors=[error_msg],
        )
    
    def retrieval_only(self, question: str) -> List[Any]:
        """
        Perform retrieval without LLM (for comparison).
        
        Args:
            question: Search query
            
        Returns:
            List of retrieval results
        """
        return self.retriever.retrieve(question)


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_agentic_result(result: AgenticResult) -> None:
    """Pretty print agentic result."""
    print("\n" + "=" * 70)
    print("AGENTIC RAG RESULT")
    print("=" * 70)
    
    print(f"\nQuery: {result.query}")
    
    if result.sub_queries:
        print(f"\nSub-queries ({len(result.sub_queries)}):")
        for i, sq in enumerate(result.sub_queries, 1):
            print(f"  {i}. {sq}")
    
    print(f"\nContext Documents: {len(result.context_docs)}")
    
    print(f"\n--- ANSWER ---")
    print(result.answer)
    print("-" * 40)
    
    print(f"\nVerification:")
    print(f"  Iterations: {result.iterations}")
    print(f"  All Verified: {result.all_verified}")
    print(f"  Verified Claims: {len(result.verified_claims)}")
    print(f"  Violated Claims: {len(result.violated_claims)}")
    
    if result.violated_claims:
        print("\n  Violations:")
        for v in result.violated_claims[:3]:
            print(f"    - {v}")
    
    print(f"\nTime: {result.total_time_ms:.1f}ms")
    
    if result.errors:
        print(f"\nErrors: {result.errors}")
    
    print("=" * 70 + "\n")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def run_interactive(pipeline: AgenticRAGPipeline) -> None:
    """Run interactive chat mode."""
    print("\n" + "=" * 70)
    print("ARTIFACT A+B: INTERACTIVE AGENTIC RAG")
    print("=" * 70)
    print("Commands:")
    print("  [question]    - Ask a question (full agentic pipeline)")
    print("  :retrieval    - Show last retrieval results")
    print("  :compare      - Compare agentic vs retrieval-only")
    print("  :stats        - Show embedding statistics")
    print("  :quit         - Exit")
    print("=" * 70 + "\n")
    
    last_result = None
    last_retrieval = None
    
    while True:
        try:
            query = input("\nQuestion> ").strip()
            
            if not query:
                continue
            
            if query.lower() in [':quit', ':exit', ':q']:
                print("Exiting...")
                break
            
            if query == ':retrieval' and last_retrieval:
                print(f"\nLast Retrieval ({len(last_retrieval)} results):")
                for i, r in enumerate(last_retrieval[:5], 1):
                    print(f"  {i}. Score: {r.relevance_score:.3f} | {r.text[:80]}...")
                continue
            
            if query == ':compare' and last_result:
                print("\nComparison: Agentic vs Retrieval-only")
                retrieval = pipeline.retrieval_only(last_result.query)
                print(f"  Retrieval: {len(retrieval)} results")
                print(f"  Agentic: {len(last_result.context_docs)} context docs")
                print(f"  Sub-queries: {len(last_result.sub_queries)}")
                print(f"  Answer length: {len(last_result.answer)} chars")
                continue
            
            if query == ':stats':
                if hasattr(pipeline.embeddings, 'print_metrics'):
                    pipeline.embeddings.print_metrics()
                continue
            
            # Execute agentic query
            last_result = pipeline.query(query)
            last_retrieval = pipeline.retrieval_only(query)
            
            print_agentic_result(last_result)
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


# ============================================================================
# BENCHMARK MODE
# ============================================================================

BENCHMARK_QUERIES = [
    # Simple factual
    "What is financial sentiment analysis?",
    "What is knowledge management?",
    
    # Multi-hop (should trigger query decomposition)
    "What is the model structure of MTMEC and how does it relate to energy consumption?",
    "How do neural networks process embeddings for retrieval?",
    
    # Complex
    "Explain how hybrid retrieval combines vector search and graph traversal",
    "What are the advantages of quantized models on edge devices?",
]


def run_benchmark(pipeline: AgenticRAGPipeline) -> None:
    """Run benchmark queries."""
    print("\n" + "=" * 70)
    print("ARTIFACT A+B: AGENTIC BENCHMARK")
    print("=" * 70 + "\n")
    
    results = []
    
    for query in BENCHMARK_QUERIES:
        print(f"\n--- Query: {query[:50]}... ---")
        
        # Run agentic
        agentic_result = pipeline.query(query)
        
        # Run retrieval-only
        retrieval_start = time.time()
        retrieval_results = pipeline.retrieval_only(query)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Collect comparison
        comparison = ComparisonResult(
            query=query,
            retrieval_results=len(retrieval_results),
            retrieval_time_ms=retrieval_time,
            retrieval_top_score=retrieval_results[0].relevance_score if retrieval_results else 0,
            agentic_answer=agentic_result.answer[:100] + "...",
            agentic_iterations=agentic_result.iterations,
            agentic_verified=agentic_result.all_verified,
            agentic_time_ms=agentic_result.total_time_ms,
        )
        results.append(comparison)
        
        print(f"  Retrieval: {comparison.retrieval_results} docs, "
              f"top={comparison.retrieval_top_score:.3f}, "
              f"{comparison.retrieval_time_ms:.0f}ms")
        print(f"  Agentic: {comparison.agentic_iterations} iter, "
              f"verified={comparison.agentic_verified}, "
              f"{comparison.agentic_time_ms:.0f}ms")
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    avg_retrieval_time = sum(r.retrieval_time_ms for r in results) / len(results)
    avg_agentic_time = sum(r.agentic_time_ms for r in results) / len(results)
    verified_count = sum(1 for r in results if r.agentic_verified)
    
    print(f"  Queries: {len(BENCHMARK_QUERIES)}")
    print(f"  Avg Retrieval Time: {avg_retrieval_time:.1f}ms")
    print(f"  Avg Agentic Time: {avg_agentic_time:.1f}ms")
    print(f"  Agentic Overhead: {(avg_agentic_time / avg_retrieval_time - 1) * 100:.1f}%")
    print(f"  Verified Answers: {verified_count}/{len(results)}")
    
    print("=" * 70 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Artifact A+B: Full Agentic RAG Pipeline Testing"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./config/settings.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents before querying"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to execute"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark queries"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare agentic vs retrieval-only"
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
    log_file = Path("./logs/agentic_pipeline.log")
    logger = setup_logging(level=args.log_level, log_file=log_file)
    
    logger.info("=" * 70)
    logger.info("ARTIFACT A+B: AGENTIC RAG PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize pipeline
        pipeline = AgenticRAGPipeline(config, logger)
        pipeline.setup()
        
        # Ingest if requested
        if args.ingest:
            doc_path = args.documents or Path(config["paths"].get("documents"))
            pipeline.ingest_documents(doc_path)
        
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
            result = pipeline.query(args.query)
            print_agentic_result(result)
        
        elif args.interactive:
            # Interactive mode
            run_interactive(pipeline)
        
        elif args.benchmark:
            # Benchmark mode
            run_benchmark(pipeline)
        
        elif args.compare:
            # Comparison demo
            demo_query = "What is knowledge management and how does it relate to AI?"
            
            print("\n--- RETRIEVAL ONLY ---")
            retrieval_results = pipeline.retrieval_only(demo_query)
            print(f"Results: {len(retrieval_results)}")
            for r in retrieval_results[:3]:
                print(f"  Score: {r.relevance_score:.3f} | {r.text[:60]}...")
            
            print("\n--- AGENTIC PIPELINE ---")
            agentic_result = pipeline.query(demo_query)
            print_agentic_result(agentic_result)
        
        else:
            # Default: demo query
            demo_query = "What is knowledge management?"
            logger.info(f"Running demo query: {demo_query}")
            
            result = pipeline.query(demo_query)
            print_agentic_result(result)
        
        logger.info("=" * 70)
        logger.info("ARTIFACT A+B: PIPELINE COMPLETED")
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