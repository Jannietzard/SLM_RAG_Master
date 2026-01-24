"""
Main Entry-Point: Graph-Augmented Edge-RAG Pipeline Orchestration.

Version: 2.1.0
Last Modified: 2026-01-13

Pipeline:
1. Load Config (settings.yaml)
2. Initiate Document Ingestion (PDF -> Chunks)
3. Initialize Embeddings (Ollama nomic-embed-text)
4. Store: Vectors (LanceDB) + Graph (NetworkX)
5. Retrieval: Hybrid (Vector + Graph Ensemble)

Scientific Foundation:
Decentralized AI Architecture for Edge Devices with quantized SLMs.
Full Stack: Ingestion -> Embedding -> Storage -> Retrieval -> Generation
All on-device, zero cloud dependencies.

CHANGES IN v2.1.0:
- Added distance_metric parameter to StorageConfig
- Explicit cosine metric for correct similarity computation
- Improved logging without UTF-8 special characters
"""
import logging
import sys
from pathlib import Path
from typing import List

import yaml
from langchain.schema import Document

# Local imports
from src/data_layer.ingestion import DocumentIngestionPipeline, load_ingestion_config
from src/data_layer.storage import HybridStore, StorageConfig
from src.retrieval import HybridRetriever, RetrievalConfig, RetrievalMode
from src.data_layer.embeddings import BatchedOllamaEmbeddings


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Path = Path("./logs/edge_rag.log")) -> logging.Logger:
    """
    Windows-compatible logging setup with UTF-8 support.
    
    Fixes Windows cp1252 encoding issues with Unicode characters.

    Args:
        log_file: Path to log file

    Returns:
        Logger instance
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler with UTF-8 (for Windows CP1252 fix)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Windows UTF-8 Fix: Force UTF-8 encoding for console output
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')

    # File Handler with UTF-8
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_configuration(config_path: Path) -> dict:
    """
    Load central configuration from YAML.

    Args:
        config_path: Path to settings.yaml

    Returns:
        Config dictionary

    Raises:
        FileNotFoundError: If config does not exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = logging.getLogger(__name__)
    logger.info(f"Configuration loaded: {config_path}")

    return config


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

class EdgeRAGPipeline:
    """
    Orchestrates the complete Edge-RAG Pipeline.
    
    Design Pattern: Pipeline/Orchestrator Pattern
    Responsibilities:
    - Config Management
    - Component Initialization
    - Data Flow Orchestration
    - Error Handling and Logging
    """

    def __init__(self, config: dict, logger_instance: logging.Logger):
        """
        Initialize pipeline with configuration.

        Args:
            config: Configuration dictionary
            logger_instance: Logger instance
        """
        self.config = config
        self.logger = logger_instance

        # Paths from config
        self.data_path = Path(config.get("paths", {}).get("documents", "./data/documents"))
        self.vector_db_path = Path(config.get("paths", {}).get("vector_db", "./data/vector_db"))
        self.graph_db_path = Path(config.get("paths", {}).get("graph_db", "./data/knowledge_graph"))

        # Components (lazy initialized)
        self.embeddings = None
        self.ingestion_pipeline = None
        self.hybrid_store = None
        self.retriever = None

        self.logger.info("EdgeRAGPipeline initialized")

    def initialize_embeddings(self) -> BatchedOllamaEmbeddings:
        """
        Initialize embedding model (Ollama nomic-embed-text).
        
        With batching + caching for 10-100x speedup.

        Returns:
            BatchedOllamaEmbeddings instance

        Raises:
            Exception: If Ollama is not reachable
        """
        try:
            embedding_config = self.config.get("embeddings", {})
            perf_config = self.config.get("performance", {})
            
            embeddings = BatchedOllamaEmbeddings(
                model_name=embedding_config.get("model_name", "nomic-embed-text"),
                base_url=embedding_config.get("base_url", "http://localhost:11434"),
                batch_size=perf_config.get("batch_size", 32),
                cache_path=Path(self.config.get("paths", {}).get("cache", "./cache")) / "embeddings.db",
                device=perf_config.get("device", "cpu"),
            )

            # Test: Embed a sample text
            test_embedding = embeddings.embed_query("test")
            embedding_dim = len(test_embedding)

            self.logger.info(
                f"[OK] Embeddings initialized: {embedding_config.get('model_name')} "
                f"(dim={embedding_dim}, batch_size={perf_config.get('batch_size', 32)}, "
                f"cached={embeddings.cache.get_stats()['total_entries']} entries)"
            )

            return embeddings

        except Exception as e:
            self.logger.error(
                f"Error initializing embeddings: {str(e)}. "
                f"Make sure Ollama is running: ollama serve"
            )
            raise

    def initialize_ingestion(self) -> DocumentIngestionPipeline:
        """
        Initialize Document Ingestion Pipeline.

        Returns:
            DocumentIngestionPipeline instance

        Raises:
            FileNotFoundError: If document path does not exist
        """
        try:
            # Load chunking config
            chunking_config = load_ingestion_config(Path("./config/settings.yaml"))

            # Create document directory if not exists
            self.data_path.mkdir(parents=True, exist_ok=True)

            ingestion = DocumentIngestionPipeline(
                chunking_config=chunking_config,
                document_path=self.data_path,
                logger_instance=self.logger,
            )

            self.logger.info("Ingestion Pipeline initialized")
            return ingestion

        except Exception as e:
            self.logger.error(f"Error initializing ingestion: {str(e)}")
            raise

    def initialize_storage(self) -> HybridStore:
        """
        Initialize Hybrid Storage (Vectors + Graph).
        
        CRITICAL UPDATE v2.1.0:
        - Added distance_metric="cosine" to ensure correct similarity computation
        - Added normalize_embeddings=True for consistent vector processing

        Returns:
            HybridStore instance
        """
        try:
            embedding_config = self.config.get("embeddings", {})
            vector_store_config = self.config.get("vector_store", {})
            
            # ================================================================
            # CRITICAL: Include distance_metric parameter
            # ================================================================
            storage_config = StorageConfig(
                vector_db_path=self.vector_db_path,
                graph_db_path=self.graph_db_path,
                embedding_dim=embedding_config.get("embedding_dim", 768),
                similarity_threshold=vector_store_config.get(
                    "similarity_threshold", 0.3
                ),
                normalize_embeddings=vector_store_config.get(
                    "normalize_embeddings", True
                ),
                distance_metric=vector_store_config.get(
                    "distance_metric", "cosine"  # EXPLICIT COSINE METRIC
                ),
            )

            store = HybridStore(
                config=storage_config,
                embeddings=self.embeddings,
            )

            self.logger.info(
                f"Hybrid Store initialized: "
                f"metric={storage_config.distance_metric}, "
                f"normalize={storage_config.normalize_embeddings}"
            )
            return store

        except Exception as e:
            self.logger.error(f"Error initializing store: {str(e)}")
            raise

    def initialize_retriever(self) -> HybridRetriever:
        """
        Initialize Hybrid Retriever (Vector + Graph Ensemble).
        
        Note:
            Currently configured with graph_weight=0 for vector-only evaluation.
            This allows isolated testing of vector retrieval performance.

        Returns:
            HybridRetriever instance
        """
        try:
            rag_config = self.config.get("rag", {})
            vector_store_config = self.config.get("vector_store", {})
            
            retrieval_config = RetrievalConfig(
                mode=RetrievalMode(rag_config.get("retrieval_mode", "hybrid")),
                top_k_vector=vector_store_config.get("top_k_vectors", 10),
                top_k_graph=rag_config.get("top_k_entities", 5),
                vector_weight=rag_config.get("vector_weight", 1.0),  # Vector-only for now
                graph_weight=rag_config.get("graph_weight", 0.0),   # Graph disabled
                similarity_threshold=vector_store_config.get(
                    "similarity_threshold", 0.3
                ),
            )

            retriever = HybridRetriever(
                config=retrieval_config,
                hybrid_store=self.hybrid_store,
                embeddings=self.embeddings,
            )

            self.logger.info(
                f"Hybrid Retriever initialized: "
                f"mode={retrieval_config.mode}, "
                f"vector_weight={retrieval_config.vector_weight}, "
                f"graph_weight={retrieval_config.graph_weight}"
            )
            return retriever

        except Exception as e:
            self.logger.error(f"Error initializing retriever: {str(e)}")
            raise

    def run_ingestion_pipeline(self) -> List[Document]:
        """
        Execute Document Ingestion.

        Returns:
            Chunked documents (not just count)

        Raises:
            Exception: On ingestion error
        """
        try:
            self.logger.info("Starting Document Ingestion...")
            documents = self.ingestion_pipeline.process_documents()
            
            self.logger.info(f"[OK] {len(documents)} documents chunked")
            return documents

        except Exception as e:
            self.logger.error(f"Ingestion Pipeline failed: {str(e)}")
            raise

    def run_storage_pipeline(self, documents: list) -> None:
        """
        Add documents to storage.

        Args:
            documents: Chunked documents
        """
        try:
            self.logger.info("Adding documents to Hybrid Store...")
            self.hybrid_store.add_documents(documents)
            self.hybrid_store.save()
            
            self.logger.info("[OK] Documents saved to storage")

        except Exception as e:
            self.logger.error(f"Storage Pipeline failed: {str(e)}")
            raise

    def retrieve(self, query: str) -> list:
        """
        Perform Hybrid Retrieval.

        Args:
            query: User query

        Returns:
            List of RetrievalResult objects
        """
        try:
            self.logger.info(f"Retrieval for query: '{query}'")
            results = self.retriever.retrieve(query)
            
            self.logger.info(f"[OK] {len(results)} results returned")
            return results

        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}")
            raise

    def setup(self) -> None:
        """
        Execute complete setup: Initialize all components in dependency order.
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING EDGE-RAG PIPELINE SETUP")
        self.logger.info("=" * 70)

        try:
            # 1. Embeddings
            self.embeddings = self.initialize_embeddings()

            # 2. Ingestion
            self.ingestion_pipeline = self.initialize_ingestion()

            # 3. Storage
            self.hybrid_store = self.initialize_storage()

            # 4. Retriever
            self.retriever = self.initialize_retriever()

            self.logger.info("=" * 70)
            self.logger.info("[OK] PIPELINE SETUP SUCCESSFUL")
            self.logger.info("=" * 70)

        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Main Entry Point for Edge-RAG Pipeline.
    """
    # Setup Logging
    logger = setup_logging()
    logger.info("Edge-RAG Pipeline Start")

    try:
        # Load Config
        config = load_configuration(Path("./config/settings.yaml"))

        # Initialize Pipeline
        pipeline = EdgeRAGPipeline(config, logger)
        pipeline.setup()

        # Execute Ingestion (once, returns documents)
        documents = pipeline.run_ingestion_pipeline()

        if len(documents) > 0:
            # Save to Storage (use already loaded documents)
            pipeline.run_storage_pipeline(documents)

            # Example Retrieval
            query = "What is the model structure of MTMEC"
            results = pipeline.retrieve(query)

            logger.info("\n" + "=" * 70)
            logger.info("RETRIEVAL RESULTS")
            logger.info("=" * 70)

            if results:
                for i, result in enumerate(results[:5], 1):
                    logger.info(f"\nResult {i}:")
                    logger.info(f"  Score: {result.relevance_score:.4f}")
                    logger.info(f"  Method: {result.retrieval_method}")
                    logger.info(f"  Text: {result.text[:200]}...")
            else:
                logger.warning("No retrieval results. Check vector store configuration.")
            
            # Print Embedding Metrics
            pipeline.embeddings.print_metrics()

        else:
            logger.warning(
                "No documents found. "
                "Please place PDFs in: ./data/documents"
            )

    except Exception as e:
        logger.critical(f"Pipeline Error: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Edge-RAG Pipeline completed")
    logger.info("=" * 70)

"""
Main Application - End-to-End Hybrid RAG System.

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

Integriert deine bestehenden Module mit dem neuen Agentic Controller:
- storage.py → HybridStore, KnowledgeGraphStore
- embeddings.py → BatchedOllamaEmbeddings  
- ingestion.py → DocumentIngestionPipeline
- retrieval.py → HybridRetriever

Neue Module (Artifact B):
- planner.py → Query Decomposition
- verifier.py → Answer Generation + Verification
- agent.py → Agentic Controller

Usage:
    python main.py ingest --documents ./data/documents
    python main.py query "What is knowledge management?"
    python main.py interactive
"""

import logging
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# ============================================================================
# IMPORTS - Deine bestehenden Module
# ============================================================================

try:
    from storage import HybridStore, StorageConfig
    from embeddings import BatchedOllamaEmbeddings
    from ingestion import DocumentIngestionPipeline, load_ingestion_config
    from retrieval import HybridRetriever, RetrievalConfig, RetrievalMode
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"⚠ Core modules nicht gefunden: {e}")

# Neue Module (Artifact B)
try:
    from planner import create_planner
    from verifier import create_verifier
    from agent import AgenticController, create_controller
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    print(f"⚠ Agent modules nicht gefunden: {e}")


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Konfiguriere Logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config(config_path: Path = Path("settings.yaml")) -> Dict[str, Any]:
    """Lade Konfiguration aus settings.yaml."""
    if not config_path.exists():
        logging.warning(f"Config nicht gefunden: {config_path}")
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Default Konfiguration."""
    return {
        "llm": {"model_name": "phi3", "base_url": "http://localhost:11434", "temperature": 0.1},
        "embeddings": {"model_name": "nomic-embed-text", "base_url": "http://localhost:11434"},
        "chunking": {"mode": "standard", "chunk_size": 512, "chunk_overlap": 128},
        "vector_store": {"db_path": "./data/vector_db", "similarity_threshold": 0.25},
        "graph": {"graph_path": "./data/knowledge_graph.graphml", "max_hops": 2},
        "rag": {"retrieval_mode": "hybrid", "top_k_vector": 10, "top_k_graph": 5,
                "vector_weight": 0.6, "graph_weight": 0.4},
        "paths": {"documents": "./data/documents"},
    }


class HybridRAGSystem:
    """
    Hauptklasse für das Hybrid RAG System.
    
    Verbindet deine bestehenden Module mit dem neuen Agentic Controller.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.embeddings = None
        self.hybrid_store = None
        self.retriever = None
        self.controller = None
        self.documents = {}
    
    def initialize_data_layer(self) -> None:
        """Initialisiere Data Layer (deine Module)."""
        self.logger.info("Initialisiere Data Layer...")
        
        if not CORE_AVAILABLE:
            raise ImportError("Core modules nicht verfügbar")
        
        # Embeddings (dein embeddings.py)
        self.embeddings = BatchedOllamaEmbeddings(
            model_name=self.config["embeddings"]["model_name"],
            base_url=self.config["embeddings"]["base_url"],
            cache_path=Path("./cache/embeddings.db"),
        )
        
        # Storage Config
        storage_config = StorageConfig(
            vector_db_path=Path(self.config["vector_store"]["db_path"]),
            graph_db_path=Path(self.config["graph"]["graph_path"]),
            similarity_threshold=self.config["vector_store"]["similarity_threshold"],
        )
        
        # Hybrid Store (dein storage.py)
        self.hybrid_store = HybridStore(storage_config, self.embeddings)
        self.logger.info("✓ Data Layer initialisiert")
    
    def initialize_logic_layer(self) -> None:
        """Initialisiere Logic Layer (neue Module)."""
        self.logger.info("Initialisiere Logic Layer...")
        
        if not AGENT_AVAILABLE:
            raise ImportError("Agent modules nicht verfügbar")
        if not CORE_AVAILABLE:
            raise ImportError("Core modules nicht verfügbar")
        
        # Retrieval Config (für deinen HybridRetriever)
        retrieval_config = RetrievalConfig(
            mode=RetrievalMode(self.config["rag"]["retrieval_mode"]),
            top_k_vector=self.config["rag"]["top_k_vector"],
            top_k_graph=self.config["rag"]["top_k_graph"],
            vector_weight=self.config["rag"]["vector_weight"],
            graph_weight=self.config["rag"]["graph_weight"],
            similarity_threshold=self.config["vector_store"]["similarity_threshold"],
        )
        
        # Dein HybridRetriever (retrieval.py)
        self.retriever = HybridRetriever(
            config=retrieval_config,
            hybrid_store=self.hybrid_store,
            embeddings=self.embeddings,
        )
        
        # Neuer Agentic Controller (agent.py)
        self.controller = create_controller(
            model_name=self.config["llm"]["model_name"],
            base_url=self.config["llm"]["base_url"],
        )
        
        # Verbinde Controller mit deinen Modulen
        self.controller.set_retriever(self.retriever, self.documents)
        self.controller.set_graph_store(self.hybrid_store.graph_store)
        
        self.logger.info("✓ Logic Layer initialisiert")
    
    def ingest_documents(self, documents_path: Path) -> int:
        """Ingestion Pipeline (dein ingestion.py)."""
        self.logger.info(f"Starte Ingestion: {documents_path}")
        
        chunking_config = load_ingestion_config(Path("settings.yaml"))
        
        pipeline = DocumentIngestionPipeline(
            documents_path=documents_path,
            chunking_config=chunking_config,
        )
        
        chunked_docs = pipeline.process_documents()
        self.hybrid_store.add_documents(chunked_docs)
        self.hybrid_store.save()
        
        # Cache für Navigator
        for doc in chunked_docs:
            doc_id = str(doc.metadata.get("chunk_id", len(self.documents)))
            self.documents[doc_id] = doc.page_content
        
        self.logger.info(f"✓ {len(chunked_docs)} Chunks indexiert")
        return len(chunked_docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Führe Query durch Agentic Pipeline."""
        if self.controller is None:
            raise RuntimeError("Logic Layer nicht initialisiert")
        return self.controller.run(question)
    
    def interactive_mode(self) -> None:
        """Interaktiver Chat-Modus."""
        print("\n" + "="*70)
        print("HYBRID RAG SYSTEM - Interactive Mode")
        print("="*70)
        print("Stelle Fragen oder 'quit' zum Beenden.\n")
        
        while True:
            try:
                question = input("📝 Frage: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Auf Wiedersehen!")
                    break
                
                if not question:
                    continue
                
                print("\n🔄 Verarbeite...")
                result = self.query(question)
                
                print(f"\n📌 Antwort: {result['answer']}")
                print(f"⏱ {result['total_time_ms']:.0f}ms | "
                      f"📄 {len(result.get('context', []))} Docs | "
                      f"🔄 {result.get('iterations', 1)} Iter.\n")
                
            except KeyboardInterrupt:
                print("\nAbgebrochen.")
                break
            except Exception as e:
                print(f"❌ Fehler: {e}\n")


# ============================================================================
# CLI COMMANDS
# ============================================================================

def cmd_ingest(args, config):
    """Ingestion Command."""
    system = HybridRAGSystem(config)
    system.initialize_data_layer()
    
    documents_path = Path(args.documents or config["paths"]["documents"])
    if not documents_path.exists():
        print(f"❌ Nicht gefunden: {documents_path}")
        return 1
    
    num_chunks = system.ingest_documents(documents_path)
    print(f"\n✅ {num_chunks} Chunks indexiert")
    return 0


def cmd_query(args, config):
    """Query Command."""
    system = HybridRAGSystem(config)
    system.initialize_data_layer()
    system.initialize_logic_layer()
    
    result = system.query(args.question)
    
    print("\n" + "="*70)
    print(f"Answer: {result['answer']}")
    print("-"*70)
    print(f"Sub-Queries: {result.get('sub_queries', [])}")
    print(f"Time: {result.get('total_time_ms', 0):.0f}ms")
    return 0


def cmd_interactive(args, config):
    """Interactive Mode."""
    system = HybridRAGSystem(config)
    system.initialize_data_layer()
    system.initialize_logic_layer()
    system.interactive_mode()
    return 0


def cmd_demo(args, config):
    """Demo mit Sample Data (ohne Ingestion)."""
    print("\n" + "="*70)
    print("DEMO MODE")
    print("="*70)
    
    if not AGENT_AVAILABLE:
        print("❌ Agent modules nicht verfügbar")
        return 1
    
    controller = create_controller()
    
    # Mock Context
    sample_context = [
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        "Bill Gates attended Harvard University but dropped out in 1975.",
        "The Eiffel Tower is located in Paris, France.",
        "Paris is the capital of France.",
    ]
    
    # Mock Navigator
    controller.retriever = type('MockRetriever', (), {
        'retrieve': lambda self, q: [
            type('Result', (), {'text': ctx})() for ctx in sample_context
        ]
    })()
    
    queries = [
        "What university did the founder of Microsoft attend?",
        "What is the capital of the country where the Eiffel Tower is located?",
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        result = controller.run(query)
        print(f"📌 Answer: {result['answer']}")
        print(f"⏱ {result['total_time_ms']:.0f}ms")
    
    return 0


def main():
    """Main Entry Point."""
    parser = argparse.ArgumentParser(description="Hybrid RAG System")
    parser.add_argument("--config", "-c", type=Path, default=Path("settings.yaml"))
    parser.add_argument("--log-level", "-l", default="INFO")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Ingest
    p = subparsers.add_parser("ingest", help="Dokumente indexieren")
    p.add_argument("--documents", "-d", type=Path)
    
    # Query
    p = subparsers.add_parser("query", help="Einzelne Frage")
    p.add_argument("question")
    
    # Interactive
    subparsers.add_parser("interactive", help="Chat-Modus")
    
    # Demo
    subparsers.add_parser("demo", help="Demo ohne Daten")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)
    
    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "interactive": cmd_interactive,
        "demo": cmd_demo,
    }
    
    if args.command in commands:
        return commands[args.command](args, config)
    
    parser.print_help()
    return 0



if __name__ == "__main__":
    main()