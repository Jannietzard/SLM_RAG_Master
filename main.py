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
from src.ingestion import DocumentIngestionPipeline, load_ingestion_config
from src.storage import HybridStore, StorageConfig
from src.retrieval import HybridRetriever, RetrievalConfig, RetrievalMode
from src.embeddings import BatchedOllamaEmbeddings


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


if __name__ == "__main__":
    main()