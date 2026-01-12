"""
Main Entry-Point: Graph-Augmented Edge-RAG Pipeline Orchestration.

Pipeline:
1. Load Config (settings.yaml)
2. Initiate Document Ingestion (PDF → Chunks)
3. Initialize Embeddings (Ollama nomic-embed-text)
4. Store: Vectors (LanceDB) + Graph (NetworkX)
5. Retrieval: Hybrid (Vector + Graph Ensemble)

Scientific Foundation:
Decentralized AI Architecture für Edge Devices mit quantisierten SLMs.
Full Stack: Ingestion → Embedding → Storage → Retrieval → Generation
All on-device, zero cloud dependencies.
"""

import logging
import sys
from pathlib import Path

import yaml

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
    Windows-kompatibles Logging Setup mit UTF-8 support.
    
    Fixt das Windows cp1252 Encoding Problem mit Unicode-Chars (✓, →, etc).

    Args:
        log_file: Pfad zur Log-Datei

    Returns:
        Logger Instance
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler mit UTF-8 (for Windows CP1252 fix)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Windows UTF-8 Fix: Force UTF-8 encoding für Console Output
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')

    # File Handler mit UTF-8 (wichtig für Windows)
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
    Lade zentrale Konfiguration aus YAML.

    Args:
        config_path: Pfad zur settings.yaml

    Returns:
        Config Dictionary

    Raises:
        FileNotFoundError: Falls Config nicht existiert
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = logging.getLogger(__name__)
    logger.info(f"Konfiguration geladen: {config_path}")

    return config


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

class EdgeRAGPipeline:
    """
    Orchestriert die gesamte Edge-RAG Pipeline.
    
    Design Pattern: Pipeline/Orchestrator Pattern
    Verantwortlichkeiten:
    - Config Management
    - Component Initialization
    - Data Flow Orchestration
    - Error Handling & Logging
    """

    def __init__(self, config: dict, logger_instance: logging.Logger):
        """
        Initialisiere Pipeline mit Config.

        Args:
            config: Konfigurationsdict
            logger_instance: Logger Instance
        """
        self.config = config
        self.logger = logger_instance

        # Pfade aus Config
        self.data_path = Path(config.get("paths", {}).get("documents", "./data/documents"))
        self.vector_db_path = Path(config.get("paths", {}).get("vector_db", "./data/vector_db"))
        self.graph_db_path = Path(config.get("paths", {}).get("graph_db", "./data/knowledge_graph"))

        # Komponenten (werden lazy initialisiert)
        self.embeddings = None
        self.ingestion_pipeline = None
        self.hybrid_store = None
        self.retriever = None

        self.logger.info("EdgeRAGPipeline initialisiert")

    def initialize_embeddings(self) -> BatchedOllamaEmbeddings:
        """
        Initialisiere Embedding-Modell (Ollama nomic-embed-text).
        
        Mit Batching + Caching für 10-100x Speedup!

        Returns:
            BatchedOllamaEmbeddings Instance

        Raises:
            Exception: Falls Ollama nicht erreichbar
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

            # Test: Embedde einen Beispieltext
            test_embedding = embeddings.embed_query("test")
            embedding_dim = len(test_embedding)

            self.logger.info(
                f"[OK] Embeddings initialisiert: {embedding_config.get('model_name')} "
                f"(dim={embedding_dim}, batch_size={perf_config.get('batch_size', 32)}, "
                f"cached={embeddings.cache.get_stats()['total_entries']} entries)"
            )

            return embeddings

        except Exception as e:
            self.logger.error(
                f"Fehler beim Initialisieren der Embeddings: {str(e)}. "
                f"Stelle sicher, dass Ollama läuft: ollama serve"
            )
            raise

    def initialize_ingestion(self) -> DocumentIngestionPipeline:
        """
        Initialisiere Document Ingestion Pipeline.

        Returns:
            DocumentIngestionPipeline Instance

        Raises:
            FileNotFoundError: Falls Document Path nicht existiert
        """
        try:
            # Lade Chunking Config
            chunking_config = load_ingestion_config(Path("./config/settings.yaml"))

            # Erstelle Dokumentverzeichnis falls nicht existiert
            self.data_path.mkdir(parents=True, exist_ok=True)

            ingestion = DocumentIngestionPipeline(
                chunking_config=chunking_config,
                document_path=self.data_path,
                logger_instance=self.logger,
            )

            self.logger.info("Ingestion Pipeline initialisiert")
            return ingestion

        except Exception as e:
            self.logger.error(f"Fehler beim Initialisieren der Ingestion: {str(e)}")
            raise

    def initialize_storage(self) -> HybridStore:
        """
        Initialisiere Hybrid Storage (Vectors + Graph).

        Returns:
            HybridStore Instance
        """
        try:
            embedding_config = self.config.get("embeddings", {})
            
            storage_config = StorageConfig(
                vector_db_path=self.vector_db_path,
                graph_db_path=self.graph_db_path,
                embedding_dim=embedding_config.get("embedding_dim", 384),
                similarity_threshold=self.config.get("vector_store", {}).get(
                    "similarity_threshold", 0.5
                ),
            )

            store = HybridStore(
                config=storage_config,
                embeddings=self.embeddings,
            )

            self.logger.info("Hybrid Store initialisiert")
            return store

        except Exception as e:
            self.logger.error(f"Fehler beim Initialisieren des Stores: {str(e)}")
            raise

    def initialize_retriever(self) -> HybridRetriever:
        """
        Initialisiere Hybrid Retriever (Vector + Graph Ensemble).

        Returns:
            HybridRetriever Instance
        """
        try:
            rag_config = self.config.get("rag", {})
            
            retrieval_config = RetrievalConfig(
                mode=RetrievalMode(rag_config.get("retrieval_mode", "hybrid")),
                top_k_vector=rag_config.get("top_k_vectors", 5),
                top_k_graph=rag_config.get("top_k_entities", 3),
                vector_weight=rag_config.get("vector_weight", 0.6),
                graph_weight=rag_config.get("graph_weight", 0.4),
                similarity_threshold=self.config.get("vector_store", {}).get(
                    "similarity_threshold", 0.5
                ),
            )

            retriever = HybridRetriever(
                config=retrieval_config,
                hybrid_store=self.hybrid_store,
                embeddings=self.embeddings,
            )

            self.logger.info(f"Hybrid Retriever initialisiert: mode={retrieval_config.mode}")
            return retriever

        except Exception as e:
            self.logger.error(f"Fehler beim Initialisieren des Retrievers: {str(e)}")
            raise

    def run_ingestion_pipeline(self) -> List[Document]:
        """
        Führe Document Ingestion aus.

        Returns:
            Gechunkte Dokumente (nicht nur Count!)

        Raises:
            Exception: Bei Ingestion-Fehler
        """
        try:
            self.logger.info("Starte Document Ingestion...")
            documents = self.ingestion_pipeline.process_documents()
            
            self.logger.info(f"✓ {len(documents)} Dokumente gechunked")
            return documents  # ← RETURN DOCUMENTS, nicht Count!

        except Exception as e:
            self.logger.error(f"Ingestion Pipeline fehlgeschlagen: {str(e)}")
            raise

    def run_storage_pipeline(self, documents: list) -> None:
        """
        Füge Dokumente zu Storage hinzu.

        Args:
            documents: Gechunkte Dokumente
        """
        try:
            self.logger.info("Füge Dokumente zu Hybrid Store hinzu...")
            self.hybrid_store.add_documents(documents)
            self.hybrid_store.save()
            
            self.logger.info("[OK] Dokumente in Storage gespeichert")

        except Exception as e:
            self.logger.error(f"Storage Pipeline fehlgeschlagen: {str(e)}")
            raise

    def retrieve(self, query: str) -> list:
        """
        Führe Hybrid Retrieval durch.

        Args:
            query: Nutzer-Anfrage

        Returns:
            Liste von RetrievalResult-Objekten
        """
        try:
            self.logger.info(f"Retrieval für Query: '{query}'")
            results = self.retriever.retrieve(query)
            
            self.logger.info(f"[OK] {len(results)} Results zurückgegeben")
            return results

        except Exception as e:
            self.logger.error(f"Retrieval fehlgeschlagen: {str(e)}")
            raise

    def setup(self) -> None:
        """
        Führe vollständiges Setup aus: Init all components in dependency order.
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTE EDGE-RAG PIPELINE SETUP")
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
            self.logger.info("[OK] PIPELINE SETUP ERFOLGREICH")
            self.logger.info("=" * 70)

        except Exception as e:
            self.logger.error(f"Setup fehlgeschlagen: {str(e)}")
            raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Main Entry Point für Edge-RAG Pipeline.
    """
    # Setup Logging
    logger = setup_logging()
    logger.info("Edge-RAG Pipeline Start")

    try:
        # Lade Config
        config = load_configuration(Path("./config/settings.yaml"))

        # Initialisiere Pipeline
        pipeline = EdgeRAGPipeline(config, logger)
        pipeline.setup()

        # Führe Ingestion aus (einmal, gibt Dokumente zurück!)
        documents = pipeline.run_ingestion_pipeline()

        if len(documents) > 0:
            # Speichere in Storage (nutze schon geladene Dokumente!)
            pipeline.run_storage_pipeline(documents)

            # Beispiel-Retrieval
            query = "What is the main concept discussed?"
            results = pipeline.retrieve(query)

            logger.info("\n" + "=" * 70)
            logger.info("RETRIEVAL RESULTS")
            logger.info("=" * 70)

            if results:
                for i, result in enumerate(results[:3], 1):
                    logger.info(f"\nResult {i}:")
                    logger.info(f"  Score: {result.relevance_score:.4f}")
                    logger.info(f"  Method: {result.retrieval_method}")
                    logger.info(f"  Text: {result.text[:200]}...")
            else:
                logger.warning("Keine Retrieval Results! Prüfe Vector Store.")
            
            # Print Embedding Metrics
            pipeline.embeddings.print_metrics()

        else:
            logger.warning(
                "Keine Dokumente gefunden. "
                "Bitte platziere PDFs in: ./data/documents"
            )

    except Exception as e:
        logger.critical(f"Pipeline Fehler: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Edge-RAG Pipeline beendet")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()