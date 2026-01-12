"""
Document Ingestion Pipeline mit Semantic Chunking (UPDATED).

Neu:
- Semantic Chunking statt fester Zeichen-Splits
- Header-based Metadata Extraction
- Context-Aware Quality Filtering
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import yaml

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# Import semantic chunking
from src.semantic_chunking import SemanticChunker, create_semantic_chunker


logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """
    Konfiguration für Chunking.
    
    Unterstützt beide Modi:
    - standard: Recursive Character Chunking
    - semantic: Intelligent Semantic Chunking
    """
    mode: str = "semantic"  # "standard" or "semantic"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_size: int = 200
    separators: List[str] = None


class DocumentIngestionPipeline:
    """
    Robuste Pipeline mit Semantic Chunking Support.
    
    Neu in dieser Version:
    - Semantic Chunking als Default
    - Header Metadata Extraction
    - Context-Aware Filtering
    - Fallback auf Standard Chunking bei Fehlern
    """

    def __init__(
        self,
        chunking_config: ChunkingConfig,
        document_path: Path,
        logger_instance: logging.Logger = logger,
    ):
        """
        Initialisiere Pipeline.

        Args:
            chunking_config: ChunkingConfig mit Chunk-Parametern
            document_path: Pfad zum Dokumentenverzeichnis
            logger_instance: Logger-Instanz
        """
        self.chunking_config = chunking_config
        self.document_path = Path(document_path)
        self.logger = logger_instance

        # Validierung
        if not self.document_path.exists():
            raise FileNotFoundError(f"Dokumentenverzeichnis nicht gefunden: {document_path}")

        # Initialize appropriate chunker
        if chunking_config.mode == "semantic":
            self.logger.info("Using SEMANTIC chunking mode")
            self.semantic_chunker = create_semantic_chunker(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap,
                min_chunk_size=chunking_config.min_chunk_size,
            )
            self.text_splitter = None  # Not used in semantic mode
            
        else:  # standard mode
            self.logger.info("Using STANDARD chunking mode")
            self.semantic_chunker = None
            
            separators = chunking_config.separators or ["\n\n", "\n", ". ", " ", ""]
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False,
            )
        
        self.logger.info(
            f"DocumentIngestionPipeline initialisiert: "
            f"mode={chunking_config.mode}, "
            f"chunk_size={chunking_config.chunk_size}, "
            f"overlap={chunking_config.chunk_overlap}"
        )

    def load_pdf_documents(self, pdf_pattern: str = "*.pdf") -> List[Document]:
        """
        Lade alle PDF-Dateien aus dem Dokumentenverzeichnis.

        Args:
            pdf_pattern: Glob-Pattern für PDF-Dateien

        Returns:
            Liste von LangChain Document-Objekten

        Raises:
            FileNotFoundError: Falls keine PDFs gefunden werden
        """
        pdf_files = list(self.document_path.glob(pdf_pattern))
        
        if not pdf_files:
            raise FileNotFoundError(f"Keine PDF-Dateien gefunden in {self.document_path}")

        documents = []
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Lade PDF: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # Metadaten enrichment
                for doc in docs:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["source_path"] = str(pdf_file.absolute())
                
                documents.extend(docs)
                self.logger.debug(f"Erfolgreich geladen: {pdf_file.name} ({len(docs)} Seiten)")
                
            except Exception as e:
                self.logger.error(f"Fehler beim Laden von {pdf_file.name}: {str(e)}")
                continue

        self.logger.info(f"Insgesamt {len(documents)} PDF-Seiten geladen")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Teile Dokumente in Chunks (semantic oder standard).

        Args:
            documents: Liste ungechunkter Dokumente

        Returns:
            Liste gechunkter Dokumente mit enriched Metadaten
        """
        if self.chunking_config.mode == "semantic":
            return self._chunk_documents_semantic(documents)
        else:
            return self._chunk_documents_standard(documents)
    
    def _chunk_documents_semantic(self, documents: List[Document]) -> List[Document]:
        """
        Semantic Chunking mit Header Extraction und Quality Filtering.
        
        Args:
            documents: Liste von Dokumenten (pages)
            
        Returns:
            Liste semantisch gechunkter Dokumente
        """
        all_chunks = []
        
        for doc in documents:
            try:
                # Semantic chunking per page
                page_chunks = self.semantic_chunker.chunk_document(doc)
                all_chunks.extend(page_chunks)
                
            except Exception as e:
                self.logger.warning(
                    f"Semantic chunking failed for page, using fallback: {e}"
                )
                # Fallback to standard chunking for this page
                if self.text_splitter is None:
                    # Create temporary standard splitter
                    fallback_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunking_config.chunk_size,
                        chunk_overlap=self.chunking_config.chunk_overlap,
                    )
                    fallback_chunks = fallback_splitter.split_documents([doc])
                else:
                    fallback_chunks = self.text_splitter.split_documents([doc])
                
                all_chunks.extend(fallback_chunks)
        
        self.logger.info(
            f"Semantic chunking complete: {len(documents)} docs → {len(all_chunks)} chunks"
        )
        
        return all_chunks
    
    def _chunk_documents_standard(self, documents: List[Document]) -> List[Document]:
        """
        Standard Recursive Character Chunking (Original-Methode).
        
        Args:
            documents: Liste von Dokumenten
            
        Returns:
            Liste gechunkter Dokumente
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        
        # Enrichere Metadaten für Traceability
        for i, chunk in enumerate(chunked_docs):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunking_method"] = "standard"
        
        self.logger.info(
            f"Standard chunking: {len(documents)} docs → {len(chunked_docs)} chunks "
            f"(avg chunk size: {sum(len(d.page_content) for d in chunked_docs) // len(chunked_docs)} chars)"
        )
        
        return chunked_docs

    def process_documents(self) -> List[Document]:
        """
        End-to-End Pipeline: Lade → Chunk → Enrichiere.

        Returns:
            Vollständig verarbeitete und gechunkte Dokumente
        """
        documents = self.load_pdf_documents()
        chunked_documents = self.chunk_documents(documents)
        return chunked_documents


def load_ingestion_config(config_path: Path) -> ChunkingConfig:
    """
    Lade Chunking-Konfiguration aus YAML.

    Args:
        config_path: Pfad zur settings.yaml

    Returns:
        ChunkingConfig-Objekt

    Raises:
        FileNotFoundError: Falls Config nicht existiert
        KeyError: Falls erforderliche Keys fehlen
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    chunking_cfg = config.get("chunking", {})
    
    return ChunkingConfig(
        mode=chunking_cfg.get("mode", "semantic"),  # Default: semantic
        chunk_size=chunking_cfg.get("chunk_size", 1024),
        chunk_overlap=chunking_cfg.get("chunk_overlap", 128),
        min_chunk_size=chunking_cfg.get("min_chunk_size", 200),
        separators=chunking_cfg.get("separators", ["\n\n", "\n", ". ", " ", ""]),
    )