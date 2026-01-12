"""
Document Ingestion Pipeline for Graph-Augmented Edge-RAG.

Scientific Foundation:
- Recursive Character Chunking reduziert Context Fragmentation und
  maximiert Information Coherence für SLMs (vgl. Gao et al., RAG Survey 2023)
- Chunk Overlap (25%) preserviert semantische Kontinuität über Grenzen hinweg,
  kritisch für Modelle mit reduziertem Context Windows (<4K tokens)
- PDF extraction nutzt PyPDF2 für Konsistenz und Reproduzierbarkeit
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import yaml

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """
    Konfiguration für Recursive Character Chunking.
    
    Wissenschaftliche Begründung:
    Recursive splitting mit hierarchischen Separatoren maximiert
    die semantische Kohärenz von Chunks für Small Language Models.
    """
    chunk_size: int
    chunk_overlap: int
    separators: List[str]


class DocumentIngestionPipeline:
    """
    Robuste Pipeline für PDF-Ingestion und intelligentes Chunking.
    
    Design Pattern: Strategy Pattern (austauschbare Chunking-Strategien)
    Dependency Injection: ChunkingConfig wird injiziert, nicht hardcoded.
    
    Scientific Rationale:
    Recursive Character Chunking mit Overlap ist state-of-the-art für RAG,
    da es sowohl semantische Grenzen (Absätze) als auch Token-Grenzen
    berücksichtigt (vgl. LangChain RAG Best Practices).
    """

    def __init__(
        self,
        chunking_config: ChunkingConfig,
        document_path: Path,
        logger_instance: logging.Logger = logger,
    ):
        """
        Initialisiere die Ingestion Pipeline mit Dependency Injection.

        Args:
            chunking_config: ChunkingConfig mit Chunk-Parametern
            document_path: Pfad zum Dokumentenverzeichnis
            logger_instance: Logger-Instanz für Debugging
        """
        self.chunking_config = chunking_config
        self.document_path = Path(document_path)
        self.logger = logger_instance

        # Validierung
        if not self.document_path.exists():
            raise FileNotFoundError(f"Dokumentenverzeichnis nicht gefunden: {document_path}")

        # Initialisiere Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap,
            separators=chunking_config.separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.logger.info(
            f"DocumentIngestionPipeline initialisiert: "
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
        Teile Dokumente in semantische Chunks mit Overlap.

        Scientific Foundation:
        Recursive Character Chunking mit 25% Overlap preserviert
        Context Boundaries und reduziert "Lost-in-the-Middle" Problem
        für SLMs (vgl. Liu et al., Position Bias in LLMs, 2023).

        Args:
            documents: Liste ungechunkter Dokumente

        Returns:
            Liste gechunkter Dokumente mit Chunk-Metadaten
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        
        # Enrichere Metadaten für Traceability
        for i, chunk in enumerate(chunked_docs):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        self.logger.info(
            f"Dokumenten geckt: {len(documents)} Docs → {len(chunked_docs)} Chunks "
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
        chunk_size=chunking_cfg.get("chunk_size", 512),
        chunk_overlap=chunking_cfg.get("chunk_overlap", 128),
        separators=chunking_cfg.get("separators", ["\n\n", "\n", " ", ""]),
    )