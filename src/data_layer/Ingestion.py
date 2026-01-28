"""
Document Ingestion Pipeline - Konfigurierbare Chunking-Strategien

Version: 3.0.0
Author: Edge-RAG Research Project
Last Modified: 2026-01-28

===============================================================================
OVERVIEW
===============================================================================

Zentrale Ingestion-Pipeline für alle Chunking-Operationen.
Unterstützt verschiedene Strategien die extern konfiguriert werden können.

CHUNKING STRATEGIES:
    1. sentence       - Regex-basiertes Sentence Chunking (schnell, einfach)
    2. sentence_spacy - SpaCy 3-Satz-Fenster (Masterthesis 2.2)
    3. semantic       - Semantische Grenzen mit TF-IDF (SemanticChunker)
    4. fixed          - Feste Zeichenanzahl mit Overlap
    5. recursive      - RecursiveCharacterTextSplitter (LangChain Standard)

===============================================================================
FIXES IN VERSION 3.0.0
===============================================================================

1. Import-Chaos behoben - alle externen Imports in try-except
2. _create_chunker() unterstützt jetzt ALLE Strategien inkl. sentence_spacy
3. Nur EIN `if __name__ == "__main__":` Block mit vollständigen Tests
4. RegexSentenceChunker als Alias für SentenceChunker definiert
5. Bug `config` -> `self.config` in __init__ behoben
6. Input-Validierung für leere Texte hinzugefügt
7. Konsistente Import-Pfade (relativ)
8. entity_extractor statt entity_pipeline

===============================================================================
USAGE
===============================================================================

    from ingestion import DocumentIngestionPipeline, IngestionConfig, create_pipeline
    
    # Quick start
    pipeline = create_pipeline(strategy="sentence_spacy")
    chunks = pipeline.process_text("Your text here...")
    
    # With configuration
    config = IngestionConfig(
        chunking_strategy="semantic",
        chunk_size=1024,
        chunk_overlap=128,
    )
    pipeline = DocumentIngestionPipeline(config)
    documents = pipeline.process_documents(raw_docs)

===============================================================================
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# OPTIONAL DEPENDENCIES - ALL IN TRY-EXCEPT
# ============================================================================

# YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.debug("PyYAML not available, config file loading disabled")

# LangChain support
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter as LCRecursiveSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LCRecursiveSplitter = None
    
    # Mock Document class for standalone usage
    @dataclass
    class Document:
        """Mock Document class when LangChain not available."""
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    logger.debug("LangChain not available, using mock Document class")

# Chunking module support (from chunking.py)
try:
    from chunking import (
        SpacySentenceChunker, 
        create_sentence_chunker, 
        create_semantic_chunker,
        SemanticChunker
    )
    CHUNKING_MODULE_AVAILABLE = True
except ImportError:
    CHUNKING_MODULE_AVAILABLE = False
    SpacySentenceChunker = None
    SemanticChunker = None
    logger.debug("chunking module not available, SpaCy/Semantic chunking disabled")


# ============================================================================
# CONFIGURATION
# ============================================================================

class ChunkingStrategy(Enum):
    """Verfügbare Chunking-Strategien."""
    SENTENCE = "sentence"             # Regex-basiertes N-Sätze pro Chunk
    SENTENCE_SPACY = "sentence_spacy" # SpaCy 3-Satz-Fenster (Masterthesis 2.2)
    SEMANTIC = "semantic"             # Semantische Grenzen mit TF-IDF
    FIXED = "fixed"                   # Feste Zeichenanzahl
    RECURSIVE = "recursive"           # LangChain RecursiveCharacterTextSplitter


@dataclass
class IngestionConfig:
    """
    Konfiguration für Document Ingestion Pipeline.
    
    Attributes:
        chunking_strategy: Welche Strategie verwenden
        chunk_size: Max Chunk-Größe (Zeichen für fixed/recursive/semantic)
        chunk_overlap: Überlappung zwischen Chunks
        min_chunk_size: Minimale Chunk-Größe (kleinere werden gefiltert)
        
        # Sentence-spezifisch
        sentences_per_chunk: Anzahl Sätze pro Chunk
        sentence_overlap: Überlappende Sätze (für sentence_spacy)
        spacy_model: SpaCy Modell Name
        entity_aware_chunking: Entity-aware Boundaries
        
        # Semantic-spezifisch  
        min_lexical_diversity: Minimum Lexical Diversity für Quality Filter
        min_information_density: Minimum Shannon Entropy
        
        # Metadata & Extraction
        extract_entities: Entitäten extrahieren (Pattern-basiert)
        add_source_metadata: Quellinformationen hinzufügen
    """
    # Core Settings
    chunking_strategy: str = "sentence"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_size: int = 50
    
    # Sentence Strategy
    sentences_per_chunk: int = 3
    sentence_overlap: int = 1
    spacy_model: str = "en_core_web_sm"
    entity_aware_chunking: bool = False

    # Semantic Strategy
    min_lexical_diversity: float = 0.3
    min_information_density: float = 2.0
    
    # Metadata Enrichment
    extract_entities: bool = False  # Default False für Performance
    add_source_metadata: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        valid_strategies = [s.value for s in ChunkingStrategy]
        if self.chunking_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid chunking_strategy: '{self.chunking_strategy}'. "
                f"Must be one of: {valid_strategies}"
            )
        
        if self.chunk_size < 50:
            raise ValueError(f"chunk_size must be >= 50, got {self.chunk_size}")
        
        if self.sentences_per_chunk < 1:
            raise ValueError(f"sentences_per_chunk must be >= 1, got {self.sentences_per_chunk}")
        
        if self.sentence_overlap >= self.sentences_per_chunk:
            raise ValueError(
                f"sentence_overlap ({self.sentence_overlap}) must be < "
                f"sentences_per_chunk ({self.sentences_per_chunk})"
            )


def load_ingestion_config(config_path: Union[str, Path] = None) -> IngestionConfig:
    """
    Load ingestion configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses defaults.
        
    Returns:
        IngestionConfig instance
    """
    if config_path is None:
        logger.info("No config path provided, using defaults")
        return IngestionConfig()
    
    if not YAML_AVAILABLE:
        logger.warning("PyYAML not installed, using defaults")
        return IngestionConfig()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return IngestionConfig()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract ingestion/chunking section if present
    if "ingestion" in config_dict:
        config_dict = config_dict["ingestion"]
    elif "chunking" in config_dict:
        config_dict = config_dict["chunking"]
    
    # Filter to only valid fields
    valid_fields = {f.name for f in IngestionConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    
    logger.info(f"Loaded ingestion config from: {config_path}")
    return IngestionConfig(**filtered_dict)


# ============================================================================
# CHUNKING IMPLEMENTATIONS
# ============================================================================

class SentenceChunker:
    """
    Regex-based Sentence Chunking.
    
    Gruppiert N Sätze pro Chunk. Schnell und einfach.
    Gut für strukturierte Dokumente (Papers, Artikel).
    """
    
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def __init__(self, sentences_per_chunk: int = 3, min_chunk_size: int = 50):
        self.sentences_per_chunk = sentences_per_chunk
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        if not text or not text.strip():
            return []
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text by grouping sentences.
        
        Args:
            text: Input text
            metadata: Base metadata to include
            
        Returns:
            List of chunk dicts with 'text' and 'metadata'
        """
        metadata = metadata or {}
        
        # FIX: Input validation
        if not text or not text.strip():
            self.logger.debug("Empty text provided to SentenceChunker")
            return []
        
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            self.logger.debug("No sentences found in text")
            return []
        
        chunks = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk_sents = sentences[i:i + self.sentences_per_chunk]
            chunk_text = " ".join(chunk_sents)
            
            # Skip short chunks
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "sentence_start": i,
                "sentence_end": i + len(chunk_sents),
                "sentence_count": len(chunk_sents),
                "chunk_method": "sentence_regex",
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
            })
        
        return chunks


# FIX: Alias für Backwards-Kompatibilität (war in Tests referenziert aber nicht definiert)
RegexSentenceChunker = SentenceChunker


class FixedSizeChunker:
    """
    Fixed-size Chunking mit Overlap.
    
    Einfachste Methode, versucht aber an Wortgrenzen zu brechen.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1024, 
        chunk_overlap: int = 128,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk text into fixed-size pieces with overlap."""
        metadata = metadata or {}
        
        # FIX: Input validation
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size * 0.8:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space
            
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "char_start": start,
                    "char_end": end,
                    "chunk_index": chunk_id,
                    "chunk_method": "fixed",
                })
                
                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": chunk_metadata,
                })
                chunk_id += 1
            
            # Move forward with overlap
            start = end - self.chunk_overlap
            if start >= len(text) - self.min_chunk_size:
                break
        
        return chunks


class RecursiveChunker:
    """
    Wrapper für RecursiveCharacterTextSplitter.
    
    Verwendet LangChain wenn verfügbar, sonst einfacher Fallback.
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._splitter = None
    
    def _get_splitter(self):
        """Lazy initialization of splitter."""
        if self._splitter is None:
            if LANGCHAIN_AVAILABLE and LCRecursiveSplitter is not None:
                self._splitter = LCRecursiveSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            else:
                # Simple fallback splitter
                self._splitter = self._fallback_split
        return self._splitter
    
    def _fallback_split(self, text: str) -> List[str]:
        """Simple fallback when LangChain not available."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk using RecursiveCharacterTextSplitter or fallback."""
        metadata = metadata or {}
        
        if not text or not text.strip():
            return []
        
        splitter = self._get_splitter()
        
        # Handle both LangChain splitter and fallback
        if callable(splitter) and splitter == self._fallback_split:
            texts = splitter(text)
        else:
            texts = splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(texts):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_method": "recursive",
            })
            
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": chunk_metadata,
            })
        
        return chunks


class SpacySentenceChunkerWrapper:
    """
    Wrapper für SpaCy-based Sentence Chunker aus chunking.py.
    
    Implementiert 3-Satz-Fenster gemäß Masterthesis Abschnitt 2.2.
    Fällt auf RegexSentenceChunker zurück wenn SpaCy nicht verfügbar.
    """
    
    def __init__(
        self,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 1,
        min_chunk_size: int = 50,
        spacy_model: str = "en_core_web_sm",
        entity_aware: bool = False,
    ):
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.min_chunk_size = min_chunk_size
        self.spacy_model = spacy_model
        self.entity_aware = entity_aware
        self._chunker = None
        self._using_spacy = False
    
    def _get_chunker(self):
        """Lazy initialization of SpaCy chunker with fallback."""
        if self._chunker is None:
            if CHUNKING_MODULE_AVAILABLE and SpacySentenceChunker is not None:
                try:
                    self._chunker = create_sentence_chunker(
                        sentences_per_chunk=self.sentences_per_chunk,
                        sentence_overlap=self.sentence_overlap,
                        spacy_model=self.spacy_model,
                        entity_aware=self.entity_aware,
                        min_chunk_chars=self.min_chunk_size,
                    )
                    self._using_spacy = True
                    logger.debug("Using SpaCy sentence chunker")
                except Exception as e:
                    logger.warning(f"SpaCy chunker init failed: {e}, using regex fallback")
                    self._chunker = SentenceChunker(
                        sentences_per_chunk=self.sentences_per_chunk,
                        min_chunk_size=self.min_chunk_size,
                    )
            else:
                logger.debug("chunking module not available, using regex fallback")
                self._chunker = SentenceChunker(
                    sentences_per_chunk=self.sentences_per_chunk,
                    min_chunk_size=self.min_chunk_size,
                )
        return self._chunker
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk using SpaCy sentence segmentation or regex fallback."""
        metadata = metadata or {}
        
        if not text or not text.strip():
            return []
        
        chunker = self._get_chunker()
        return chunker.chunk(text, metadata)
    
    @property
    def using_spacy(self) -> bool:
        """Check if actually using SpaCy or fallback."""
        self._get_chunker()  # Ensure initialized
        return self._using_spacy


class SemanticChunkerWrapper:
    """
    Wrapper für SemanticChunker aus chunking.py.
    
    Nutzt semantische Grenzen, TF-IDF und Quality Filtering.
    Fällt auf RecursiveChunker zurück wenn nicht verfügbar.
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        min_chunk_size: int = 200,
        min_lexical_diversity: float = 0.3,
        min_information_density: float = 2.0,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.min_lexical_diversity = min_lexical_diversity
        self.min_information_density = min_information_density
        self._chunker = None
        self._using_semantic = False
    
    def _get_chunker(self):
        """Lazy initialization of semantic chunker with fallback."""
        if self._chunker is None:
            if CHUNKING_MODULE_AVAILABLE and SemanticChunker is not None:
                try:
                    self._chunker = create_semantic_chunker(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        min_chunk_size=self.min_chunk_size,
                    )
                    self._using_semantic = True
                    logger.debug("Using semantic chunker")
                except Exception as e:
                    logger.warning(f"Semantic chunker init failed: {e}, using recursive fallback")
                    self._chunker = None
            
            if self._chunker is None:
                logger.debug("Using recursive fallback for semantic strategy")
                self._chunker = RecursiveChunker(
                    self.chunk_size, self.chunk_overlap, self.min_chunk_size
                )
        return self._chunker
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk using semantic boundaries or fallback."""
        metadata = metadata or {}
        
        if not text or not text.strip():
            return []
        
        chunker = self._get_chunker()
        
        # If using actual SemanticChunker, need to wrap in Document
        if self._using_semantic:
            temp_doc = Document(page_content=text, metadata=metadata)
            try:
                result_docs = chunker.chunk_document(temp_doc)
                chunks = []
                for doc in result_docs:
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata["chunk_method"] = "semantic"
                    chunks.append({
                        "text": doc.page_content,
                        "metadata": chunk_metadata,
                    })
                return chunks
            except Exception as e:
                logger.warning(f"Semantic chunking failed: {e}")
                # Fall through to recursive
                fallback = RecursiveChunker(
                    self.chunk_size, self.chunk_overlap, self.min_chunk_size
                )
                return fallback.chunk(text, metadata)
        else:
            # Using fallback RecursiveChunker
            return chunker.chunk(text, metadata)


# ============================================================================
# ENTITY EXTRACTION (Simple Pattern-based)
# ============================================================================

class SimpleEntityExtractor:
    """
    Einfache Entity-Extraktion basierend auf Patterns.
    
    Extrahiert Named Entities für Knowledge Graph Konstruktion.
    Für produktiven Einsatz sollte SpaCy NER verwendet werden.
    """
    
    ENTITY_PATTERNS = [
        r'"([^"]+)"',                              # Quoted strings
        r"'([^']+)'",                              # Single quoted
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",  # Multi-word proper nouns
        r"\b([A-Z][a-z]{2,})\b",                   # Single proper nouns (min 3 chars)
    ]
    
    STOPWORDS = {
        'The', 'This', 'That', 'These', 'Those', 'There', 'Here',
        'However', 'Therefore', 'Furthermore', 'Moreover', 'Although',
        'Because', 'Since', 'While', 'When', 'Where', 'After', 'Before',
        'Also', 'Just', 'Only', 'Even', 'Still', 'Already', 'Always',
    }
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p) for p in self.ENTITY_PATTERNS]
    
    def extract(self, text: str) -> List[str]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of unique entity strings
        """
        if not text:
            return []
        
        entities = set()
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                entity = match.strip()
                if len(entity) > 2 and entity not in self.STOPWORDS:
                    entities.add(entity)
        
        return sorted(list(entities))


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class DocumentIngestionPipeline:
    """
    Zentrale Document Ingestion Pipeline.
    
    Orchestriert Chunking, Entity Extraction, und Metadata Enrichment.
    """
    
    def __init__(self, config: IngestionConfig = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: IngestionConfig instance. Uses defaults if None.
        """
        self.config = config or IngestionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize chunker based on strategy
        self.chunker = self._create_chunker()
        
        # FIX: War `config.extract_entities` - muss `self.config` sein!
        self.entity_extractor = SimpleEntityExtractor() if self.config.extract_entities else None
        
        self.logger.info(
            f"DocumentIngestionPipeline initialized: "
            f"strategy={self.config.chunking_strategy}, "
            f"chunk_size={self.config.chunk_size}"
        )
    
    def _create_chunker(self):
        """
        Create chunker based on configuration.
        
        FIX: Unterstützt jetzt ALLE Strategien inkl. sentence_spacy
        """
        strategy = self.config.chunking_strategy
        
        if strategy == ChunkingStrategy.SENTENCE.value:
            return SentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
                min_chunk_size=self.config.min_chunk_size,
            )
        
        elif strategy == ChunkingStrategy.SENTENCE_SPACY.value:
            # FIX: War in _create_chunker_extended aber nie aufgerufen
            return SpacySentenceChunkerWrapper(
                sentences_per_chunk=self.config.sentences_per_chunk,
                sentence_overlap=self.config.sentence_overlap,
                min_chunk_size=self.config.min_chunk_size,
                spacy_model=self.config.spacy_model,
                entity_aware=self.config.entity_aware_chunking,
            )
        
        elif strategy == ChunkingStrategy.SEMANTIC.value:
            return SemanticChunkerWrapper(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
                min_lexical_diversity=self.config.min_lexical_diversity,
                min_information_density=self.config.min_information_density,
            )
        
        elif strategy == ChunkingStrategy.FIXED.value:
            return FixedSizeChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
            )
        
        elif strategy == ChunkingStrategy.RECURSIVE.value:
            return RecursiveChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
            )
        
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def process_text(
        self,
        text: str,
        metadata: Dict = None,
        source_id: str = None,
    ) -> List[Dict]:
        """
        Process single text into chunks.
        
        Args:
            text: Raw text content
            metadata: Optional base metadata
            source_id: Optional source identifier
            
        Returns:
            List of chunk dicts with 'text', 'metadata', optionally 'entities'
        """
        metadata = metadata or {}
        
        # FIX: Input validation
        if not text or not text.strip():
            self.logger.debug("Empty text provided to pipeline")
            return []
        
        if source_id and self.config.add_source_metadata:
            metadata["source_id"] = source_id
        
        # Chunk text
        chunks = self.chunker.chunk(text, metadata)
        
        # Enrich with entities
        if self.entity_extractor:
            for chunk in chunks:
                entities = self.entity_extractor.extract(chunk["text"])
                chunk["entities"] = entities
                chunk["metadata"]["entity_count"] = len(entities)
        
        return chunks
    
    def process_texts(
        self,
        texts: List[str],
        metadatas: List[Dict] = None,
        source_ids: List[str] = None,
    ) -> List[Dict]:
        """
        Process multiple texts into chunks.
        
        Args:
            texts: List of raw texts
            metadatas: Optional list of metadata dicts
            source_ids: Optional list of source identifiers
            
        Returns:
            List of all chunks from all texts
        """
        if not texts:
            return []
        
        metadatas = metadatas or [{}] * len(texts)
        source_ids = source_ids or [None] * len(texts)
        
        all_chunks = []
        chunk_id = 0
        
        for text, meta, src_id in zip(texts, metadatas, source_ids):
            chunks = self.process_text(text, meta, src_id)
            
            # Add global chunk IDs
            for chunk in chunks:
                chunk["metadata"]["global_chunk_id"] = chunk_id
                chunk_id += 1
            
            all_chunks.extend(chunks)
        
        self.logger.info(
            f"Processed {len(texts)} texts → {len(all_chunks)} chunks "
            f"(strategy: {self.config.chunking_strategy})"
        )
        
        return all_chunks
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process Document objects.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document objects (chunked)
        """
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        chunks = self.process_texts(texts, metadatas)
        
        # Convert back to Documents
        result_docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"],
            )
            if "entities" in chunk:
                doc.metadata["entities"] = chunk["entities"]
            result_docs.append(doc)
        
        return result_docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "chunking_strategy": self.config.chunking_strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "sentences_per_chunk": self.config.sentences_per_chunk,
            "entity_extraction": self.config.extract_entities,
            "chunking_module_available": CHUNKING_MODULE_AVAILABLE,
            "langchain_available": LANGCHAIN_AVAILABLE,
        }
        
        # Add SpaCy status if relevant
        if hasattr(self.chunker, 'using_spacy'):
            stats["using_spacy"] = self.chunker.using_spacy
        
        return stats


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_pipeline(
    strategy: str = "sentence",
    chunk_size: int = 1024,
    sentences_per_chunk: int = 3,
    **kwargs
) -> DocumentIngestionPipeline:
    """
    Factory function für schnelle Pipeline-Erstellung.
    
    Args:
        strategy: "sentence", "sentence_spacy", "semantic", "fixed", or "recursive"
        chunk_size: Max chunk size
        sentences_per_chunk: Sentences per chunk (for sentence strategies)
        **kwargs: Additional config options
        
    Returns:
        Configured DocumentIngestionPipeline
        
    Example:
        pipeline = create_pipeline(strategy="sentence_spacy", sentences_per_chunk=3)
        chunks = pipeline.process_text("Your text here...")
    """
    config = IngestionConfig(
        chunking_strategy=strategy,
        chunk_size=chunk_size,
        sentences_per_chunk=sentences_per_chunk,
        **kwargs,
    )
    return DocumentIngestionPipeline(config)


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """
    Comprehensive tests for the ingestion pipeline.
    
    Tests all chunking strategies and validates output format.
    """
    print("\n" + "=" * 70)
    print("INGESTION PIPELINE TESTS (v3.0.0)")
    print("=" * 70)
    
    # Test texts
    short_text = "First sentence here. Second sentence follows. Third one too."
    
    long_text = """
    Albert Einstein was a German-born theoretical physicist. He developed the 
    theory of relativity, one of the two pillars of modern physics. His work 
    is also known for its influence on the philosophy of science.
    
    Einstein received the Nobel Prize in Physics in 1921. He was awarded for 
    his explanation of the photoelectric effect. Einstein published more than 
    300 scientific papers during his career.
    
    In 1905, Einstein published four groundbreaking papers. This year is 
    sometimes called his "miracle year" by physicists. The papers covered the 
    photoelectric effect, Brownian motion, special relativity, and mass-energy 
    equivalence. These contributions revolutionized our understanding of physics.
    """
    
    all_tests_passed = True
    
    # -------------------------------------------------------------------------
    # TEST 1: Configuration Validation
    # -------------------------------------------------------------------------
    print("\n--- Test 1: Configuration Validation ---")
    try:
        # Valid config
        config = IngestionConfig(chunking_strategy="sentence", chunk_size=500)
        assert config.chunking_strategy == "sentence"
        print(f"  ✓ Valid config created")
        
        # Invalid strategy should raise
        try:
            IngestionConfig(chunking_strategy="invalid_strategy")
            print(f"  ✗ Should have raised ValueError")
            all_tests_passed = False
        except ValueError:
            print(f"  ✓ Invalid strategy correctly rejected")
        
        # Invalid sentence_overlap should raise
        try:
            IngestionConfig(sentences_per_chunk=3, sentence_overlap=5)
            print(f"  ✗ Should have raised ValueError for overlap >= sentences")
            all_tests_passed = False
        except ValueError:
            print(f"  ✓ Invalid overlap correctly rejected")
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 2: Sentence Strategy (Regex)
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Sentence Strategy (Regex) ---")
    try:
        pipeline = create_pipeline(strategy="sentence", sentences_per_chunk=2, min_chunk_size=20)
        chunks = pipeline.process_text(long_text, {"source": "test"})
        
        assert len(chunks) > 0, "Should produce chunks"
        assert all("text" in c and "metadata" in c for c in chunks), "Should have text and metadata"
        assert all("chunk_method" in c["metadata"] for c in chunks), "Should have chunk_method"
        assert chunks[0]["metadata"]["chunk_method"] == "sentence_regex"
        
        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"  ✓ Chunk method: {chunks[0]['metadata']['chunk_method']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 3: Sentence SpaCy Strategy
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Sentence SpaCy Strategy ---")
    try:
        pipeline = create_pipeline(strategy="sentence_spacy", sentences_per_chunk=3, min_chunk_size=20)
        chunks = pipeline.process_text(long_text, {"source": "test"})
        
        assert len(chunks) > 0, "Should produce chunks"
        
        stats = pipeline.get_stats()
        using_spacy = stats.get("using_spacy", False)
        
        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"  ✓ Using SpaCy: {using_spacy} (fallback if False)")
        print(f"  ✓ Chunking module available: {CHUNKING_MODULE_AVAILABLE}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 4: Fixed Size Strategy
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Fixed Size Strategy ---")
    try:
        pipeline = create_pipeline(strategy="fixed", chunk_size=200, chunk_overlap=50, min_chunk_size=20)
        chunks = pipeline.process_text(long_text, {"source": "test"})
        
        assert len(chunks) > 0, "Should produce chunks"
        assert all(c["metadata"]["chunk_method"] == "fixed" for c in chunks)
        
        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"  ✓ All chunks have 'fixed' method")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 5: Recursive Strategy
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Recursive Strategy ---")
    try:
        pipeline = create_pipeline(strategy="recursive", chunk_size=200, chunk_overlap=50, min_chunk_size=20)
        chunks = pipeline.process_text(long_text, {"source": "test"})
        
        assert len(chunks) > 0, "Should produce chunks"
        
        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"  ✓ LangChain available: {LANGCHAIN_AVAILABLE}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 6: Semantic Strategy
    # -------------------------------------------------------------------------
    print("\n--- Test 6: Semantic Strategy ---")
    try:
        pipeline = create_pipeline(strategy="semantic", chunk_size=300, min_chunk_size=50)
        chunks = pipeline.process_text(long_text, {"source": "test"})
        
        assert len(chunks) > 0, "Should produce chunks"
        
        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"  ✓ Semantic chunking available: {CHUNKING_MODULE_AVAILABLE}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 7: Entity Extraction
    # -------------------------------------------------------------------------
    print("\n--- Test 7: Entity Extraction ---")
    try:
        pipeline = create_pipeline(strategy="sentence", extract_entities=True, min_chunk_size=20)
        chunks = pipeline.process_text(long_text, {"source": "test"})
        
        assert len(chunks) > 0, "Should produce chunks"
        assert all("entities" in c for c in chunks), "Should have entities"
        
        total_entities = sum(len(c["entities"]) for c in chunks)
        all_entities = []
        for c in chunks:
            all_entities.extend(c["entities"])
        
        print(f"  ✓ Extracted {total_entities} entities across {len(chunks)} chunks")
        
        # Check Einstein is found
        einstein_found = any("Einstein" in e for e in all_entities)
        if einstein_found:
            print(f"  ✓ 'Einstein' correctly extracted")
        else:
            print(f"  ⚠ 'Einstein' not found (entities: {all_entities[:5]}...)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 8: Empty Input Handling
    # -------------------------------------------------------------------------
    print("\n--- Test 8: Empty Input Handling ---")
    try:
        pipeline = create_pipeline(strategy="sentence")
        
        # Empty string
        chunks = pipeline.process_text("", {})
        assert chunks == [], "Should return empty list for empty string"
        
        # Whitespace only
        chunks = pipeline.process_text("   \n\t  ", {})
        assert chunks == [], "Should return empty list for whitespace"
        
        # Empty list of texts
        chunks = pipeline.process_texts([], [])
        assert chunks == [], "Should return empty list for empty input"
        
        # None documents
        docs = pipeline.process_documents([])
        assert docs == [], "Should return empty list for no documents"
        
        print(f"  ✓ Empty inputs handled correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 9: Multiple Texts Processing
    # -------------------------------------------------------------------------
    print("\n--- Test 9: Multiple Texts Processing ---")
    try:
        pipeline = create_pipeline(strategy="sentence", sentences_per_chunk=2, min_chunk_size=20)
        
        texts = [
            "First document has sentences. It contains information.",
            "Second document is different. It has other content. More sentences here.",
        ]
        metadatas = [{"doc_id": 1}, {"doc_id": 2}]
        
        chunks = pipeline.process_texts(texts, metadatas)
        
        assert len(chunks) > 0, "Should produce chunks"
        assert all("global_chunk_id" in c["metadata"] for c in chunks), "Should have global IDs"
        
        # Verify global IDs are sequential
        global_ids = [c["metadata"]["global_chunk_id"] for c in chunks]
        assert global_ids == list(range(len(chunks))), "Global IDs should be sequential"
        
        print(f"  ✓ Processed 2 texts → {len(chunks)} chunks with sequential IDs")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 10: Document Processing
    # -------------------------------------------------------------------------
    print("\n--- Test 10: Document Processing ---")
    try:
        pipeline = create_pipeline(strategy="sentence", sentences_per_chunk=2, min_chunk_size=20)
        
        docs = [
            Document(page_content=long_text, metadata={"source": "einstein.txt"}),
        ]
        
        result_docs = pipeline.process_documents(docs)
        
        assert len(result_docs) > 0, "Should produce documents"
        assert all(isinstance(d, Document) for d in result_docs), "Should return Document objects"
        assert all(hasattr(d, 'page_content') and hasattr(d, 'metadata') for d in result_docs)
        
        print(f"  ✓ Processed 1 Document → {len(result_docs)} Document chunks")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 11: Pipeline Stats
    # -------------------------------------------------------------------------
    print("\n--- Test 11: Pipeline Stats ---")
    try:
        pipeline = create_pipeline(strategy="sentence_spacy", chunk_size=500)
        stats = pipeline.get_stats()
        
        assert "chunking_strategy" in stats
        assert "chunk_size" in stats
        assert stats["chunking_strategy"] == "sentence_spacy"
        
        print(f"  ✓ Stats keys: {list(stats.keys())}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 12: RegexSentenceChunker Alias
    # -------------------------------------------------------------------------
    print("\n--- Test 12: RegexSentenceChunker Alias ---")
    try:
        # FIX: Both should work now
        chunker1 = SentenceChunker(sentences_per_chunk=3)
        chunker2 = RegexSentenceChunker(sentences_per_chunk=3)
        
        assert type(chunker1) == type(chunker2), "Should be same type"
        
        # Both should produce same results
        result1 = chunker1.chunk(short_text, {})
        result2 = chunker2.chunk(short_text, {})
        
        assert len(result1) == len(result2), "Should produce same number of chunks"
        
        print(f"  ✓ RegexSentenceChunker alias works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # TEST 13: 3-Sentence Window Logic
    # -------------------------------------------------------------------------
    print("\n--- Test 13: 3-Sentence Window Logic ---")
    try:
        # 6 sentences should produce 2 chunks with 3 sentences each
        six_sentences = (
            "Sentence one is here. Sentence two follows next. Sentence three ends first group. "
            "Sentence four starts new group. Sentence five continues. Sentence six ends."
        )
        
        chunker = SentenceChunker(sentences_per_chunk=3, min_chunk_size=10)
        chunks = chunker.chunk(six_sentences, {})
        
        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        assert "Sentence one" in chunks[0]["text"]
        assert "Sentence four" in chunks[1]["text"]
        
        print(f"  ✓ 3-sentence window correctly splits 6 sentences into 2 chunks")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_tests_passed = False
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    
    return all_tests_passed


def demonstrate_pipeline():
    """Demonstrate the ingestion pipeline with different strategies."""
    print("\n" + "=" * 70)
    print("INGESTION PIPELINE DEMONSTRATION")
    print("=" * 70)
    
    test_text = """
    Albert Einstein was a German-born theoretical physicist. He developed the 
    theory of relativity, one of the two pillars of modern physics. His work 
    is also known for its influence on the philosophy of science.
    
    Einstein received the Nobel Prize in Physics in 1921. He was awarded for 
    his explanation of the photoelectric effect. Einstein published more than 
    300 scientific papers during his career.
    """
    
    strategies = ["sentence", "sentence_spacy", "fixed", "recursive", "semantic"]
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        
        try:
            pipeline = create_pipeline(
                strategy=strategy,
                chunk_size=250,
                sentences_per_chunk=2,
                min_chunk_size=30,
            )
            
            chunks = pipeline.process_text(test_text, {"source": "demo"})
            
            print(f"  Chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks[:2]):  # Show first 2
                preview = chunk["text"][:50].replace("\n", " ") + "..."
                method = chunk["metadata"].get("chunk_method", "unknown")
                print(f"    [{i}] {len(chunk['text'])} chars ({method}): \"{preview}\"")
            if len(chunks) > 2:
                print(f"    ... and {len(chunks) - 2} more")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN - SINGLE ENTRY POINT (FIX: war doppelt)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    tests_passed = run_tests()
    
    # Run demonstration
    print("\n")
    demonstrate_pipeline()
    
    # Exit with appropriate code
    exit(0 if tests_passed else 1)