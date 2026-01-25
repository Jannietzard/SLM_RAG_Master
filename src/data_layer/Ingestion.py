"""
Document Ingestion Pipeline - Konfigurierbare Chunking-Strategien

Version: 2.1.0
Author: Edge-RAG Research Project

===============================================================================
OVERVIEW
===============================================================================

Zentrale Ingestion-Pipeline für alle Chunking-Operationen.
Unterstützt verschiedene Strategien die extern konfiguriert werden können.

CHUNKING STRATEGIES:
    1. sentence   - Gruppiert N Sätze pro Chunk (schnell, einfach)
    2. semantic   - Semantische Grenzen (SemanticChunker)
    3. fixed      - Feste Zeichenanzahl mit Overlap
    4. recursive  - RecursiveCharacterTextSplitter (LangChain Standard)

USAGE:
    from src.data_layer.ingestion import DocumentIngestionPipeline, IngestionConfig
    
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

import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class ChunkingStrategy(Enum):
    """Verfügbare Chunking-Strategien."""
    SENTENCE = "sentence"       # N Sätze pro Chunk
    SEMANTIC = "semantic"       # Semantische Grenzen
    FIXED = "fixed"             # Feste Zeichenanzahl
    RECURSIVE = "recursive"     # LangChain RecursiveCharacterTextSplitter


@dataclass
class IngestionConfig:
    """
    Konfiguration für Document Ingestion Pipeline.
    
    Attributes:
        chunking_strategy: Welche Strategie verwenden
        chunk_size: Max Chunk-Größe (Zeichen für fixed/recursive, Sätze für sentence)
        chunk_overlap: Überlappung zwischen Chunks
        min_chunk_size: Minimale Chunk-Größe (kleinere werden gefiltert)
        
        # Sentence-spezifisch
        sentences_per_chunk: Anzahl Sätze pro Chunk (nur für sentence strategy)
        
        # Semantic-spezifisch  
        min_lexical_diversity: Minimum Lexical Diversity für Quality Filter
        min_information_density: Minimum Shannon Entropy
        
        # Quality Filtering
        filter_short_chunks: Kurze Chunks entfernen
        filter_low_quality: Low-quality Chunks entfernen
        
        # Metadata
        extract_entities: Entitäten extrahieren für Graph
        add_source_metadata: Quellinformationen hinzufügen
    """
    # Core Settings
    chunking_strategy: str = "sentence"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_size: int = 50
    
    # Sentence Strategy
    sentences_per_chunk: int = 3
    
    # Semantic Strategy
    min_lexical_diversity: float = 0.3
    min_information_density: float = 2.0
    
    # Quality Filtering
    filter_short_chunks: bool = True
    filter_low_quality: bool = False  # Nur für semantic
    
    # Metadata Enrichment
    extract_entities: bool = True
    add_source_metadata: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        valid_strategies = [s.value for s in ChunkingStrategy]
        if self.chunking_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid chunking_strategy: {self.chunking_strategy}. "
                f"Must be one of: {valid_strategies}"
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
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return IngestionConfig()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract ingestion section if present
    if "ingestion" in config_dict:
        config_dict = config_dict["ingestion"]
    
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
    Sentence-based Chunking.
    
    Gruppiert N Sätze pro Chunk. Schnell und einfach.
    Gut für strukturierte Dokumente (Papers, Artikel).
    """
    
    # Sentence splitting pattern
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def __init__(self, sentences_per_chunk: int = 3, min_chunk_size: int = 50):
        self.sentences_per_chunk = sentences_per_chunk
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Use regex for sentence splitting
        sentences = self.SENTENCE_PATTERN.split(text)
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
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
        sentences = self.split_into_sentences(text)
        
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
                "chunk_method": "sentence",
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
            })
        
        return chunks


class FixedSizeChunker:
    """
    Fixed-size Chunking mit Overlap.
    
    Einfachste Methode, aber kann semantische Einheiten zerschneiden.
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
    Wrapper around LangChain's RecursiveCharacterTextSplitter.
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
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                self._splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            except ImportError:
                raise ImportError(
                    "langchain required for recursive chunking. "
                    "Install with: pip install langchain"
                )
        return self._splitter
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk using RecursiveCharacterTextSplitter."""
        metadata = metadata or {}
        splitter = self._get_splitter()
        
        # Split text
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


class SemanticChunkerWrapper:
    """
    Wrapper für SemanticChunker aus semantic_chunking.py.
    
    Nutzt semantische Grenzen und Quality Filtering.
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
    
    def _get_chunker(self):
        """Lazy initialization of semantic chunker."""
        if self._chunker is None:
            try:
                from src.data_layer.semantic_chunking import SemanticChunker
                self._chunker = SemanticChunker(
                    max_chunk_size=self.chunk_size,
                    min_chunk_size=self.min_chunk_size,
                    overlap=self.chunk_overlap,
                )
                # Update quality filter settings
                self._chunker.quality_filter.min_lexical_diversity = self.min_lexical_diversity
                self._chunker.quality_filter.min_information_density = self.min_information_density
            except ImportError as e:
                logger.warning(f"SemanticChunker not available: {e}")
                logger.warning("Falling back to RecursiveChunker")
                return None
        return self._chunker
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk using semantic boundaries."""
        metadata = metadata or {}
        chunker = self._get_chunker()
        
        if chunker is None:
            # Fallback to recursive
            fallback = RecursiveChunker(
                self.chunk_size, self.chunk_overlap, self.min_chunk_size
            )
            return fallback.chunk(text, metadata)
        
        # Create temporary Document for SemanticChunker
        try:
            from langchain.schema import Document
            temp_doc = Document(page_content=text, metadata=metadata)
            
            # Process
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
            logger.warning(f"Semantic chunking failed: {e}, using fallback")
            fallback = RecursiveChunker(
                self.chunk_size, self.chunk_overlap, self.min_chunk_size
            )
            return fallback.chunk(text, metadata)


# ============================================================================
# ENTITY EXTRACTION (for Knowledge Graph)
# ============================================================================

class SimpleEntityExtractor:
    """
    Einfache Entity-Extraktion basierend auf Patterns.
    
    Extrahiert Named Entities für Knowledge Graph Konstruktion.
    """
    
    # Patterns for entity extraction
    ENTITY_PATTERNS = [
        r'"([^"]+)"',                              # Quoted strings
        r"'([^']+)'",                              # Single quoted
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",  # Multi-word proper nouns
        r"\b([A-Z][a-z]{2,})\b",                   # Single proper nouns
    ]
    
    # Common words to skip
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
        'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'just', 'also', 'now', 'here', 'there', 'then', 'once',
        # Common sentence starters
        'However', 'Therefore', 'Furthermore', 'Moreover', 'Although',
        'Because', 'Since', 'While', 'When', 'Where', 'After', 'Before',
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
        entities = set()
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Clean and validate
                entity = match.strip()
                if len(entity) > 2 and entity not in self.STOPWORDS:
                    entities.add(entity)
        
        return list(entities)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class DocumentIngestionPipeline:
    """
    Zentrale Document Ingestion Pipeline.
    
    Orchestriert Chunking, Entity Extraction, und Metadata Enrichment.
    
    USAGE:
        config = IngestionConfig(chunking_strategy="semantic")
        pipeline = DocumentIngestionPipeline(config)
        
        # Process raw documents
        documents = pipeline.process_texts(texts, metadatas)
        
        # Or process LangChain Documents
        documents = pipeline.process_documents(docs)
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
        
        # Initialize entity extractor
        self.entity_extractor = SimpleEntityExtractor() if config.extract_entities else None
        
        self.logger.info(
            f"DocumentIngestionPipeline initialized: "
            f"strategy={self.config.chunking_strategy}, "
            f"chunk_size={self.config.chunk_size}"
        )
    
    def _create_chunker(self):
        """Create chunker based on configuration."""
        strategy = self.config.chunking_strategy
        
        if strategy == ChunkingStrategy.SENTENCE.value:
            return SentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
                min_chunk_size=self.config.min_chunk_size,
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
        
        elif strategy == ChunkingStrategy.SEMANTIC.value:
            return SemanticChunkerWrapper(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
                min_lexical_diversity=self.config.min_lexical_diversity,
                min_information_density=self.config.min_information_density,
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
            List of chunk dicts with 'text', 'metadata', 'entities'
        """
        metadata = metadata or {}
        
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
        metadatas = metadatas or [{}] * len(texts)
        source_ids = source_ids or [None] * len(texts)
        
        all_chunks = []
        chunk_id = 0
        
        for text, meta, src_id in zip(texts, metadatas, source_ids):
            chunks = self.process_text(text, meta, src_id)
            
            # Add global chunk IDs
            for chunk in chunks:
                chunk["metadata"]["chunk_id"] = chunk_id
                chunk_id += 1
            
            all_chunks.extend(chunks)
        
        self.logger.info(
            f"Processed {len(texts)} texts → {len(all_chunks)} chunks "
            f"(strategy: {self.config.chunking_strategy})"
        )
        
        return all_chunks
    
    def process_documents(self, documents: List) -> List:
        """
        Process LangChain Document objects.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of LangChain Document objects (chunked)
        """
        try:
            from langchain.schema import Document
        except ImportError:
            raise ImportError("langchain required. Install with: pip install langchain")
        
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
            # Store entities in metadata if present
            if "entities" in chunk:
                doc.metadata["entities"] = chunk["entities"]
            result_docs.append(doc)
        
        return result_docs
    
    def process_articles(
        self,
        articles: List,  # List of Article dataclass
        chunk_id_start: int = 0,
    ) -> Tuple[List, int]:
        """
        Process Article objects (from benchmark datasets).
        
        Provides compatibility with benchmark_datasets.py Article format.
        
        Args:
            articles: List of Article dataclass instances
            chunk_id_start: Starting chunk ID
            
        Returns:
            Tuple of (List of Document objects, next_chunk_id)
        """
        try:
            from langchain.schema import Document
        except ImportError:
            raise ImportError("langchain required")
        
        all_docs = []
        chunk_id = chunk_id_start
        
        for article in articles:
            # Build text from sentences if available
            if hasattr(article, 'sentences') and article.sentences:
                text = " ".join(article.sentences)
            elif hasattr(article, 'text'):
                text = article.text
            else:
                continue
            
            # Base metadata
            metadata = {
                "article_title": getattr(article, 'title', 'unknown'),
                "dataset": getattr(article, 'dataset', 'unknown'),
                "article_id": getattr(article, 'id', 'unknown'),
            }
            
            # Process
            chunks = self.process_text(text, metadata, source_id=article.title)
            
            # Convert to Documents
            for chunk in chunks:
                chunk["metadata"]["chunk_id"] = chunk_id
                chunk["metadata"]["source_file"] = f"{metadata['dataset']}_{metadata['article_title']}"
                
                doc = Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"],
                )
                all_docs.append(doc)
                chunk_id += 1
        
        self.logger.info(
            f"Processed {len(articles)} articles → {len(all_docs)} chunks"
        )
        
        return all_docs, chunk_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "chunking_strategy": self.config.chunking_strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "entity_extraction": self.config.extract_entities,
        }


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
        strategy: "sentence", "semantic", "fixed", or "recursive"
        chunk_size: Max chunk size
        sentences_per_chunk: Sentences per chunk (for sentence strategy)
        **kwargs: Additional config options
        
    Returns:
        Configured DocumentIngestionPipeline
    """
    config = IngestionConfig(
        chunking_strategy=strategy,
        chunk_size=chunk_size,
        sentences_per_chunk=sentences_per_chunk,
        **kwargs,
    )
    return DocumentIngestionPipeline(config)


# ============================================================================
# MAIN (Testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test text
    test_text = """
    Albert Einstein was a German-born theoretical physicist. He developed the 
    theory of relativity, one of the two pillars of modern physics. His work 
    is also known for its influence on the philosophy of science.
    
    Einstein received the Nobel Prize in Physics in 1921. He was awarded for 
    his explanation of the photoelectric effect. Einstein published more than 
    300 scientific papers.
    
    In 1905, Einstein published four groundbreaking papers. This year is 
    sometimes called his "miracle year". The papers covered the photoelectric 
    effect, Brownian motion, special relativity, and mass-energy equivalence.
    """
    
    print("=" * 70)
    print("DOCUMENT INGESTION PIPELINE TEST")
    print("=" * 70)
    
    # Test different strategies
    strategies = ["sentence", "fixed", "recursive"]
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        
        pipeline = create_pipeline(
            strategy=strategy,
            chunk_size=300,
            sentences_per_chunk=2,
        )
        
        chunks = pipeline.process_text(test_text, {"source": "test"})
        
        print(f"Chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"  [{i}] {len(chunk['text'])} chars, "
                  f"entities: {chunk.get('entities', [])[:3]}")
    
    print("\n" + "=" * 70)
    print("SUCCESS - All strategies working!")
    print("=" * 70)