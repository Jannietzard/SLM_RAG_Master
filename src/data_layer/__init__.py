"""
Data Layer Package for Edge-RAG System.

Modules:
- ingestion: PDF processing and chunking
- embeddings: Batched embeddings with caching
- storage: Vector store + Knowledge graph
- retrieval: Hybrid retrieval engine
- preprocessing: Content filtering

Import directly from modules:
    from src.data_layer.embeddings import BatchedOllamaEmbeddings
    from src.data_layer.storage import HybridStore, StorageConfig
    from src.data_layer.retrieval import HybridRetriever, RetrievalConfig
"""

from .entity_extraction import EntityExtractionPipeline
from .hybrid_retriever import HybridRetriever, RRFFusion

# Sentence Chunking (SpaCy-basiert)
try:
    from .sentence_chunking import (
        SpacySentenceChunker,
        SentenceChunkingConfig,
        SentenceChunk,
        SentenceInfo,
        create_sentence_chunker,
    )
except ImportError:
    pass  # SpaCy not installed


__version__ = "2.1.0"
__author__ = "Edge-RAG Research Project"

# Don't auto-import to avoid circular import issues
# Users should import directly from submodules