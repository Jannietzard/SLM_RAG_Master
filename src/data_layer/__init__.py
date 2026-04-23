"""
Data Layer — Artifact A of the Edge-RAG System.

================================================================================
ARCHITECTURAL OVERVIEW
================================================================================

This package encapsulates all data operations for the Edge-RAG pipeline:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  chunking.py        Document segmentation (SpaCy + semantic + utils) │
  │  embeddings.py      Batched Ollama embeddings with SQLite cache       │
  │  entity_extraction.py  GLiNER NER + REBEL relation extraction         │
  │  storage.py         HybridStore façade: LanceDB + KuzuDB              │
  │  hybrid_retriever.py  HybridRetriever, RRF fusion, pre-gen filter    │
  │  ingestion.py       DocumentIngestionPipeline (chunking orchestration)│
  └──────────────────────────────────────────────────────────────────────┘

Ingestion data flow:
  raw text → ingestion.py → chunking.py → embeddings.py → storage.py

Query data flow:
  query → hybrid_retriever.py → (embeddings.py + storage.py) → results

================================================================================
REQUIRED EXTERNAL SERVICES (all local, no cloud dependency)
================================================================================

  * Ollama (http://localhost:11434) — nomic-embed-text + phi3
  * SpaCy model en_core_web_sm  → python -m spacy download en_core_web_sm
  * GLiNER urchade/gliner_small-v2.1 (auto-downloaded by HuggingFace)

================================================================================
COMMON IMPORT PATTERNS
================================================================================

Retrieval (Logic Layer / Navigator):
    from src.data_layer import HybridRetriever, RetrievalConfig
    from src.data_layer import HybridStore, StorageConfig

Ingestion (Pipeline Layer):
    from src.data_layer import DocumentIngestionPipeline, IngestionConfig
    from src.data_layer import create_ingestion_config
    from src.data_layer import BatchedOllamaEmbeddings
    from src.data_layer import EntityExtractionPipeline

Chunking (direct use):
    from src.data_layer import SpacySentenceChunker, create_sentence_chunker

================================================================================
"""

__version__ = "4.0.0"
__author__ = "Edge-RAG Research Project"

# ── Storage ───────────────────────────────────────────────────────────────────
from .storage import (
    HybridStore,
    StorageConfig,
    VectorStoreAdapter,
    KuzuGraphStore,
    create_storage_config,
)

# ── Embeddings ────────────────────────────────────────────────────────────────
from .embeddings import BatchedOllamaEmbeddings, create_embeddings

# ── Retrieval ─────────────────────────────────────────────────────────────────
from .hybrid_retriever import (
    HybridRetriever,
    RetrievalConfig,
    RetrievalMode,
    RetrievalResult,
    RetrievalMetrics,
    PreGenerativeFilter,
    ImprovedQueryEntityExtractor,
)

# ── Entity Extraction ─────────────────────────────────────────────────────────
from .entity_extraction import (
    EntityExtractionPipeline,
    ExtractionConfig,
    create_extraction_pipeline,
    normalize_entity_name,
)

# ── Chunking ──────────────────────────────────────────────────────────────────
# Public API: SpacySentenceChunker and create_sentence_chunker.
# SemanticChunker / FixedSizeChunker / RecursiveChunker / SentenceChunker are
# internal implementation details consumed only by ingestion.py.
from .chunking import (
    SpacySentenceChunker,
    SentenceChunkingConfig,
    SentenceChunk,
    create_sentence_chunker,
)

# ── Ingestion Pipeline ────────────────────────────────────────────────────────
from .ingestion import (
    DocumentIngestionPipeline,
    IngestionConfig,
    ChunkingStrategy,
    create_ingestion_config,
    create_data_layer_pipeline,
)

__all__ = [
    # Storage
    "HybridStore",
    "StorageConfig",
    "VectorStoreAdapter",
    "KuzuGraphStore",
    "create_storage_config",
    # Embeddings
    "BatchedOllamaEmbeddings",
    "create_embeddings",
    # Retrieval
    "HybridRetriever",
    "RetrievalConfig",
    "RetrievalMode",
    "RetrievalResult",
    "RetrievalMetrics",
    "PreGenerativeFilter",
    "ImprovedQueryEntityExtractor",
    # Entity Extraction
    "EntityExtractionPipeline",
    "ExtractionConfig",
    "create_extraction_pipeline",
    "normalize_entity_name",
    # Chunking — public API
    "SpacySentenceChunker",
    "SentenceChunkingConfig",
    "SentenceChunk",
    "create_sentence_chunker",
    # SemanticChunker, SentenceChunker, FixedSizeChunker, RecursiveChunker,
    # SentenceInfo, create_semantic_chunker are internal implementation details
    # used only by ingestion.py — not part of the public API.
    # Ingestion pipeline
    "DocumentIngestionPipeline",
    "IngestionConfig",
    "ChunkingStrategy",
    "create_ingestion_config",
    "create_data_layer_pipeline",
]

