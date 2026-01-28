"""
Pipeline: Unified Orchestration Layer

Provides:
    - IngestionPipeline: Document → Chunks → Entities → Storage
    - AgentPipeline: Query → S_P → S_N → S_V → Answer

Features:
    - Batch processing for evaluation
    - Query caching
    - Early-exit optimizations
    - Comprehensive metrics
"""

from .agent_pipeline import (
    AgentPipeline,
    PipelineResult,
    BatchProcessor,
    create_pipeline,
    create_full_pipeline
)
from .ingestion_pipeline import (
    IngestionPipeline,
    IngestionConfig,
    IngestionMetrics,
    DocumentLoader,
    EmbeddingGenerator,
    create_ingestion_pipeline
)

__all__ = [
    # Agent Pipeline
    "AgentPipeline",
    "PipelineResult",
    "BatchProcessor",
    "create_pipeline",
    "create_full_pipeline",
    
    # Ingestion Pipeline
    "IngestionPipeline",
    "IngestionConfig",
    "IngestionMetrics",
    "DocumentLoader",
    "EmbeddingGenerator",
    "create_ingestion_pipeline"
]
