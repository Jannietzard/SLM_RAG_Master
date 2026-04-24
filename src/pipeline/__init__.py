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
    AgentPipelineConfig,
    PipelineResult,
    BatchProcessor,
    create_full_pipeline,
)
from .ingestion_pipeline import (
    IngestionPipeline,
    IngestionConfig,
    IngestionMetrics,
    DocumentLoader,
    create_ingestion_pipeline,
)

__version__ = "4.0.0"
__author__ = "Edge-RAG Research Project"

__all__ = [
    # Agent Pipeline
    "AgentPipeline",
    "AgentPipelineConfig",
    "PipelineResult",
    "BatchProcessor",
    "create_full_pipeline",

    # Ingestion Pipeline
    "IngestionPipeline",
    "IngestionConfig",
    "IngestionMetrics",
    "DocumentLoader",
    "create_ingestion_pipeline",
]
