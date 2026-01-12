"""
Graph-Augmented Edge-RAG Library.

Submodules:
- ingestion: PDF Ingestion & Chunking
- storage: Hybrid Storage (Vector DB + Graph)
- retrieval: Hybrid Retrieval (Vector + Graph Ensemble)

Usage:
    from src.ingestion import DocumentIngestionPipeline
    from src.storage import HybridStore
    from src.retrieval import HybridRetriever
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__institution__ = "RWTH Aachen"

# Optional: Expose häufig genutzte Klassen
from src.ingestion import DocumentIngestionPipeline
from src.storage import HybridStore
from src.retrieval import HybridRetriever

__all__ = [
    "DocumentIngestionPipeline",
    "HybridStore",
    "HybridRetriever",
]