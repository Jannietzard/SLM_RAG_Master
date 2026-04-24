"""
Evaluation Layer — Artifact C

Thesis benchmark runners, metric utilities, and diagnostic tools.

Public API:
    metrics.normalize_answer        — HotpotQA normalisation pipeline
    metrics.compute_exact_match     — Exact Match with word-boundary substring fallback
    metrics.compute_f1              — Token-level F1 matching the official HotpotQA evaluator

All three metric functions are the single canonical implementation shared by
evaluate_hotpotqa.py, ablation_study.py, and agent_pipeline.BatchProcessor.
"""

from .metrics import normalize_answer, compute_exact_match, compute_f1

__version__ = "4.0.0"
__author__ = "Edge-RAG Research Project"

__all__ = [
    "normalize_answer",
    "compute_exact_match",
    "compute_f1",
]
