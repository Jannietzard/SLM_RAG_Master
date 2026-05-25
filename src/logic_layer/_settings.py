"""
Shared utilities for the logic layer.

Internal module — not part of the public API. Imported by planner.py,
navigator.py, controller.py, and verifier.py.

Exports:
    _load_settings()      — load config/settings.yaml from the project root.
    _validate_settings()  — warn when required keys are absent. Backed by
                            the 35-key ``_REQUIRED_SETTINGS`` tuple covering
                            every parameter that meaningfully affects EM/SF
                            metrics (LLM context budget, embeddings, vector
                            store, graph, RAG fusion + BM25, the Navigator
                            filter chain, Verifier validation thresholds,
                            agent pipeline flags, entity extraction,
                            ingestion, benchmark Soft-EM threshold). A
                            missing key emits a WARNING and the system
                            silently falls back to the dataclass default —
                            this is the reproducibility guard that catches
                            single-source-of-truth violations (see
                            TECHNICAL_ARCHITECTURE.md §11.16.5).
    _PROPER_NOUN_RE       — compiled regex for multi-word capitalized phrases.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared regex — multi-word capitalized proper-noun proxy.
#
# Matches: "Ed Wood", "Scott Derrickson", "New York", "Eiffel Tower"
# Misses:  ALL-CAPS acronyms (NATO), names with lowercase particles
#          (Tower of London).  These known gaps are acceptable for a
#          heuristic that avoids loading a second NER model at inference.
#
# Used by:
#   planner.py      — EntityExtractor.ENTITY_PATTERNS (regex NER fallback)
#   navigator.py    — _entity_overlap_pruning(), _extract_bridge_entities()
#   controller.py   — _extract_bridge_entities()
#   verifier.py     — Verifier._MULTI_PROPER_NOUN_RE (claim verification)
# ---------------------------------------------------------------------------
_PROPER_NOUN_RE: re.Pattern = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)


def _load_settings() -> Dict[str, Any]:
    """
    Load config/settings.yaml from the project root.

    Path is resolved relative to this file's location so the function works
    regardless of the current working directory.  Returns {} if the file is
    missing or unparseable so callers can still fall back to their dataclass
    defaults.

    Returns:
        Parsed settings dict, or {} on any error.
    """
    import yaml  # PyYAML is a required dependency (see requirements.txt)

    settings_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    if not settings_path.exists():
        logger.warning(
            "_load_settings: settings.yaml not found at %s — "
            "config dataclass defaults will be used as emergency fallbacks.",
            settings_path,
        )
        return {}
    try:
        with open(settings_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        _validate_settings(cfg)
        return cfg
    except yaml.YAMLError as exc:
        logger.error(
            "_load_settings: failed to parse %s (%s) — "
            "config dataclass defaults will be used as emergency fallbacks.",
            settings_path,
            exc,
        )
        return {}


# Keys that must be present for reproducible thesis evaluation.
# Format: tuple of nested keys, e.g. ("llm", "temperature") → cfg["llm"]["temperature"].
#
# Every parameter listed here MEANINGFULLY affects EM/SF metrics. If any one is
# missing from settings.yaml the system silently falls back to a hardcoded
# dataclass default, which can change results without notice (precise case: the
# 2026-05-24 audit found vector_store.top_k_vectors was silently 10 — the
# dataclass default — instead of the documented settings value 20, halving the
# vector retrieval funnel during evaluation). Growing this list is the
# reproducibility guard that catches that class of bug.
_REQUIRED_SETTINGS: tuple = (
    # ── LLM / Verifier prompt context budget ──────────────────────────
    ("llm", "model_name"),
    ("llm", "base_url"),
    ("llm", "temperature"),
    ("llm", "max_tokens"),
    ("llm", "timeout"),
    ("llm", "max_docs"),
    ("llm", "max_context_chars"),
    ("llm", "max_chars_per_doc"),
    # ── Embeddings ────────────────────────────────────────────────────
    ("embeddings", "model_name"),
    # ── Vector store ──────────────────────────────────────────────────
    ("vector_store", "top_k_vectors"),
    ("vector_store", "similarity_threshold"),
    ("vector_store", "distance_metric"),
    # ── Graph ─────────────────────────────────────────────────────────
    ("graph", "max_hops"),
    ("graph", "top_k_entities"),
    ("graph", "hub_mention_cap"),
    ("graph", "enable_hop3"),
    # ── RAG / retrieval fusion ────────────────────────────────────────
    ("rag", "retrieval_mode"),
    ("rag", "rrf_k"),
    ("rag", "cross_source_boost"),
    ("rag", "enable_bm25"),
    ("rag", "bm25_top_k"),
    # ── Navigator filter chain ────────────────────────────────────────
    ("navigator", "relevance_threshold_factor"),
    ("navigator", "redundancy_threshold"),
    ("navigator", "max_context_chunks"),
    ("navigator", "top_k_per_subquery"),
    ("navigator", "rrf_k"),
    ("navigator", "enable_reranker"),
    ("navigator", "contradiction_min_value"),
    # ── Verifier validation ───────────────────────────────────────────
    ("verifier", "entity_coverage_threshold"),
    ("verifier", "confidence_high_threshold"),
    # ── Agent pipeline ────────────────────────────────────────────────
    ("agent", "max_verification_iterations"),
    ("agent", "enable_verifier"),
    # ── Entity extraction (ingestion + query-time) ────────────────────
    ("entity_extraction", "gliner", "confidence_threshold"),
    # ── Ingestion ─────────────────────────────────────────────────────
    ("ingestion", "sentences_per_chunk"),
    # ── Benchmark (Soft-EM verdict threshold) ─────────────────────────
    ("benchmark", "answer_f1_threshold"),
)


def _validate_settings(cfg: Dict[str, Any]) -> None:
    """
    Warn when required settings.yaml keys are absent.

    A missing key means the system silently falls back to a hardcoded
    dataclass default — a reproducibility risk for thesis evaluation.
    This function emits a WARNING (not an error) so missing keys never
    prevent the system from starting, but are always visible in the logs.

    Args:
        cfg: Parsed settings dict returned by yaml.safe_load().
    """
    import warnings as _warnings

    for key_path in _REQUIRED_SETTINGS:
        node = cfg
        found = True
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                found = False
                break
            node = node[key]
        if not found:
            dotted = ".".join(key_path)
            _warnings.warn(
                "settings.yaml: key %r is missing — hardcoded dataclass default "
                "will be used instead. Evaluation results may not be reproducible. "
                "Check config/settings.yaml." % dotted,
                stacklevel=4,
            )
            logger.warning(
                "_validate_settings: required key %r absent from settings.yaml",
                dotted,
            )
