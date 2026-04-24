"""
Shared utilities for the logic layer.

Internal module — not part of the public API.  Imported by planner.py,
navigator.py, controller.py, and verifier.py.

Exports:
    _load_settings()      — load config/settings.yaml from the project root.
    _validate_settings()  — warn when required keys are absent (reproducibility guard).
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
_REQUIRED_SETTINGS: tuple = (
    ("llm", "model_name"),
    ("llm", "temperature"),
    ("llm", "base_url"),
    ("embeddings", "model_name"),
    ("navigator", "rrf_k"),
    ("entity_extraction", "gliner", "confidence_threshold"),
    ("ingestion", "sentences_per_chunk"),
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
