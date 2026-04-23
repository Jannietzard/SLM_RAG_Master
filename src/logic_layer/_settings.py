"""
Shared utilities for the logic layer.

Internal module — not part of the public API.  Imported by planner.py,
navigator.py, controller.py, and verifier.py.

Exports:
    _load_settings()   — load config/settings.yaml from the project root.
    _PROPER_NOUN_RE    — compiled regex for multi-word capitalized phrases.
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
            return yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        logger.error(
            "_load_settings: failed to parse %s (%s) — "
            "config dataclass defaults will be used as emergency fallbacks.",
            settings_path,
            exc,
        )
        return {}
