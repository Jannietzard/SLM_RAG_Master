"""
Canonical entity-type label maps — single source of truth.

All entity-type normalisation in the pipeline MUST import from this module so
that ingestion-time (GLiNERExtractor, SpaCyEntityExtractor) and query-time
(ImprovedQueryEntityExtractor, _normalize_query_entity) type assignments are
identical.  Duplicate local dicts that previously lived in entity_extraction.py
and hybrid_retriever.py have been removed in favour of these constants.

Maps exported:
    GLINER_LABEL_MAP       — lowercase GLiNER / query-time labels → canonical type
    SPACY_LABEL_MAP        — uppercase SpaCy NER labels → canonical type
                             (preserves GPE as distinct from LOCATION for graph
                              lookups; used by GLiNERExtractor._spacy_extract and
                              ImprovedQueryEntityExtractor._spacy_extract)
    SPACY_LABEL_MAP_FLAT   — uppercase SpaCy NER labels → canonical type
                             (flattens GPE → LOCATION; used by SpaCyEntityExtractor
                              where no graph-store GPE/LOCATION distinction is needed)
"""

from typing import Dict

# ---------------------------------------------------------------------------
# GLiNER label map — lowercase GLiNER output / query label → canonical type
# Unifies GLiNERExtractor._LABEL_MAP (entity_extraction.py) and
# _QUERY_LABEL_MAP (hybrid_retriever.py); all keys are lower-case.
# ---------------------------------------------------------------------------
GLINER_LABEL_MAP: Dict[str, str] = {
    # People
    "person":       "PERSON",
    "director":     "PERSON",
    "actor":        "PERSON",
    "politician":   "PERSON",
    "scientist":    "PERSON",
    "athlete":      "PERSON",
    # Organisations
    "organization": "ORGANIZATION",
    "company":      "ORGANIZATION",
    "studio":       "ORGANIZATION",
    "institution":  "ORGANIZATION",
    # Geopolitical / location
    "city":         "GPE",
    "country":      "GPE",
    "state":        "GPE",
    "gpe":          "GPE",
    "location":     "LOCATION",
    "place":        "LOCATION",
    "landmark":     "LOCATION",
    "monument":     "LOCATION",
    "building":     "LOCATION",
    # Creative works
    "film":         "WORK_OF_ART",
    "movie":        "WORK_OF_ART",
    "book":         "WORK_OF_ART",
    "album":        "WORK_OF_ART",
    "song":         "WORK_OF_ART",
    "work_of_art":  "WORK_OF_ART",
    "work of art":  "WORK_OF_ART",
    "award":        "WORK_OF_ART",
    "prize":        "WORK_OF_ART",
    # Events / other
    "event":        "EVENT",
    "product":      "PRODUCT",
    "technology":   "TECHNOLOGY",
}

# ---------------------------------------------------------------------------
# SpaCy label map — UPPERCASE SpaCy NER output → canonical type.
# Preserves GPE as a distinct type so that entity IDs generated at ingestion
# time (GLiNERExtractor._spacy_extract) match entity IDs generated at query
# time (ImprovedQueryEntityExtractor._spacy_extract).
# ---------------------------------------------------------------------------
SPACY_LABEL_MAP: Dict[str, str] = {
    "PERSON":      "PERSON",
    "ORG":         "ORGANIZATION",
    "GPE":         "GPE",       # city / country — kept distinct for graph lookups
    "LOC":         "LOCATION",
    "DATE":        "DATE",
    "EVENT":       "EVENT",
}

# ---------------------------------------------------------------------------
# SpaCy label map (flat) — UPPERCASE SpaCy NER output → canonical type.
# Collapses GPE → LOCATION and adds WORK_OF_ART / FAC entries.
# Used exclusively by SpaCyEntityExtractor, where graph-store GPE/LOCATION
# discrimination is not required.
# ---------------------------------------------------------------------------
SPACY_LABEL_MAP_FLAT: Dict[str, str] = {
    "PERSON":      "PERSON",
    "ORG":         "ORGANIZATION",
    "GPE":         "LOCATION",  # flattened: no GPE/LOCATION distinction needed
    "LOC":         "LOCATION",
    "DATE":        "DATE",
    "EVENT":       "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "FAC":         "LOCATION",
}
