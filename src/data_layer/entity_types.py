"""
Canonical entity-type label maps -- the type-normalisation contract.

Edge-RAG mints entity IDs at two distinct points in the pipeline:

  - Ingestion time (`entity_extraction.GLiNERExtractor` and
    `SpaCyEntityExtractor`), when a document is chunked and stored.
  - Query time (`hybrid_retriever.ImprovedQueryEntityExtractor`), when
    a user question is parsed and its entities are matched against the
    graph.

Both phases call separate NER backends (GLiNER for ingestion, SpaCy as
a fallback and at query time) that emit overlapping but non-identical
label vocabularies. If the two phases assigned different canonical
types to the same surface form, the graph lookup would miss its target
even when the entity is present. This module is the single point of
agreement: every label-to-type translation in the pipeline MUST go
through one of the three maps below; per-class private dictionaries
have been removed.

Exported tables
---------------
GLINER_LABEL_MAP
    Lowercase GLiNER prompts and lowercase query-time labels -> canonical
    OntoNotes-style type. Keys are a SUPERSET of the GLiNER prompts
    actually configured in `config/settings.yaml` (`person`,
    `organization`, `location`, `city`, `country`, `date`, `event`,
    `work of art`, `product`). The extra keys (`director`, `actor`,
    `album`, `film`, `landmark`, ...) are tolerated so that:
      * query-time entity extractors that synthesise finer-grained
        sub-labels still normalise to the same canonical type, and
      * legacy ingestion traces produced under a broader prompt set
        remain interpretable.
    Adding new keys here is safe and idempotent; adding new GLiNER
    *prompts* to settings.yaml is a separate, methodologically
    significant decision (see settings.yaml comment).

SPACY_LABEL_MAP
    Uppercase SpaCy NER labels -> canonical type. Preserves `GPE` as a
    type distinct from `LOCATION` because the graph store keys on this
    distinction. Consumed by `GLiNERExtractor._spacy_extract` and by
    `ImprovedQueryEntityExtractor._spacy_extract`, where ingestion-time
    and query-time IDs must collide bit-for-bit.

SPACY_LABEL_MAP_FLAT
    Uppercase SpaCy NER labels -> canonical type, with `GPE` collapsed
    to `LOCATION` and with `WORK_OF_ART` / `FAC` added. Consumed only
    by `SpaCyEntityExtractor`, an ablation/alternative extractor where
    the graph-store GPE/LOCATION distinction is not required.

Canonical type universe
-----------------------
The reachable image of `GLINER_LABEL_MAP` over the configured GLiNER
prompts is eight types: PERSON, ORGANIZATION, LOCATION, GPE, DATE,
EVENT, WORK_OF_ART, PRODUCT. SpaCy maps add no new types (FAC -> LOCATION;
WORK_OF_ART is already in the universe). This is the OntoNotes-5 core
set (Weischedel et al., 2013, LDC2013T19), chosen for cross-dataset
transferability across HotpotQA, 2WikiMultiHopQA, and StrategyQA.

Why two SpaCy maps?
-------------------
`SPACY_LABEL_MAP` is the canonical one used wherever a label appears
in a flow that also stores or queries the graph (the GPE/LOCATION
distinction matters there). `SPACY_LABEL_MAP_FLAT` is used by a
SpaCy-only extractor that never reaches the graph store, so flattening
GPE simplifies downstream matching without breaking any contract.

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

from typing import Dict

__all__ = [
    "GLINER_LABEL_MAP",
    "SPACY_LABEL_MAP",
    "SPACY_LABEL_MAP_FLAT",
]

# ---------------------------------------------------------------------------
# GLiNER label map -- lowercase GLiNER output / query label -> canonical type.
# Unifies what used to be GLiNERExtractor._LABEL_MAP (entity_extraction.py)
# and _QUERY_LABEL_MAP (hybrid_retriever.py); all keys are lower-case.
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
    # Both "work_of_art" (underscore) and "work of art" (space) are kept
    # intentionally: upstream sources differ in whitespace convention and
    # collapsing them would silently drop one form.
    "film":         "WORK_OF_ART",
    "movie":        "WORK_OF_ART",
    "book":         "WORK_OF_ART",
    "album":        "WORK_OF_ART",
    "song":         "WORK_OF_ART",
    "work_of_art":  "WORK_OF_ART",
    "work of art":  "WORK_OF_ART",
    "award":        "WORK_OF_ART",
    "prize":        "WORK_OF_ART",
    # Temporal
    "date":         "DATE",
    "year":         "DATE",
    "time":         "DATE",
    # Events / other
    "event":        "EVENT",
    "product":      "PRODUCT",
    "technology":   "TECHNOLOGY",
}

# ---------------------------------------------------------------------------
# SpaCy label map -- UPPERCASE SpaCy NER output -> canonical type.
# Preserves GPE as a distinct type so that entity IDs generated at ingestion
# time (GLiNERExtractor._spacy_extract) match entity IDs generated at query
# time (ImprovedQueryEntityExtractor._spacy_extract).
# ---------------------------------------------------------------------------
SPACY_LABEL_MAP: Dict[str, str] = {
    "PERSON":      "PERSON",
    "ORG":         "ORGANIZATION",
    "GPE":         "GPE",       # city / country -- kept distinct for graph lookups
    "LOC":         "LOCATION",
    "DATE":        "DATE",
    "EVENT":       "EVENT",
}

# ---------------------------------------------------------------------------
# SpaCy label map (flat) -- UPPERCASE SpaCy NER output -> canonical type.
# Collapses GPE -> LOCATION and adds WORK_OF_ART / FAC entries.
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
