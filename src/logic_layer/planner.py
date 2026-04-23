"""
===============================================================================
S_P: Rule-Based Query Planner
===============================================================================

Master's Thesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"
Artifact B: Agent-Based Query Processing

===============================================================================
OVERVIEW
===============================================================================

The Planner (S_P) is the first stage of the Agentic RAG system and acts as a
deterministic router for query analysis and retrieval plan generation.

Core functions per thesis specification:
1. QUERY CLASSIFICATION (Heuristic)
   - Uses SpaCy's Rule-Based Matcher instead of ML models
   - Classifies: Single-Hop, Multi-Hop, Comparison, Temporal Reasoning
   - Minimises latency through lightweight linguistic heuristics

2. ENTITY & BRIDGE DETECTION
   - Extraction via SpaCy NER (confidence > min_entity_confidence)
   - Dependency parsing for syntactic relationships
   - Bridge entities for multi-hop graph traversal

3. PLAN GENERATION
   - Structured JSON retrieval plan
   - Defines: strategy, hop sequence, constraints
   - Strategies: Vector-Only, Hybrid (Graph-Only reserved for future work)

===============================================================================
ARCHITECTURE
===============================================================================

    User Query
        │
        ▼
    ┌─────────────────────────────────────────────────────┐
    │                    S_P (PLANNER)                     │
    │                                                      │
    │   ┌──────────────┐    ┌──────────────┐              │
    │   │  SpaCy NLP   │───▶│   Query      │              │
    │   │  Processing  │    │   Classifier │              │
    │   └──────────────┘    └──────────────┘              │
    │          │                    │                      │
    │          ▼                    ▼                      │
    │   ┌──────────────┐    ┌──────────────┐              │
    │   │   Entity &   │    │   Bridge     │              │
    │   │   NER Extract│    │   Detection  │              │
    │   └──────────────┘    └──────────────┘              │
    │          │                    │                      │
    │          └────────┬───────────┘                      │
    │                   ▼                                  │
    │          ┌──────────────┐                            │
    │          │  Retrieval   │                            │
    │          │  Plan Gen    │                            │
    │          └──────────────┘                            │
    │                   │                                  │
    └───────────────────┼──────────────────────────────────┘
                        ▼
                Retrieval Plan (JSON)
                   → Navigator (S_N)

===============================================================================
ACADEMIC REFERENCES
===============================================================================

- Honnibal, M., & Montani, I. (2017). "spaCy 2: Natural language understanding
  with Bloom embeddings, convolutional neural networks and incremental parsing."
  Unpublished; arXiv:1802.04016.
- Yang, Z., et al. (2018). "HotpotQA: A Dataset for Diverse, Explainable
  Multi-hop Question Answering." EMNLP 2018.
- Khattab, O., et al. (2022). "Demonstrate-Search-Predict: Composing retrieval
  and language models for knowledge-intensive NLP." arXiv:2212.14024.
  (Motivation for structured query decomposition in RAG pipelines.)
- Weischedel, R., et al. (2013). "OntoNotes Release 5.0." LDC2013T19.
  (Label-level NER confidence estimates for en_core_web_sm in
  EntityExtractor._estimate_confidence; see _LABEL_CONFIDENCE dict.)

===============================================================================
Review History:
    Last Reviewed:  2026-04-21
    Review Result:  0 CRITICAL, 4 IMPORTANT, 7 RECOMMENDED
    Reviewer:       Code Review Prompt v2.1
    Next Review:    After re-ingestion with updated entity types or SpaCy lazy-load
                    refactor (Finding 7)
===============================================================================
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


from ._settings import _load_settings, _PROPER_NOUN_RE


# =============================================================================
# MODULE-LEVEL CONSTANTS
# =============================================================================

# Year pattern used in temporal constraint extraction.
# Matches historical years (1000–1999) and 21st-century years (2000–2099).
# Narrower than the 4-digit patterns in TEMPORAL_PATTERNS / ENTITY_PATTERNS to
# avoid false positives on port numbers or other 4-digit tokens.
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")


# =============================================================================
# SPACY INTEGRATION
# =============================================================================
# SpaCy is used for entity extraction and dependency parsing.
# If unavailable, regex fallbacks are used throughout.

try:
    import spacy
    from spacy.matcher import Matcher

    # Model loaded once at import time using this default.
    # KNOWN LIMITATION: PlannerConfig.spacy_model (from settings.yaml →
    # ingestion.spacy_model) cannot affect this module-level load — the
    # setting is read after the model is already in memory. A lazy-load
    # refactor (module-level _NLP = None + _get_nlp() accessor) would fix
    # this but is deferred; en_core_web_sm matches settings.yaml currently.
    _DEFAULT_SPACY_MODEL = "en_core_web_sm"

    try:
        NLP = spacy.load(_DEFAULT_SPACY_MODEL)
        SPACY_AVAILABLE = True
        logger.info("SpaCy model '%s' loaded for query analysis", _DEFAULT_SPACY_MODEL)
    except OSError:
        NLP = None
        SPACY_AVAILABLE = False
        logger.warning(
            "SpaCy model '%s' not found. Install with:\n"
            "  python -m spacy download en_core_web_sm\n"
            "Regex fallbacks will be used for entity extraction.",
            _DEFAULT_SPACY_MODEL,
        )
except ImportError:
    NLP = None
    SPACY_AVAILABLE = False
    Matcher = None
    logger.warning(
        "SpaCy not installed. Install with:\n"
        "  pip install spacy\n"
        "  python -m spacy download en_core_web_sm\n"
        "Regex fallbacks will be used for entity extraction."
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class QueryType(Enum):
    """
    Query type classification based on reasoning complexity.

    Per thesis Section 3.2:
    - SINGLE_HOP:   Simple factual question, one retrieval step.
    - MULTI_HOP:    Sequential dependencies requiring bridge entities.
    - COMPARISON:   Parallel retrieval + comparison logic.
    - TEMPORAL:     Temporal reasoning component.
    - AGGREGATE:    Aggregation of multiple results.
    - INTERSECTION: Shared properties between two subjects.
    """
    SINGLE_HOP   = "single_hop"    # e.g. "What is the capital of France?"
    MULTI_HOP    = "multi_hop"     # e.g. "Who directed the film starring Tom Hanks?"
    COMPARISON   = "comparison"    # e.g. "Is Berlin older than Munich?"
    TEMPORAL     = "temporal"      # e.g. "What happened after WW2?"
    AGGREGATE    = "aggregate"     # e.g. "List all films from 2020."
    INTERSECTION = "intersection"  # e.g. "What do A and B have in common?"


class RetrievalStrategy(Enum):
    """
    Retrieval strategy selected based on query complexity.

    Per thesis Section 3.2:
    - VECTOR_ONLY: Fast, for simple single-hop queries.
    - GRAPH_ONLY:  For relation-based queries (reserved for future work —
                   not currently selected by _determine_strategy; included
                   for ablation interface compatibility).
    - HYBRID:      Combined for complex multi-hop and comparison queries.
    """
    VECTOR_ONLY = "vector_only"  # Fast; for simple queries
    GRAPH_ONLY  = "graph_only"   # Reserved for future work (not yet active)
    HYBRID      = "hybrid"       # Combined; for complex queries


@dataclass
class EntityInfo:
    """
    Information about an extracted entity.

    Attributes:
        text:        Entity surface form.
        label:       NER label (PERSON, ORG, GPE, etc.).
        confidence:  Extraction confidence (0.0–1.0).
        start_char:  Start character offset in the original text.
        end_char:    End character offset in the original text.
        is_bridge:   True if this entity acts as a bridge in multi-hop reasoning.
    """
    text: str
    label: str = "UNKNOWN"
    confidence: float = 1.0
    start_char: int = 0
    end_char: int = 0
    is_bridge: bool = False


@dataclass
class HopStep:
    """
    A single step in a multi-hop reasoning chain.

    Attributes:
        step_id:         Unique step identifier.
        sub_query:       The sub-query for this step.
        target_entities: Target entities for this step.
        depends_on:      IDs of steps this step depends on.
        is_bridge:       True if this step resolves a bridge entity.
    """
    step_id: int
    sub_query: str
    target_entities: List[str] = field(default_factory=list)
    depends_on: List[int] = field(default_factory=list)
    is_bridge: bool = False


@dataclass
class RetrievalPlan:
    """
    Structured retrieval plan for the Navigator (S_N).

    This is the primary output format of the Planner, consumed by the Navigator
    to execute hybrid retrieval.

    Attributes:
        original_query:  The original user query.
        query_type:      Classified query type.
        strategy:        Selected retrieval strategy.
        entities:        List of extracted entities with metadata.
        hop_sequence:    Ordered list of hop steps.
        sub_queries:     Flat list of all sub-queries for retrieval.
        constraints:     Additional constraints (temporal, comparison, etc.).
        estimated_hops:  Estimated number of retrieval hops.
        confidence:      Query classification confidence.
        metadata:        Additional metadata for debugging.
    """
    original_query: str
    query_type: QueryType
    strategy: RetrievalStrategy
    entities: List[EntityInfo] = field(default_factory=list)
    hop_sequence: List[HopStep] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_hops: int = 1
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary for JSON output."""
        return {
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "strategy": self.strategy.value,
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "confidence": e.confidence,
                    "is_bridge": e.is_bridge,
                }
                for e in self.entities
            ],
            "hop_sequence": [
                {
                    "step_id": h.step_id,
                    "sub_query": h.sub_query,
                    "target_entities": h.target_entities,
                    "depends_on": h.depends_on,
                    "is_bridge": h.is_bridge,
                }
                for h in self.hop_sequence
            ],
            "sub_queries": self.sub_queries,
            "constraints": self.constraints,
            "estimated_hops": self.estimated_hops,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        """
        Serialise to a JSON string.

        Public API convenience method for external consumers (e.g. REST endpoints,
        logging pipelines). Internal pipeline code uses to_dict() directly.
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class PlannerConfig:
    """
    Configuration for the Query Planner.

    All numeric thresholds are empirically tuned on the HotpotQA dev set
    (see thesis Section 4.3 for ablation results). All values are sourced
    from ``config/settings.yaml → planner`` via ``from_yaml()``; the
    dataclass defaults serve only as documented emergency fallbacks.

    Attributes:
        min_entity_confidence:    Minimum NER confidence for entity extraction.
        max_entities:             Maximum number of entities to extract per query.
        enable_bridge_detection:  Enable bridge entity detection for multi-hop.
        enable_temporal_parsing:  Parse temporal constraints from queries.
        default_strategy:         Fallback strategy when classification is ambiguous.
        spacy_model:              SpaCy model name. Sourced from
                                  ``settings.yaml → ingestion.spacy_model``.
        regex_entity_confidence:  Confidence assigned to entities found by the
                                  regex fallback (not SpaCy NER). Lower than NER
                                  confidence to reflect noisier extraction.
        entity_density_threshold: Named-entity count above which the multi-hop
                                  score boost fires in classify() Phase 3.
        noun_density_threshold:   Noun/proper-noun count threshold for the same.
        classifier_spacy_weight:  Score bonus for SpaCy Matcher hits (Phase 2).
        classifier_entity_boost:  Score bonus when entity density is high (Phase 3).
        classifier_confidence_base:  Base confidence added to scaled score.
        classifier_confidence_scale: Score multiplier for confidence calculation.
        classifier_confidence_cap:   Upper cap on returned confidence.
        classifier_fallback_confidence: Confidence for SINGLE_HOP fallback.
    """
    min_entity_confidence: float = 0.7       # settings.yaml: planner.min_entity_confidence
    max_entities: int = 10                    # settings.yaml: planner.max_entities
    enable_bridge_detection: bool = True
    enable_temporal_parsing: bool = True
    default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    spacy_model: str = "en_core_web_sm"      # settings.yaml: ingestion.spacy_model

    # Confidence for regex-matched entities (lower than NER-based estimates)
    regex_entity_confidence: float = 0.75    # settings.yaml: planner.regex_entity_confidence

    # Entity/noun density thresholds for multi-hop heuristic (Phase 3)
    entity_density_threshold: int = 2        # settings.yaml: planner.entity_density_threshold
    noun_density_threshold: int = 4          # settings.yaml: planner.noun_density_threshold

    # Classifier weight constants — empirically tuned on HotpotQA dev set.
    # Changing these may affect classification accuracy on other benchmarks.
    classifier_spacy_weight: float = 1.5     # settings.yaml: planner.classifier_spacy_weight
    classifier_entity_boost: float = 0.5     # settings.yaml: planner.classifier_entity_boost
    classifier_confidence_base: float = 0.6  # settings.yaml: planner.classifier_confidence_base
    classifier_confidence_scale: float = 0.15  # settings.yaml: planner.classifier_confidence_scale
    classifier_confidence_cap: float = 0.95  # settings.yaml: planner.classifier_confidence_cap
    classifier_fallback_confidence: float = 0.8  # settings.yaml: planner.classifier_fallback_confidence

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "PlannerConfig":
        """
        Build a PlannerConfig from a settings.yaml dict.

        Reads the ``planner`` block for planner-specific settings and
        ``ingestion.spacy_model`` for the shared SpaCy model name.
        All defaults match the thesis evaluation settings documented in
        settings.yaml. Follows the same pattern as IngestionConfig.from_yaml().

        Args:
            config: Full settings.yaml dict (or the relevant sub-dict).

        Returns:
            PlannerConfig populated from the provided settings dict.
        """
        planner = config.get("planner", {})
        ingestion = config.get("ingestion", {})
        return cls(
            min_entity_confidence=planner.get("min_entity_confidence", 0.7),
            max_entities=planner.get("max_entities", 10),
            enable_bridge_detection=planner.get("enable_bridge_detection", True),
            enable_temporal_parsing=planner.get("enable_temporal_parsing", True),
            spacy_model=ingestion.get("spacy_model", "en_core_web_sm"),
            regex_entity_confidence=planner.get("regex_entity_confidence", 0.75),
            entity_density_threshold=planner.get("entity_density_threshold", 2),
            noun_density_threshold=planner.get("noun_density_threshold", 4),
            classifier_spacy_weight=planner.get("classifier_spacy_weight", 1.5),
            classifier_entity_boost=planner.get("classifier_entity_boost", 0.5),
            classifier_confidence_base=planner.get("classifier_confidence_base", 0.6),
            classifier_confidence_scale=planner.get("classifier_confidence_scale", 0.15),
            classifier_confidence_cap=planner.get("classifier_confidence_cap", 0.95),
            classifier_fallback_confidence=planner.get("classifier_fallback_confidence", 0.8),
        )


# =============================================================================
# QUERY CLASSIFIER
# =============================================================================

class QueryClassifier:
    """
    Rule-based query classifier using SpaCy Matcher.

    Per thesis Section 3.2:
    "Instead of an ML model, SpaCy's Rule-Based Matcher is used. It identifies
    query types such as Comparison, Temporal, or Multi-Hop through lexical
    pattern matching, minimising inference latency."
    (Honnibal & Montani, 2017; arXiv:1802.04016)

    Classification uses four phases:
    1. Lexical regex pattern counts per query type.
    2. SpaCy Matcher boost for syntactic patterns.
    3. Entity-density heuristic for multi-hop identification.
    4. Priority-ordered tie-break with confidence scaling.

    All weight constants are read from PlannerConfig (settings.yaml) for full
    reproducibility.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # LEXICAL PATTERNS FOR QUERY CLASSIFICATION
    # Tuples (immutable) prevent accidental mutation by subclasses.
    # ─────────────────────────────────────────────────────────────────────────

    # Comparison indicators: comparative words and structures
    COMPARISON_PATTERNS = (
        r"\b(older|younger|taller|shorter|bigger|smaller|larger|higher|lower)\s+than\b",
        r"\b(more|less|fewer)\s+\w+\s+than\b",
        r"\b(compare|comparison|versus|vs\.?|vs)\b",
        r"\bdifference\s+between\b",
        r"\bwhich\s+(is|was|are|were)\s+\w*(er|est)\b",
        r"\b(better|worse|best|worst)\b.*\bor\b",
        r"\bor\b.*\?(which|what)\s+(is|was)\s+\w*(er|est)",
        r"\bsame\s+\w+\b",          # "same nationality", "same country", "same field"
        r"\bboth\s+.+\s+(born|from|have|had|were|are)\b",  # "both born in", "both from"
        # "Who is older/taller/..., X or Y?" — no "than" required
        r"\b(older|younger|taller|shorter|bigger|smaller|larger|higher|lower|richer|poorer)\b.{0,60}\bor\b",
    )

    # Temporal indicators: time references and temporal structures
    TEMPORAL_PATTERNS = (
        r"\b(before|after|during|since|until|when|while)\b",
        r"\b(year|month|day|century|decade|era)\s+\d+",
        r"\b\d{4}\b",  # four-digit years
        r"\b(first|last|latest|earliest|recent|previous|next)\b",
        r"\b(history|historical|timeline|chronolog)\w*\b",
        r"\b(began|started|ended|founded|established)\b",
    )

    # Multi-hop indicators: nested structures
    MULTI_HOP_PATTERNS = (
        r"\bof\s+(a|an|the)\s+\w+\s+(that|which|who)\b",   # "of a/the X that/who" (bridge)
        r"\bwhere\s+.+\s+(was|is|were|are)\b",
        r"\b\w+\s+of\s+the\s+\w+\s+of\b",
        r"'s\s+\w+'s",  # possessive chains
        r"\b(who|what)\s+\w+\s+(the|a)\s+\w+\s+(that|which)\b",
        # Bridge-relation patterns (HotpotQA / 2WikiMultiHop — Yang et al., 2018):
        r"\b(starring|featuring|directed by|written by|authored by|composed by)\b",
        r"\b(father|mother|son|daughter|wife|husband|creator|founder)\s+of\b",
        r"\b(located|situated)\s+in\s+the\b",
        r"\b(formed|created|founded|established|organized|produced|released)\s+by\b",
        r"\b\w+\s+(group|band|company|team|studio|label)\s+(that|which|who)\b",
        r"\bformed\s+by\b",           # "group that was formed by who?"
        r"\bwas\s+\w+ed\s+by\b",      # "was founded/formed/created by"
        r"\b(debut|first|second)\s+(album|single|film|movie|show)\s+of\b",
    )

    # Intersection indicators: shared properties
    INTERSECTION_PATTERNS = (
        r"\bboth\s+.+\s+and\b",
        r"\bin\s+common\b",
        r"\b(also|too)\b.*\band\b",
        r"\bshared\s+(by|between)\b",
    )

    # Aggregation indicators: lists and summaries
    AGGREGATE_PATTERNS = (
        r"\b(list|enumerate|all|every|count|how\s+many)\b",
        r"\b(summarize|summary|overview)\b",
        r"\bwhat\s+(are|were)\s+the\b",
    )

    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialise the Query Classifier.

        Args:
            config: Planner configuration.
        """
        self.config = config or PlannerConfig()

        # Compile regex patterns once at construction time
        self._compiled_patterns = {
            QueryType.COMPARISON:   [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS],
            QueryType.TEMPORAL:     [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS],
            QueryType.MULTI_HOP:    [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS],
            QueryType.INTERSECTION: [re.compile(p, re.IGNORECASE) for p in self.INTERSECTION_PATTERNS],
            QueryType.AGGREGATE:    [re.compile(p, re.IGNORECASE) for p in self.AGGREGATE_PATTERNS],
        }

        # SpaCy Matcher for syntactic patterns
        self._setup_spacy_matcher()

        logger.info("QueryClassifier initialised")

    def _setup_spacy_matcher(self) -> None:
        """
        Initialise the SpaCy Matcher with linguistic patterns.

        The Matcher identifies syntactic structures that indicate query
        complexity (Honnibal & Montani, 2017; arXiv:1802.04016).
        """
        if not SPACY_AVAILABLE or NLP is None:
            self.matcher = None
            return

        self.matcher = Matcher(NLP.vocab)

        # Pattern for multi-hop: "of the X that/which Y"
        # Example: "the director of the film that won"
        multi_hop_pattern = [
            {"LOWER": "of"},
            {"LOWER": "the"},
            {"POS": {"IN": ["NOUN", "PROPN"]}},
            {"LOWER": {"IN": ["that", "which", "who"]}},
        ]
        self.matcher.add("MULTI_HOP", [multi_hop_pattern])

        # Pattern for comparison: comparative adjective/adverb + "than"
        comparison_pattern = [
            {"TAG": {"IN": ["JJR", "RBR"]}},  # comparative form
            {"LOWER": "than"},
        ]
        self.matcher.add("COMPARISON", [comparison_pattern])

        logger.debug("SpaCy Matcher configured")

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify a query and determine its type.

        Algorithm:
        1. Count regex pattern matches for each query type.
        2. Apply SpaCy Matcher boost for syntactic hits.
        3. Apply entity-density heuristic for multi-hop detection.
        4. Select highest-scoring type with priority tie-break.
        5. Fall back to SINGLE_HOP if no pattern matches.

        Weight constants are read from PlannerConfig (settings.yaml):
        - spacy_weight (default 1.5):    syntactic match carries more weight
          than a single regex hit (SpaCy pattern is more precise).
        - entity_boost (default 0.5):    partial nudge — entity density is a
          weak but useful signal for multi-hop.
        - confidence_base / scale / cap: map raw score to [0.6, 0.95] range.

        Note: SINGLE_HOP has no patterns in _compiled_patterns and therefore
        always scores 0; it is selected only via the fallback path at the end.

        Args:
            query: The query to classify.

        Returns:
            Tuple of (QueryType, confidence).
        """
        query = query.strip()
        scores = {qt: 0.0 for qt in QueryType}

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 1: Regex pattern matching
        # ─────────────────────────────────────────────────────────────────────

        for query_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[query_type] += 1.0

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2: SpaCy Matcher (when available)
        # ─────────────────────────────────────────────────────────────────────

        doc = None
        if self.matcher and NLP:
            doc = NLP(query)
            matches = self.matcher(doc)

            for match_id, start, end in matches:
                rule_name = NLP.vocab.strings[match_id]
                if rule_name == "MULTI_HOP":
                    scores[QueryType.MULTI_HOP] += self.config.classifier_spacy_weight
                elif rule_name == "COMPARISON":
                    scores[QueryType.COMPARISON] += self.config.classifier_spacy_weight

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3: Entity-density heuristic (multi-hop indicator)
        # ─────────────────────────────────────────────────────────────────────

        # High entity density suggests multi-hop reasoning.
        # Thresholds sourced from PlannerConfig (settings.yaml).
        # Re-use the doc parsed in Phase 2 if available to avoid a second call.
        if SPACY_AVAILABLE and NLP:
            if doc is None:
                doc = NLP(query)
            entity_count = len(doc.ents)
            noun_count = sum(1 for token in doc if token.pos_ in ("NOUN", "PROPN"))

            if (entity_count > self.config.entity_density_threshold
                    or noun_count > self.config.noun_density_threshold):
                scores[QueryType.MULTI_HOP] += self.config.classifier_entity_boost

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 4: Determine final query type
        # ─────────────────────────────────────────────────────────────────────

        max_score = max(scores.values())

        if max_score == 0:
            # No pattern matched → default to SINGLE_HOP
            logger.debug("classify: no pattern matched for '%s' → SINGLE_HOP", query[:80])
            return QueryType.SINGLE_HOP, self.config.classifier_fallback_confidence

        # Priority order resolves ties.
        # IMPORTANT: MULTI_HOP before TEMPORAL — year tokens in bridge questions
        # (e.g. "2014 S/S is the debut album of ... formed by who?") would
        # otherwise be incorrectly classified as TEMPORAL.
        priority = [
            QueryType.COMPARISON,
            QueryType.MULTI_HOP,
            QueryType.TEMPORAL,
            QueryType.INTERSECTION,
            QueryType.AGGREGATE,
        ]

        for qt in priority:
            if scores[qt] == max_score:
                confidence = min(
                    self.config.classifier_confidence_cap,
                    self.config.classifier_confidence_base
                    + (max_score * self.config.classifier_confidence_scale),
                )
                return qt, confidence

        # Should not be reached (SINGLE_HOP scores 0, caught above); guard
        # against floating-point edge cases.
        logger.debug("classify: priority exhausted for '%s' → SINGLE_HOP", query[:80])
        return QueryType.SINGLE_HOP, self.config.classifier_fallback_confidence


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class EntityExtractor:
    """
    Entity extractor using SpaCy NER and regex fallback.

    Per thesis Section 3.2:
    "Entity extraction uses SpaCy NER (confidence > min_entity_confidence).
    For complex queries, dependency parsing resolves syntactic relationships,
    enabling identification of bridge entities as necessary intermediate steps
    (hops) for graph traversal."
    (Honnibal & Montani, 2017; arXiv:1802.04016;
     Yang et al., 2018 HotpotQA EMNLP)
    """

    # SpaCy NER labels relevant to RAG queries
    RELEVANT_ENTITY_TYPES = frozenset({
        "PERSON",      # people
        "ORG",         # organisations
        "GPE",         # geo-political entities (countries, cities)
        "LOC",         # locations
        "PRODUCT",     # products
        "EVENT",       # events
        "WORK_OF_ART", # artworks, films, books
        "LAW",         # laws
        "DATE",        # dates
        "TIME",        # times
        "MONEY",       # monetary amounts
        "QUANTITY",    # quantities
        "NORP",        # nationalities, religious/political groups
    })

    # Regex fallback patterns for entity extraction when SpaCy is unavailable
    ENTITY_PATTERNS = (
        (r'"([^"]+)"',                             "QUOTED"),  # double-quoted strings
        (r"'([^']+)'",                             "QUOTED"),  # single-quoted strings
        (_PROPER_NOUN_RE.pattern,                   "PROPN"),   # multi-word proper nouns (shared from _settings)
        (r"\b([A-Z][a-z]{2,})\b",                  "PROPN"),   # single proper nouns
        (r"\b(\d{4})\b",                           "DATE"),    # four-digit years
    )

    # Common stopwords that must not be extracted as named entities.
    # Built once as a frozenset class constant to avoid per-call reconstruction
    # (called for every regex-matched candidate in extract()).
    _STOPWORDS: frozenset = frozenset({
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'however', 'therefore', 'furthermore', 'moreover',
        'although', 'because', 'since', 'while', 'when',
        'what', 'which', 'who', 'whom', 'whose', 'where',
        'how', 'why', 'if', 'then', 'else', 'but', 'and', 'or',
    })

    # Per-label confidence estimates.
    # Values are approximate — SpaCy does not expose per-entity confidence scores.
    # Derived from label-level reliability observed in SpaCy documentation and
    # en_core_web_sm evaluation on OntoNotes 5 (Weischedel et al., 2013):
    # high-precision labels (DATE, PERSON, GPE) receive higher base confidence
    # than ambiguous labels (WORK_OF_ART) which are frequently mis-categorised.
    # Length bonus (up to +0.1, at +0.03 per token) rewards unambiguous multi-word spans.
    _LABEL_CONFIDENCE: Dict[str, float] = {
        "PERSON":     0.9,
        "ORG":        0.85,
        "GPE":        0.9,
        "LOC":        0.85,
        "DATE":       0.95,
        "EVENT":      0.8,
        "WORK_OF_ART": 0.75,
    }

    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialise the Entity Extractor.

        Args:
            config: Planner configuration.
        """
        self.config = config or PlannerConfig()

        # Compile regex patterns once at construction time
        self._compiled_patterns = [
            (re.compile(pattern), label)
            for pattern, label in self.ENTITY_PATTERNS
        ]

        logger.info("EntityExtractor initialised")

    def extract(self, query: str) -> List[EntityInfo]:
        """
        Extract entities from a query.

        Uses SpaCy NER when available, with regex fallback for additional
        coverage or when SpaCy is not installed.

        Args:
            query: The query from which to extract entities.

        Returns:
            List of EntityInfo objects sorted by character position.
        """
        entities = []
        seen_texts: set = set()  # for deduplication

        # ─────────────────────────────────────────────────────────────────────
        # METHOD 1: SpaCy NER (when available)
        # ─────────────────────────────────────────────────────────────────────

        if SPACY_AVAILABLE and NLP:
            doc = NLP(query)

            for ent in doc.ents:
                if ent.label_ in self.RELEVANT_ENTITY_TYPES:
                    # SpaCy does not expose per-entity confidence scores;
                    # estimate from label type and entity length.
                    confidence = self._estimate_confidence(ent)

                    if confidence >= self.config.min_entity_confidence:
                        entity_text = ent.text.strip()

                        if entity_text.lower() not in seen_texts:
                            entities.append(EntityInfo(
                                text=entity_text,
                                label=ent.label_,
                                confidence=confidence,
                                start_char=ent.start_char,
                                end_char=ent.end_char,
                                is_bridge=False,
                            ))
                            seen_texts.add(entity_text.lower())

        # ─────────────────────────────────────────────────────────────────────
        # METHOD 2: Regex fallback (supplementary or primary)
        # ─────────────────────────────────────────────────────────────────────

        for pattern, label in self._compiled_patterns:
            for match in pattern.finditer(query):
                text = match.group(1) if match.lastindex else match.group(0)
                text = text.strip()

                if len(text) > 2 and text.lower() not in seen_texts:
                    if not self._is_stopword(text):
                        entities.append(EntityInfo(
                            text=text,
                            label=label,
                            # Regex confidence is lower than NER; value
                            # configurable via PlannerConfig.regex_entity_confidence
                            confidence=self.config.regex_entity_confidence,
                            start_char=match.start(),
                            end_char=match.end(),
                            is_bridge=False,
                        ))
                        seen_texts.add(text.lower())

        # Sort by position in text
        entities.sort(key=lambda e: e.start_char)

        # ─────────────────────────────────────────────────────────────────────
        # POST-PROCESSING: remove substring-duplicate entities
        # e.g. remove "Scott" when "Scott Derrickson" is already in the list;
        # remove "Wood" when "Ed Wood" is in the list.
        # Also filter out PROPN-labelled single tokens shorter than 5 chars to
        # avoid spurious hits like "Were", "Wood" from the regex fallback.
        # ─────────────────────────────────────────────────────────────────────
        all_texts_lower = [e.text.lower() for e in entities]
        filtered: List[EntityInfo] = []
        for entity in entities:
            txt_lower = entity.text.lower()
            # Drop short PROPN tokens — likely regex noise
            if entity.label == "PROPN" and len(entity.text) < 5:
                continue
            # Drop if any other entity's text contains this one as a substring
            # (only when this entity is a single token — multi-word spans are kept)
            if " " not in entity.text.strip():
                is_substring = any(
                    txt_lower != other_lower and txt_lower in other_lower
                    for other_lower in all_texts_lower
                )
                if is_substring:
                    continue
            filtered.append(entity)

        return filtered[:self.config.max_entities]

    def detect_bridge_entities(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> List[EntityInfo]:
        """
        Identify bridge entities for multi-hop reasoning.

        Bridge entities are intermediate nodes required for graph traversal.
        Per thesis Section 3.2 and Yang et al. (2018) HotpotQA EMNLP:
        "Bridge entities (e.g. subject-object relationships in nested sentences)
        serve as necessary intermediate steps (hops) for graph traversal."

        Detection uses SpaCy dependency parsing:
        - Prepositional objects in nested structures (dep=pobj)
        - Possessive modifiers in chains (dep=poss)
        - Relative clause subjects (dep=relcl)

        Args:
            query:    The original query.
            entities: Already-extracted entities.

        Returns:
            Entities with the is_bridge flag updated where appropriate.
        """
        if not self.config.enable_bridge_detection:
            logger.debug("detect_bridge_entities: bridge detection disabled by config")
            return entities

        if not SPACY_AVAILABLE or NLP is None or len(entities) < 2:
            logger.debug(
                "detect_bridge_entities: skipped (spacy=%s, entities=%d)",
                SPACY_AVAILABLE,
                len(entities),
            )
            return entities

        doc = NLP(query)
        bridge_candidates: set = set()

        for token in doc:
            # Prepositional objects inside nested structures
            if token.dep_ == "pobj" and token.head.dep_ == "prep":
                if token.head.head.pos_ in ("NOUN", "PROPN"):
                    bridge_candidates.add(token.text.lower())

            # Possessive chains (e.g. "John's sister's husband")
            if token.dep_ == "poss":
                bridge_candidates.add(token.text.lower())

            # Relative clause subjects
            if token.dep_ == "relcl":
                for child in token.children:
                    if child.dep_ == "nsubj":
                        bridge_candidates.add(child.text.lower())

        # Mark entities as bridge if in bridge_candidates.
        # First and last entities are typically anchors, not bridges.
        for i, entity in enumerate(entities):
            if entity.text.lower() in bridge_candidates:
                if 0 < i < len(entities) - 1:
                    entity.is_bridge = True

        return entities

    def _estimate_confidence(self, ent: "spacy.tokens.Span") -> float:
        """
        Estimate confidence for a SpaCy entity span.

        SpaCy does not expose native per-entity confidence scores.
        Confidence is estimated from label type (see _LABEL_CONFIDENCE) and
        entity length (longer spans are less ambiguous).

        Args:
            ent: A SpaCy Span object.

        Returns:
            Estimated confidence in [0.0, 1.0].
        """
        base = self._LABEL_CONFIDENCE.get(ent.label_, 0.7)
        length_bonus = min(0.1, len(ent.text.split()) * 0.03)
        return min(1.0, base + length_bonus)

    def _is_stopword(self, text: str) -> bool:
        """Return True if text is a common stopword that should not be extracted."""
        return text.lower() in self._STOPWORDS


# =============================================================================
# RETRIEVAL PLAN GENERATOR
# =============================================================================

class PlanGenerator:
    """
    Generator for structured retrieval plans.

    Produces a detailed plan for the Navigator (S_N) based on query
    classification and extracted entities.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # CLASS-LEVEL COMPILED CONSTANTS
    # ─────────────────────────────────────────────────────────────────────────

    # NER labels recognised as proper named entities (SpaCy OntoNotes label set).
    # Intentionally a strict subset of EntityExtractor.RELEVANT_ENTITY_TYPES:
    # LAW, TIME, MONEY, QUANTITY are excluded because comparison decomposition
    # operates on entities that refer to real-world objects with comparable
    # attributes, not temporal values or monetary amounts.
    # FAC (facility) is included as it is a named place (comparable location).
    _NER_LABELS: frozenset = frozenset({
        "PERSON", "GPE", "ORG", "LOC", "PRODUCT",
        "EVENT", "WORK_OF_ART", "NORP", "DATE", "FAC",
    })

    # Vague pronoun / generic reference pattern used in multi-hop enrichment.
    # Matches definite NP generics like "the director", "the woman", etc.
    # See _decompose_multi_hop Fall B for usage context.
    _VAGUE_REFS = re.compile(
        r'\b(the\s+(?:woman|man|person|actor|actress|director|author|artist'
        r'|president|player|team|group|band|company|film|movie|show|song|book))\b',
        re.IGNORECASE,
    )

    # Attribute rewriting map for comparison queries.
    # Transforms "Were X and Y of the same nationality?" into two factual lookups
    # "What is the nationality of X?" / "What is the nationality of Y?".
    # This improves vector similarity to factual chunks (e.g. "X is American").
    # 8 patterns covering the most frequent comparison attributes in HotpotQA
    # (Yang et al., 2018 EMNLP).
    _ATTR_MAP = (
        (re.compile(r'\bsame\s+nationality\b', re.IGNORECASE),
         "What is the nationality of {entity}?"),
        (re.compile(r'\bsame\s+(?:birth\s*place|birthplace|hometown)\b', re.IGNORECASE),
         "Where was {entity} born?"),
        (re.compile(r'\bborn\s+in\s+the\s+same\b', re.IGNORECASE),
         "Where was {entity} born?"),
        (re.compile(r'\bsame\s+(?:profession|occupation|job)\b', re.IGNORECASE),
         "What is the profession of {entity}?"),
        (re.compile(r'\bsame\s+(?:genre|style)\b', re.IGNORECASE),
         "What genre is {entity}?"),
        (re.compile(r'\bsame\s+(?:age|birth\s*year)\b', re.IGNORECASE),
         "When was {entity} born?"),
        (re.compile(r'\bsame\s+(?:country|state|city)\b', re.IGNORECASE),
         "What country is {entity} from?"),
        (re.compile(r'\bsame\s+(?:religion|faith|belief)\b', re.IGNORECASE),
         "What is the religion of {entity}?"),
    )

    # ── Pattern C: "for a/an/the [category] [description]" ──────────────────────
    # Handles hidden-bridge queries where the bridge entity is described but not
    # named, e.g. "What year did GNR do a promo for a movie starring Schwarzenegger?"
    # → bridge_query = "movie starring Schwarzenegger as NY police detective"
    # → final_query  = original (controller injects bridge name at runtime)
    _CATEGORY_WORDS = (
        "film|movie|show|song|album|game|book|video|series|documentary"
    )
    _FOR_CAT = re.compile(
        r'\bfor (?:a|an|the)\s+(?P<cat>' + _CATEGORY_WORDS + r')\b'
        r'(?P<desc>.{5,}?)(?:\?|$)',
        re.IGNORECASE,
    )

    # ── Pattern D: "What [role] with/having [qualifier] [verb]..." ─────────────
    # Handles: "What screenwriter with credits for X co-wrote a film ...?"
    # The role + qualifier identifies the unknown bridge person.
    _ROLE_PAT = re.compile(
        r'^(?:what|which|who)\s+(?P<role>\w+(?:\s+\w+)?)\s+'
        r'(?:with|having|known for|who)\s+(?P<qual>.+?)\s+'
        r'(?:co-?wrote|wrote|directed|produced|starred|co-?directed)\b',
        re.IGNORECASE,
    )

    # Pre-compiled regexes for _form_sub_query (called on every multi-hop step)
    _STRIP_LEADING_CONJ = re.compile(
        r"^(and|or|but|that|which|who|where)\s+", re.IGNORECASE
    )
    _INTERROGATIVE_PREFIX = re.compile(
        r"^(what|who|where|when|why|how|which|is|are|was|were|did|does|do)\b",
        re.IGNORECASE,
    )

    # Pre-compiled regexes for _extract_constraints (called on every query)
    _TEMPORAL_TERMS_RE = re.compile(
        r"\b(before|after|during|since|until|recent|latest|first|last)\b",
        re.IGNORECASE,
    )
    _COMPARISON_GREATER_RE = re.compile(
        r"\b(older|bigger|larger|more|higher|greater)\b", re.IGNORECASE
    )
    _COMPARISON_LESS_RE = re.compile(
        r"\b(younger|smaller|less|lower|fewer)\b", re.IGNORECASE
    )
    _COMPARISON_ATTR_RE = re.compile(
        r"\b(older|younger|taller|shorter|bigger|smaller|richer|poorer)\b",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialise the Plan Generator.

        Args:
            config: Planner configuration.
        """
        self.config = config or PlannerConfig()

    def generate(
        self,
        query: str,
        query_type: QueryType,
        confidence: float,
        entities: List[EntityInfo],
    ) -> RetrievalPlan:
        """
        Generate a retrieval plan.

        Args:
            query:      Original query.
            query_type: Classified query type.
            confidence: Classification confidence.
            entities:   Extracted entities.

        Returns:
            Complete RetrievalPlan.
        """
        strategy = self._determine_strategy(query_type, entities)
        hop_sequence, sub_queries = self._generate_hops(query, query_type, entities)
        constraints = self._extract_constraints(query, query_type)

        plan = RetrievalPlan(
            original_query=query,
            query_type=query_type,
            strategy=strategy,
            entities=entities,
            hop_sequence=hop_sequence,
            sub_queries=sub_queries,
            constraints=constraints,
            estimated_hops=len(hop_sequence),
            confidence=confidence,
            metadata={
                "entity_count": len(entities),
                "bridge_count": sum(1 for e in entities if e.is_bridge),
                "spacy_available": SPACY_AVAILABLE,
            },
        )

        logger.info(
            "Plan generated: type=%s strategy=%s hops=%d sub_queries=%d",
            query_type.value,
            strategy.value,
            len(hop_sequence),
            len(sub_queries),
        )

        return plan

    def _determine_strategy(
        self,
        query_type: QueryType,
        entities: List[EntityInfo],
    ) -> RetrievalStrategy:
        """
        Select the optimal retrieval strategy for the given query type.

        Per thesis Section 3.2:
        - VECTOR_ONLY: For simple single-hop queries (fast path).
        - HYBRID:      For all complex query types requiring graph traversal.

        Note: RetrievalStrategy.GRAPH_ONLY is reserved for future work and
        is not currently selected by this method. A dedicated graph-only
        ablation path would require explicit graph-relation queries (e.g.
        "Who is the founder of X?") to be routed here; this is left as a
        planned extension for the thesis evaluation.
        """
        # Simple queries with at most one entity → vector search is sufficient
        if query_type == QueryType.SINGLE_HOP and len(entities) <= 1:
            return RetrievalStrategy.VECTOR_ONLY

        # All multi-hop queries use HYBRID regardless of bridge-entity presence.
        # Even without detected bridges the graph may surface indirect relations.
        if query_type == QueryType.MULTI_HOP:
            return RetrievalStrategy.HYBRID

        # Comparison and intersection → parallel retrieval + graph relations
        if query_type in (QueryType.COMPARISON, QueryType.INTERSECTION):
            return RetrievalStrategy.HYBRID

        # Temporal and aggregate → graph can provide structured time-stamped facts
        if query_type in (QueryType.TEMPORAL, QueryType.AGGREGATE):
            return RetrievalStrategy.HYBRID

        return self.config.default_strategy

    def _generate_hops(
        self,
        query: str,
        query_type: QueryType,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Generate the hop sequence and sub-query list.

        Returns an ordered sequence of retrieval steps with dependencies
        for multi-hop reasoning.
        """
        hop_sequence = []
        sub_queries = []

        if query_type == QueryType.SINGLE_HOP:
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False,
            ))
            sub_queries = [query]

        elif query_type == QueryType.MULTI_HOP:
            hop_sequence, sub_queries = self._decompose_multi_hop(query, entities)

        elif query_type == QueryType.COMPARISON:
            hop_sequence, sub_queries = self._decompose_comparison(query, entities)

        elif query_type == QueryType.INTERSECTION:
            hop_sequence, sub_queries = self._decompose_intersection(query, entities)

        elif query_type == QueryType.TEMPORAL:
            hop_sequence, sub_queries = self._decompose_temporal(query, entities)

        elif query_type == QueryType.AGGREGATE:
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False,
            ))
            sub_queries = [query]

        return hop_sequence, sub_queries

    def _decompose_multi_hop(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose a multi-hop query into ordered retrieval steps.

        Thesis Section 4.2 — Sub-query enrichment strategy:

        The query is split at bridge connectors ("that", "which", "who",
        "of the") to isolate the anchor part (contains named entities) from
        the bridge part (contains the unknown to resolve).

        After splitting, parts are reversed so that the bridge step (which
        must be resolved first using the anchor entities) comes before the
        final step.

        Two enrichment cases handle missing entity context:

        Fall A — Bridge step without entities:
          "was formed by who?" has no named entity. Donor entities are taken
          from the other (anchor) parts and prepended:
          → "2014 S/S was formed by who?"

        Fall B — Final step with only vague generic references:
          "What position was held by the woman?" contains no proper name.
          The vague NP is replaced with the bridge-part entity:
          → "What position was held by Shirley Temple?"
        """
        hop_sequence = []
        sub_queries = []

        # Split at bridge connectors; maxsplit=1 per pattern preserves the
        # full tail of the query on the right side of the split.
        split_patterns = [
            r"\s+(that|which|who)\s+",
            r"\s+of\s+the\s+",
        ]

        parts = [query]
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                split_result = re.split(pattern, part, maxsplit=1, flags=re.IGNORECASE)
                new_parts.extend(split_result)
            parts = [p.strip() for p in new_parts if p.strip() and len(p.strip()) > 5]

        # ── Pattern C: "for a/an/the [category] [description]" ──────────────────
        # See class-level _FOR_CAT for full rationale.
        if len(parts) <= 1:
            cm = self._FOR_CAT.search(query)
            if cm:
                bridge_q = f"{cm.group('cat')} {cm.group('desc').strip()}"
                hop_sequence = [
                    HopStep(
                        step_id=0,
                        sub_query=bridge_q,
                        target_entities=[e.text for e in entities],
                        depends_on=[],
                        is_bridge=True,
                    ),
                    HopStep(
                        step_id=1,
                        sub_query=query,
                        target_entities=[e.text for e in entities],
                        depends_on=[0],
                        is_bridge=False,
                    ),
                ]
                logger.debug(
                    "_decompose_multi_hop: Pattern C bridge → %r", bridge_q[:60]
                )
                return hop_sequence, [bridge_q, query]

        # ── Pattern D: "What [role] with/having [qualifier] [verb]..." ────────
        # See class-level _ROLE_PAT for full rationale.
        if len(parts) <= 1 and entities:
            rm = self._ROLE_PAT.match(query)
            if rm:
                bridge_q = f"{rm.group('role')} {rm.group('qual').strip()}"
                hop_sequence = [
                    HopStep(
                        step_id=0,
                        sub_query=bridge_q,
                        target_entities=[e.text for e in entities],
                        depends_on=[],
                        is_bridge=True,
                    ),
                    HopStep(
                        step_id=1,
                        sub_query=query,
                        target_entities=[e.text for e in entities],
                        depends_on=[0],
                        is_bridge=False,
                    ),
                ]
                logger.debug(
                    "_decompose_multi_hop: Pattern D role-bridge → %r", bridge_q[:60]
                )
                return hop_sequence, [bridge_q, query]

        if len(parts) > 1:
            reversed_parts = list(reversed(parts))

            for i, part in enumerate(reversed_parts):
                depends = list(range(i)) if i > 0 else []
                part_entities = [
                    e.text for e in entities
                    if e.text.lower() in part.lower()
                ]

                enriched_part = part
                is_bridge_step = (i < len(parts) - 1)
                is_final_step  = (i == len(parts) - 1)

                if is_bridge_step and not part_entities:
                    # Fall A: bridge step has no known entities; borrow from anchor.
                    # other_parts = all non-bridge parts (anchor side); may be >1.
                    other_parts = reversed_parts[1:]
                    donor_entity_texts = [
                        e.text for e in entities
                        if any(e.text.lower() in ap.lower() for ap in other_parts)
                        and e.text.lower() not in part.lower()
                    ]
                    # Secondary fallback: extract a noun phrase from the anchor part
                    if not donor_entity_texts:
                        for op in other_parts:
                            m = re.search(
                                r'\b(?:a|an|the)\s+((?:\w+\s+){1,3}'
                                r'(?:group|band|company|team|label|artist|person|film|movie|show))\b',
                                op, re.IGNORECASE,
                            )
                            if m:
                                donor_entity_texts.append(m.group(1).strip())
                                break
                    if donor_entity_texts:
                        ctx = " ".join(donor_entity_texts[:2])
                        enriched_part = f"{ctx} {part}"

                elif is_final_step and self._VAGUE_REFS.search(part) and not part_entities:
                    # Fall B: final step contains only a vague generic; replace with entity
                    bridge_parts = reversed_parts[:i]
                    donor_entity_texts = [
                        e.text for e in entities
                        if any(e.text.lower() in bp.lower() for bp in bridge_parts)
                        and e.text.lower() not in part.lower()
                    ]
                    if donor_entity_texts:
                        ctx = " ".join(donor_entity_texts[:2])
                        enriched_part = self._VAGUE_REFS.sub(ctx, part, count=1)

                sub_query = self._form_sub_query(enriched_part)

                hop_sequence.append(HopStep(
                    step_id=i,
                    sub_query=sub_query,
                    target_entities=part_entities,
                    depends_on=depends,
                    is_bridge=(i < len(parts) - 1),
                ))
                sub_queries.append(sub_query)
        else:
            # Could not split → treat as single hop
            logger.debug("_decompose_multi_hop: split failed for '%s' → single hop", query[:80])
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False,
            ))
            sub_queries = [query]

        return hop_sequence, sub_queries

    def _decompose_comparison(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose a comparison query into parallel retrieval steps.

        Strategy:
        1. Identify the two primary entities to compare (prefer NER over regex).
        2. Generate one sub-query per entity (can run in parallel).
        3. Apply attribute rewriting (_ATTR_MAP) to improve vector similarity.
        4. Append the original query as the final comparison step.

        Edge case — zero entities detected:
          If no named entities are found (SpaCy below threshold, regex filtered),
          the original query is used as a single retrieval step. The Navigator
          will retrieve broadly and the Verifier synthesises from the context.
        """
        hop_sequence = []
        sub_queries = []

        # Use only proper NER entities; regex-PROPN entities include noisy
        # sentence-initial tokens and are filtered out here.
        ner_entities = [e for e in entities if e.label in self._NER_LABELS]

        if len(ner_entities) >= 2:
            comparison_entities = ner_entities[:2]
        else:
            # Fallback: greedily select non-overlapping entities from the full list
            selected: List[EntityInfo] = list(ner_entities)
            for e in entities:
                if len(selected) >= 2:
                    break
                overlaps = any(
                    not (e.end_char <= sel.start_char or e.start_char >= sel.end_char)
                    for sel in selected
                )
                if not overlaps and e not in selected:
                    selected.append(e)
            comparison_entities = selected[:2]

        # ── Zero-entity guard ─────────────────────────────────────────────────
        # If no entities were detected at all, fall back to using the original
        # query as the sole sub-query. The Navigator retrieves broadly and the
        # Verifier synthesises the comparison from the retrieved context.
        if not comparison_entities:
            logger.debug(
                "_decompose_comparison: no entities detected for '%s' → single fallback step",
                query[:80],
            )
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[],
                depends_on=[],
                is_bridge=False,
            ))
            return hop_sequence, [query]

        sub_query_templates = []

        if len(comparison_entities) >= 2:
            idx = [query.find(e.text) for e in comparison_entities]
            if all(i >= 0 for i in idx):
                span_start = min(idx)
                span_end   = max(
                    idx[j] + len(comparison_entities[j].text)
                    for j in range(len(comparison_entities))
                )
                prefix = query[:span_start]
                suffix = query[span_end:]
                for e in comparison_entities:
                    sq = re.sub(r'\s+', ' ', (prefix + e.text + suffix).strip())
                    sub_query_templates.append((e, sq))

        if not sub_query_templates:
            # Fallback: entity positions not found in query string
            # (e.g. entity text was normalised or contains special characters)
            logger.debug(
                "_decompose_comparison: entity position not found for query '%s'"
                " → using generic template",
                query[:80],
            )
            sub_query_templates = [
                (e, f"What is {e.text}?") for e in comparison_entities
            ]

        # Attribute rewriting: "Were X of the same nationality?" →
        # "What is the nationality of X?" (per _ATTR_MAP above).
        # Improves vector similarity to factual chunks such as
        # "X is an American filmmaker" (Yang et al., 2018 EMNLP).
        rewritten = []
        for pattern, template in self._ATTR_MAP:
            if pattern.search(query):
                rewritten = [
                    (e, template.format(entity=e.text))
                    for e, _ in sub_query_templates
                ]
                break
        if rewritten:
            sub_query_templates = rewritten

        # One step per entity (steps are independent → can run in parallel).
        # NOTE: The original query is intentionally NOT appended as a third
        # sub-query.  Empirical diagnosis (idx=0, diagnose_verbose.py) showed
        # that re-issuing the full comparison question as a retrieval query
        # produces noise hits (John MacGregor, Gerald R. Ford Freeway) and
        # inflates RRF scores for irrelevant chunks via the cross-query boost.
        # The two entity-specific attribute-template queries are sufficient.
        for i, (entity, sub_query) in enumerate(sub_query_templates):
            hop_sequence.append(HopStep(
                step_id=i,
                sub_query=sub_query,
                target_entities=[entity.text],
                depends_on=[],  # no dependency → parallel execution
                is_bridge=True,
            ))
            sub_queries.append(sub_query)

        return hop_sequence, sub_queries

    def _decompose_intersection(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose an intersection query.

        Intersection and comparison share identical retrieval decomposition:
        both require parallel per-entity lookups followed by a synthesis step.
        The difference lies in the Verifier's synthesis step (intersection
        identifies shared attributes; comparison ranks or contrasts), not in
        the retrieval plan structure. Both therefore route through
        _decompose_comparison.
        """
        return self._decompose_comparison(query, entities)

    def _decompose_temporal(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose a temporal query.

        Temporal queries are typically single-hop with a temporal constraint;
        the time component is captured in the constraints dict rather than
        as a separate hop.
        """
        hop_sequence = [
            HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False,
            )
        ]
        return hop_sequence, [query]

    def _form_sub_query(self, part: str) -> str:
        """Convert a query fragment into a well-formed sub-query."""
        part = part.strip()

        # Remove leading conjunctions left over from splitting
        part = self._STRIP_LEADING_CONJ.sub("", part)

        # Ensure the fragment ends with "?" and has an interrogative prefix
        if not part.endswith("?"):
            if not self._INTERROGATIVE_PREFIX.match(part):
                part = f"What is {part}?"
            else:
                part = f"{part}?"

        return part

    def _extract_constraints(
        self,
        query: str,
        query_type: QueryType,
    ) -> Dict[str, Any]:
        """
        Extract constraints from the query.

        Constraints are additional conditions such as:
        - Temporal: year ranges, date references.
        - Comparison: direction (greater/less) and attribute.

        Args:
            query:      The user query.
            query_type: Classified query type.

        Returns:
            Dict of constraint key → value (may be empty).
        """
        constraints: Dict[str, Any] = {}

        # ─────────────────────────────────────────────────────────────────────
        # TEMPORAL CONSTRAINTS
        # ─────────────────────────────────────────────────────────────────────

        if query_type == QueryType.TEMPORAL or self.config.enable_temporal_parsing:
            # Extract historical / 21st-century years using module-level constant
            years = _YEAR_RE.findall(query)
            if years:
                constraints["years"] = years

            temporal_match = self._TEMPORAL_TERMS_RE.search(query)
            if temporal_match:
                constraints["temporal_relation"] = temporal_match.group(1).lower()

        # ─────────────────────────────────────────────────────────────────────
        # COMPARISON CONSTRAINTS
        # ─────────────────────────────────────────────────────────────────────

        if query_type == QueryType.COMPARISON:
            if self._COMPARISON_GREATER_RE.search(query):
                constraints["comparison_direction"] = "greater"
            elif self._COMPARISON_LESS_RE.search(query):
                constraints["comparison_direction"] = "less"

            attr_match = self._COMPARISON_ATTR_RE.search(query)
            if attr_match:
                constraints["comparison_attribute"] = attr_match.group(1).lower()

        return constraints


# =============================================================================
# MAIN PLANNER CLASS
# =============================================================================

class Planner:
    """
    S_P: Rule-based Query Planner.

    Orchestrates query classification, entity extraction, and plan generation.

    Usage:
        planner = Planner()
        plan = planner.plan("Who directed the movie with Tom Hanks?")
        sub_queries = planner.decompose_query("Is Berlin older than Munich?")
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        # Kept for API compatibility with LLM-based planner signatures
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialise the Planner.

        Args:
            config:     PlannerConfig (optional).
            model_name: Ignored (API compatibility shim).
            base_url:   Ignored (API compatibility shim).
        """
        self.config = config or PlannerConfig()

        self.classifier       = QueryClassifier(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.plan_generator   = PlanGenerator(self.config)

        logger.info(
            "Planner initialised: SpaCy=%s",
            "available" if SPACY_AVAILABLE else "unavailable",
        )

    def plan(self, query: str) -> RetrievalPlan:
        """
        Generate a complete retrieval plan for a query.

        This is the primary entry point for the Planner. It runs the full
        classification → entity extraction → plan generation pipeline and
        returns a structured RetrievalPlan for the Navigator (S_N).

        Args:
            query: The user query.

        Returns:
            RetrievalPlan with strategy, entities, hops, and constraints.
        """
        start_time = time.perf_counter()

        if query is None:
            return self._empty_plan("")
        query = query.strip()
        if not query:
            return self._empty_plan(query)

        # Step 1: Query classification
        query_type, confidence = self.classifier.classify(query)
        logger.debug("Query classified: %s (conf=%.2f)", query_type.value, confidence)

        # Step 2: Entity extraction
        entities = self.entity_extractor.extract(query)

        if query_type == QueryType.MULTI_HOP:
            entities = self.entity_extractor.detect_bridge_entities(query, entities)

        logger.debug("Entities extracted: %d", len(entities))

        # Step 3: Plan generation
        plan = self.plan_generator.generate(
            query=query,
            query_type=query_type,
            confidence=confidence,
            entities=entities,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        plan.metadata["planning_time_ms"] = elapsed_ms

        logger.info(
            "Plan generated in %.1fms: type=%s entities=%d hops=%d",
            elapsed_ms,
            query_type.value,
            len(entities),
            plan.estimated_hops,
        )

        return plan

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a query into sub-queries.

        Simplified entry point for agent compatibility.
        Returns only the flat sub-query list.

        Args:
            query: The user query.

        Returns:
            List of sub-queries for retrieval.
        """
        plan = self.plan(query)
        return plan.sub_queries

    def _empty_plan(self, query: str) -> RetrievalPlan:
        """Return a minimal plan for empty or invalid queries."""
        return RetrievalPlan(
            original_query=query,
            query_type=QueryType.SINGLE_HOP,
            strategy=RetrievalStrategy.VECTOR_ONLY,
            entities=[],
            hop_sequence=[],
            sub_queries=[query] if query else [],
            constraints={},
            estimated_hops=0,  # consistent with empty hop_sequence
            confidence=0.0,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_planner(
    cfg: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,  # ignored — API compatibility
    base_url: Optional[str] = None,    # ignored — API compatibility
) -> Planner:
    """
    Factory function for Planner.

    When ``cfg`` is None, settings.yaml is auto-loaded from
    ``config/settings.yaml`` relative to the project root — so a bare
    ``create_planner()`` call always picks up the live settings.yaml values
    without any hardcoded fallbacks in the call site.

    ``model_name`` and ``base_url`` are accepted but ignored for API
    compatibility with LLM-based planner signatures (the Planner is
    rule-based and does not call an LLM).

    Args:
        cfg:        Full settings.yaml dict.  Auto-loaded when None.
        model_name: Ignored (API compatibility shim).
        base_url:   Ignored (API compatibility shim).

    Returns:
        Configured Planner instance.
    """
    if cfg is None:
        cfg = _load_settings()
    config = PlannerConfig.from_yaml(cfg)
    return Planner(config=config)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    test_queries = [
        ("What is the capital of France?",                                  QueryType.SINGLE_HOP),
        ("Who is the director of the film that stars Tom Hanks?",           QueryType.MULTI_HOP),
        ("What is the capital of the country where Einstein was born?",     QueryType.MULTI_HOP),
        ("Is Berlin older than Munich?",                                    QueryType.COMPARISON),
        ("Which is taller, the Eiffel Tower or Big Ben?",                   QueryType.COMPARISON),
        ("What happened after World War 2?",                                QueryType.TEMPORAL),
        ("Who was president in 1990?",                                      QueryType.TEMPORAL),
        ("Which movies star both Brad Pitt and Leonardo DiCaprio?",         QueryType.INTERSECTION),
    ]

    print("=" * 70)
    print("S_P: RULE-BASED QUERY PLANNER SMOKE TEST")
    print(f"SpaCy available: {SPACY_AVAILABLE}")
    print("=" * 70)

    planner = create_planner()  # auto-loads config/settings.yaml

    total_time = 0
    correct = 0

    for query, expected_type in test_queries:
        plan = planner.plan(query)
        elapsed = plan.metadata.get("planning_time_ms", 0)
        total_time += elapsed

        is_correct = plan.query_type == expected_type
        correct += int(is_correct)
        status = "OK" if is_correct else "FAIL"

        print(f"\n[{status}] {query}")
        print(f"  Expected: {expected_type.value}  Got: {plan.query_type.value}")
        print(f"  Strategy: {plan.strategy.value}")
        print(f"  Entities: {[e.text for e in plan.entities]}")
        print(f"  Sub-queries: {plan.sub_queries}")
        print(f"  Hops: {plan.estimated_hops}  Time: {elapsed:.1f}ms")

    print("\n" + "=" * 70)
    print(
        f"Accuracy: {correct}/{len(test_queries)} "
        f"({100 * correct / len(test_queries):.1f}%)"
    )
    print(f"Average planning time: {total_time / len(test_queries):.1f}ms")
    print("=" * 70)
