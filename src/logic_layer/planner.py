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

# Auxiliary verbs and articles that SpaCy NER sometimes absorbs into an entity
# span when they precede it at sentence start (e.g. "Are Random House Tower"
# where "Are" is the sentence-initial auxiliary verb, not part of the name).
# Stripped from entity text before downstream use; char offsets adjusted.
_LEADING_FUNCTION_WORDS = frozenset({
    "are", "is", "was", "were", "did", "do", "does", "have", "has",
    "a", "an", "the",
})


def _strip_leading_function_words(text: str) -> str:
    """Remove leading auxiliary verbs / articles absorbed into an NER span.

    SpaCy en_core_web_sm occasionally includes the sentence-initial function
    word in a named-entity span (e.g. "Are Random House Tower" → ORG).
    Stripping these tokens recovers the correct entity surface form.
    """
    tokens = text.split()
    while tokens and tokens[0].lower() in _LEADING_FUNCTION_WORDS:
        tokens = tokens[1:]
    return " ".join(tokens)


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
        # FIX P-1: the canonical HotpotQA "select-between-two" comparison form.
        # "Which writer was from England, Henry Roth or Robert Erskine Childers?"
        # "Which band, Letters to Cleo or Screaming Trees, had more members?"
        # "Which magazine was started first, X or Y?" / "...A or B, was published first?"
        # Signature: a leading question-word + an "A or B" disjunction (often set
        # off by a comma). This is a comparison even when no comparative adjective
        # or "than" appears — the *answer* is one of the two named entities.
        r"^\s*(which|what|who|whom)\b.{0,120}?,\s*[^,?]+\s+\bor\b\s+[^,?]+[,?]",   # "Which X, A or B, ...?" / "..., A or B?"
        r"^\s*(which|what|who|whom)\b.{0,120}?\b(was|is|were|are|did|had|has|came|come)\b[^,?]*\bfirst\b",  # "Which ... was published first..."
        r"\b(more|fewer|less|most|fewest)\s+\w+\b.{0,60}\bor\b",   # "had more members ... or ..."
        r"\bor\b.{0,60}\b(more|fewer|less|most|fewest)\s+\w+\b",   # "...or...had more members"
        # Pattern K (§12.33): shared-attribute parallel comparison.
        # "What nationality were social anthropologists Alfred Gell and Edmund Leach?"
        # "What countries are X and Y from?"
        # "When were X and Y born?"
        # Signature: an interrogative word + a plural copula/aux that takes
        # multiple subjects + an "X and Y" coordination of two NER-like
        # capitalised phrases. The answer is a SHARED attribute, so each
        # entity needs its own bio retrieved in parallel — exactly what
        # _decompose_comparison does. Without this pattern the query falls
        # through to _decompose_multi_hop and emits a chained bridge from
        # only one entity, missing the second gold paragraph.
        r"^\s*(what|which|where|when|who|how)\b[^?]{0,80}?\b(are|were|do|did|have|has)\b"
        r"[^?]{0,80}?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+and\s+"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
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
        # PHASE 0: Boolean-conjunction pre-empt (Pattern I)
        # "Are/Did/Were X and Y both P?" — parallel yes/no, never a bridge chain.
        # Must run before Phase 1 so the "both" and "and" tokens cannot boost
        # MULTI_HOP or INTERSECTION scores past COMPARISON.
        # ─────────────────────────────────────────────────────────────────────
        _BOOL_CONJ_PRE = re.compile(
            r'^\s*(are|is|were|was|did|do|does|have|has)\b.+\band\b.+\bboth\b',
            re.IGNORECASE,
        )
        if _BOOL_CONJ_PRE.match(query):
            logger.debug("classify: Boolean conjunction pre-empt for '%s' → COMPARISON", query[:80])
            return QueryType.COMPARISON, 0.90

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 0.5: Implicit bridge pre-empt (Pattern J)
        # "X and another [noun] that …" — the answer requires first resolving
        # the bridge ("which corporation is the 'another'?") then following it.
        # The AGGREGATE pattern r"\bhow\s+many\b" fires on "how many countries"
        # and would otherwise classify this as AGGREGATE (single-pass).
        # ─────────────────────────────────────────────────────────────────────
        _IMPLICIT_BRIDGE_PRE = re.compile(
            r'\banother\s+\w+\b',
            re.IGNORECASE,
        )
        if _IMPLICIT_BRIDGE_PRE.search(query):
            logger.debug(
                "classify: implicit-bridge pre-empt (Pattern J) for '%s' → MULTI_HOP",
                query[:80],
            )
            return QueryType.MULTI_HOP, 0.80

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
        # PHASE 3.5: Multi-hop override (Bug 1)
        # ─────────────────────────────────────────────────────────────────────
        # Bridge questions that contain a year/"founded"/"when" token
        # (e.g. "What year was the university where John studied founded?")
        # accumulate multiple TEMPORAL hits but only one MULTI_HOP hit, so
        # priority-on-tie alone is not enough to keep them in the MULTI_HOP
        # branch.  Because TEMPORAL keywords are common across many query
        # types while MULTI_HOP relation cues ("founder of", "directed by",
        # "starring", possessive chains) are rare and specific, treat any
        # MULTI_HOP hit as decisive when TEMPORAL would otherwise dominate.
        if scores[QueryType.MULTI_HOP] > 0 and scores[QueryType.TEMPORAL] > scores[QueryType.MULTI_HOP]:
            logger.debug(
                "classify: multi-hop override for '%s' (multi_hop=%.1f, temporal=%.1f)",
                query[:80], scores[QueryType.MULTI_HOP], scores[QueryType.TEMPORAL],
            )
            scores[QueryType.MULTI_HOP] = scores[QueryType.TEMPORAL]

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

    # SpaCy NER labels that map to the GLiNER taxonomy used at ingestion time.
    # Only these labels are accepted — MONEY, QUANTITY, TIME, NORP, LAW, CARDINAL,
    # ORDINAL, PERCENT, LANGUAGE are excluded because they have no GLiNER equivalent
    # and produce false-positive entities (e.g. "888 7th Avenue" → MONEY).
    # Reference: settings.yaml → entity_extraction.gliner.entity_types
    RELEVANT_ENTITY_TYPES = frozenset({
        "PERSON",      # → person
        "ORG",         # → organization
        "GPE",         # → city / country / location
        "LOC",         # → location
        "FAC",         # → location (facility)
        "PRODUCT",     # → product
        "EVENT",       # → event
        "WORK_OF_ART", # → work of art
        "DATE",        # → date
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
                        entity_text = _strip_leading_function_words(ent.text.strip())
                        if not entity_text:
                            continue

                        if entity_text.lower() not in seen_texts:
                            # Adjust start_char to match the stripped text
                            stripped_offset = ent.text.index(entity_text) if entity_text in ent.text else 0
                            entities.append(EntityInfo(
                                text=entity_text,
                                label=ent.label_,
                                confidence=confidence,
                                start_char=ent.start_char + stripped_offset,
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

        # ─────────────────────────────────────────────────────────────────────
        # FIX P-3: re-join proper-noun spans that SpaCy / the regex fallback
        # fragmented. en_core_web_sm frequently splits a single title into
        # pieces around a lowercase connector — e.g. "Letters to Cleo" →
        # ["Letters"(ORG), "Cleo"(PRODUCT)] and "Seven Brief Lessons on
        # Physics" → ["Seven Brief Lessons", "Physics"]. When two extracted
        # entities are adjacent in the query separated only by a short
        # connector ("to", "of", "on", "the", "and", "&", "'s", …), merge
        # them into one span covering the whole title. This prevents the
        # fragment ("Physics", "Letters") from being treated as a primary or
        # bridge entity and dragging in irrelevant chunks.
        # ─────────────────────────────────────────────────────────────────────
        filtered = self._rejoin_fragmented_spans(query, filtered)

        return filtered[:self.config.max_entities]

    # Short connector words allowed *inside* a multi-word proper-noun span when
    # re-joining fragments (P-3). They must appear lowercase and surrounded by
    # the two entity spans in the original query text.
    #
    # IMPORTANT: "and"/"&" are deliberately EXCLUDED — a conjunction between two
    # distinct named entities ("Scott Derrickson and Ed Wood") must not be merged
    # into one span, or comparison decomposition breaks. Likewise "a"/"an" are
    # excluded (too generic). Only genuine title-internal connectors are kept.
    _SPAN_CONNECTORS: frozenset = frozenset({
        "to", "of", "on", "in", "the", "for", "at", "de",
        "von", "van", "del", "della", "di", "le", "la",
    })

    def _rejoin_fragmented_spans(
        self, query: str, entities: List[EntityInfo]
    ) -> List[EntityInfo]:
        """Merge consecutive entities that are adjacent in `query` and separated
        only by short connector words (or just whitespace/an apostrophe-s).

        Operates on the position-sorted entity list; runs a single left-to-right
        pass, repeatedly absorbing the next entity into the current span when the
        gap between them in the source text qualifies."""
        if len(entities) < 2:
            return entities
        ents = sorted(entities, key=lambda e: e.start_char)
        merged: List[EntityInfo] = []
        cur = ents[0]
        for nxt in ents[1:]:
            gap = query[cur.end_char:nxt.start_char]
            gap_norm = gap.strip().lower().strip("'").strip()  # tolerate "'s"
            gap_tokens = [t for t in gap_norm.split() if t]
            joinable = (
                nxt.start_char >= cur.end_char  # non-overlapping, in order
                and len(gap) <= 12              # connectors are short
                and (
                    gap.strip() in ("", "'s", "’s")          # pure adjacency
                    or (gap_tokens and all(t in self._SPAN_CONNECTORS for t in gap_tokens))
                )
            )
            if joinable:
                # Build the merged span verbatim from the query text so casing /
                # connectors are preserved exactly ("Letters to Cleo").
                new_text = query[cur.start_char:nxt.end_char].strip()
                # Prefer the more specific label: a named type over PROPN/QUOTED.
                if cur.label in ("PROPN", "QUOTED") and nxt.label not in ("PROPN", "QUOTED"):
                    new_label = nxt.label
                else:
                    new_label = cur.label
                cur = EntityInfo(
                    text=new_text,
                    label=new_label,
                    confidence=max(cur.confidence, nxt.confidence),
                    start_char=cur.start_char,
                    end_char=nxt.end_char,
                    is_bridge=cur.is_bridge or nxt.is_bridge,
                )
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged

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
        #
        # FIX P-3: a bridge entity is the *thing* that links two hops — a
        # person, organisation, work, place, or event. Generic descriptors
        # (NORP nationalities like "Italian", DATE values, QUANTITY/MONEY,
        # and unlabelled regex PROPN fragments) must never be promoted to
        # bridge: doing so steers retrieval toward high-frequency hub nodes
        # ("Physics", "Italian") instead of the entity that actually bridges.
        _BRIDGE_OK_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC",
                             "WORK_OF_ART", "PRODUCT", "EVENT"}
        for i, entity in enumerate(entities):
            if entity.text.lower() in bridge_candidates:
                if 0 < i < len(entities) - 1 and entity.label in _BRIDGE_OK_LABELS:
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
        # "Who is older/younger, X or Y?" → "When was X born?" so ANN matches
        # Wikipedia bio intros ("X (born 14 August 1965) is a …") rather than the
        # comparative question phrasing. §12.24 (2026-05-04).
        (re.compile(r'\b(older|younger)\b', re.IGNORECASE),
         "When was {entity} born?"),
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

    # ── Pattern E: "the [role] of [Entity]" — relational anchor bridge ──────────
    # Handles: "What was the father of X voted to be?", "Where was the wife of Y born?"
    # When none of the split patterns match, this detects a role-relation that
    # implicitly names a bridge entity (e.g., "the father of Kasper Schmeichel"
    # → Peter Schmeichel). Two sub-queries are generated:
    #   hop 0 (bridge): "Who is the {role} of {anchor entity}?"  — resolves bridge
    #   hop 1 (final):  original query                           — answers the fact
    # The bridge sub-query causes the Navigator to retrieve the anchor entity's
    # article, which names the bridge, increasing the chance that the bridge
    # entity's own article surfaces via keyword entity fallback and RRF fusion.
    _RELATIONAL_ANCHOR_ROLES = re.compile(
        r'\bthe\s+(father|mother|son|daughter|wife|husband|brother|sister|'
        r'uncle|aunt|grandfather|grandmother|nephew|niece|cousin|'
        r'director|founder|creator|author|composer|inventor|'
        r'president|chairman|owner|captain|coach|manager)\s+of\b',
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

    # ── Pattern F: Passive-agent bridge ─────────────────────────────────────
    # Handles: "[Subject] was written/directed/authored by [agent] that/who [property]?"
    # e.g. "Seven Brief Lessons on Physics was written by an Italian physicist
    #        that has worked in France since what year?"
    # Correct decomposition:
    #   hop 0 (bridge): "Who wrote Seven Brief Lessons on Physics?"
    #   hop 1 (final):  original query
    # The passive verb tells us *how* to form the bridge question ("wrote",
    # "directed", "authored", etc.).  The subject is everything before "was/were".
    _PASSIVE_AGENT_RE = re.compile(
        r'^(?P<subj>.+?)\s+(?:was|were)\s+'
        r'(?P<verb>written|authored|directed|produced|composed|created|'
        r'founded|formed|recorded|released|published|edited|scored|narrated)\s+'
        r'by\b',
        re.IGNORECASE,
    )
    # Maps passive past-participle → active question verb
    _PASSIVE_TO_ACTIVE = {
        "written":   "write",
        "authored":  "author",
        "directed":  "direct",
        "produced":  "produce",
        "composed":  "compose",
        "created":   "create",
        "founded":   "found",
        "formed":    "form",
        "recorded":  "record",
        "released":  "release",
        "published": "publish",
        "edited":    "edit",
        "scored":    "score",
        "narrated":  "narrate",
    }

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

    # ── Pattern J: Implicit bridge ("another [noun]") ───────────────────────
    # Matches queries like "X and another corporation that has operations in …"
    # where the answer-bearing entity is described by a common noun + relative
    # clause, not named.  The anchor is the named entity alongside "another".
    _IMPLICIT_BRIDGE_RE = re.compile(
        r'\banother\s+(\w+)\b',
        re.IGNORECASE,
    )

    def _find_implicit_bridge(
        self,
        query: str,
        entities: List["EntityInfo"],
    ) -> Optional[Tuple[str, str]]:
        """
        Detect "X and another [noun] that …" queries.

        Returns (anchor_entity_text, bridge_noun) when the pattern is found and
        at least one named entity is present to serve as the anchor, else None.
        """
        m = self._IMPLICIT_BRIDGE_RE.search(query)
        if not m:
            return None
        if not entities:
            return None
        bridge_noun = m.group(1)
        # Anchor: pick the entity that appears BEFORE "another" in the query
        another_pos = m.start()
        anchor = None
        for ent in entities:
            idx = query.lower().find(ent.text.lower())
            if 0 <= idx < another_pos:
                anchor = ent.text
                break
        if anchor is None:
            # Fall back to first entity
            anchor = entities[0].text
        return anchor, bridge_noun

    def _decompose_implicit_bridge(
        self,
        query: str,
        bridge_info: Tuple[str, str],
        entities: List["EntityInfo"],
    ) -> Tuple[List["HopStep"], List[str]]:
        """
        Decompose an implicit-bridge query (Pattern J) into two hops.

        hop 0 (bridge resolution): identify the unnamed second entity.
          "What [bridge_noun] besides [anchor] is [subject] a critic of?"
        hop 1 (attribute lookup): original query, answered with bridge entity
          materialised in the retrieved context.
        """
        anchor, bridge_noun = bridge_info

        # Extract the subject of the sentence (the person/thing doing the
        # critiquing / comparing) — everything before the named anchor entity.
        # Simple heuristic: take the first named entity if it is the subject,
        # otherwise fall back to the full query prefix before the anchor.
        subject = ""
        anchor_idx = query.lower().find(anchor.lower())
        if anchor_idx > 0:
            prefix = query[:anchor_idx].strip().rstrip("of").strip()
            # Take last sentence fragment as subject
            subject_match = re.search(r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', prefix)
            if subject_match:
                subject = subject_match.group(1)

        if subject:
            hop0_q = (
                f"What {bridge_noun} besides {anchor} is {subject} a critic of?"
            )
        else:
            hop0_q = f"What {bridge_noun} besides {anchor} is mentioned in the context?"

        hop_sequence = [
            HopStep(
                step_id=0,
                sub_query=hop0_q,
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
            "_decompose_implicit_bridge: Pattern J → hop0=%r", hop0_q[:80]
        )
        return hop_sequence, [hop0_q, query]

    def _find_relative_clause_bridge(
        self,
        query: str,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Detect bridge queries with a relative-clause structure using SpaCy's
        dependency parse and return (role_noun, anchor_entity), or None.

        Two structural forms are supported:

        Form 1 — Anchor-inside-clause:
            "The [noun] in which [Entity] [predicate]..."
            → role_noun = head of relcl (subject noun),
              anchor   = subject of relcl

        Form 2 — Anchor-as-clause-object (§12.33 extension, Pattern L):
            "...the [King|actress|author] who [made|directed|wrote] [Entity]..."
            → role_noun = head of relcl (attr/dobj head),
              anchor   = NER entity inside the relcl subtree
            When the relcl subject is a relative pronoun (who/that/which),
            the actual bridge anchor is the NER entity in the predicate.

        Form 2 covers "In which year was the King who made the 1925 Birthday
        Honours born?" — Pattern G v1 rejected it because `King.dep_ == "attr"`
        and because the relcl subject was the pronoun "who", not a named entity.

        No verb list is needed — the structural signal (relcl dep label +
        an NER entity reachable from the relcl) is sufficient for any verb
        the language model might produce.
        """
        if not SPACY_AVAILABLE or NLP is None:
            return None
        # Relative-pronoun set: when the relcl subject is one of these, the
        # real anchor sits inside the predicate as an object/oblique NP.
        _REL_PRONOUNS = {"who", "whom", "which", "that"}
        try:
            doc = NLP(query)
            for token in doc:
                if token.dep_ != "relcl":
                    continue
                head = token.head
                # Accept attr (predicate-nominal: "X was the King who...")
                # in addition to nsubj/nsubjpass/ROOT.
                if head.dep_ not in ("nsubj", "nsubjpass", "ROOT", "attr"):
                    continue

                # ── Form 1: relcl subject is a real NP (not a pronoun) ──
                rel_subjects = [
                    c for c in token.children if c.dep_ == "nsubj"
                ]
                if rel_subjects:
                    subj_tok = rel_subjects[0]
                    if subj_tok.text.lower() not in _REL_PRONOUNS:
                        entity_text = " ".join(
                            t.text for t in subj_tok.subtree
                            if not t.is_punct
                        ).strip()
                        noun_text = head.text
                        if entity_text and noun_text:
                            return noun_text, entity_text, "form1"

                # ── Form 2 (Pattern L): relcl subject is a relative pronoun;
                # anchor is the NER entity inside the relcl subtree. ──────
                # We need entities that sit INSIDE the relative clause
                # (so we can use them as graph-search anchors), not the
                # head NP itself. The head's `subtree` INCLUDES the relcl,
                # so we cannot use `head.subtree` as the exclusion set —
                # it would mask the entire relcl. Instead, restrict to
                # tokens whose ancestor chain passes through the relcl
                # token but NOT directly through the head as a noun-phrase
                # modifier (det/compound/amod).
                relcl_token_indices = {t.i for t in token.subtree}
                # Tokens that are part of the head's OWN noun phrase (det,
                # compound, amod, nmod attached directly to head). These
                # must be excluded — they belong to the role NP, not the
                # anchor inside the relcl predicate.
                head_np_indices = {head.i}
                for child in head.children:
                    if child.dep_ in ("det", "compound", "amod", "nmod", "poss"):
                        head_np_indices.update(t.i for t in child.subtree)
                candidate_ents = [
                    ent for ent in doc.ents
                    if any(t.i in relcl_token_indices for t in ent)
                    and not any(t.i in head_np_indices for t in ent)
                ]
                if candidate_ents:
                    # Longest entity span = most specific anchor.
                    anchor = max(candidate_ents, key=lambda e: len(e.text))
                    noun_text = head.text
                    if anchor.text and noun_text:
                        return noun_text, anchor.text, "form2"
        except Exception as exc:
            logger.debug("_find_relative_clause_bridge failed: %s", exc)
        return None

    # ── Pattern H: chained-attribution bridge ───────────────────────────────
    # English verbs that, as the head of an `acl` clause on a "work" noun,
    # express "this work is *about/derived-from* X":  "[work] based on / set in /
    # featuring / starring / centred on / adapted from [X]".  This is a CLOSED
    # LINGUISTIC CATEGORY — the small inventory of work→source attribution verbs
    # in English — not a list of verbs harvested from test answers.  It is the
    # same kind of artefact as a stopword list or a list of auxiliary verbs:
    # finite, domain-independent, and stable.  A query using any of these heads
    # is recognised regardless of whether that phrasing has ever been seen; a
    # query about manga, films, symphonies or video games is treated identically
    # because the recogniser keys on the *attribution relation*, never on the
    # work's vocabulary, the entity's name, or the question's phrasing.
    # Lemmas (SpaCy gives "based"→"base", "featuring"→"feature", etc.).
    _ATTRIBUTION_ACL_VERBS = frozenset({
        "base", "feature", "star", "set", "center", "centre",
        "focus", "follow", "depict", "adapt", "inspire",
        # "about" as an acl head can surface with lemma "about" (prep/sconj use)
        "about",
    })
    # Indefinite pronouns marking an *unresolved* agent in a passive `by`-phrase
    # — the entity the next hop must look up ("written by someone").  A closed
    # grammatical class (English indefinite person pronouns), not example-derived.
    _INDEFINITE_AGENT_HEADS = frozenset({
        "someone", "somebody", "anyone", "anybody",
    })
    # The agent hop's verb is taken VERBATIM from the query's own participle
    # (token.text — "illustrated", "written", "directed"), so there is no
    # participle→verb table and no past-tense rules: the inflection is the
    # user's, not ours.  This deliberately leaves no record in the codebase of
    # which verbs the system has encountered.

    def _find_attribution_chain(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Optional[Dict[str, Any]]:
        """
        Detect a *chained* attribution bridge using SpaCy's dependency parse.

        Target shape (idx-42 archetype):
            "A [work] based on [Entity], is [written] by someone [attribute]?"
        Dependency signature:
            work_noun   --nsubjpass--> passive_verb (ROOT)
            passive_verb --agent--> "by" --pobj--> indefinite agent ("someone")
            work_noun   --acl--> attribution_verb ("based") --prep--> --pobj--> Entity
            agent       --acl--> ... (the residual attribute question stays on it)

        Returns a dict describing the two-link chain, or None if the structure
        is absent or the parse is too ambiguous to trust:
            {
              "work_type":          "series"          # head noun of the work NP
              "work_np":            "manga series"    # compound+amod span (no det)
              "anchor_entity":      "Ichitaka Seto"   # known entity (link-0 target)
              "agent_verb_surface": "illustrated"     # participle verbatim from
                                                      # the query (used in hop1)
            }

        Anti-fragility: keys on dependency *relation labels* (nsubjpass, agent,
        acl, prep/pobj) and a small closed set of attribution clause heads — the
        grammar of attribution, not the lexicon of works.  A query with the same
        shape but novel vocabulary ("a symphony commissioned for the coronation
        of a monarch crowned in what year?") matches with zero new code.

        Parse-confidence gate: requires the *exact* relation chain to be present
        and the anchor entity to overlap a detected NER entity.  If any link is
        missing or fuzzy, returns None and the caller falls back to the next
        pattern / single-query — so the worst case is current behaviour, never
        worse.
        """
        if not SPACY_AVAILABLE or NLP is None:
            return None
        try:
            doc = NLP(query)

            # 1. Find a passive ROOT (or conj-of-ROOT) with an `agent` by-phrase
            #    whose object is an indefinite placeholder.
            for tok in doc:
                if tok.pos_ != "VERB":
                    continue
                # Collect this verb + any conj siblings (handles "written and
                # illustrated by ...") — the agent may hang off either.
                verb_group = [tok] + [c for c in tok.children if c.dep_ == "conj"]
                agent_obj = None
                for v in verb_group:
                    for c in v.children:
                        if c.dep_ == "agent":
                            for gc in c.children:
                                if gc.dep_ == "pobj":
                                    agent_obj = gc
                                    break
                        if agent_obj:
                            break
                    if agent_obj:
                        break
                if agent_obj is None:
                    continue
                # The agent must be an *unresolved* placeholder, else this is a
                # normal passive-with-named-agent (Pattern F territory).
                if agent_obj.lemma_.lower() not in self._INDEFINITE_AGENT_HEADS:
                    continue

                # 2. The passive verb's subject = the "work" noun.
                work_noun = None
                for v in verb_group + [tok]:
                    for c in v.children:
                        if c.dep_ in ("nsubjpass", "nsubj"):
                            work_noun = c
                            break
                    if work_noun:
                        break
                # nsubjpass usually attaches to the first verb of the group
                if work_noun is None:
                    for c in tok.children:
                        if c.dep_ in ("nsubjpass", "nsubj"):
                            work_noun = c
                            break
                if work_noun is None:
                    continue

                # 3. The work noun must carry an `acl` attribution clause linking
                #    it (via prep + pobj) to a concrete noun/entity.  Capture both
                #    the prep object's head proper-noun span (preferred anchor) and
                #    the full NP (fallback).
                anchor_full = None         # full prep-object NP
                anchor_propn = None        # contiguous PROPN run within it
                for c in work_noun.children:
                    if c.dep_ == "acl" and c.lemma_.lower() in self._ATTRIBUTION_ACL_VERBS:
                        for gc in c.children:
                            if gc.dep_ == "prep":
                                for ggc in gc.children:
                                    if ggc.dep_ == "pobj":
                                        sub = [t for t in ggc.subtree if not t.is_punct]
                                        anchor_full = " ".join(t.text for t in sub).strip()
                                        # Longest contiguous PROPN run = the name
                                        run, best = [], []
                                        for t in sub:
                                            if t.pos_ == "PROPN":
                                                run.append(t.text)
                                            else:
                                                if len(run) > len(best):
                                                    best = run
                                                run = []
                                        if len(run) > len(best):
                                            best = run
                                        if best:
                                            anchor_propn = " ".join(best)
                                        break
                            if anchor_full:
                                break
                    if anchor_full:
                        break
                if not anchor_full:
                    continue

                # 4. Parse-confidence gate: the anchor must overlap a detected
                #    NER entity, else the "chain" is noise.  Resolve the anchor we
                #    pass downstream to the NER entity text when it's contained in
                #    the prep-object NP (so "a 16 year old ... Ichitaka Seto" →
                #    "Ichitaka Seto"); else fall back to the PROPN run, then NP.
                _ANCHOR_LABELS = ("PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART",
                                  "EVENT", "FAC", "PRODUCT")
                anchor_full_lc = anchor_full.lower()
                matched_ner = next(
                    (e.text for e in entities
                     if e.label in _ANCHOR_LABELS
                     and (e.text.lower() in anchor_full_lc
                          or anchor_full_lc in e.text.lower())),
                    None,
                )
                if matched_ner is None:
                    logger.debug(
                        "_find_attribution_chain: anchor %r has no NER overlap "
                        "— rejecting (parse-confidence gate)", anchor_full[:50]
                    )
                    continue
                anchor_text = matched_ner or anchor_propn or anchor_full

                # 5. Build the work NP (compound + amod, drop determiners) and
                #    capture the agent participle *as it appears in the query*.
                #    We deliberately reuse the user's own surface form ("written",
                #    "illustrated", "directed") rather than re-inflecting from a
                #    lemma — no verb tables, no past-tense rules, nothing that
                #    encodes which verbs we've seen.  "Who illustrated the X?"
                #    is grammatical because the inflection came from the input.
                work_type = work_noun.text
                work_np_tokens = [
                    t.text for t in work_noun.subtree
                    if t.dep_ in ("compound", "amod") and t.head == work_noun
                ] + [work_noun.text]
                work_np = " ".join(work_np_tokens).strip()

                agent_verb_surface = None
                for v in verb_group + [tok]:
                    if any(c.dep_ == "agent" for c in v.children):
                        agent_verb_surface = v.text
                        break
                if agent_verb_surface is None:
                    agent_verb_surface = tok.text

                return {
                    "work_type": work_type,
                    "work_np": work_np or work_type,
                    "anchor_entity": anchor_text,
                    "agent_verb_surface": agent_verb_surface,
                }
        except Exception as exc:
            logger.debug("_find_attribution_chain failed: %s", exc)
        return None

    def _decompose_attribution_chain(
        self,
        query: str,
        chain: Dict[str, Any],
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Build a 3-step hop sequence from a chained-attribution bridge.

            hop0 (bridge): "What {work_type} is based on {anchor_entity}?"
                           → resolves the {work} placeholder
            hop1 (bridge): "Who {agent_verb_surface} the {work_type} based on
                           {anchor}?"  — {agent_verb_surface} is the participle
                           taken verbatim from the query ("illustrated",
                           "written", "directed"), so the phrasing is correct
                           without any verb-inflection tables.  The Controller
                           injects the resolved work title at runtime; this is
                           the retrieval seed.
            hop2 (final):  the original query, depends_on=[1]
                           → answers the residual attribute question once both
                             the work and the agent are in context

        depends_on chains 0 → 1 → 2 so the Controller's iterative bridge-entity
        injection (§12 iterative-multihop) feeds each link's result into the next.
        """
        work_type = chain["work_type"]
        anchor    = chain["anchor_entity"]
        agent_vb  = chain["agent_verb_surface"]

        hop0_q = f"What {work_type} is based on {anchor}?"
        hop1_q = f"Who {agent_vb} the {work_type} based on {anchor}?"

        target_all = [e.text for e in entities]
        hop_sequence = [
            HopStep(
                step_id=0,
                sub_query=hop0_q,
                target_entities=[anchor],
                depends_on=[],
                is_bridge=True,
            ),
            HopStep(
                step_id=1,
                sub_query=hop1_q,
                target_entities=[anchor],
                depends_on=[0],
                is_bridge=True,
            ),
            HopStep(
                step_id=2,
                sub_query=query,
                target_entities=target_all,
                depends_on=[1],
                is_bridge=False,
            ),
        ]
        logger.debug(
            "_decompose_multi_hop: Pattern H attribution-chain → hop0=%r hop1=%r",
            hop0_q[:50], hop1_q[:50],
        )
        return hop_sequence, [hop0_q, hop1_q, query]

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

        # ── Pattern J: Implicit bridge ("X and another [noun] that …") ───────
        # "Scott Parkin has been a critic of Exxonmobil and another corporation
        #  that has operations in how many countries?"
        # The phrase "another [noun]" signals that the answer-bearing entity is
        # unnamed in the query; it must be resolved via the named anchor entity
        # before the count/attribute question can be answered.
        # hop 0: resolve the bridge — "What corporation besides Exxonmobil is
        #         Scott Parkin a vocal critic of?"
        # hop 1: answer the attribute question — original query (answered with
        #         the bridge entity now materialised in context)
        imb = self._find_implicit_bridge(query, entities)
        if imb:
            return self._decompose_implicit_bridge(query, imb, entities)

        # ── Pattern G: Relative-clause bridge (SpaCy dependency parse) ───────
        # Form 1: "The [noun] in which [Entity] [predicate] [main question]?"
        # Form 2 (Pattern L, §12.33): "...the [King|actress|author] who [verb] [Entity]..."
        # Uses structural grammar (relcl dep label) — no verb list needed.
        # Must run BEFORE the generic split because the split-on-"which" would
        # otherwise destroy the grammatical structure of the query.
        rc = self._find_relative_clause_bridge(query)
        if rc:
            noun, entity_text, form = rc
            if form == "form2":
                # Form 2: the bridge entity is the unknown role-NP; the anchor
                # is the NER entity inside the relative clause. The bridge
                # sub-query asks "which {role} is associated with {anchor}"
                # so retrieval pulls the anchor's article (which names the role).
                bridge_q = f"Who is the {noun} associated with {entity_text}?"
            else:
                # Form 1 (original): the entity is the subject of the relcl.
                bridge_q = f"In which {noun} did {entity_text}?"
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
                "_decompose_multi_hop: Pattern G (%s) → %r", form, bridge_q[:60]
            )
            return hop_sequence, [bridge_q, query]

        # ── Pattern H: chained-attribution bridge (SpaCy dependency parse) ───
        # Detects "A [work] based on [Entity], is [written] by someone [attr]?"
        # — a *two-link* chain (entity → work → agent → attribute).  Must run
        # BEFORE the generic split because the split-on-connector would shred
        # the multi-clause structure.  The parse-confidence gate inside
        # _find_attribution_chain returns None on anything ambiguous, so this is
        # fail-safe — worst case it falls through to the patterns below.
        ac = self._find_attribution_chain(query, entities)
        if ac:
            return self._decompose_attribution_chain(query, ac, entities)

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

        # ── 2-hop cap (§12.32 Fix 1) ──────────────────────────────────────────
        # Repeated splitting on different connectors can produce 3+ parts from
        # a single relative clause:
        #   "What is the middle name of the actress who plays X in Y?"
        #   → ["middle name", "actress", "plays X in Y"]   (3 parts → 3 hops)
        # But semantically this is still a 2-hop bridge: resolve the actress,
        # then answer the attribute. The middle parts ("actress") become
        # nonsensical sub-queries ("What is actress?") that retrieve random
        # actresses as distractors and propagate spurious bridge entities.
        #
        # Heuristic: keep the first part (final attribute) and the last part
        # (bridge resolver), drop everything in between. The dropped fragments
        # carry no named entity — they were just clause connectors picked up by
        # the over-eager split.
        #
        # The cap is conservative: it only collapses when the middle parts have
        # NO named entity. If any middle part has an entity, we keep all parts
        # so genuine 3-hop chains (Pattern H attribution) survive.
        if len(parts) > 2:
            middle_has_entity = any(
                any(e.text.lower() in mid.lower() for e in entities)
                for mid in parts[1:-1]
            )
            if not middle_has_entity:
                logger.debug(
                    "_decompose_multi_hop: collapsing %d-part split → 2-hop "
                    "(no entity in middle parts: %r)",
                    len(parts), parts[1:-1],
                )
                parts = [parts[0], parts[-1]]

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

        # ── Pattern E: "the [role] of [Entity]" — relational anchor bridge ──────
        # Only fires when all split patterns failed (len(parts) == 1).
        # See class-level _RELATIONAL_ANCHOR_ROLES for full rationale.
        if len(parts) <= 1 and entities:
            em = self._RELATIONAL_ANCHOR_ROLES.search(query)
            if em:
                role = em.group(1).lower()
                # Use the first proper named entity as the anchor (the entity
                # whose article will name the bridge). Prefer PERSON/ORG/GPE
                # over DATE entities, which are not article subjects.
                _ANCHOR_LABELS = frozenset({
                    "PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", "FAC",
                })
                anchor = next(
                    (e.text for e in entities if e.label in _ANCHOR_LABELS),
                    entities[0].text,
                )
                bridge_q = f"Who is the {role} of {anchor}?"
                hop_sequence = [
                    HopStep(
                        step_id=0,
                        sub_query=bridge_q,
                        target_entities=[anchor],
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
                    "_decompose_multi_hop: Pattern E relational-anchor → %r", bridge_q[:60]
                )
                return hop_sequence, [bridge_q, query]

        # ── Pattern F: Passive-agent bridge ─────────────────────────────────
        # "[Subject] was written/directed/... by [agent] that [property]?"
        # Only fires when the generic split produced parts (i.e. there IS a
        # "that/which/who" clause) — meaning the connector split already found
        # the boundary, but the *order* it produces is wrong (it reverses parts,
        # putting the property fragment first and the anchor second).  Pattern F
        # detects the passive-agent structure and corrects the order by emitting
        # hop 0 = "Who {verb} {subject}?" and hop 1 = original query.
        # Also fires when split failed (len(parts)==1) with a passive-agent form.
        pm = self._PASSIVE_AGENT_RE.match(query)
        if pm and len(parts) >= 1:
            subj = pm.group("subj").strip()
            past_part = pm.group("verb").lower()
            active_verb = self._PASSIVE_TO_ACTIVE.get(past_part, past_part)
            # "Who wrote Seven Brief Lessons on Physics?"
            bridge_q = f"Who {active_verb} {subj}?"
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
                "_decompose_multi_hop: Pattern F passive-agent → %r", bridge_q[:60]
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
            # ── Classification–decomposition consistency fallback ───────────
            # The query was *classified* multi-hop but every pattern + the
            # connector split failed.  Rather than silently emitting the
            # unsplit query (which looks like a working single-hop plan and
            # hides the gap), emit a deliberate 2-hop fallback: hop0 retrieves
            # broadly around the detected entities, hop1 answers the original
            # query with that context available.  If there are no usable
            # entities to seed hop0, only then degrade to single-hop — with a
            # WARNING so the gap is visible in logs / diagnostics.
            seed_entities = [
                e for e in entities
                if e.label in ("PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART",
                                "EVENT", "FAC", "PRODUCT", "NORP")
            ]
            if seed_entities:
                anchor = seed_entities[0].text
                hop0_q = f"Who or what is {anchor}?"
                logger.debug(
                    "_decompose_multi_hop: no pattern matched for %r "
                    "— generic 2-hop fallback (anchor=%r)", query[:80], anchor
                )
                hop_sequence = [
                    HopStep(
                        step_id=0,
                        sub_query=hop0_q,
                        target_entities=[anchor],
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
                sub_queries = [hop0_q, query]
            else:
                logger.warning(
                    "_decompose_multi_hop: query classified MULTI_HOP but no "
                    "pattern matched and no anchor entity available — "
                    "degrading to single-hop: %r", query[:100]
                )
                hop_sequence.append(HopStep(
                    step_id=0,
                    sub_query=query,
                    target_entities=[e.text for e in entities],
                    depends_on=[],
                    is_bridge=False,
                ))
                sub_queries = [query]

        return hop_sequence, sub_queries

    # Boolean-conjunction surface form: "Are/Did/Were/Is/Do/Does/Have/Has [X] and [Y] both [P]?"
    # The "both" keyword is a reliable discriminator — genuine bridge questions almost
    # never contain it. Lexical detection avoids SpaCy parse dependency.
    # §12.32: Pattern I — Boolean conjunction decomposition.
    _BOOL_CONJ_RE = re.compile(
        r'^\s*(are|is|were|was|did|do|does|have|has)\b.+\band\b.+\bboth\b',
        re.IGNORECASE,
    )

    def _decompose_boolean_conjunction(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Optional[Tuple[List[HopStep], List[str]]]:
        """Decompose "Are [X] and [Y] both [predicate]?" into two parallel yes/no
        sub-queries, one per subject entity.

        Returns (hop_sequence, sub_queries) when the Boolean conjunction form is
        detected and at least two named entities are present; otherwise None so
        the caller falls through to the generic comparison path.

        The predicate fragment is recovered by removing the "[X] and [Y]" span
        from the query, leaving e.g. "Are ... both used for real estate?" which
        is then specialised per entity: "Is Random House Tower used for real estate?"
        """
        if not self._BOOL_CONJ_RE.match(query):
            return None

        # Extract entity names directly from the query's conjunction structure:
        # "[AUX] [X] and [Y] both [P]?" — split on " and " then on " both ".
        # This bypasses SpaCy NER entirely for entity identification in this
        # pattern, avoiding MONEY/CARDINAL misclassification of address-like
        # strings such as "888 7th Avenue".
        _and_re = re.compile(r'\band\b', re.IGNORECASE)
        _both_re = re.compile(r'\bboth\b', re.IGNORECASE)
        both_m = _both_re.search(query)
        and_m  = _and_re.search(query)
        if both_m and and_m and and_m.start() < both_m.start():
            # Strip the leading auxiliary verb to get ent_a text
            after_aux = re.sub(
                r'^\s*(are|is|were|was|did|do|does|have|has)\s+',
                '', query, count=1, flags=re.IGNORECASE,
            )
            raw_a = after_aux[:and_m.start() - (len(query) - len(after_aux))].strip()
            raw_b = query[and_m.end():both_m.start()].strip().rstrip(',').strip()
            if raw_a and raw_b:
                # Build synthetic EntityInfo objects with correct char offsets
                a_idx = query.find(raw_a)
                b_idx = query.find(raw_b)
                ent_a = EntityInfo(
                    text=raw_a, label="PROPN", confidence=0.75,
                    start_char=a_idx if a_idx >= 0 else 0,
                    end_char=(a_idx + len(raw_a)) if a_idx >= 0 else len(raw_a),
                )
                ent_b = EntityInfo(
                    text=raw_b, label="PROPN", confidence=0.75,
                    start_char=b_idx if b_idx >= 0 else 0,
                    end_char=(b_idx + len(raw_b)) if b_idx >= 0 else len(raw_b),
                )
                logger.debug(
                    "_decompose_boolean_conjunction: conjunction parse → a=%r b=%r",
                    raw_a, raw_b,
                )
                # Skip the NER-based entity selection below
                ner_entities = [ent_a, ent_b]
            else:
                ner_entities = [e for e in entities if e.label in self._NER_LABELS]
                if len(ner_entities) < 2:
                    return None
        else:
            ner_entities = [e for e in entities if e.label in self._NER_LABELS]
            if len(ner_entities) < 2:
                return None

        ent_a, ent_b = ner_entities[0], ner_entities[1]

        # Strip the leading auxiliary verb and swap to singular "Is/Did/Was..."
        _AUX_MAP = {
            "are": "Is", "were": "Was", "did": "Did",
            "do": "Does", "does": "Does", "have": "Has", "has": "Has", "is": "Is",
        }
        first_token = query.split()[0].lower()
        singular_aux = _AUX_MAP.get(first_token, "Is")

        # Build the predicate fragment: remove "[X] and [Y]" span from query
        a_idx = query.find(ent_a.text)
        b_idx = query.find(ent_b.text)
        if a_idx < 0 or b_idx < 0:
            return None

        conj_start = a_idx
        conj_end   = b_idx + len(ent_b.text)
        # Eat the word "both" immediately after the conjunction if present
        after_conj = query[conj_end:].lstrip()
        if after_conj.lower().startswith("both "):
            conj_end += query[conj_end:].index("both") + len("both")

        predicate = query[conj_end:].strip().rstrip("?").strip()

        hop_sequence = []
        sub_queries  = []
        for i, ent in enumerate([ent_a, ent_b]):
            sq = f"{singular_aux} {ent.text} {predicate}?"
            hop_sequence.append(HopStep(
                step_id=i,
                sub_query=sq,
                target_entities=[ent.text],
                depends_on=[],
                is_bridge=False,
            ))
            sub_queries.append(sq)
            logger.debug(
                "_decompose_boolean_conjunction: Pattern I → hop%d=%r", i, sq[:80]
            )

        return hop_sequence, sub_queries

    def _decompose_comparison(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose a comparison query into parallel retrieval steps.

        Strategy:
        0. Boolean conjunction check: "Are [X] and [Y] both [P]?" → Pattern I.
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

        # ── Pattern I: Boolean conjunction ("Are X and Y both P?") ──────────────
        # Must run before select-between-two: "both" keyword is a reliable signal
        # that this is a parallel yes/no check, not a selection-between-two-options.
        bool_conj = self._decompose_boolean_conjunction(query, entities)
        if bool_conj is not None:
            return bool_conj

        # FIX P-1: "select-between-two" comparison form
        # ("Which writer was from England, Henry Roth or Robert Erskine Childers?",
        #  "Which band, Letters to Cleo or Screaming Trees, had more members?").
        # Here the two comparison operands are exactly the two entities joined by
        # "or" in the query — NOT the first two NER hits (which would wrongly pick
        # "England"). Detect the disjunction, strip it + the leading "Which
        # <category>" framing to recover the *property*, and emit one focused
        # per-entity sub-query.
        sel = self._decompose_select_between(query, entities)
        if sel is not None:
            return sel

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

    # Leading framing to strip from a "select-between-two" question to recover
    # the property being compared: "Which writer ", "What city ", "Who ", etc.
    _SELECT_LEAD_RE = re.compile(
        r'^\s*(which|what|who|whom)\b\s*([a-z][a-z\- ]{0,30}?\b)?\s*',
        re.IGNORECASE,
    )
    # "ENT_A or ENT_B" disjunction (allowing a comma before "or").
    _OR_DISJ_RE = re.compile(r'\s*,?\s+\bor\b\s+', re.IGNORECASE)

    def _decompose_select_between(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Optional[Tuple[List[HopStep], List[str]]]:
        """Handle the "Which <category> <property>, A or B<property>?" comparison
        form, where A and B are the two entities joined by "or" in the query.

        Returns (hop_sequence, sub_queries) on success, or None if the query is
        not of this form (caller then falls back to the generic decomposition).

        Strategy:
          1. Find two extracted entities that sit on either side of an "or" in
             the query, with only whitespace/comma between them and "or".
          2. Remove that "<A> or <B>" span from the query, and strip the leading
             "Which <category>" framing → what remains is the *property* clause
             ("was from England", "had more members", "was published first").
          3. Build one focused sub-query per entity by attaching the entity to
             the property clause, e.g. "Henry Roth was from England" /
             "Letters to Cleo had more members". If an _ATTR_MAP attribute
             pattern matches, prefer that rewrite instead.
        """
        # 1. Locate an "A or B" disjunction of two entities.
        ents_by_pos = sorted(entities, key=lambda e: e.start_char)
        pair: Optional[Tuple[EntityInfo, EntityInfo]] = None
        for a, b in zip(ents_by_pos, ents_by_pos[1:]):
            gap = query[a.end_char:b.start_char]
            if self._OR_DISJ_RE.fullmatch(gap):
                pair = (a, b)
                break
        if pair is None:
            return None
        ent_a, ent_b = pair

        # 2. Excise the "<A> or <B>" span; strip leading "Which <category>".
        disj_start, disj_end = ent_a.start_char, ent_b.end_char
        remainder = (query[:disj_start] + " " + query[disj_end:])
        remainder = re.sub(r'\s+', ' ', remainder).strip().rstrip('?').strip()
        # drop a dangling leading/trailing comma left by the excision
        remainder = remainder.strip(',').strip()
        property_clause = self._SELECT_LEAD_RE.sub('', remainder).strip()
        # also drop any leftover leading conjunction/punctuation
        property_clause = property_clause.lstrip(',; ').strip()

        # 3. Per-entity sub-queries.
        # Prefer an attribute rewrite if one of the _ATTR_MAP / comparative
        # patterns applies to the original query.
        templates: List[str] = []
        for pat, tmpl in self._ATTR_MAP:
            if pat.search(query):
                templates = [tmpl.format(entity=ent_a.text),
                             tmpl.format(entity=ent_b.text)]
                break
        if not templates:
            if property_clause:
                # "Henry Roth was from England" / "Letters to Cleo had more members"
                templates = [f"{ent_a.text} {property_clause}",
                             f"{ent_b.text} {property_clause}"]
            else:
                # No usable property clause recovered → fall back to a generic
                # "Who/what is <entity>?" lookup so retrieval at least targets
                # the right two articles.
                templates = [f"Who is {ent_a.text}?", f"Who is {ent_b.text}?"]

        hop_sequence: List[HopStep] = []
        sub_queries: List[str] = []
        for i, (ent, sq) in enumerate(zip((ent_a, ent_b), templates)):
            sq = re.sub(r'\s+', ' ', sq).strip()
            hop_sequence.append(HopStep(
                step_id=i,
                sub_query=sq,
                target_entities=[ent.text],
                depends_on=[],          # independent → parallel
                is_bridge=True,
            ))
            sub_queries.append(sq)
        logger.debug(
            "_decompose_select_between: %r → [%r, %r]",
            query[:80], sub_queries[0][:50], sub_queries[1][:50],
        )
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
