"""
===============================================================================
S_V: Verifier — Pre-Generation Validation and Self-Correction
===============================================================================

Master's Thesis: Edge-RAG — Hybrid Retrieval-Augmented Generation on
Resource-Constrained Edge Devices.
Artifact B: Agent-Based Query Processing (Logic Layer).

===============================================================================
OVERVIEW
===============================================================================

The Verifier (S_V) is the final stage of the three-agent RAG pipeline
(S_P → S_N → S_V) and implements a dual-stage approach:

1. PRE-GENERATION VALIDATION
   Three checks are applied before generation:

   a) Entity-Path Validation (multi-hop queries)
      Verifies that retrieved chunks cover the query entities.  When a
      KuzuDB graph store is available, ``find_chunks_by_entity_multihop``
      is used; otherwise falls back to substring matching.

   b) Contradiction Detection
      NLI-based pairwise detection on adjacent chunk pairs.  Only
      consecutive pairs are checked (O(n)) to stay within edge CPU budget;
      non-adjacent contradictions are not detected.  Disabled by default
      to avoid the ~270 MB NLI model download; a numeric-divergence
      heuristic serves as the offline fallback.
      Reference: Bowman et al. (2015). arXiv:1508.05326 (NLI task);
      Reimers & Gurevych (2019). arXiv:1908.10084 (cross-encoder model)..

   c) Source Credibility Scoring
      Weighted combination (40 % cross-references, 30 % entity-mention
      frequency, 30 % retrieval provenance).  Weights were chosen
      empirically on a HotpotQA dev-set sample.  The provenance signal
      is currently always False because Navigator does not forward
      retrieval-source metadata to S_V (known limitation).

2. GENERATION WITH A QUANTISED SLM
   Phi-3-Mini (or any Ollama-hosted model) is prompted with a compact
   context budget tuned for edge hardware (<16 GB RAM).  Context limits
   are set in config/settings.yaml under the ``llm`` block.

3. SELF-CORRECTION LOOP
   Iterative refinement following the Self-Refine paradigm:
       Madaan, A., et al. (2023). "Self-Refine: Iterative Refinement
       with Self-Feedback." NeurIPS 2023. arXiv:2303.17651.

   Up to ``max_iterations`` rounds are run.  Each round extracts atomic
   claims from the generated answer, verifies each claim against the graph
   store and the retrieved context, and — if violations remain — re-prompts
   the LLM with explicit violation feedback.  Claim-level verification is
   a conservative proxy (entity presence, not logical entailment); see
   Kryscinski et al. (2020) for the full entailment framing.

===============================================================================
ARCHITECTURE
===============================================================================

    Query + Context (from Navigator)
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                      S_V (VERIFIER)                           │
    │                                                               │
    │   ┌─────────────────────────────────────────────────────┐    │
    │   │            PRE-GENERATION VALIDATION                 │    │
    │   │                                                      │    │
    │   │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │    │
    │   │  │Entity-Path │  │Contradiction│  │  Source    │    │    │
    │   │  │ Validation │  │ Detection  │  │ Credibility│    │    │
    │   │  └────────────┘  └────────────┘  └────────────┘    │    │
    │   │         │              │               │            │    │
    │   │         └──────────────┴───────────────┘            │    │
    │   │                        │                             │    │
    │   │              Pass/Fail + Filtered Context           │    │
    │   └────────────────────────┼────────────────────────────┘    │
    │                            │                                  │
    │                            ▼                                  │
    │   ┌─────────────────────────────────────────────────────┐    │
    │   │               GENERATION LOOP                        │    │
    │   │                                                      │    │
    │   │         ┌──────────┐                                │    │
    │   │    ┌───▶│ GENERATE │                                │    │
    │   │    │    │  Answer  │                                │    │
    │   │    │    └────┬─────┘                                │    │
    │   │    │         │                                       │    │
    │   │    │    ┌────▼─────┐                                │    │
    │   │    │    │ EXTRACT  │                                │    │
    │   │    │    │  Claims  │                                │    │
    │   │    │    └────┬─────┘                                │    │
    │   │    │         │                                       │    │
    │   │    │    ┌────▼─────┐     ┌─────────────┐            │    │
    │   │    │    │  VERIFY  │────▶│ Violations? │            │    │
    │   │    │    │  Claims  │     └──────┬──────┘            │    │
    │   │    │    └──────────┘            │                    │    │
    │   │    │                      Yes   │   No               │    │
    │   │    │    ┌─────────────┐◀───────┘     │              │    │
    │   │    └────┤ SELF-CORRECT│              │              │    │
    │   │         │ (Feedback)  │              │              │    │
    │   │         └─────────────┘              │              │    │
    │   │                                      ▼              │    │
    │   │                              ┌─────────────┐        │    │
    │   │                              │   RETURN    │        │    │
    │   │                              │   Answer    │        │    │
    │   │                              └─────────────┘        │    │
    │   └──────────────────────────────────────────────────────┘    │
    │                                                               │
    └───────────────────────────────────────────────────────────────┘

===============================================================================
REFERENCES
===============================================================================

- Madaan, A., et al. (2023). "Self-Refine: Iterative Refinement with
  Self-Feedback." NeurIPS 2023. arXiv:2303.17651.
- Bowman, S., et al. (2015). "A large annotated corpus for learning natural
  language inference." EMNLP 2015. arXiv:1508.05326.
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings
  using Siamese BERT-Networks." EMNLP 2019. arXiv:1908.10084.
- Kryscinski, W., et al. (2020). "Evaluating the Factual Consistency of
  Abstractive Text Summarization." EMNLP 2020. arXiv:1910.12840.

===============================================================================

===============================================================================
Review History:
    Last Reviewed:  2026-04-21
    Review Result:  1 CRITICAL, 4 IMPORTANT, 7 RECOMMENDED
    Reviewer:       Code Review Prompt v2.1
    Next Review:    After implementing multi-hop graph-path planning and
                    forwarding retrieval-source metadata from Navigator to S_V
===============================================================================
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


from ._settings import _load_settings, _PROPER_NOUN_RE


# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

_DEFAULT_SPACY_MODEL = "en_core_web_sm"

# KNOWN LIMITATION: _DEFAULT_SPACY_MODEL (from this constant) cannot be
# overridden by settings.yaml at module load time because this block runs
# at import before any config is read.  Changing the SpaCy model requires
# editing _DEFAULT_SPACY_MODEL here directly.  The same limitation applies
# to planner.py and navigator.py.  A future refactor could defer the load
# into __init__ and read the model name from VerifierConfig.

# SpaCy: used for sentence splitting in _extract_claims and for NER-based
# entity density in _compute_credibility.
try:
    import spacy
    try:
        NLP = spacy.load(_DEFAULT_SPACY_MODEL)
        SPACY_AVAILABLE = True
        logger.info("SpaCy model '%s' loaded for claim extraction", _DEFAULT_SPACY_MODEL)
    except (OSError, IOError):
        NLP = None
        SPACY_AVAILABLE = False
        logger.warning(
            "SpaCy model '%s' not found. Install with:\n"
            "  python -m spacy download en_core_web_sm\n"
            "Regex fallbacks will be used for claim extraction and credibility scoring.",
            _DEFAULT_SPACY_MODEL,
        )
except ImportError:
    NLP = None
    SPACY_AVAILABLE = False
    logger.warning(
        "SpaCy not installed. Regex fallbacks will be used for claim "
        "extraction and credibility scoring. Install with: pip install spacy"
    )

# Transformers: used only when enable_contradiction_detection is True.
# The NLI model (~270 MB) is lazy-loaded on first use.
try:
    from transformers import pipeline as _hf_pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers available for NLI contradiction detection")
except ImportError:
    _hf_pipeline = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "Transformers not available; NLI contradiction detection disabled. "
        "Install with: pip install transformers"
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class ValidationStatus(Enum):
    """Status codes returned by the pre-generation validation stage."""

    PASSED = "passed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTION_DETECTED = "contradiction_detected"
    LOW_CREDIBILITY = "low_credibility"


@dataclass
class SourceCredibility:
    """
    Credibility score for a single context chunk.

    Computes a weighted combination of three signals:

    - ``cross_references``: how often information in this chunk is corroborated
      by other chunks (weight: ``credibility_weight_cross_ref``, default 40 %).
    - ``entity_frequency``: named-entity mention density as a proxy for
      information richness (weight: ``credibility_weight_entity_freq``,
      default 30 %).
    - ``retrieval_provenance``: graph-based retrieval receives a higher score
      than vector-only retrieval (weight: ``credibility_weight_provenance``,
      default 30 %).  Currently always False because Navigator does not
      forward retrieval-source metadata to S_V (known limitation).

    Weights are empirically chosen on a HotpotQA dev-set sample.  Their
    individual contribution is modest relative to entity-path filtering, so
    they are a first approximation rather than an optimised calibration.
    """

    text: str
    score: float = 0.5
    cross_references: int = 0
    entity_frequency: float = 0.0
    is_graph_based: bool = False

    def compute_score(
        self,
        weight_cross_ref: float = 0.4,
        weight_entity_freq: float = 0.3,
        weight_provenance: float = 0.3,
        cross_ref_max: float = 3.0,
        provenance_baseline: float = 0.5,
    ) -> float:
        """
        Compute the weighted credibility score and store it in ``self.score``.

        Parameters
        ----------
        weight_cross_ref, weight_entity_freq, weight_provenance :
            Signal weights (should sum to 1.0).
        cross_ref_max :
            Divisor for normalising ``cross_references`` to [0, 1].
        provenance_baseline :
            Score assigned to vector-only (non-graph) sources.

        Returns
        -------
        float
            Credibility score in [0, 1].
        """
        ref_score = min(1.0, self.cross_references / max(1.0, cross_ref_max))
        entity_score = self.entity_frequency
        provenance_score = 1.0 if self.is_graph_based else provenance_baseline
        self.score = (
            weight_cross_ref * ref_score
            + weight_entity_freq * entity_score
            + weight_provenance * provenance_score
        )
        return self.score


@dataclass
class PreValidationResult:
    """
    Result of the pre-generation validation stage.

    Attributes
    ----------
    status :
        Overall validation outcome.
    entity_path_valid :
        Whether retrieved chunks cover query entities.
    contradictions :
        Index-based contradiction pairs: ``(idx1, idx2, score)``.
    filtered_context :
        Context chunks that passed all validation filters.
    credibility_scores :
        Per-chunk credibility scores aligned with ``filtered_context``.
    validation_time_ms :
        Wall-clock time for the validation stage in milliseconds.
    details :
        Per-step diagnostic information.
    """

    status: ValidationStatus = ValidationStatus.PASSED
    entity_path_valid: bool = True
    contradictions: List[Tuple[int, int, float]] = field(default_factory=list)
    filtered_context: List[str] = field(default_factory=list)
    credibility_scores: List[float] = field(default_factory=list)
    validation_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierConfig:
    """
    Configuration for the Verifier stage (S_V).

    All fields are loaded from config/settings.yaml via ``from_yaml()``.
    Hardcoded defaults serve as emergency fallbacks only and match the
    thesis evaluation settings documented in settings.yaml.

    LLM settings
    ------------
    model_name : Ollama model name.
    base_url : Ollama API endpoint.
    temperature : Sampling temperature (0.1 = near-deterministic).
    max_tokens : Maximum answer tokens.
    timeout : HTTP timeout in seconds for a single Ollama call.

    Context settings (settings.yaml: llm.*)
    ----------------------------------------
    max_context_chars : Total character budget for the prompt context.
    max_docs : Maximum chunks forwarded to the LLM.
    max_chars_per_doc : Per-chunk truncation limit.

    Pre-validation settings (settings.yaml: verifier.*)
    -----------------------------------------------------
    enable_entity_path_validation, enable_contradiction_detection,
    enable_credibility_scoring : Feature flags for the three checks.
    contradiction_threshold : NLI confidence threshold.
    min_credibility_score : Chunks below this score are removed.
    entity_coverage_threshold : Minimum entity-coverage fraction for
        entity-path validation to pass.
    nli_model : HuggingFace model ID for NLI.
        Reference: Reimers & Gurevych (2019). arXiv:1908.10084.
    nli_max_input_chars : Per-chunk truncation before NLI inference.
    spacy_max_chars : Per-chunk truncation before SpaCy processing.

    Credibility scoring weights (settings.yaml: verifier.credibility_*)
    ----------------------------------------------------------------------
    Weights must sum to 1.0.

    Agentic loop settings (settings.yaml: agent.*)
    -----------------------------------------------
    max_iterations : Maximum self-correction rounds.
        1 = generation only (ablation baseline).
        2 = thesis default (one correction round).
        Reference: Madaan et al. (2023). arXiv:2303.17651.
    stop_on_first_success : Exit when all claims are verified.

    Confidence thresholds (settings.yaml: verifier.confidence_*)
    --------------------------------------------------------------
    confidence_high_threshold : Verified-ratio >= this → HIGH.
    confidence_medium_threshold : Verified-ratio >= this → MEDIUM.

    Claim extraction limits (settings.yaml: verifier.*)
    -----------------------------------------------------
    min_claim_chars : Minimum characters for a sentence to count as a claim.
    max_entities_to_verify : Maximum entities checked per claim.
    max_key_phrases : Maximum key phrases per chunk in cross-reference scoring.
    """

    # LLM settings — emergency fallbacks; live values read from settings.yaml via from_yaml()
    model_name: str = "qwen2:1.5b"          # settings.yaml: llm.model_name
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 200
    timeout: int = 60

    # Context settings
    max_context_chars: int = 900
    max_docs: int = 3
    max_chars_per_doc: int = 500             # settings.yaml: llm.max_chars_per_doc

    # Pre-validation flags
    enable_entity_path_validation: bool = True
    enable_contradiction_detection: bool = False
    enable_credibility_scoring: bool = True

    # Pre-validation parameters
    contradiction_threshold: float = 0.85
    min_credibility_score: float = 0.5
    entity_coverage_threshold: float = 0.5
    nli_model: str = "cross-encoder/nli-distilroberta-base"
    nli_max_input_chars: int = 200
    spacy_max_chars: int = 500

    # Credibility scoring weights
    credibility_weight_cross_ref: float = 0.4
    credibility_weight_entity_freq: float = 0.3
    credibility_weight_provenance: float = 0.3
    credibility_cross_ref_max: float = 3.0
    credibility_provenance_baseline: float = 0.5

    # Agentic loop
    max_iterations: int = 2

    # Confidence thresholds
    confidence_high_threshold: float = 0.8
    confidence_medium_threshold: float = 0.5

    # Claim extraction limits
    min_claim_chars: int = 15
    max_entities_to_verify: int = 5
    max_key_phrases: int = 10

    # Heuristic contradiction threshold (Finding 6)
    heuristic_contradiction_threshold: float = 0.5  # settings.yaml: verifier.heuristic_contradiction_threshold

    # Sentence-boundary fraction for context truncation (Finding 7)
    format_sentence_boundary_fraction: float = 0.7  # settings.yaml: verifier.format_sentence_boundary_fraction

    # Entity-density normalizers for credibility scoring (Finding 8)
    credibility_entity_freq_normalizer_spacy: float = 5.0   # settings.yaml: verifier.credibility_entity_freq_normalizer_spacy
    credibility_entity_freq_normalizer_regex: float = 10.0  # settings.yaml: verifier.credibility_entity_freq_normalizer_regex

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "VerifierConfig":
        """
        Build a VerifierConfig from a settings.yaml dict.

        Reads the ``llm``, ``agent``, and ``verifier`` blocks.  All
        defaults match the thesis evaluation settings in settings.yaml and
        serve as emergency fallbacks when a block is absent.  Follows the
        same pattern as PlannerConfig.from_yaml().

        Args:
            config: Full settings.yaml dict (or any compatible sub-dict).

        Returns:
            VerifierConfig populated from the provided settings dict.
        """
        llm = config.get("llm", {})
        agent = config.get("agent", {})
        v = config.get("verifier", {})
        return cls(
            model_name=llm.get("model_name", "qwen2:1.5b"),
            base_url=llm.get("base_url", "http://localhost:11434"),
            temperature=llm.get("temperature", 0.1),
            max_tokens=llm.get("max_tokens", 200),
            timeout=llm.get("timeout", 60),
            max_context_chars=llm.get("max_context_chars", 900),
            max_docs=llm.get("max_docs", 3),
            max_chars_per_doc=llm.get("max_chars_per_doc", 500),
            max_iterations=agent.get("max_verification_iterations", 2),
            enable_entity_path_validation=v.get("enable_entity_path_validation", True),
            enable_contradiction_detection=v.get("enable_contradiction_detection", False),
            enable_credibility_scoring=v.get("enable_credibility_scoring", True),
            contradiction_threshold=v.get("contradiction_threshold", 0.85),
            min_credibility_score=v.get("min_credibility_score", 0.5),
            entity_coverage_threshold=v.get("entity_coverage_threshold", 0.5),
            nli_model=v.get("nli_model", "cross-encoder/nli-distilroberta-base"),
            nli_max_input_chars=v.get("nli_max_input_chars", 200),
            spacy_max_chars=v.get("spacy_max_chars", 500),
            credibility_weight_cross_ref=v.get("credibility_weight_cross_ref", 0.4),
            credibility_weight_entity_freq=v.get("credibility_weight_entity_freq", 0.3),
            credibility_weight_provenance=v.get("credibility_weight_provenance", 0.3),
            credibility_cross_ref_max=v.get("credibility_cross_ref_max", 3.0),
            credibility_provenance_baseline=v.get("credibility_provenance_baseline", 0.5),
            confidence_high_threshold=v.get("confidence_high_threshold", 0.8),
            confidence_medium_threshold=v.get("confidence_medium_threshold", 0.5),
            min_claim_chars=v.get("min_claim_chars", 15),
            max_entities_to_verify=v.get("max_entities_to_verify", 5),
            max_key_phrases=v.get("max_key_phrases", 10),
            heuristic_contradiction_threshold=v.get("heuristic_contradiction_threshold", 0.5),
            format_sentence_boundary_fraction=v.get("format_sentence_boundary_fraction", 0.7),
            credibility_entity_freq_normalizer_spacy=v.get("credibility_entity_freq_normalizer_spacy", 5.0),
            credibility_entity_freq_normalizer_regex=v.get("credibility_entity_freq_normalizer_regex", 10.0),
        )


class ConfidenceLevel(Enum):
    """Confidence level derived from the fraction of verified claims."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class VerificationResult:
    """
    Result of the verification stage.

    Attributes
    ----------
    answer :
        Generated (and potentially self-corrected) answer string.
    iterations :
        Number of self-correction iterations executed.
    verified_claims :
        Claims whose entities were found in graph or context.
    violated_claims :
        Claims that could not be verified.
    all_verified :
        True when all extracted claims passed verification.
    pre_validation :
        Output of the pre-generation validation stage.
    timing_ms :
        Total wall-clock time in milliseconds.
    iteration_history :
        Per-iteration diagnostics (answer, claims, latency, error flag).
    confidence_high_threshold :
        Verified-ratio threshold for HIGH confidence (stored with result
        for reproducibility independent of the config in scope at read time).
    confidence_medium_threshold :
        Verified-ratio threshold for MEDIUM confidence.
    """

    answer: str
    iterations: int
    verified_claims: List[str] = field(default_factory=list)
    violated_claims: List[str] = field(default_factory=list)
    all_verified: bool = False
    pre_validation: Optional[PreValidationResult] = None
    timing_ms: float = 0.0
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    confidence_high_threshold: float = 0.8
    confidence_medium_threshold: float = 0.5

    @property
    def confidence(self) -> ConfidenceLevel:
        """
        Confidence level based on the fraction of verified claims.

        Returns LOW when no claims were extracted (e.g., one-word answers
        that contain no verifiable entities).
        """
        total = len(self.verified_claims) + len(self.violated_claims)
        if total == 0:
            return ConfidenceLevel.LOW
        ratio = len(self.verified_claims) / total
        if ratio >= self.confidence_high_threshold:
            return ConfidenceLevel.HIGH
        if ratio >= self.confidence_medium_threshold:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


# =============================================================================
# PRE-GENERATION VALIDATOR
# =============================================================================


class PreGenerationValidator:
    """
    Pre-generation validation stage for S_V.

    Implements three sequential checks before answer generation:

    1. **Entity-Path Validation** — verifies that retrieved chunks cover all
       query entities.  Uses ``find_chunks_by_entity_multihop`` when a
       KuzuDB graph store is available; falls back to substring matching.

    2. **Contradiction Detection** — pairwise NLI check on adjacent chunk
       pairs (O(n); non-adjacent pairs are not checked).
       Reference: Bowman et al. (2015). arXiv:1508.05326;
       Reimers & Gurevych (2019). arXiv:1908.10084.
       Disabled by default; numeric-divergence heuristic as fallback.

    3. **Source Credibility Scoring** — weighted combination of
       cross-reference corroboration, entity-mention density, and retrieval
       provenance.  Chunks below ``min_credibility_score`` are filtered;
       at least one chunk is always retained.
    """

    # Compiled at class level — reused across all instances.
    # Matches "<CapitalisedWord> was/is/has/had <number>" for the numeric
    # contradiction heuristic.
    _NUMBER_PATTERN = re.compile(
        r"(\b[A-Z][a-z]+\b)\s+(?:was|is|has|had)\s+(\d+(?:\.\d+)?)"
    )
    # Capitalised proper-noun sequences for key-phrase extraction.
    _PROPER_NOUN_PATTERN = re.compile(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    )
    # Numeric tokens with optional unit word.
    _NUMERIC_PHRASE_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:\s+\w+)?")

    def __init__(
        self,
        config: VerifierConfig,
        graph_store: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : VerifierConfig
        graph_store : KuzuGraphStore or compatible, optional
            When provided, entity-path validation uses graph lookups.
        """
        self.config = config
        self.graph_store = graph_store
        # Lazy-loaded; only instantiated when contradiction detection is
        # enabled and Transformers is available.
        self._nli_pipeline: Optional[Any] = None
        logger.info(
            "PreGenerationValidator initialised: entity_path=%s, "
            "contradiction=%s, credibility=%s",
            config.enable_entity_path_validation,
            config.enable_contradiction_detection,
            config.enable_credibility_scoring,
        )

    def validate(
        self,
        context: List[str],
        query: str,
        entities: Optional[List[str]] = None,
        hop_sequence: Optional[List[Dict[str, Any]]] = None,
    ) -> PreValidationResult:
        """
        Run all three pre-generation validation checks sequentially.

        Parameters
        ----------
        context :
            Context chunks from the Navigator.
        query :
            Original user query (used for logging).
        entities :
            Query entities from the Planner (used by entity-path check).
        hop_sequence :
            Hop plan from the Planner.  Reserved for future graph-path
            planning; not used in the current implementation.

        Returns
        -------
        PreValidationResult
        """
        start_time = time.time()
        result = PreValidationResult()

        if not context:
            result.status = ValidationStatus.INSUFFICIENT_EVIDENCE
            result.filtered_context = []
            result.details["error"] = "No context available"
            return result

        result.filtered_context = context.copy()

        # ── Check 1: Entity-Path Validation ──────────────────────────────────
        if self.config.enable_entity_path_validation and entities:
            path_valid, path_details = self._validate_entity_path(context, entities)
            result.entity_path_valid = path_valid
            result.details["entity_path"] = path_details
            if not path_valid:
                logger.warning(
                    "Entity-path validation failed for query: '%s'", query[:60]
                )
                # Continue with generation; the INSUFFICIENT_EVIDENCE prompt
                # instructs the LLM to qualify its answer accordingly.
                result.status = ValidationStatus.INSUFFICIENT_EVIDENCE

        # ── Check 2: Contradiction Detection ─────────────────────────────────
        if self.config.enable_contradiction_detection and len(context) > 1:
            contradictions = self._detect_contradictions(context)
            result.contradictions = contradictions
            if contradictions:
                logger.warning("%d contradiction(s) detected", len(contradictions))
                result.details["contradictions"] = [
                    {"chunk1_idx": c[0], "chunk2_idx": c[1], "score": c[2]}
                    for c in contradictions
                ]
                result.filtered_context = self._resolve_contradictions(
                    context, contradictions
                )
                if len(result.filtered_context) < len(context):
                    result.status = ValidationStatus.CONTRADICTION_DETECTED

        # ── Check 3: Source Credibility Scoring ───────────────────────────────
        if self.config.enable_credibility_scoring:
            credibility_scores = self._compute_credibility(
                result.filtered_context, context
            )
            result.credibility_scores = credibility_scores
            high_cred = [
                chunk
                for chunk, score in zip(result.filtered_context, credibility_scores)
                if score >= self.config.min_credibility_score
            ]
            if high_cred:
                result.filtered_context = high_cred
            elif credibility_scores:
                # Always retain the highest-credibility chunk so the
                # generator is never given an empty context.
                best_idx = credibility_scores.index(max(credibility_scores))
                result.filtered_context = [result.filtered_context[best_idx]]
                result.status = ValidationStatus.LOW_CREDIBILITY

        result.validation_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Pre-validation: status=%s, context=%d/%d, time=%.0fms",
            result.status.value,
            len(result.filtered_context),
            len(context),
            result.validation_time_ms,
        )
        return result

    def _validate_entity_path(
        self,
        context: List[str],
        entities: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check whether retrieved chunks cover all query entities.

        Prefers KuzuDB ``find_chunks_by_entity_multihop``; falls back to
        ``graph_search`` (HybridStore) and then to substring matching.

        Returns
        -------
        (is_valid, details)
            ``is_valid`` is True when the fraction of found entities meets
            ``config.entity_coverage_threshold``.
        """
        details: Dict[str, Any] = {
            "entities_found": [],
            "entities_missing": [],
            "path_exists": False,
        }

        if not self.graph_store:
            # Fallback: substring matching against concatenated context.
            context_text = " ".join(context).lower()
            for entity in entities:
                if entity.lower() in context_text:
                    details["entities_found"].append(entity)
                else:
                    details["entities_missing"].append(entity)
        else:
            for entity in entities:
                try:
                    found = False
                    if hasattr(self.graph_store, "find_chunks_by_entity_multihop"):
                        results = self.graph_store.find_chunks_by_entity_multihop(
                            entity_name=entity, max_results=1
                        )
                        found = bool(results)
                    elif hasattr(self.graph_store, "graph_search"):
                        results = self.graph_store.graph_search(
                            entities=[entity], top_k=1
                        )
                        found = bool(results)
                    if found:
                        details["entities_found"].append(entity)
                    else:
                        details["entities_missing"].append(entity)
                except (AttributeError, TypeError, RuntimeError) as exc:
                    logger.warning(
                        "Entity-path graph lookup failed for '%s': %s", entity, exc
                    )
                    details["entities_missing"].append(entity)

        coverage = len(details["entities_found"]) / max(1, len(entities))
        details["coverage"] = coverage
        details["path_exists"] = coverage >= self.config.entity_coverage_threshold
        return details["path_exists"], details

    def _detect_contradictions(
        self,
        context: List[str],
    ) -> List[Tuple[int, int, float]]:
        """
        Detect contradictions between consecutive chunk pairs.

        Uses an NLI cross-encoder when Transformers is available; otherwise
        falls back to the numeric-divergence heuristic.

        Note: only consecutive pairs (i, i+1) are checked — O(n) — to stay
        within edge CPU budget.  Non-adjacent contradictions are not detected.

        Reference (NLI): Bowman et al. (2015). arXiv:1508.05326.
        Reference (model): Reimers & Gurevych (2019). arXiv:1908.10084.

        Returns
        -------
        list of (idx1, idx2, score) tuples
            Index-based so downstream resolution addresses original chunks.
        """
        if TRANSFORMERS_AVAILABLE and self.config.enable_contradiction_detection:
            try:
                if self._nli_pipeline is None:
                    self._nli_pipeline = _hf_pipeline(
                        "text-classification",
                        model=self.config.nli_model,
                        device=-1,  # CPU; HuggingFace uses -1 (not "cpu")
                    )
                contradictions: List[Tuple[int, int, float]] = []
                for i in range(len(context) - 1):
                    c1 = context[i][: self.config.nli_max_input_chars]
                    c2 = context[i + 1][: self.config.nli_max_input_chars]
                    result = self._nli_pipeline(
                        "%s [SEP] %s" % (c1, c2), truncation=True
                    )
                    if (
                        result
                        and result[0]["label"] == "CONTRADICTION"
                        and result[0]["score"] >= self.config.contradiction_threshold
                    ):
                        contradictions.append((i, i + 1, result[0]["score"]))
                return contradictions
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning(
                    "NLI contradiction detection failed (%s); "
                    "falling back to heuristic detection.",
                    exc,
                )
                return self._heuristic_contradiction_detection(context)
        else:
            logger.warning(
                "Transformers unavailable — falling back to heuristic contradiction detection."
            )
            return self._heuristic_contradiction_detection(context)

    def _heuristic_contradiction_detection(
        self,
        context: List[str],
    ) -> List[Tuple[int, int, float]]:
        """
        Numeric-divergence heuristic for offline contradiction detection.

        Flags chunk pairs where the same capitalised entity is assigned
        substantially different numeric values (relative difference > 50 %).
        Score is set to the actual divergence magnitude (capped at 1.0)
        rather than a fixed constant.

        This is a conservative approximation: it misses semantic
        contradictions and can over-fire on entities with multiple numeric
        attributes.  Appropriate only as a last-resort fallback.

        Returns
        -------
        list of (idx1, idx2, score) tuples
        """
        contradictions: List[Tuple[int, int, float]] = []
        entity_values: Dict[str, List[Tuple[int, float]]] = {}
        for i, chunk in enumerate(context):
            for entity, value in self._NUMBER_PATTERN.findall(chunk):
                entity_values.setdefault(entity, []).append((i, float(value)))
        for entity, values in entity_values.items():
            for j in range(len(values)):
                for k in range(j + 1, len(values)):
                    v1, v2 = values[j][1], values[k][1]
                    if min(v1, v2) > 0:
                        diff = abs(v1 - v2) / max(v1, v2)
                        if diff > self.config.heuristic_contradiction_threshold:
                            contradictions.append(
                                (values[j][0], values[k][0], min(1.0, diff))
                            )
        return contradictions

    def _resolve_contradictions(
        self,
        context: List[str],
        contradictions: List[Tuple[int, int, float]],
    ) -> List[str]:
        """
        Remove the most-contradicted chunks from the context.

        Uses index-based counting so the lookup is immune to string
        truncation differences between detection (chunk[:nli_max_input_chars])
        and resolution (full-length chunk strings).

        Returns the original context unchanged if filtering would remove
        all chunks.
        """
        contradiction_counts: Dict[int, int] = {}
        for idx1, idx2, _ in contradictions:
            contradiction_counts[idx1] = contradiction_counts.get(idx1, 0) + 1
            contradiction_counts[idx2] = contradiction_counts.get(idx2, 0) + 1
        if not contradiction_counts:
            return context
        max_count = max(contradiction_counts.values())
        filtered = [
            chunk
            for i, chunk in enumerate(context)
            if contradiction_counts.get(i, 0) < max_count
        ]
        return filtered if filtered else context

    def _compute_credibility(
        self,
        filtered_context: List[str],
        original_context: List[str],
    ) -> List[float]:
        """
        Compute credibility scores for each chunk in ``filtered_context``.

        Three signals:
        1. Cross-references: number of other chunks sharing a key phrase
           (corroboration proxy).
        2. Entity frequency: SpaCy NER density as an information-richness
           proxy.  Regex proper-noun count as fallback.
        3. Retrieval provenance: currently always False because Navigator
           does not forward retrieval-source metadata (known limitation).
           The provenance weight is applied uniformly at the
           ``credibility_provenance_baseline`` level (0.5).
        """
        scores: List[float] = []
        for chunk in filtered_context:
            cred = SourceCredibility(text=chunk)
            key_phrases = self._extract_key_phrases(chunk)

            # Cross-reference: any other chunk that shares a key phrase.
            for other in original_context:
                if other != chunk:
                    other_lower = other.lower()
                    for phrase in key_phrases:
                        if phrase.lower() in other_lower:
                            cred.cross_references += 1
                            break

            # Entity-frequency signal.
            if SPACY_AVAILABLE and NLP:
                doc = NLP(chunk[: self.config.spacy_max_chars])
                cred.entity_frequency = min(1.0, len(doc.ents) / self.config.credibility_entity_freq_normalizer_spacy)
            else:
                proper_count = len(self._PROPER_NOUN_PATTERN.findall(chunk))
                cred.entity_frequency = min(1.0, proper_count / self.config.credibility_entity_freq_normalizer_regex)

            # Provenance: always False (see docstring).
            cred.is_graph_based = False

            scores.append(
                cred.compute_score(
                    weight_cross_ref=self.config.credibility_weight_cross_ref,
                    weight_entity_freq=self.config.credibility_weight_entity_freq,
                    weight_provenance=self.config.credibility_weight_provenance,
                    cross_ref_max=self.config.credibility_cross_ref_max,
                    provenance_baseline=self.config.credibility_provenance_baseline,
                )
            )
        return scores

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract named entities and numeric phrases for cross-reference scoring.

        Returns a deterministically sorted, deduplicated list of up to
        ``config.max_key_phrases`` items so results are reproducible
        regardless of set() insertion order.
        """
        phrases: List[str] = []
        if SPACY_AVAILABLE and NLP:
            doc = NLP(text[: self.config.spacy_max_chars])
            phrases.extend(ent.text for ent in doc.ents)
        phrases.extend(self._PROPER_NOUN_PATTERN.findall(text))
        phrases.extend(self._NUMERIC_PHRASE_PATTERN.findall(text))
        # Sort before deduplication for deterministic ordering (reproducibility).
        return sorted(set(phrases))[: self.config.max_key_phrases]


# =============================================================================
# MAIN VERIFIER CLASS
# =============================================================================


class Verifier:
    """
    S_V: Verifier with pre-generation validation and self-correction.

    Primary public interface: ``generate_and_verify()``.
    """

    # ── Prompt Templates ──────────────────────────────────────────────────────

    ANSWER_PROMPT = """You are a factual QA assistant. Answer based ONLY on the context below.

Rules:
- Give the shortest possible answer: a name, place, date, or yes/no.
- Do NOT explain or add sentences beyond the direct answer.
- If the answer is a person, place, or thing: reply with just that name.
- If the answer is yes/no: reply with just "yes" or "no".
- If the context does not contain the answer: reply with "I don't know."

Context:
{context}

Question: {query}

Answer (as short as possible):"""

    CORRECTION_PROMPT = """Your previous answer contained unverified claims.

Unverified claims:
{violations}

Context:
{context}

Question: {query}

Give the shortest correct answer (name, place, date, or yes/no only):"""

    INSUFFICIENT_EVIDENCE_PROMPT = """Based on the available context, I could not find sufficient evidence to fully answer your question.

Context:
{context}

Question: {query}

Please provide a partial answer based on the available evidence, clearly indicating what information is missing:"""

    # ── Class-Level Compiled Regex Constants ──────────────────────────────────
    # Hoisted from per-call compilation in _verify_claim for performance.

    # Multi-word proper noun sequences — shared constant from _settings.py.
    _MULTI_PROPER_NOUN_RE = _PROPER_NOUN_RE
    # Single capitalised proper nouns of at least 3 characters.
    _SINGLE_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
    # Quoted strings treated as entity mentions.
    _QUOTED_RE = re.compile(r'"([^"]+)"')
    # Sentence boundary splitter for regex fallback in _extract_claims.
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    # Stopwords excluded from entity extraction in _verify_claim.
    # Capitalised sentence-starters and demonyms that match proper-noun
    # patterns but are never factual entities worth verifying.
    _CLAIM_STOPWORDS: frozenset = frozenset({
        "The", "This", "That", "These", "Those",
        "However", "Therefore", "Furthermore", "Moreover", "Although",
        "American", "British", "European", "Australian", "Canadian",
        "Yes", "No",
    })

    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        graph_store: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : VerifierConfig, optional
            Uses VerifierConfig() defaults when None.
        graph_store : KuzuGraphStore or compatible, optional
            Injected for claim verification and entity-path validation.
            Can be set or replaced later via ``set_graph_store()``.
        """
        self.config = config or VerifierConfig()
        self.graph_store = graph_store
        self.pre_validator = PreGenerationValidator(self.config, graph_store)
        logger.info(
            "Verifier initialised: model=%s, max_iterations=%d, "
            "entity_path_validation=%s",
            self.config.model_name,
            self.config.max_iterations,
            self.config.enable_entity_path_validation,
        )

    def set_graph_store(self, graph_store: Any) -> None:
        """Inject or replace the graph store at runtime."""
        self.graph_store = graph_store
        self.pre_validator.graph_store = graph_store
        logger.info("Graph store connected to Verifier and PreGenerationValidator")

    # ── LLM Interaction ───────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Tuple[str, float]:
        """
        Call the Ollama generate endpoint.

        Returns ``(response_text, latency_ms)``.  On failure, returns an
        error-sentinel string beginning with ``"[Error:"`` so callers can
        detect failures without raising exceptions.
        """
        start = time.time()
        try:
            response = requests.post(
                "%s/api/generate" % self.config.base_url,
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
                timeout=self.config.timeout,
            )
            latency_ms = (time.time() - start) * 1000
            if response.status_code != 200:
                logger.error("Ollama API error: HTTP %d", response.status_code)
                return "[Error: API returned %d]" % response.status_code, latency_ms
            data = response.json()
            # Guard against Ollama error responses (e.g., model not found).
            if "error" in data:
                logger.error("Ollama error response: %s", data["error"])
                return "[Error: %s]" % str(data["error"])[:100], latency_ms
            return data.get("response", "").strip(), latency_ms
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start) * 1000
            logger.error("Ollama timeout after %ds", self.config.timeout)
            return "[Error: LLM timeout - try reducing context size]", latency_ms
        except requests.exceptions.ConnectionError:
            latency_ms = (time.time() - start) * 1000
            logger.error("Cannot connect to Ollama at %s", self.config.base_url)
            return "[Error: Cannot connect to Ollama - is it running?]", latency_ms
        except requests.exceptions.RequestException as exc:
            latency_ms = (time.time() - start) * 1000
            logger.error("LLM request failed: %s", exc)
            return "[Error: %s]" % str(exc)[:100], latency_ms

    # ── Context Formatting ────────────────────────────────────────────────────

    def _format_context(self, context: List[str]) -> str:
        """
        Format context chunks into a single prompt string with size limits.

        Strategy:
        1. Take at most ``max_docs`` chunks.
        2. Truncate each chunk at ``max_chars_per_doc``, preferring a
           sentence boundary if one falls in the last 30 % of the window.
        3. Stop adding chunks once ``max_context_chars`` is reached.
        """
        if not context:
            return "No context available."
        formatted_parts: List[str] = []
        total_chars = 0
        for i, doc in enumerate(context[: self.config.max_docs]):
            if len(doc) > self.config.max_chars_per_doc:
                truncated = doc[: self.config.max_chars_per_doc]
                last_period = truncated.rfind(". ")
                if last_period > self.config.max_chars_per_doc * self.config.format_sentence_boundary_fraction:
                    truncated = truncated[: last_period + 1]
                else:
                    last_space = truncated.rfind(" ")
                    if last_space > 0:
                        truncated = truncated[:last_space] + "..."
            else:
                truncated = doc
            part = "[%d] %s" % (i + 1, truncated)
            if total_chars + len(part) > self.config.max_context_chars:
                logger.debug(
                    "Context budget reached at doc %d/%d", i + 1, len(context)
                )
                break
            formatted_parts.append(part)
            total_chars += len(part) + 2  # +2 for "\n\n" separator
        logger.debug(
            "Context formatted: %d docs, %d chars (~%d tokens)",
            len(formatted_parts),
            total_chars,
            total_chars // 4,
        )
        return "\n\n".join(formatted_parts)

    # ── Claim Extraction ──────────────────────────────────────────────────────

    def _extract_claims(self, answer: str) -> List[str]:
        """
        Split a generated answer into atomic factual claims.

        Uses SpaCy sentence segmentation when available; falls back to
        punctuation-based regex splitting.  Meta-statements (hedges, "I don't
        know", etc.) are filtered out because they are not verifiable claims.

        Reference: Kryscinski et al. (2020). "Evaluating the Factual
        Consistency of Abstractive Text Summarization." EMNLP 2020.
        arXiv:1910.12840. — Motivation for claim-level factual consistency
        checking as a quality proxy.
        """
        if answer.startswith("[Error:"):
            return []
        if SPACY_AVAILABLE and NLP:
            doc = NLP(answer)
            claims = [
                s.text.strip()
                for s in doc.sents
                if len(s.text.strip()) > self.config.min_claim_chars
            ]
        else:
            claims = self._SENTENCE_SPLIT_RE.split(answer)
            claims = [
                c.strip()
                for c in claims
                if len(c.strip()) > self.config.min_claim_chars
            ]
        meta_patterns = (
            "based on the context",
            "according to the",
            "i cannot answer",
            "i don't know",
            "not enough information",
            "the context does not",
            "the context doesn't",
            "error:",
            "insufficient evidence",
        )
        filtered = [
            c for c in claims if not any(p in c.lower() for p in meta_patterns)
        ]
        logger.debug("Extracted %d claims from answer", len(filtered))
        return filtered

    # ── Claim Verification ────────────────────────────────────────────────────

    def _verify_claim(
        self,
        claim: str,
        context: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Verify a single claim against the graph store and/or context text.

        Strategy:
        1. Extract proper-noun entities via regex.
        2. Query KuzuDB (``find_chunks_by_entity_multihop``) or fall back to
           ``graph_search`` / ``get_entity_relations`` depending on interface.
        3. If graph store is absent or returns no results, fall back to
           substring matching in the retrieved context.
        4. Claims with no extractable entities are considered verified by
           default (nothing to contradict).

        This is a conservative entity-presence proxy, not logical entailment.
        For entailment-based verification see Kryscinski et al. (2020).
        arXiv:1910.12840.

        Returns
        -------
        (is_verified, reason_code) : (bool, str)
        """
        entities: List[str] = []
        entities.extend(self._MULTI_PROPER_NOUN_RE.findall(claim))
        entities.extend(self._SINGLE_PROPER_NOUN_RE.findall(claim))
        entities.extend(self._QUOTED_RE.findall(claim))
        entities = [e for e in entities if e not in self._CLAIM_STOPWORDS]
        if not entities:
            return True, "no_entities_to_verify"

        # ── Graph store verification ──────────────────────────────────────────
        if self.graph_store:
            for entity in entities[: self.config.max_entities_to_verify]:
                try:
                    found = False
                    if hasattr(self.graph_store, "find_chunks_by_entity_multihop"):
                        results = self.graph_store.find_chunks_by_entity_multihop(
                            entity_name=entity, max_results=1
                        )
                        found = bool(results)
                    elif hasattr(self.graph_store, "graph_search"):
                        results = self.graph_store.graph_search(
                            entities=[entity], top_k=1
                        )
                        found = bool(results)
                    elif hasattr(self.graph_store, "get_entity_relations"):
                        results = self.graph_store.get_entity_relations(entity)
                        found = bool(results)
                    if found:
                        return True, "graph_verified_%s" % entity
                except (AttributeError, TypeError, RuntimeError) as exc:
                    logger.debug(
                        "Graph query failed for entity '%s': %s", entity, exc
                    )

        # ── Context substring verification ────────────────────────────────────
        if context:
            context_text = " ".join(context).lower()
            for entity in entities[: self.config.max_entities_to_verify]:
                if entity.lower() in context_text:
                    return True, "context_verified_%s" % entity

        return False, "entities_not_found"

    # ── Main Verification Loop ────────────────────────────────────────────────

    def generate_and_verify(
        self,
        query: str,
        context: List[str],
        entities: Optional[List[str]] = None,
        hop_sequence: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        """
        Main entry point: pre-validation, generation, and self-correction.

        Algorithm
        ---------
        1. Run pre-generation validation (entity-path, contradiction,
           credibility filtering).
        2. Select the initial prompt based on validation status.
        3. For each iteration up to ``max_iterations``:
           a. Call the LLM.
           b. Skip empty or error responses (with best-answer fallback).
           c. Extract atomic claims from the answer.
           d. Verify each claim; return immediately if all pass.
           e. Otherwise re-prompt with CORRECTION_PROMPT listing violated
              claims.
        4. Return the best answer seen across all iterations.

        Reference: Madaan, A., et al. (2023). "Self-Refine: Iterative
        Refinement with Self-Feedback." NeurIPS 2023. arXiv:2303.17651.

        Parameters
        ----------
        query :
            Original user question.
        context :
            Retrieved chunks from the Navigator.
        entities :
            Query entities from the Planner (forwarded to pre-validation).
        hop_sequence :
            Hop plan from the Planner (reserved for future graph-path
            planning; passed through to pre-validation but not used there).

        Returns
        -------
        VerificationResult
        """
        if query is None:
            query = ""
        start_time = time.time()
        logger.info("[Verifier] query='%s'", query[:60])
        logger.info("[Verifier] context docs: %d", len(context))

        # ── Pre-generation validation ─────────────────────────────────────────
        pre_validation = self.pre_validator.validate(
            context=context,
            query=query,
            entities=entities,
            hop_sequence=hop_sequence,
        )
        working_context = pre_validation.filtered_context
        logger.info(
            "[Verifier] Pre-validation: %s, context %d/%d",
            pre_validation.status.value,
            len(working_context),
            len(context),
        )
        formatted_context = self._format_context(working_context)

        best_answer: Optional[str] = None
        best_verified: List[str] = []
        best_violated: List[str] = []
        iteration_history: List[Dict[str, Any]] = []
        violated_claims: List[str] = []

        # ── Self-correction loop ──────────────────────────────────────────────
        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()
            logger.info(
                "[Verifier] === Iteration %d/%d ===",
                iteration,
                self.config.max_iterations,
            )

            if iteration == 1:
                if pre_validation.status == ValidationStatus.INSUFFICIENT_EVIDENCE:
                    prompt = self.INSUFFICIENT_EVIDENCE_PROMPT.format(
                        context=formatted_context, query=query
                    )
                else:
                    prompt = self.ANSWER_PROMPT.format(
                        context=formatted_context, query=query
                    )
            else:
                prompt = self.CORRECTION_PROMPT.format(
                    violations="\n".join("- %s" % v for v in violated_claims),
                    context=formatted_context,
                    query=query,
                )

            answer, llm_latency = self._call_llm(prompt)
            logger.info("[Verifier] LLM response in %.0fms", llm_latency)

            # Guard: empty answers cannot be verified and must not be
            # returned as correct results.
            if not answer or answer.isspace():
                logger.warning(
                    "[Verifier] Empty answer in iteration %d; skipping.",
                    iteration,
                )
                iteration_history.append({
                    "iteration": iteration,
                    "answer": answer,
                    "claims": [],
                    "verified": [],
                    "violated": [],
                    "llm_latency_ms": llm_latency,
                    "error": True,
                })
                if best_answer:
                    break
                continue

            if answer.startswith("[Error:"):
                logger.warning("[Verifier] LLM error: %s", answer)
                iteration_history.append({
                    "iteration": iteration,
                    "answer": answer,
                    "claims": [],
                    "verified": [],
                    "violated": [],
                    "llm_latency_ms": llm_latency,
                    "error": True,
                })
                if best_answer:
                    break
                continue

            claims = self._extract_claims(answer)
            logger.info("[Verifier] %d claims extracted", len(claims))

            verified_claims: List[str] = []
            violated_claims = []
            for claim in claims:
                is_ok, reason = self._verify_claim(claim, working_context)
                if is_ok:
                    verified_claims.append(claim)
                    logger.debug(
                        "[Verifier] VERIFIED '%s...' (%s)", claim[:50], reason
                    )
                else:
                    violated_claims.append(claim)
                    logger.debug(
                        "[Verifier] VIOLATED '%s...' (%s)", claim[:50], reason
                    )
            logger.info(
                "[Verifier] Verification: %d verified, %d violated",
                len(verified_claims),
                len(violated_claims),
            )

            iter_time = (time.time() - iter_start) * 1000
            iteration_history.append({
                "iteration": iteration,
                "answer": answer,
                "claims": claims,
                "verified": verified_claims,
                "violated": violated_claims,
                "llm_latency_ms": llm_latency,
                "total_time_ms": iter_time,
                "error": False,
            })

            # Track the best answer across iterations (fewest violations).
            if best_answer is None or len(violated_claims) < len(best_violated):
                best_answer = answer
                best_verified = verified_claims
                best_violated = violated_claims

            if len(violated_claims) == 0:
                logger.info(
                    "[Verifier] All claims verified in iteration %d.", iteration
                )
                total_time = (time.time() - start_time) * 1000
                return VerificationResult(
                    answer=answer,
                    iterations=iteration,
                    verified_claims=verified_claims,
                    violated_claims=[],
                    all_verified=True,
                    pre_validation=pre_validation,
                    timing_ms=total_time,
                    iteration_history=iteration_history,
                    confidence_high_threshold=self.config.confidence_high_threshold,
                    confidence_medium_threshold=self.config.confidence_medium_threshold,
                )
            logger.info(
                "[Verifier] %d unverified claim(s); attempting correction.",
                len(violated_claims),
            )

        # ── Max iterations reached ────────────────────────────────────────────
        total_time = (time.time() - start_time) * 1000
        logger.warning(
            "[Verifier] Max iterations reached. Best result: %d verified, %d violated.",
            len(best_verified),
            len(best_violated),
        )
        return VerificationResult(
            answer=best_answer if best_answer is not None
            else "[Error: No valid answer generated]",
            iterations=self.config.max_iterations,
            verified_claims=best_verified,
            violated_claims=best_violated,
            all_verified=False,
            pre_validation=pre_validation,
            timing_ms=total_time,
            iteration_history=iteration_history,
            confidence_high_threshold=self.config.confidence_high_threshold,
            confidence_medium_threshold=self.config.confidence_medium_threshold,
        )

    def __call__(
        self,
        query: str,
        context: List[str],
        entities: Optional[List[str]] = None,
        hop_sequence: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        """Callable interface — forwards all arguments to generate_and_verify."""
        return self.generate_and_verify(
            query=query,
            context=context,
            entities=entities,
            hop_sequence=hop_sequence,
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_verifier(
    cfg: Optional[Dict[str, Any]] = None,
    graph_store: Optional[Any] = None,
    enable_pre_validation: bool = False,
) -> Verifier:
    """
    Factory function for Verifier — reads all values from a settings.yaml dict.

    Delegates to ``VerifierConfig.from_yaml()`` for YAML parsing.  The
    ``enable_pre_validation`` flag overrides the ``verifier.enable_*`` flags
    in the settings dict (useful for one-liner test construction).

    Parameters
    ----------
    cfg : dict, optional
        Full settings.yaml dict.  Relevant keys:
        ``llm.*``, ``agent.max_verification_iterations``, ``verifier.*``.
        Pass ``{"agent": {"max_verification_iterations": 1}}`` to construct
        a single-iteration verifier for unit tests.
    graph_store : KuzuGraphStore or compatible, optional
    enable_pre_validation : bool
        When True, activates entity-path and credibility validation,
        overriding the settings dict values.

    Returns
    -------
    Verifier
    """
    if cfg is None:
        cfg = _load_settings()
    config = VerifierConfig.from_yaml(cfg)
    if enable_pre_validation:
        config.enable_entity_path_validation = True
        config.enable_credibility_scoring = True
    return Verifier(config, graph_store)


# =============================================================================
# SMOKE TEST  (python verifier.py)
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    print("=" * 70)
    print("S_V: Verifier smoke test")
    print("SpaCy available: %s" % SPACY_AVAILABLE)
    print("Transformers available: %s" % TRANSFORMERS_AVAILABLE)
    print("=" * 70)

    _test_context = [
        "Albert Einstein was a German-born theoretical physicist who developed"
        " the theory of relativity.",
        "Einstein received the Nobel Prize in Physics in 1921 for his"
        " explanation of the photoelectric effect.",
        "He published more than 300 scientific papers and became a symbol of"
        " genius.",
        "Einstein was born in Ulm, Germany, on March 14, 1879.",
        "He worked at the Swiss Patent Office while developing his groundbreaking"
        " theories.",
    ]
    _test_query = (
        "When was Einstein born and what did he receive the Nobel Prize for?"
    )

    print("\nQuery: %s" % _test_query)
    print("Context docs: %d" % len(_test_context))

    _test_cfg: Dict[str, Any] = {
        "llm": {
            "max_context_chars": 2000,
            "max_docs": 5,
            "max_chars_per_doc": 400,
        },
        "agent": {"max_verification_iterations": 3},
        "verifier": {
            "enable_entity_path_validation": True,
            "enable_credibility_scoring": True,
        },
    }
    _verifier = create_verifier(cfg=_test_cfg, enable_pre_validation=True)

    print("\n--- Pre-Generation Validation ---")
    _pre = _verifier.pre_validator.validate(
        context=_test_context,
        query=_test_query,
        entities=["Einstein", "Nobel Prize"],
    )
    print("Status: %s" % _pre.status.value)
    print("Entity-path valid: %s" % _pre.entity_path_valid)
    print("Contradictions: %d" % len(_pre.contradictions))
    print(
        "Filtered context: %d/%d"
        % (len(_pre.filtered_context), len(_test_context))
    )
    print(
        "Credibility scores: %s" % [("%.2f" % s) for s in _pre.credibility_scores]
    )
    print("Validation time: %.0fms" % _pre.validation_time_ms)

    print("\n--- Full Verification (requires Ollama) ---")
    try:
        _result = _verifier.generate_and_verify(
            query=_test_query,
            context=_test_context,
            entities=["Einstein", "Nobel Prize"],
        )
        print("Answer: %s" % _result.answer)
        print("Iterations: %d" % _result.iterations)
        print("All verified: %s" % _result.all_verified)
        print("Verified claims: %d" % len(_result.verified_claims))
        print("Violated claims: %d" % len(_result.violated_claims))
        print("Confidence: %s" % _result.confidence.value)
        print("Total time: %.0fms" % _result.timing_ms)
    except Exception as exc:
        print("Ollama not available: %s" % exc)
        print("Verifier logic functional; LLM generation requires Ollama.")

    print("\n" + "=" * 70)
