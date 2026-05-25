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
   The Verifier defaults to TWO checks before generation. A third check
   exists as an ablation-only toggle.

   a) Entity-Path Validation (multi-hop queries) — DEFAULT ON
      Verifies that retrieved chunks cover the query entities.  When a
      KuzuDB graph store is available, ``find_chunks_by_entity_multihop``
      is used; otherwise falls back to substring matching.

   b) Source Credibility Scoring — DEFAULT ON
      Weighted combination (40 % cross-references, 30 % entity-mention
      frequency, 30 % retrieval provenance — see SourceCredibility for
      the per-weight defense). Graph-vs-vector provenance is now a real
      signal from the Navigator (B2 fix); pre-B2 it was a constant
      baseline because Navigator did not forward retrieval-source metadata.

   c) Contradiction Detection — DEFAULT OFF (ablation-only, B6)
      NLI-based pairwise detection on adjacent chunk pairs requires a
      ~270 MB cross-encoder download (Reimers & Gurevych, 2019;
      arXiv:1908.10084) and contradicts the edge-deployment constraint of
      the thesis. The Navigator already runs a numeric-divergence
      heuristic filter (``enable_contradiction_filter`` in settings.yaml)
      which removes obviously contradictory chunks before they reach the
      Verifier. The Verifier-side check is therefore retained only as a
      research-mode toggle for ablation studies — set
      ``enable_contradiction_detection: true`` to enable. The thesis
      methodology paragraph describes the system with this off.

2. GENERATION WITH A QUANTISED SLM
   Phi-3-Mini (or any Ollama-hosted model) is prompted with a compact
   context budget tuned for edge hardware (<16 GB RAM). Context limits
   are set in config/settings.yaml under the ``llm`` block.

   Between pre-validation and generation, two passes shape the prompt
   (`§11.13` + `§11.16.1`):

   - The verifier caps the Navigator's context to ``max_docs`` (default
     5) **by RRF order first**, then ``_reorder_by_question_relevance``
     reorders only **within that kept window**. Selection (set
     membership) is owned by the retrieval-score signal; the reorder
     mitigates small-LLM positional bias (Liu et al. 2023, *Lost in the
     Middle*, TACL/arXiv:2307.03172). The reorder cannot evict a chunk.
   - The reorder scoring combines IDF-weighted query-term overlap
     (Spärck Jones 1972), sqrt-length normalisation, and a
     structural-coverage floor that protects distinctive-entity chunks
     from demotion (Robertson 2004).

   The headline answer-correctness verdict applied downstream by the
   evaluator is **Soft-EM** (token-F1 ≥ `benchmark.answer_f1_threshold`,
   default 0.6) — strict EM systematically under-counts answers that
   differ from gold only by a trailing category word. The Verifier
   itself does not compute it; it lives in
   ``src.thesis_evaluations.benchmark_datasets`` and is mirrored by
   ``diagnose_verbose.py``.

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
import math
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
    - ``retrieval_provenance``: graph-retrieved chunks score higher than
      vector/BM25-only chunks (weight: ``credibility_weight_provenance``,
      default 30 %). Real provenance is supplied via
      ``chunk_is_graph_based`` from the Navigator (B2-fix); when absent the
      historical constant baseline (0.5) is applied so the term degenerates
      to a uniform offset rather than crashing.

    B5 note — defense of the 40 / 30 / 30 weights:
        The weights are documented as a deliberate inspection-time choice
        rather than the output of a grid-search calibration.  Cross-reference
        corroboration receives the largest share because two independent
        chunks agreeing on a fact is a stronger correctness signal than
        either signal in isolation (Knowledge-Vault style multi-source
        fusion, Dong et al., 2014, KDD).  Entity-frequency and
        retrieval-provenance are weighted equally because they measure
        independent dimensions (information density vs. retrieval-path
        quality) and the thesis does not claim one dominates the other.

        The total contribution of the credibility filter is bounded above
        by the chunk-eviction rate at ``min_credibility_score``; on the
        thesis HotpotQA evaluation this filter evicts < 10 % of chunks,
        so the weights' individual influence on final EM/F1 is small.
        The thesis reports a single ablation row ("Verifier w/o credibility
        filter") rather than a weight-sweep, which is methodologically
        sufficient at this scale.

    The provenance term used to be a constant baseline because the Navigator
    did not forward retrieval-source metadata.  This is fixed in B2 — see
    ``PreGenerationValidator.validate``'s ``chunk_is_graph_based`` parameter.
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
    temperature : Sampling temperature (0.0 = fully deterministic).
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
    temperature: float = 0.0
    max_tokens: int = 200
    timeout: int = 60

    # Context settings
    max_context_chars: int = 900
    max_docs: int = 5
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
            temperature=llm.get("temperature", 0.0),
            max_tokens=llm.get("max_tokens", 200),
            timeout=llm.get("timeout", 60),
            max_context_chars=llm.get("max_context_chars", 900),
            max_docs=llm.get("max_docs", 5),
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

    DEFAULT pipeline (two checks):

    1. **Entity-Path Validation** — verifies that retrieved chunks cover all
       query entities.  Uses ``find_chunks_by_entity_multihop`` when a
       KuzuDB graph store is available; falls back to substring matching.

    2. **Source Credibility Scoring** — weighted combination of
       cross-reference corroboration, entity-mention density, and retrieval
       provenance.  Chunks below ``min_credibility_score`` are filtered;
       at least one chunk is always retained.

    ABLATION-ONLY check (default OFF, B6):

    3. **Contradiction Detection** — pairwise NLI check on adjacent chunk
       pairs (O(n); non-adjacent pairs are not checked).
       Reference: Bowman et al. (2015). arXiv:1508.05326;
       Reimers & Gurevych (2019). arXiv:1908.10084.
       The Navigator already runs a numeric-divergence contradiction filter
       on the same context; this Verifier-side check is research-mode only.
       Enable via ``enable_contradiction_detection: true``.
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
        chunk_is_graph_based: Optional[List[bool]] = None,
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
        chunk_is_graph_based :
            B2-fix: per-chunk retrieval-provenance flag (parallel to ``context``).
            True for chunks retrieved via the KuzuDB graph path. Used by the
            credibility scorer to give graph-corroborated chunks a higher
            provenance score. None disables the provenance signal (falls back
            to the constant baseline used pre-B2).

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
            # B2-fix: remap the provenance flag onto the (possibly trimmed)
            # filtered_context by exact-text lookup against the original
            # context. None entries fall back to the constant baseline inside
            # _compute_credibility so older callers that pass no provenance
            # see no behavior change.
            filtered_graph_flags: Optional[List[bool]] = None
            if chunk_is_graph_based is not None and len(chunk_is_graph_based) == len(context):
                _by_text = {c: g for c, g in zip(context, chunk_is_graph_based)}
                filtered_graph_flags = [
                    _by_text.get(c, False) for c in result.filtered_context
                ]

            credibility_scores = self._compute_credibility(
                result.filtered_context, context, entities=entities,
                chunk_is_graph_based=filtered_graph_flags,
            )
            result.credibility_scores = credibility_scores
            # Track which chunks survive credibility filtering so the high-cred
            # subset stays aligned with filtered_graph_flags for downstream use.
            keep_indices = [
                i for i, score in enumerate(credibility_scores)
                if score >= self.config.min_credibility_score
            ]
            if keep_indices:
                result.filtered_context = [result.filtered_context[i] for i in keep_indices]
                if filtered_graph_flags is not None:
                    filtered_graph_flags = [filtered_graph_flags[i] for i in keep_indices]
            elif credibility_scores:
                # Always retain the highest-credibility chunk so the
                # generator is never given an empty context.
                best_idx = credibility_scores.index(max(credibility_scores))
                result.filtered_context = [result.filtered_context[best_idx]]
                if filtered_graph_flags is not None:
                    filtered_graph_flags = [filtered_graph_flags[best_idx]]
                result.status = ValidationStatus.LOW_CREDIBILITY
            # Expose the final per-chunk provenance flags for callers that want
            # to log retrieval-source statistics (e.g. ablation diagnostics).
            if filtered_graph_flags is not None:
                result.details["chunk_is_graph_based"] = filtered_graph_flags

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
        entities: Optional[List[str]] = None,
        chunk_is_graph_based: Optional[List[bool]] = None,
    ) -> List[float]:
        """
        Compute credibility scores for each chunk in ``filtered_context``.

        Three signals:
        1. Cross-references: number of other chunks sharing a key phrase
           (corroboration proxy).
        2. Entity frequency: SpaCy NER density as an information-richness
           proxy.  Regex proper-noun count as fallback.
        3. Retrieval provenance (B2-fix): graph-retrieved chunks get the full
           weight (1.0); vector/BM25-only chunks get ``credibility_provenance_baseline``.
           Before B2, ``is_graph_based`` was always False — see the docstring
           addendum in compute_score(). Pass ``chunk_is_graph_based`` (parallel
           to ``filtered_context``) to enable the real signal; None falls back
           to the constant baseline for callers that haven't been updated.

        ``entities`` adds token-level cross-reference matching: if a word
        (≥4 chars) from an entity name appears in both this chunk and another
        chunk, that counts as a cross-reference even when the full entity name
        is absent as a substring.  This handles surface-form mismatches such
        as 'Terrence "Uncle Terry" Richardson' which does not contain the
        substring 'Terry Richardson' but does contain 'Terry' and 'Richardson'.
        """
        # Pre-compute entity name tokens for token-level cross-reference.
        entity_tokens_lower: List[str] = []
        if entities:
            seen_tokens: set = set()
            for name in entities:
                for tok in name.split():
                    tok_l = tok.lower()
                    if len(tok_l) >= 4 and tok_l not in seen_tokens:
                        entity_tokens_lower.append(tok_l)
                        seen_tokens.add(tok_l)

        # B2-fix: align the provenance flag with filtered_context. None means
        # the caller didn't supply provenance, so we keep the historic baseline
        # behavior (every chunk is treated as non-graph).
        if chunk_is_graph_based is not None and len(chunk_is_graph_based) != len(filtered_context):
            logger.warning(
                "chunk_is_graph_based length mismatch (%d vs %d filtered chunks); "
                "ignoring provenance signal.",
                len(chunk_is_graph_based), len(filtered_context),
            )
            chunk_is_graph_based = None

        scores: List[float] = []
        for chunk_idx, chunk in enumerate(filtered_context):
            cred = SourceCredibility(text=chunk)
            key_phrases = self._extract_key_phrases(chunk)
            chunk_lower = chunk.lower()

            # Cross-reference: any other chunk that shares a key phrase OR
            # shares an entity name token with this chunk.
            for other in original_context:
                if other != chunk:
                    other_lower = other.lower()
                    phrase_matched = any(
                        phrase.lower() in other_lower for phrase in key_phrases
                    )
                    if phrase_matched:
                        cred.cross_references += 1
                    elif entity_tokens_lower:
                        # Token-level fallback: entity word present in both chunks.
                        chunk_has = any(t in chunk_lower for t in entity_tokens_lower)
                        other_has = any(t in other_lower for t in entity_tokens_lower)
                        if chunk_has and other_has:
                            cred.cross_references += 1

            # Self-relevance boost: if this chunk itself mentions a query entity
            # it is directly on-topic regardless of what other chunks say.
            # Without this, a bridge-target chunk (e.g. the Strasbourg article)
            # always scores cross_references=0 because no other Hop-2 noise chunk
            # mentions Strasbourg — and then falls below min_credibility_score.
            if entity_tokens_lower and cred.cross_references == 0:
                if any(t in chunk_lower for t in entity_tokens_lower):
                    cred.cross_references = 1

            # Entity-frequency signal.
            if SPACY_AVAILABLE and NLP:
                doc = NLP(chunk[: self.config.spacy_max_chars])
                cred.entity_frequency = min(1.0, len(doc.ents) / self.config.credibility_entity_freq_normalizer_spacy)
            else:
                proper_count = len(self._PROPER_NOUN_PATTERN.findall(chunk))
                cred.entity_frequency = min(1.0, proper_count / self.config.credibility_entity_freq_normalizer_regex)

            # B2-fix: use real retrieval-provenance when supplied. Falls back
            # to the historical False/baseline path for callers (older tests,
            # AgenticController direct path) that don't pass provenance.
            if chunk_is_graph_based is not None:
                cred.is_graph_based = bool(chunk_is_graph_based[chunk_idx])
            else:
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
- Give the shortest possible answer: a name, place, date, number, or yes/no.
- Do NOT explain or add sentences beyond the direct answer.
- If the answer is a person, place, or thing: reply with just that name.
- If the answer is a number or statistic (e.g. population, count, year): reply with just the number.
- If the answer is yes/no: reply with just "yes" or "no".
- If the context does not contain the answer: reply with "I don't know."

Context:
{context}

Question: {query}

Answer (as short as possible):"""

    BRIDGE_PROMPT = """You are a factual QA assistant. Answer based ONLY on the context below.

This is a multi-step question. Use the following reasoning chain to find the answer:
{bridge_chain}

Rules:
- Give the shortest possible answer: a name, place, date, number, or yes/no.
- Do NOT explain or add sentences beyond the direct answer.
- If the answer is a number or statistic (e.g. population, count, year): reply with just the number.
- If the context does not contain the answer: reply with "I don't know."

Context:
{context}

Question: {query}

Answer (as short as possible):"""

    COMPARISON_PROMPT = """You are a factual QA assistant. Answer based ONLY on the context below.

The question compares two people or things. Follow these steps:
1. Find the relevant fact for the FIRST person/thing in the context.
2. Find the relevant fact for the SECOND person/thing in the context.
3. Compare the two facts and give the answer.

Rules:
- For yes/no questions: reply with just "yes" or "no".
- For "which one" questions: reply with just the name.
- Do NOT explain beyond the direct answer.
- If the context does not contain enough information: reply with "I don't know."

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

    # B4-fix: stopwords excluded from the token-grounding check for short /
    # numeric claims with no proper noun. Articles, auxiliaries, and copulas
    # add no factual content, so demanding their literal presence in the
    # context would over-violate paraphrases. Lowercase — matched against
    # tokenized claim text.
    _CLAIM_VERIFY_STOPWORDS: frozenset = frozenset({
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with",
        "is", "was", "are", "were", "be", "been", "being", "has", "have", "had",
        "do", "does", "did", "and", "or", "but", "if", "so",
        "it", "its", "this", "that", "these", "those",
    })

    # B10-fix: per-string truncation budget for iteration_history entries.
    # Across 500 questions × 2 iterations × ~400-char answers + claim lists,
    # the raw history balloons the per-question JSONL by several MB and
    # slows down jq/pandas analysis. 200 chars per string is enough to
    # diagnose failures (LLM error sentinels, hallucination first line) while
    # keeping the file flat-text grep-friendly.
    _HISTORY_STR_TRUNCATE_CHARS: int = 200

    @classmethod
    def _truncate_history_str(cls, s: str) -> str:
        """Truncate a string for storage in iteration_history (B10)."""
        if not isinstance(s, str):
            return s
        if len(s) <= cls._HISTORY_STR_TRUNCATE_CHARS:
            return s
        return s[: cls._HISTORY_STR_TRUNCATE_CHARS] + "...[truncated]"

    @classmethod
    def _truncate_history_list(cls, items: List[str]) -> List[str]:
        """Truncate every string in a list for iteration_history storage (B10)."""
        return [cls._truncate_history_str(x) for x in items]

    # Epistemic-disclaimer phrases that signal the LLM did NOT answer.
    # When an answer matches any of these, it must not be reported as
    # HIGH confidence with all_verified=True (Bug 4).
    _DISCLAIMER_PATTERNS: Tuple[str, ...] = (
        "i don't know",
        "i do not know",
        "i cannot determine",
        "i can't determine",
        "i cannot find",
        "i can't find",
        "i was unable to find",
        "unable to find",
        "no information",
        "no specific information",
        "not provided in the context",
        "not provided",
        "not mentioned",
        "not specified",
        "not stated",
        "not available",
        "is not in the context",
        "the context does not",
        "the context doesn't",
        "context does not contain",
        "context doesn't contain",
        "insufficient evidence",
        "insufficient information",
        "based on the available",
        "no answer can be",
        "cannot be determined",
        "however, there is no",
    )

    @classmethod
    def _is_disclaimer_answer(cls, answer: str) -> bool:
        """Return True if the answer is an epistemic disclaimer (Bug 4)."""
        if not answer or answer.startswith("[Error:"):
            return True
        a = answer.lower()
        return any(p in a for p in cls._DISCLAIMER_PATTERNS)

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

    # ── Context Relevance Reordering ──────────────────────────────────────────

    # Content words to exclude from question-keyword scoring.
    # These appear in almost every question and carry no discriminative signal.
    _QR_STOPWORDS: frozenset = frozenset({
        "what", "who", "where", "when", "which", "whom", "whose", "that",
        "this", "these", "those", "have", "been", "from", "with", "their",
        "they", "were", "there", "about", "also", "into", "more", "some",
        "does", "will", "would", "could", "should", "than", "then", "them",
    })

    def _reorder_by_question_relevance(
        self,
        query: str,
        context: List[str],
        entities: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Stable-sort context chunks so those sharing more content words with the
        query appear first in the LLM prompt.

        Rationale (§12.27): small LLMs tend to latch onto the first plausible
        entity in the context (positional bias; Liu et al. 2023, "Lost in the
        Middle", arXiv:2307.03172). By positioning the most answer-relevant
        chunk first, the LLM is exposed to the fact it needs before encountering
        distractors.

        Contract (revised): this method reorders ONLY the chunks already
        selected for the prompt — it operates on PRESENTATION ORDER, never on
        SET MEMBERSHIP. The caller caps the context to ``max_docs`` by the
        Navigator's fused RRF ranking BEFORE calling this method, so the
        retrieval signal (Cormack et al. 2009) decides which chunks survive the
        cap and this lexical-overlap heuristic cannot evict a high-RRF answer
        chunk that happens to be sparse in question terms (regression: idx143).

        Scoring combines three terms:
        1. IDF-weighted query-term overlap (F1a) — a query term occurring in
           MANY candidate chunks (a generic category word like "magazines")
           carries little discriminative power; a rare term (the specific
           entity) is decisive. Classic inverse document frequency (Spärck
           Jones 1972; Robertson 2004), computed over the candidate set in
           hand. Applied only when there are >= _IDF_MIN_CANDIDATES chunks —
           below that, document frequency is degenerate, so the score falls
           back to the validated length-normalised hit count (Fix E).
        2. sqrt(word_count) length normalisation (Fix E) — short direct-answer
           chunks are not penalised against long topic chunks that accumulate
           hits from sheer length.
        3. Structural-coverage floor (D1) — a chunk that names a DISTINCTIVE
           query entity (multi-word, or a single token >= 8 chars) receives a
           score floor so a required entity's article (e.g. a comparison
           conjunct, or a bridge target) cannot be demoted below the cap by
           keyword sparsity. Restricted to distinctive entities so a common
           single-word name does not over-fire.

        Stable-sort descending — ties preserve original (Navigator RRF) order.
        """
        if len(context) <= 1:
            return context

        query_tokens = {
            t for t in re.findall(r"\b\w{4,}\b", query.lower())
            if t not in self._QR_STOPWORDS
        }
        if not query_tokens:
            return context

        # F1a: IDF over the candidate set, guarded by a minimum count below
        # which document frequency is statistically meaningless.
        _IDF_MIN_CANDIDATES = 4
        if len(context) >= _IDF_MIN_CANDIDATES:
            n_docs = len(context)
            lowered_all = [c.lower() for c in context]
            idf = {
                t: math.log((n_docs + 1.0) / (1.0 + sum(1 for c in lowered_all if t in c)))
                for t in query_tokens
            }
        else:
            idf = {t: 1.0 for t in query_tokens}

        # D1: distinctive query entities that earn a coverage floor.
        distinctive_entities = [
            e.lower() for e in (entities or [])
            if len(e.split()) >= 2 or len(e) >= 8
        ]
        # The floor is the maximum achievable IDF mass, so an entity-bearing
        # chunk always outranks a chunk that merely shares generic terms.
        coverage_floor = sum(idf.values()) if idf else 1.0

        def _score(chunk: str) -> float:
            chunk_lower = chunk.lower()
            weighted = sum(idf[t] for t in query_tokens if t in chunk_lower)
            word_count = max(1, len(chunk_lower.split()))
            base = weighted / (word_count ** 0.5)
            if distinctive_entities and any(e in chunk_lower for e in distinctive_entities):
                base += coverage_floor
            return base

        return sorted(context, key=_score, reverse=True)

    # ── Context Formatting ────────────────────────────────────────────────────

    @staticmethod
    def _truncate_sentence_aware(doc: str, budget: int, query: str) -> str:
        """F2: truncate a doc to `budget` chars by keeping the most
        query-relevant SENTENCES (in original order), not the first N chars.

        Head-truncation silently drops an answer-bearing sentence that sits in
        the tail of a chunk (confirmed failure: a chunk whose defining fact was
        in its last sentence). Selecting by query overlap and re-emitting the
        kept sentences in their ORIGINAL order preserves local coherence (which
        matters for a small LLM) while ensuring the answer sentence survives.
        """
        sentences = re.split(r"(?<=[.!?])\s+", doc)
        if len(sentences) <= 1:
            # No sentence structure to exploit — fall back to head truncation.
            cut = doc[:budget]
            sp = cut.rfind(" ")
            return (cut[:sp] + "...") if sp > 0 else cut
        q_tokens = {t for t in re.findall(r"\b\w{4,}\b", query.lower())}

        def _rel(s: str) -> int:
            sl = s.lower()
            return sum(1 for t in q_tokens if t in sl)

        # Rank sentence INDICES by relevance, take the highest-scoring ones
        # that fit the budget, then emit them in original order.
        order = sorted(range(len(sentences)), key=lambda i: -_rel(sentences[i]))
        keep: set = set()
        used = 0
        for idx in order:
            s_len = len(sentences[idx]) + 1
            if used + s_len > budget and keep:
                break
            keep.add(idx)
            used += s_len
        kept = [sentences[i] for i in range(len(sentences)) if i in keep]
        return " ".join(kept).strip()

    def _format_context(self, context: List[str], query: str = "") -> str:
        """
        Format context chunks into a single prompt string with size limits.

        Strategy:
        1. Take at most ``max_docs`` chunks.
        2. Truncate each chunk at ``max_chars_per_doc`` (F2: sentence-aware,
           keeping the most query-relevant sentences in original order when a
           query is supplied; head-truncation otherwise).
        3. Stop adding chunks once ``max_context_chars`` is reached.
        """
        if not context:
            return "No context available."
        formatted_parts: List[str] = []
        total_chars = 0
        for i, doc in enumerate(context[: self.config.max_docs]):
            if len(doc) > self.config.max_chars_per_doc:
                if query:
                    truncated = self._truncate_sentence_aware(
                        doc, self.config.max_chars_per_doc, query,
                    )
                else:
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
            # B4-fix: claims with no extractable proper nouns previously
            # auto-verified, which inflated all_verified=True for short factual
            # answers like "1995", "9 million inhabitants", "ice hockey" — the
            # exact answer shape HotpotQA produces for "how many" / "in what
            # year" / "what sport" questions. We now check the claim against
            # the retrieved context whenever it contains a numeric token OR is
            # very short (<= 6 tokens). If the claim isn't grounded in context,
            # treat it as a violation rather than auto-verified. Multi-clause
            # narrative sentences with no proper noun still auto-verify (the
            # historical behavior) because no falsifiable anchor exists.
            claim_lower = claim.lower().strip()
            tokens = claim_lower.split()
            has_number = bool(re.search(r"\d", claim_lower))
            is_short = len(tokens) <= 6

            if (has_number or is_short) and context:
                # Token-level grounding check: every non-stopword content token
                # of the claim must appear somewhere in the joined context.
                # Stopwords/articles/auxiliary verbs are ignored so phrasing
                # differences ("was founded in 1995" vs "founded 1995") don't
                # falsely violate.
                #
                # B4-fix: numeric tokens are KEPT regardless of length because
                # they are the falsifiable signal the check exists to catch
                # (e.g. "9 million inhabitants" vs "1.5 million inhabitants" —
                # without keeping the digit, "million" and "inhabitants" both
                # match and the hallucination would slip through). Strip
                # trailing punctuation so "1995." still matches "1995" in
                # the context.
                context_text = " ".join(context).lower()
                content_tokens = []
                for raw in tokens:
                    t = raw.strip(".,;:!?\"'()[]")
                    if not t:
                        continue
                    if t in self._CLAIM_VERIFY_STOPWORDS:
                        continue
                    is_numeric_token = any(ch.isdigit() for ch in t)
                    if is_numeric_token or len(t) >= 2:
                        content_tokens.append(t)
                if not content_tokens:
                    return True, "no_content_tokens_to_verify"
                grounded = all(t in context_text for t in content_tokens)
                if grounded:
                    return True, "context_token_grounded"
                return False, "no_entities_and_tokens_ungrounded"

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

    # ── Bridge Chain Builder ──────────────────────────────────────────────────

    @staticmethod
    def _build_bridge_chain(
        query: str,
        entities: List[str],
        bridge_entities: List[str],
        hop_sequence: Optional[List[Dict[str, Any]]],
        context: Optional[List[str]] = None,
    ) -> str:
        """
        Build a human-readable reasoning scaffold for multi-hop prompts.

        Safety guarantees (§12.32):
        1. Never emits a literal sentinel like "THIS IS THE ANSWER" — small
           quantized models echo such strings verbatim. The final step uses
           a directive verb ("→ derive the final answer") instead.
        2. Only injects a bridge entity into a hop's substitution if the
           entity is actually present in the retrieved context. Prevents
           propagation of spurious/distractor bridge entities into the
           reasoning chain (e.g. "New York" appearing because the gold
           chunk had a tangential clause about Robert Durst).
        3. If no bridge entity passes the grounding check, the hop is
           rendered as a directive ("→ identify the intermediate result")
           rather than left blank or pre-filled with a wrong value.
        """
        lines: List[str] = []
        # Lowercased context for cheap substring grounding-check.
        context_blob = " ".join(context or []).lower()

        def _grounded(entity: str) -> bool:
            """Entity is grounded if it appears in the retrieved context.

            Empty context → trust the upstream extractor (cannot ground-
            check, but emitting nothing is worse than emitting unverified)."""
            if not context_blob:
                return True
            return entity.lower() in context_blob

        if hop_sequence:
            hops = sorted(hop_sequence, key=lambda h: h.get("step_id", 0))
            last_step_id = max(h.get("step_id", 0) for h in hops)
            # Filter bridge entities to those actually grounded in context.
            # Preserves order; first grounded entity goes to first bridge hop.
            grounded_bridges = [e for e in bridge_entities if _grounded(e)]
            bridge_idx = 0
            for hop in hops:
                step = hop.get("step_id", 0) + 1
                sub_q = hop.get("sub_query", "")
                is_bridge = hop.get("is_bridge", False)
                is_last = hop.get("step_id", 0) == last_step_id
                if is_last:
                    # Directive verb form — NOT a literal placeholder string
                    # the LLM could mistake for an answer.
                    lines.append(
                        "Step %d: %s → derive the final answer" % (step, sub_q)
                    )
                else:
                    # Bridge entity injection disabled (2026-05-20).
                    # The grounding check (entity appears in context) is
                    # insufficient — appearing in context does not mean the
                    # entity answers the sub-query. Observed failure modes
                    # in 20-sample diagnostic:
                    #   - injected current name where the question asked for
                    #     the former name (SLM copied the wrong value),
                    #   - injected a random PERSON pulled from a noisy
                    #     sub-query (SLM picked a different label),
                    #   - injected a useless synonym (SLM abstained despite
                    #     both gold paragraphs being present in context).
                    # The directive form was already deployed in the prior
                    # ungrounded-fallback branch; we unify both bridge paths
                    # onto it. `grounded_bridges` and `bridge_idx` remain in
                    # scope for any caller-side telemetry.
                    lines.append(
                        "Step %d: %s → identify the intermediate result"
                        % (step, sub_q)
                    )
        else:
            # Fallback: generic chain from entity names + (grounded) bridges.
            grounded_bridges = [e for e in bridge_entities if _grounded(e)]
            anchor = entities[0] if entities else "the subject"
            for i, bridge in enumerate(grounded_bridges, start=1):
                lines.append("Step %d: find information about %s → %s" % (i, anchor, bridge))
                anchor = bridge
            lines.append(
                "Step %d: derive the final answer about %s"
                % (len(grounded_bridges) + 1, anchor)
            )

        return "\n".join(lines)

    # ── Main Verification Loop ────────────────────────────────────────────────

    def generate_and_verify(
        self,
        query: str,
        context: List[str],
        entities: Optional[List[str]] = None,
        hop_sequence: Optional[List[Dict[str, Any]]] = None,
        query_type: Optional[str] = None,
        bridge_entities: Optional[List[str]] = None,
        chunk_is_graph_based: Optional[List[bool]] = None,
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

        # B1-followup (2026-05-15): the signature documents hop_sequence as
        # List[Dict[str, Any]], but AgentPipeline.process() passes the
        # Planner's List[HopStep] dataclasses directly. The bridge-chain
        # builder uses .get() on each entry, which works on dicts but not on
        # dataclasses. Normalise at the boundary so both call paths work:
        # dataclass → dict via dataclasses.asdict(); dicts pass through.
        if hop_sequence:
            try:
                from dataclasses import asdict, is_dataclass
                hop_sequence = [
                    asdict(h) if is_dataclass(h) else h
                    for h in hop_sequence
                ]
            except Exception as exc:  # defensive — never break the eval over this
                logger.debug(
                    "hop_sequence normalisation skipped (%s); leaving as-is.",
                    exc,
                )

        # ── Pre-generation validation ─────────────────────────────────────────
        pre_validation = self.pre_validator.validate(
            context=context,
            query=query,
            entities=entities,
            hop_sequence=hop_sequence,
            chunk_is_graph_based=chunk_is_graph_based,
        )
        working_context = pre_validation.filtered_context
        logger.info(
            "[Verifier] Pre-validation: %s, context %d/%d",
            pre_validation.status.value,
            len(working_context),
            len(context),
        )

        # ── Hard early-return on truly empty evidence (Bug 3) ────────────────
        # Skip the ~18 s LLM call when pre-validation produced no usable
        # context: an empty filtered_context, or INSUFFICIENT_EVIDENCE
        # combined with zero entities found by the entity-path check.
        # We still hand non-empty INSUFFICIENT_EVIDENCE contexts to the LLM
        # via INSUFFICIENT_EVIDENCE_PROMPT — the LLM may extract a partial
        # answer from the surviving chunks.
        path_details = pre_validation.details.get("entity_path", {}) or {}
        entities_found = path_details.get("entities_found", []) or []
        if not working_context or (
            pre_validation.status == ValidationStatus.INSUFFICIENT_EVIDENCE
            and not entities_found
        ):
            logger.warning(
                "[Verifier] Hard early-return: no usable context "
                "(working_context=%d, entities_found=%d) — skipping LLM call.",
                len(working_context),
                len(entities_found),
            )
            total_time = (time.time() - start_time) * 1000
            return VerificationResult(
                answer="I cannot determine the answer from the provided context.",
                iterations=0,
                verified_claims=[],
                violated_claims=[],
                all_verified=False,
                pre_validation=pre_validation,
                timing_ms=total_time,
                iteration_history=[],
                confidence_high_threshold=self.config.confidence_high_threshold,
                confidence_medium_threshold=self.config.confidence_medium_threshold,
            )

        # §12.27 contract (revised): the Navigator's fused RRF ranking owns SET
        # MEMBERSHIP — which chunks survive the max_docs cap — while the
        # question-relevance reorder owns only PRESENTATION ORDER within that
        # window, to mitigate LLM positional bias (Liu et al. 2023, "Lost in the
        # Middle", arXiv:2307.03172). Selecting by lexical query-overlap let a
        # distractor that merely echoes the question evict a question-term-sparse
        # answer chunk that the retriever had ranked #1 (observed idx143). Cap by
        # RRF order first, then reorder only inside the kept window.
        selected = working_context[: self.config.max_docs]
        working_context = self._reorder_by_question_relevance(
            query, selected, entities=entities,
        )
        formatted_context = self._format_context(working_context, query=query)

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
                elif query_type == "comparison":
                    prompt = self.COMPARISON_PROMPT.format(
                        context=formatted_context, query=query
                    )
                elif query_type in ("multi_hop", "bridge") and hop_sequence:
                    bridge_chain = self._build_bridge_chain(
                        query, entities or [], bridge_entities or [],
                        hop_sequence, context=context,
                    )
                    prompt = self.BRIDGE_PROMPT.format(
                        bridge_chain=bridge_chain,
                        context=formatted_context,
                        query=query,
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
                    "answer": self._truncate_history_str(answer),
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
                    "answer": self._truncate_history_str(answer),
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

            # ── Disclaimer override (Bug 4) ──────────────────────────────
            # If the answer is an epistemic disclaimer, _extract_claims has
            # already stripped the meta-statement and the claim list is
            # often empty — which silently maps to all_verified=True / HIGH
            # confidence via the (0,0) ratio in VerificationResult.confidence.
            # Force a single violated claim so downstream sees this as a
            # non-answer with LOW confidence.
            if self._is_disclaimer_answer(answer):
                logger.warning(
                    "[Verifier] Disclaimer answer detected — forcing LOW confidence."
                )
                violated_claims = [answer.strip()[:200]]
                verified_claims = []
            logger.info(
                "[Verifier] Verification: %d verified, %d violated",
                len(verified_claims),
                len(violated_claims),
            )

            iter_time = (time.time() - iter_start) * 1000
            # B10-fix: truncate stored strings (answer + claim/verified/violated
            # lists) to 200 chars each. Keeps the per-question JSONL flat-text
            # grep-friendly across 500-question runs.
            iteration_history.append({
                "iteration": iteration,
                "answer": self._truncate_history_str(answer),
                "claims": self._truncate_history_list(claims),
                "verified": self._truncate_history_list(verified_claims),
                "violated": self._truncate_history_list(violated_claims),
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
        query_type: Optional[str] = None,
        bridge_entities: Optional[List[str]] = None,
        chunk_is_graph_based: Optional[List[bool]] = None,
    ) -> VerificationResult:
        """Callable interface — forwards all arguments to generate_and_verify."""
        return self.generate_and_verify(
            query=query,
            context=context,
            entities=entities,
            hop_sequence=hop_sequence,
            query_type=query_type,
            bridge_entities=bridge_entities,
            chunk_is_graph_based=chunk_is_graph_based,
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
