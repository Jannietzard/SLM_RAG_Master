"""
Agent Pipeline: Orchestration of S_P → S_N → S_V

Version: 4.1.0
Author: Edge-RAG Research Project
Last Modified: 2026-04-08

================================================================================
ARCHITECTURAL POSITION
================================================================================

This module is the top-level entry point of Artifact B (Logic Layer) and the
primary interface consumed by the Evaluation Layer (benchmark_datasets.py).
It sits one level above the three reasoning agents:

    benchmark_datasets.py
           │
           ▼  create_full_pipeline()
    AgentPipeline.process(query)
           │
    ┌──────┴────────────────────────────┐
    │  S_P  Planner  (planner.py)       │  Query decomposition, strategy
    │  S_N  Navigator (navigator.py)    │  Hybrid retrieval + RRF fusion
    │  S_V  Verifier  (verifier.py)     │  Self-correcting answer generation
    └───────────────────────────────────┘

================================================================================
SCIENTIFIC CONTRIBUTION
================================================================================

The central thesis contribution of Artifact B is the self-correction loop in
S_V: the Verifier generates an initial answer, checks it against the retrieved
context via entity-presence verification (named entities in the answer must
appear in at least one context chunk), and iteratively applies a
CORRECTION_PROMPT with concrete violation feedback for up to
max_verification_iterations rounds. An NLI-based contradiction detector
(`enable_contradiction_detection`) is available as an ablation flag but is
OFF by default — it is not active in the thesis evaluation.

This design follows the Self-Refine paradigm:
    Madaan, A. et al. (2023). "Self-Refine: Iterative Refinement with
    Self-Feedback." NeurIPS 2023.

Crucially, the self-correction loop is implemented EXCLUSIVELY inside
Verifier.generate_and_verify(). AgentPipeline.process() calls this method
exactly once. There is no outer retry loop — a previous outer retry was removed
because a fully deterministic LLM (temperature=0.0) produces identical outputs
for identical inputs, so outer repetition provides no correction benefit.
See TECHNICAL_ARCHITECTURE.md §12.2 for the full rationale.

================================================================================
ABLATION FLAGS
================================================================================

The pipeline supports component ablation via config/settings.yaml or at
construction time:

    agent.enable_planner: false     → --no-planner: passthrough RetrievalPlan
    agent.enable_verifier: false    → --no-verifier: return top retrieved chunk
    agent.max_verification_iterations: 1  → no self-correction (baseline)
    agent.max_verification_iterations: 2  → thesis default (1 correction round)

================================================================================
USAGE
================================================================================

    from src.pipeline import create_full_pipeline

    pipeline = create_full_pipeline(
        hybrid_retriever=retriever,
        graph_store=store.graph_store,
        config=yaml_config,
    )
    result = pipeline.process("What is the capital of France?")
    print(result.answer, result.confidence)

================================================================================
"""

import dataclasses
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..logic_layer.planner import Planner, PlannerConfig, RetrievalPlan
    from ..logic_layer.navigator import ControllerConfig, Navigator
    from ..logic_layer.verifier import Verifier, VerifierConfig

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION BRIDGE HELPERS
# ============================================================================

def _verifier_config_from_cfg(config: Dict[str, Any]) -> "VerifierConfig":
    """
    Build VerifierConfig from a settings.yaml dict.

    Delegates to ``VerifierConfig.from_yaml(config)``. Emits a warning when
    no ``llm`` block is present so callers detect missing configuration early.
    All defaults are emergency fallbacks that match settings.yaml thesis values.
    """
    from ..logic_layer.verifier import VerifierConfig
    if not config.get("llm"):
        logger.warning(
            "FALLBACK ACTIVE: No llm config provided to _verifier_config_from_cfg. "
            "Verifier context budget (max_context_chars/max_docs/max_chars_per_doc) "
            "uses hardcoded defaults. Pass settings.yaml content for reproducible results."
        )
    return VerifierConfig.from_yaml(config)


def _navigator_config_from_cfg(config: Dict[str, Any]) -> "ControllerConfig":
    """
    Build ControllerConfig (Navigator) from a settings.yaml dict.

    Delegates to ``ControllerConfig.from_yaml(config)``.
    """
    from ..logic_layer.navigator import ControllerConfig
    return ControllerConfig.from_yaml(config)


def _planner_config_from_cfg(config: Dict[str, Any]) -> "PlannerConfig":
    """
    Build PlannerConfig from a settings.yaml dict.

    Delegates to ``PlannerConfig.from_yaml(config)``.
    """
    from ..logic_layer.planner import PlannerConfig
    return PlannerConfig.from_yaml(config)


def _create_passthrough_plan(query: str) -> "RetrievalPlan":
    """
    Create a minimal RetrievalPlan for --no-planner ablation mode.

    Bypasses S_P entirely: forces HYBRID strategy with full confidence so that
    S_N receives a valid plan without any LLM call.

    ``QueryType.MULTI_HOP`` is chosen as the conservative default because it
    triggers the full hybrid retrieval path in Navigator. A SINGLE_HOP default
    would skip multi-source fusion and risk missing supporting documents for
    complex questions.
    """
    from ..logic_layer.planner import RetrievalPlan, QueryType, RetrievalStrategy
    return RetrievalPlan(
        original_query=query,
        query_type=QueryType.MULTI_HOP,
        strategy=RetrievalStrategy.HYBRID,
        confidence=1.0,
    )


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

@dataclass
class AgentPipelineConfig:
    """
    Pipeline-level configuration sourced from settings.yaml → agent.*.

    Analogous to IngestionConfig for the ingestion side. All parameters have
    dataclass defaults that match the thesis evaluation settings; in production
    always construct via from_yaml() so settings.yaml is the single source of
    truth.

    Attributes:
        enable_planner: When False, S_P is skipped and a passthrough
            RetrievalPlan (HYBRID, MULTI_HOP) is used instead.
        enable_verifier: When False, S_V is skipped and the top retrieved
            chunk is returned directly as the answer.
        enable_caching: Cache query results keyed by SHA-256 of the
            normalised query string (FIFO eviction).
        cache_max_size: Maximum number of cached entries before the oldest
            is evicted.
        enable_confidence_gate: Architecture A (2026-05-27). When True, the
            pipeline runs a cheap single-pass baseline retrieval first and,
            if retrieval confidence is HIGH, answers directly from the
            baseline -- skipping the Planner's iterative decomposition. Only
            escalates to the full agentic retrieval path when confidence is
            not HIGH. Default False (opt-in) so existing behaviour is
            unchanged. Refs: Geifman & El-Yaniv 2019 (selective prediction,
            NeurIPS); Asai et al. 2024 (Self-RAG adaptive retrieval, ICLR).
        confidence_score_gap_threshold: top1/top2 RRF-score ratio above which
            the top chunk is considered "dominant" (Signal A).
        confidence_require_signals: number of the 3 confidence signals
            (score-gap / multi-source / entity-coverage) that must fire for
            a HIGH verdict.
    """
    enable_planner: bool = True
    enable_verifier: bool = True
    enable_caching: bool = True
    cache_max_size: int = 1000
    # Architecture A — confidence gate (opt-in)
    enable_confidence_gate: bool = False
    confidence_score_gap_threshold: float = 1.5
    confidence_require_signals: int = 2

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "AgentPipelineConfig":
        """
        Construct AgentPipelineConfig from a settings.yaml dict.

        Args:
            config: Full settings.yaml dict (or the relevant sub-dict). Missing
                keys fall back to dataclass defaults.

        Returns:
            AgentPipelineConfig populated from config[\"agent\"].
        """
        agent = config.get("agent", {})
        return cls(
            enable_planner=agent.get("enable_planner", True),
            enable_verifier=agent.get("enable_verifier", True),
            enable_caching=agent.get("enable_caching", True),
            cache_max_size=agent.get("cache_max_size", 1000),
            enable_confidence_gate=agent.get("enable_confidence_gate", False),
            confidence_score_gap_threshold=agent.get(
                "confidence_score_gap_threshold", 1.5
            ),
            confidence_require_signals=agent.get("confidence_require_signals", 2),
        )


# ============================================================================
# PIPELINE RESULT
# ============================================================================

@dataclass
class PipelineResult:
    """
    Unified result of the complete S_P → S_N → S_V agent pipeline.

    Contains the final answer, per-stage intermediate results, and
    per-stage timing information. Used by benchmark_datasets.py to compute
    EM and F1 metrics and to record retrieval coverage.
    """
    # Final output
    answer: str
    confidence: str

    # Input
    query: str

    # Per-stage results
    planner_result: Dict[str, Any]
    navigator_result: Dict[str, Any]
    verifier_result: Dict[str, Any]

    # Per-stage latency (milliseconds)
    planner_time_ms: float
    navigator_time_ms: float
    verifier_time_ms: float
    total_time_ms: float

    # Optimization flags
    cached_result: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a nested dict for JSON export and benchmark logging."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "query": self.query,
            "stages": {
                "planner": self.planner_result,
                "navigator": self.navigator_result,
                "verifier": self.verifier_result,
            },
            "timing": {
                "planner_ms": self.planner_time_ms,
                "navigator_ms": self.navigator_time_ms,
                "verifier_ms": self.verifier_time_ms,
                "total_ms": self.total_time_ms,
            },
            "optimization": {
                "cached": self.cached_result,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """
        JSON-serialise to_dict() output.

        Convenience method for interactive inspection and result export.
        Not used in the benchmark pipeline itself.
        """
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# AGENT PIPELINE
# ============================================================================

class AgentPipeline:
    """
    Orchestrator for the three-agent pipeline S_P → S_N → S_V.

    Responsibilities:
    - Chain Planner, Navigator, and Verifier in a fixed sequential order.
    - Bridge settings.yaml configuration to each agent's typed config object.
    - Provide ablation controls (enable_planner, enable_verifier).
    - Maintain a FIFO query-result cache for repeated evaluation queries.
    - Track per-stage timing and aggregate statistics.

    The self-correction loop is NOT managed here. It is implemented entirely
    inside Verifier.generate_and_verify() and controlled by
    VerifierConfig.max_iterations (sourced from settings.yaml:
    agent.max_verification_iterations). This pipeline calls
    generate_and_verify() exactly once per query.
    Reference: Madaan, A. et al. (2023). "Self-Refine: Iterative Refinement
    with Self-Feedback." NeurIPS 2023. DOI: 10.48550/arXiv.2303.17651

    Thread safety: Not thread-safe. This pipeline is designed for the
    single-threaded edge deployment described in the thesis. Do not share
    an AgentPipeline instance across threads without external locking.
    """

    def __init__(
        self,
        planner: Optional["Planner"] = None,
        navigator: Optional["Navigator"] = None,
        verifier: Optional["Verifier"] = None,
        hybrid_retriever: Optional[Any] = None,
        graph_store: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise the pipeline.

        Args:
            planner: S_P agent. Created lazily on first process() call if None.
            navigator: S_N agent. Created lazily if None.
            verifier: S_V agent. Created lazily if None.
            hybrid_retriever: Injected into Navigator for retrieval. If None,
                Navigator is created but will fail on first retrieval call.
            graph_store: Injected into Verifier for provenance lookup.
            config: Full settings.yaml dict. If None or empty, all agent configs
                use hardcoded fallback defaults — including max_iterations=2 (thesis
                self-correction default). A warning is emitted in this case.
        """
        self.config: Dict[str, Any] = config or {}
        if not self.config:
            logger.warning(
                "FALLBACK ACTIVE: No config provided to AgentPipeline. "
                "All agent configs use hardcoded defaults. "
                "Pass settings.yaml content for reproducible results."
            )

        # Resolve pipeline-level flags via AgentPipelineConfig so all reads
        # from settings.yaml → agent.* go through a single, typed path.
        _pipeline_cfg = AgentPipelineConfig.from_yaml(self.config)
        self.enable_planner: bool = _pipeline_cfg.enable_planner
        self.enable_verifier: bool = _pipeline_cfg.enable_verifier
        # Architecture A — confidence gate flags
        self.enable_confidence_gate: bool = _pipeline_cfg.enable_confidence_gate
        self._conf_score_gap_threshold: float = (
            _pipeline_cfg.confidence_score_gap_threshold
        )
        self._conf_require_signals: int = _pipeline_cfg.confidence_require_signals

        # Agent instances (injected or lazy-created on first process() call)
        self.planner: Optional["Planner"] = planner
        self.navigator: Optional["Navigator"] = navigator
        self.verifier: Optional["Verifier"] = verifier

        # Store dependencies for lazy init
        self.hybrid_retriever = hybrid_retriever
        self.graph_store = graph_store

        # FIFO query-result cache.
        # Python 3.7+ dicts preserve insertion order, so next(iter(cache))
        # reliably evicts the oldest entry. This is FIFO, not LRU — adequate
        # for the benchmark evaluation pattern where queries are processed once.
        self.enable_caching: bool = _pipeline_cfg.enable_caching
        self._cache: Dict[str, PipelineResult] = {}
        self._cache_max_size: int = _pipeline_cfg.cache_max_size

        # Online statistics — use Welford's incremental mean for avg_latency_ms
        # to avoid storing all latency values.
        # Reference: Welford, B.P. (1962). "Note on a method for calculating
        # corrected sums of squares and products." Technometrics, 4(3), 419-420.
        # DOI: 10.2307/1266577
        self._stats: Dict[str, Any] = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0.0,
        }

        logger.info(
            "AgentPipeline initialised: caching=%s, cache_max_size=%d, "
            "planner=%s, verifier=%s",
            self.enable_caching, self._cache_max_size,
            self.enable_planner, self.enable_verifier,
        )

    def _lazy_init_agents(self) -> None:
        """
        Lazily construct agent instances on first process() call.

        Deferred construction avoids importing SpaCy, GLiNER, and the Ollama
        client at module import time — critical for edge devices where startup
        latency and memory budget matter. Agents passed in at construction
        time are never replaced.
        """
        if self.planner is None and self.enable_planner:
            from ..logic_layer.planner import Planner
            self.planner = Planner(config=_planner_config_from_cfg(self.config))
            logger.info("Planner (S_P) lazy-initialised")

        if self.navigator is None:
            from ..logic_layer.navigator import Navigator
            nav_config = _navigator_config_from_cfg(self.config)
            self.navigator = Navigator(nav_config)
            if self.hybrid_retriever is not None:
                self.navigator.set_retriever(self.hybrid_retriever)
            else:
                logger.warning(
                    "FALLBACK ACTIVE: Navigator created without hybrid_retriever. "
                    "Retrieval calls will fail until set_retriever() is called."
                )
            logger.info("Navigator (S_N) lazy-initialised")

        if self.verifier is None and self.enable_verifier:
            from ..logic_layer.verifier import Verifier
            self.verifier = Verifier(
                config=_verifier_config_from_cfg(self.config),
                graph_store=self.graph_store,
            )
            logger.info("Verifier (S_V) lazy-initialised")

    def _compute_retrieval_confidence(
        self, nav_result: Any, query_entities: List[str]
    ) -> str:
        """
        Architecture A: classify single-pass retrieval confidence as
        'high' / 'medium' / 'low' from ranking features.

        Used by the confidence gate to decide whether to answer directly
        from the cheap baseline retrieval (HIGH) or escalate to the full
        agentic decomposition path (MEDIUM / LOW).

        Signals (all derived from the existing NavigatorResult -- no new
        compute):
          A. score_gap     scores[0] / scores[1] >= threshold -> a dominant
                           top chunk (clear winner vs a tied field).
          B. multi_source  the top chunk was produced by >=2 retrieval lanes
                           (vector + BM25 + graph), surfaced by the Navigator
                           as metadata["top_chunk_methods"][0] == "hybrid".
          C. entity_cov    all query entities appear in the top-3 chunks.

        A HIGH verdict requires >= self._conf_require_signals signals to fire
        (default 2 of 3). MEDIUM = exactly 1. LOW = 0.

        Refs: Geifman & El-Yaniv 2019 (selective prediction, NeurIPS);
              Asai et al. 2024 (Self-RAG adaptive retrieval, ICLR).
        """
        scores = getattr(nav_result, "scores", None) or []
        ctx = getattr(nav_result, "filtered_context", None) or []
        if len(scores) < 2 or len(ctx) < 1:
            # Too little signal to judge -> treat as not-confident.
            return "low"

        # Signal A: dominant top chunk.
        gap_ok = (scores[0] / max(scores[1], 1e-6)) >= self._conf_score_gap_threshold

        # Signal B: top chunk hit by >=2 retrieval lanes (vector/bm25/graph).
        # The Navigator surfaces per-chunk provenance as
        # metadata["chunk_retrieval_methods"] (list-of-lists aligned with
        # filtered_context; B2-fix). The top chunk is "multi-source" when its
        # lane list has >=2 entries.
        meta = getattr(nav_result, "metadata", {}) or {}
        methods_per_chunk = (
            meta.get("chunk_retrieval_methods", []) if isinstance(meta, dict) else []
        )
        multi_source = bool(methods_per_chunk) and len(methods_per_chunk[0]) >= 2

        # Signal C: every query entity present in the top-3 chunks.
        top3 = " ".join(ctx[:3]).lower()
        entity_cov = bool(query_entities) and all(
            (e or "").lower() in top3 for e in query_entities
        )

        n_strong = int(gap_ok) + int(multi_source) + int(entity_cov)
        if n_strong >= self._conf_require_signals:
            verdict = "high"
        elif n_strong >= 1:
            # Some signal fired but below the HIGH threshold. MEDIUM and LOW
            # both escalate to the agentic path today (only HIGH short-circuits
            # to baseline); the distinction is retained for diagnostics.
            verdict = "medium"
        else:
            verdict = "low"
        logger.debug(
            "[Gate] confidence=%s (gap=%s multi_source=%s entity_cov=%s "
            "n_strong=%d/%d)",
            verdict, gap_ok, multi_source, entity_cov,
            n_strong, self._conf_require_signals,
        )
        return verdict

    def _iterative_navigate(self, plan: Any, original_query: str) -> Any:
        """
        §12.37: iterative multi-hop navigation with bridge-entity propagation.

        Executes each hop sequentially:
          1. Run Navigator.navigate() for the hop's sub-query.
          2. Extract bridge entities from the retrieved chunks (via
             AgenticController._extract_bridge_entities — relevance-ranked,
             query-aware).
          3. Rewrite the NEXT hop's sub-query by appending the resolved
             bridges (via _rewrite_hop_query_with_bridges).
          4. Accumulate raw_context + filtered_context across all hops.

        Returns a merged NavigatorResult.

        Refs:
          - IRCoT (Trivedi et al., 2023, ACL, arXiv:2212.10509): iterative
            retrieval-with-reasoning, feeds back retrieved entities.
          - HippoRAG (Gutiérrez et al., 2024, NeurIPS): personalized-PageRank
            seeding from hop-1.
        """
        from ..logic_layer.controller import AgenticController
        from ..logic_layer.navigator import NavigatorResult

        # Sort hops by step_id so dependencies resolve in order. Fall back
        # to original list order for hops whose step_id is missing or
        # non-comparable (e.g. test doubles where step_id is a MagicMock
        # that can't be sorted against another MagicMock). Using
        # `(sort_key, list_index)` keeps the sort stable for real plans and
        # degrades to insertion order for malformed ones.
        def _hop_sort_key(item):
            idx, hop = item
            sid = getattr(hop, "step_id", None)
            if isinstance(sid, int):
                return (0, sid, idx)
            return (1, idx, idx)

        hops = [
            h for _, h in sorted(
                enumerate(plan.hop_sequence), key=_hop_sort_key,
            )
        ]

        # Initial entity hints come from the planner's NER pass.
        initial_entities = [e.text for e in (plan.entities or [])]
        current_hints: List[str] = list(initial_entities)
        resolved_bridges: List[str] = []

        accumulated_raw: List[str] = []
        accumulated_filtered: List[str] = []
        seen_raw: set = set()
        seen_filtered: set = set()
        merged_scores: List[float] = []
        # Fix 2a (2026-05-26): track which hop each chunk came from so the
        # post-loop cap can allocate a FAIR share per hop instead of a pure
        # score sort (which systematically evicted hop-1 chunks because hop-0
        # chunks score higher against the original query).
        chunk_hop_index: List[int] = []
        merged_metadata: Dict[str, Any] = {}
        last_result: Optional[NavigatorResult] = None

        for hop_idx, hop in enumerate(hops):
            sub_query = hop.sub_query

            # §12.37 Fix F: inject resolved bridge entities into this hop's
            # sub_query before retrieval. Only fires when the hop depends
            # on earlier bridges AND those bridges weren't already in the
            # planner-generated sub_query.
            if resolved_bridges and hop.depends_on:
                sub_query = AgenticController._rewrite_hop_query_with_bridges(
                    sub_query, resolved_bridges
                )

            try:
                nav_result = self.navigator.navigate(
                    retrieval_plan=plan,
                    sub_queries=[sub_query],
                    entity_names=current_hints,
                )
            except Exception as exc:
                logger.warning("[AgentPipeline iterative] hop %d failed: %s",
                               hop.step_id, exc)
                continue

            last_result = nav_result

            for chunk in nav_result.raw_context:
                if chunk not in seen_raw:
                    accumulated_raw.append(chunk)
                    seen_raw.add(chunk)
            for chunk, score in zip(nav_result.filtered_context,
                                     nav_result.scores or [0.0] * len(nav_result.filtered_context)):
                if chunk not in seen_filtered:
                    accumulated_filtered.append(chunk)
                    seen_filtered.add(chunk)
                    merged_scores.append(score)
                    chunk_hop_index.append(hop_idx)

            # After a bridge hop: extract bridges and remember them.
            if hop.is_bridge and nav_result.filtered_context:
                bridges = AgenticController._extract_bridge_entities(
                    nav_result.filtered_context,
                    exclude=current_hints,
                    query=original_query,
                )
                if bridges:
                    logger.info(
                        "[AgentPipeline iterative] hop %d bridges: %s",
                        hop.step_id, bridges,
                    )
                    current_hints = current_hints + bridges
                    resolved_bridges = resolved_bridges + bridges

        if last_result is None:
            return NavigatorResult(
                filtered_context=[], raw_context=[], scores=[], metadata={},
            )

        # Fix 2a (2026-05-26): per-hop FAIR cap replaces the previous score-
        # only cap. The score-only variant systematically evicted hop-1
        # chunks because hop-0 chunks naturally score higher against the
        # original query (the rewriter only injects bridge entities for
        # hop>=1, so hop-0's sub_query is closer to the surface query in
        # the cross-encoder reranker's input). Empirically that caused 4
        # of the 9 retrieval_miss cases in the 2026-05-26 50-q row3.
        #
        # Fair-cap allocates an equal base_quota per hop, then redistributes
        # the remainder by best score across all hops, then fills any
        # leftover budget from hops that had fewer chunks than their quota.
        # Refs: Radlinski 2008 source fairness; Diaz & Croft 2012 federated IR.
        total_cap = getattr(self.navigator.config, "max_context_chunks", 8)
        if len(accumulated_filtered) > total_cap and chunk_hop_index:
            from collections import defaultdict as _dd

            # Group (original_index, chunk, score) by hop_index.
            by_hop: Dict[int, List[Tuple[int, str, float]]] = _dd(list)
            for idx, (chunk, score, h) in enumerate(
                zip(accumulated_filtered, merged_scores, chunk_hop_index)
            ):
                by_hop[h].append((idx, chunk, score))

            num_hops = max(1, len(by_hop))
            base_quota = total_cap // num_hops
            remainder = total_cap - base_quota * num_hops

            kept_indices: set = set()
            leftovers: List[Tuple[int, str, float]] = []

            # Pass 1: take top base_quota chunks per hop (by score)
            for h in sorted(by_hop):
                ranked = sorted(by_hop[h], key=lambda t: -t[2])
                for i, (orig_idx, _c, _s) in enumerate(ranked):
                    if i < base_quota:
                        kept_indices.add(orig_idx)
                    else:
                        leftovers.append((orig_idx, _c, _s))

            # Pass 2: distribute remainder across hops by next-best score
            if remainder > 0 and leftovers:
                leftovers.sort(key=lambda t: -t[2])
                for orig_idx, _c, _s in leftovers[:remainder]:
                    kept_indices.add(orig_idx)
                leftovers = leftovers[remainder:]

            # Pass 3: if a hop had fewer chunks than its quota, the budget
            # is under total_cap; fill from highest-score remaining leftovers.
            while len(kept_indices) < total_cap and leftovers:
                leftovers.sort(key=lambda t: -t[2])
                orig_idx, _c, _s = leftovers.pop(0)
                kept_indices.add(orig_idx)

            # Restore original insertion order (Navigator RRF order within
            # each hop is preserved). The Verifier's §12.27 reorder then
            # operates on this stable input.
            kept_order = sorted(kept_indices)
            new_filtered = [accumulated_filtered[i] for i in kept_order]
            new_scores   = [merged_scores[i]        for i in kept_order]
            new_hops     = [chunk_hop_index[i]      for i in kept_order]
            per_hop_kept = {h: sum(1 for hh in new_hops if hh == h) for h in by_hop}
            logger.info(
                "[AgentPipeline iterative] fair-cap: %d -> %d chunks "
                "(per-hop quotas: %s)",
                len(accumulated_filtered), len(new_filtered), per_hop_kept,
            )
            accumulated_filtered = new_filtered
            merged_scores = new_scores
            chunk_hop_index = new_hops

        return NavigatorResult(
            filtered_context=accumulated_filtered,
            raw_context=accumulated_raw,
            scores=merged_scores,
            metadata={
                **(last_result.metadata or {}),
                "iterative_hops": len(hops),
                "resolved_bridges": resolved_bridges,
            },
        )

    def process(self, query: str) -> PipelineResult:
        """
        Process a single query through the full S_P → S_N → S_V pipeline.

        Inference-time errors (LLM unavailable, OOM, network timeout) are
        caught and returned as a structured PipelineResult with
        ``confidence="error"`` rather than propagating as unhandled exceptions.
        This protects callers that invoke process() directly (as opposed to
        going through BatchProcessor, which has its own per-query guard).

        Args:
            query: Natural language question (must be non-empty).

        Returns:
            PipelineResult with the final answer, per-stage metadata, and
            timing. On inference failure the answer is ``"Error: <message>"``
            and confidence is ``"error"``.

        Raises:
            ValueError: If query is None or empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        start_time = time.time()
        self._stats["total_queries"] += 1

        # ── Cache check ───────────────────────────────────────────────────────
        if self.enable_caching:
            cache_key = self._get_cache_key(query)
            if cache_key in self._cache:
                self._stats["cache_hits"] += 1
                # Use dataclasses.replace() to avoid mutating the stored object
                # in-place — callers that hold a reference to the original result
                # must not see its cached_result flag change retroactively.
                cached = dataclasses.replace(self._cache[cache_key], cached_result=True)
                logger.debug("Cache hit for query: %.50s", query)
                return cached

        # ── Lazy agent initialisation ─────────────────────────────────────────
        self._lazy_init_agents()

        # ── Inference stages (S_P → S_N → S_V) ───────────────────────────────
        # Wrapped in a single try/except so that infrastructure failures
        # (LLM unavailable, OOM, Ollama timeout) return a structured error
        # result instead of propagating an unhandled exception to the caller.
        # ValueError from query validation above is intentionally excluded —
        # invalid queries must fail loudly.
        planner_time: float = 0.0
        navigator_time: float = 0.0
        verifier_time: float = 0.0
        planner_result: Dict[str, Any] = {}
        navigator_result: Dict[str, Any] = {}
        verifier_result: Dict[str, Any] = {}
        answer: str = ""
        confidence_val: str = "low"

        try:
            # ── Stage 1: S_P (Planner) ────────────────────────────────────────
            planner_start = time.time()
            if self.enable_planner:
                plan = self.planner.plan(query)
                planner_result = plan.to_dict()
                logger.debug(
                    "S_P completed: type=%s strategy=%s (%.2fms)",
                    plan.query_type.value, plan.strategy.value,
                    (time.time() - planner_start) * 1000,
                )
            else:
                # Ablation: --no-planner mode. Forces HYBRID strategy without
                # any LLM call. Documented in TECHNICAL_ARCHITECTURE.md §5.1.
                plan = _create_passthrough_plan(query)
                from ..logic_layer.planner import QueryType, RetrievalStrategy
                planner_result = {
                    "planner_skipped": True,
                    "query_type": QueryType.MULTI_HOP.value,
                    "strategy": RetrievalStrategy.HYBRID.value,
                }
                logger.info("S_P skipped (enable_planner=False)")
            planner_time = (time.time() - planner_start) * 1000

            # Fix 2c + Tier 1.5 extension (2026-05-26): gate over-decomposing
            # patterns to single-pass when their decompositions are likely to
            # confuse the SLM more than they help retrieval. Three rules:
            #
            #   1. `fallback_generic_2hop` (10/50 in diagnostic): the catch-all
            #      when classifier said multi_hop but no pattern matched.
            #      Unconditionally gate -- the diagnostic confirms these are
            #      the dominant over-decomposition source.
            #
            #   2. `connector_split` with short halves (Tier 1.5): the
            #      connector-split pattern aggressively splits on "and"/"or"
            #      connectors. When BOTH halves are <=4 content words, the
            #      result is two near-trivial sub-queries the single-pass
            #      retriever handles just as well, without the cross-hop
            #      context dilution cost. Long halves still get the split.
            #
            #   3. (Future) `comparison_parallel` same-source: deferred --
            #      requires runtime check of entity source_file overlap.
            if self.enable_planner and plan.hop_sequence:
                _matched = getattr(plan, "matched_pattern", "") or ""

                _gate_reason: Optional[str] = None
                if _matched == "fallback_generic_2hop":
                    _gate_reason = "fallback_generic_2hop"
                elif _matched == "connector_split":
                    # Word-count check on each hop's sub_query
                    halves_short = all(
                        len((getattr(h, "sub_query", "") or "").split()) <= 4
                        for h in plan.hop_sequence
                    )
                    if halves_short:
                        _gate_reason = "connector_split_short_halves"

                if _gate_reason is not None:
                    logger.info(
                        "[Pipeline] over-decomposition gate fired (%s) for "
                        "query=%.60s -- routing to single-pass retrieval",
                        _gate_reason, query,
                    )
                    plan = _create_passthrough_plan(query)
                    planner_result = {
                        **(planner_result if isinstance(planner_result, dict) else {}),
                        "fallback_gate_triggered": True,
                        "original_matched_pattern": _matched,
                        "gate_reason": _gate_reason,
                    }

            # ── Stage 2: S_N (Navigator) ──────────────────────────────────────
            # Extract sub-queries from the planner's hop_sequence. For single-hop
            # queries or passthrough plans the hop_sequence is empty, so the
            # original query is used directly.
            sub_queries: List[str] = (
                [h.sub_query for h in plan.hop_sequence]
                if plan.hop_sequence
                else [query]
            )
            if self.enable_planner and not plan.hop_sequence:
                logger.debug(
                    "Planner returned empty hop_sequence — using original query "
                    "as single sub-query."
                )

            # §12.37: route through iterative multi-hop when the plan has
            # dependent steps. Previously AgentPipeline ran ALL sub-queries
            # in parallel via a single Navigator.navigate() call, which meant
            # the Hop-2 sub-query never received the bridge entity resolved in
            # Hop-1 — this is the dominant retrieval-failure mode for bridge
            # questions (29% gold-recall in the 20-question retrieval-only
            # baseline). The iterative path delegates to AgenticController so
            # the bridge-entity propagation logic (Fix F query rewriting) and
            # Navigator entity-hints filter both fire.
            #
            # Refs:
            #   - IRCoT iterative retrieval-with-reasoning: Trivedi et al.
            #     (2023), ACL, arXiv:2212.10509.
            #   - HippoRAG bridge-aware multi-hop: Gutiérrez et al. (2024),
            #     NeurIPS, arXiv:2405.14831.
            has_bridge_deps = (
                self.enable_planner
                and plan.hop_sequence
                and len(plan.hop_sequence) > 1
                and any(h.depends_on for h in plan.hop_sequence)
            )

            navigator_start = time.time()
            confidence_route = "agentic"   # default route, recorded in metadata

            # Architecture A (2026-05-27): confidence gate. Run a cheap
            # single-pass baseline retrieval first; if retrieval confidence is
            # HIGH, answer directly from it and skip the Planner's iterative
            # decomposition (which the ablation shows is a net cost when the
            # answer is already cleanly retrievable). Only escalate to the
            # agentic path when confidence is not HIGH.
            #
            # The gate only engages when the Planner is on AND the plan would
            # otherwise take the multi-subquery / iterative path -- there is
            # nothing to gate on a passthrough single-hop plan (it is already
            # the baseline). Refs: Geifman & El-Yaniv 2019; Asai et al. 2024.
            gate_applies = (
                self.enable_confidence_gate
                and self.enable_planner
                and bool(plan.hop_sequence)
                and len(sub_queries) > 1
            )
            if gate_applies:
                baseline_nav = self.navigator.navigate(plan, [query])
                query_entities = [
                    e.text for e in (getattr(plan, "entities", None) or [])
                ]
                confidence = self._compute_retrieval_confidence(
                    baseline_nav, query_entities
                )
                if confidence == "high":
                    confidence_route = "baseline_confident"
                    nav_result = baseline_nav
                    logger.info(
                        "[Gate] HIGH retrieval confidence -> baseline path "
                        "(skipping Planner decomposition) for query=%.60s",
                        query,
                    )
                else:
                    confidence_route = "agentic_escalated"
                    if has_bridge_deps:
                        nav_result = self._iterative_navigate(plan, query)
                    else:
                        nav_result = self.navigator.navigate(plan, sub_queries)
            else:
                # Gate disabled or not applicable -> existing routing logic.
                if has_bridge_deps:
                    nav_result = self._iterative_navigate(plan, query)
                else:
                    nav_result = self.navigator.navigate(plan, sub_queries)
            navigator_time = (time.time() - navigator_start) * 1000

            # asdict() performs a deep copy and recursively converts nested
            # dataclasses, but converts Enum fields to their raw Enum objects
            # (not .value strings). NavigatorResult currently contains only
            # str/int/float/list/bool fields, so json.dumps() in to_json() is
            # safe. If NavigatorResult gains Enum fields in the future, add an
            # explicit Enum→str conversion here (see verifier_result handling
            # below for the pattern).
            navigator_result = asdict(nav_result)
            # Architecture A: record which route the gate chose so the
            # benchmark / diagnostic can stratify EM by route.
            if isinstance(navigator_result, dict):
                _meta = navigator_result.get("metadata")
                if not isinstance(_meta, dict):
                    _meta = {}
                _meta["confidence_route"] = confidence_route
                navigator_result["metadata"] = _meta
            logger.debug(
                "S_N completed: %d chunks (%.2fms)",
                len(nav_result.filtered_context), navigator_time,
            )

            # ── Stage 3: S_V (Verifier) ───────────────────────────────────────
            # process() calls generate_and_verify() exactly once. The self-correction
            # loop (up to max_verification_iterations rounds) is managed entirely
            # inside the Verifier. An outer retry loop would be a no-op because
            # temperature=0.1 makes the LLM nearly deterministic.
            # See TECHNICAL_ARCHITECTURE.md §12.2.
            verifier_start = time.time()
            if self.enable_verifier:
                # Extract entities and hop_sequence from the plan so the Verifier
                # can run entity-path validation (enable_entity_path_validation in
                # settings.yaml). Without these, pre-validation is silently skipped.
                plan_entities: List[str] = (
                    [e.text for e in plan.entities] if plan.entities else []
                )
                plan_hop_sequence = plan.hop_sequence if plan.hop_sequence else None

                # B1-fix: forward query_type and bridge_entities so the Verifier
                # actually selects BRIDGE_PROMPT for multi-hop and COMPARISON_PROMPT
                # for comparison queries (instead of falling through to ANSWER_PROMPT
                # for every query). query_type is sourced from the Planner; bridge
                # entities are sourced from the iterative-navigate metadata when the
                # plan had dependent hops, falling back to an empty list otherwise.
                plan_query_type: Optional[str] = None
                if hasattr(plan, "query_type") and plan.query_type is not None:
                    try:
                        plan_query_type = plan.query_type.value  # Enum -> str
                    except AttributeError:
                        plan_query_type = str(plan.query_type)

                bridge_entities_for_verifier: List[str] = []
                chunk_is_graph_based: Optional[List[bool]] = None
                nav_metadata = getattr(nav_result, "metadata", None) or {}
                if isinstance(nav_metadata, dict):
                    bridge_entities_for_verifier = (
                        nav_metadata.get("resolved_bridges", []) or []
                    )
                    # B2-fix: forward per-chunk graph-provenance from the
                    # Navigator so the Verifier credibility scorer uses a real
                    # signal instead of a constant baseline.
                    flags = nav_metadata.get("chunk_is_graph_based")
                    if isinstance(flags, list) and len(flags) == len(nav_result.filtered_context):
                        chunk_is_graph_based = [bool(f) for f in flags]

                gen_result = self.verifier.generate_and_verify(
                    query=query,
                    context=nav_result.filtered_context,
                    entities=plan_entities,
                    hop_sequence=plan_hop_sequence,
                    query_type=plan_query_type,
                    bridge_entities=bridge_entities_for_verifier or None,
                    chunk_is_graph_based=chunk_is_graph_based,
                )
                # asdict() deep-copies the dataclass but converts Enum fields to raw
                # Enum objects, not .value strings. Override every Enum field explicitly
                # so json.dumps() in to_json() never raises TypeError.
                gen_dict = asdict(gen_result)
                gen_dict["confidence"] = gen_result.confidence.value
                if gen_result.pre_validation is not None and gen_dict.get("pre_validation"):
                    gen_dict["pre_validation"]["status"] = (
                        gen_result.pre_validation.status.value
                    )
                verifier_result = gen_dict
                answer = gen_result.answer
                confidence_val = gen_result.confidence.value
                logger.debug(
                    "S_V completed: confidence=%s iterations=%d (%.2fms)",
                    confidence_val, gen_result.iterations,
                    (time.time() - verifier_start) * 1000,
                )
            else:
                # Ablation: --no-verifier mode.
                answer = (
                    nav_result.filtered_context[0].strip()
                    if nav_result.filtered_context
                    else "No answer found."
                )
                confidence_val = "low"
                verifier_result = {
                    "verifier_skipped": True,
                    "answer": answer,
                    "confidence": "low",
                }
                logger.info("S_V skipped (enable_verifier=False)")
            verifier_time = (time.time() - verifier_start) * 1000

        except Exception as exc:  # noqa: BLE001
            # Structured error result — never swallowed silently.
            # Callers detect failure via result.confidence == "error".
            logger.error(
                "Pipeline inference error for query '%.60s': %s",
                query, exc, exc_info=True,
            )
            answer = f"Error: {exc}"
            confidence_val = "error"
            verifier_result = {"error": str(exc), "confidence": "error"}

        total_time = (time.time() - start_time) * 1000

        # ── Welford online mean for avg_latency_ms ────────────────────────────
        # Avoids accumulating all latency values in a list.
        # Reference: Welford, B.P. (1962). Technometrics, 4(3), 419-420.
        # DOI: 10.2307/1266577
        n = self._stats["total_queries"]
        self._stats["avg_latency_ms"] += (
            (total_time - self._stats["avg_latency_ms"]) / n
        )

        result = PipelineResult(
            answer=answer,
            confidence=confidence_val,
            query=query,
            planner_result=planner_result,
            navigator_result=navigator_result,
            verifier_result=verifier_result,
            planner_time_ms=planner_time,
            navigator_time_ms=navigator_time,
            verifier_time_ms=verifier_time,
            total_time_ms=total_time,
        )

        self._update_cache(query, result)

        logger.info(
            "Pipeline completed: total=%.2fms (S_P=%.1f S_N=%.1f S_V=%.1f)",
            total_time, planner_time, navigator_time, verifier_time,
        )

        return result

    def _get_cache_key(self, query: str) -> str:
        """
        Return a 16-hex-char SHA-256 key for the normalised query string.

        Truncation to 64 bits is safe for a cache of ≤10,000 entries
        (collision probability < 10⁻¹⁴).
        """
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    def _update_cache(self, query: str, result: PipelineResult) -> None:
        """
        Insert result into the FIFO cache, evicting the oldest entry if full.

        Eviction uses next(iter(self._cache)), which returns the
        insertion-order oldest key in Python 3.7+ dicts.
        """
        if not self.enable_caching:
            return
        cache_key = self._get_cache_key(query)
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result

    def get_stats(self) -> Dict[str, Any]:
        """
        Return aggregate pipeline statistics.

        ``total_queries`` counts every call to process(), including cache hits.
        ``cache_hit_rate`` is therefore cache_hits / total_queries (cache hits
        included in denominator), not cache_hits / non_cached_queries.
        The denominator is clamped to 1 so the rate is 0.0 before any queries.
        """
        n = max(self._stats["total_queries"], 1)
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "cache_hit_rate": self._stats["cache_hits"] / n,
        }

    def clear_cache(self) -> None:
        """Clear the query result cache."""
        self._cache.clear()
        logger.info("Pipeline cache cleared")


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """
    Convenience wrapper for multi-query batch processing.

    Designed for interactive experimentation and unit testing.
    The primary evaluation pipeline (benchmark_datasets.py) implements its own
    batch loop with full EM/F1 normalisation and does NOT use this class.
    Do not use BatchProcessor.evaluate() for thesis results — its exact-match
    implementation omits article/punctuation normalisation.
    """

    def __init__(
        self,
        pipeline: AgentPipeline,
        show_progress: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.show_progress = show_progress

    def process_batch(
        self,
        queries: List[str],
        return_details: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process a list of queries sequentially.

        Args:
            queries: List of natural language questions.
            return_details: If True, return full PipelineResult dicts;
                otherwise return simplified {query, answer, confidence,
                latency_ms} dicts.

        Returns:
            One dict per query; failed queries contain an ``"error"`` key.
        """
        results: List[Dict[str, Any]] = []
        total = len(queries)

        for i, query in enumerate(queries):
            try:
                result = self.pipeline.process(query)
                if return_details:
                    results.append(result.to_dict())
                else:
                    results.append({
                        "query": query,
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "latency_ms": result.total_time_ms,
                    })
                if self.show_progress and (i + 1) % 10 == 0:
                    logger.info("Progress: %d/%d queries processed", i + 1, total)
            except Exception as exc:
                logger.error(
                    "Error processing query %d: %s", i, exc, exc_info=True
                )
                results.append({"query": query, "error": str(exc)})

        return results

    @staticmethod
    def _exact_match(prediction: str, ground_truth: str) -> bool:
        """
        Quick sanity-check EM: case-insensitive strip comparison.

        For thesis evaluation numbers use src.evaluations.metrics.compute_exact_match
        which applies the full HotpotQA normalisation pipeline (articles, punctuation,
        whitespace, word-boundary containment).
        """
        return prediction.strip().lower() == ground_truth.strip().lower()

    def evaluate(
        self,
        questions: List[str],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        """
        Run exact-match evaluation over a question/answer list.

        Uses _exact_match (simple strip/lowercase) for quick sanity checks.
        For publication-grade thesis results use src/evaluations/evaluate_hotpotqa.py
        which applies the canonical metrics from src/evaluations/metrics.py.
        """
        if len(questions) != len(ground_truths):
            raise ValueError(
                f"questions and ground_truths must be the same length "
                f"(got {len(questions)} vs {len(ground_truths)})"
            )
        results = self.process_batch(questions)
        correct = sum(
            self._exact_match(r.get("answer", ""), gt)
            for r, gt in zip(results, ground_truths)
            if "error" not in r
        )
        total = len(questions)
        return {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total_queries": total,
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_full_pipeline(
    hybrid_retriever: Any,
    graph_store: Any,
    config: Optional[Dict[str, Any]] = None,
) -> AgentPipeline:
    """
    Primary factory for a fully configured AgentPipeline.

    Injects ``hybrid_retriever`` and ``graph_store`` into the pipeline, then
    calls ``_lazy_init_agents()`` to construct all three agents using the same
    code path as the lazy-init route in ``process()``. This ensures that
    ``create_full_pipeline`` and lazy initialisation always produce identical
    agent configurations (DRY principle).

    This is the entry point called by benchmark_datasets.py and
    evaluate_hotpotqa.py for all evaluation runs.

    Args:
        hybrid_retriever: Data-layer retriever (LanceDB + KuzuDB) for Navigator.
        graph_store: Graph store for Verifier provenance lookup.
        config: Full settings.yaml dict. Pass None only for unit tests — doing
            so applies all hardcoded defaults, which may not match the thesis
            evaluation configuration.

    Returns:
        Fully initialised AgentPipeline with all three agents wired.
    """
    config = config or {}
    if not config:
        logger.warning(
            "FALLBACK ACTIVE: No config provided to create_full_pipeline. "
            "All agent configs use hardcoded defaults. "
            "Pass settings.yaml content for reproducible evaluation results."
        )

    pipeline = AgentPipeline(
        hybrid_retriever=hybrid_retriever,
        graph_store=graph_store,
        config=config,
    )
    # Eagerly build all three agents using the shared _lazy_init_agents() path.
    # This is the only place where agent construction should occur so that
    # create_full_pipeline() and the lazy-init path stay in sync.
    pipeline._lazy_init_agents()
    return pipeline


# ============================================================================
# SELF-VERIFICATION
# ============================================================================

def _main() -> None:
    """
    Smoke demo and test runner for direct module invocation.

    Constructs a pipeline with mock agents, runs three queries, and validates
    basic result structure. Then invokes the associated pytest test file.
    """
    # Local imports — only needed for this dev-utility function, not at
    # module load time (reduces startup cost on edge hardware).
    import subprocess
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from unittest.mock import MagicMock

    # ── Build mock agents ─────────────────────────────────────────────────────
    def _make_plan(query):
        plan = MagicMock()
        plan.query_type.value = "multi_hop"
        plan.strategy.value = "hybrid"
        plan.hop_sequence = []
        plan.to_dict.return_value = {"query_type": "multi_hop", "strategy": "hybrid"}
        return plan

    def _make_nav_result():
        nav = MagicMock()
        nav.filtered_context = ["Albert Einstein was born in Ulm, Germany in 1879."]
        return nav

    def _make_gen_result():
        from ..logic_layer.verifier import ConfidenceLevel, VerificationResult
        return VerificationResult(
            answer="Ulm, Germany",
            iterations=1,
            verified_claims=["claim"],
            violated_claims=[],
            all_verified=True,
            timing_ms=10.0,
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )

    mock_planner = MagicMock()
    mock_planner.plan.side_effect = _make_plan

    mock_navigator = MagicMock()
    mock_navigator.navigate.return_value = _make_nav_result()

    mock_verifier = MagicMock()
    mock_verifier.generate_and_verify.return_value = _make_gen_result()

    pipeline = AgentPipeline(
        planner=mock_planner,
        navigator=mock_navigator,
        verifier=mock_verifier,
        config={"agent": {"max_verification_iterations": 2}},
    )

    # ── Run smoke queries ─────────────────────────────────────────────────────
    queries = [
        "What is the capital of France?",
        "How tall is Mount Everest?",
        "What is the capital of France?",   # should be a cache hit
    ]
    for i, q in enumerate(queries):
        result = pipeline.process(q)
        assert isinstance(result, PipelineResult), "Expected PipelineResult"
        assert result.answer, "Expected non-empty answer"
        assert result.total_time_ms >= 0
        logger.info(
            "Query %d: answer=%r cached=%s latency=%.2fms",
            i, result.answer, result.cached_result, result.total_time_ms,
        )

    stats = pipeline.get_stats()
    assert stats["total_queries"] == 3
    assert stats["cache_hits"] == 1, "Third query should be a cache hit"
    logger.info("Stats: %s", stats)
    logger.info("Smoke demo passed.")

    # ── pytest ────────────────────────────────────────────────────────────────
    test_file = Path(__file__).parent.parent.parent / "test_system" / "test_pipeline.py"
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", str(test_file), "-v"],
        check=False,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    _main()
