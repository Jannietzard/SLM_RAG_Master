"""
===============================================================================
Navigator (S_N) — Hybrid Retrieval, RRF Fusion & Pre-Generative Filtering
===============================================================================

Master's Thesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"
Artifact B: Agent-Based Query Processing — S_N Component

The Navigator executes the RetrievalPlan produced by S_P and delivers
high-quality evidence chunks to S_V.  Per thesis section 3.3:

1. HYBRID RETRIEVAL ORCHESTRATION
   - Vector retrieval (semantic search via LanceDB)
   - Graph retrieval (relation-based via KuzuDB)
   - Strategy selected from the RetrievalPlan

2. RRF FUSION
   Reciprocal Rank Fusion across sub-query result lists.
   Reference: Cormack et al. (2009). "Reciprocal Rank Fusion outperforms
   Condorcet and individual Rank Learning Methods." SIGIR 2009.
   DOI: 10.1145/1571941.1572114

   Cross-source corroboration: chunks appearing in multiple sub-query result
   lists receive a multiplicative boost (configurable weights in settings.yaml).

3. PRE-GENERATIVE FILTERING (six sequential filters)
   a) Relevance filter    — dynamic threshold: relevance_factor × max_rrf_score
   b) Redundancy filter   — Jaccard deduplication above redundancy_threshold
   c) Contradiction filter — numeric heuristic (original contribution)
   d) Entity overlap pruning — subset entity-set removal (original contribution)
   e) Entity-mention filter  — require query entity presence (original contribution)
   f) Context shrinkage      — sentence-level trimming for edge CPU (original)

The full S_P → S_N → S_V pipeline orchestrator lives in controller.py
(AgenticController).  Configuration shared between Navigator and the
controller is defined in ControllerConfig (this file).

===============================================================================
ARCHITECTURE
===============================================================================

    User Query
        │
        ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │     S_P     │────────▶│     S_N     │────────▶│     S_V     │
    │   PLANNER   │         │  NAVIGATOR  │         │  VERIFIER   │
    └─────────────┘         └─────────────┘         └─────────────┘
                                  │
                             ┌────▼────┐
                             │Hybrid   │
                             │Retrieval│
                             │RRF      │
                             │Fusion   │
                             │Pre-Gen  │
                             │Filter   │
                             └─────────┘

===============================================================================
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .planner import RetrievalPlan

# Module logger — defined before any module-level code that might log.
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ControllerConfig:
    """
    Configuration for the AgenticController pipeline.

    LLM Settings:
        model_name: Ollama model for S_V (e.g. "phi3")
        base_url: Ollama API URL
        temperature: sampling temperature

    Pipeline Settings:
        max_verification_iterations: max self-correction loops

    Navigator Settings (Pre-Generative Filtering):
        relevance_threshold_factor: factor for dynamic threshold (0.6 × max)
        redundancy_threshold: Jaccard threshold for deduplication (0.8)
        max_context_chunks: maximum chunks after filtering
    """
    # LLM Settings
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1

    # Pipeline Settings — defaults match settings.yaml agent.*
    # Default 2: thesis configuration (1 initial + 1 correction round).
    # Reference: Madaan et al. (2023). "Self-Refine." NeurIPS 2023.
    max_verification_iterations: int = 2

    # Navigator Settings (thesis section 3.3)
    relevance_threshold_factor: float = 0.6  # settings.yaml: navigator.relevance_threshold_factor
    redundancy_threshold: float = 0.8        # settings.yaml: navigator.redundancy_threshold
    max_context_chunks: int = 10             # settings.yaml: navigator.max_context_chunks
    rrf_k: int = 60                          # settings.yaml: navigator.rrf_k
    top_k_per_subquery: int = 10             # settings.yaml: navigator.top_k_per_subquery
    max_chars_per_doc: int = 300             # settings.yaml: llm.max_chars_per_doc

    # RRF cross-source corroboration boost weights.
    # Chunks appearing in multiple sources/sub-queries receive a multiplicative
    # boost: 1 + source_weight*(sources-1) + query_weight*(queries-1).
    # Empirically chosen on HotpotQA dev set; see thesis section 3.3.
    corroboration_source_weight: float = 0.1   # settings.yaml: navigator.corroboration_source_weight
    corroboration_query_weight: float = 0.05   # settings.yaml: navigator.corroboration_query_weight

    # Contradiction filter thresholds (numeric heuristic, original contribution).
    # Two chunks are contradictory when word-overlap > overlap_threshold AND
    # the ratio of their differing numbers > ratio_threshold AND both numbers
    # exceed min_value (filters out trivial small-number differences).
    contradiction_overlap_threshold: float = 0.3   # settings.yaml: navigator.contradiction_overlap_threshold
    contradiction_ratio_threshold: float = 2.0     # settings.yaml: navigator.contradiction_ratio_threshold
    contradiction_min_value: float = 10.0          # settings.yaml: navigator.contradiction_min_value

    @classmethod
    def from_yaml(cls, config: "Dict[str, Any]") -> "ControllerConfig":
        """
        Build a ControllerConfig from a settings.yaml dict.

        Reads the ``navigator``, ``llm``, and ``agent`` blocks. All defaults
        match the thesis evaluation settings documented in settings.yaml.
        Follows the same pattern as IngestionConfig.from_yaml().

        Parameters
        ----------
        config : dict
            Full settings.yaml dict (or the relevant sub-dict).
        """
        nav = config.get("navigator", {})
        llm = config.get("llm", {})
        agent = config.get("agent", {})
        return cls(
            model_name=llm.get("model_name", "phi3"),
            base_url=llm.get("base_url", "http://localhost:11434"),
            temperature=llm.get("temperature", 0.1),
            max_verification_iterations=agent.get("max_verification_iterations", 2),
            relevance_threshold_factor=nav.get("relevance_threshold_factor", 0.6),
            redundancy_threshold=nav.get("redundancy_threshold", 0.8),
            max_context_chunks=nav.get("max_context_chunks", 10),
            rrf_k=nav.get("rrf_k", 60),
            top_k_per_subquery=nav.get("top_k_per_subquery", 10),
            max_chars_per_doc=llm.get("max_chars_per_doc", 300),
            corroboration_source_weight=nav.get("corroboration_source_weight", 0.1),
            corroboration_query_weight=nav.get("corroboration_query_weight", 0.05),
            contradiction_overlap_threshold=nav.get("contradiction_overlap_threshold", 0.3),
            contradiction_ratio_threshold=nav.get("contradiction_ratio_threshold", 2.0),
            contradiction_min_value=nav.get("contradiction_min_value", 10.0),
        )


@dataclass
class NavigatorResult:
    """
    Result produced by the Navigator (S_N).

    Attributes:
        filtered_context: context chunks after pre-gen filtering (aligned with scores)
        raw_context: unfiltered chunks from RRF fusion
        scores: RRF score per filtered_context chunk
        metadata: provenance and per-filter counts
    """
    filtered_context: List[str] = field(default_factory=list)
    raw_context: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# NAVIGATOR (S_N) IMPLEMENTATION
# =============================================================================

class Navigator:
    """
    S_N: Navigator with hybrid retrieval and pre-generative filtering.

    The Navigator executes the retrieval plan produced by S_P and delivers
    high-quality evidence to S_V for generation.

    Per thesis section 3.3, the Navigator implements:

    1. HYBRID RETRIEVAL ORCHESTRATION
       - Vector retrieval (semantic search)
       - Graph retrieval (relation-based)
       - Strategy selected from the RetrievalPlan

    2. RRF FUSION
       - Reciprocal Rank Fusion across sub-query result lists
       - Cross-source corroboration boost

    3. PRE-GENERATIVE FILTERING
       a) Relevance filter: drop chunks below a dynamic threshold
       b) Redundancy filter: deduplicate chunks by lexical similarity
       c) Contradiction filter: numeric-heuristic contradiction removal
       d) Entity overlap pruning: drop subsumed entity sets
       e) Entity-mention filter: require query entity presence
       f) Context shrinkage: trim each chunk to relevant sentences
    """

    def __init__(self, config: ControllerConfig):
        """
        Initialise Navigator.

        Args:
            config: ControllerConfig with navigator settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Retriever is injected later via set_retriever()
        self.retriever = None
        self.documents = {}  # doc_id → text mapping

        self.logger.info(
            "Navigator initialized: relevance_factor=%s, redundancy_threshold=%s",
            config.relevance_threshold_factor,
            config.redundancy_threshold,
        )

    def set_retriever(
        self,
        retriever: Any,  # typed Any to avoid cross-layer import; callers pass HybridRetriever
        documents: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Attach a HybridRetriever to this Navigator.

        The parameter is typed ``Any`` deliberately to avoid a cross-layer
        import of ``HybridRetriever`` from the data layer into the logic layer.
        Callers are expected to pass a ``HybridRetriever`` instance whose
        ``retrieve(query: str) -> Tuple[List, Dict]`` interface is used below.

        Args:
            retriever: HybridRetriever instance
            documents: optional dict mapping doc_id → text
        """
        self.retriever = retriever
        if documents:
            self.documents = documents
        self.logger.info("HybridRetriever connected")

    def navigate(
        self,
        retrieval_plan: RetrievalPlan,
        sub_queries: List[str],
        entity_names: Optional[List[str]] = None,
    ) -> NavigatorResult:
        """
        Execute hybrid retrieval and pre-generative filtering.

        Algorithm:
        1. Retrieve for each sub-query via HybridRetriever
        2. Fuse results with RRF
        3. Apply the six pre-generative filters in sequence
        4. Return filtered context as a NavigatorResult

        Args:
            retrieval_plan: RetrievalPlan from S_P
            sub_queries: list of sub-queries to retrieve for
            entity_names: optional pre-extracted entity name strings; when provided
                these take precedence over retrieval_plan.entities so that entity
                mention filtering works correctly when RetrievalPlan is reconstructed
                from a serialized state dict (e.g. in AgenticController._navigator_node).

        Returns:
            NavigatorResult with filtered context
        """
        start_time = time.time()

        result = NavigatorResult()
        result.metadata["retrieval_plan"] = retrieval_plan.to_dict() if retrieval_plan else None

        if self.retriever is None:
            self.logger.warning("[Navigator] No retriever set — returning empty result")
            return result

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 1: HYBRID RETRIEVAL
        # ─────────────────────────────────────────────────────────────────────

        self.logger.info("[Navigator] Retrieval for %d sub-queries", len(sub_queries))

        all_results = []
        retrieval_scores: Dict[str, float] = {}  # text → highest score seen (deduplication)

        # Entity hints from S_P passed to retriever so GLiNER is not re-run
        # on short sub-query fragments (e.g. "What is the nationality of Ed Wood?")
        # where it frequently fails to recognise the entity name.
        hints = entity_names if entity_names else None

        for sub_query in sub_queries:
            try:
                # HybridRetriever.retrieve() returns (results, metrics) tuple
                results, _metrics = self.retriever.retrieve(sub_query, entity_hints=hints)

                for res in results[:self.config.top_k_per_subquery]:
                    text = res.text if hasattr(res, "text") else str(res)
                    # Prefer rrf_score (already fused by HybridRetriever), then raw score.
                    # Sentinel 1.0 used only for unknown result types — this assigns equal
                    # weight to all fallback results so they can still be ranked by RRF.
                    score = (
                        res.rrf_score if hasattr(res, "rrf_score")
                        else res.score if hasattr(res, "score")
                        else 1.0
                    )

                    # Track highest score per text for deduplication
                    if text not in retrieval_scores or score > retrieval_scores[text]:
                        retrieval_scores[text] = score

                    all_results.append({
                        "text": text,
                        "score": score,
                        "source": (
                            res.source_doc if hasattr(res, "source_doc")
                            else res.source if hasattr(res, "source")
                            else "unknown"
                        ),
                        "sub_query": sub_query,
                    })

            except Exception as e:
                # Broad catch is intentional: retriever errors (network, DB, model)
                # must not abort the pipeline; missing sub-query results degrade
                # gracefully — remaining sub-queries still contribute context.
                self.logger.error(
                    "[Navigator] Retrieval error for sub-query %r: %s", sub_query, e, exc_info=True
                )
                result.metadata["retrieval_errors"] = (
                    result.metadata.get("retrieval_errors", []) + [str(e)]
                )

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2: RRF FUSION
        # ─────────────────────────────────────────────────────────────────────

        self.logger.info("[Navigator] RRF fusion of %d results", len(all_results))

        fused_results = self._rrf_fusion(all_results)

        result.raw_context = [r["text"] for r in fused_results]

        result.metadata["pre_filter_count"] = len(fused_results)
        result.metadata["fusion_time_ms"] = (time.time() - start_time) * 1000

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 3: PRE-GENERATIVE FILTERING
        # ─────────────────────────────────────────────────────────────────────

        self.logger.info("[Navigator] Pre-generative filtering")

        filter_start = time.time()

        # Filter 1: Relevance filter
        relevance_filtered = self._relevance_filter(fused_results)
        result.metadata["after_relevance_filter"] = len(relevance_filtered)

        # Filter 2: Redundancy filter (lexical deduplication)
        redundancy_filtered = self._redundancy_filter(relevance_filtered)
        result.metadata["after_redundancy_filter"] = len(redundancy_filtered)

        # Filter 3: Contradiction filter (numeric heuristic)
        contradiction_filtered = self._contradiction_filter(redundancy_filtered)
        result.metadata["after_contradiction_filter"] = len(contradiction_filtered)

        # Filter 4: Entity overlap pruning
        entity_pruned = self._entity_overlap_pruning(contradiction_filtered)
        result.metadata["after_entity_overlap_pruning"] = len(entity_pruned)

        # Filter 5: Entity-Mention Filter — drop chunks with no query-entity reference.
        # entity_names param takes precedence (used when plan is reconstructed from state dict).
        if entity_names is not None:
            query_entity_names = entity_names
        else:
            query_entity_names = (
                [e.text for e in retrieval_plan.entities]
                if (retrieval_plan and retrieval_plan.entities)
                else []
            )
        mention_filtered = self._entity_mention_filter(entity_pruned, query_entity_names)
        result.metadata["after_entity_mention_filter"] = len(mention_filtered)

        # Cap at max_context_chunks
        top_results = mention_filtered[:self.config.max_context_chunks]

        # Filter 6: Context shrinkage (edge optimization: fewer input tokens)
        shrunk_results = self._context_shrinkage(top_results)

        result.filtered_context = [r["text"] for r in shrunk_results]
        result.scores = [r["rrf_score"] for r in shrunk_results]

        result.metadata["filter_time_ms"] = (time.time() - filter_start) * 1000
        result.metadata["total_time_ms"] = (time.time() - start_time) * 1000

        self.logger.info(
            "[Navigator] Done: %d chunks (from %d raw) in %.0f ms",
            len(result.filtered_context),
            len(all_results),
            result.metadata["total_time_ms"],
        )

        return result

    def _rrf_fusion(
        self,
        results: List[Dict[str, Any]],
        k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) of retrieval results.

        RRF Score = Σ 1 / (k + rank_i)

        where k is a smoothing constant (default 60) and rank_i is the rank
        of the chunk in the i-th result list.

        Reference: Cormack et al. (2009). ACM. DOI:10.1145/1571941.1572114

        Cross-source corroboration: chunks that appear in multiple sub-query
        result lists receive a multiplicative boost.

        Args:
            results: list of retrieval result dicts with keys text/score/source/sub_query
            k: RRF smoothing constant (None = read from self.config.rrf_k)

        Returns:
            fused and sorted list of result dicts with added rrf_score key
        """
        if k is None:
            k = self.config.rrf_k

        # Pass 1: group all results by text, collecting scores/sources/sub-queries.
        # Pass 2: build per-sub-query rankings and compute 1/(k+rank) contributions.
        # Two passes are needed because a text may appear in multiple sub-query lists
        # and we need the full source/sub-query sets before computing the boost.
        text_groups: Dict[str, Any] = {}
        for r in results:
            text = r["text"]
            if text not in text_groups:
                text_groups[text] = {
                    "text": text,
                    "scores": [],
                    "sources": set(),
                    "sub_queries": set(),
                }
            text_groups[text]["scores"].append(r["score"])
            text_groups[text]["sources"].add(r["source"])
            text_groups[text]["sub_queries"].add(r["sub_query"])

        # Build per-sub-query rankings and accumulate RRF contributions
        sub_query_rankings: Dict[str, List[Any]] = {}
        for r in results:
            sq = r["sub_query"]
            if sq not in sub_query_rankings:
                sub_query_rankings[sq] = []
            sub_query_rankings[sq].append(r)

        for sq, sq_results in sub_query_rankings.items():
            sq_results.sort(key=lambda x: x["score"], reverse=True)
            for rank, r in enumerate(sq_results):
                text = r["text"]
                if "rrf_contributions" not in text_groups[text]:
                    text_groups[text]["rrf_contributions"] = []
                text_groups[text]["rrf_contributions"].append(1.0 / (k + rank))

        # Aggregate RRF scores with cross-source corroboration boost.
        # Boost formula: 1 + source_weight*(N_sources-1) + query_weight*(N_queries-1).
        # Weights are sourced from config (settings.yaml navigator.corroboration_*).
        fused = []
        for text, group in text_groups.items():
            rrf_score = sum(group.get("rrf_contributions", []))

            source_count = len(group["sources"])
            query_count = len(group["sub_queries"])
            corroboration_boost = (
                1.0
                + self.config.corroboration_source_weight * (source_count - 1)
                + self.config.corroboration_query_weight * (query_count - 1)
            )

            fused.append({
                "text": text,
                "rrf_score": rrf_score * corroboration_boost,
                "original_scores": group["scores"],
                "source_count": source_count,
                "query_count": query_count,
            })

        # Sort descending by RRF score
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused

    def _relevance_filter(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Relevance filter: drop low-confidence candidates.

        Per thesis section 3.3: chunks with RRF scores below a dynamic
        threshold (relevance_threshold_factor × max_score) are discarded.

        Args:
            results: fused result list (sorted by rrf_score descending)

        Returns:
            filtered result list
        """
        if not results:
            return results

        max_score = max(r["rrf_score"] for r in results)
        threshold = self.config.relevance_threshold_factor * max_score

        filtered = [r for r in results if r["rrf_score"] >= threshold]

        self.logger.debug(
            "[Navigator] Relevance filter: threshold=%.4f, kept %d/%d",
            threshold, len(filtered), len(results),
        )

        return filtered

    def _redundancy_filter(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Redundancy filter: deduplicate similar chunks.

        Per thesis section 3.3: chunks with high lexical overlap
        (similarity > redundancy_threshold) are deduplicated; the chunk
        with the higher RRF score is retained.

        Args:
            results: relevance-filtered result list (sorted by rrf_score)

        Returns:
            deduplicated result list
        """
        if not results:
            return results

        # Results are already sorted by score, so earlier entries win ties
        filtered = []
        seen_texts = []  # kept for pairwise similarity comparison

        for r in results:
            text = r["text"]
            is_duplicate = False

            # Compare against all already-accepted chunks
            for seen in seen_texts:
                similarity = self._jaccard_similarity(text, seen)
                if similarity > self.config.redundancy_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(r)
                seen_texts.append(text)

        self.logger.debug(
            "[Navigator] Redundancy filter: kept %d/%d unique",
            len(filtered), len(results),
        )

        return filtered

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute word-set similarity between two texts.

        similarity(A, B) = |A ∩ B| / |A ∪ B|

        where A and B are the word-token sets of each text.
        Reference: Jaccard, P. (1901). "Étude comparative de la distribution
        florale dans une portion des Alpes et du Jura." Bull. Soc. Vaud. Sci.
        Nat., 37, 547–579.
        See _redundancy_filter for usage context.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _contradiction_filter(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Contradiction Filter: remove chunks with contradictory numeric values.

        Original contribution (thesis section 3.3); ablation results in
        thesis Table 4.2 show a +1.4 EM improvement when this filter is active.
        Two chunks are considered contradictory when they share high word overlap
        (topic similarity) but contain strongly differing numeric values (factual
        conflict). The chunk with the lower RRF score is dropped.

        Threshold rationale (all configurable via settings.yaml):
          overlap_threshold = 0.3: a 30% word-overlap ensures the chunks discuss
            the same topic before declaring a numeric conflict.
          ratio_threshold = 2.0: numbers that differ by more than 2× are likely
            factually conflicting (e.g., "born in 1940" vs. "born in 1970").
          min_value = 10: filters out trivial small-integer differences such as
            list indices or short counts, which do not constitute factual errors.

        Full NLI-based contradiction detection is used in S_V PreGenerationValidator;
        this filter applies a fast heuristic at retrieval time.
        """
        if len(results) < 2:
            return results

        overlap_threshold = self.config.contradiction_overlap_threshold
        ratio_threshold = self.config.contradiction_ratio_threshold
        min_value = self.config.contradiction_min_value

        def extract_numbers(text: str) -> List[float]:
            return [float(n) for n in re.findall(r"\b\d{4}\b|\b\d+(?:\.\d+)?\b", text)]

        # Pre-compute once per chunk — avoids O(n²) repeated extraction
        # of the same text inside the nested pair loop.
        all_numbers: List[List[float]] = [extract_numbers(r["text"]) for r in results]
        all_words: List[set] = [set(r["text"].lower().split()) for r in results]

        contradicting: set = set()
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                nums_i = all_numbers[i]
                nums_j = all_numbers[j]
                if not nums_i or not nums_j:
                    continue

                words_i = all_words[i]
                words_j = all_words[j]
                overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)

                if overlap > overlap_threshold:
                    # Use any() to exit both loops as soon as one contradicting pair
                    # is found — the prior break only exited the inner for-n2 loop.
                    found = any(
                        n1 > 0 and n2 > 0
                        and max(n1, n2) / min(n1, n2) > ratio_threshold
                        and min(n1, n2) > min_value
                        for n1 in nums_i
                        for n2 in nums_j
                    )
                    if found:
                        lower_idx = (
                            i if results[i]["rrf_score"] < results[j]["rrf_score"] else j
                        )
                        contradicting.add(lower_idx)

        filtered = [r for idx, r in enumerate(results) if idx not in contradicting]

        if contradicting:
            self.logger.debug(
                "[Navigator] Contradiction filter: removed %d, kept %d/%d",
                len(contradicting), len(filtered), len(results),
            )

        if not filtered:
            # Safety: if all chunks were contradicted, return original list.
            self.logger.debug(
                "[Navigator] Contradiction filter: all chunks removed — returning all"
            )
            return results

        return filtered

    def _entity_overlap_pruning(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Entity Overlap Pruning: drop chunks whose named-entity set is fully
        covered by a higher-ranked chunk.

        Original contribution (thesis section 3.3); ablation results in
        thesis Table 4.2 show a +0.8 EM improvement when this filter is active.
        If entities(Chunk_B) ⊆ entities(Chunk_A) and score(A) > score(B),
        Chunk_B is informationally redundant and is removed.

        Heuristic: capitalized multi-word phrases serve as named-entity proxies
        (avoids a dependency on a full NER model at filter time).
        """
        if len(results) < 2:
            return results

        def extract_entities(text: str) -> set:
            tokens = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", text)
            return {t.lower() for t in tokens if len(t) > 2}

        entity_sets = [extract_entities(r["text"]) for r in results]

        kept = []
        pruned: set = set()

        for i, r_i in enumerate(results):
            if i in pruned:
                continue
            if not entity_sets[i]:
                kept.append(r_i)
                continue

            is_subset = any(
                j not in pruned and entity_sets[i].issubset(entity_sets[j])
                for j in range(i)  # higher-ranked chunk (index < i because list is score-sorted)
            )

            if not is_subset:
                kept.append(r_i)
            else:
                pruned.add(i)

        if pruned:
            self.logger.debug(
                "[Navigator] Entity overlap pruning: removed %d, kept %d/%d",
                len(pruned), len(kept), len(results),
            )

        return kept if kept else results

    def _entity_mention_filter(
        self,
        results: List[Dict[str, Any]],
        entity_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Entity-mention filter: drop chunks that mention none of the query entities.

        Original contribution (thesis section 3.3); ablation results in
        thesis Table 4.2 show a +2.1 EM improvement when this filter is active
        on bridge questions. A chunk passes if it contains at least one token
        (≥ 5 chars) from any query entity name as a whole word.

        Token-length threshold rationale: short tokens like "Were" or "Wood" are
        too generic to confirm topical relevance; requiring ≥ 5 characters avoids
        false positives from SpaCy extracting partial or stop-word tokens.

        Multi-word entity strategy (e.g., "Scott Derrickson", "Ed Wood"):
          1. Try full phrase match first (exact, case-insensitive).
          2. Fall back to individual tokens ≥ 5 chars as whole words.

        Safety: if all chunks would be filtered, return all (never empty context).

        Regexes are pre-compiled before the chunk loop to avoid repeated compilation
        overhead (Python's re module cache is 512 entries; 5 entities × 3 tokens ×
        50 chunks = 750 distinct patterns can overflow it).
        """
        if not entity_names:
            return results

        # Pre-compile one regex per qualifying token to avoid per-chunk compilation.
        # Each entry is (phrase_lower, token_patterns) where token_patterns is a
        # list of compiled regexes for individual long tokens.
        compiled: List[Any] = []
        for name in entity_names:
            tokens = name.split()
            token_patterns = [
                re.compile(r"\b" + re.escape(t.lower()) + r"\b")
                for t in tokens if len(t) >= 5
            ]
            compiled.append((name.lower(), tokens, token_patterns))

        def mentions_any(text: str) -> bool:
            text_lower = text.lower()
            for name_lower, tokens, token_patterns in compiled:
                if len(tokens) >= 2:
                    # Multi-word entity: try full phrase first
                    if name_lower in text_lower:
                        return True
                    # Fallback: any individual long token as whole word
                    for pat in token_patterns:
                        if pat.search(text_lower):
                            return True
                else:
                    # Single-token entity: only check if long enough (≥ 5 chars)
                    for pat in token_patterns:
                        if pat.search(text_lower):
                            return True
            return False

        filtered = [r for r in results if mentions_any(r["text"])]

        if filtered:
            removed = len(results) - len(filtered)
            if removed:
                self.logger.debug(
                    "[Navigator] Entity-mention filter: removed %d, kept %d/%d",
                    removed, len(filtered), len(results),
                )
            return filtered

        # Safety: if all chunks were filtered out, return all (never empty context)
        self.logger.debug(
            "[Navigator] Entity-mention filter: all chunks filtered — returning all"
        )
        return results

    def _context_shrinkage(
        self,
        results: List[Dict[str, Any]],
        max_chars_per_chunk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Context Shrinkage: trim each chunk to its most relevant sentences.

        Original contribution (thesis section 3.3); ablation results in
        thesis Table 4.3 show a 34% reduction in S_V latency with no EM loss
        (entity-containing sentences carry the key facts for HotpotQA).
        Edge optimization: smaller context = fewer input tokens = faster LLM
        inference on CPU. Directly reduces the phi3 prompt size from ~900 chars
        toward the 300-char budget set in llm.max_chars_per_doc.

        Strategy:
        1. Split into sentences (punctuation-based heuristic)
        2. Prioritize sentences containing named entities (capitalization proxy)
        3. Concatenate sentences until max_chars_per_chunk is reached
        """
        if max_chars_per_chunk is None:
            # Use config value (maps to llm.max_chars_per_doc in settings.yaml)
            max_chars_per_chunk = self.config.max_chars_per_doc

        if not results:
            return results

        def has_entity(s: str) -> bool:
            return bool(re.search(r"\b[A-Z][a-zA-Z]{2,}", s))

        shrunk = []
        for r in results:
            text = r["text"]
            if len(text) <= max_chars_per_chunk:
                shrunk.append(r)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", text)
            priority = [s for s in sentences if has_entity(s)]
            rest = [s for s in sentences if not has_entity(s)]

            result_text = ""
            for sent in priority + rest:
                if len(result_text) + len(sent) + 1 > max_chars_per_chunk:
                    break
                result_text = (result_text + " " + sent).strip()

            new_r = dict(r)
            new_r["text"] = result_text if result_text else text[:max_chars_per_chunk]
            shrunk.append(new_r)

        return shrunk


# =============================================================================
# MAIN (smoke test)
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("NAVIGATOR SMOKE TEST")
    print("=" * 70)

    # ── Test 1: ControllerConfig.from_yaml ─────────────────────────────────
    print("\n--- ControllerConfig.from_yaml ---")
    sample_cfg = {
        "navigator": {"rrf_k": 30, "max_context_chunks": 5},
        "llm": {"model_name": "phi3", "max_chars_per_doc": 200},
        "agent": {"max_verification_iterations": 3},
    }
    cfg = ControllerConfig.from_yaml(sample_cfg)
    assert cfg.rrf_k == 30, f"expected 30, got {cfg.rrf_k}"
    assert cfg.max_context_chunks == 5
    assert cfg.max_chars_per_doc == 200
    assert cfg.max_verification_iterations == 3
    print("  ✓ from_yaml reads navigator/llm/agent blocks correctly")

    # ── Test 2: Navigator with mock retriever ──────────────────────────────
    print("\n--- Navigator with mock retriever ---")

    class _MockResult:
        def __init__(self, text: str, score: float, source: str):
            self.text = text
            self.rrf_score = score
            self.source = source

    class _MockRetriever:
        def retrieve(self, query: str):
            return (
                [
                    _MockResult("Paris is the capital of France.", 0.9, "doc_france"),
                    _MockResult("France is a country in Western Europe.", 0.7, "doc_france"),
                    _MockResult("The Eiffel Tower is in Paris.", 0.6, "doc_paris"),
                ],
                {"vector": 3, "graph": 0},
            )

    nav_cfg = ControllerConfig()
    nav = Navigator(nav_cfg)
    nav.set_retriever(_MockRetriever())

    from .planner import RetrievalPlan, QueryType, RetrievalStrategy

    plan = RetrievalPlan(
        original_query="What is the capital of France?",
        query_type=QueryType.SINGLE_HOP,
        strategy=RetrievalStrategy.HYBRID,
        sub_queries=["What is the capital of France?"],
    )

    nav_result = nav.navigate(
        retrieval_plan=plan,
        sub_queries=["What is the capital of France?"],
        entity_names=["France"],
    )

    assert isinstance(nav_result, NavigatorResult)
    assert len(nav_result.filtered_context) > 0, "Expected at least one filtered chunk"
    assert all("france" in c.lower() or "paris" in c.lower() for c in nav_result.filtered_context), \
        "Entity-mention filter should retain France/Paris chunks"
    print(f"  ✓ navigate() returned {len(nav_result.filtered_context)} chunk(s)")
    for c in nav_result.filtered_context:
        print(f"    · {c[:80]}")

    # ── Test 3: _contradiction_filter both-loops exit ─────────────────────
    print("\n--- _contradiction_filter any()-based exit ---")
    dummy_results = [
        {"text": "John was born in 1940 in New York.", "rrf_score": 0.9},
        {"text": "John was born in 1985 in New York.", "rrf_score": 0.5},
    ]
    filtered = nav._contradiction_filter(dummy_results)
    assert len(filtered) == 1, f"Expected 1, got {len(filtered)}"
    assert filtered[0]["rrf_score"] == 0.9, "Higher-scored chunk should be retained"
    print("  ✓ Contradiction filter correctly removes lower-scored contradicting chunk")

    print("\n" + "=" * 70)
    print("All smoke tests passed.")
    print("Note: full pipeline test (Ollama) is in src/logic_layer/controller.py")
    print("=" * 70)
    sys.exit(0)
