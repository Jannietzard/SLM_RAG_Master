"""
===============================================================================
AgenticController — S_P → S_N → S_V Pipeline Orchestrator
===============================================================================

Master's Thesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"
Artifact B: Agent-Based Query Processing — Pipeline Controller

The AgenticController orchestrates the three agents per the thesis design.
It supports two execution modes:

  LangGraph mode  — StateGraph workflow (when langgraph is installed)
  Fallback mode   — sequential _run_simple_pipeline (always available)

Both modes produce an identical AgentState result dict. The thesis evaluation
was conducted in fallback mode (LangGraph is not a hard dependency).

Reference for self-correction loop (S_V): Madaan et al. (2023). "Self-Refine:
Iterative Refinement with Self-Feedback." NeurIPS 2023.

===============================================================================
ARCHITECTURE
===============================================================================

    User Query
        │
        ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                   Pipeline Controller                              │
    │                                                                    │
    │   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐│
    │   │     S_P     │────────▶│     S_N     │────────▶│     S_V     ││
    │   │   PLANNER   │         │  NAVIGATOR  │         │  VERIFIER   ││
    │   └─────────────┘         └─────────────┘         └─────────────┘│
    │         │                       │                       │         │
    │    ┌────▼────┐            ┌────▼────┐            ┌────▼────┐    │
    │    │Query    │            │Hybrid   │            │Pre-     │    │
    │    │Analysis │            │Retrieval│            │Validation│    │
    │    │Entity   │            │RRF      │            │Generation│    │
    │    │Extract  │            │Fusion   │            │Self-    │    │
    │    │Plan Gen │            │Pre-Gen  │            │Correct  │    │
    │    └─────────┘            │Filter   │            └─────────┘    │
    │                           └─────────┘                           │
    └───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            Final Answer + Metadata

===============================================================================
INTER-AGENT COMMUNICATION
===============================================================================

Agents communicate via structured messages (JSON-compatible dicts) that carry
intermediate results and metadata (confidence scores, retrieval provenance).

S_P → S_N:
    - RetrievalPlan (query type, strategy, entities, hop sequence)

S_N → S_V:
    - Filtered context (after RRF fusion and pre-gen filtering)
    - Retrieval metadata (scores, provenance)

===============================================================================
"""

import logging
import time
from typing import Any, Dict, List, NotRequired, Optional, TypedDict, cast

from .navigator import ControllerConfig, Navigator, NavigatorResult
from .planner import Planner, QueryType, RetrievalPlan, RetrievalStrategy, create_planner
from .verifier import Verifier, create_verifier

# Module logger — defined before the LangGraph import block so that the
# ImportError warning uses the module-namespaced logger (Action 2).
logger = logging.getLogger(__name__)

# LangGraph is optional: when absent the controller falls back to a sequential
# _run_simple_pipeline.  The thesis evaluation was conducted in fallback mode.
try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning(
        "LangGraph not installed — using sequential fallback: pip install langgraph"
    )


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    Shared state for the S_P → S_N → S_V pipeline.

    Carries all intermediate results between pipeline stages.

    Planner output:
        query: original user query
        retrieval_plan: full RetrievalPlan from S_P (serialized dict); NotRequired
            because node functions return partial update dicts that omit unchanged keys.
        sub_queries: flat list of sub-queries
        entities: extracted entity name strings
        query_type: classified query type

    Navigator output:
        raw_context: unfiltered chunks from retrieval
        context: filtered chunks after pre-gen filtering
        retrieval_scores: RRF score per filtered chunk
        retrieval_metadata: additional provenance metadata

    Verifier output:
        answer: final answer string
        iterations: number of self-correction iterations run
        verified_claims: claims confirmed by the verifier
        violated_claims: claims that failed verification
        all_verified: True when all claims are verified
        pre_validation: pre-generation validation result dict; NotRequired —
            same reason as retrieval_plan.

    Metadata:
        total_time_ms: total pipeline wall time
        errors: accumulated error messages from all stages
        stage_timings: per-stage timing breakdown
    """
    # Input
    query: str

    # Planner Output
    retrieval_plan: NotRequired[Optional[Dict[str, Any]]]
    sub_queries: List[str]
    entities: List[str]
    query_type: str

    # Navigator Output
    raw_context: List[str]
    context: List[str]
    retrieval_scores: List[float]
    retrieval_metadata: Dict[str, Any]

    # Verifier Output
    answer: str
    iterations: int
    verified_claims: List[str]
    violated_claims: List[str]
    all_verified: bool
    pre_validation: NotRequired[Optional[Dict[str, Any]]]

    # Metadata
    total_time_ms: float
    errors: List[str]
    stage_timings: Dict[str, float]


# =============================================================================
# AGENTIC CONTROLLER
# =============================================================================

class AgenticController:
    """
    S_P → S_N → S_V pipeline controller.

    Orchestrates the three agents and manages pipeline state.

    Usage::

        controller = create_controller()
        controller.set_retriever(hybrid_retriever)
        result = controller.run("Who directed Inception?")
        print(result["answer"])
    """

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        planner: Optional[Planner] = None,
        verifier: Optional[Verifier] = None,
        full_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the pipeline controller.

        Args:
            config: ControllerConfig (defaults to ControllerConfig()).
            planner: optional pre-configured Planner instance.
            verifier: optional pre-configured Verifier instance.
            full_cfg: full settings.yaml dict; when provided, all sub-blocks
                (including ``verifier:``) are passed to ``create_verifier`` so
                that every setting from settings.yaml is honoured.  When None,
                a minimal cfg is constructed from the ControllerConfig fields.
        """
        self.config = config or ControllerConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize pipeline components
        self.planner = planner or create_planner()

        if verifier is not None:
            self.verifier = verifier
        elif full_cfg is not None:
            # Pass the complete settings dict so the verifier block is honoured.
            # ControllerConfig fields take precedence for llm/agent sub-keys.
            merged_cfg: Dict[str, Any] = dict(full_cfg)
            merged_cfg.setdefault("llm", {})
            merged_cfg["llm"]["model_name"] = self.config.model_name
            merged_cfg["llm"]["base_url"] = self.config.base_url
            merged_cfg["llm"]["max_chars_per_doc"] = self.config.max_chars_per_doc
            merged_cfg.setdefault("agent", {})
            merged_cfg["agent"]["max_verification_iterations"] = (
                self.config.max_verification_iterations
            )
            self.verifier = create_verifier(cfg=merged_cfg)
        else:
            # Minimal cfg — verifier block defaults apply (no settings.yaml verifier section).
            self.verifier = create_verifier(
                cfg={
                    "llm": {
                        "model_name": self.config.model_name,
                        "base_url": self.config.base_url,
                        "max_chars_per_doc": self.config.max_chars_per_doc,
                    },
                    "agent": {
                        "max_verification_iterations": self.config.max_verification_iterations,
                    },
                }
            )

        self.navigator = Navigator(self.config)

        # Build Workflow
        if LANGGRAPH_AVAILABLE:
            self.app = self._build_workflow()
            self.logger.info("Pipeline controller initialized with LangGraph")
        else:
            self.app = None
            self.logger.info("Pipeline controller initialized with simple pipeline fallback")

    def set_retriever(self, retriever: Any, documents: Optional[Dict[str, str]] = None) -> None:
        """
        Attach a HybridRetriever to the Navigator.

        Args:
            retriever: HybridRetriever instance (typed Any to avoid cross-layer import)
            documents: optional dict mapping doc_id → text
        """
        self.navigator.set_retriever(retriever, documents)
        self.logger.info("HybridRetriever connected to Navigator")

    def set_graph_store(self, graph_store: Any) -> None:
        """
        Attach a KnowledgeGraphStore to the Verifier for pre-validation.

        Args:
            graph_store: HybridStore or KuzuGraphStore instance (typed Any to
                avoid cross-layer import)
        """
        self.verifier.set_graph_store(graph_store)
        self.logger.info("GraphStore connected to Verifier")

    # ─────────────────────────────────────────────────────────────────────────
    # LANGGRAPH WORKFLOW
    # ─────────────────────────────────────────────────────────────────────────

    def _build_workflow(self) -> Any:
        """Build and compile the LangGraph StateGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("navigator", self._navigator_node)
        workflow.add_node("verifier", self._verifier_node)

        workflow.add_edge("planner", "navigator")
        workflow.add_edge("navigator", "verifier")
        workflow.add_edge("verifier", END)

        workflow.set_entry_point("planner")

        return workflow.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE NODES
    # ─────────────────────────────────────────────────────────────────────────

    def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """
        S_P (Planner) node: query analysis and retrieval plan generation.

        Input:  query
        Output: retrieval_plan, sub_queries, entities, query_type
        """
        self.logger.info("--- [S_P PLANNER] Query Analysis ---")

        start_time = time.time()

        try:
            plan = self.planner.plan(state["query"])

            sub_queries = plan.sub_queries
            entities = [e.text for e in plan.entities]
            query_type = plan.query_type.value

            self.logger.info("[S_P] Query type: %s", query_type)
            self.logger.info("[S_P] Strategy: %s", plan.strategy.value)
            self.logger.info("[S_P] Entities: %s", entities)
            self.logger.info("[S_P] Sub-queries: %d", len(sub_queries))
            for i, sq in enumerate(sub_queries, 1):
                self.logger.info("      %d. %s", i, sq)

            elapsed = (time.time() - start_time) * 1000

            return {
                "retrieval_plan": plan.to_dict(),
                "sub_queries": sub_queries,
                "entities": entities,
                "query_type": query_type,
                "stage_timings": {"planner_ms": elapsed},
            }

        except Exception as e:
            # Broad catch is intentional: Planner errors (SpaCy, malformed query)
            # must not abort the pipeline — fall back to treating the raw query
            # as a single sub-query so retrieval can still proceed.
            self.logger.error("[S_P] Error: %s", e, exc_info=True)
            elapsed = (time.time() - start_time) * 1000
            return {
                "retrieval_plan": None,
                "sub_queries": [state["query"]],
                "entities": [],
                "query_type": QueryType.SINGLE_HOP.value,
                "errors": [f"Planner error: {e}"],
                "stage_timings": {"planner_ms": elapsed},
            }

    @staticmethod
    def _extract_bridge_entities(chunks: List[str], exclude: List[str]) -> List[str]:
        """
        Extract candidate bridge entity names from retrieved text chunks.

        Simple heuristic: capitalized multi-word phrases (2+ tokens) that are
        not already in the known entity list.  Used by iterative multi-hop to
        discover the bridge entity name from step-N results before step-N+1.

        Args:
            chunks: Filtered context chunks from a bridge retrieval step.
            exclude: Entity strings already known (from S_P extraction).

        Returns:
            Up to 3 candidate bridge entity strings.
        """
        import re as _re
        exclude_lower = {e.lower() for e in exclude}
        seen: set = set()
        candidates: List[str] = []
        for chunk in chunks:
            for m in _re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', chunk):
                phrase = m.group(1)
                phrase_lower = phrase.lower()
                if (
                    phrase_lower not in exclude_lower
                    and len(phrase) > 4
                    and phrase_lower not in seen
                ):
                    seen.add(phrase_lower)
                    candidates.append(phrase)
            if len(candidates) >= 5:
                break
        return candidates[:3]

    def _iterative_navigator_node(
        self,
        state: AgentState,
        hop_sequence_raw: List[Dict[str, Any]],
        entity_names: List[str],
        plan_dict: Optional[Dict[str, Any]],
        start_time: float,
    ) -> Dict[str, Any]:
        """
        Execute HopSteps sequentially, feeding bridge entities discovered in
        step N as entity_hints into step N+1.

        This enables "hidden bridge" resolution: when the query asks about
        entity A via an unnamed intermediate entity B (described in the query),
        step 0 retrieves B, the controller extracts B's name, and step 1 uses
        B's name to retrieve the final answer.

        Reference: multi-hop retrieval augmentation; thesis section 3.x.
        """
        accumulated_raw: List[str] = []
        accumulated_context: List[str] = []
        seen_raw: set = set()
        seen_filtered: set = set()
        current_hints = list(entity_names)

        hops = sorted(hop_sequence_raw, key=lambda h: h.get("step_id", 0))

        for hop in hops:
            sub_query = hop.get("sub_query", state["query"])
            is_bridge = hop.get("is_bridge", False)
            step_id = hop.get("step_id", 0)

            self.logger.info(
                "[S_N Iterative] Step %d (bridge=%s) query=%r  hints=%s",
                step_id, is_bridge, sub_query[:60], current_hints,
            )

            # Build minimal plan for single-step navigation
            if plan_dict:
                plan = RetrievalPlan(
                    original_query=state["query"],
                    query_type=QueryType(
                        plan_dict.get("query_type", QueryType.SINGLE_HOP.value)
                    ),
                    strategy=RetrievalStrategy(
                        plan_dict.get("strategy", RetrievalStrategy.HYBRID.value)
                    ),
                    sub_queries=[sub_query],
                )
            else:
                plan = None

            try:
                nav_result = self.navigator.navigate(
                    retrieval_plan=plan,
                    sub_queries=[sub_query],
                    entity_names=current_hints,
                )
            except Exception as e:
                self.logger.error("[S_N Iterative] Step %d failed: %s", step_id, e)
                continue

            # Accumulate unique chunks
            for chunk in nav_result.raw_context:
                if chunk not in seen_raw:
                    accumulated_raw.append(chunk)
                    seen_raw.add(chunk)

            for chunk in nav_result.filtered_context:
                if chunk not in seen_filtered:
                    accumulated_context.append(chunk)
                    seen_filtered.add(chunk)

            # After a bridge step: discover the bridge entity name
            if is_bridge and nav_result.filtered_context:
                bridge_entities = self._extract_bridge_entities(
                    nav_result.filtered_context[:2],
                    exclude=current_hints,
                )
                if bridge_entities:
                    self.logger.info(
                        "[S_N Iterative] Bridge entities discovered: %s",
                        bridge_entities,
                    )
                    current_hints = current_hints + bridge_entities

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(
            "[S_N Iterative] %d steps done — %d raw / %d filtered chunks, %.1f ms",
            len(hops), len(accumulated_raw), len(accumulated_context), elapsed,
        )

        return {
            "raw_context": accumulated_raw,
            "context": accumulated_context,
            "retrieval_scores": [],
            "retrieval_metadata": {"iterative_hints": current_hints, "hop_count": len(hops)},
            "stage_timings": {
                **state.get("stage_timings", {}),
                "navigator_ms": elapsed,
            },
        }

    def _navigator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        S_N (Navigator) node: hybrid retrieval and pre-generative filtering.

        For multi-hop plans with dependent steps, delegates to
        _iterative_navigator_node which executes steps in order and feeds
        bridge entities discovered in step N into step N+1.

        Input:  retrieval_plan, sub_queries
        Output: raw_context, context, retrieval_scores, retrieval_metadata
        """
        self.logger.info("--- [S_N NAVIGATOR] Hybrid Retrieval + Filtering ---")

        start_time = time.time()

        if self.navigator.retriever is None:
            self.logger.warning("[S_N] No retriever set — returning empty context")
            return {
                "raw_context": [],
                "context": [],
                "retrieval_scores": [],
                "retrieval_metadata": {"error": "No retriever"},
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "navigator_ms": 0,
                },
            }

        try:
            plan_dict = state.get("retrieval_plan")
            entity_names = state.get("entities", [])
            hop_sequence_raw: List[Dict[str, Any]] = (
                plan_dict.get("hop_sequence", []) if plan_dict else []
            )

            # ── Iterative multi-hop: execute steps in dependency order ────────
            # When the planner produced dependent bridge steps (depends_on is
            # non-empty for at least one step), run them sequentially so that
            # bridge entities discovered in step N can refine step N+1 retrieval.
            has_bridge_deps = any(
                h.get("depends_on") for h in hop_sequence_raw
            )
            if has_bridge_deps and len(hop_sequence_raw) > 1:
                self.logger.info(
                    "[S_N] Iterative multi-hop: %d dependent steps detected",
                    len(hop_sequence_raw),
                )
                return self._iterative_navigator_node(
                    state, hop_sequence_raw, entity_names, plan_dict, start_time
                )

            # ── Single-pass (original behaviour) ─────────────────────────────
            # Reconstruct a minimal RetrievalPlan from state (if available).
            # The reconstructed plan omits .entities; entity_names is passed
            # explicitly so the entity-mention filter works correctly.
            if plan_dict:
                plan = RetrievalPlan(
                    original_query=state["query"],
                    query_type=QueryType(
                        plan_dict.get("query_type", QueryType.SINGLE_HOP.value)
                    ),
                    strategy=RetrievalStrategy(
                        plan_dict.get("strategy", RetrievalStrategy.HYBRID.value)
                    ),
                    sub_queries=state["sub_queries"],
                )
            else:
                plan = None

            nav_result = self.navigator.navigate(
                retrieval_plan=plan,
                sub_queries=state["sub_queries"],
                entity_names=entity_names,
            )

            self.logger.info("[S_N] Raw context: %d chunks", len(nav_result.raw_context))
            self.logger.info(
                "[S_N] Filtered context: %d chunks", len(nav_result.filtered_context)
            )

            elapsed = (time.time() - start_time) * 1000

            return {
                "raw_context": nav_result.raw_context,
                "context": nav_result.filtered_context,
                "retrieval_scores": nav_result.scores,
                "retrieval_metadata": nav_result.metadata,
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "navigator_ms": elapsed,
                },
            }

        except Exception as e:
            # Broad catch is intentional: Navigator errors (retriever timeout,
            # filter exceptions) must not abort the pipeline — an empty context
            # will cause S_V to produce a low-confidence answer, which is the
            # correct degraded behaviour.
            self.logger.error("[S_N] Error: %s", e, exc_info=True)
            elapsed = (time.time() - start_time) * 1000
            return {
                "raw_context": [],
                "context": [],
                "retrieval_scores": [],
                "retrieval_metadata": {"error": str(e)},
                "errors": state.get("errors", []) + [f"Navigator error: {e}"],
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "navigator_ms": elapsed,
                },
            }

    def _verifier_node(self, state: AgentState) -> Dict[str, Any]:
        """
        S_V (Verifier) node: pre-validation, answer generation, and self-correction.

        Input:  query, context, entities
        Output: answer, iterations, verified_claims, violated_claims, all_verified
        """
        self.logger.info("--- [S_V VERIFIER] Pre-Validation + Generation ---")

        start_time = time.time()

        try:
            # Extract hop sequence for pre-validation (may be None)
            plan_dict = state.get("retrieval_plan", {})
            hop_sequence = plan_dict.get("hop_sequence") if plan_dict else None

            result = self.verifier.generate_and_verify(
                query=state["query"],
                context=state["context"],
                entities=state.get("entities", []),
                hop_sequence=hop_sequence,
            )

            self.logger.info("[S_V] Iterations: %d", result.iterations)
            self.logger.info("[S_V] All verified: %s", result.all_verified)
            self.logger.info("[S_V] Verified claims: %d", len(result.verified_claims))
            self.logger.info("[S_V] Violated claims: %d", len(result.violated_claims))

            elapsed = (time.time() - start_time) * 1000

            pre_val_dict = None
            if result.pre_validation:
                pre_val_dict = {
                    "status": result.pre_validation.status.value,
                    "entity_path_valid": result.pre_validation.entity_path_valid,
                    "contradictions_count": len(result.pre_validation.contradictions),
                    "validation_time_ms": result.pre_validation.validation_time_ms,
                }

            return {
                "answer": result.answer,
                "iterations": result.iterations,
                "verified_claims": result.verified_claims,
                "violated_claims": result.violated_claims,
                "all_verified": result.all_verified,
                "pre_validation": pre_val_dict,
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "verifier_ms": elapsed,
                },
            }

        except Exception as e:
            # Broad catch is intentional: Verifier errors (Ollama timeout, NLI
            # model failure) must not surface as unhandled exceptions — the
            # pipeline returns a clearly marked error answer instead.
            self.logger.error("[S_V] Error: %s", e, exc_info=True)
            elapsed = (time.time() - start_time) * 1000
            return {
                "answer": f"[Error: {e}]",
                "iterations": 0,
                "verified_claims": [],
                "violated_claims": [],
                "all_verified": False,
                "pre_validation": None,
                "errors": state.get("errors", []) + [f"Verifier error: {e}"],
                "stage_timings": {
                    **state.get("stage_timings", {}),
                    "verifier_ms": elapsed,
                },
            }

    # ─────────────────────────────────────────────────────────────────────────
    # SIMPLE PIPELINE (sequential fallback — no LangGraph required)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_initial_state(query: str) -> AgentState:
        """
        Build a blank AgentState for a new pipeline execution.

        Centralises initial state construction so that both the LangGraph
        and the simple-pipeline paths share a single definition (DRY).
        """
        return AgentState(
            query=query,
            retrieval_plan=None,
            sub_queries=[],
            entities=[],
            query_type=QueryType.SINGLE_HOP.value,
            raw_context=[],
            context=[],
            retrieval_scores=[],
            retrieval_metadata={},
            answer="",
            iterations=0,
            verified_claims=[],
            violated_claims=[],
            all_verified=False,
            pre_validation=None,
            total_time_ms=0.0,
            errors=[],
            stage_timings={},
        )

    def _run_simple_pipeline(self, query: str) -> AgentState:
        """
        Execute the pipeline sequentially without LangGraph.

        Uses cast(AgentState, {**state, **update}) instead of state.update()
        to maintain TypedDict type safety — TypedDict has no .update() in the
        type system even though at runtime it is a plain dict.
        """
        state: AgentState = self._make_initial_state(query)
        state = cast(AgentState, {**state, **self._planner_node(state)})
        state = cast(AgentState, {**state, **self._navigator_node(state)})
        state = cast(AgentState, {**state, **self._verifier_node(state)})
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, query: str) -> AgentState:
        """
        Execute the full S_P → S_N → S_V pipeline.

        Args:
            query: user query string

        Returns:
            AgentState with answer and full pipeline metadata
        """
        self.logger.info("--- Pipeline start ---")
        self.logger.info("Query: %s", query)

        start_time = time.time()

        initial_state: AgentState = self._make_initial_state(query)

        if self.app is not None:
            final_state = self.app.invoke(initial_state)
        else:
            final_state = self._run_simple_pipeline(query)

        total_time = (time.time() - start_time) * 1000
        final_state["total_time_ms"] = total_time

        self.logger.info(
            "--- Pipeline complete: %.0f ms | context=%d | iterations=%d | verified=%s ---",
            total_time,
            len(final_state["context"]),
            final_state["iterations"],
            final_state["all_verified"],
        )

        return final_state

    def __call__(self, query: str) -> str:
        """Callable shortcut — returns only the answer string."""
        return self.run(query)["answer"]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_controller(
    cfg: Optional[Dict[str, Any]] = None,
    model_name: str = "phi3",
    base_url: str = "http://localhost:11434",
    max_iterations: int = 2,
    relevance_threshold: float = 0.6,
    redundancy_threshold: float = 0.8,
) -> AgenticController:
    """
    Factory function for AgenticController.

    When ``cfg`` is provided (full settings.yaml dict), all pipeline settings
    including the ``verifier:`` block are read from it — this is the
    reproducible entry point that honours every parameter in settings.yaml.

    When ``cfg`` is None, individual keyword arguments act as overrides on top
    of the ControllerConfig defaults.  **Important:** the keyword arguments only
    cover five of the fifteen ControllerConfig fields (model_name, base_url,
    max_iterations, relevance_threshold, redundancy_threshold).  The remaining
    ten fields — corroboration weights, contradiction thresholds, rrf_k,
    top_k_per_subquery, max_chars_per_doc, and max_context_chunks — use the
    ControllerConfig dataclass defaults and cannot be overridden without
    passing a full ``cfg`` dict.  For evaluation runs, always pass ``cfg``.

    Args:
        cfg: full settings.yaml dict (recommended for evaluation runs).
        model_name: Ollama model for S_V (ignored when cfg is provided).
        base_url: Ollama API URL (ignored when cfg is provided).
        max_iterations: max self-correction iterations — thesis default: 2
            (Madaan et al., 2023). Ignored when cfg is provided.
        relevance_threshold: factor for relevance filter (0.6 × max score).
            Ignored when cfg is provided.
        redundancy_threshold: Jaccard threshold for deduplication.
            Ignored when cfg is provided.

    Returns:
        Configured pipeline controller instance.
    """
    if cfg is not None:
        config = ControllerConfig.from_yaml(cfg)
        return AgenticController(config=config, full_cfg=cfg)

    config = ControllerConfig(
        model_name=model_name,
        base_url=base_url,
        max_verification_iterations=max_iterations,
        relevance_threshold_factor=relevance_threshold,
        redundancy_threshold=redundancy_threshold,
    )
    return AgenticController(config=config)


# =============================================================================
# MAIN (smoke test — Planner-only, no Ollama required)
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("CONTROLLER SMOKE TEST")
    print(f"LangGraph available: {LANGGRAPH_AVAILABLE}")
    print("=" * 70)

    controller = create_controller(
        max_iterations=2,
        relevance_threshold=0.6,
        redundancy_threshold=0.8,
    )

    # Planner-only test (no retriever needed)
    print("\n--- Planner-only test (no retriever) ---")

    test_queries = [
        "What is the capital of France?",
        "Who directed the movie that stars Tom Hanks in Forrest Gump?",
        "Is Berlin older than Munich?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        plan = controller.planner.plan(query)
        print(f"  Type:        {plan.query_type.value}")
        print(f"  Strategy:    {plan.strategy.value}")
        print(f"  Entities:    {[e.text for e in plan.entities]}")
        print(f"  Sub-queries: {plan.sub_queries}")

    print("\n" + "=" * 70)
    print("Note: full pipeline test requires:")
    print("  1. HybridRetriever (data_layer/hybrid_retriever.py)")
    print("  2. Ollama with phi3 model running")
    print("=" * 70)
    sys.exit(0)
