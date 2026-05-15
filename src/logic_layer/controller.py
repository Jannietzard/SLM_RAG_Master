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

Review History:
    Last Reviewed:  2026-04-21
    Review Result:  0 CRITICAL, 4 IMPORTANT, 9 RECOMMENDED
    Reviewer:       Code Review Prompt v2.1
    All findings applied: create_controller() defaults aligned to
    settings.yaml, iterative multi-hop silent failure fixed, module
    structure cleaned up.
    Next Review:    After thesis section numbers are finalized

===============================================================================
"""

import logging
import re
import time
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict, cast

from ._config import ControllerConfig
from .navigator import Navigator, NavigatorResult
from .planner import Planner, QueryType, RetrievalPlan, RetrievalStrategy, create_planner
from .verifier import Verifier, create_verifier

# Module logger — defined before the LangGraph import block so that the
# ImportError warning uses the module-namespaced logger (Action 2).
logger = logging.getLogger(__name__)


from ._settings import _load_settings, _PROPER_NOUN_RE

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

        # Initialize pipeline components — pass full_cfg so each component
        # reads its own block from the same settings.yaml dict rather than
        # triggering a second file-load.
        self.planner = planner or create_planner(cfg=full_cfg)

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
            logger.info("Pipeline controller initialized with LangGraph")
        else:
            self.app = None
            logger.info("Pipeline controller initialized with simple pipeline fallback")

    def set_retriever(self, retriever: Any) -> None:
        """
        Attach a HybridRetriever to the Navigator.

        Args:
            retriever: HybridRetriever instance (typed Any to avoid cross-layer import)
        """
        self.navigator.set_retriever(retriever)
        logger.info("HybridRetriever connected to Navigator")

    def set_graph_store(self, graph_store: Any) -> None:
        """
        Attach a KnowledgeGraphStore to the Verifier for pre-validation.

        Args:
            graph_store: HybridStore or KuzuGraphStore instance (typed Any to
                avoid cross-layer import)
        """
        self.verifier.set_graph_store(graph_store)
        logger.info("GraphStore connected to Verifier")

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
        logger.info("--- [S_P PLANNER] Query Analysis ---")

        start_time = time.time()

        try:
            plan = self.planner.plan(state["query"])

            sub_queries = plan.sub_queries
            entities = [e.text for e in plan.entities]
            query_type = plan.query_type.value

            logger.info("[S_P] Query type: %s", query_type)
            logger.info("[S_P] Strategy: %s", plan.strategy.value)
            logger.info("[S_P] Entities: %s", entities)
            logger.info("[S_P] Sub-queries: %d", len(sub_queries))
            for i, sq in enumerate(sub_queries, 1):
                logger.info("      %d. %s", i, sq)

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
            logger.error("[S_P] Error: %s", e, exc_info=True)
            elapsed = (time.time() - start_time) * 1000
            return {
                "retrieval_plan": None,
                "sub_queries": [state["query"]],
                "entities": [],
                "query_type": QueryType.SINGLE_HOP.value,
                "errors": [f"Planner error: {e}"],
                "stage_timings": {"planner_ms": elapsed},
            }

    # Query-keyword stopwords for relevance ranking. These are non-content
    # words removed before measuring proximity between a candidate entity and
    # the query keywords in the source chunk (§12.32 Fix 2).
    _QUERY_STOPWORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "has", "have", "he", "her", "him", "his", "how",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
        "they", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "whom", "whose", "why", "will", "with", "would",
        "you", "your", "i", "we", "us", "our", "she", "them",
    })

    # Expected entity-type hints from interrogative words. Used to prefer
    # candidates whose surface form matches the expected category.
    _QUERY_TYPE_HINTS = {
        # who/whose → person
        "who": "PERSON", "whose": "PERSON", "whom": "PERSON",
        # where → location/place
        "where": "GPE",
        # when → date/year (matched by digit pattern, not name-style)
        "when": "DATE",
    }

    @classmethod
    def _score_bridge_candidate(
        cls,
        candidate: str,
        chunk: str,
        query: str,
        expected_type: Optional[str] = None,
    ) -> float:
        """
        Relevance score for a bridge-entity candidate.

        Combines three signals:
            +α  Proximity to query keywords in the source chunk (the closer
                the candidate sits to terms the user actually asked about,
                the more likely it is the intended bridge).
            +β  Match to expected entity type (PERSON candidates score
                higher when the query says "who", GPE higher for "where").
            -γ  Position penalty for being far from the start of the chunk
                (top-of-chunk entities are usually the topic; later mentions
                are tangential — "Robert Durst... Galveston" appears after
                the subject's main definition).

        Returns a float in roughly [0, 3]. Higher = more likely the bridge.
        """
        chunk_lower = chunk.lower()
        candidate_lower = candidate.lower()
        cand_pos = chunk_lower.find(candidate_lower)
        if cand_pos < 0:
            return 0.0

        # ── Signal 1: proximity to query keywords ─────────────────────────
        # Score inversely proportional to the distance (in characters)
        # between the candidate and the nearest query keyword in the chunk.
        query_keywords = [
            w.lower() for w in re.findall(r"\b\w{3,}\b", query)
            if w.lower() not in cls._QUERY_STOPWORDS
        ]
        min_dist = float("inf")
        for kw in query_keywords:
            kw_pos = chunk_lower.find(kw)
            if kw_pos >= 0:
                dist = abs(kw_pos - cand_pos)
                if dist < min_dist:
                    min_dist = dist
        # Convert distance → score in (0, 1]: 0 chars → 1.0, 500 chars → ~0.5
        if min_dist == float("inf"):
            proximity = 0.0
        else:
            proximity = 1.0 / (1.0 + min_dist / 200.0)

        # ── Signal 2: expected-type match ─────────────────────────────────
        # PERSON type-bonus is gated by a role-noun blocklist. Multi-word
        # capitalised phrases like "Texas Private Investigator" or "Sony
        # Pictures Movie" match the surface pattern of a person name but
        # are role descriptions, not names. They get NO type bonus —
        # otherwise they outrank real names ("Sela Ward") when both happen
        # to sit near the same query keywords.
        type_bonus = 0.0
        # Tokens that, when present, mark the candidate as a role/title
        # rather than a person name. Conservative — only common role words.
        _ROLE_TOKENS = {
            "investigator", "director", "producer", "manager", "president",
            "chairman", "founder", "owner", "captain", "coach", "lawyer",
            "attorney", "officer", "secretary", "minister", "governor",
            "actor", "actress", "author", "writer", "composer",
            # Org/title fragments
            "pictures", "studios", "movie", "film", "company", "corporation",
            "investigations", "agency", "department", "division", "group",
            "industries", "limited", "incorporated", "associates",
            # Place-like that show up in addresses
            "private", "public",
        }
        if expected_type == "PERSON":
            tokens = candidate.split()
            tokens_lower = [t.lower() for t in tokens]
            has_role_token = any(t in _ROLE_TOKENS for t in tokens_lower)
            # Heuristic: PERSON names typically have 2-3 capitalised tokens
            # AND no role/title noun. Allow optional middle name.
            if (
                2 <= len(tokens) <= 3
                and all(t[0].isupper() for t in tokens if t)
                and not has_role_token
            ):
                type_bonus = 0.5
            elif has_role_token:
                # Active anti-bonus: this is almost certainly a role, not
                # a person. Subtract to push it below real names.
                type_bonus = -0.3
        elif expected_type == "GPE":
            # GPE: single capitalised token, often shorter (city/country).
            if len(candidate.split()) <= 2:
                type_bonus = 0.3

        # ── Signal 3: position penalty ────────────────────────────────────
        # Candidates in the first 200 chars are the topic; after that,
        # they're usually distractors. Penalty grows linearly to -0.5.
        position_penalty = min(0.5, cand_pos / 600.0)

        return proximity + type_bonus - position_penalty

    @classmethod
    def _detect_expected_type(cls, query: str) -> Optional[str]:
        """Infer the expected bridge-entity type from interrogative words.

        Returns the first matching hint type, or None if the query has no
        recognised question word. Conservative — only fires on canonical
        question starters."""
        q = query.lower().strip()
        for prefix, etype in cls._QUERY_TYPE_HINTS.items():
            if q.startswith(prefix + " ") or f" {prefix} " in q:
                return etype
        # Special case: "the actress who"/"the director who" → PERSON
        if re.search(r"\bthe\s+\w+\s+who\b", q):
            return "PERSON"
        return None

    @staticmethod
    def _extract_bridge_entities(
        chunks: List[str], exclude: List[str], query: str = "",
    ) -> List[str]:
        """
        Extract candidate bridge entity names from retrieved text chunks.

        Three-pass heuristic (§12.32):

        Pass 0 — Location-context extraction (highest priority):
          Scans the top chunk for place names introduced by location prepositions
          ("in the city of X", "capital of X", "in X", "of X").  A single
          capitalised token found this way becomes the primary bridge entity and
          short-circuits the remaining passes.  This recovers city/country names
          (e.g. "Strasbourg") that are the natural bridge in geography questions
          but are invisible to the person-name heuristics in Passes 1 and 2.

        Pass 1 — Surname-anchor search (higher precision):
          For each known entity (e.g. "Kasper Schmeichel"), look for its
          surname-length token ("Schmeichel", ≥6 chars) in ALL provided chunks
          using a unicode-aware pattern that allows an optional middle name
          between the first name and the surname.  This recovers names like
          "Peter Schmeichel" even when the stored text has the full form
          "Peter Bolesław Schmeichel" (the ł character breaks the ASCII-only
          _PROPER_NOUN_RE).  The constructed full name (first + surname) is
          preferred as the bridge entity.

        Pass 2 — General proper-noun fallback (higher recall):
          Falls back to _PROPER_NOUN_RE over all chunks if Pass 1 yields no
          results.  Still caps at 3 candidates.

        Previously this only looked at filtered_context[:2], which meant the
        bridge entity (typically in a later chunk) was never found.
        """
        # Matches a capitalised word that follows a location-preposition phrase.
        # Covers patterns like:
        #   "in the city of Strasbourg"
        #   "the capital of France"
        #   "formed in Strasbourg"
        #   "born in Paris"
        _LOCATION_CTX_RE = re.compile(
            r"(?:in\s+the\s+(?:city|town|village|capital|region|province|district)\s+of"
            r"|capital\s+of"
            r"|(?:in|at|near|of)\s+)"
            r"([A-Z][a-z]{2,}(?:[- ][A-Z][a-z]+)*)",
            re.UNICODE,
        )

        exclude_lower = {e.lower() for e in exclude}
        seen: set = set()
        candidates: List[str] = []

        # ── Pass 0: location-context extraction (geography queries only) ─────
        # Pass 0 was originally unconditional — fires on every chunk regardless
        # of whether the query is asking about a place. That caused
        # over-extraction on person-bridge questions: "What is the middle name
        # of the actress who plays Bobbi Bacha?" returned "New York" and
        # "Galveston" from the gold chunk's tangential clause about Robert
        # Durst's murder trial.
        #
        # Fix (§12.32): gate Pass 0 to queries whose *expected type* is GPE.
        # When the user asks "where/in which city/in which country", Pass 0
        # is the right tool. When they ask "who/which actress/the X of Y",
        # skip Pass 0 entirely so the relevance-ranked Pass 2 can do its job.
        expected_type = AgenticController._detect_expected_type(query)
        if chunks and expected_type == "GPE":
            for m in _LOCATION_CTX_RE.finditer(chunks[0]):
                place = m.group(1).strip()
                place_lower = place.lower()
                if (
                    place_lower not in exclude_lower
                    and place_lower not in seen
                    and len(place) >= 4
                ):
                    seen.add(place_lower)
                    candidates.append(place)

        if candidates:
            return candidates[:3]

        # ── Pass 1: surname-anchor search ────────────────────────────────────
        # Only the LAST token of a 2-3 token exclusion is treated as a
        # surname anchor. Previously this pass iterated EVERY long token of
        # every exclusion, which caused token-level leaks when the exclude
        # list contained compound entities like "Bobbi Bacha in Suburban
        # Madness" — "Suburban" and "Madness" were both ≥6 chars, so the
        # regex matched "Pictures Suburban" and "Movie Madness" as surname
        # anchors. The fix restricts the heuristic to its intended scope:
        # person names of the form "First [Middle] Surname".
        for known in exclude:
            tokens = known.split()
            # Restrict to 2- or 3-token exclusions; longer phrases (those
            # with prepositional context like "X in Y") are not person names.
            if len(tokens) not in (2, 3):
                continue
            # Skip exclusions containing prepositions or articles — they're
            # not person names regardless of token count.
            if any(t.lower() in {"in", "on", "of", "at", "by", "the", "a", "an"}
                   for t in tokens):
                continue
            # Only the last token may be a surname anchor.
            surname = tokens[-1]
            if len(surname) < 6:
                continue
            for _ in (surname,):  # preserve loop indent for minimal diff
                pat = re.compile(
                    r"\b([A-Z][^\s,.()\[\]]{1,})\s+(?:[A-Z][^\s,.()\[\]]+\s+)?"
                    + re.escape(surname)
                    + r"\b",
                    re.UNICODE,
                )
                for chunk in chunks:
                    for m in pat.finditer(chunk):
                        first = m.group(1)
                        full = f"{first} {surname}"
                        full_lower = full.lower()
                        if (
                            full_lower not in exclude_lower
                            and full_lower not in seen
                            and len(full) > 4
                            # reject noise tokens (articles, punct residue)
                            and first not in {"The", "A", "An", "This", "In", "Of"}
                            # reject [About: X] annotation artifacts — the colon
                            # signals a prefix tag, not a proper-noun first name
                            and ":" not in first
                        ):
                            seen.add(full_lower)
                            candidates.append(full)

        if candidates:
            return candidates[:3]

        # ── Pass 2: general proper-noun fallback, query-relevance ranked ──
        # Previously: 2-token names first, then multi-token, in extraction
        # order. That picked up the first geographic noun in a chunk even
        # when a person name was the intended bridge — e.g. "Bobbi Bacha
        # ... Sela Ward ... New York ... Galveston" returned "New York"
        # before "Sela Ward" because of token-count tiebreaking.
        #
        # Now (§12.32 Fix 2): each candidate is scored against the query
        # using proximity to query keywords, expected-type match, and a
        # position penalty for late-in-chunk mentions (which tend to be
        # tangential). The top-scoring candidates win.
        #
        # Also: the exclusion check is now token-substring-aware. When the
        # exclude list contains a compound entity like "Bobbi Bacha in
        # Suburban Madness", the constituent phrases "Bobbi Bacha" and
        # "Suburban Madness" must also be excluded — they are not the
        # bridge, they are the subject of the query.
        expected_type = AgenticController._detect_expected_type(query)

        # Build a set of all sub-phrases of every exclusion (for substring
        # check): "Bobbi Bacha in Suburban Madness" → {"Bobbi Bacha",
        # "Suburban Madness", "Bobbi", "Suburban Madness in", ...}. We only
        # need 2+ token sub-phrases — single tokens are too noisy.
        excluded_subphrases: set = set(exclude_lower)
        for known in exclude:
            tokens = known.split()
            for i in range(len(tokens)):
                for j in range(i + 2, len(tokens) + 1):
                    sub = " ".join(tokens[i:j]).lower()
                    # Skip sub-phrases starting/ending with a preposition
                    # ("in suburban madness" should not become a key).
                    if tokens[i].lower() in {"in", "on", "of", "at", "by", "the", "a", "an"}:
                        continue
                    if tokens[j - 1].lower() in {"in", "on", "of", "at", "by", "the", "a", "an"}:
                        continue
                    excluded_subphrases.add(sub)

        scored: List[Tuple[float, str]] = []
        for chunk in chunks:
            for m in _PROPER_NOUN_RE.finditer(chunk):
                phrase = m.group(1)
                phrase_lower = phrase.lower()
                if (
                    phrase_lower not in excluded_subphrases
                    and len(phrase) > 4
                    and phrase_lower not in seen
                ):
                    seen.add(phrase_lower)
                    score = AgenticController._score_bridge_candidate(
                        phrase, chunk, query, expected_type,
                    )
                    # Two-token bias preserved as a small tie-breaker
                    # bonus (NOT the primary sort key). PERSON names are
                    # almost always 2 tokens; 3+ token phrases are typically
                    # role descriptions ("Texas Private Investigator") or
                    # multi-word titles, less likely to be bridges.
                    n_tokens = len(phrase.split())
                    if n_tokens == 2:
                        score += 0.10
                    elif n_tokens >= 4:
                        score -= 0.20
                    scored.append((score, phrase))
        # Sort descending by score; preserve stable order for ties.
        scored.sort(key=lambda x: -x[0])
        return [phrase for _, phrase in scored[:3]]

    @staticmethod
    def _rewrite_hop_query_with_bridges(
        sub_query: str, bridges: List[str],
    ) -> str:
        """
        §12.37 Fix F: inject resolved bridge entities into a hop's sub-query.

        Iterative multi-hop retrieval relies on each hop being able to find
        its supporting paragraph in the index. Planner-generated sub-queries
        are written before retrieval runs, so Hop-2 sub-queries are usually
        under-specified (e.g. "What is the population?" with no entity).
        After Hop-1 resolves a bridge entity (e.g. "Strasbourg"), Hop-2 must
        be rewritten to "What is the population of Strasbourg?" before the
        retriever sees it — otherwise the retriever returns random
        population-discussing chunks.

        Heuristic (conservative — only fires when SAFE):
          - If sub_query already mentions any of the bridge entities (case-
            insensitive), do nothing (no double-injection).
          - Otherwise append `" — about <bridge_1>, <bridge_2>"` to the
            sub-query. Vector retrievers tolerate this format well; BM25
            picks up the entity tokens; graph search uses the entity names
            from `current_hints` regardless.

        Refs:
          - Query rewriting in iterative retrieval: IRCoT (Trivedi et al.,
            2023, ACL, arXiv:2212.10509).
          - Sub-question augmentation: DSP / Self-Ask (Khattab et al., 2022,
            arXiv:2212.14024; Press et al., 2022, EMNLP).
        """
        if not bridges or not sub_query:
            return sub_query
        sq_lower = sub_query.lower()
        # Already mentioned? skip (avoid "...of Strasbourg of Strasbourg")
        new_bridges = [b for b in bridges if b.lower() not in sq_lower]
        if not new_bridges:
            return sub_query
        # Conservative join: "X — about Y, Z". Does not assume any verb form.
        # The em-dash is rare in natural text → minimal risk of confusing
        # entity extractors downstream.
        injection = ", ".join(new_bridges[:3])  # cap at 3 to keep query short
        rewritten = f"{sub_query.rstrip(' ?.')} — about {injection}"
        logger.info(
            "[S_N Iterative] Rewrote sub-query: %r → %r",
            sub_query[:60], rewritten[:80],
        )
        return rewritten

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

        Reference: iterative multi-hop retrieval; thesis Section 4.2
        (TECHNICAL_ARCHITECTURE.md, pending implementation plan).
        Trivedi et al. (2022). "Interleaving Retrieval with Chain-of-Thought
        Reasoning for Knowledge-Intensive Multi-Step Questions." ACL 2023.
        """
        accumulated_raw: List[str] = []
        accumulated_context: List[str] = []
        seen_raw: set = set()
        seen_filtered: set = set()
        current_hints = list(entity_names)
        # §12.37 Fix F: track resolved bridge entities so subsequent hops can
        # inject them into their sub_query before retrieval. Without this,
        # Hop-2 sub-queries like "What is the population?" go to the
        # retriever as-is and return irrelevant chunks because the bridge
        # entity (e.g. "Strasbourg", "Sela Ward") is known but not used.
        # Refs:
        #   - Iterative multi-hop retrieval: IRCoT (Trivedi et al., 2023, ACL,
        #     arXiv:2212.10509) — feeds retrieved entities back into the
        #     next-hop query.
        #   - Query rewriting with resolved entities: HippoRAG (Gutiérrez et
        #     al., 2024, NeurIPS) — uses personalized-PageRank seeds from
        #     hop-1 to guide hop-2 retrieval.
        resolved_bridges: List[str] = []

        hops = sorted(hop_sequence_raw, key=lambda h: h.get("step_id", 0))

        for hop in hops:
            sub_query = hop.get("sub_query", state["query"])
            is_bridge = hop.get("is_bridge", False)
            step_id = hop.get("step_id", 0)

            # §12.37 Fix F: rewrite this hop's sub-query by injecting the
            # bridge entities resolved in earlier hops. Only fires when:
            #   1. We have at least one resolved bridge entity from prior hops
            #   2. The bridge entity is NOT already mentioned in the sub_query
            #   3. The sub_query depends on at least one prior bridge hop
            depends_on = hop.get("depends_on") or []
            if resolved_bridges and depends_on:
                sub_query = self._rewrite_hop_query_with_bridges(
                    sub_query, resolved_bridges
                )

            logger.info(
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
                nav_result: NavigatorResult = self.navigator.navigate(
                    retrieval_plan=plan,
                    sub_queries=[sub_query],
                    entity_names=current_hints,
                )
            except Exception as e:
                logger.error("[S_N Iterative] Step %d failed: %s", step_id, e)
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

            # After a bridge step: discover the bridge entity name.
            # All filtered chunks are used (not just first 2) so that bridge
            # entities in lower-ranked chunks are not missed.
            if is_bridge and nav_result.filtered_context:
                # Pass the ORIGINAL query (not the sub-query) so the
                # type-detection picks up "who"/"where"/"the actress who"
                # from the user's surface phrasing rather than from the
                # planner-generated stub.
                bridge_entities = self._extract_bridge_entities(
                    nav_result.filtered_context,
                    exclude=current_hints,
                    query=state["query"],
                )
                if bridge_entities:
                    logger.info(
                        "[S_N Iterative] Bridge entities discovered: %s",
                        bridge_entities,
                    )
                    current_hints = current_hints + bridge_entities
                    # §12.37 Fix F: remember bridge entities so the NEXT hop
                    # can inject them into its sub_query (see top of loop).
                    resolved_bridges = resolved_bridges + bridge_entities

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            "[S_N Iterative] %d steps done — %d raw / %d filtered chunks, %.1f ms",
            len(hops), len(accumulated_raw), len(accumulated_context), elapsed,
        )

        errors: List[str] = list(state.get("errors", []))
        if not accumulated_context:
            # All hops failed or produced no results — surface in state.errors so
            # S_V and callers can detect the failure rather than silently receiving
            # an empty context with no explanation.
            errors.append(
                f"Iterative navigator: all {len(hops)} hop(s) produced no context"
            )
            logger.warning("[S_N Iterative] All hops produced empty context")

        return {
            "raw_context": accumulated_raw,
            "context": accumulated_context,
            # retrieval_scores: per-hop RRF scores are not aggregated across steps;
            # downstream receives [] for iterative multi-hop queries.
            # Future work: accumulate nav_result.scores per step.
            "retrieval_scores": [],
            "retrieval_metadata": {"iterative_hints": current_hints, "hop_count": len(hops)},
            # Propagate accumulated entity hints (original + bridge) so S_V uses
            # the full entity set for credibility scoring and entity-path validation.
            "entities": current_hints,
            "errors": errors,
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
        logger.info("--- [S_N NAVIGATOR] Hybrid Retrieval + Filtering ---")

        start_time = time.time()

        if self.navigator.retriever is None:
            logger.warning("[S_N] No retriever set — returning empty context")
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
                logger.info(
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

            nav_result: NavigatorResult = self.navigator.navigate(
                retrieval_plan=plan,
                sub_queries=state["sub_queries"],
                entity_names=entity_names,
            )

            logger.info("[S_N] Raw context: %d chunks", len(nav_result.raw_context))
            logger.info(
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
            logger.error("[S_N] Error: %s", e, exc_info=True)
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
        logger.info("--- [S_V VERIFIER] Pre-Validation + Generation ---")

        start_time = time.time()

        try:
            # Extract hop sequence for pre-validation (may be None)
            plan_dict = state.get("retrieval_plan", {})
            hop_sequence = plan_dict.get("hop_sequence") if plan_dict else None

            # Derive bridge entities: everything added to the entity list during
            # iterative navigation that was not in the original planner output.
            all_entities = state.get("entities", [])
            original_entity_names = {
                e.get("text", "") if isinstance(e, dict) else str(e)
                for e in (plan_dict.get("entities") or [])
            }
            bridge_entities = [
                e for e in all_entities
                if e not in original_entity_names
            ]

            result = self.verifier.generate_and_verify(
                query=state["query"],
                context=state["context"],
                entities=all_entities,
                hop_sequence=hop_sequence,
                query_type=plan_dict.get("query_type") if plan_dict else None,
                bridge_entities=bridge_entities or None,
            )

            logger.info("[S_V] Iterations: %d", result.iterations)
            logger.info("[S_V] All verified: %s", result.all_verified)
            logger.info("[S_V] Verified claims: %d", len(result.verified_claims))
            logger.info("[S_V] Violated claims: %d", len(result.violated_claims))

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
            logger.error("[S_V] Error: %s", e, exc_info=True)
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
        logger.info("--- Pipeline start ---")
        # Fix run-together tokens from noisy datasets (e.g. "in2014" → "in 2014").
        query = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", query)
        query = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", query)
        logger.info("Query: %s", query)

        start_time = time.time()

        if self.app is not None:
            final_state = self.app.invoke(self._make_initial_state(query))
        else:
            final_state = self._run_simple_pipeline(query)

        total_time = (time.time() - start_time) * 1000
        final_state["total_time_ms"] = total_time

        logger.info(
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
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    max_iterations: Optional[int] = None,
    relevance_threshold: Optional[float] = None,
    redundancy_threshold: Optional[float] = None,
    max_chars_per_doc: Optional[int] = None,
) -> AgenticController:
    """
    Factory function for AgenticController.

    When ``cfg`` is None, ``config/settings.yaml`` is auto-loaded from the
    project root — so a bare ``create_controller()`` call always picks up the
    live settings.yaml values without any hardcoded fallbacks in the call site.

    Optional keyword arguments are **overrides** applied after loading from
    settings.yaml (useful for testing or single-parameter ablations).  They
    are only applied when explicitly passed (i.e. not None); otherwise the
    value from settings.yaml is used for every field.

    For evaluation runs, passing ``cfg`` directly is recommended so all
    fifteen ControllerConfig fields — including corroboration weights,
    contradiction thresholds, rrf_k, and top_k_per_subquery — are sourced
    from the same settings.yaml rather than only the six covered by the
    keyword arguments.

    Args:
        cfg:                  Full settings.yaml dict.  Auto-loaded when None.
        model_name:           Override for llm.model_name.
        base_url:             Override for llm.base_url.
        max_iterations:       Override for agent.max_verification_iterations.
        relevance_threshold:  Override for navigator.relevance_threshold_factor.
        redundancy_threshold: Override for navigator.redundancy_threshold.
        max_chars_per_doc:    Override for llm.max_chars_per_doc.

    Returns:
        Configured pipeline controller instance.
    """
    if cfg is None:
        cfg = _load_settings()

    config = ControllerConfig.from_yaml(cfg)

    # Apply explicit keyword overrides (only when the caller passed a value).
    if model_name is not None:
        config.model_name = model_name
    if base_url is not None:
        config.base_url = base_url
    if max_iterations is not None:
        config.max_verification_iterations = max_iterations
    if relevance_threshold is not None:
        config.relevance_threshold_factor = relevance_threshold
    if redundancy_threshold is not None:
        config.redundancy_threshold = redundancy_threshold
    if max_chars_per_doc is not None:
        config.max_chars_per_doc = max_chars_per_doc

    return AgenticController(config=config, full_cfg=cfg)


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

    # No keyword overrides needed — all values come from config/settings.yaml.
    controller = create_controller()

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
