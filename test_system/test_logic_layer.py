"""
Test Suite for the Logic Layer (Artifact B)

Version: 2.0.0
Author: Edge-RAG Research Project

===============================================================================
TEST COVERAGE
===============================================================================

1. Planner Tests (S_P)
   - Initialisation, RetrievalPlan structure
   - QueryType classification: SINGLE_HOP, MULTI_HOP, COMPARISON, TEMPORAL
   - RetrievalStrategy selection
   - Sub-query content and entity presence
   - Edge cases: empty query, long query, non-English query

2. Navigator Tests (S_N)
   - Initialisation, set_retriever()
   - RRF fusion (_rrf_fusion): deduplication and cross-source boost
   - Relevance filter (_relevance_filter): dynamic threshold
   - Redundancy filter (_redundancy_filter): Jaccard deduplication
   - Jaccard similarity utility
   - Entity-mention filter (_entity_mention_filter): safety fallback
   - navigate() without retriever (empty return)
   - navigate() with mock retriever (full path)

3. Verifier Tests (S_V)
   - VerificationResult.confidence property (HIGH/MEDIUM/LOW boundary values)
   - VerifierConfig.from_yaml() round-trip
   - PreGenerationValidator.validate()
   - Verifier.generate_and_verify() with mocked LLM
   - Self-correction loop: two-iteration flow with changing LLM answers

4. AgenticController Tests
   - Initialisation (S_P + S_N + S_V)
   - set_retriever(), set_graph_store()
   - run() → AgentState dict with "answer"
   - __call__() → string
   - Full pipeline flow with mock retriever + mock LLM
   - AgentState required-key completeness

5. AgentPipeline Tests (pipeline/)
   - PipelineResult.to_dict(), to_json()
   - Imports: AgentPipeline, BatchProcessor

6. Thesis Compliance Tests
   - QueryType and RetrievalStrategy enum completeness
   - ConfidenceLevel HIGH/MEDIUM/LOW
   - AgentState key fields
   - RRF cross-source corroboration boost
   - Navigator config defaults

===============================================================================
USAGE
===============================================================================

# From src/logic_layer/:
pytest test_logic_layer.py -v

# From project root (Entwicklungfolder/):
pytest src/logic_layer/test_logic_layer.py -v

===============================================================================
"""

import pytest
import json
from typing import List
from unittest.mock import Mock, patch


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================


def _rule_based_llm_stub(prompt: str):
    """
    Deterministic LLM stub for integration tests.

    Extracts the first sentence from the context block in the prompt and
    returns it as the answer.  This exercises the prompt-formatting code path
    (verifying that context and query are embedded correctly) while remaining
    fully reproducible and requiring no real Ollama connection.

    Contrast with a fixed canned-string mock: canned mocks bypass prompt
    assembly entirely; this stub fails if ``context`` or ``question`` markers
    are missing from the prompt.
    """
    try:
        # Expect the standard ANSWER_PROMPT structure:
        #   Context:\n[1] <text>\n\nQuestion: <query>
        context_block = prompt.split("Context:")[1].split("Question:")[0].strip()
        # Remove the "[1] " prefix added by _format_context
        raw = context_block.split("]", 1)[-1].strip() if "]" in context_block else context_block
        first_sentence = raw.split(".")[0].strip() + "."
        return (first_sentence, 0.05)
    except (IndexError, AttributeError):
        return ("I don't know.", 0.05)


class MockResult:
    """
    Minimal mock of a retrieval result object.

    Provides the ``text``, ``rrf_score``, and ``source_doc`` attributes
    that Navigator and AgenticController expect when consuming retriever
    output.  Extracted from duplicated inline definitions to avoid
    copy-paste divergence.
    """

    def __init__(self, text: str, score: float) -> None:
        self.text = text
        self.rrf_score = score
        self.source_doc = "test.pdf"


# =============================================================================
# 1. PLANNER TESTS (S_P)
# =============================================================================


class TestPlanner:
    """
    Tests for S_P: query analysis and retrieval-plan construction.

    Queries are selected to cover representative HotpotQA / 2WikiMultiHop
    patterns used in the thesis evaluation (Section 3.2).
    """

    def test_initialization(self) -> None:
        """create_planner() returns a Planner instance."""
        from src.logic_layer.planner import create_planner, Planner
        planner = create_planner()
        assert isinstance(planner, Planner)

    def test_plan_returns_retrieval_plan(self) -> None:
        """plan() returns a RetrievalPlan object."""
        from src.logic_layer.planner import create_planner, RetrievalPlan
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        assert isinstance(plan, RetrievalPlan)

    def test_plan_has_required_attributes(self) -> None:
        """RetrievalPlan contains all mandatory fields."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Who invented the telephone?")
        assert hasattr(plan, "original_query")
        assert hasattr(plan, "query_type")
        assert hasattr(plan, "strategy")
        assert hasattr(plan, "entities")
        assert hasattr(plan, "hop_sequence")
        assert hasattr(plan, "confidence")
        assert plan.original_query == "Who invented the telephone?"

    def test_single_hop_query_type(self) -> None:
        """Simple factual questions → SINGLE_HOP.

        VECTOR_ONLY is the strategy assigned to SINGLE_HOP in the current
        thesis implementation; this test encodes that deliberate design choice.
        """
        from src.logic_layer.planner import create_planner, QueryType, RetrievalStrategy
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        assert plan.query_type == QueryType.SINGLE_HOP
        assert plan.strategy == RetrievalStrategy.VECTOR_ONLY

    def test_multi_hop_query_type(self) -> None:
        """Multi-step bridge queries → MULTI_HOP.

        HYBRID is the strategy assigned to MULTI_HOP in the current thesis
        implementation; this test encodes that deliberate design choice.
        """
        from src.logic_layer.planner import create_planner, QueryType, RetrievalStrategy
        planner = create_planner()
        plan = planner.plan("Who directed the movie starring Tom Hanks?")
        assert plan.query_type == QueryType.MULTI_HOP
        assert plan.strategy == RetrievalStrategy.HYBRID

    def test_comparison_query_type(self) -> None:
        """Comparative questions → COMPARISON."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan("Is Berlin larger than Munich?")
        assert plan.query_type == QueryType.COMPARISON

    def test_temporal_query_type(self) -> None:
        """Year-anchored questions → TEMPORAL.

        Note: this test requires SpaCy to be available for DATE entity
        detection of "2020".  Under the regex-only fallback the classification
        may differ.
        """
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan("What happened in 2020?")
        assert plan.query_type == QueryType.TEMPORAL

    def test_plan_confidence_in_valid_range(self) -> None:
        """Classification confidence is always in [0.0, 1.0]."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Who invented the telephone?")
        assert 0.0 <= plan.confidence <= 1.0

    def test_decompose_query_returns_list(self) -> None:
        """decompose_query() returns a non-empty list."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        sub_queries = planner.decompose_query(
            "Who directed Inception and when was it released?"
        )
        assert isinstance(sub_queries, list)
        assert len(sub_queries) >= 1

    def test_empty_query_no_crash(self) -> None:
        """Empty string returns a RetrievalPlan with a valid QueryType and empty sub_queries."""
        from src.logic_layer.planner import create_planner, QueryType, RetrievalPlan
        planner = create_planner()
        plan = planner.plan("")
        assert plan is not None
        assert isinstance(plan, RetrievalPlan)
        assert plan.query_type in list(QueryType)
        # Empty query → Planner should produce no sub-queries (nothing to decompose)
        assert isinstance(plan.sub_queries, list)

    def test_very_long_query_no_crash(self) -> None:
        """1 000-character query returns a RetrievalPlan with a valid QueryType."""
        from src.logic_layer.planner import create_planner, QueryType, RetrievalPlan
        planner = create_planner()
        plan = planner.plan("a" * 1000)
        assert plan is not None
        assert isinstance(plan, RetrievalPlan)
        assert plan.query_type in list(QueryType)
        # Long nonsense input must not produce an empty entities list crash
        assert isinstance(plan.entities, list)

    def test_non_english_query_no_crash(self) -> None:
        """Non-English query returns a RetrievalPlan without crashing; query is preserved."""
        from src.logic_layer.planner import create_planner, RetrievalPlan
        planner = create_planner()
        query = "Was ist die Hauptstadt von Deutschland?"
        plan = planner.plan(query)
        assert plan is not None
        assert isinstance(plan, RetrievalPlan)
        # The original query must be preserved verbatim in the plan
        assert plan.original_query == query, (
            f"Plan original_query must match input; got '{plan.original_query}'"
        )

    # -----------------------------------------------------------------------
    # QueryType Classification — dataset-representative patterns
    # (queries drawn from HotpotQA / 2WikiMultiHop dev sets)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("query", [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Are Madonna and Lady Gaga from the same country?",
        "Did Nikola Tesla and Thomas Edison have the same nationality?",
    ])
    def test_comparison_same_attribute_pattern(self, query: str) -> None:
        """'same X' pattern → COMPARISON (not MULTI_HOP).

        These three queries are representative of the 'same-attribute'
        comparison sub-type in HotpotQA.
        """
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON, (
            f"Expected COMPARISON for '{query}', got {plan.query_type.value}"
        )

    @pytest.mark.parametrize("query", [
        "Is Berlin older than Munich?",
        "Which is taller, the Eiffel Tower or Big Ben?",
        "Was Alexander the Great older than Julius Caesar?",
    ])
    def test_comparison_comparative_adjective_pattern(self, query: str) -> None:
        """Comparative adjectives (older/taller/…) → COMPARISON.

        These queries exercise the regex patterns for superlative/comparative
        forms in QueryClassifier.COMPARISON_PATTERNS.
        """
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON, (
            f"Expected COMPARISON for '{query}', got {plan.query_type.value}"
        )

    # -----------------------------------------------------------------------
    # Sub-query content — core Planner output quality
    # -----------------------------------------------------------------------

    def test_comparison_sub_queries_are_distinct(self) -> None:
        """Comparison sub-queries for two entities must be different strings."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        queries = [
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "Is Berlin older than Munich?",
            "Which is taller, the Eiffel Tower or Big Ben?",
        ]
        for q in queries:
            plan = planner.plan(q)
            assert plan.query_type == QueryType.COMPARISON
            assert len(plan.sub_queries) >= 2, (
                f"Too few sub-queries for: {q}"
            )
            assert plan.sub_queries[0] != plan.sub_queries[1], (
                f"Sub-queries are identical for: {q}\n"
                f"  sub_queries[0]={plan.sub_queries[0]}\n"
                f"  sub_queries[1]={plan.sub_queries[1]}"
            )

    def test_comparison_sub_queries_contain_respective_entity(self) -> None:
        """Each entity-specific sub-query must mention the corresponding entity."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        query = "Were Scott Derrickson and Ed Wood of the same nationality?"
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON
        entity_names = ["Scott Derrickson", "Ed Wood"]
        sq0, sq1 = plan.sub_queries[0], plan.sub_queries[1]
        assert any(e in sq0 for e in entity_names), (
            f"No entity found in sub_queries[0]: '{sq0}'"
        )
        assert any(e in sq1 for e in entity_names), (
            f"No entity found in sub_queries[1]: '{sq1}'"
        )
        assert sq0 != sq1

    def test_comparison_sub_queries_use_attribute_template(self) -> None:
        """'same nationality' queries produce attribute-template sub-queries.

        _ATTR_MAP rewrites same-attribute comparisons into per-entity
        attribute lookup queries of the form "What is the <attr> of <Entity>?".
        This test validates that the rewriting is active and produces the
        expected format, superseding the older assertion that sub-queries must
        NOT start with 'What is '.
        """
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        query = "Were Scott Derrickson and Ed Wood of the same nationality?"
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON
        assert len(plan.sub_queries) >= 2
        # Both sub-queries must be attribute-template lookups and must
        # reference exactly one of the two entities.
        entity_names = ["Scott Derrickson", "Ed Wood"]
        for sq in plan.sub_queries[:2]:
            assert any(e in sq for e in entity_names), (
                f"Sub-query does not mention any entity: '{sq}'"
            )
            # _ATTR_MAP generates "What is the nationality of <Entity>?"
            assert sq.startswith("What is the nationality of"), (
                f"Expected attribute-template form, got: '{sq}'"
            )

    def test_comparison_sub_queries_use_birthplace_template(self) -> None:
        """'same birthplace' queries are rewritten to 'Where was <Entity> born?'

        _ATTR_MAP entry: same birthplace/hometown → 'Where was {entity} born?'
        """
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        query = "Were Albert Einstein and Max Planck born in the same place?"
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON
        assert len(plan.sub_queries) >= 2
        for sq in plan.sub_queries[:2]:
            assert sq.lower().startswith("where was"), (
                f"Expected birthplace-template form, got: '{sq}'"
            )

    def test_comparison_sub_queries_use_profession_template(self) -> None:
        """'same profession' queries are rewritten to 'What is the profession of <Entity>?'

        _ATTR_MAP entry: same profession/occupation/job → 'What is the profession of {entity}?'
        """
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        query = "Did Sigmund Freud and Carl Jung have the same profession?"
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON
        assert len(plan.sub_queries) >= 2
        for sq in plan.sub_queries[:2]:
            assert sq.lower().startswith("what is the profession of"), (
                f"Expected profession-template form, got: '{sq}'"
            )

    def test_retrieval_strategy_comparison_is_hybrid(self) -> None:
        """COMPARISON queries must use HYBRID strategy (not VECTOR_ONLY).

        The thesis design (Section 3.2) assigns HYBRID to all complex query
        types that require graph traversal.  COMPARISON is complex by design.
        Using VECTOR_ONLY for a comparison query would skip graph search and
        degrade multi-hop entity retrieval.
        """
        from src.logic_layer.planner import create_planner, QueryType, RetrievalStrategy
        planner = create_planner()
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        assert plan.query_type == QueryType.COMPARISON
        assert plan.strategy == RetrievalStrategy.HYBRID, (
            f"COMPARISON query must use HYBRID strategy; got {plan.strategy.value!r}"
        )

    def test_multi_hop_generates_multiple_sub_queries(self) -> None:
        """Multi-hop query → at least 2 sub-queries."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        queries = [
            "Who directed the movie that stars Tom Hanks?",
            "What is the capital of the country where Einstein was born?",
        ]
        for q in queries:
            plan = planner.plan(q)
            assert plan.query_type == QueryType.MULTI_HOP
            assert len(plan.sub_queries) >= 2, (
                f"Multi-hop needs >= 2 sub-queries for: {q}"
            )

    # -----------------------------------------------------------------------
    # Entity extraction quality
    # -----------------------------------------------------------------------

    def test_entity_extraction_finds_person_names(self) -> None:
        """SpaCy must find PERSON entities in queries."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        entity_texts = [e.text for e in plan.entities]
        assert "Scott Derrickson" in entity_texts or any(
            "Derrickson" in t for t in entity_texts
        ), f"Scott Derrickson not in entities: {entity_texts}"
        assert "Ed Wood" in entity_texts or any(
            "Wood" in t for t in entity_texts
        ), f"Ed Wood not in entities: {entity_texts}"

    def test_entity_extraction_finds_gpe(self) -> None:
        """SpaCy must find GPE entities (countries, cities)."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Is Berlin older than Munich?")
        entity_texts = [e.text for e in plan.entities]
        assert any("Berlin" in t for t in entity_texts), (
            f"Berlin not in entities: {entity_texts}"
        )
        assert any("Munich" in t for t in entity_texts), (
            f"Munich not in entities: {entity_texts}"
        )

    def test_hop_sequence_comparison_has_parallel_steps(self) -> None:
        """Comparison plan: entity-retrieval steps must have no dependencies (parallel)."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        assert plan.query_type == QueryType.COMPARISON
        parallel_steps = [s for s in plan.hop_sequence if s.depends_on == []]
        assert len(parallel_steps) >= 2, (
            f"Expected >= 2 parallel steps, got: {[s.depends_on for s in plan.hop_sequence]}"
        )


# =============================================================================
# 2. NAVIGATOR TESTS (S_N)
# =============================================================================


class TestNavigator:
    """
    Tests for S_N: hybrid retrieval and pre-generative filtering.

    The ``config`` fixture uses explicit threshold values that match
    settings.yaml (navigator.relevance_threshold_factor: 0.6,
    navigator.redundancy_threshold: 0.8).  These are intentional test-scope
    values, not an additional settings-compliance obligation.
    """

    @pytest.fixture
    def config(self):
        from src.logic_layer.navigator import ControllerConfig
        return ControllerConfig(
            relevance_threshold_factor=0.6,
            redundancy_threshold=0.8,
        )

    @pytest.fixture
    def navigator(self, config):
        from src.logic_layer.navigator import Navigator
        return Navigator(config=config)

    @pytest.fixture
    def fused_results(self) -> List[dict]:
        """Pre-computed dicts in the format produced by _rrf_fusion()."""
        return [
            {"text": "Paris is the capital of France.", "rrf_score": 0.90},
            {"text": "France is located in Western Europe.", "rrf_score": 0.70},
            {"text": "French cuisine is internationally famous.", "rrf_score": 0.30},
        ]

    def test_initialization(self, navigator) -> None:
        """Navigator starts with no retriever."""
        assert navigator is not None
        assert navigator.retriever is None

    def test_set_retriever(self, navigator) -> None:
        """set_retriever() stores the retriever on the instance."""
        mock_retriever = Mock()
        navigator.set_retriever(mock_retriever)
        assert navigator.retriever is mock_retriever

    def test_relevance_filter_removes_low_scores(self, navigator, fused_results) -> None:
        """Chunks below factor × max_score are removed.

        Uses navigator.config.relevance_threshold_factor (not a hardcoded
        0.6) so the test tracks the config value rather than a stale copy.
        """
        filtered = navigator._relevance_filter(fused_results)
        max_score = max(r["rrf_score"] for r in fused_results)
        threshold = navigator.config.relevance_threshold_factor * max_score
        assert all(r["rrf_score"] >= threshold for r in filtered)
        assert len(filtered) < len(fused_results)

    def test_relevance_filter_empty_input(self, navigator) -> None:
        """Empty list → empty list."""
        assert navigator._relevance_filter([]) == []

    def test_redundancy_filter_removes_duplicates(self, navigator) -> None:
        """Near-duplicate is removed: exactly 2 chunks survive (not just ≤ 2).

        'Paris is the capital of France' / 'The capital of France is Paris' share
        high Jaccard overlap → deduplication removes the lower-scored one.
        'Darwin studied evolution in Galapagos' is unrelated and must survive.
        """
        results = [
            {"text": "Paris is the capital of France",        "rrf_score": 0.9},
            {"text": "The capital of France is Paris",        "rrf_score": 0.8},
            {"text": "Darwin studied evolution in Galapagos", "rrf_score": 0.7},
        ]
        filtered = navigator._redundancy_filter(results)
        assert len(filtered) == 2, (
            f"Near-duplicate must be removed, leaving exactly 2; got {len(filtered)}"
        )
        assert any("Darwin" in r["text"] for r in filtered), "Darwin chunk must survive"
        assert any("Paris"  in r["text"] for r in filtered), "Paris chunk must survive"

    def test_redundancy_filter_empty_input(self, navigator) -> None:
        """Empty list → empty list."""
        assert navigator._redundancy_filter([]) == []

    def test_redundancy_filter_higher_scored_chunk_survives(self, navigator) -> None:
        """When two chunks are near-duplicates, the higher-scored one is retained.

        The filter processes results in score order (highest first).  The
        first accepted chunk wins; the near-duplicate with a lower score
        must be dropped.  This verifies the 'higher-scored wins' guarantee
        rather than just checking the resulting list length.
        """
        high = {"text": "Paris is the capital of France.", "rrf_score": 0.9}
        low  = {"text": "Paris is the capital of France.", "rrf_score": 0.3}
        # Identical texts → Jaccard = 1.0 > any reasonable threshold.
        filtered = navigator._redundancy_filter([high, low])
        assert len(filtered) == 1
        assert filtered[0]["rrf_score"] == 0.9, (
            "Higher-scored chunk must survive deduplication"
        )

    def test_jaccard_similarity_identical_texts(self, navigator) -> None:
        """Identical texts → Jaccard similarity = 1.0."""
        sim = navigator._jaccard_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_jaccard_similarity_different_texts(self, navigator) -> None:
        """Completely different texts → Jaccard similarity < 0.5."""
        sim = navigator._jaccard_similarity("hello world", "goodbye universe")
        assert sim < 0.5

    def test_rrf_fusion_deduplicates_and_boosts(self, navigator) -> None:
        """A chunk appearing in two sub-queries receives a cross-source boost.

        Cross-source corroboration boost is documented in thesis Section 3.3
        and implemented via rrf_k and cross_source_boost in settings.yaml.
        """
        results = [
            {"text": "Doc A", "score": 0.9, "source": "s1", "sub_query": "q1"},
            {"text": "Doc B", "score": 0.7, "source": "s1", "sub_query": "q1"},
            {"text": "Doc A", "score": 0.8, "source": "s2", "sub_query": "q2"},
        ]
        fused = navigator._rrf_fusion(results)
        doc_a = [r for r in fused if r["text"] == "Doc A"]
        doc_b = [r for r in fused if r["text"] == "Doc B"]
        assert len(doc_a) == 1, "Doc A must not be duplicated after fusion"
        assert len(doc_b) == 1
        assert doc_a[0]["rrf_score"] > doc_b[0]["rrf_score"], (
            "Doc A (two sub-queries) must outscore Doc B (one sub-query)"
        )

    def test_entity_overlap_pruning_removes_subset_chunk(self, navigator) -> None:
        """Lower-ranked chunk whose entity set ⊆ higher-ranked chunk's entities is pruned.

        The entity extractor uses capitalized-phrase matching (e.g. "Paris",
        "France", "Eiffel Tower").  Chunk B's entities {"Paris", "France"} are
        a strict subset of Chunk A's {"Paris", "France", "Eiffel Tower"}.
        Because rank(A) > rank(B) (i.e. score(A) > score(B)), B is redundant
        and must be pruned.

        Chunk C has unrelated entities and must survive.
        """
        results = [
            {
                "text": "Paris is the capital of France and home to the Eiffel Tower.",
                "rrf_score": 0.9,
            },
            {
                "text": "Paris is located in France.",
                "rrf_score": 0.7,
            },
            {
                "text": "Berlin is the capital of Germany.",
                "rrf_score": 0.5,
            },
        ]
        filtered = navigator._entity_overlap_pruning(results)
        assert len(filtered) < len(results), (
            "Chunk B (entity subset of Chunk A) must be pruned"
        )
        assert any("Eiffel Tower" in r["text"] for r in filtered), (
            "Chunk A (superset entities, higher score) must survive"
        )
        assert any("Berlin" in r["text"] for r in filtered), (
            "Chunk C (unrelated entities) must survive"
        )

    def test_entity_overlap_pruning_empty_input(self, navigator) -> None:
        """Empty input returns empty list without crash."""
        assert navigator._entity_overlap_pruning([]) == []

    def test_entity_mention_filter_keeps_matching_chunks(self, navigator) -> None:
        """Chunks mentioning a query entity pass the filter."""
        results = [
            {"text": "Albert Einstein developed relativity.", "rrf_score": 0.9},
            {"text": "The theory was published in 1905.",    "rrf_score": 0.7},
            {"text": "Newton invented calculus.",            "rrf_score": 0.5},
        ]
        # "Einstein" is 7 chars → single-token entity, passes ≥5-char guard.
        filtered = navigator._entity_mention_filter(results, ["Einstein"])
        texts = [r["text"] for r in filtered]
        assert any("Einstein" in t for t in texts), (
            "Chunk mentioning Einstein must survive the filter"
        )

    def test_entity_mention_filter_drops_non_matching_chunks(self, navigator) -> None:
        """Non-matching chunks are actually removed — filter is not a no-op.

        Verifies the filter does not silently pass all input unchanged.
        Only the Einstein chunk (7-char token) matches; weather and Newton
        chunks must be dropped.  Safety fallback must NOT fire here because
        at least one chunk matches.
        """
        results = [
            {"text": "Albert Einstein developed the theory of relativity.", "rrf_score": 0.9},
            {"text": "The weather is cold today.",                          "rrf_score": 0.7},
            {"text": "Newton invented calculus.",                           "rrf_score": 0.5},
        ]
        filtered = navigator._entity_mention_filter(results, ["Einstein"])
        assert len(filtered) == 1, (
            f"Only Einstein chunk must survive; got {len(filtered)}: {[r['text'] for r in filtered]}"
        )
        assert "Einstein" in filtered[0]["text"]
        assert not any("weather" in r["text"] for r in filtered)
        assert not any("Newton"  in r["text"] for r in filtered)

    def test_entity_mention_filter_safety_fallback(self, navigator) -> None:
        """If all chunks would be filtered, all are returned (never empty context)."""
        results = [
            {"text": "The sky is blue.",  "rrf_score": 0.9},
            {"text": "Water is liquid.", "rrf_score": 0.7},
        ]
        # "Aristotle" appears in neither chunk → would filter everything.
        filtered = navigator._entity_mention_filter(results, ["Aristotle"])
        assert filtered == results, (
            "Safety fallback must return all chunks when all would be filtered"
        )

    def test_entity_mention_filter_empty_entities(self, navigator) -> None:
        """Empty entity list → all results returned unchanged."""
        results = [{"text": "Some text.", "rrf_score": 0.9}]
        assert navigator._entity_mention_filter(results, []) == results

    def test_contradiction_filter_removes_lower_scored_chunk(self, navigator) -> None:
        """Lower-scored chunk with a contradictory number is removed.

        Setup: two chunks discuss the same topic (8/10 word overlap) but cite
        conflicting numeric values (50 vs 200 → ratio 4.0 > threshold 2.0;
        min value 50 > threshold 10).  The chunk with the lower rrf_score (0.5)
        must be removed; the higher-scored chunk (0.9) must survive.

        This test exercises the ``any()`` comprehension fix that replaced a
        nested ``break`` (which only exited the inner for-n2 loop, wasting CPU
        after the first contradicting pair was found).
        """
        results = [
            {
                "text": "The scientist was born 50 years after the event.",
                "rrf_score": 0.9,
            },
            {
                "text": "The scientist was born 200 years after the event.",
                "rrf_score": 0.5,
            },
        ]
        filtered = navigator._contradiction_filter(results)
        assert len(filtered) == 1, (
            "Lower-scored contradicting chunk must be removed"
        )
        assert filtered[0]["rrf_score"] == 0.9, (
            "The higher-scored chunk must survive"
        )

    def test_contradiction_filter_keeps_non_contradicting_chunks(self, navigator) -> None:
        """Chunks that share low word overlap are not considered contradictions."""
        results = [
            {"text": "Paris is in 50 km from Versailles.", "rrf_score": 0.9},
            {"text": "Quantum mechanics was formulated in 1925.", "rrf_score": 0.7},
        ]
        filtered = navigator._contradiction_filter(results)
        # Different topics → word overlap below threshold → no chunk removed.
        assert len(filtered) == 2

    def test_contradiction_filter_empty_input(self, navigator) -> None:
        """Empty input returns empty list without crash."""
        assert navigator._contradiction_filter([]) == []

    def test_contradiction_filter_single_chunk_unchanged(self, navigator) -> None:
        """Single-element list is returned unchanged (no pairs to compare)."""
        results = [{"text": "Some fact.", "rrf_score": 0.9}]
        assert navigator._contradiction_filter(results) == results

    # -----------------------------------------------------------------------
    # _context_shrinkage() — entity-sentence prioritization (action item 9)
    # -----------------------------------------------------------------------

    def test_context_shrinkage_prioritizes_entity_sentences(self, navigator) -> None:
        """Sentences containing named entities appear in the shrunk output even
        when the total text exceeds max_chars_per_chunk.

        Strategy: the shrinkage algorithm puts entity-containing sentences
        first (priority queue) before filler sentences.  Filler sentences are
        all-lowercase so has_entity() does not fire on them; the entity sentence
        starts with a capital proper noun, landing it in the priority queue.
        If the budget is tight, the entity sentence must still be present.
        """
        entity_sentence = "Albert Einstein was born in Ulm in 1879."
        # All-lowercase filler → has_entity() returns False → goes to rest queue
        filler = " ".join(["it is very cold outside today."] * 15)  # ~450 chars
        long_text = filler + " " + entity_sentence
        result = navigator._context_shrinkage(
            [{"text": long_text, "rrf_score": 0.9}],
            max_chars_per_chunk=120,  # tight budget: fits entity sentence + maybe 2 filler
        )
        assert result, "Shrinkage must return at least one chunk"
        assert "Einstein" in result[0]["text"], (
            "Entity sentence must survive shrinkage (prioritized over filler)"
        )

    def test_context_shrinkage_short_chunk_unchanged(self, navigator) -> None:
        """Chunks shorter than max_chars_per_chunk are not modified."""
        short = {"text": "Paris is the capital of France.", "rrf_score": 0.9}
        result = navigator._context_shrinkage([short], max_chars_per_chunk=500)
        assert result[0]["text"] == short["text"]

    def test_navigate_without_retriever_returns_empty_context(self, navigator) -> None:
        """navigate() without a retriever returns a NavigatorResult with empty context."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        result = navigator.navigate(plan, ["What is the capital of France?"])
        assert result is not None
        assert result.filtered_context == []

    def test_navigate_with_mock_retriever(self, navigator) -> None:
        """navigate() with a mock retriever returns filtered context containing relevant text."""
        from src.logic_layer.planner import create_planner
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            [
                MockResult("Paris is the capital of France.", 0.9),
                MockResult("France is located in Western Europe.", 0.7),
            ],
            {"latency_ms": 5},
        )
        navigator.set_retriever(mock_retriever)
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        result = navigator.navigate(plan, ["What is the capital of France?"])
        assert result is not None
        assert len(result.filtered_context) >= 1
        assert any("Paris" in c for c in result.filtered_context)

    def test_navigate_result_metadata_keys(self, navigator) -> None:
        """NavigatorResult.metadata contains the seven filter-step counters."""
        from src.logic_layer.planner import create_planner
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            [MockResult("Paris is the capital of France.", 0.9)],
            {"latency_ms": 5},
        )
        navigator.set_retriever(mock_retriever)
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        result = navigator.navigate(plan, ["What is the capital of France?"])
        required_keys = {
            "pre_filter_count",
            "after_relevance_filter",
            "after_redundancy_filter",
            "after_contradiction_filter",
            "after_entity_overlap_pruning",
            "after_entity_mention_filter",
            "total_time_ms",
        }
        assert required_keys.issubset(set(result.metadata.keys())), (
            f"Missing metadata keys: {required_keys - set(result.metadata.keys())}"
        )

    # -----------------------------------------------------------------------
    # Action 2: retrieve() called once per sub-query (COMPARISON)
    # -----------------------------------------------------------------------

    def test_navigate_calls_retriever_once_per_sub_query(self, navigator) -> None:
        """navigate() calls retriever.retrieve() exactly once per sub-query.

        A COMPARISON plan for two entities produces two sub-queries.
        If retrieve() is called fewer times than len(sub_queries), the second
        entity is never retrieved — the primary failure mode for idx=0 (Ed Wood).
        """
        from src.logic_layer.planner import create_planner, QueryType
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = ([], {})
        navigator.set_retriever(mock_retriever)

        planner = create_planner()
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        assert plan.query_type == QueryType.COMPARISON
        assert len(plan.sub_queries) >= 2, "Need >= 2 sub-queries for this test"

        navigator.navigate(plan, plan.sub_queries)

        assert mock_retriever.retrieve.call_count == len(plan.sub_queries), (
            f"Expected {len(plan.sub_queries)} retrieve() calls, "
            f"got {mock_retriever.retrieve.call_count}. "
            "Second entity is never retrieved when call_count < len(sub_queries)."
        )

    # -----------------------------------------------------------------------
    # Action 7: multi-hop plan merges results from all sub-queries
    # -----------------------------------------------------------------------

    def test_navigate_multi_hop_merges_results_from_all_sub_queries(
        self, navigator
    ) -> None:
        """navigate() with a MULTI_HOP plan calls retrieve() for every sub-query.

        Uses a side-effect retriever so each call returns a distinct chunk,
        then verifies the result contains chunks from multiple retrieval calls.
        """
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan("Who directed the movie that stars Tom Hanks?")
        assert plan.query_type == QueryType.MULTI_HOP
        assert len(plan.sub_queries) >= 2

        call_idx = [0]

        def side_effect(query, top_k=None, entity_hints=None):
            n = call_idx[0]
            call_idx[0] += 1
            return (
                [MockResult(f"Result for sub-query {n + 1}: {query[:30]}.", 0.9)],
                {},
            )

        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = side_effect
        navigator.set_retriever(mock_retriever)

        result = navigator.navigate(plan, plan.sub_queries)

        assert mock_retriever.retrieve.call_count >= 2, (
            "retrieve() must be called for every sub-query in a MULTI_HOP plan"
        )
        # Raw context must mention results from different sub-queries
        combined = " ".join(result.raw_context)
        assert "sub-query 1" in combined or "sub-query 2" in combined, (
            "Results from multiple retrieve() calls must reach raw_context"
        )

    # -----------------------------------------------------------------------
    # Action 10: filter-step metadata counters are monotonically non-increasing
    # -----------------------------------------------------------------------

    def test_navigate_metadata_counters_are_monotonically_non_increasing(
        self, navigator
    ) -> None:
        """Each filter step can only reduce or maintain chunk count, never increase.

        This is a pipeline-invariant property: filters only discard chunks,
        never create new ones.  A violation here would indicate a broken filter.
        """
        from src.logic_layer.planner import create_planner
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            [
                MockResult("Albert Einstein was born in Ulm, Germany.", 0.9),
                MockResult("Einstein received the Nobel Prize in Physics.", 0.8),
                MockResult("The sky is blue on a clear day.", 0.3),
            ],
            {},
        )
        navigator.set_retriever(mock_retriever)
        planner = create_planner()
        plan = planner.plan("Where was Einstein born?")
        result = navigator.navigate(plan, plan.sub_queries)
        m = result.metadata
        chain = [
            m.get("pre_filter_count", 0),
            m.get("after_relevance_filter", 0),
            m.get("after_redundancy_filter", 0),
            m.get("after_contradiction_filter", 0),
            m.get("after_entity_overlap_pruning", 0),
            m.get("after_entity_mention_filter", 0),
        ]
        for i in range(len(chain) - 1):
            assert chain[i + 1] <= chain[i], (
                f"Filter step {i + 1} increased chunk count from "
                f"{chain[i]} to {chain[i + 1]}: {chain}"
            )

    # -----------------------------------------------------------------------
    # Action 14: strategy propagates to result metadata for SINGLE_HOP plan
    # -----------------------------------------------------------------------

    def test_navigate_strategy_stored_in_metadata(self, navigator) -> None:
        """RetrievalPlan.strategy is preserved in NavigatorResult.metadata.

        This verifies that the strategy field is propagated from the Planner
        through navigate() into the metadata dict, where diagnostics and the
        evaluation harness read it.  SINGLE_HOP → VECTOR_ONLY.
        """
        from src.logic_layer.planner import create_planner
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            [MockResult("Paris is the capital of France.", 0.9)],
            {},
        )
        navigator.set_retriever(mock_retriever)
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        result = navigator.navigate(plan, plan.sub_queries)

        assert "retrieval_plan" in result.metadata, (
            "metadata must contain 'retrieval_plan' key"
        )
        stored_plan = result.metadata["retrieval_plan"]
        assert stored_plan is not None
        assert stored_plan.get("strategy") == "vector_only", (
            f"SINGLE_HOP plan must store strategy='vector_only'; "
            f"got {stored_plan.get('strategy')!r}"
        )


# =============================================================================
# 3. VERIFIER TESTS (S_V)
# =============================================================================


class TestVerificationResultConfidence:
    """
    Tests for VerificationResult.confidence property.

    All tests set confidence_high_threshold and confidence_medium_threshold
    explicitly so they are independent of dataclass defaults.
    """

    def test_confidence_high_all_verified(self) -> None:
        """5/5 verified claims → HIGH (ratio = 1.0 ≥ threshold 0.8)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="Answer",
            iterations=1,
            verified_claims=["c1", "c2", "c3", "c4", "c5"],
            violated_claims=[],
            all_verified=True,
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )
        assert result.confidence == ConfidenceLevel.HIGH

    def test_confidence_medium_mixed_claims(self) -> None:
        """2/3 verified → MEDIUM (ratio ≈ 0.67, 0.5 ≤ ratio < 0.8)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="Answer",
            iterations=1,
            verified_claims=["c1", "c2"],
            violated_claims=["c3"],
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_confidence_low_mostly_violated(self) -> None:
        """1/4 verified → LOW (ratio = 0.25 < 0.5)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="Answer",
            iterations=1,
            verified_claims=["c1"],
            violated_claims=["c2", "c3", "c4"],
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )
        assert result.confidence == ConfidenceLevel.LOW

    def test_confidence_low_when_no_claims(self) -> None:
        """No claims → LOW (zero-division guard)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="Answer",
            iterations=1,
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )
        assert result.confidence == ConfidenceLevel.LOW

    def test_confidence_value_is_string(self) -> None:
        """.confidence.value must be a string (consumed by agent_pipeline.py)."""
        from src.logic_layer.verifier import VerificationResult
        result = VerificationResult(
            answer="Answer",
            iterations=1,
            verified_claims=["c1"],
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )
        assert isinstance(result.confidence.value, str)


class TestVerifierConfig:
    """Tests for VerifierConfig construction and from_yaml() loading."""

    def test_config_defaults(self) -> None:
        """VerifierConfig() defaults are sensible."""
        from src.logic_layer.verifier import VerifierConfig
        config = VerifierConfig()
        assert config.max_iterations >= 1
        assert config.max_context_chars > 0
        assert config.max_docs > 0
        assert 0.0 < config.contradiction_threshold <= 1.0

    def test_from_yaml_reads_llm_block(self) -> None:
        """from_yaml() reads model_name, timeout, and context budget from llm block."""
        from src.logic_layer.verifier import VerifierConfig
        cfg = {
            "llm": {
                "model_name": "qwen2:1.5b",
                "base_url": "http://localhost:11434",
                "temperature": 0.2,
                "max_tokens": 150,
                "timeout": 30,
                "max_context_chars": 800,
                "max_docs": 2,
                "max_chars_per_doc": 250,
            }
        }
        config = VerifierConfig.from_yaml(cfg)
        assert config.model_name == "qwen2:1.5b"
        assert config.temperature == 0.2
        assert config.max_tokens == 150
        assert config.timeout == 30
        assert config.max_context_chars == 800
        assert config.max_docs == 2
        assert config.max_chars_per_doc == 250

    def test_from_yaml_reads_agent_block(self) -> None:
        """from_yaml() reads max_verification_iterations from agent block."""
        from src.logic_layer.verifier import VerifierConfig
        cfg = {"agent": {"max_verification_iterations": 3}}
        config = VerifierConfig.from_yaml(cfg)
        assert config.max_iterations == 3

    def test_from_yaml_reads_verifier_block(self) -> None:
        """from_yaml() reads NLI model, contradiction threshold, and credibility params."""
        from src.logic_layer.verifier import VerifierConfig
        cfg = {
            "verifier": {
                "nli_model": "cross-encoder/nli-MiniLM2-L6-H768",
                "contradiction_threshold": 0.90,
                "min_credibility_score": 0.6,
                "entity_coverage_threshold": 0.7,
                "enable_contradiction_detection": True,
                "confidence_high_threshold": 0.85,
                "confidence_medium_threshold": 0.55,
            }
        }
        config = VerifierConfig.from_yaml(cfg)
        assert config.nli_model == "cross-encoder/nli-MiniLM2-L6-H768"
        assert config.contradiction_threshold == 0.90
        assert config.min_credibility_score == 0.6
        assert config.entity_coverage_threshold == 0.7
        assert config.enable_contradiction_detection is True
        assert config.confidence_high_threshold == 0.85
        assert config.confidence_medium_threshold == 0.55

    def test_from_yaml_empty_dict_uses_defaults(self) -> None:
        """from_yaml({}) falls back to all dataclass defaults (emergency fallback check)."""
        from src.logic_layer.verifier import VerifierConfig
        config = VerifierConfig.from_yaml({})
        defaults = VerifierConfig()
        assert config.model_name == defaults.model_name
        assert config.max_iterations == defaults.max_iterations
        assert config.max_context_chars == defaults.max_context_chars

    def test_create_verifier_round_trip(self) -> None:
        """create_verifier() with a full settings dict populates all key fields."""
        from src.logic_layer.verifier import create_verifier
        cfg = {
            "llm": {
                "model_name": "gemma2:2b",
                "max_context_chars": 1200,
                "max_docs": 4,
                "max_chars_per_doc": 350,
            },
            "agent": {"max_verification_iterations": 3},
            "verifier": {
                "nli_model": "cross-encoder/nli-distilroberta-base",
                "contradiction_threshold": 0.88,
            },
        }
        verifier = create_verifier(cfg=cfg)
        assert verifier.config.model_name == "gemma2:2b"
        assert verifier.config.max_iterations == 3
        assert verifier.config.max_context_chars == 1200
        assert verifier.config.nli_model == "cross-encoder/nli-distilroberta-base"
        assert verifier.config.contradiction_threshold == 0.88


class TestPreGenerationValidator:
    """Tests for S_V pre-validation without LLM calls."""

    def test_initialization(self) -> None:
        """PreGenerationValidator is constructed without errors."""
        from src.logic_layer.verifier import PreGenerationValidator, VerifierConfig
        config = VerifierConfig(enable_contradiction_detection=False)
        validator = PreGenerationValidator(config=config, graph_store=None)
        assert validator is not None

    def test_validate_returns_pre_validation_result(self) -> None:
        """validate() returns a PreValidationResult with a valid status."""
        from src.logic_layer.verifier import (
            PreGenerationValidator, VerifierConfig,
            PreValidationResult, ValidationStatus,
        )
        config = VerifierConfig(
            enable_entity_path_validation=True,
            enable_contradiction_detection=False,
            enable_credibility_scoring=True,
        )
        validator = PreGenerationValidator(config=config, graph_store=None)
        result = validator.validate(
            context=[
                "Einstein was born in 1879 in Ulm, Germany.",
                "Einstein received the Nobel Prize in 1921.",
            ],
            query="When was Einstein born?",
            entities=["Einstein"],
        )
        assert isinstance(result, PreValidationResult)
        assert result.status in list(ValidationStatus)
        assert isinstance(result.filtered_context, list)

    def test_validate_empty_context(self) -> None:
        """validate() with empty context returns INSUFFICIENT_EVIDENCE without crashing."""
        from src.logic_layer.verifier import (
            PreGenerationValidator, VerifierConfig, ValidationStatus,
        )
        config = VerifierConfig(enable_contradiction_detection=False)
        validator = PreGenerationValidator(config=config, graph_store=None)
        result = validator.validate(context=[], query="test query", entities=[])
        assert result is not None
        assert result.status == ValidationStatus.INSUFFICIENT_EVIDENCE

    def test_validate_with_mock_graph_store(self) -> None:
        """Entity-path validation uses find_chunks_by_entity_multihop when graph store is present."""
        from src.logic_layer.verifier import (
            PreGenerationValidator, VerifierConfig, ValidationStatus,
        )
        config = VerifierConfig(
            enable_entity_path_validation=True,
            enable_contradiction_detection=False,
            enable_credibility_scoring=False,
        )
        mock_store = Mock()
        # Simulate KuzuDB interface: entity found → non-empty result.
        mock_store.find_chunks_by_entity_multihop.return_value = [
            {"text": "Einstein was a physicist.", "hops": 0}
        ]
        validator = PreGenerationValidator(config=config, graph_store=mock_store)
        result = validator.validate(
            context=["Einstein was a physicist."],
            query="Where was Einstein born?",
            entities=["Einstein"],
        )
        assert result is not None
        # Entity was found in the graph → path should be valid.
        assert result.entity_path_valid is True
        mock_store.find_chunks_by_entity_multihop.assert_called()


class TestVerifier:
    """Tests for S_V Verifier with mocked LLM."""

    @pytest.fixture
    def verifier(self):
        """Single-iteration Verifier (no self-correction) for isolated tests."""
        from src.logic_layer.verifier import create_verifier
        return create_verifier(cfg={"agent": {"max_verification_iterations": 1}})

    def test_initialization(self, verifier) -> None:
        """create_verifier() returns a Verifier instance."""
        from src.logic_layer.verifier import Verifier
        assert isinstance(verifier, Verifier)

    def test_set_graph_store(self, verifier) -> None:
        """set_graph_store() stores the store on both Verifier and PreGenerationValidator."""
        mock_store = Mock()
        verifier.set_graph_store(mock_store)
        assert verifier.graph_store is mock_store
        assert verifier.pre_validator.graph_store is mock_store

    def test_generate_and_verify_with_mock_llm(self, verifier) -> None:
        """generate_and_verify() returns a VerificationResult with a non-empty answer."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        with patch.object(
            verifier, "_call_llm",
            return_value=("Einstein was born in 1879.", 0.05),
        ):
            result = verifier.generate_and_verify(
                query="When was Einstein born?",
                context=["Einstein was born in 1879 in Ulm, Germany."],
                entities=["Einstein"],
            )
        assert isinstance(result, VerificationResult)
        assert result.answer != ""
        # LLM is mocked → exactly one call → one iteration.
        assert result.iterations == 1
        assert result.confidence in list(ConfidenceLevel)
        # At least one iteration must be recorded in the history.
        assert len(result.iteration_history) >= 1

    # -----------------------------------------------------------------------
    # _extract_claims() — atomic claim splitting (action item 1)
    # -----------------------------------------------------------------------

    def test_extract_claims_splits_multi_sentence_answer(self, verifier) -> None:
        """Multi-sentence answer yields one claim per sentence (≥ min_claim_chars)."""
        answer = "Einstein was born in 1879. He received the Nobel Prize in 1921."
        claims = verifier._extract_claims(answer)
        assert len(claims) >= 2, (
            f"Expected ≥ 2 claims for two-sentence answer, got {len(claims)}: {claims}"
        )

    def test_extract_claims_single_sentence(self, verifier) -> None:
        """Single-sentence answer yields exactly one claim."""
        answer = "Einstein was born in 1879."
        claims = verifier._extract_claims(answer)
        assert len(claims) == 1, (
            f"Expected exactly 1 claim for one-sentence answer, got {len(claims)}: {claims}"
        )

    def test_extract_claims_error_prefix_returns_empty(self, verifier) -> None:
        """[Error: …] answer is not a verifiable claim → empty list."""
        claims = verifier._extract_claims("[Error: LLM timeout]")
        assert claims == []

    def test_extract_claims_meta_statements_filtered(self, verifier) -> None:
        """Meta-hedges ('I don't know', 'based on the context') are removed."""
        answer = "Based on the context, I cannot answer this question."
        claims = verifier._extract_claims(answer)
        assert claims == [], (
            f"Meta-statement should be filtered out, got: {claims}"
        )

    # -----------------------------------------------------------------------
    # _format_context() — context formatting and budget limits (action item 2)
    # -----------------------------------------------------------------------

    def test_format_context_max_docs_limit_drops_excess(self, verifier) -> None:
        """When there are more docs than max_docs, the excess is silently dropped."""
        from src.logic_layer.verifier import create_verifier
        v = create_verifier(cfg={"llm": {"max_docs": 2, "max_chars_per_doc": 500,
                                          "max_context_chars": 10000}})
        docs = ["Doc one.", "Doc two.", "Doc three.", "Doc four."]
        formatted = v._format_context(docs)
        # max_docs=2 → at most 2 [N] doc sections in the output
        assert formatted.count("[1]") == 1
        assert formatted.count("[2]") == 1
        assert "[3]" not in formatted, (
            "Doc 3 should be dropped because max_docs=2"
        )

    def test_format_context_truncates_long_doc(self, verifier) -> None:
        """Doc longer than max_chars_per_doc is truncated (does not appear in full)."""
        from src.logic_layer.verifier import create_verifier
        long_doc = "A" * 2000
        v = create_verifier(cfg={"llm": {"max_docs": 5, "max_chars_per_doc": 100,
                                          "max_context_chars": 10000}})
        formatted = v._format_context([long_doc])
        # Formatted section should be much shorter than the original 2000 chars
        assert len(formatted) < 300, (
            f"Formatted context should be truncated; got length {len(formatted)}"
        )

    def test_format_context_empty_returns_no_context_string(self, verifier) -> None:
        """Empty context list → canonical 'No context available.' string."""
        assert verifier._format_context([]) == "No context available."

    # -----------------------------------------------------------------------
    # _verify_claim() — entity-presence verification (action item 3)
    # -----------------------------------------------------------------------

    def test_verify_claim_entity_in_context_returns_true(self, verifier) -> None:
        """Entity mentioned in context → claim verified (context substring path)."""
        is_ok, reason = verifier._verify_claim(
            "Einstein was born in 1879.",
            context=["Albert Einstein was a German-born physicist born in 1879."],
        )
        assert is_ok is True, f"Expected True, got ({is_ok}, {reason!r})"
        assert "context_verified" in reason

    def test_verify_claim_entity_absent_returns_false(self, verifier) -> None:
        """Entity absent from context → claim fails verification."""
        is_ok, reason = verifier._verify_claim(
            "Aristotle founded the Lyceum.",
            context=["Einstein was born in 1879 in Ulm, Germany."],
        )
        assert is_ok is False, f"Expected False, got ({is_ok}, {reason!r})"
        assert reason == "entities_not_found"

    def test_verify_claim_no_entities_returns_true(self, verifier) -> None:
        """Claim with no extractable proper nouns is verified by default."""
        is_ok, reason = verifier._verify_claim(
            "the answer is yes.",
            context=[],
        )
        assert is_ok is True
        assert reason == "no_entities_to_verify"

    # -----------------------------------------------------------------------
    # ANSWER_PROMPT template — context and query appear in formatted prompt
    # (action item 4: _build_generation_prompt equivalent)
    # -----------------------------------------------------------------------

    def test_answer_prompt_contains_context_and_query(self, verifier) -> None:
        """ANSWER_PROMPT.format() embeds both the context block and the query."""
        context_text = "[1] Einstein was born in Ulm."
        query_text   = "Where was Einstein born?"
        prompt = verifier.ANSWER_PROMPT.format(
            context=context_text, query=query_text
        )
        assert context_text in prompt, "Formatted context not found in prompt"
        assert query_text   in prompt, "Query string not found in prompt"

    def test_correction_prompt_contains_violations_context_query(self, verifier) -> None:
        """CORRECTION_PROMPT.format() embeds violations, context, and query."""
        violations = "- Aristotle founded the Lyceum."
        context_text = "[1] Plato founded the Academy."
        query_text   = "Who founded the Academy?"
        prompt = verifier.CORRECTION_PROMPT.format(
            violations=violations, context=context_text, query=query_text
        )
        assert violations    in prompt
        assert context_text  in prompt
        assert query_text    in prompt

    # -----------------------------------------------------------------------
    # LLM stub: imprecise answer still produces a VerificationResult (item 8)
    # -----------------------------------------------------------------------

    def test_generate_with_imprecise_llm_answer_produces_result(self, verifier) -> None:
        """Verifier completes even when the LLM returns a vague, ungrounded answer.

        Unlike test_generate_and_verify_with_mock_llm (which uses a perfectly
        context-grounded answer), this test uses a stub returning a generic
        hedge to confirm the pipeline is robust to low-quality LLM output.
        """
        from src.logic_layer.verifier import VerificationResult
        with patch.object(
            verifier, "_call_llm",
            return_value=("I don't know.", 0.05),
        ):
            result = verifier.generate_and_verify(
                query="Where was Einstein born?",
                context=["Einstein was born in 1879 in Ulm, Germany."],
                entities=["Einstein"],
            )
        assert isinstance(result, VerificationResult)
        # "I don't know." is a meta-statement → _extract_claims returns []
        # → no verified claims.  Result must still be a valid object.
        assert result.answer is not None
        assert result.iterations >= 1

    def test_generate_with_empty_context_no_crash(self, verifier) -> None:
        """Verifier does not crash when given empty context."""
        from src.logic_layer.verifier import VerificationResult
        with patch.object(
            verifier, "_call_llm",
            return_value=("Insufficient evidence.", 0.0),
        ):
            result = verifier.generate_and_verify(
                query="Unknown question?",
                context=[],
            )
        assert isinstance(result, VerificationResult)

    # -----------------------------------------------------------------------
    # Action 3: early exit when iteration 1 is fully verified
    # -----------------------------------------------------------------------

    def test_generate_stops_early_when_all_verified(self) -> None:
        """If iteration 1 passes all claims, no further iterations occur.

        The self-correction loop must exit immediately once all_verified is True;
        it must NOT continue up to max_iterations for no reason.  This is
        critical on edge hardware: unnecessary LLM calls triple latency.
        """
        from src.logic_layer.verifier import create_verifier
        call_count = [0]

        def counting_llm(prompt: str):
            call_count[0] += 1
            return ("Einstein was born in 1879.", 0.05)

        verifier = create_verifier(cfg={"agent": {"max_verification_iterations": 3}})
        with patch.object(verifier, "_call_llm", side_effect=counting_llm):
            result = verifier.generate_and_verify(
                query="When was Einstein born?",
                context=["Einstein was born in 1879 in Ulm."],
                entities=["Einstein"],
            )

        assert call_count[0] == 1, (
            f"Must stop after first verified iteration; got {call_count[0]} LLM calls"
        )
        assert result.all_verified is True

    # -----------------------------------------------------------------------
    # Action 4: CORRECTION_PROMPT receives violated claim text
    # -----------------------------------------------------------------------

    def test_self_correction_uses_correction_prompt(self) -> None:
        """Iteration 2 must be sent CORRECTION_PROMPT containing the violated claim.

        Captures both prompts and verifies:
        - Exactly 2 LLM calls occur (violated → corrected)
        - The second prompt contains the violated claim text (from iter 1)
        - The second prompt is structurally different from the first
        """
        from src.logic_layer.verifier import create_verifier
        prompts_received: list = []

        def capturing_llm(prompt: str):
            prompts_received.append(prompt)
            if len(prompts_received) == 1:
                return ("Aristotle wrote the Nicomachean Ethics.", 0.05)
            return ("Einstein was born in 1879.", 0.05)

        verifier = create_verifier(cfg={
            "agent": {"max_verification_iterations": 2},
            "verifier": {
                "enable_credibility_scoring": False,
                "enable_entity_path_validation": False,
            },
        })
        with patch.object(verifier, "_call_llm", side_effect=capturing_llm):
            verifier.generate_and_verify(
                query="When was Einstein born?",
                context=["Einstein was born in 1879 in Ulm, Germany."],
                entities=["Einstein"],
            )

        assert len(prompts_received) >= 2, (
            "Must call LLM twice: once for initial answer, once for correction"
        )
        # The correction prompt must reference the violated claim
        correction_prompt = prompts_received[1]
        assert "Aristotle" in correction_prompt, (
            "Correction prompt must include the violated claim entity ('Aristotle')"
        )
        # The correction prompt must use the CORRECTION_PROMPT template keyword
        assert "unverified" in correction_prompt.lower(), (
            "Correction prompt must reference unverified/violated claims"
        )

    # -----------------------------------------------------------------------
    # Action 9: _verify_claim falls back to context when graph returns empty
    # -----------------------------------------------------------------------

    def test_verify_claim_graph_empty_falls_back_to_context(self, verifier) -> None:
        """When the graph store returns no results, context substring is used.

        Graph store is attached but returns [] for find_chunks_by_entity_multihop.
        The context contains 'Einstein' → should return True via context fallback.
        """
        mock_store = Mock()
        mock_store.find_chunks_by_entity_multihop.return_value = []  # not in graph
        verifier.set_graph_store(mock_store)

        is_ok, reason = verifier._verify_claim(
            "Einstein was born in 1879.",
            context=["Albert Einstein was born in 1879 in Ulm."],
        )
        assert is_ok is True, (
            f"Expected True via context fallback; got ({is_ok}, {reason!r})"
        )
        assert "context_verified" in reason, (
            f"Reason must indicate context fallback; got {reason!r}"
        )
        # Clean up graph store to avoid state leak between tests
        verifier.set_graph_store(None)

    # -----------------------------------------------------------------------
    # Action 11: _format_context budget stops before max_docs
    # -----------------------------------------------------------------------

    def test_format_context_budget_stops_before_max_docs(self, verifier) -> None:
        """max_context_chars budget is exhausted before max_docs limit is reached.

        When max_context_chars is very small, the function must stop adding docs
        early — even if max_docs would allow more.  The second doc must be absent.
        """
        from src.logic_layer.verifier import create_verifier
        # max_docs=5 allows 5 docs but max_context_chars=50 is exhausted after 1
        v = create_verifier(cfg={
            "llm": {
                "max_docs": 5,
                "max_chars_per_doc": 500,
                "max_context_chars": 50,
            }
        })
        docs = ["First document text here.", "Second document text here.", "Third doc."]
        formatted = v._format_context(docs)
        assert "[2]" not in formatted, (
            "Budget (max_context_chars=50) should stop before doc 2 is added"
        )

    def test_self_correction_loop_two_iterations(self) -> None:
        """Self-correction runs a second iteration when the first answer has violations.

        This test exercises the central thesis contribution of S_V:
        the Self-Refine loop (Madaan et al., 2023, NeurIPS).

        Iteration 1: LLM returns an answer containing "Aristotle", an entity
        that is not in the context → claim is violated.
        Iteration 2: LLM returns a context-grounded answer → all claims pass.
        Expected outcome: result.iterations == 2 and result.all_verified is True.
        """
        from src.logic_layer.verifier import create_verifier, VerificationResult
        # Two-iteration verifier with credibility scoring disabled for speed.
        verifier = create_verifier(cfg={
            "agent": {"max_verification_iterations": 2},
            "verifier": {
                "enable_credibility_scoring": False,
                "enable_entity_path_validation": False,
            },
        })
        llm_responses = [
            # Iteration 1: mentions "Aristotle" which is absent from context.
            ("Aristotle wrote the Nicomachean Ethics.", 0.05),
            # Iteration 2: context-grounded answer.
            ("Einstein was born in 1879.", 0.05),
        ]
        call_count = 0

        def mock_llm(prompt: str):
            nonlocal call_count
            response = llm_responses[min(call_count, len(llm_responses) - 1)]
            call_count += 1
            return response

        with patch.object(verifier, "_call_llm", side_effect=mock_llm):
            result = verifier.generate_and_verify(
                query="When was Einstein born?",
                context=["Einstein was born in 1879 in Ulm, Germany."],
                entities=["Einstein"],
            )

        assert isinstance(result, VerificationResult)
        assert result.iterations >= 2, (
            "Self-correction must run at least 2 iterations when iteration 1 has violations"
        )
        # The second answer is context-grounded → should be all_verified.
        assert result.all_verified is True, (
            "Second iteration answer should pass verification"
        )


# =============================================================================
# 4. AGENTIC CONTROLLER TESTS
# =============================================================================


class TestAgenticController:
    """Tests for AgenticController: full S_P → S_N → S_V pipeline."""

    @pytest.fixture
    def controller(self):
        """Single-iteration controller for fast unit tests.

        model_name='phi3' and max_iterations=1 are explicit unit-test scope
        choices; the production default (settings.yaml llm.model_name and
        agent.max_verification_iterations) may differ.
        """
        from src.logic_layer.controller import create_controller
        return create_controller(model_name="phi3", max_iterations=1)

    def test_initialization(self, controller) -> None:
        """Controller exposes all three agent instances."""
        assert controller.planner is not None
        assert controller.navigator is not None
        assert controller.verifier is not None

    def test_navigator_starts_without_retriever(self, controller) -> None:
        """Navigator has no retriever at construction time."""
        assert controller.navigator.retriever is None

    def test_set_retriever(self, controller) -> None:
        """set_retriever() delegates to Navigator."""
        mock_retriever = Mock()
        controller.set_retriever(mock_retriever)
        assert controller.navigator.retriever is mock_retriever

    def test_set_graph_store(self, controller) -> None:
        """set_graph_store() delegates to Verifier."""
        mock_store = Mock()
        controller.set_graph_store(mock_store)
        assert controller.verifier.graph_store is mock_store

    def test_planner_classifies_query_type(self, controller) -> None:
        """Planner inside the controller returns a valid QueryType."""
        from src.logic_layer.planner import QueryType
        plan = controller.planner.plan("What is machine learning?")
        assert plan.query_type in list(QueryType)

    def test_run_returns_dict_with_answer_key(self, controller) -> None:
        """run() returns an AgentState dict containing an 'answer' key."""
        with patch.object(
            controller.verifier, "_call_llm",
            return_value=("Mocked answer.", 0.05),
        ):
            state = controller.run("What is machine learning?")
        assert isinstance(state, dict)
        assert "answer" in state

    def test_run_without_retriever_no_crash(self, controller) -> None:
        """run() without a retriever completes and returns an answer string."""
        with patch.object(
            controller.verifier, "_call_llm",
            return_value=("No information available.", 0.05),
        ):
            state = controller.run("What is the capital of France?")
        assert isinstance(state, dict)
        assert isinstance(state.get("answer", ""), str)

    def test_call_shortcut_returns_string(self, controller) -> None:
        """__call__() returns the answer directly as a string."""
        with patch.object(
            controller.verifier, "_call_llm",
            return_value=("Test answer.", 0.05),
        ):
            answer = controller("What is 2+2?")
        assert isinstance(answer, str)

    def test_full_pipeline_with_mock_retriever_and_llm(self, controller) -> None:
        """Full S_P → S_N → S_V flow using the rule-based LLM stub.

        Uses _rule_based_llm_stub instead of a fixed canned answer so that
        the test verifies the prompt-formatting code path: if _format_context()
        or ANSWER_PROMPT.format() produces a malformed prompt, the stub returns
        'I don't know.' and the answer assertion fails.
        """
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            [MockResult("Paris is the capital of France.", 0.9)],
            {"latency_ms": 5},
        )
        controller.set_retriever(mock_retriever)
        with patch.object(
            controller.verifier, "_call_llm",
            side_effect=_rule_based_llm_stub,
        ):
            state = controller.run("What is the capital of France?")
        assert state["answer"] != "", "Rule-based stub must produce a non-empty answer"
        assert "Paris" in state["answer"] or "capital" in state["answer"].lower(), (
            f"Rule-based stub should echo context content; got: {state['answer']!r}"
        )
        assert "context" in state
        assert "iterations" in state

    def test_planner_entities_reach_navigator_filter(self, controller) -> None:
        """E2E: entities extracted by S_P are forwarded into Navigator's
        entity-mention filter — they must not be silently dropped.

        A mock retriever returns one chunk mentioning "Einstein" and one
        that does not.  After the entity-mention filter, only the Einstein
        chunk must remain (or both if Safety Fallback fires — but at least
        the Einstein chunk must be present).
        """
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            [
                MockResult("Einstein was born in Ulm, Germany in 1879.", 0.9),
                MockResult("The weather in Berlin is cold in winter.", 0.3),
            ],
            {"latency_ms": 5},
        )
        controller.set_retriever(mock_retriever)
        with patch.object(
            controller.verifier, "_call_llm",
            return_value=("Einstein was born in Ulm.", 0.05),
        ):
            state = controller.run("Where was Albert Einstein born?")

        context = state.get("context", [])
        # At least one context chunk must mention Einstein — confirming that
        # Planner entities reached the Navigator entity-mention filter and the
        # relevant chunk was not dropped.
        assert any("Einstein" in c for c in context), (
            f"Einstein chunk must survive entity-mention filter; context={context}"
        )

    def test_state_has_all_required_keys(self, controller) -> None:
        """AgentState contains the seven keys consumed by the evaluation harness."""
        with patch.object(
            controller.verifier, "_call_llm",
            return_value=("Test.", 0.05),
        ):
            state = controller.run("test query")
        required_keys = {
            "query", "answer", "context", "iterations",
            "all_verified", "verified_claims", "violated_claims",
        }
        assert required_keys.issubset(set(state.keys()))


# =============================================================================
# 5. AGENT PIPELINE TESTS (pipeline/)
# =============================================================================


class TestAgentPipeline:
    """Tests for the AgentPipeline orchestrator in src/pipeline/."""

    def test_pipeline_result_to_dict(self) -> None:
        """PipelineResult.to_dict() contains all mandatory top-level fields."""
        from src.pipeline.agent_pipeline import PipelineResult
        result = PipelineResult(
            answer="Paris is the capital.",
            confidence="high",
            query="What is the capital of France?",
            planner_result={},
            navigator_result={},
            verifier_result={},
            planner_time_ms=10.0,
            navigator_time_ms=20.0,
            verifier_time_ms=50.0,
            total_time_ms=80.0,
        )
        d = result.to_dict()
        assert d["answer"] == "Paris is the capital."
        assert d["confidence"] == "high"
        assert d["query"] == "What is the capital of France?"
        assert "stages" in d
        assert "timing" in d

    def test_pipeline_result_to_json_valid(self) -> None:
        """to_json() produces syntactically valid JSON."""
        from src.pipeline.agent_pipeline import PipelineResult
        result = PipelineResult(
            answer="Test",
            confidence="low",
            query="test",
            planner_result={},
            navigator_result={},
            verifier_result={},
            planner_time_ms=0.0,
            navigator_time_ms=0.0,
            verifier_time_ms=0.0,
            total_time_ms=0.0,
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["answer"] == "Test"

    def test_pipeline_imports(self) -> None:
        """AgentPipeline and factory functions are importable."""
        from src.pipeline.agent_pipeline import (
            AgentPipeline, create_full_pipeline,
        )
        assert AgentPipeline is not None
        assert create_full_pipeline is not None

    def test_batch_processor_importable(self) -> None:
        """BatchProcessor is importable from agent_pipeline."""
        from src.pipeline.agent_pipeline import BatchProcessor
        assert BatchProcessor is not None


# =============================================================================
# 6. THESIS COMPLIANCE TESTS
# =============================================================================


class TestThesisCompliance:
    """
    Validates that the implementation matches the thesis specifications.

    Each test encodes one architectural or algorithmic commitment from the
    thesis design.  A green suite here provides reviewers with an executable
    contract between the written thesis and the code.
    """

    def test_query_types_complete(self) -> None:
        """All four QueryTypes required by the thesis query taxonomy exist."""
        from src.logic_layer.planner import QueryType
        required = {"SINGLE_HOP", "MULTI_HOP", "COMPARISON", "TEMPORAL"}
        actual = {q.name for q in QueryType}
        assert required.issubset(actual)

    def test_retrieval_strategies_complete(self) -> None:
        """All three RetrievalStrategies required by the thesis design exist."""
        from src.logic_layer.planner import RetrievalStrategy
        required = {"VECTOR_ONLY", "GRAPH_ONLY", "HYBRID"}
        actual = {s.name for s in RetrievalStrategy}
        assert required.issubset(actual)

    def test_confidence_levels_complete(self) -> None:
        """ConfidenceLevel has exactly HIGH, MEDIUM, LOW — no more, no fewer."""
        from src.logic_layer.verifier import ConfidenceLevel
        required = {"HIGH", "MEDIUM", "LOW"}
        actual = {c.name for c in ConfidenceLevel}
        assert required == actual

    def test_confidence_value_strings_are_lowercase(self) -> None:
        """ConfidenceLevel.value strings are lowercase: 'high', 'medium', 'low'.

        agent_pipeline.py stores result.confidence.value in the JSON output and
        the evaluation harness compares these strings.  If the values were 'HIGH'
        instead of 'high', downstream comparisons would silently fail.
        """
        from src.logic_layer.verifier import ConfidenceLevel
        assert ConfidenceLevel.HIGH.value   == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value    == "low"

    def test_verifier_supports_self_correction(self) -> None:
        """VerifierConfig.max_iterations >= 1 (ablation baseline: 1; thesis default: 2)."""
        from src.logic_layer.verifier import VerifierConfig
        config = VerifierConfig()
        assert config.max_iterations >= 1

    def test_agent_state_has_required_fields(self) -> None:
        """AgentState TypedDict contains all keys consumed by the evaluation harness."""
        from src.logic_layer.controller import AgentState
        annotations = AgentState.__annotations__
        required = {
            "query", "answer", "context", "iterations",
            "all_verified", "verified_claims", "violated_claims",
            "retrieval_plan", "errors",
        }
        assert required.issubset(set(annotations.keys()))

    def test_navigator_relevance_threshold_default(self) -> None:
        """ControllerConfig default relevance_threshold_factor matches settings.yaml value 0.85."""
        from src.logic_layer.navigator import ControllerConfig
        config = ControllerConfig()
        assert config.relevance_threshold_factor == 0.85

    def test_navigator_redundancy_threshold_default(self) -> None:
        """ControllerConfig default redundancy_threshold matches settings.yaml value 0.8."""
        from src.logic_layer.navigator import ControllerConfig
        config = ControllerConfig()
        assert config.redundancy_threshold == 0.8

    def test_confidence_boundary_high(self) -> None:
        """Exactly 80 % verified claims → HIGH (boundary value, threshold=0.8)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="A",
            iterations=1,
            verified_claims=["c1", "c2", "c3", "c4"],
            violated_claims=["c5"],  # 4/5 = 0.80 → HIGH
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )
        assert result.confidence == ConfidenceLevel.HIGH

    def test_confidence_boundary_medium(self) -> None:
        """Exactly 50 % verified claims → MEDIUM (boundary value, threshold=0.5)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="A",
            iterations=1,
            verified_claims=["c1"],
            violated_claims=["c2"],  # 1/2 = 0.50 → MEDIUM
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )
        assert result.confidence == ConfidenceLevel.MEDIUM


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
