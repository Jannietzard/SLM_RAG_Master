"""
Test Suite für Logic Layer (Artefakt B)

Version: 1.0.0
Author: Edge-RAG Research Project

===============================================================================
TEST COVERAGE
===============================================================================

1. Planner Tests (S_P)
   - Initialisierung, RetrievalPlan-Struktur
   - QueryType-Erkennung: SINGLE_HOP, MULTI_HOP, COMPARISON, TEMPORAL
   - RetrievalStrategy-Auswahl
   - Edge Cases (leere Query, lange Query, Deutsch)

2. Navigator Tests (S_N)
   - Initialisierung, set_retriever()
   - RRF-Fusion (_rrf_fusion)
   - Relevance Filter (_relevance_filter)
   - Redundancy Filter (_redundancy_filter)
   - navigate() ohne Retriever (leere Rückgabe)
   - navigate() mit Mock-Retriever (Vollpfad)

3. Verifier Tests (S_V)
   - VerificationResult.confidence Property (HIGH/MEDIUM/LOW)
   - PreGenerationValidator.validate()
   - Verifier.generate_and_verify() mit gemocktem LLM

4. AgenticController Tests
   - Initialisierung (S_P + S_N + S_V)
   - set_retriever(), set_graph_store()
   - run() → AgentState-Dict mit "answer"
   - __call__() → String
   - Vollständiger Pipeline-Flow mit Mock-Retriever + Mock-LLM

5. AgentPipeline Tests (pipeline/)
   - PipelineResult.to_dict(), to_json()
   - Imports: AgentPipeline, BatchProcessor

6. Thesis Compliance Tests
   - QueryType- und RetrievalStrategy-Enum-Vollständigkeit
   - ConfidenceLevel HIGH/MEDIUM/LOW Berechnung
   - AgentState-Schlüsselfelder
   - RRF Cross-Source Corroboration Boost

===============================================================================
USAGE
===============================================================================

# Aus src/logic_layer/:
pytest test_logic_layer.py -v

# Aus Projekt-Root (Entwicklungfolder/):
pytest src/logic_layer/test_logic_layer.py -v

===============================================================================
"""

import pytest
import json
from unittest.mock import Mock, patch


# ============================================================================
# 1. PLANNER TESTS (S_P)
# ============================================================================

class TestPlanner:
    """Tests für S_P: Query Analysis und Retrieval-Plan-Erstellung."""

    def test_initialization(self):
        """Planner kann erstellt werden."""
        from src.logic_layer.planner import create_planner, Planner
        planner = create_planner()
        assert isinstance(planner, Planner)

    def test_plan_returns_retrieval_plan(self):
        """plan() gibt ein RetrievalPlan-Objekt zurück."""
        from src.logic_layer.planner import create_planner, RetrievalPlan
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        assert isinstance(plan, RetrievalPlan)

    def test_plan_has_required_attributes(self):
        """RetrievalPlan enthält alle Pflichtfelder."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Who invented the telephone?")
        assert hasattr(plan, 'original_query')
        assert hasattr(plan, 'query_type')
        assert hasattr(plan, 'strategy')
        assert hasattr(plan, 'entities')
        assert hasattr(plan, 'hop_sequence')
        assert hasattr(plan, 'confidence')
        assert plan.original_query == "Who invented the telephone?"

    def test_single_hop_query_type(self):
        """Einfache Faktenfragen → SINGLE_HOP."""
        from src.logic_layer.planner import create_planner, QueryType, RetrievalStrategy
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        assert plan.query_type == QueryType.SINGLE_HOP
        assert plan.strategy == RetrievalStrategy.VECTOR_ONLY

    def test_multi_hop_query_type(self):
        """Mehrstufige Fragen → MULTI_HOP."""
        from src.logic_layer.planner import create_planner, QueryType, RetrievalStrategy
        planner = create_planner()
        plan = planner.plan("Who directed the movie starring Tom Hanks?")
        assert plan.query_type == QueryType.MULTI_HOP
        assert plan.strategy == RetrievalStrategy.HYBRID

    def test_comparison_query_type(self):
        """Vergleichsfragen → COMPARISON."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan("Is Berlin larger than Munich?")
        assert plan.query_type == QueryType.COMPARISON

    def test_temporal_query_type(self):
        """Zeitbezogene Fragen → TEMPORAL."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan("What happened in 2020?")
        assert plan.query_type == QueryType.TEMPORAL

    def test_plan_confidence_in_valid_range(self):
        """Confidence liegt immer zwischen 0 und 1."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Who invented the telephone?")
        assert 0.0 <= plan.confidence <= 1.0

    def test_decompose_query_returns_list(self):
        """decompose_query() gibt eine nicht-leere Liste zurück."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        sub_queries = planner.decompose_query(
            "Who directed Inception and when was it released?"
        )
        assert isinstance(sub_queries, list)
        assert len(sub_queries) >= 1

    def test_empty_query_no_crash(self):
        """Leere Query löst keinen Exception aus."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("")
        assert plan is not None

    def test_very_long_query_no_crash(self):
        """Sehr lange Query wird verarbeitet ohne Absturz."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("a" * 1000)
        assert plan is not None

    def test_non_english_query_no_crash(self):
        """Nicht-englische Query wird verarbeitet ohne Absturz."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Was ist die Hauptstadt von Deutschland?")
        assert plan is not None

    # -----------------------------------------------------------------------
    # QueryType Classification — datensatzrepräsentative Muster
    # (HotpotQA / 2WikiMultiHop)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("query", [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Are Madonna and Lady Gaga from the same country?",
        "Did Nikola Tesla and Thomas Edison have the same nationality?",
    ])
    def test_comparison_same_attribute_pattern(self, query):
        """'same X'-Muster → COMPARISON (nicht multi_hop)."""
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
    def test_comparison_comparative_adjective_pattern(self, query):
        """Komparative Adjektive (older/taller/...) → COMPARISON."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON, (
            f"Expected COMPARISON for '{query}', got {plan.query_type.value}"
        )

    # -----------------------------------------------------------------------
    # Sub-Query-Inhalt — Kernfunktion des Planners
    # -----------------------------------------------------------------------

    def test_comparison_sub_queries_are_distinct(self):
        """Comparison-Sub-Queries für zwei Entities müssen verschieden sein."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        # Verwende mehrere repräsentative Queries
        queries = [
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "Is Berlin older than Munich?",
            "Which is taller, the Eiffel Tower or Big Ben?",
        ]
        for q in queries:
            plan = planner.plan(q)
            assert plan.query_type == QueryType.COMPARISON
            # Die ersten beiden Sub-Queries (Entity-spezifisch) müssen unterschiedlich sein
            assert len(plan.sub_queries) >= 2, f"Zu wenige Sub-Queries für: {q}"
            assert plan.sub_queries[0] != plan.sub_queries[1], (
                f"Sub-Queries sind identisch für: {q}\n"
                f"  sub_queries[0]={plan.sub_queries[0]}\n"
                f"  sub_queries[1]={plan.sub_queries[1]}"
            )

    def test_comparison_sub_queries_contain_respective_entity(self):
        """Jede Entity-spezifische Sub-Query muss die zugehörige Entity enthalten."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        # Beide Entities bekannt und eindeutig
        query = "Were Scott Derrickson and Ed Wood of the same nationality?"
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON
        # sub_queries[0] und sub_queries[1] sollten je eine der Entities enthalten
        entity_names = ["Scott Derrickson", "Ed Wood"]
        sq0, sq1 = plan.sub_queries[0], plan.sub_queries[1]
        assert any(e in sq0 for e in entity_names), (
            f"Keine Entity in sub_queries[0]: '{sq0}'"
        )
        assert any(e in sq1 for e in entity_names), (
            f"Keine Entity in sub_queries[1]: '{sq1}'"
        )
        # Die beiden Sub-Queries müssen verschiedene Entities abdecken
        assert sq0 != sq1

    def test_comparison_sub_queries_preserve_query_structure(self):
        """Sub-Queries sollen Fragmente der Original-Query sein (nicht generisch 'What is X?')."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        query = "Were Scott Derrickson and Ed Wood of the same nationality?"
        plan = planner.plan(query)
        assert plan.query_type == QueryType.COMPARISON
        # Sub-Queries dürfen nicht das generische Fallback-Muster sein
        for sq in plan.sub_queries[:2]:
            assert not sq.startswith("What is "), (
                f"Generischer Fallback nicht akzeptabel: '{sq}'"
            )

    def test_multi_hop_generates_multiple_sub_queries(self):
        """Multi-Hop-Query → mindestens 2 Sub-Queries."""
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
                f"Multi-Hop braucht >= 2 Sub-Queries für: {q}"
            )

    # -----------------------------------------------------------------------
    # Entity Extraction — Qualität
    # -----------------------------------------------------------------------

    def test_entity_extraction_finds_person_names(self):
        """SpaCy soll PERSON-Entities in Queries finden."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        entity_texts = [e.text for e in plan.entities]
        assert "Scott Derrickson" in entity_texts or any(
            "Derrickson" in t for t in entity_texts
        ), f"Scott Derrickson nicht in Entities: {entity_texts}"
        assert "Ed Wood" in entity_texts or any(
            "Wood" in t for t in entity_texts
        ), f"Ed Wood nicht in Entities: {entity_texts}"

    def test_entity_extraction_finds_gpe(self):
        """SpaCy soll GPE-Entities (Länder, Städte) finden."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("Is Berlin older than Munich?")
        entity_texts = [e.text for e in plan.entities]
        assert any("Berlin" in t for t in entity_texts), (
            f"Berlin nicht in Entities: {entity_texts}"
        )
        assert any("Munich" in t for t in entity_texts), (
            f"Munich nicht in Entities: {entity_texts}"
        )

    def test_hop_sequence_comparison_has_parallel_steps(self):
        """Comparison-Plan: Entity-Steps sollen keine Abhängigkeiten haben (parallel)."""
        from src.logic_layer.planner import create_planner, QueryType
        planner = create_planner()
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        assert plan.query_type == QueryType.COMPARISON
        # Ersten Schritte sind parallel (depends_on=[])
        parallel_steps = [s for s in plan.hop_sequence if s.depends_on == []]
        assert len(parallel_steps) >= 2, (
            f"Mindestens 2 parallele Steps erwartet, got: {[s.depends_on for s in plan.hop_sequence]}"
        )


# ============================================================================
# 2. NAVIGATOR TESTS (S_N)
# ============================================================================

class TestNavigator:
    """Tests für S_N: Hybrid Retrieval und Pre-Generative Filtering."""

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
    def fused_results(self):
        """Vorberechnete Dicts im Format nach _rrf_fusion()."""
        return [
            {"text": "Paris is the capital of France.", "rrf_score": 0.90},
            {"text": "France is located in Western Europe.",  "rrf_score": 0.70},
            {"text": "French cuisine is internationally famous.", "rrf_score": 0.30},
        ]

    def test_initialization(self, navigator):
        """Navigator startet ohne Retriever."""
        assert navigator is not None
        assert navigator.retriever is None

    def test_set_retriever(self, navigator):
        """set_retriever() speichert den Retriever korrekt."""
        mock_retriever = Mock()
        navigator.set_retriever(mock_retriever)
        assert navigator.retriever is mock_retriever

    def test_relevance_filter_removes_low_scores(self, navigator, fused_results):
        """Chunks unter 0.6 × max_score werden entfernt."""
        # threshold = 0.6 * 0.90 = 0.54 → "French cuisine" (0.30) fliegt raus
        filtered = navigator._relevance_filter(fused_results)
        max_score = max(r["rrf_score"] for r in fused_results)
        threshold = 0.6 * max_score
        assert all(r["rrf_score"] >= threshold for r in filtered)
        assert len(filtered) < len(fused_results)

    def test_relevance_filter_empty_input(self, navigator):
        """Leere Liste → leere Liste."""
        assert navigator._relevance_filter([]) == []

    def test_redundancy_filter_removes_duplicates(self, navigator):
        """Hochähnliche Chunks werden dedupliziert."""
        results = [
            {"text": "Paris is the capital of France",      "rrf_score": 0.9},
            {"text": "The capital of France is Paris",      "rrf_score": 0.8},  # ~gleich
            {"text": "Darwin studied evolution in Galapagos", "rrf_score": 0.7},
        ]
        filtered = navigator._redundancy_filter(results)
        assert len(filtered) <= 2

    def test_redundancy_filter_empty_input(self, navigator):
        """Leere Liste → leere Liste."""
        assert navigator._redundancy_filter([]) == []

    def test_jaccard_similarity_identical_texts(self, navigator):
        """Identische Texte → Jaccard = 1.0."""
        sim = navigator._jaccard_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_jaccard_similarity_different_texts(self, navigator):
        """Komplett unterschiedliche Texte → Jaccard < 0.5."""
        sim = navigator._jaccard_similarity("hello world", "goodbye universe")
        assert sim < 0.5

    def test_rrf_fusion_deduplicates_and_boosts(self, navigator):
        """Chunk der in zwei Sub-Queries auftaucht erhält Corroboration-Boost."""
        results = [
            {"text": "Doc A", "score": 0.9, "source": "s1", "sub_query": "q1"},
            {"text": "Doc B", "score": 0.7, "source": "s1", "sub_query": "q1"},
            {"text": "Doc A", "score": 0.8, "source": "s2", "sub_query": "q2"},  # Duplikat
        ]
        fused = navigator._rrf_fusion(results)
        doc_a = [r for r in fused if r["text"] == "Doc A"]
        doc_b = [r for r in fused if r["text"] == "Doc B"]
        assert len(doc_a) == 1          # Kein Duplikat
        assert len(doc_b) == 1
        assert doc_a[0]["rrf_score"] > doc_b[0]["rrf_score"]  # Boost durch Cross-Source

    def test_navigate_without_retriever_returns_empty_context(self, navigator):
        """Ohne Retriever gibt navigate() leeres NavigatorResult zurück."""
        from src.logic_layer.planner import create_planner
        planner = create_planner()
        plan = planner.plan("What is the capital of France?")
        result = navigator.navigate(plan, ["What is the capital of France?"])
        assert result is not None
        assert result.filtered_context == []

    def test_navigate_with_mock_retriever(self, navigator):
        """Mit Mock-Retriever enthält NavigatorResult gefilterte Texte."""
        from src.logic_layer.planner import create_planner

        class MockResult:
            def __init__(self, text, score):
                self.text = text
                self.rrf_score = score
                self.source_doc = "test.pdf"

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


# ============================================================================
# 3. VERIFIER TESTS (S_V)
# ============================================================================

class TestVerificationResultConfidence:
    """Tests für VerificationResult.confidence Property."""

    def test_confidence_high_all_verified(self):
        """Alle Claims verifiziert → HIGH (≥ 0.8)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="Answer",
            iterations=1,
            verified_claims=["c1", "c2", "c3", "c4", "c5"],
            violated_claims=[],
            all_verified=True,
        )
        assert result.confidence == ConfidenceLevel.HIGH

    def test_confidence_medium_mixed_claims(self):
        """2/3 verifiziert → MEDIUM (0.5 ≤ ratio < 0.8)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="Answer",
            iterations=1,
            verified_claims=["c1", "c2"],
            violated_claims=["c3"],
        )
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_confidence_low_mostly_violated(self):
        """1/4 verifiziert → LOW (< 0.5)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="Answer",
            iterations=1,
            verified_claims=["c1"],
            violated_claims=["c2", "c3", "c4"],
        )
        assert result.confidence == ConfidenceLevel.LOW

    def test_confidence_low_when_no_claims(self):
        """Keine Claims → LOW."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(answer="Answer", iterations=1)
        assert result.confidence == ConfidenceLevel.LOW

    def test_confidence_value_is_string(self):
        """.confidence.value muss ein String sein (für agent_pipeline.py)."""
        from src.logic_layer.verifier import VerificationResult
        result = VerificationResult(
            answer="Answer", iterations=1, verified_claims=["c1"]
        )
        assert isinstance(result.confidence.value, str)


class TestPreGenerationValidator:
    """Tests für S_V Pre-Validation ohne LLM."""

    def test_initialization(self):
        from src.logic_layer.verifier import PreGenerationValidator, VerifierConfig
        config = VerifierConfig(enable_contradiction_detection=False)
        validator = PreGenerationValidator(config=config, graph_store=None)
        assert validator is not None

    def test_validate_returns_pre_validation_result(self):
        """validate() gibt PreValidationResult mit korrektem Status zurück."""
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

    def test_validate_empty_context(self):
        """validate() mit leerem Context stürzt nicht ab."""
        from src.logic_layer.verifier import PreGenerationValidator, VerifierConfig
        config = VerifierConfig(enable_contradiction_detection=False)
        validator = PreGenerationValidator(config=config, graph_store=None)
        result = validator.validate(context=[], query="test query", entities=[])
        assert result is not None


class TestVerifier:
    """Tests für S_V Verifier mit gemocktem LLM."""

    def test_initialization(self):
        from src.logic_layer.verifier import create_verifier, Verifier
        verifier = create_verifier(max_iterations=1)
        assert isinstance(verifier, Verifier)

    def test_config_defaults(self):
        """VerifierConfig hat sinnvolle Standardwerte."""
        from src.logic_layer.verifier import VerifierConfig
        config = VerifierConfig()
        assert config.max_iterations >= 1
        assert config.max_context_chars > 0

    def test_set_graph_store(self):
        """set_graph_store() speichert den Store korrekt."""
        from src.logic_layer.verifier import create_verifier
        verifier = create_verifier(max_iterations=1)
        mock_store = Mock()
        verifier.set_graph_store(mock_store)
        assert verifier.graph_store is mock_store

    def test_generate_and_verify_with_mock_llm(self):
        """generate_and_verify() gibt VerificationResult zurück (gemocktes LLM)."""
        from src.logic_layer.verifier import (
            create_verifier, VerificationResult, ConfidenceLevel,
        )
        verifier = create_verifier(max_iterations=1)
        with patch.object(
            verifier, '_call_llm',
            return_value=("Einstein was born in 1879.", 0.05)
        ):
            result = verifier.generate_and_verify(
                query="When was Einstein born?",
                context=["Einstein was born in 1879 in Ulm, Germany."],
                entities=["Einstein"],
            )
        assert isinstance(result, VerificationResult)
        assert result.answer != ""
        assert result.iterations >= 1
        assert result.confidence in list(ConfidenceLevel)

    def test_generate_with_empty_context_no_crash(self):
        """Verifier stürzt nicht ab wenn kein Context vorhanden."""
        from src.logic_layer.verifier import create_verifier, VerificationResult
        verifier = create_verifier(max_iterations=1)
        with patch.object(
            verifier, '_call_llm',
            return_value=("Insufficient evidence.", 0.0)
        ):
            result = verifier.generate_and_verify(
                query="Unknown question?",
                context=[],
            )
        assert isinstance(result, VerificationResult)


# ============================================================================
# 4. AGENTIC CONTROLLER TESTS
# ============================================================================

class TestAgenticController:
    """Tests für AgenticController: vollständige S_P → S_N → S_V Pipeline."""

    @pytest.fixture
    def controller(self):
        from src.logic_layer.navigator import create_controller
        return create_controller(model_name="phi3", max_iterations=1)

    def test_initialization(self, controller):
        """Controller hat alle drei Agenten."""
        assert controller.planner is not None
        assert controller.navigator is not None
        assert controller.verifier is not None

    def test_navigator_starts_without_retriever(self, controller):
        """Navigator hat initial keinen Retriever."""
        assert controller.navigator.retriever is None

    def test_set_retriever(self, controller):
        """set_retriever() delegiert an Navigator."""
        mock_retriever = Mock()
        controller.set_retriever(mock_retriever)
        assert controller.navigator.retriever is mock_retriever

    def test_set_graph_store(self, controller):
        """set_graph_store() delegiert an Verifier."""
        mock_store = Mock()
        controller.set_graph_store(mock_store)
        assert controller.verifier.graph_store is mock_store

    def test_planner_classifies_query_type(self, controller):
        """Planner im Controller erkennt QueryType korrekt."""
        from src.logic_layer.planner import QueryType
        plan = controller.planner.plan("What is machine learning?")
        assert plan.query_type in list(QueryType)

    def test_run_returns_dict_with_answer_key(self, controller):
        """run() gibt AgentState-Dict mit 'answer'-Schlüssel zurück."""
        with patch.object(
            controller.verifier, '_call_llm',
            return_value=("Mocked answer.", 0.05)
        ):
            state = controller.run("What is machine learning?")
        assert isinstance(state, dict)
        assert "answer" in state

    def test_run_without_retriever_no_crash(self, controller):
        """run() ohne Retriever läuft durch (leerer Kontext)."""
        with patch.object(
            controller.verifier, '_call_llm',
            return_value=("No information available.", 0.05)
        ):
            state = controller.run("What is the capital of France?")
        assert state is not None
        assert isinstance(state.get("answer", ""), str)

    def test_call_shortcut_returns_string(self, controller):
        """__call__() gibt direkt den Answer-String zurück."""
        with patch.object(
            controller.verifier, '_call_llm',
            return_value=("Test answer.", 0.05)
        ):
            answer = controller("What is 2+2?")
        assert isinstance(answer, str)

    def test_full_pipeline_with_mock_retriever_and_llm(self, controller):
        """Vollständiger S_P → S_N → S_V Flow mit gemockten Abhängigkeiten."""
        class MockResult:
            def __init__(self, text, score):
                self.text = text
                self.rrf_score = score
                self.source_doc = "test.pdf"

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = (
            [MockResult("Paris is the capital of France.", 0.9)],
            {"latency_ms": 5},
        )
        controller.set_retriever(mock_retriever)

        with patch.object(
            controller.verifier, '_call_llm',
            return_value=("Paris is the capital of France.", 0.05)
        ):
            state = controller.run("What is the capital of France?")

        assert state["answer"] != ""
        assert "context" in state
        assert "iterations" in state

    def test_state_has_all_required_keys(self, controller):
        """AgentState enthält alle Pflicht-Schlüssel."""
        with patch.object(
            controller.verifier, '_call_llm',
            return_value=("Test.", 0.05)
        ):
            state = controller.run("test query")
        required_keys = {
            "query", "answer", "context", "iterations",
            "all_verified", "verified_claims", "violated_claims",
        }
        assert required_keys.issubset(set(state.keys()))


# ============================================================================
# 5. AGENT PIPELINE TESTS (pipeline/)
# ============================================================================

class TestAgentPipeline:
    """Tests für den Orchestrator in src/pipeline/."""

    def test_pipeline_result_to_dict(self):
        """PipelineResult.to_dict() enthält alle Pflichtfelder."""
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

    def test_pipeline_result_to_json_valid(self):
        """to_json() produziert valides JSON."""
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

    def test_pipeline_imports(self):
        """AgentPipeline und Factory-Funktionen importierbar."""
        from src.pipeline.agent_pipeline import (
            AgentPipeline, create_pipeline, create_full_pipeline,
        )
        assert AgentPipeline is not None
        assert create_pipeline is not None
        assert create_full_pipeline is not None

    def test_batch_processor_importable(self):
        """BatchProcessor importierbar."""
        from src.pipeline.agent_pipeline import BatchProcessor
        assert BatchProcessor is not None


# ============================================================================
# 6. THESIS COMPLIANCE TESTS
# ============================================================================

class TestThesisCompliance:
    """Überprüft ob Implementierung den Masterthesis-Spezifikationen entspricht."""

    def test_query_types_complete(self):
        """Alle benötigten QueryTypes existieren (Abschnitt 3.2)."""
        from src.logic_layer.planner import QueryType
        required = {"SINGLE_HOP", "MULTI_HOP", "COMPARISON", "TEMPORAL"}
        actual = {q.name for q in QueryType}
        assert required.issubset(actual)

    def test_retrieval_strategies_complete(self):
        """Alle benötigten RetrievalStrategies existieren (Abschnitt 3.3)."""
        from src.logic_layer.planner import RetrievalStrategy
        required = {"VECTOR_ONLY", "GRAPH_ONLY", "HYBRID"}
        actual = {s.name for s in RetrievalStrategy}
        assert required.issubset(actual)

    def test_confidence_levels_complete(self):
        """ConfidenceLevel hat HIGH, MEDIUM, LOW (Abschnitt 3.4)."""
        from src.logic_layer.verifier import ConfidenceLevel
        required = {"HIGH", "MEDIUM", "LOW"}
        actual = {c.name for c in ConfidenceLevel}
        assert required == actual

    def test_verifier_supports_self_correction(self):
        """Verifier führt mindestens 1 Iteration durch (Edge: max_iterations=1 für <30s)."""
        from src.logic_layer.verifier import VerifierConfig
        config = VerifierConfig()
        assert config.max_iterations >= 1

    def test_agent_state_has_required_fields(self):
        """AgentState TypedDict enthält alle Pipeline-Schlüssel."""
        from src.logic_layer.navigator import AgentState
        annotations = AgentState.__annotations__
        required = {
            "query", "answer", "context", "iterations",
            "all_verified", "verified_claims", "violated_claims",
            "retrieval_plan", "errors",
        }
        assert required.issubset(set(annotations.keys()))

    def test_navigator_relevance_threshold_default(self):
        """Relevance Threshold ist 0.6 × max (Abschnitt 3.3)."""
        from src.logic_layer.navigator import ControllerConfig
        config = ControllerConfig()
        assert config.relevance_threshold_factor == 0.6

    def test_navigator_redundancy_threshold_default(self):
        """Redundancy Threshold ist 0.8 Jaccard (Abschnitt 3.3)."""
        from src.logic_layer.navigator import ControllerConfig
        config = ControllerConfig()
        assert config.redundancy_threshold == 0.8

    def test_confidence_boundary_high(self):
        """Exakt 80% verifiziert → HIGH (Grenzwert)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="A",
            iterations=1,
            verified_claims=["c1", "c2", "c3", "c4"],
            violated_claims=["c5"],  # 4/5 = 0.80 → HIGH
        )
        assert result.confidence == ConfidenceLevel.HIGH

    def test_confidence_boundary_medium(self):
        """Exakt 50% verifiziert → MEDIUM (Grenzwert)."""
        from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
        result = VerificationResult(
            answer="A",
            iterations=1,
            verified_claims=["c1"],
            violated_claims=["c2"],  # 1/2 = 0.50 → MEDIUM
        )
        assert result.confidence == ConfidenceLevel.MEDIUM


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
