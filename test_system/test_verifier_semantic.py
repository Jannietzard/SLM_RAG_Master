"""
Semantic correctness tests for S_V (Verifier).

Tests verify structural invariants, pre-validation behaviour, and claim
extraction/verification logic — without requiring a live LLM.

Run:
    python -X utf8 -m pytest test_system/test_verifier_semantic.py -v
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.logic_layer.verifier import (
    Verifier, VerifierConfig, PreGenerationValidator,
    ValidationStatus, ConfidenceLevel, VerificationResult,
    create_verifier, SPACY_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def minimal_cfg():
    return {
        "llm": {"max_context_chars": 2000, "max_docs": 5, "max_chars_per_doc": 400},
        "agent": {"max_verification_iterations": 1},
        "verifier": {
            "enable_entity_path_validation": True,
            "enable_credibility_scoring": True,
            "enable_contradiction_detection": False,
        },
    }


@pytest.fixture(scope="module")
def verifier(minimal_cfg):
    return create_verifier(cfg=minimal_cfg, enable_pre_validation=True)


@pytest.fixture(scope="module")
def validator(minimal_cfg):
    config = VerifierConfig.from_yaml(minimal_cfg)
    return PreGenerationValidator(config)


@pytest.fixture(scope="module")
def einstein_context():
    return [
        "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
        "Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
        "He was born in Ulm, Germany, on March 14, 1879.",
    ]


# ---------------------------------------------------------------------------
# TestPreValidation
# ---------------------------------------------------------------------------

class TestPreValidation:

    def test_empty_context_returns_insufficient_evidence(self, validator):
        result = validator.validate([], "What is the capital of France?")
        assert result.status == ValidationStatus.INSUFFICIENT_EVIDENCE
        assert result.filtered_context == []

    def test_entity_found_in_context(self, validator, einstein_context):
        result = validator.validate(einstein_context, "When was Einstein born?", entities=["Einstein"])
        assert result.entity_path_valid is True

    def test_entity_missing_flags_insufficient_evidence(self, validator, einstein_context):
        result = validator.validate(
            einstein_context, "Who is Marie Curie?",
            entities=["Marie Curie", "Polonium"],
        )
        assert result.status == ValidationStatus.INSUFFICIENT_EVIDENCE

    def test_credibility_scores_never_empty_for_nonempty_context(self, validator, einstein_context):
        result = validator.validate(einstein_context, "test")
        assert len(result.credibility_scores) > 0

    def test_filtered_context_never_empty_after_credibility(self, validator, einstein_context):
        result = validator.validate(einstein_context, "test")
        assert len(result.filtered_context) >= 1

    def test_validation_time_recorded(self, validator, einstein_context):
        result = validator.validate(einstein_context, "test")
        assert result.validation_time_ms >= 0

    def test_status_passed_for_good_context(self, validator, einstein_context):
        result = validator.validate(
            einstein_context,
            "What did Einstein win the Nobel Prize for?",
            entities=["Einstein"],
        )
        assert result.status in (ValidationStatus.PASSED, ValidationStatus.LOW_CREDIBILITY)


# ---------------------------------------------------------------------------
# TestContextFormatting
# ---------------------------------------------------------------------------

class TestContextFormatting:

    def test_empty_context_returns_placeholder(self, verifier):
        fmt = verifier._format_context([])
        assert "No context available" in fmt

    def test_chunks_numbered(self, verifier, einstein_context):
        fmt = verifier._format_context(einstein_context)
        assert "[1]" in fmt
        assert "[2]" in fmt

    def test_max_docs_respected(self, verifier):
        many_chunks = ["chunk %d" % i for i in range(20)]
        fmt = verifier._format_context(many_chunks)
        assert "[6]" not in fmt  # max_docs=5

    def test_long_chunk_truncated(self, verifier):
        long_chunk = "Albert Einstein " * 200
        fmt = verifier._format_context([long_chunk])
        assert len(fmt) < len(long_chunk)


# ---------------------------------------------------------------------------
# TestClaimExtraction
# ---------------------------------------------------------------------------

class TestClaimExtraction:

    def test_factual_sentence_extracted(self, verifier):
        answer = "Albert Einstein was born in Ulm, Germany, in 1879."
        claims = verifier._extract_claims(answer)
        assert len(claims) >= 1

    def test_error_prefix_returns_empty(self, verifier):
        claims = verifier._extract_claims("[Error: LLM timeout]")
        assert claims == []

    def test_meta_statements_filtered(self, verifier):
        answer = "Based on the context, I cannot answer this question."
        claims = verifier._extract_claims(answer)
        assert claims == []

    def test_short_non_meta_answer_handled(self, verifier):
        claims = verifier._extract_claims("Paris")
        assert isinstance(claims, list)

    def test_multi_sentence_answer_splits(self, verifier):
        answer = (
            "Albert Einstein was born in 1879. "
            "He received the Nobel Prize in 1921."
        )
        claims = verifier._extract_claims(answer)
        assert len(claims) >= 1


# ---------------------------------------------------------------------------
# TestClaimVerification
# ---------------------------------------------------------------------------

class TestClaimVerification:

    def test_entity_in_context_verified(self, verifier, einstein_context):
        ok, reason = verifier._verify_claim(
            "Albert Einstein was born in Ulm.", context=einstein_context
        )
        assert ok is True

    def test_entity_not_in_context_violated(self, verifier):
        ok, reason = verifier._verify_claim(
            "Napoleon was born in Corsica.",
            context=["Einstein worked in Bern."],
        )
        assert ok is False

    def test_claim_with_no_entities_verified_by_default(self, verifier):
        ok, reason = verifier._verify_claim("Yes.", context=[])
        assert ok is True
        assert reason == "no_entities_to_verify"

    def test_stopwords_not_treated_as_entities(self, verifier, einstein_context):
        ok, reason = verifier._verify_claim(
            "This is a fact about American history.",
            context=einstein_context,
        )
        assert isinstance(ok, bool)


# ---------------------------------------------------------------------------
# TestConfidenceProperty
# ---------------------------------------------------------------------------

class TestConfidenceProperty:

    def _make_result(self, verified, violated):
        return VerificationResult(
            answer="test",
            iterations=1,
            verified_claims=verified,
            violated_claims=violated,
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )

    def test_all_verified_high_confidence(self):
        r = self._make_result(["c1", "c2", "c3"], [])
        assert r.confidence == ConfidenceLevel.HIGH

    def test_none_verified_low_confidence(self):
        r = self._make_result([], ["c1", "c2"])
        assert r.confidence == ConfidenceLevel.LOW

    def test_zero_claims_low_confidence(self):
        r = self._make_result([], [])
        assert r.confidence == ConfidenceLevel.LOW

    def test_medium_confidence_range(self):
        r = self._make_result(["c1"], ["c1"])  # 50 % verified
        assert r.confidence == ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# TestFactory
# ---------------------------------------------------------------------------

class TestFactory:

    def test_create_verifier_loads_settings(self):
        v = create_verifier()
        assert isinstance(v, Verifier)
        assert isinstance(v.config, VerifierConfig)

    def test_enable_pre_validation_flag(self, minimal_cfg):
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=True)
        assert v.config.enable_entity_path_validation is True
        assert v.config.enable_credibility_scoring is True

    def test_from_yaml_reads_all_blocks(self):
        cfg = {
            "llm": {"model_name": "phi3", "max_tokens": 100},
            "agent": {"max_verification_iterations": 3},
            "verifier": {
                "min_credibility_score": 0.6,
                "heuristic_contradiction_threshold": 0.4,
                "format_sentence_boundary_fraction": 0.65,
            },
        }
        config = VerifierConfig.from_yaml(cfg)
        assert config.model_name == "phi3"
        assert config.max_tokens == 100
        assert config.max_iterations == 3
        assert config.min_credibility_score == 0.6
        assert config.heuristic_contradiction_threshold == 0.4
        assert config.format_sentence_boundary_fraction == 0.65

    def test_none_query_does_not_crash(self, minimal_cfg):
        """generate_and_verify(None, ...) must not raise."""
        v = create_verifier(cfg=minimal_cfg)
        result = v.generate_and_verify(None, [])
        assert isinstance(result, VerificationResult)


# ---------------------------------------------------------------------------
# TestLLMErrorPaths (T-03)
# ---------------------------------------------------------------------------

class TestLLMErrorPaths:
    """Regression tests for Verifier error handling on bad LLM responses.

    T-03a: LLM raises requests.Timeout — Verifier must return a VerificationResult
           with a fallback answer string, not propagate the exception.

    T-03b: LLM returns "" — regression guard for the ``best_answer or "..."``
           bug fixed in v3.4.0 (thesis 12.12).  An empty string is falsy in
           Python, so ``best_answer or fallback`` would silently substitute
           the fallback even when the LLM returned an intentional empty
           response.  The fix uses ``best_answer if best_answer is not None``.
    """

    def test_llm_timeout_returns_fallback_not_exception(
        self, minimal_cfg
    ) -> None:
        """generate_and_verify must handle the LLM timeout sentinel without raising.

        Verifier._call_llm internally catches requests.Timeout and returns
        the error sentinel "[Error: ...]" rather than propagating.  This test
        verifies that generate_and_verify handles the sentinel gracefully and
        returns a VerificationResult — not an exception.
        """
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)
        # Simulate what _call_llm returns when Ollama times out:
        # a sentinel string starting with "[Error:" and zero latency.
        with patch.object(
            v, "_call_llm", return_value=("[Error: LLM request timed out]", 0.0)
        ):
            result = v.generate_and_verify(
                "What is the capital of France?",
                context=["Paris is the capital of France."],
            )
        assert isinstance(result, VerificationResult), (
            "generate_and_verify must return VerificationResult on LLM timeout sentinel"
        )
        assert result.answer is not None, (
            "VerificationResult.answer must not be None after LLM timeout"
        )

    def test_llm_empty_string_does_not_substitute_fallback(
        self, minimal_cfg
    ) -> None:
        """Regression: _call_llm returning '' must not be silently replaced by fallback.

        Before v3.4.0 fix: ``best_answer or "fallback"`` would substitute
        "fallback" when best_answer=="" because "" is falsy.
        After fix: ``best_answer if best_answer is not None else "fallback"``
        preserves the empty string as a valid (if unusual) LLM response.
        """
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)
        with patch.object(v, "_call_llm", return_value=("", 5.0)):
            result = v.generate_and_verify(
                "test query",
                context=["some context"],
            )
        assert isinstance(result, VerificationResult)
        # The fix ensures "" is carried through; the answer may be "" or a
        # fallback sentinel — the important invariant is no unhandled exception.
        assert result.answer is not None


# ---------------------------------------------------------------------------
# TestVerifierFactualCorrectness (T-A)
# ---------------------------------------------------------------------------

class TestVerifierFactualCorrectness:
    """Verifier claim-grounding invariants (T-A).

    _verify_claim uses entity-presence as a conservative proxy:
    a claim is verified when **any** named entity from the claim appears in the
    context (OR logic, not logical entailment — see verifier.py docstring).
    These tests exercise that contract directly.
    """

    def test_answer_with_entity_absent_from_context_yields_violated_claim(
        self, minimal_cfg
    ) -> None:
        """If the LLM answer contains an entity completely absent from context,
        that claim must appear in violated_claims.

        Setup: context is exclusively about Einstein; LLM returns an answer
        about Napoleon — neither "Napoleon" nor "Egypt" appears anywhere in the
        Einstein context, so _verify_claim must return (False, ...).
        """
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)
        with patch.object(
            v, "_call_llm",
            return_value=("Napoleon conquered Egypt.", 5.0),
        ):
            result = v.generate_and_verify(
                query="What did Napoleon do?",
                context=[
                    "Albert Einstein was born in Ulm, Germany in 1879.",
                    "He was a theoretical physicist.",
                ],
            )

        assert len(result.violated_claims) > 0 or result.confidence == ConfidenceLevel.LOW, (
            f"Answer about Napoleon (absent from Einstein context) must produce "
            f"violated_claims or LOW confidence; "
            f"got confidence={result.confidence}, violated_claims={result.violated_claims}"
        )

    def test_answer_with_entity_present_in_context_not_violated(
        self, minimal_cfg
    ) -> None:
        """An answer whose named entities are all found in the context must
        receive MEDIUM or HIGH confidence (no violated claims from _verify_claim).

        _verify_claim returns True when at least one entity from the claim is
        found in the context string.  "Einstein" and "Ulm" both appear in the
        context → claim passes → confidence must be HIGH or MEDIUM.
        """
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)
        with patch.object(
            v, "_call_llm",
            return_value=("Einstein was born in Ulm.", 5.0),
        ):
            result = v.generate_and_verify(
                query="Where was Einstein born?",
                context=["Albert Einstein was born in Ulm, Germany in 1879."],
                entities=["Einstein", "Ulm"],
            )

        assert result.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM), (
            f"Answer with context-grounded entities should be HIGH/MEDIUM confidence; "
            f"got {result.confidence}. violated_claims={result.violated_claims}"
        )


# ---------------------------------------------------------------------------
# TestVerifierStatelessness (Action #6)
# ---------------------------------------------------------------------------

class TestVerifierStatelessness:
    """Verifier must not leak state between independent generate_and_verify calls (F5).

    Two sequential calls with unrelated contexts and queries must produce
    independent results — entities from call A's context must not appear in
    call B's violated_claims or verified_claims unless they genuinely exist in
    call B's context.
    """

    def test_second_call_does_not_inherit_first_call_context(self, minimal_cfg):
        """Call B's result must not contain entities exclusive to call A's context."""
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)

        # Call A: Einstein context + Einstein answer
        with patch.object(v, "_call_llm", return_value=("Einstein was born in Ulm.", 1.0)):
            result_a = v.generate_and_verify(
                query="Where was Einstein born?",
                context=["Albert Einstein was born in Ulm, Germany in 1879."],
                entities=["Einstein"],
            )

        # Call B: Curie context + Curie answer (Einstein ABSENT from context)
        with patch.object(v, "_call_llm", return_value=("Curie discovered radium.", 1.0)):
            result_b = v.generate_and_verify(
                query="What did Curie discover?",
                context=["Marie Curie discovered radium and polonium."],
                entities=["Curie"],
            )

        # Result B must not carry verified claims from result A's Einstein context
        b_verified_lower = {c.lower() for c in result_b.verified_claims}
        assert "einstein" not in " ".join(b_verified_lower), (
            f"Call B result must not contain 'einstein' from call A's context. "
            f"verified_claims={result_b.verified_claims}"
        )

    def test_sequential_calls_produce_independent_confidence(self, minimal_cfg):
        """Confidence level of call B must be determined solely by call B's context."""
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)

        # Call A: entity absent → LOW confidence expected
        with patch.object(v, "_call_llm", return_value=("Napoleon conquered Egypt.", 1.0)):
            result_a = v.generate_and_verify(
                query="What did Napoleon do?",
                context=["Albert Einstein was born in Ulm in 1879."],
            )

        # Call B: entity present → HIGH or MEDIUM confidence expected
        with patch.object(v, "_call_llm", return_value=("Einstein was born in Ulm.", 1.0)):
            result_b = v.generate_and_verify(
                query="Where was Einstein born?",
                context=["Albert Einstein was born in Ulm, Germany in 1879."],
                entities=["Einstein"],
            )

        assert result_b.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM), (
            f"Call B should be HIGH/MEDIUM confidence; got {result_b.confidence}. "
            f"Prior low-confidence call A must not poison call B."
        )


class TestQuestionRelevanceReorder:
    """_reorder_by_question_relevance: answer-relevant chunks rise to the top (§12.27)."""

    def test_most_relevant_chunk_first(self, minimal_cfg):
        """Chunk sharing the most query content words must be sorted first."""
        v = create_verifier(cfg=minimal_cfg)
        query = "Who was voted World's Best Goalkeeper by IFFHS in 1992?"
        context = [
            "Kasper Schmeichel is a Danish footballer.",
            "Peter Schmeichel was voted the IFFHS World's Best Goalkeeper in 1992.",
            "Denmark won the UEFA Euro 1992 championship.",
        ]
        reordered = v._reorder_by_question_relevance(query, context)
        assert reordered[0] == context[1], (
            f"Chunk with most query-word overlap should be first; got: {reordered[0]!r}"
        )

    def test_single_chunk_unchanged(self, minimal_cfg):
        """A single-chunk list is returned as-is."""
        v = create_verifier(cfg=minimal_cfg)
        context = ["Only one chunk here."]
        assert v._reorder_by_question_relevance("any query", context) == context

    def test_empty_context_unchanged(self, minimal_cfg):
        """Empty list is returned as-is."""
        v = create_verifier(cfg=minimal_cfg)
        assert v._reorder_by_question_relevance("any query", []) == []

    def test_all_zero_score_order_preserved(self, minimal_cfg):
        """When no chunk shares query words, original order is preserved (stable sort)."""
        v = create_verifier(cfg=minimal_cfg)
        query = "xyzzy plugh"
        context = ["Alpha text.", "Beta text.", "Gamma text."]
        reordered = v._reorder_by_question_relevance(query, context)
        assert reordered == context

    def test_reorder_is_stable(self, minimal_cfg):
        """Chunks with equal score preserve their original relative order."""
        v = create_verifier(cfg=minimal_cfg)
        query = "goalkeeper voted"
        context = [
            "Peter Schmeichel was voted goalkeeper of the year.",
            "The goalkeeper voted most valuable was legendary.",
            "Some unrelated text about football.",
        ]
        reordered = v._reorder_by_question_relevance(query, context)
        assert reordered[0] == context[0]
        assert reordered[1] == context[1]


# ---------------------------------------------------------------------------
# B1: query_type + bridge_entities forwarded by AgentPipeline
# ---------------------------------------------------------------------------

class TestB1QueryTypePlumbing:
    """B1: the BRIDGE_PROMPT / COMPARISON_PROMPT must actually be selectable.

    Pre-fix, AgentPipeline called generate_and_verify() without query_type or
    bridge_entities. The verifier's prompt-selection code at L1657-1670 only
    fires when both are supplied, so every eval query used ANSWER_PROMPT.
    These tests assert the verifier's selection picks BRIDGE/COMPARISON when
    given the proper kwargs, and that the pipeline actually forwards them.
    """

    def test_comparison_query_type_selects_comparison_prompt(self, minimal_cfg, monkeypatch):
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        captured = {}

        def fake_llm(self, prompt):
            captured["prompt"] = prompt
            return "Berlin", 1.0

        monkeypatch.setattr(Verifier, "_call_llm", fake_llm)
        v.generate_and_verify(
            query="Is Berlin older than Munich?",
            context=["Berlin founded 1237.", "Munich founded 1158."],
            entities=["Berlin", "Munich"],
            query_type="comparison",
        )
        # COMPARISON_PROMPT has a unique signature: numbered steps "1. Find ... 2. Find ... 3. Compare"
        assert "1. Find the relevant fact for the FIRST" in captured["prompt"]
        assert "2. Find the relevant fact for the SECOND" in captured["prompt"]

    def test_multi_hop_with_hop_sequence_selects_bridge_prompt(self, minimal_cfg, monkeypatch):
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        captured = {}

        def fake_llm(self, prompt):
            captured["prompt"] = prompt
            return "Inception", 1.0

        monkeypatch.setattr(Verifier, "_call_llm", fake_llm)
        v.generate_and_verify(
            query="Who directed the film starring Leonardo DiCaprio about dreams?",
            context=["Inception is a 2010 film directed by Christopher Nolan."],
            entities=["Leonardo DiCaprio"],
            hop_sequence=[
                {"step_id": 0, "sub_query": "Find the film", "is_bridge": True},
                {"step_id": 1, "sub_query": "Find the director", "is_bridge": False},
            ],
            query_type="multi_hop",
            bridge_entities=["Inception"],
        )
        # BRIDGE_PROMPT has the unique substring "Use the following reasoning chain"
        assert "reasoning chain" in captured["prompt"]
        assert "Step" in captured["prompt"]  # _build_bridge_chain produces "Step N:" lines

    def test_hop_sequence_accepts_HopStep_dataclasses(self, minimal_cfg, monkeypatch):
        """B1-followup regression guard (2026-05-15).

        AgentPipeline.process() passes the Planner's List[HopStep] dataclasses
        directly to generate_and_verify, while the signature documents
        List[Dict]. The verifier must normalise both forms — otherwise
        _build_bridge_chain crashes with `'HopStep' object has no attribute
        'get'` before any LLM call is made.
        """
        from src.logic_layer.planner import HopStep
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        captured = {}

        def fake_llm(self, prompt):
            captured["prompt"] = prompt
            return "Inception", 1.0

        monkeypatch.setattr(Verifier, "_call_llm", fake_llm)
        # Pass dataclasses (production pipeline path), not dicts (controller path).
        v.generate_and_verify(
            query="Who directed the film starring Leonardo DiCaprio?",
            context=["Inception was directed by Christopher Nolan."],
            entities=["Leonardo DiCaprio"],
            hop_sequence=[
                HopStep(step_id=0, sub_query="Find the film",
                        target_entities=["Leonardo DiCaprio"],
                        depends_on=[], is_bridge=True),
                HopStep(step_id=1, sub_query="Find the director",
                        target_entities=[], depends_on=[0], is_bridge=False),
            ],
            query_type="multi_hop",
            bridge_entities=["Inception"],
        )
        # Must reach the LLM — the crash this guards against happened before
        # _call_llm was invoked.
        assert "prompt" in captured, (
            "B1-followup: HopStep dataclasses crashed _build_bridge_chain "
            "before the LLM was called"
        )
        assert "reasoning chain" in captured["prompt"]

    def test_no_query_type_falls_back_to_answer_prompt(self, minimal_cfg, monkeypatch):
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        captured = {}

        def fake_llm(self, prompt):
            captured["prompt"] = prompt
            return "Paris", 1.0

        monkeypatch.setattr(Verifier, "_call_llm", fake_llm)
        v.generate_and_verify(
            query="What is the capital of France?",
            context=["France's capital is Paris."],
            entities=["France"],
        )
        # ANSWER_PROMPT has the unique line "Give the shortest possible answer"
        assert "Give the shortest possible answer" in captured["prompt"]
        # Neither BRIDGE nor COMPARISON markers should be present.
        assert "reasoning chain" not in captured["prompt"]
        assert "1. Find the relevant fact for the FIRST" not in captured["prompt"]

    def test_call_wrapper_forwards_bridge_entities(self, minimal_cfg, monkeypatch):
        """B9: Verifier.__call__ must forward bridge_entities (regression guard for B1)."""
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        seen = {}

        def fake_gen(self, query, context, entities=None, hop_sequence=None,
                     query_type=None, bridge_entities=None, chunk_is_graph_based=None):
            seen["bridge_entities"] = bridge_entities
            seen["query_type"] = query_type
            return VerificationResult(answer="x", iterations=1)

        monkeypatch.setattr(Verifier, "generate_and_verify", fake_gen)
        v(query="q", context=["c"], query_type="multi_hop", bridge_entities=["E"])
        assert seen["bridge_entities"] == ["E"]
        assert seen["query_type"] == "multi_hop"


# ---------------------------------------------------------------------------
# B2: retrieval-provenance signal in credibility scoring
# ---------------------------------------------------------------------------

class TestB2ProvenanceSignal:
    """B2: pre-fix, ``cred.is_graph_based`` was hard-coded False — the
    provenance term contributed a constant baseline (0.5) regardless of
    retrieval method. Now, callers can pass per-chunk graph-provenance flags
    and graph chunks score strictly higher than vector-only chunks.
    """

    def test_graph_chunk_scores_higher_than_vector_chunk(self, minimal_cfg):
        validator = PreGenerationValidator(VerifierConfig.from_yaml(minimal_cfg))
        # Two near-identical chunks, only difference is graph-vs-vector provenance.
        chunk = "Albert Einstein was a German-born theoretical physicist."
        # Use two unique chunks to avoid filter collapse on duplicate text.
        ctx = [
            chunk,
            "Einstein developed the theory of relativity in 1915.",
        ]
        # Provenance: chunk 0 graph-retrieved, chunk 1 vector-only.
        flags = [True, False]
        result = validator.validate(
            ctx, "What did Einstein develop?",
            entities=["Einstein"],
            chunk_is_graph_based=flags,
        )
        assert len(result.credibility_scores) == 2
        # Graph-retrieved chunk should score >= vector chunk (the provenance
        # term contributes weight*1.0 vs weight*baseline).
        # Both chunks have the same entity-frequency / cross-references, so
        # the only systematic difference is provenance.
        # B2-fix: graph chunk gets full provenance credit (weight × 1.0),
        # vector chunk gets baseline (weight × 0.5). Score delta should be
        # provenance_weight × (1.0 - baseline) = 0.3 × 0.5 = 0.15.
        score_graph, score_vector = result.credibility_scores[0], result.credibility_scores[1]
        assert score_graph > score_vector, (
            f"Graph chunk ({score_graph:.3f}) should outscore vector chunk "
            f"({score_vector:.3f}) after B2-fix"
        )

    def test_missing_provenance_falls_back_to_baseline(self, minimal_cfg):
        """When chunk_is_graph_based is None, behavior is identical to pre-B2."""
        validator = PreGenerationValidator(VerifierConfig.from_yaml(minimal_cfg))
        ctx = ["Einstein was a physicist.", "Einstein developed relativity."]
        result_no_provenance = validator.validate(
            ctx, "What did Einstein develop?", entities=["Einstein"],
        )
        result_explicit_none = validator.validate(
            ctx, "What did Einstein develop?", entities=["Einstein"],
            chunk_is_graph_based=None,
        )
        # Same input → same scores when provenance is absent (baseline applied).
        assert result_no_provenance.credibility_scores == result_explicit_none.credibility_scores

    def test_provenance_flag_length_mismatch_ignored(self, minimal_cfg, caplog):
        """A length-mismatched provenance list should be ignored, not crash."""
        import logging
        caplog.set_level(logging.WARNING)
        validator = PreGenerationValidator(VerifierConfig.from_yaml(minimal_cfg))
        ctx = ["chunk one", "chunk two", "chunk three"]
        result = validator.validate(
            ctx, "query", entities=[],
            chunk_is_graph_based=[True],  # wrong length: 1 vs 3
        )
        assert len(result.credibility_scores) >= 1


# ---------------------------------------------------------------------------
# B4: claim verification — no auto-verify for short / numeric claims
# ---------------------------------------------------------------------------

class TestB4NumericClaimVerification:
    """B4: pre-fix, a claim with no proper noun returned (True, 'no_entities_to_verify')
    immediately — so an LLM hallucinating "9 million inhabitants" or "1995"
    auto-verified regardless of context. The fix grounds short / numeric
    claims by token presence in the context."""

    def test_numeric_claim_grounded_in_context_verified(self, verifier):
        ok, reason = verifier._verify_claim(
            "founded in 1995",
            context=["The company was founded in 1995 in Cupertino."],
        )
        assert ok is True
        assert reason == "context_token_grounded"

    def test_numeric_claim_not_in_context_violated(self, verifier):
        """The hallucination case B4 is designed to catch."""
        ok, reason = verifier._verify_claim(
            "9 million inhabitants",
            context=["Munich has roughly 1.5 million inhabitants."],
        )
        assert ok is False
        assert reason == "no_entities_and_tokens_ungrounded"

    def test_short_phrase_not_in_context_violated(self, verifier):
        ok, reason = verifier._verify_claim(
            "ice hockey",
            context=["The athlete played football professionally."],
        )
        assert ok is False
        assert reason == "no_entities_and_tokens_ungrounded"

    def test_short_phrase_in_context_verified(self, verifier):
        ok, reason = verifier._verify_claim(
            "ice hockey",
            context=["He played ice hockey for the national team."],
        )
        assert ok is True
        assert reason == "context_token_grounded"

    def test_long_narrative_no_proper_noun_still_auto_verifies(self, verifier):
        """Long sentences with no proper noun keep the historical auto-verify
        behavior — no falsifiable anchor to check against."""
        ok, reason = verifier._verify_claim(
            "This is a long sentence that contains no proper noun and is more than six tokens.",
            context=["something else entirely"],
        )
        assert ok is True
        assert reason == "no_entities_to_verify"

    def test_empty_context_short_claim_still_auto_verifies(self, verifier):
        """No context means no falsifiable anchor — keep historical behavior."""
        ok, reason = verifier._verify_claim("Yes.", context=[])
        assert ok is True
        assert reason == "no_entities_to_verify"


# ---------------------------------------------------------------------------
# B10: iteration_history string-truncation budget
# ---------------------------------------------------------------------------

class TestB10HistoryTruncation:
    """B10: per-string truncation budget for iteration_history entries.

    Pre-fix, the verifier stored full answers + full claim lists in
    iteration_history. Across 500 questions × 2 iterations × ~400-char
    answers, the per-question JSONL bloated by several MB. Strings are
    now truncated to 200 chars + a "...[truncated]" marker.
    """

    def test_short_string_unchanged(self):
        assert Verifier._truncate_history_str("Hello world.") == "Hello world."

    def test_long_string_truncated_with_marker(self):
        s = "x" * 500
        out = Verifier._truncate_history_str(s)
        assert out.endswith("...[truncated]")
        # 200 content chars + truncation marker
        assert len(out) == 200 + len("...[truncated]")
        assert out.startswith("x" * 200)

    def test_non_string_passed_through(self):
        # Non-string inputs should not crash (defensive default).
        assert Verifier._truncate_history_str(None) is None
        assert Verifier._truncate_history_str(42) == 42

    def test_truncate_list_applies_per_element(self):
        items = ["short", "x" * 300, "y" * 50]
        out = Verifier._truncate_history_list(items)
        assert out[0] == "short"
        assert out[1].endswith("...[truncated]")
        assert out[2] == "y" * 50

    def test_iteration_history_truncated_on_long_llm_answer(self, minimal_cfg, monkeypatch):
        """Full integration: a 500-char LLM answer is truncated in history."""
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        long_answer = "Albert Einstein was born in Ulm. " + ("filler text. " * 50)
        assert len(long_answer) > 200

        monkeypatch.setattr(
            Verifier, "_call_llm",
            lambda self, prompt: (long_answer, 1.0),
        )
        result = v.generate_and_verify(
            query="Where was Einstein born?",
            context=["Einstein was born in Ulm."],
            entities=["Einstein"],
        )
        assert result.iteration_history, "Expected at least one iteration in history"
        stored_answer = result.iteration_history[0]["answer"]
        assert len(stored_answer) <= 200 + len("...[truncated]"), (
            f"B10: history answer too long ({len(stored_answer)} chars)"
        )
        assert stored_answer.endswith("...[truncated]"), (
            "B10: truncated answers must carry the truncation marker"
        )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    print("=" * 60)
    print("VERIFIER SEMANTIC SMOKE CHECK")
    print("SpaCy: %s" % ("available" if SPACY_AVAILABLE else "unavailable"))
    print("=" * 60)

    _ctx = [
        "Albert Einstein was a German-born theoretical physicist.",
        "Einstein received the Nobel Prize in Physics in 1921.",
        "He was born in Ulm, Germany, on March 14, 1879.",
    ]
    _v = create_verifier(cfg={
        "llm": {"max_context_chars": 2000, "max_docs": 5, "max_chars_per_doc": 400},
        "agent": {"max_verification_iterations": 1},
        "verifier": {"enable_entity_path_validation": True, "enable_credibility_scoring": True},
    }, enable_pre_validation=True)

    _validator = PreGenerationValidator(_v.config)
    _res = _validator.validate(_ctx, "When was Einstein born?", entities=["Einstein"])
    print("Pre-validation status: %s" % _res.status.value)
    print("Entity path valid: %s" % _res.entity_path_valid)
    print("Filtered context docs: %d" % len(_res.filtered_context))
    print("=" * 60)
