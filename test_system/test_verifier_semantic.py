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
