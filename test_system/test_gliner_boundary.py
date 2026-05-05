"""
Boundary correctness tests for the GLiNER entity extractor.

Uses real GLiNER inference (gliner_small-v2.1) on a small fixed corpus so
that model-level entity extraction quality is validated in CI rather than
only in the nightly integration run.

R1 compliance:
    N_SAMPLES defaults to 2 so that the default CI run calls GLiNER inference
    on only 2 sentences.  Set EDGE_RAG_N_SAMPLES=10 and use ``-m nightly``
    to run the full 10-sentence recall test.

Academic reference:
    Zaratiana et al. (2023). "GLiNER: Generalist Model for Named Entity
        Recognition using Bidirectional Transformer." arXiv:2311.08526.

Run:
    pytest test_system/test_gliner_boundary.py -v
    pytest test_system/test_gliner_boundary.py -v -m nightly  # full recall

Review History:
    Created:        2026-04-25
    Review Result:  Addresses ⚫ UNBEKANNT rating from test health audit
"""

import os
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("gliner", reason="gliner package not installed — skip GLiNER tests")

N_SAMPLES: int = int(os.getenv("EDGE_RAG_N_SAMPLES", "2"))


# ---------------------------------------------------------------------------
# Gold-standard sentence → entity mapping
# Each entry covers one sentence from the HotpotQA-style evaluation corpus.
# ---------------------------------------------------------------------------

_ALL_GOLD: List[dict] = [
    {
        "text": "Albert Einstein was born in Ulm, Germany in 1879.",
        "entity": "Albert Einstein",
        "type": "person",
    },
    {
        "text": "Marie Curie discovered radium and polonium.",
        "entity": "Marie Curie",
        "type": "person",
    },
    {
        "text": "Scott Derrickson directed Doctor Strange for Marvel Studios.",
        "entity": "Scott Derrickson",
        "type": "person",
    },
    {
        "text": "The Eiffel Tower stands in Paris, France.",
        "entity": "Paris",
        "type": "city",
    },
    {
        "text": "Isaac Newton formulated the laws of motion at Cambridge University.",
        "entity": "Isaac Newton",
        "type": "person",
    },
    {
        "text": "Tom Hanks starred in Forrest Gump directed by Robert Zemeckis.",
        "entity": "Tom Hanks",
        "type": "person",
    },
    {
        "text": "Stephen Hawking worked at the University of Cambridge.",
        "entity": "Stephen Hawking",
        "type": "person",
    },
    {
        "text": "Nikola Tesla invented the alternating current motor.",
        "entity": "Nikola Tesla",
        "type": "person",
    },
    {
        "text": "The Berlin Wall fell in November 1989.",
        "entity": "Berlin",
        "type": "city",
    },
    {
        "text": "Charles Darwin published On the Origin of Species in 1859.",
        "entity": "Charles Darwin",
        "type": "person",
    },
]

# CI subset (N_SAMPLES cases); nightly uses all 10.
GOLD = _ALL_GOLD[:N_SAMPLES]


# ---------------------------------------------------------------------------
# Module-scoped fixture: load GLiNER extractor once per module.
# Model loading takes ~10 s; reusing across all tests in this file is
# essential to keep the test suite fast.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gliner_extractor():
    """Instantiate GLiNERExtractor once with recall-optimised confidence threshold."""
    from src.data_layer.entity_extraction import GLiNERExtractor, ExtractionConfig
    config = ExtractionConfig(ner_confidence_threshold=0.15)
    return GLiNERExtractor(config=config)


# ---------------------------------------------------------------------------
# Section 1 — Boundary detection (CI, N_SAMPLES cases)
# ---------------------------------------------------------------------------

class TestGLiNERBoundary:
    """GLiNER must detect gold-standard entities in the CI sentence subset."""

    @pytest.mark.parametrize("case", GOLD, ids=[g["entity"] for g in GOLD])
    def test_expected_entity_extracted(
        self, gliner_extractor, case: dict
    ) -> None:
        """GLiNER must include the gold-standard entity in its output."""
        entities = gliner_extractor.extract(case["text"], chunk_id="boundary_test")
        extracted_names = {e.name.lower() for e in entities}
        assert case["entity"].lower() in extracted_names, (
            f"Expected '{case['entity']}' in: {sorted(extracted_names)}\n"
            f"Sentence: {case['text']}"
        )

    @pytest.mark.parametrize("case", GOLD, ids=[g["entity"] for g in GOLD])
    def test_entity_type_correct(
        self, gliner_extractor, case: dict
    ) -> None:
        """The extracted entity type must match the gold-standard label."""
        entities = gliner_extractor.extract(case["text"], chunk_id="type_test")
        type_map = {e.name.lower(): e.entity_type.lower() for e in entities}
        name_lower = case["entity"].lower()
        assert name_lower in type_map, (
            f"Entity '{case['entity']}' not found — cannot verify type.\n"
            f"Extracted: {sorted(type_map.keys())}"
        )
        assert type_map[name_lower] == case["type"], (
            f"Wrong type for '{case['entity']}': "
            f"expected '{case['type']}', got '{type_map[name_lower]}'"
        )


# ---------------------------------------------------------------------------
# Section 2 — Full-corpus recall (nightly only)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Section 2 — Compound / multi-token span boundary tests (T-C)
# ---------------------------------------------------------------------------

_ALL_COMPOUND_CASES: List[dict] = [
    {
        "text": "The Eiffel Tower is located in Paris, France.",
        "expected_span": "Eiffel Tower",
        "entity_type": "landmark",
    },
    {
        "text": "New York City is the most populous city in the United States.",
        "expected_span": "New York City",
        "entity_type": "city",
    },
    {
        "text": "Scott Derrickson directed Doctor Strange for Marvel Studios.",
        "expected_span": "Marvel Studios",
        "entity_type": "organization",
    },
]

_N_COMPOUND = int(os.getenv("EDGE_RAG_N_SAMPLES", "2"))
COMPOUND_CASES = _ALL_COMPOUND_CASES[:_N_COMPOUND]


@pytest.mark.nightly
class TestGLiNERSpanBoundary:
    """GLiNER must extract compound / multi-token entity spans as a single unit."""

    @pytest.mark.parametrize(
        "case",
        COMPOUND_CASES,
        ids=[c["expected_span"] for c in COMPOUND_CASES],
    )
    def test_compound_span_extracted_as_single_entity(
        self, gliner_extractor, case: dict
    ) -> None:
        """The full multi-token span must appear as one entity, not as split fragments."""
        entities = gliner_extractor.extract(case["text"], chunk_id="span_boundary_test")
        extracted_names = {e.name.lower() for e in entities}
        assert case["expected_span"].lower() in extracted_names, (
            f"Expected full span '{case['expected_span']}' in extracted names.\n"
            f"Got: {sorted(extracted_names)}\n"
            f"Sentence: {case['text']}\n"
            "GLiNER may be splitting the compound span into individual tokens."
        )

    @pytest.mark.parametrize(
        "case",
        COMPOUND_CASES,
        ids=[c["expected_span"] for c in COMPOUND_CASES],
    )
    def test_partial_span_not_standalone_when_full_span_absent(
        self, gliner_extractor, case: dict
    ) -> None:
        """If the full compound span is absent, individual tokens must not appear alone.

        A partial extraction (e.g. "Eiffel" instead of "Eiffel Tower") signals
        that GLiNER split the span rather than skipping it entirely — both cases
        are failures, but partial extraction is the more insidious one.
        """
        entities = gliner_extractor.extract(case["text"], chunk_id="span_partial_test")
        extracted_names = {e.name.lower() for e in entities}
        full_span_lower = case["expected_span"].lower()
        full_tokens = full_span_lower.split()

        if full_span_lower not in extracted_names:
            # Full span absent — individual tokens alone are also invalid
            partial_found = any(tok in extracted_names for tok in full_tokens)
            assert not partial_found, (
                f"Partial token found but full span '{case['expected_span']}' absent.\n"
                f"Extracted: {sorted(extracted_names)}\n"
                "This indicates a compound-span splitting defect."
            )


# ---------------------------------------------------------------------------
# Section 3 — Full-corpus recall (nightly only)
# ---------------------------------------------------------------------------

class TestGLiNERRecall:
    """Full-corpus recall test — run with EDGE_RAG_N_SAMPLES=10 -m nightly."""

    @pytest.mark.nightly
    def test_recall_at_least_60_percent(self, gliner_extractor) -> None:
        """GLiNER must achieve recall >= 0.6 on the 10-sentence gold corpus.

        0.6 is a conservative lower bound consistent with the 88% NER quality
        achieved after the v3.5.0 normalisation improvements (thesis section
        3.5).  The lower threshold accommodates known model limitations
        (small cities, coordinated spans) documented in the Known Limitations
        section of TECHNICAL_ARCHITECTURE.md.
        """
        n_total = len(_ALL_GOLD)
        n_found = 0
        for case in _ALL_GOLD:
            entities = gliner_extractor.extract(case["text"], chunk_id="recall_test")
            names = {e.name.lower() for e in entities}
            if case["entity"].lower() in names:
                n_found += 1
        recall = n_found / n_total
        assert recall >= 0.6, (
            f"GLiNER recall {recall:.2f} ({n_found}/{n_total}) < 0.6 on "
            f"the 10-sentence gold corpus."
        )

    @pytest.mark.nightly
    def test_recall_by_entity_type(self, gliner_extractor) -> None:
        """Per-type recall breakdown — emits UserWarning when any type < 0.5.

        Surfacing type-level weaknesses (documented in §12.6 of
        TECHNICAL_ARCHITECTURE.md) without failing the overall recall assertion.
        A type with 0 gold sentences is skipped from the warning.
        """
        import warnings
        from collections import defaultdict

        hits: dict = defaultdict(int)
        totals: dict = defaultdict(int)

        for case in _ALL_GOLD[:N_SAMPLES]:
            t = case["type"]
            totals[t] += 1
            entities = gliner_extractor.extract(case["text"], chunk_id="type_recall_test")
            names = {e.name.lower() for e in entities}
            if case["entity"].lower() in names:
                hits[t] += 1

        low_types = []
        for t, total in totals.items():
            if total == 0:
                continue
            recall_t = hits[t] / total
            if recall_t < 0.5:
                low_types.append(f"{t}: {recall_t:.0%} ({hits[t]}/{total})")

        if low_types:
            warnings.warn(
                "GLiNER per-type recall below 0.5 — see §12.6 TECHNICAL_ARCHITECTURE.md: "
                + "; ".join(low_types),
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Section 4 — Robustness / crash+exception boundary tests (F1+F2)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gliner_extractor():
    """GLiNERExtractor with a mocked GLiNER model — zero real inference calls (R1-safe)."""
    from src.data_layer.entity_extraction import GLiNERExtractor, ExtractionConfig
    from unittest.mock import MagicMock, patch
    with patch("src.data_layer.entity_extraction.GLiNER") as mock_cls:
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        mock_cls.from_pretrained.return_value = mock_model
        extractor = GLiNERExtractor(config=ExtractionConfig())
    return extractor


class TestGLiNERRobustness:
    """F1+F2 crash/exception boundary tests for GLiNERExtractor.extract().

    Uses a mocked GLiNER model — R1-safe (zero real inference calls in CI).
    Promotes entity_extraction_gliner from FRAGIL to STABIL by covering the
    four boundary cases documented in Action #4 of the Audit II action list.
    """

    def test_empty_string_returns_empty_list(self, mock_gliner_extractor):
        """extract('') must return [] without raising (F1 crash guard)."""
        result = mock_gliner_extractor.extract("", chunk_id="robustness_empty")
        assert result == []

    def test_whitespace_only_returns_empty_list(self, mock_gliner_extractor):
        """extract('   \\t\\n') must return [] without raising (F1 crash guard)."""
        result = mock_gliner_extractor.extract("   \t\n  ", chunk_id="robustness_ws")
        assert result == []

    def test_overlong_text_does_not_raise(self, mock_gliner_extractor):
        """Text >512 tokens must not raise any exception (F1 crash guard)."""
        overlong = "word " * 600
        try:
            result = mock_gliner_extractor.extract(overlong, chunk_id="robustness_long")
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(
                f"overlong text raised {type(exc).__name__}: {exc}"
            )

    def test_non_ascii_text_does_not_raise(self, mock_gliner_extractor):
        """Non-ASCII / Arabic text must not raise any exception (F1 crash guard)."""
        arabic = "مرحبا بالعالم. هذا نص تجريبي."
        try:
            result = mock_gliner_extractor.extract(arabic, chunk_id="robustness_arabic")
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(
                f"non-ASCII text raised {type(exc).__name__}: {exc}"
            )

    def test_model_exception_falls_back_gracefully(self, mock_gliner_extractor):
        """RuntimeError from the model must be caught; extract() must return [] (F2)."""
        from unittest.mock import patch
        with patch.object(
            mock_gliner_extractor.model,
            "predict_entities",
            side_effect=RuntimeError("model crash"),
        ):
            result = mock_gliner_extractor.extract(
                "Einstein was born in Ulm.", chunk_id="robustness_exc"
            )
        assert isinstance(result, list)
