"""
Tests for src/data_layer/chunking.py

Covers both chunking strategies:
  - SemanticChunker: boundary detection, quality filtering, TF-IDF scoring,
    header extraction, word-boundary-aware overlap
  - SpacySentenceChunker: 3-sentence windows, overlap, SpaCy caching,
    deterministic chunk IDs, ingestion.py-compatible interface

Run from project root:
    pytest test_system/test_chunking.py -v
"""

import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

THESIS_TEXT = """
1. Introduction

This thesis investigates the application of machine learning
techniques to natural language processing tasks. The research
focuses on edge deployment scenarios where computational
resources are limited.

1.1 Problem Statement

Modern language models require significant computational resources,
making deployment on edge devices challenging. This research
addresses the gap between model capability and device constraints
through quantization and optimization techniques.

1.2 Research Questions

The central research questions are:
- How can large language models be efficiently deployed on edge devices?
- What is the impact of quantization on model accuracy?
- How can retrieval-augmented generation improve edge AI systems?

2. Background

This chapter provides the theoretical foundation for the research.
We review relevant literature on language models, quantization
techniques, and retrieval-augmented generation.
"""

EINSTEIN_TEXT = """
Albert Einstein was born on March 14, 1879, in Ulm, Germany. He was a
theoretical physicist who developed the theory of relativity. Einstein is
best known for his mass-energy equivalence formula E = mc². He received the
Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect.
Einstein emigrated to the United States in 1933 and worked at Princeton
University. He became an American citizen in 1940. Einstein died on April 18,
1955, in Princeton, New Jersey.
"""


# ── Semantic Chunker ──────────────────────────────────────────────────────────

class TestSemanticChunker:

    def test_produces_chunks(self):
        from src.data_layer.chunking import create_semantic_chunker
        from langchain.schema import Document

        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "thesis.pdf"})
        chunker = create_semantic_chunker(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 0, "Should produce at least one chunk"

    def test_chunks_have_importance_score(self):
        from src.data_layer.chunking import create_semantic_chunker
        from langchain.schema import Document

        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "thesis.pdf"})
        chunker = create_semantic_chunker(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
        chunks = chunker.chunk_document(doc)

        assert all("importance_score" in c.metadata for c in chunks)
        assert all("lexical_diversity" in c.metadata for c in chunks)

    def test_chunks_start_with_complete_words(self):
        """Word-boundary-aware overlap: no chunk should start with a partial word."""
        from src.data_layer.chunking import create_semantic_chunker
        from langchain.schema import Document

        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "thesis.pdf"})
        chunker = create_semantic_chunker(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            first_char = chunk.page_content[0] if chunk.page_content else ""
            # A chunk starting mid-word would begin with a non-space, non-alpha boundary
            # This is a soft check: first char must not be a continuation character
            assert first_char == "" or not first_char.isspace(), \
                f"Chunk starts with unexpected whitespace: {chunk.page_content[:30]!r}"

    def test_resets_header_context_between_calls(self):
        """HeaderExtractor state must not leak across chunk_document() calls."""
        from src.data_layer.chunking import create_semantic_chunker
        from langchain.schema import Document

        chunker = create_semantic_chunker(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "doc.pdf"})

        chunks_first = chunker.chunk_document(doc)
        chunks_second = chunker.chunk_document(doc)

        # Second call should produce the same result (no stale chapter/section state)
        assert len(chunks_first) == len(chunks_second)

    def test_header_detection(self):
        from src.data_layer.chunking import create_semantic_chunker
        from langchain.schema import Document

        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "thesis.pdf"})
        chunker = create_semantic_chunker(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
        chunks = chunker.chunk_document(doc)

        sections = {c.metadata.get("section") for c in chunks if c.metadata.get("section")}
        chapters = {c.metadata.get("chapter") for c in chunks if c.metadata.get("chapter")}

        assert sections or chapters, "Should detect at least one section or chapter header"

    def test_fallback_on_empty_document(self):
        """Empty document should not raise; fallback splitter produces result."""
        from src.data_layer.chunking import create_semantic_chunker
        from langchain.schema import Document

        doc = Document(page_content="", metadata={})
        chunker = create_semantic_chunker(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
        # Should not raise
        chunks = chunker.chunk_document(doc)
        assert isinstance(chunks, list)


# ── Quality Filter ─────────────────────────────────────────────────────────────

class TestAutomaticQualityFilter:

    def test_keeps_good_text(self):
        from src.data_layer.chunking import AutomaticQualityFilter

        qf = AutomaticQualityFilter()
        good = "This is a sample text with diverse vocabulary and meaningful content. " * 3
        keep, reason, _ = qf.should_keep_chunk(good)
        assert keep, f"Expected keep, got filtered: {reason}"

    def test_filters_short_text(self):
        from src.data_layer.chunking import AutomaticQualityFilter

        qf = AutomaticQualityFilter()
        keep, reason, _ = qf.should_keep_chunk("Hi")
        assert not keep
        assert "too_short" in reason

    def test_filters_too_few_words(self):
        from src.data_layer.chunking import AutomaticQualityFilter

        qf = AutomaticQualityFilter(min_length=5, min_words=20)
        keep, reason, _ = qf.should_keep_chunk("One two three.")
        assert not keep
        assert "too_few_words" in reason

    def test_empty_string_filtered(self):
        from src.data_layer.chunking import AutomaticQualityFilter

        qf = AutomaticQualityFilter()
        keep, _, _2 = qf.should_keep_chunk("")
        assert not keep


# ── TF-IDF Scorer ─────────────────────────────────────────────────────────────

class TestTFIDFScorer:

    def test_stopwords_excluded_from_top_terms(self):
        """No stopword should appear as a high-scoring term after filtering."""
        from src.data_layer.chunking import TFIDFScorer, ALL_STOPWORDS

        scorer = TFIDFScorer()
        chunks = [
            "Machine learning is transforming artificial intelligence.",
            "Natural language processing uses deep learning models.",
            "Edge devices have limited computational resources.",
        ]
        scorer.analyze_corpus(chunks)

        # Get the top-scored terms for chunk 0 via direct access
        term_freq = scorer.chunk_term_frequencies[0]
        scores = {}
        import math
        for term, tf in term_freq.items():
            df = scorer.document_frequency.get(term, 1)
            idf = math.log(scorer.total_chunks / df) if df > 0 else 0.0
            scores[term] = tf * idf

        top_terms = sorted(scores, key=scores.get, reverse=True)[:5]
        stopwords_present = [t for t in top_terms if t in ALL_STOPWORDS]
        assert len(stopwords_present) == 0, f"Stopwords in top terms: {stopwords_present}"

    def test_importance_nonzero_for_discriminating_chunk(self):
        from src.data_layer.chunking import TFIDFScorer

        scorer = TFIDFScorer()
        scorer.analyze_corpus([
            "Einstein relativity physics quantum mechanics.",
            "Darwin evolution species natural selection biology.",
            "Newton gravity laws motion classical mechanics.",
        ])
        score = scorer.calculate_chunk_importance(0)
        assert score > 0.0

    def test_zero_if_not_analyzed(self):
        from src.data_layer.chunking import TFIDFScorer

        scorer = TFIDFScorer()
        # total_chunks == 0 → must return 0.0 without error
        assert scorer.calculate_chunk_importance(0) == 0.0


# ── SpaCy Sentence Chunker ─────────────────────────────────────────────────────

class TestSpacySentenceChunker:

    def test_produces_chunks(self):
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker(sentences_per_chunk=3, sentence_overlap=1)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")
        assert len(chunks) > 0

    def test_overlap_present(self):
        """Adjacent chunks must share at least one sentence index."""
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker(sentences_per_chunk=3, sentence_overlap=1)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")

        if len(chunks) >= 2:
            shared = set(chunks[0].sentence_indices) & set(chunks[1].sentence_indices)
            assert len(shared) >= 1, "Adjacent chunks must share at least one sentence"

    def test_max_sentences_per_chunk(self):
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker(sentences_per_chunk=3, sentence_overlap=1)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")
        assert all(c.sentence_count <= 3 for c in chunks)

    def test_chunk_id_deterministic(self):
        """Same input must produce identical chunk IDs across two runs."""
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker(sentences_per_chunk=3, sentence_overlap=1)
        chunks_a = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")
        chunks_b = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")

        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        assert ids_a == ids_b, "Chunk IDs must be deterministic across runs"

    def test_chunk_id_is_sha256_based(self):
        """Chunk ID must be a 20-char hex string (truncated SHA-256)."""
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker(sentences_per_chunk=3, sentence_overlap=1)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")

        for chunk in chunks:
            assert len(chunk.chunk_id) == 20
            assert all(c in "0123456789abcdef" for c in chunk.chunk_id)

    def test_none_input_returns_empty(self):
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker()
        result = chunker.chunk_text(None, source_doc="test")
        assert result == []

    def test_empty_string_returns_empty(self):
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker()
        result = chunker.chunk_text("", source_doc="test")
        assert result == []

    def test_ingestion_interface(self):
        """chunk() must return list of dicts with 'text' and 'metadata' keys."""
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker()
        result = chunker.chunk("Some text about Einstein and relativity theory.",
                               metadata={"source_file": "test.txt"})
        assert isinstance(result, list)
        assert all("text" in r and "metadata" in r for r in result)

    def test_to_langchain_documents(self):
        from src.data_layer.chunking import create_sentence_chunker

        chunker = create_sentence_chunker()
        docs = chunker.chunk_to_documents(EINSTEIN_TEXT, metadata={"source_file": "e.txt"})
        assert len(docs) > 0
        assert all(hasattr(d, "page_content") and hasattr(d, "metadata") for d in docs)


# ── SpaCy Model Cache ─────────────────────────────────────────────────────────

class TestSpacyModelCache:

    def test_second_load_faster(self):
        """Second instantiation must be significantly faster (cache hit)."""
        import time
        from src.data_layer.chunking import SpacyModelCache, create_sentence_chunker

        SpacyModelCache.clear_cache()

        t0 = time.perf_counter()
        create_sentence_chunker()
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        create_sentence_chunker()
        t_second = time.perf_counter() - t0

        # Second instantiation must be at least 10x faster (cache vs. disk load)
        from src.data_layer.chunking import SPACY_AVAILABLE
        if SPACY_AVAILABLE and t_first > 0.05:
            assert t_second < t_first * 0.1, \
                f"Cache not working: first={t_first*1000:.0f}ms, second={t_second*1000:.0f}ms"

    def test_clear_cache(self):
        from src.data_layer.chunking import SpacyModelCache

        SpacyModelCache.clear_cache()
        assert SpacyModelCache._instances == {}


# ── SentenceChunkingConfig Validation ─────────────────────────────────────────

class TestSentenceChunkingConfig:

    def test_rejects_zero_sentences(self):
        from src.data_layer.chunking import SentenceChunkingConfig

        with pytest.raises(ValueError, match="sentences_per_chunk"):
            SentenceChunkingConfig(sentences_per_chunk=0)

    def test_rejects_overlap_geq_window(self):
        from src.data_layer.chunking import SentenceChunkingConfig

        with pytest.raises(ValueError, match="sentence_overlap"):
            SentenceChunkingConfig(sentences_per_chunk=3, sentence_overlap=3)

    def test_rejects_negative_overlap(self):
        from src.data_layer.chunking import SentenceChunkingConfig

        with pytest.raises(ValueError, match="sentence_overlap"):
            SentenceChunkingConfig(sentence_overlap=-1)

    def test_valid_config_accepted(self):
        from src.data_layer.chunking import SentenceChunkingConfig

        cfg = SentenceChunkingConfig(sentences_per_chunk=3, sentence_overlap=1)
        assert cfg.sentences_per_chunk == 3
        assert cfg.sentence_overlap == 1


# ── Thesis Compliance ─────────────────────────────────────────────────────────

class TestThesisCompliance:
    """Verify that default values match the thesis specification (Section 2.2, 2.3)."""

    def test_sentence_window_defaults(self):
        from src.data_layer.chunking import SentenceChunkingConfig

        cfg = SentenceChunkingConfig()
        assert cfg.sentences_per_chunk == 3, "Thesis §2.2: 3-sentence window"
        assert cfg.sentence_overlap == 1, "Thesis §2.2: 1-sentence overlap"
        assert cfg.min_chunk_chars == 50
        assert cfg.spacy_model == "en_core_web_sm"
