"""
Chunking Module: Document Segmentation for the Edge-RAG System

Version: 4.2.0
Last Modified: 2026-04-03

================================================================================
ARCHITECTURAL ROLE
================================================================================

This module implements the document segmentation stage of Artifact A (Data
Layer). It sits between raw document intake and the vector/graph storage layer,
producing LangChain Document objects consumed by ingestion.py → HybridStore
(LanceDB + KuzuDB).

Two complementary strategies are provided:

1. SENTENCE-BASED CHUNKING (SpacySentenceChunker) — primary strategy
   Implements the 3-sentence sliding-window approach specified in Thesis
   Section 2.2. Produces overlapping context windows that preserve entity
   bridges across chunk boundaries, which is critical for multi-hop retrieval
   in HotpotQA-style queries.

2. SEMANTIC CHUNKING (SemanticChunker) — alternative for structured documents
   Segments documents at structural and statistical boundaries using TF-IDF
   importance scoring and Shannon-entropy-based quality filtering. Designed for
   thesis-style documents with numbered section hierarchies.

================================================================================
DESIGN DECISIONS
================================================================================

3-Sentence Window:
    Empirical evaluation on HotpotQA (Thesis Section 4.2) demonstrated that
    3-sentence windows achieve the best F1/recall tradeoff for bridge queries.
    An overlap of 1 sentence prevents entity-bridge fragmentation at boundaries.
    Reference: Lewis, P. et al. (2020). "Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks." NeurIPS 2020.

Quality Filtering (Semantic Chunker):
    Chunks pass the quality gate only if they satisfy all of: minimum length,
    lexical diversity (type-token ratio), information density (Shannon entropy),
    and absence of transcript/whitespace artifacts. Thresholds are configurable
    via settings.yaml (ingestion.quality_filter.*).

Deterministic Chunk IDs:
    Chunk identifiers are derived from SHA-256(source_doc + position + text
    prefix). This ensures that re-ingestion produces identical KuzuDB node IDs,
    preserving graph integrity across runs.

TF-IDF Importance Scoring:
    Chunk importance is the mean TF-IDF score over all non-stopword content
    terms, normalised by total term count. Stopwords are excluded from both TF
    and DF to prevent high-frequency function words from dominating.
    Reference: Salton, G. & Buckley, C. (1988). "Term-weighting approaches in
    automatic text retrieval." Information Processing & Management, 24(5),
    513–523.

================================================================================
CONFIGURATION
================================================================================

All tunable parameters originate from config/settings.yaml. Defaults in this
file serve as documented emergency fallbacks only.

  ingestion.sentences_per_chunk     → SpacySentenceChunker.sentences_per_chunk (3)
  ingestion.sentence_overlap        → SpacySentenceChunker.sentence_overlap (1)
  ingestion.min_chunk_size          → SentenceChunkingConfig.min_chunk_chars (50)
  ingestion.max_chunk_chars         → SentenceChunkingConfig.max_chunk_chars (2000)
  ingestion.word_boundary_factor    → SemanticBoundaryDetector.word_boundary_factor (0.8)
  ingestion.spacy_model             → SentenceChunkingConfig.spacy_model ("en_core_web_sm")
  ingestion.entity_aware_chunking   → SpacySentenceChunker.entity_aware (false)
  ingestion.min_lexical_diversity   → AutomaticQualityFilter.min_lexical_diversity (0.3)
  ingestion.min_information_density → AutomaticQualityFilter.min_information_density (2.0)
  ingestion.quality_filter.min_length → AutomaticQualityFilter.min_length (100)
  ingestion.quality_filter.min_words  → AutomaticQualityFilter.min_words (15)
  chunking.chunk_size               → SemanticChunker.max_chunk_size (1024)
  chunking.chunk_overlap            → SemanticChunker.overlap (128)
  chunking.semantic.min_chunk_size  → SemanticChunker.min_chunk_size (200)

================================================================================
USAGE
================================================================================

Sentence-Based Chunking (primary):
    from src.data_layer.chunking import create_sentence_chunker
    chunker = create_sentence_chunker(sentences_per_chunk=3, sentence_overlap=1)
    chunks = chunker.chunk_text(text, source_doc="document.txt")

Semantic Chunking (structured documents):
    from src.data_layer.chunking import create_semantic_chunker
    chunker = create_semantic_chunker(chunk_size=1024, chunk_overlap=128)
    chunks = chunker.chunk_document(document)

================================================================================
"""

import re
import math
import logging
import hashlib
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter

if TYPE_CHECKING:
    from spacy.language import Language

logger = logging.getLogger(__name__)

# ─── LangChain ────────────────────────────────────────────────────────────────

try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning(
        "LangChain not installed — RecursiveCharacterTextSplitter unavailable. "
        "Install with: pip install langchain"
    )

    class Document:  # type: ignore[no-redef]
        """Minimal Document stub used only when langchain is absent."""
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
        """Minimal text splitter stub used only when langchain is absent."""
        def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", " ", ""]

        def split_documents(self, documents):
            result = []
            for doc in documents:
                text = doc.page_content
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                    chunk_text = text[i:i + self.chunk_size].strip()
                    if chunk_text:
                        result.append(Document(
                            page_content=chunk_text,
                            metadata=doc.metadata.copy()
                        ))
            return result

# ─── SpaCy ────────────────────────────────────────────────────────────────────

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning(
        "SpaCy not available — sentence chunker falls back to regex segmentation. "
        "Install with: pip install spacy && python -m spacy download en_core_web_sm"
    )


# ============================================================================
# SPACY MODEL CACHE
# ============================================================================

class SpacyModelCache:
    """
    Module-level singleton cache for SpaCy Language models.

    Prevents repeated disk loads when multiple chunkers are instantiated in
    the same process (e.g., parallel ingestion workers). Thread-safety is not
    guaranteed; this is acceptable for the target single-threaded edge pipeline.
    """

    _instances: Dict[str, Any] = {}

    @classmethod
    def get_model(
        cls, model_name: str, disable: List[str] = None
    ) -> Optional["Language"]:
        """Return a cached SpaCy Language model, loading it from disk on first access."""
        if not SPACY_AVAILABLE:
            return None

        disable = disable or []
        cache_key = f"{model_name}__{'_'.join(sorted(disable))}"

        if cache_key not in cls._instances:
            try:
                nlp = spacy.load(model_name, disable=disable)
                # Add a sentencizer only if neither senter nor sentencizer is present.
                # SpaCy small models ship with a rule-based sentencizer; full
                # pipeline models include the neural senter. We add the lightweight
                # rule-based sentencizer as a safe default when neither is active.
                if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                cls._instances[cache_key] = nlp
                logger.info(f"SpaCy model loaded and cached: {model_name}")
            except OSError as e:
                logger.warning(
                    f"SpaCy model '{model_name}' not found: {e}. "
                    "Falling back to regex sentence segmentation."
                )
                cls._instances[cache_key] = None

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls) -> None:
        """Evict all cached models. Primarily used in tests to reset state."""
        cls._instances.clear()


# ============================================================================
# STOPWORDS FOR TF-IDF SCORING
# ============================================================================

ENGLISH_STOPWORDS = frozenset({
    'a', 'an', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'may',
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'at',
    'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'by',
    'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'near', 'of',
    'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'since', 'through',
    'throughout', 'till', 'to', 'toward', 'towards', 'under', 'underneath', 'until',
    'unto', 'up', 'upon', 'with', 'within', 'without', 'and', 'but', 'or', 'nor',
    'yet', 'so', 'both', 'either', 'neither', 'not', 'only', 'own', 'same', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'every', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'any', 'many', 'much', 'as', 'if', 'then', 'because', 'while', 'although',
    'though', 'once', 'unless', 'whether', 's', 't', 'd', 'll', 've', 're', 'm',
})

GERMAN_STOPWORDS = frozenset({
    'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'einem',
    'einen', 'eines', 'und', 'oder', 'aber', 'wenn', 'weil', 'dass', 'ist', 'sind',
    'war', 'waren', 'wird', 'werden', 'wurde', 'wurden', 'hat', 'haben', 'hatte',
    'hatten', 'sein', 'ihr', 'ihre', 'ihrer', 'ihrem', 'ihren', 'sich', 'auch',
    'als', 'so', 'wie', 'bei', 'mit', 'zu', 'zur', 'zum', 'von', 'vom', 'für',
    'auf', 'aus', 'an', 'in', 'im', 'am', 'um', 'nach', 'über', 'unter', 'vor',
    'hinter', 'neben', 'zwischen', 'durch', 'gegen', 'ohne', 'bis', 'seit',
    'während', 'trotz', 'wegen', 'es', 'er', 'sie', 'wir', 'ich', 'du', 'man',
    'nicht', 'nur', 'noch', 'schon', 'sehr', 'mehr', 'kann', 'können', 'muss',
    'müssen', 'soll', 'sollen', 'will', 'wollen', 'darf', 'dürfen',
})

ALL_STOPWORDS = ENGLISH_STOPWORDS | GERMAN_STOPWORDS


# ============================================================================
# PART 1: SEMANTIC CHUNKING
# ============================================================================

@dataclass
class ChunkMetadata:
    """Structured metadata for a semantically-chunked document segment."""

    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    heading_level: int = 0
    is_header: bool = False
    page_number: Optional[int] = None
    importance_score: float = 0.0
    lexical_diversity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "heading_level": self.heading_level,
            "is_header": self.is_header,
            "page_number": self.page_number,
            "importance_score": self.importance_score,
            "lexical_diversity": self.lexical_diversity,
        }


class HeaderExtractor:
    """
    Extract hierarchical document structure using language-agnostic regex patterns.

    Maintains stateful chapter/section context across successive calls to
    extract_headers() within a single document traversal. Call reset() before
    processing a new document to prevent context from leaking across document
    boundaries. SemanticChunker.chunk_document() calls reset() automatically.
    """

    CHAPTER_PATTERNS = [
        r'^(\d+)\.\s+([A-Z\u00C0-\u024F][^\n]{3,80})$',
        r'^([IVX]+)\.\s+([A-Z\u00C0-\u024F][^\n]{3,80})$',
        r'^\w+\s+(\d+)[:\s]+([^\n]{3,80})$',
    ]

    SECTION_PATTERNS = [
        r'^(\d+\.\d+)[\.\s]+([A-Z\u00C0-\u024F][^\n]{3,80})$',
    ]

    SUBSECTION_PATTERNS = [
        r'^(\d+\.\d+\.\d+)[\.\s]+([A-Z\u00C0-\u024F][^\n]{3,80})$',
    ]

    def __init__(self):
        self.current_chapter: Optional[str] = None
        self.current_section: Optional[str] = None
        self.current_subsection: Optional[str] = None

    def reset(self) -> None:
        """Reset document-level header context. Must be called between documents."""
        self.current_chapter = None
        self.current_section = None
        self.current_subsection = None

    def _scan_all_headers(self, text: str) -> None:
        """Scan all lines of a chunk to update the running chapter/section context."""
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            for pattern in self.CHAPTER_PATTERNS:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    number, title = match.groups()
                    self.current_chapter = f"{number}. {title}"
                    self.current_section = None
                    self.current_subsection = None
                    break
            for pattern in self.SECTION_PATTERNS:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    number, title = match.groups()
                    self.current_section = f"{number} {title}"
                    self.current_subsection = None
                    break
            for pattern in self.SUBSECTION_PATTERNS:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    number, title = match.groups()
                    self.current_subsection = f"{number} {title}"
                    break

    def extract_headers(self, text: str) -> Tuple[ChunkMetadata, str]:
        """
        Extract header information from a text chunk.

        Updates the running chapter/section context, detects whether the first
        line is a header, and returns a (ChunkMetadata, cleaned_text) pair
        where cleaned_text has the leading header line removed if applicable.
        """
        self._scan_all_headers(text)

        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""

        metadata = ChunkMetadata()
        is_first_line_header = False
        header_level = 0

        for pattern in self.CHAPTER_PATTERNS:
            if re.match(pattern, first_line, re.MULTILINE):
                is_first_line_header, header_level = True, 1
                break
        if not is_first_line_header:
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, first_line, re.MULTILINE):
                    is_first_line_header, header_level = True, 2
                    break
        if not is_first_line_header:
            for pattern in self.SUBSECTION_PATTERNS:
                if re.match(pattern, first_line, re.MULTILINE):
                    is_first_line_header, header_level = True, 3
                    break

        metadata.chapter = self.current_chapter
        metadata.section = self.current_section
        metadata.subsection = self.current_subsection
        metadata.heading_level = header_level
        metadata.is_header = is_first_line_header

        cleaned_text = '\n'.join(lines[1:]).strip() if is_first_line_header else text
        return metadata, cleaned_text


class SemanticBoundaryDetector:
    """
    Detect natural semantic boundaries in text for the SemanticChunker.

    Paragraph breaks (double newlines), sentence-final newlines before
    uppercase characters, and colon-terminated lines are treated as boundary
    candidates. A candidate is accepted only when the distance from the
    previous accepted boundary exceeds word_boundary_factor * max_chunk_size,
    preventing excessively short chunks.

    word_boundary_factor maps to settings.yaml: ingestion.word_boundary_factor
    (default 0.8).
    """

    BOUNDARY_PATTERNS = [
        r'\n\n+',
        r'\.\s*\n(?=[A-Z\u00C0-\u024F])',
        r'[.!?]\s*\n\s*\n',
        r':\s*\n',
    ]

    def __init__(
        self,
        min_boundary_distance: int = 200,
        word_boundary_factor: float = 0.8,  # settings.yaml: ingestion.word_boundary_factor
    ):
        self.min_boundary_distance = min_boundary_distance
        self.word_boundary_factor = word_boundary_factor

    def find_semantic_boundaries(self, text: str, max_chunk_size: int = 1024) -> List[int]:
        """
        Return sorted character positions at which the text may be split.

        A boundary is accepted when the distance from the previous boundary is
        >= word_boundary_factor * max_chunk_size AND >= min_boundary_distance.
        """
        boundaries = [0]
        potential_boundaries: List[int] = []

        for pattern in self.BOUNDARY_PATTERNS:
            for match in re.finditer(pattern, text):
                pos = match.end()
                if pos >= self.min_boundary_distance:
                    potential_boundaries.append(pos)

        potential_boundaries = sorted(set(potential_boundaries))

        current_position = 0
        acceptance_threshold = self.word_boundary_factor * max_chunk_size

        for boundary in potential_boundaries:
            distance = boundary - current_position
            if distance >= self.min_boundary_distance and distance >= acceptance_threshold:
                boundaries.append(boundary)
                current_position = boundary

        if boundaries[-1] != len(text):
            boundaries.append(len(text))

        return boundaries


class AutomaticQualityFilter:
    """
    Statistical quality gate for text chunks.

    A chunk is retained only if it passes all four filters (applied in order
    of ascending computational cost). Thresholds map to settings.yaml under
    ingestion.quality_filter.*
    """

    def __init__(
        self,
        min_length: int = 100,                 # settings.yaml: ingestion.quality_filter.min_length
        min_words: int = 15,                   # settings.yaml: ingestion.quality_filter.min_words
        min_lexical_diversity: float = 0.3,    # settings.yaml: ingestion.min_lexical_diversity
        min_information_density: float = 2.0,  # settings.yaml: ingestion.min_information_density
    ):
        self.min_length = min_length
        self.min_words = min_words
        self.min_lexical_diversity = min_lexical_diversity
        self.min_information_density = min_information_density

    def calculate_lexical_diversity(self, text: str) -> float:
        """
        Compute type-token ratio (TTR) as a measure of vocabulary richness.

        TTR = |unique_tokens| / |total_tokens|, range [0, 1]. Values below
        min_lexical_diversity indicate repetitive or boilerplate text.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def calculate_information_density(self, text: str) -> float:
        """
        Compute Shannon entropy of the unigram word distribution.

        H = -sum_i p_i * log2(p_i) over all word types.

        Reference: Shannon, C.E. (1948). "A Mathematical Theory of
        Communication." Bell System Technical Journal, 27, 379–423.

        Low entropy (< min_information_density) indicates layout artifacts,
        highly repetitive lists, or near-empty chunks.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        word_counts = Counter(words)
        total_words = len(words)

        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def detect_transcript_pattern(self, text: str) -> bool:
        """Return True if the text resembles a dialog transcript (short speaker labels)."""
        pattern = r'(?:^|\n)\s*\w{1,3}\s*:\s*.{10,}'
        matches = re.findall(pattern, text)
        lines = text.split('\n')
        if not lines:
            return False
        return len(matches) / len(lines) > 0.3

    def detect_excessive_whitespace(self, text: str) -> bool:
        """Return True if whitespace exceeds 40% of text length (layout artifact)."""
        if not text:
            return True
        return (text.count(' ') + text.count('\t')) / len(text) > 0.4

    def should_keep_chunk(self, text: str) -> Tuple[bool, str]:
        """
        Apply all quality filters in sequence. Returns (keep, reason_string).

        Filters are ordered from cheapest to most expensive.
        """
        if len(text) < self.min_length:
            return False, f"too_short ({len(text)} chars)"

        words = re.findall(r'\b\w+\b', text)
        if len(words) < self.min_words:
            return False, f"too_few_words ({len(words)} words)"

        if self.detect_transcript_pattern(text):
            return False, "transcript_pattern_detected"

        diversity = self.calculate_lexical_diversity(text)
        if diversity < self.min_lexical_diversity:
            return False, f"low_lexical_diversity ({diversity:.2f})"

        density = self.calculate_information_density(text)
        if density < self.min_information_density:
            return False, f"low_information_density ({density:.2f} bits/word)"

        if self.detect_excessive_whitespace(text):
            return False, "layout_artifact"

        return True, "passed"


class TFIDFScorer:
    """
    TF-IDF importance scorer for text chunks.

    Computes normalised TF-IDF scores to rank chunk relevance within a
    document corpus. Stopwords are excluded from both TF and DF calculations.

    Reference: Salton, G. & Buckley, C. (1988). "Term-weighting approaches
    in automatic text retrieval." Information Processing & Management,
    24(5), 513–523.
    """

    def __init__(self, stopwords: frozenset = None):
        self.stopwords = stopwords if stopwords is not None else ALL_STOPWORDS
        self.document_frequency: Dict[str, int] = {}
        self.total_chunks: int = 0
        self.chunk_term_frequencies: List[Counter] = []

    def reset(self) -> None:
        """Reset scorer state before analyzing a new corpus."""
        self.document_frequency = {}
        self.total_chunks = 0
        self.chunk_term_frequencies = []

    def _tokenize_and_filter(self, text: str) -> List[str]:
        """Tokenize and remove stopwords and single/two-character tokens."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in self.stopwords and len(w) > 2]

    def analyze_corpus(self, chunks: List[str]) -> None:
        """Build per-chunk TF tables and corpus-level DF table."""
        self.reset()
        self.total_chunks = len(chunks)

        for chunk in chunks:
            words = self._tokenize_and_filter(chunk)
            term_freq = Counter(words)
            self.chunk_term_frequencies.append(term_freq)
            for term in set(words):
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

    def calculate_chunk_importance(self, chunk_index: int) -> float:
        """
        Return mean TF-IDF score for all content terms in the chunk.

        Returns 0.0 if the corpus has not been analyzed or the chunk is empty.
        """
        if self.total_chunks == 0 or chunk_index >= len(self.chunk_term_frequencies):
            return 0.0

        term_freq = self.chunk_term_frequencies[chunk_index]
        if not term_freq:
            return 0.0

        tfidf_score = 0.0
        for term, tf in term_freq.items():
            df = self.document_frequency.get(term, 1)
            idf = math.log(self.total_chunks / df) if df > 0 else 0.0
            tfidf_score += tf * idf

        total_terms = sum(term_freq.values())
        return tfidf_score / total_terms if total_terms > 0 else 0.0


class SemanticChunker:
    """
    Semantic chunking orchestrator for structured documents (Thesis Section 2.3).

    Combines SemanticBoundaryDetector, HeaderExtractor, AutomaticQualityFilter,
    and TFIDFScorer. Falls back to RecursiveCharacterTextSplitter only on
    ValueError, RuntimeError, or AttributeError; other exceptions propagate.

    Parameter → settings.yaml mapping:
      max_chunk_size       → chunking.chunk_size (1024)
      min_chunk_size       → chunking.semantic.min_chunk_size (200)
      overlap              → chunking.chunk_overlap (128)
      word_boundary_factor → ingestion.word_boundary_factor (0.8)
    """

    def __init__(
        self,
        max_chunk_size: int = 1024,            # settings.yaml: chunking.chunk_size
        min_chunk_size: int = 200,             # settings.yaml: chunking.semantic.min_chunk_size
        overlap: int = 128,                    # settings.yaml: chunking.chunk_overlap
        word_boundary_factor: float = 0.8,     # settings.yaml: ingestion.word_boundary_factor
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

        self.header_extractor = HeaderExtractor()
        self.boundary_detector = SemanticBoundaryDetector(
            min_boundary_distance=min_chunk_size,
            word_boundary_factor=word_boundary_factor,
        )
        self.quality_filter = AutomaticQualityFilter(
            min_length=min_chunk_size,
            min_words=15,                      # settings.yaml: ingestion.quality_filter.min_words
            min_lexical_diversity=0.3,         # settings.yaml: ingestion.min_lexical_diversity
            min_information_density=2.0,       # settings.yaml: ingestion.min_information_density
        )
        self.tfidf_scorer = TFIDFScorer()

        # Ordered separator priority: paragraph > line > sentence > word > char
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"SemanticChunker initialized: "
            f"max_size={max_chunk_size}, min_size={min_chunk_size}, overlap={overlap}"
        )

    def _find_overlap_start(self, text: str, boundary: int, target_overlap: int) -> int:
        """
        Find overlap start that respects word boundaries.

        Walks backward from (boundary - target_overlap) to the nearest
        whitespace, then forward past the whitespace to find the word start.
        Prevents mid-word splits at the beginning of overlapping chunks.
        """
        if boundary < target_overlap:
            return 0

        pos = boundary - target_overlap
        while pos > 0 and not text[pos].isspace():
            pos -= 1
        while pos < boundary and text[pos].isspace():
            pos += 1
        return pos

    def _extract_raw_chunks(self, text: str) -> List[str]:
        """Extract raw text chunks using detected semantic boundaries."""
        boundaries = self.boundary_detector.find_semantic_boundaries(
            text, self.max_chunk_size
        )

        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            # Apply word-boundary-aware overlap for all chunks except the first
            if i > 0 and start >= self.overlap:
                start = self._find_overlap_start(text, boundaries[i], self.overlap)

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Segment a Document into semantically-coherent chunks with metadata.

        HeaderExtractor state is reset at the start of each call to prevent
        chapter/section context from leaking across document boundaries when
        this chunker instance is reused.
        """
        text = document.page_content
        base_metadata = document.metadata.copy()

        try:
            raw_chunks = self._extract_raw_chunks(text)
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning(
                f"Semantic chunking failed ({type(e).__name__}), using fallback: {e}"
            )
            return self.fallback_splitter.split_documents([document])

        if not raw_chunks:
            # Short or structure-free document: delegate to character-level splitter
            logger.debug("No semantic boundaries found; delegating to fallback splitter")
            return self.fallback_splitter.split_documents([document])

        self.tfidf_scorer.analyze_corpus(raw_chunks)

        processed_chunks: List[Document] = []
        filter_stats: Dict[str, Any] = {"kept": 0, "filtered": 0, "reasons": {}}

        self.header_extractor.reset()

        for i, chunk_text in enumerate(raw_chunks):
            metadata, cleaned_text = self.header_extractor.extract_headers(chunk_text)

            keep, reason = self.quality_filter.should_keep_chunk(cleaned_text)
            if not keep:
                filter_stats["filtered"] += 1
                filter_stats["reasons"][reason] = filter_stats["reasons"].get(reason, 0) + 1
                logger.debug(f"Filtered chunk {i}: {reason}")
                continue

            filter_stats["kept"] += 1

            importance_score = self.tfidf_scorer.calculate_chunk_importance(i)
            lexical_diversity = self.quality_filter.calculate_lexical_diversity(cleaned_text)

            enriched_metadata = base_metadata.copy()
            enriched_metadata.update({
                "chunk_id": len(processed_chunks),
                "chunk_size": len(cleaned_text),
                "chapter": metadata.chapter,
                "section": metadata.section,
                "subsection": metadata.subsection,
                "heading_level": metadata.heading_level,
                "is_header": metadata.is_header,
                "chunking_method": "semantic_automatic",
                "importance_score": round(importance_score, 4),
                "lexical_diversity": round(lexical_diversity, 4),
            })

            processed_chunks.append(
                Document(page_content=cleaned_text, metadata=enriched_metadata)
            )

        if filter_stats["filtered"] > 0 or len(raw_chunks) > 10:
            logger.info(
                f"Semantic chunking: {len(raw_chunks)} raw -> "
                f"{filter_stats['kept']} kept "
                f"(filtered {filter_stats['filtered']}: {filter_stats['reasons']})"
            )

        return processed_chunks


def create_semantic_chunker(
    chunk_size: int = 1024,            # settings.yaml: chunking.chunk_size
    chunk_overlap: int = 128,           # settings.yaml: chunking.chunk_overlap
    min_chunk_size: int = 200,          # settings.yaml: chunking.semantic.min_chunk_size
    word_boundary_factor: float = 0.8,  # settings.yaml: ingestion.word_boundary_factor
) -> SemanticChunker:
    """
    Factory for SemanticChunker. All parameters map to settings.yaml entries.
    Defaults are emergency fallbacks; production code should pass values from
    config/settings.yaml.
    """
    return SemanticChunker(
        max_chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        overlap=chunk_overlap,
        word_boundary_factor=word_boundary_factor,
    )


# ============================================================================
# PART 2: SENTENCE-BASED CHUNKING (Primary Strategy)
# ============================================================================

@dataclass
class SentenceChunkingConfig:
    """
    Configuration for the 3-sentence sliding-window chunker.

    Default values match the settings.yaml entries listed below. All values
    are validated in __post_init__ to surface misconfiguration at instantiation
    time rather than during processing.

      sentences_per_chunk → settings.yaml: ingestion.sentences_per_chunk (3)
      sentence_overlap    → settings.yaml: ingestion.sentence_overlap (1)
      min_chunk_chars     → settings.yaml: ingestion.min_chunk_size (50)
      max_chunk_chars     → settings.yaml: ingestion.max_chunk_chars (2000)
      spacy_model         → settings.yaml: ingestion.spacy_model ("en_core_web_sm")
      entity_aware        → settings.yaml: ingestion.entity_aware_chunking (false)
    """

    sentences_per_chunk: int = 3
    sentence_overlap: int = 1
    min_chunk_chars: int = 50
    max_chunk_chars: int = 2000
    spacy_model: str = "en_core_web_sm"
    # NER and parser are disabled for sentence segmentation to minimise latency.
    # Named entity recognition is handled separately by GLiNER (entity_extraction.py).
    disable_components: List[str] = field(default_factory=lambda: ["ner", "parser"])
    entity_aware: bool = False
    include_sentence_offsets: bool = True

    def __post_init__(self) -> None:
        if self.sentences_per_chunk < 1:
            raise ValueError(
                f"sentences_per_chunk must be >= 1, got {self.sentences_per_chunk}"
            )
        if self.sentence_overlap < 0:
            raise ValueError(
                f"sentence_overlap must be >= 0, got {self.sentence_overlap}"
            )
        if self.sentence_overlap >= self.sentences_per_chunk:
            raise ValueError(
                f"sentence_overlap ({self.sentence_overlap}) must be < "
                f"sentences_per_chunk ({self.sentences_per_chunk})"
            )


@dataclass
class SentenceInfo:
    """Metadata for a single sentence within a chunk."""

    text: str
    start_char: int
    end_char: int
    index: int


@dataclass
class SentenceChunk:
    """
    A chunk comprising multiple consecutive sentences.

    chunk_id is a deterministic SHA-256-based identifier derived from
    source_doc, position, and the first 50 characters of the chunk text.
    This ensures that re-ingestion produces identical KuzuDB node IDs,
    preserving graph integrity across ingestion runs.
    """

    chunk_id: str
    text: str
    sentences: List[SentenceInfo]
    position: int
    source_doc: str
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @property
    def char_length(self) -> int:
        return len(self.text)

    @property
    def sentence_indices(self) -> List[int]:
        return [s.index for s in self.sentences]

    def to_langchain_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "position": self.position,
                "source_doc": self.source_doc,
                "source_file": self.source_doc,
                "sentence_count": self.sentence_count,
                "char_start": self.char_start,
                "char_end": self.char_end,
                "sentence_indices": self.sentence_indices,
                "chunk_method": "sentence_spacy_3_window",
                **self.metadata,
            }
        )


class SpacySentenceSegmenter:
    """
    Sentence boundary detection backed by SpaCy.

    Loads the model from SpacyModelCache to avoid repeated disk reads.
    Automatically falls back to a regex-based segmenter when SpaCy is
    unavailable or the requested model is not installed.
    """

    # Minimum character length for a token span to be treated as a sentence.
    # Shorter spans are typically fragment artifacts from the sentencizer.
    MIN_SENTENCE_CHARS: int = 5

    def __init__(self, config: SentenceChunkingConfig):
        self.config = config
        self.nlp: Optional["Language"] = None
        self.using_spacy: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Load SpaCy model via SpacyModelCache (no-op if already cached)."""
        if not SPACY_AVAILABLE:
            logger.warning("SpaCy not available; using regex sentence segmentation fallback")
            return

        self.nlp = SpacyModelCache.get_model(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )
        if self.nlp is not None:
            self.using_spacy = True

    def segment(self, text: str) -> List[SentenceInfo]:
        """Segment text into SentenceInfo objects; returns [] for None or empty input."""
        if not text:
            return []
        if self.using_spacy and self.nlp is not None:
            return self._spacy_segment(text)
        return self._regex_segment(text)

    def _spacy_segment(self, text: str) -> List[SentenceInfo]:
        doc = self.nlp(text)
        sentences = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if len(sent_text) < self.MIN_SENTENCE_CHARS:
                continue
            sentences.append(SentenceInfo(
                text=sent_text,
                start_char=sent.start_char,
                end_char=sent.end_char,
                index=idx,
            ))
        return sentences

    def _regex_segment(self, text: str) -> List[SentenceInfo]:
        """
        Regex-based fallback segmenter.

        Splits on sentence-final punctuation followed by whitespace and an
        uppercase character. Accuracy degrades for text with dense abbreviation
        usage; SpaCy is strongly preferred when available.
        """
        if not text.strip():
            return []

        sentences = []
        current_pos = 0
        for sent_idx, part in enumerate(re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)):
            part_stripped = part.strip()
            if len(part_stripped) < self.MIN_SENTENCE_CHARS:
                continue

            start = text.find(part_stripped, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(part_stripped)
            current_pos = end

            sentences.append(SentenceInfo(
                text=part_stripped,
                start_char=start,
                end_char=end,
                index=sent_idx,
            ))

        return sentences


class SpacySentenceChunker:
    """
    3-sentence sliding-window chunker — primary ingestion strategy (Thesis §2.2).

    Each chunk spans sentences_per_chunk consecutive sentences with
    sentence_overlap sentences shared between adjacent windows. The overlap
    prevents entity bridges from being severed at chunk boundaries, which is
    critical for multi-hop retrieval in HotpotQA-style bridge queries.

    Reference: Lewis, P. et al. (2020). "Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks." NeurIPS 2020.
    (3-sentence window size validated by ablation study, Thesis §4.2.)

    Chunk identifiers are deterministic SHA-256 hashes over (source_doc,
    position, text_prefix), ensuring that KuzuDB graph node IDs remain stable
    across re-ingestion runs.
    """

    def __init__(
        self,
        sentences_per_chunk: int = 3,       # settings.yaml: ingestion.sentences_per_chunk
        sentence_overlap: int = 1,           # settings.yaml: ingestion.sentence_overlap
        min_chunk_chars: int = 50,           # settings.yaml: ingestion.min_chunk_size
        max_chunk_chars: int = 2000,         # settings.yaml: ingestion.max_chunk_chars
        spacy_model: str = "en_core_web_sm", # settings.yaml: ingestion.spacy_model
        entity_aware: bool = False,          # settings.yaml: ingestion.entity_aware_chunking
    ):
        self.config = SentenceChunkingConfig(
            sentences_per_chunk=sentences_per_chunk,
            sentence_overlap=sentence_overlap,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            spacy_model=spacy_model,
            entity_aware=entity_aware,
        )

        self.segmenter = SpacySentenceSegmenter(self.config)
        self._ner_nlp = None
        if entity_aware:
            self._ner_nlp = SpacyModelCache.get_model(spacy_model, disable=["parser"])

        logger.info(
            f"SpacySentenceChunker initialized: "
            f"{self.config.sentences_per_chunk}-sentence windows, "
            f"overlap={self.config.sentence_overlap}"
        )

    @staticmethod
    def _generate_chunk_id(source_doc: str, position: int, text: str) -> str:
        """
        Generate a deterministic, content-addressed chunk identifier.

        SHA-256 over (source_doc + position + text_prefix) ensures identical
        IDs on re-ingestion, preserving KuzuDB foreign key references.
        """
        content = f"{source_doc}:{position}:{text[:50]}"
        return hashlib.sha256(content.encode()).hexdigest()[:20]

    def chunk_text(
        self,
        text: str,
        source_doc: str = "unknown",
        base_metadata: Dict[str, Any] = None,
    ) -> List[SentenceChunk]:
        """Segment text into overlapping SentenceChunk objects."""
        if text is None:
            logger.warning(f"chunk_text received None for source_doc={source_doc!r}")
            return []

        base_metadata = base_metadata or {}
        sentences = self.segmenter.segment(text)

        if not sentences:
            logger.warning(f"No sentences found in document: {source_doc!r}")
            return []

        return self._sliding_window_chunk(sentences, source_doc, base_metadata)

    def _sliding_window_chunk(
        self,
        sentences: List[SentenceInfo],
        source_doc: str,
        base_metadata: Dict[str, Any],
    ) -> List[SentenceChunk]:
        """
        Build overlapping sentence windows.

        Window size = sentences_per_chunk; step = window_size - overlap.
        A trailing-sentences handler ensures no sentences are dropped when
        len(sentences) is not evenly divisible by step_size.
        """
        window_size = self.config.sentences_per_chunk
        step_size = max(1, window_size - self.config.sentence_overlap)

        chunks: List[SentenceChunk] = []
        position = 0
        i = 0

        while i < len(sentences):
            window_end = min(i + window_size, len(sentences))
            window_sentences = sentences[i:window_end]
            chunk_text = " ".join(s.text for s in window_sentences)

            # Extend window if chunk falls below minimum size
            if len(chunk_text) < self.config.min_chunk_chars and window_end < len(sentences):
                window_end = min(window_end + 1, len(sentences))
                window_sentences = sentences[i:window_end]
                chunk_text = " ".join(s.text for s in window_sentences)

            # Truncate window if chunk exceeds maximum size
            while len(chunk_text) > self.config.max_chunk_chars and len(window_sentences) > 1:
                window_sentences = window_sentences[:-1]
                chunk_text = " ".join(s.text for s in window_sentences)

            if len(chunk_text) < self.config.min_chunk_chars:
                i += step_size
                continue

            chunks.append(SentenceChunk(
                chunk_id=self._generate_chunk_id(source_doc, position, chunk_text),
                text=chunk_text,
                sentences=window_sentences,
                position=position,
                source_doc=source_doc,
                char_start=window_sentences[0].start_char,
                char_end=window_sentences[-1].end_char,
                metadata={
                    **base_metadata,
                    "chunk_method": "sentence_spacy_3_window",
                    "sentences_per_chunk": self.config.sentences_per_chunk,
                    "sentence_overlap": self.config.sentence_overlap,
                },
            ))

            position += 1
            i += step_size

            if i >= len(sentences) - 1 and window_end >= len(sentences):
                break

        # Trailing sentences: emit a final chunk for sentences not covered by the
        # main loop (occurs when len(sentences) % step_size != 0).
        if sentences and chunks:
            last_covered_idx = chunks[-1].sentences[-1].index
            remaining = [s for s in sentences if s.index > last_covered_idx]
            if remaining:
                chunk_text = " ".join(s.text for s in remaining)
                if len(chunk_text) >= self.config.min_chunk_chars:
                    chunks.append(SentenceChunk(
                        chunk_id=self._generate_chunk_id(source_doc, position, chunk_text),
                        text=chunk_text,
                        sentences=remaining,
                        position=position,
                        source_doc=source_doc,
                        char_start=remaining[0].start_char,
                        char_end=remaining[-1].end_char,
                        metadata={
                            **base_metadata,
                            "chunk_method": "sentence_spacy_3_window",
                            "is_final_chunk": True,
                        },
                    ))

        return chunks

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict]:
        """
        Primary interface consumed by ingestion.py (DocumentIngestionPipeline).

        Returns a list of dicts with keys 'text' and 'metadata', compatible
        with the ingestion pipeline's internal format.
        """
        metadata = metadata or {}
        source_doc = metadata.get("source_file", metadata.get("source", "unknown"))
        chunks = self.chunk_text(text, source_doc=source_doc, base_metadata=metadata)

        return [
            {
                "text": chunk.text,
                "metadata": {
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                    "position": chunk.position,
                    "sentence_start": chunk.sentences[0].index if chunk.sentences else 0,
                    "sentence_end": chunk.sentences[-1].index + 1 if chunk.sentences else 0,
                    "sentence_count": chunk.sentence_count,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            }
            for chunk in chunks
        ]

    def chunk_to_documents(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """Convert chunked text directly to LangChain Document objects."""
        metadata = metadata or {}
        source_doc = metadata.get("source_file", metadata.get("source", "unknown"))
        chunks = self.chunk_text(text, source_doc=source_doc, base_metadata=metadata)
        return [chunk.to_langchain_document() for chunk in chunks]

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk a list of LangChain Documents into sentence-window Documents."""
        all_chunks: List[Document] = []
        for doc in documents:
            source = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            chunks = self.chunk_text(
                doc.page_content, source_doc=source, base_metadata=doc.metadata
            )
            all_chunks.extend(chunk.to_langchain_document() for chunk in chunks)
        return all_chunks


def create_sentence_chunker(
    sentences_per_chunk: int = 3,       # settings.yaml: ingestion.sentences_per_chunk
    sentence_overlap: int = 1,           # settings.yaml: ingestion.sentence_overlap
    spacy_model: str = "en_core_web_sm", # settings.yaml: ingestion.spacy_model
    entity_aware: bool = False,          # settings.yaml: ingestion.entity_aware_chunking
    **kwargs,
) -> SpacySentenceChunker:
    """
    Factory for SpacySentenceChunker. All parameters map to settings.yaml entries.
    Defaults are emergency fallbacks; production code should pass values from
    config/settings.yaml.
    """
    return SpacySentenceChunker(
        sentences_per_chunk=sentences_per_chunk,
        sentence_overlap=sentence_overlap,
        spacy_model=spacy_model,
        entity_aware=entity_aware,
        **kwargs,
    )
