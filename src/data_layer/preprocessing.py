"""
Content Filtering and Preprocessing for RAG Quality

Version: 2.1.0
Author: Edge-RAG Research Project
Last Modified: 2026-01-13

===============================================================================
OVERVIEW
===============================================================================

This module provides content filtering functions to improve RAG retrieval
quality by removing low-value text chunks before embedding and indexing.

The core insight is that not all text in a document contributes equally
to answering user queries. Filtering out non-semantic content:
- Reduces index size (storage savings)
- Improves retrieval precision (less noise)
- Speeds up embedding generation (fewer texts to process)

===============================================================================
SCIENTIFIC FOUNDATION
===============================================================================

INFORMATION DENSITY IN DOCUMENTS:

Academic documents contain several types of content with varying
information density for retrieval purposes:

High Information Density (KEEP):
    - Abstract and introduction
    - Methodology descriptions
    - Results and findings
    - Discussion and conclusions
    - Definitions and explanations

Low Information Density (FILTER):
    - Bibliography/References
    - Table of contents
    - Headers and footers
    - Page numbers
    - Acknowledgments
    - Legal disclaimers

FILTERING RATIONALE:

1. Bibliography Sections:
   - Contain citations, not explanations
   - High density of proper nouns and dates
   - Pattern: "Author, A. (2020). Title..."
   - Impact: Confuses semantic matching

2. URL-Heavy Content:
   - URLs are not semantically meaningful
   - Often appear in references or footnotes
   - Pattern: Multiple http:// or https://

3. Short Fragments:
   - Likely headers, captions, or artifacts
   - Insufficient context for embedding
   - Pattern: < 100 characters

4. Repetitive Structures:
   - Tables of contents
   - List of figures/tables
   - Pattern: "Chapter X ... page Y"

QUALITY METRICS:

The filtering decisions can be evaluated using:
- Precision: Relevant chunks / Retrieved chunks
- Recall: Retrieved relevant / Total relevant
- F1 Score: Harmonic mean of precision and recall

Filtering improves precision at potential cost to recall.
The threshold parameters control this tradeoff.

===============================================================================
IMPLEMENTATION NOTES
===============================================================================

LANGUAGE CONSIDERATIONS:

The current implementation includes patterns for:
- German: "Literaturverzeichnis", "Quellenverzeichnis"
- English: "References", "Bibliography"

For other languages, extend the keyword lists accordingly.

THRESHOLD TUNING:

The filtering thresholds should be tuned based on:
- Document type (academic, technical, general)
- Language distribution
- Desired precision/recall tradeoff

INTEGRATION:

This module integrates with the ingestion pipeline:

    from src.preprocessing import should_skip_chunk, ContentFilter
    
    chunks = text_splitter.split_documents(documents)
    filtered_chunks = [c for c in chunks if not should_skip_chunk(c.page_content)]

===============================================================================
MODULE STRUCTURE
===============================================================================

Functions:
    should_skip_chunk()     - Simple boolean filter (legacy API)
    
Classes:
    ContentFilter           - Configurable filter with statistics
    FilterConfig            - Configuration dataclass
    FilterStatistics        - Statistics tracking

Constants:
    BIBLIOGRAPHY_KEYWORDS   - Keywords indicating reference sections
    CITATION_PATTERN        - Regex for citation detection
    URL_PATTERN             - Regex for URL detection
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND PATTERNS
# ============================================================================

# Keywords indicating bibliography/reference sections
# Covers German and English academic documents
BIBLIOGRAPHY_KEYWORDS = frozenset([
    # German
    "literaturverzeichnis",
    "quellenverzeichnis",
    "literatur",
    "quellen",
    "bibliographie",
    "referenzen",
    # English
    "references",
    "bibliography",
    "works cited",
    "citations",
    "literature cited",
    # Common headers
    "letzter zugriff",      # "last accessed" in German
    "accessed on",
    "retrieved from",
])

# Keywords indicating table of contents
TOC_KEYWORDS = frozenset([
    "inhaltsverzeichnis",   # German
    "table of contents",
    "contents",
    "gliederung",           # German: outline
])

# Keywords indicating acknowledgments (often low information density)
ACKNOWLEDGMENT_KEYWORDS = frozenset([
    "danksagung",           # German
    "acknowledgments",
    "acknowledgements",
])

# Regex pattern for academic citations
# Matches: "Author, A. (2020)" or "Author et al. (2019)"
CITATION_PATTERN = re.compile(
    r'\b[A-Z][a-z]+,?\s+[A-Z][a-z]*\.?\s*'  # Author name
    r'(?:et\s+al\.?)?\s*'                     # Optional "et al."
    r'\(\d{4}\)',                              # Year in parentheses
    re.UNICODE
)

# Regex pattern for URLs
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)

# Regex pattern for DOIs
DOI_PATTERN = re.compile(
    r'\b(?:doi[:\s]*)?10\.\d{4,}/[^\s]+',
    re.IGNORECASE
)

# Regex pattern for page references in TOC
# Matches: "Chapter 1 ... 15" or "1.2 Section Name.....23"
TOC_LINE_PATTERN = re.compile(
    r'^[\d.]+\s+.{5,50}\.{2,}\s*\d+$',
    re.MULTILINE
)


# ============================================================================
# FILTER CONFIGURATION
# ============================================================================

class FilterReason(Enum):
    """Enumeration of reasons for filtering a chunk."""
    PASSED = "passed"
    TOO_SHORT = "too_short"
    TOO_MANY_URLS = "too_many_urls"
    TOO_MANY_CITATIONS = "too_many_citations"
    BIBLIOGRAPHY_SECTION = "bibliography_section"
    TABLE_OF_CONTENTS = "table_of_contents"
    LOW_WORD_COUNT = "low_word_count"
    HIGH_NUMERIC_RATIO = "high_numeric_ratio"


@dataclass
class FilterConfig:
    """
    Configuration for content filtering.
    
    These thresholds control the precision/recall tradeoff.
    Higher thresholds = more aggressive filtering = higher precision, lower recall.
    
    Attributes:
        min_length: Minimum character count for a chunk
        min_word_count: Minimum word count for a chunk
        max_url_count: Maximum URLs before filtering
        max_citation_count: Maximum citations before filtering
        max_numeric_ratio: Maximum ratio of numeric characters
        check_bibliography: Enable bibliography keyword detection
        check_toc: Enable table of contents detection
    """
    min_length: int = 100
    min_word_count: int = 15
    max_url_count: int = 2
    max_citation_count: int = 3
    max_numeric_ratio: float = 0.3
    check_bibliography: bool = True
    check_toc: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "min_length": self.min_length,
            "min_word_count": self.min_word_count,
            "max_url_count": self.max_url_count,
            "max_citation_count": self.max_citation_count,
            "max_numeric_ratio": self.max_numeric_ratio,
            "check_bibliography": self.check_bibliography,
            "check_toc": self.check_toc,
        }


@dataclass
class FilterStatistics:
    """
    Statistics for content filtering operations.
    
    Tracks how many chunks were filtered and why, useful for:
    - Tuning filter thresholds
    - Documenting preprocessing in thesis
    - Quality assurance
    
    Attributes:
        total_processed: Total chunks evaluated
        total_passed: Chunks that passed filtering
        total_filtered: Chunks that were filtered out
        reasons: Count of chunks filtered by each reason
    """
    total_processed: int = 0
    total_passed: int = 0
    total_filtered: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    
    def record(self, passed: bool, reason: FilterReason) -> None:
        """Record a filtering decision."""
        self.total_processed += 1
        if passed:
            self.total_passed += 1
        else:
            self.total_filtered += 1
            reason_key = reason.value
            self.reasons[reason_key] = self.reasons.get(reason_key, 0) + 1
    
    @property
    def pass_rate(self) -> float:
        """Percentage of chunks that passed filtering."""
        if self.total_processed == 0:
            return 0.0
        return (self.total_passed / self.total_processed) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_processed": self.total_processed,
            "total_passed": self.total_passed,
            "total_filtered": self.total_filtered,
            "pass_rate_percent": self.pass_rate,
            "filter_reasons": self.reasons.copy(),
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Filter Statistics:",
            f"  Processed: {self.total_processed}",
            f"  Passed:    {self.total_passed} ({self.pass_rate:.1f}%)",
            f"  Filtered:  {self.total_filtered}",
        ]
        if self.reasons:
            lines.append("  Reasons:")
            for reason, count in sorted(self.reasons.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")
        return "\n".join(lines)


# ============================================================================
# CONTENT FILTER CLASS
# ============================================================================

class ContentFilter:
    """
    Configurable content filter for RAG preprocessing.
    
    This class provides comprehensive content filtering with:
    - Configurable thresholds
    - Multiple filter criteria
    - Statistics tracking
    - Detailed reasoning for filtering decisions
    
    FILTER CRITERIA:
    
    1. Length Check:
       - Chunks shorter than min_length are filtered
       - Rationale: Short chunks lack context for meaningful embedding
    
    2. Word Count Check:
       - Chunks with fewer than min_word_count words are filtered
       - Rationale: Ensures minimum semantic content
    
    3. URL Density Check:
       - Chunks with more than max_url_count URLs are filtered
       - Rationale: URL-heavy content is typically references
    
    4. Citation Density Check:
       - Chunks with more than max_citation_count citations are filtered
       - Rationale: Citation-heavy content is bibliography
    
    5. Bibliography Keyword Check:
       - Chunks containing bibliography keywords are filtered
       - Rationale: Reference sections add noise
    
    6. Table of Contents Check:
       - Chunks matching TOC patterns are filtered
       - Rationale: TOC has no semantic content
    
    7. Numeric Ratio Check:
       - Chunks with high numeric character ratio are filtered
       - Rationale: Likely tables, page numbers, or data
    
    USAGE:
    
        filter = ContentFilter(FilterConfig(min_length=100))
        
        for chunk in chunks:
            keep, reason = filter.evaluate(chunk.page_content)
            if keep:
                filtered_chunks.append(chunk)
        
        print(filter.statistics.summary())
    
    Attributes:
        config: FilterConfig instance
        statistics: FilterStatistics instance
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize content filter.
        
        Args:
            config: Filter configuration (uses defaults if None)
        """
        self.config = config or FilterConfig()
        self.statistics = FilterStatistics()
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, text: str) -> Tuple[bool, FilterReason]:
        """
        Evaluate whether a text chunk should be kept or filtered.
        
        Applies all configured filter criteria in order of computational
        cost (cheapest first). Returns immediately when a filter triggers.
        
        Args:
            text: Text content to evaluate
            
        Returns:
            Tuple of (should_keep, reason)
            - should_keep: True if chunk passes all filters
            - reason: FilterReason explaining the decision
        """
        # Check 1: Minimum length (cheapest check)
        if len(text.strip()) < self.config.min_length:
            self.statistics.record(False, FilterReason.TOO_SHORT)
            return False, FilterReason.TOO_SHORT
        
        # Check 2: Minimum word count
        words = text.split()
        if len(words) < self.config.min_word_count:
            self.statistics.record(False, FilterReason.LOW_WORD_COUNT)
            return False, FilterReason.LOW_WORD_COUNT
        
        # Check 3: URL density
        url_count = len(URL_PATTERN.findall(text))
        if url_count > self.config.max_url_count:
            self.statistics.record(False, FilterReason.TOO_MANY_URLS)
            return False, FilterReason.TOO_MANY_URLS
        
        # Check 4: Citation density
        citation_count = len(CITATION_PATTERN.findall(text))
        if citation_count > self.config.max_citation_count:
            self.statistics.record(False, FilterReason.TOO_MANY_CITATIONS)
            return False, FilterReason.TOO_MANY_CITATIONS
        
        # Check 5: Bibliography keywords
        if self.config.check_bibliography:
            text_lower = text.lower()
            for keyword in BIBLIOGRAPHY_KEYWORDS:
                if keyword in text_lower:
                    self.statistics.record(False, FilterReason.BIBLIOGRAPHY_SECTION)
                    return False, FilterReason.BIBLIOGRAPHY_SECTION
        
        # Check 6: Table of contents patterns
        if self.config.check_toc:
            text_lower = text.lower()
            for keyword in TOC_KEYWORDS:
                if keyword in text_lower:
                    self.statistics.record(False, FilterReason.TABLE_OF_CONTENTS)
                    return False, FilterReason.TABLE_OF_CONTENTS
            
            # Check for TOC line patterns
            toc_matches = TOC_LINE_PATTERN.findall(text)
            if len(toc_matches) > 3:  # Multiple TOC-like lines
                self.statistics.record(False, FilterReason.TABLE_OF_CONTENTS)
                return False, FilterReason.TABLE_OF_CONTENTS
        
        # Check 7: Numeric ratio
        numeric_chars = sum(1 for c in text if c.isdigit())
        total_chars = len(text.replace(" ", ""))
        if total_chars > 0:
            numeric_ratio = numeric_chars / total_chars
            if numeric_ratio > self.config.max_numeric_ratio:
                self.statistics.record(False, FilterReason.HIGH_NUMERIC_RATIO)
                return False, FilterReason.HIGH_NUMERIC_RATIO
        
        # All checks passed
        self.statistics.record(True, FilterReason.PASSED)
        return True, FilterReason.PASSED
    
    def filter_chunks(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """
        Filter a list of text chunks.
        
        Args:
            texts: List of text chunks to filter
            
        Returns:
            Tuple of (filtered_texts, kept_indices)
            - filtered_texts: Texts that passed filtering
            - kept_indices: Original indices of kept texts
        """
        filtered = []
        indices = []
        
        for i, text in enumerate(texts):
            keep, _ = self.evaluate(text)
            if keep:
                filtered.append(text)
                indices.append(i)
        
        return filtered, indices
    
    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self.statistics = FilterStatistics()
    
    def get_statistics(self) -> FilterStatistics:
        """Get current statistics."""
        return self.statistics


# ============================================================================
# LEGACY API (for backwards compatibility)
# ============================================================================

def should_skip_chunk(text: str) -> bool:
    """
    Determine if a text chunk should be skipped (filtered out).
    
    LEGACY API: This function provides a simple boolean interface
    for backwards compatibility. For new code, use ContentFilter class.
    
    FILTER CRITERIA:
    
    1. URL Density:
       Skip if text contains more than 2 URLs.
       Rationale: High URL density indicates reference/link sections.
    
    2. Citation Density:
       Skip if text contains more than 3 academic citations.
       Rationale: Citation-heavy text is likely bibliography.
       Pattern: "Author, First (Year)"
    
    3. Length:
       Skip if text is shorter than 100 characters.
       Rationale: Short fragments lack semantic context.
    
    4. Bibliography Keywords:
       Skip if text contains bibliography section indicators.
       Supports German and English keywords.
    
    Args:
        text: Text content to evaluate
        
    Returns:
        True if chunk should be skipped (filtered out),
        False if chunk should be kept
        
    Example:
        chunks = text_splitter.split_documents(docs)
        filtered = [c for c in chunks if not should_skip_chunk(c.page_content)]
    """
    # Check 1: URL density
    url_count = text.count("http://") + text.count("https://")
    if url_count > 2:
        return True
    
    # Check 2: Citation density (academic citation pattern)
    citation_matches = CITATION_PATTERN.findall(text)
    if len(citation_matches) > 3:
        return True
    
    # Check 3: Minimum length
    if len(text.strip()) < 100:
        return True
    
    # Check 4: Bibliography keywords
    text_lower = text.lower()
    for keyword in BIBLIOGRAPHY_KEYWORDS:
        if keyword in text_lower:
            return True
    
    return False


def filter_chunks_simple(texts: List[str]) -> List[str]:
    """
    Filter a list of text chunks using simple criteria.
    
    LEGACY API: For new code, use ContentFilter class.
    
    Args:
        texts: List of text chunks
        
    Returns:
        List of chunks that passed filtering
    """
    return [t for t in texts if not should_skip_chunk(t)]


# ============================================================================
# ADVANCED FILTERING UTILITIES
# ============================================================================

def compute_information_density(text: str) -> float:
    """
    Compute information density score for text.
    
    Information density is estimated using lexical diversity:
    the ratio of unique words to total words.
    
    High diversity = more information
    Low diversity = repetitive content
    
    FORMULA:
        density = len(unique_words) / len(total_words)
    
    INTERPRETATION:
        0.0 - 0.3: Low density (repetitive, boilerplate)
        0.3 - 0.6: Medium density (typical prose)
        0.6 - 1.0: High density (varied vocabulary)
    
    Args:
        text: Input text
        
    Returns:
        Information density score in [0, 1]
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    
    unique_words = set(words)
    return len(unique_words) / len(words)


def compute_academic_score(text: str) -> float:
    """
    Compute academic content score for text.
    
    Higher scores indicate more academic/technical content.
    Useful for prioritizing chunks in retrieval.
    
    SCORING FACTORS:
        + Long sentences (complex ideas)
        + Formal vocabulary
        + Structured paragraphs
        - High URL density
        - High citation density (bibliography)
        - Short fragments
    
    Args:
        text: Input text
        
    Returns:
        Academic score in [0, 1]
    """
    score = 0.5  # Start neutral
    
    # Length factor
    if len(text) > 500:
        score += 0.1
    elif len(text) < 100:
        score -= 0.2
    
    # Sentence complexity (average words per sentence)
    sentences = re.split(r'[.!?]+', text)
    if sentences:
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_length > 15:
            score += 0.1
        elif avg_sentence_length < 5:
            score -= 0.1
    
    # URL penalty
    url_count = len(URL_PATTERN.findall(text))
    score -= url_count * 0.1
    
    # Citation penalty (too many = bibliography)
    citation_count = len(CITATION_PATTERN.findall(text))
    if citation_count > 3:
        score -= 0.2
    
    # Information density bonus
    density = compute_information_density(text)
    score += (density - 0.5) * 0.2
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def detect_language(text: str) -> str:
    """
    Simple language detection based on common words.
    
    This is a heuristic approach suitable for German/English detection.
    For production use, consider langdetect or fasttext.
    
    Args:
        text: Input text
        
    Returns:
        Language code: "de" for German, "en" for English, "unknown" otherwise
    """
    text_lower = text.lower()
    
    # German indicators
    german_words = ["und", "der", "die", "das", "ist", "ein", "eine", "mit", "auf"]
    german_count = sum(1 for w in german_words if f" {w} " in f" {text_lower} ")
    
    # English indicators
    english_words = ["the", "and", "is", "are", "this", "that", "with", "for", "on"]
    english_count = sum(1 for w in english_words if f" {w} " in f" {text_lower} ")
    
    if german_count > english_count and german_count >= 3:
        return "de"
    elif english_count > german_count and english_count >= 3:
        return "en"
    else:
        return "unknown"


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

def preprocess_documents(
    texts: List[str],
    config: Optional[FilterConfig] = None,
    return_statistics: bool = False
) -> Any:
    """
    Preprocess a batch of text chunks with filtering.
    
    Convenience function combining filtering with statistics.
    
    Args:
        texts: List of text chunks to process
        config: Filter configuration (uses defaults if None)
        return_statistics: If True, return statistics along with filtered texts
        
    Returns:
        If return_statistics is False: List of filtered texts
        If return_statistics is True: Tuple of (filtered_texts, statistics_dict)
    """
    content_filter = ContentFilter(config)
    filtered_texts, _ = content_filter.filter_chunks(texts)
    
    if return_statistics:
        return filtered_texts, content_filter.statistics.to_dict()
    else:
        return filtered_texts


def analyze_chunk_quality(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze quality metrics for a collection of text chunks.
    
    Useful for thesis documentation and quality assurance.
    
    Args:
        texts: List of text chunks
        
    Returns:
        Dictionary containing quality metrics:
        - count: Number of chunks
        - avg_length: Average character count
        - avg_words: Average word count
        - avg_density: Average information density
        - language_distribution: Count by detected language
        - quality_distribution: Count by quality tier
    """
    if not texts:
        return {"count": 0, "error": "No texts provided"}
    
    lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    densities = [compute_information_density(t) for t in texts]
    languages = [detect_language(t) for t in texts]
    scores = [compute_academic_score(t) for t in texts]
    
    # Language distribution
    lang_dist = {}
    for lang in languages:
        lang_dist[lang] = lang_dist.get(lang, 0) + 1
    
    # Quality tiers
    quality_dist = {"high": 0, "medium": 0, "low": 0}
    for score in scores:
        if score >= 0.6:
            quality_dist["high"] += 1
        elif score >= 0.4:
            quality_dist["medium"] += 1
        else:
            quality_dist["low"] += 1
    
    return {
        "count": len(texts),
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "word_stats": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": sum(word_counts) / len(word_counts),
        },
        "density_stats": {
            "min": min(densities),
            "max": max(densities),
            "mean": sum(densities) / len(densities),
        },
        "language_distribution": lang_dist,
        "quality_distribution": quality_dist,
    }