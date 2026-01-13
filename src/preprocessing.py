"""
Content Preprocessing and Quality Filtering for RAG Systems

Version: 2.1.0
Author: Edge-RAG Research Project
Last Modified: 2026-01-13

===============================================================================
OVERVIEW
===============================================================================

This module provides content filtering and quality assessment functions
for RAG (Retrieval-Augmented Generation) systems. Its primary purpose is
to identify and filter out low-quality text chunks that would degrade
retrieval performance.

MOTIVATION:

Not all text chunks are equally valuable for retrieval:

1. Bibliography Sections:
   - Lists of references with author names and years
   - High keyword density but low semantic content
   - Would match many queries incorrectly

2. Boilerplate Content:
   - Headers, footers, page numbers
   - Legal disclaimers, copyright notices
   - Table of contents entries

3. Low-Information Content:
   - Very short fragments
   - Mostly whitespace or punctuation
   - Repetitive text patterns

4. Transcript Artifacts:
   - Interview markers (I:, B1:, etc.)
   - Timestamps and speaker labels
   - Often low semantic density

Filtering these improves:
- Retrieval precision (fewer false positives)
- Index efficiency (smaller vector store)
- Response quality (better context for LLM)

===============================================================================
SCIENTIFIC FOUNDATION
===============================================================================

QUALITY METRICS:

This module implements several quality metrics based on information theory
and linguistic analysis:

1. Lexical Diversity (Type-Token Ratio):
   
   TTR = |unique_tokens| / |total_tokens|
   
   - High TTR (> 0.5): Diverse vocabulary, likely informative
   - Low TTR (< 0.3): Repetitive, possibly boilerplate
   
   Reference: Richards (1987). "Type/Token Ratios: What Do They Really Tell Us?"

2. Information Density (Shannon Entropy):
   
   H(X) = -sum(p(x) * log2(p(x))) for all unique tokens x
   
   - High entropy: Varied word distribution
   - Low entropy: Skewed distribution (few dominant words)
   
   Reference: Shannon (1948). "A Mathematical Theory of Communication"

3. Content Indicators:
   - Citation density (academic reference patterns)
   - URL density (web references)
   - Structural markers (section headers, lists)

FILTERING STRATEGY:

The filtering approach uses a multi-stage pipeline:

Stage 1: Length Filtering
    - Minimum character count
    - Minimum word count
    
Stage 2: Pattern-Based Filtering
    - Bibliography detection
    - Transcript detection
    - Boilerplate detection
    
Stage 3: Statistical Filtering
    - Lexical diversity threshold
    - Information density threshold

Each stage can be individually enabled/disabled for experimentation.

===============================================================================
USAGE
===============================================================================

Basic Usage:
    from preprocessing import should_skip_chunk, ChunkQualityAnalyzer
    
    # Simple filtering
    if should_skip_chunk(text):
        continue  # Skip this chunk
    
    # Detailed analysis
    analyzer = ChunkQualityAnalyzer()
    result = analyzer.analyze(text)
    
    if not result.should_keep:
        print(f"Filtered: {result.rejection_reason}")

Integration with Ingestion Pipeline:
    chunks = pipeline.process_documents()
    filtered_chunks = [c for c in chunks if not should_skip_chunk(c.page_content)]

===============================================================================
MODULE STRUCTURE
===============================================================================

Classes:
    ChunkQualityResult   - Result container for quality analysis
    ChunkQualityAnalyzer - Comprehensive quality analyzer
    
Functions:
    should_skip_chunk()           - Simple boolean filter
    calculate_lexical_diversity() - TTR computation
    calculate_entropy()           - Shannon entropy computation
    detect_bibliography()         - Bibliography section detection
    detect_transcript()           - Interview transcript detection
    clean_and_normalize()         - Text normalization
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND PATTERNS
# ============================================================================

class FilterReason(Enum):
    """
    Enumeration of reasons for chunk rejection.
    
    Used for logging and analysis of filtering decisions.
    """
    TOO_SHORT = "too_short"
    TOO_FEW_WORDS = "too_few_words"
    BIBLIOGRAPHY = "bibliography_section"
    TRANSCRIPT = "transcript_content"
    LOW_DIVERSITY = "low_lexical_diversity"
    LOW_ENTROPY = "low_information_density"
    HIGH_URL_DENSITY = "high_url_density"
    HIGH_CITATION_DENSITY = "high_citation_density"
    BOILERPLATE = "boilerplate_content"
    WHITESPACE_DOMINANT = "excessive_whitespace"


# Bibliography detection keywords (multilingual)
BIBLIOGRAPHY_KEYWORDS: Set[str] = {
    # German
    "literaturverzeichnis",
    "quellenverzeichnis", 
    "quellenangaben",
    "bibliographie",
    "literaturangaben",
    "letzter zugriff",
    "abgerufen am",
    # English
    "references",
    "bibliography",
    "works cited",
    "citations",
    "further reading",
    "accessed on",
    "retrieved from",
    # French
    "bibliographie",
    "references",
}

# Boilerplate detection keywords
BOILERPLATE_KEYWORDS: Set[str] = {
    # German
    "inhaltsverzeichnis",
    "abbildungsverzeichnis",
    "tabellenverzeichnis",
    "abkuerzungsverzeichnis",
    "anhang",
    "impressum",
    # English
    "table of contents",
    "list of figures",
    "list of tables",
    "appendix",
    "index",
    "glossary",
}

# Citation patterns (academic references)
# Pattern: "Author, A. (2023)" or "Author (2023)"
CITATION_PATTERN = re.compile(
    r'\b[A-Z][a-z]+(?:,\s*[A-Z]\.?\s*)?(?:\s*(?:&|and|und)\s*[A-Z][a-z]+(?:,\s*[A-Z]\.?\s*)?)?\s*\(\d{4}[a-z]?\)',
    re.UNICODE
)

# URL pattern
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)

# Transcript speaker pattern (e.g., "I:", "B1:", "Interviewer:")
TRANSCRIPT_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:[A-Z]{1,2}\d*|Interviewer|Interviewee|Speaker\s*\d*):\s*',
    re.IGNORECASE | re.MULTILINE
)

# Page number patterns
PAGE_NUMBER_PATTERN = re.compile(
    r'\b(?:page|seite|p\.?)\s*\d+\b|\b\d+\s*/\s*\d+\b',
    re.IGNORECASE
)


# ============================================================================
# RESULT CONTAINER
# ============================================================================

@dataclass
class ChunkQualityResult:
    """
    Container for chunk quality analysis results.
    
    This dataclass provides comprehensive information about a chunk's
    quality assessment, useful for debugging, logging, and thesis analysis.
    
    Attributes:
        should_keep: Whether the chunk passed all quality filters
        rejection_reason: Reason for rejection (if rejected)
        char_count: Number of characters in chunk
        word_count: Number of words in chunk
        lexical_diversity: Type-Token Ratio [0.0, 1.0]
        entropy: Shannon entropy in bits
        url_count: Number of URLs detected
        citation_count: Number of academic citations detected
        is_bibliography: Whether chunk appears to be from bibliography
        is_transcript: Whether chunk appears to be transcript content
        confidence: Confidence score for the decision [0.0, 1.0]
    
    Usage:
        result = analyzer.analyze(text)
        if result.should_keep:
            process_chunk(text)
        else:
            logger.info(f"Skipped: {result.rejection_reason}")
    """
    should_keep: bool
    rejection_reason: Optional[FilterReason] = None
    char_count: int = 0
    word_count: int = 0
    lexical_diversity: float = 0.0
    entropy: float = 0.0
    url_count: int = 0
    citation_count: int = 0
    is_bibliography: bool = False
    is_transcript: bool = False
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "should_keep": self.should_keep,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "lexical_diversity": round(self.lexical_diversity, 4),
            "entropy": round(self.entropy, 4),
            "url_count": self.url_count,
            "citation_count": self.citation_count,
            "is_bibliography": self.is_bibliography,
            "is_transcript": self.is_transcript,
            "confidence": round(self.confidence, 4),
        }


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def tokenize_simple(text: str) -> List[str]:
    """
    Simple whitespace-based tokenization with normalization.
    
    This tokenizer is designed for quality assessment, not for
    linguistic analysis. It prioritizes speed over accuracy.
    
    ALGORITHM:
    1. Convert to lowercase
    2. Split on whitespace
    3. Remove punctuation from token boundaries
    4. Filter empty tokens
    
    Args:
        text: Input text
        
    Returns:
        List of lowercase tokens
    """
    # Convert to lowercase and split
    tokens = text.lower().split()
    
    # Clean tokens: remove surrounding punctuation
    cleaned = []
    for token in tokens:
        # Strip common punctuation
        token = token.strip('.,;:!?()[]{}"\'-')
        if token:  # Skip empty tokens
            cleaned.append(token)
    
    return cleaned


def calculate_lexical_diversity(text: str) -> float:
    """
    Calculate lexical diversity using Type-Token Ratio (TTR).
    
    TYPE-TOKEN RATIO (TTR):
    
    TTR measures vocabulary richness by comparing unique words (types)
    to total words (tokens):
    
        TTR = |types| / |tokens|
    
    Properties:
    - Range: [0.0, 1.0]
    - TTR = 1.0: Every word is unique (maximum diversity)
    - TTR = 0.0: All words are the same (minimum diversity)
    
    INTERPRETATION FOR RAG:
    
    - TTR > 0.6: High diversity, likely informative content
    - TTR 0.4-0.6: Moderate diversity, typical prose
    - TTR 0.3-0.4: Low diversity, possibly repetitive
    - TTR < 0.3: Very low diversity, likely boilerplate/lists
    
    LIMITATIONS:
    
    TTR is sensitive to text length - longer texts tend to have
    lower TTR due to word repetition. For consistent comparison,
    use texts of similar length or apply correction factors.
    
    Args:
        text: Input text
        
    Returns:
        Type-Token Ratio in range [0.0, 1.0]
    """
    tokens = tokenize_simple(text)
    
    if not tokens:
        return 0.0
    
    types = set(tokens)
    ttr = len(types) / len(tokens)
    
    return ttr


def calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of word distribution.
    
    SHANNON ENTROPY:
    
    Entropy measures the average information content (in bits) per word:
    
        H(X) = -sum(p(x) * log2(p(x))) for all words x
    
    where p(x) is the probability (frequency) of word x.
    
    Properties:
    - H >= 0 (always non-negative)
    - H = 0 when all words are the same
    - H is maximized when words are uniformly distributed
    - Maximum H = log2(|vocabulary|)
    
    INTERPRETATION FOR RAG:
    
    - High entropy (> 5 bits): Varied vocabulary, informative
    - Medium entropy (3-5 bits): Typical prose
    - Low entropy (< 3 bits): Repetitive, possibly low quality
    
    EXAMPLE:
    
    Text: "the cat sat on the mat"
    Frequencies: {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
    Total: 6 words
    
    H = -(2/6 * log2(2/6) + 4 * 1/6 * log2(1/6))
    H = -(0.333 * -1.585 + 4 * 0.167 * -2.585)
    H = 0.528 + 1.723 = 2.251 bits
    
    Args:
        text: Input text
        
    Returns:
        Shannon entropy in bits (non-negative float)
    """
    tokens = tokenize_simple(text)
    
    if not tokens:
        return 0.0
    
    # Count word frequencies
    word_counts = Counter(tokens)
    total_words = len(tokens)
    
    # Calculate entropy
    entropy = 0.0
    for count in word_counts.values():
        probability = count / total_words
        if probability > 0:  # Avoid log(0)
            entropy -= probability * math.log2(probability)
    
    return entropy


def calculate_whitespace_ratio(text: str) -> float:
    """
    Calculate ratio of whitespace to total characters.
    
    High whitespace ratio often indicates:
    - Table/list formatting artifacts
    - Excessive line breaks
    - Poor PDF extraction
    
    Args:
        text: Input text
        
    Returns:
        Whitespace ratio in range [0.0, 1.0]
    """
    if not text:
        return 1.0
    
    whitespace_count = sum(1 for c in text if c.isspace())
    return whitespace_count / len(text)


# ============================================================================
# PATTERN DETECTION FUNCTIONS
# ============================================================================

def detect_bibliography(text: str) -> Tuple[bool, float]:
    """
    Detect if text chunk is from a bibliography section.
    
    DETECTION CRITERIA:
    
    1. Keyword presence: "References", "Bibliography", etc.
    2. Citation density: High frequency of author-year patterns
    3. URL density: Many web references
    
    Each criterion contributes to a confidence score.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (is_bibliography: bool, confidence: float)
    """
    text_lower = text.lower()
    confidence = 0.0
    
    # Check for bibliography keywords
    keyword_found = any(kw in text_lower for kw in BIBLIOGRAPHY_KEYWORDS)
    if keyword_found:
        confidence += 0.4
    
    # Count citations
    citations = CITATION_PATTERN.findall(text)
    citation_density = len(citations) / max(len(text.split()), 1)
    
    if len(citations) > 3:
        confidence += 0.3
    if citation_density > 0.1:  # More than 10% of words are citations
        confidence += 0.2
    
    # Count URLs
    urls = URL_PATTERN.findall(text)
    if len(urls) > 2:
        confidence += 0.1
    
    is_bibliography = confidence >= 0.5
    
    return is_bibliography, min(confidence, 1.0)


def detect_transcript(text: str) -> Tuple[bool, float]:
    """
    Detect if text chunk is from an interview transcript.
    
    DETECTION CRITERIA:
    
    1. Speaker markers: "I:", "B1:", "Interviewer:", etc.
    2. Pattern density: Frequency of speaker changes
    3. Dialogue structure: Short turns, question-answer patterns
    
    Transcripts often have low semantic density and should be
    filtered or processed differently.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (is_transcript: bool, confidence: float)
    """
    # Count speaker markers
    matches = TRANSCRIPT_PATTERN.findall(text)
    
    # Calculate line count for density
    lines = text.strip().split('\n')
    line_count = len([l for l in lines if l.strip()])
    
    if line_count == 0:
        return False, 0.0
    
    # Speaker marker density
    marker_density = len(matches) / line_count
    
    confidence = 0.0
    
    if len(matches) >= 2:
        confidence += 0.3
    if marker_density > 0.2:  # More than 20% of lines have markers
        confidence += 0.4
    if marker_density > 0.4:
        confidence += 0.3
    
    is_transcript = confidence >= 0.5
    
    return is_transcript, min(confidence, 1.0)


def detect_boilerplate(text: str) -> Tuple[bool, float]:
    """
    Detect if text chunk is boilerplate content.
    
    BOILERPLATE TYPES:
    
    1. Table of contents
    2. List of figures/tables
    3. Appendix headers
    4. Index entries
    5. Legal disclaimers
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (is_boilerplate: bool, confidence: float)
    """
    text_lower = text.lower()
    confidence = 0.0
    
    # Check for boilerplate keywords
    keyword_matches = sum(1 for kw in BOILERPLATE_KEYWORDS if kw in text_lower)
    
    if keyword_matches >= 1:
        confidence += 0.5
    if keyword_matches >= 2:
        confidence += 0.3
    
    # Check for page number density
    page_numbers = PAGE_NUMBER_PATTERN.findall(text)
    if len(page_numbers) > 3:
        confidence += 0.2
    
    is_boilerplate = confidence >= 0.5
    
    return is_boilerplate, min(confidence, 1.0)


# ============================================================================
# SIMPLE FILTERING FUNCTION
# ============================================================================

def should_skip_chunk(
    text: str,
    min_chars: int = 100,
    min_words: int = 15,
    max_url_count: int = 2,
    max_citation_count: int = 3,
    min_diversity: float = 0.25,
) -> bool:
    """
    Determine if a text chunk should be skipped during indexing.
    
    This is a simple, fast filter for basic quality control.
    For detailed analysis, use ChunkQualityAnalyzer.
    
    FILTERING CRITERIA (any triggers rejection):
    
    1. Too short (< min_chars characters)
    2. Too few words (< min_words)
    3. Too many URLs (> max_url_count)
    4. Too many citations (> max_citation_count)
    5. Bibliography section detected
    6. Low lexical diversity (< min_diversity)
    
    DEFAULT THRESHOLDS:
    
    The default values are calibrated for academic documents:
    - min_chars=100: Skip very short fragments
    - min_words=15: Ensure meaningful content
    - max_url_count=2: Filter URL-heavy sections
    - max_citation_count=3: Filter bibliography entries
    - min_diversity=0.25: Filter repetitive content
    
    Args:
        text: Text chunk to evaluate
        min_chars: Minimum character count
        min_words: Minimum word count
        max_url_count: Maximum allowed URLs
        max_citation_count: Maximum allowed citations
        min_diversity: Minimum lexical diversity (TTR)
        
    Returns:
        True if chunk should be skipped, False if it should be kept
        
    Example:
        chunks = load_chunks()
        filtered = [c for c in chunks if not should_skip_chunk(c.text)]
    """
    # Length check
    if len(text.strip()) < min_chars:
        return True
    
    # Word count check
    words = text.split()
    if len(words) < min_words:
        return True
    
    # URL density check
    urls = URL_PATTERN.findall(text)
    if len(urls) > max_url_count:
        return True
    
    # Citation density check
    citations = CITATION_PATTERN.findall(text)
    if len(citations) > max_citation_count:
        return True
    
    # Bibliography keyword check
    text_lower = text.lower()
    if any(kw in text_lower for kw in BIBLIOGRAPHY_KEYWORDS):
        return True
    
    # Lexical diversity check
    diversity = calculate_lexical_diversity(text)
    if diversity < min_diversity:
        return True
    
    return False


# ============================================================================
# COMPREHENSIVE QUALITY ANALYZER
# ============================================================================

class ChunkQualityAnalyzer:
    """
    Comprehensive chunk quality analyzer with configurable thresholds.
    
    This class provides detailed quality analysis for text chunks,
    suitable for:
    - Fine-grained filtering decisions
    - Quality metric collection for thesis
    - Debugging filtering behavior
    - Ablation studies on filter components
    
    CONFIGURATION:
    
    All thresholds can be adjusted via constructor parameters or
    the configure() method. This enables experimentation with
    different quality criteria.
    
    ANALYSIS PIPELINE:
    
    1. Basic statistics (length, word count)
    2. Pattern detection (bibliography, transcript, boilerplate)
    3. Statistical analysis (diversity, entropy)
    4. Decision based on configured thresholds
    
    USAGE:
    
        analyzer = ChunkQualityAnalyzer(
            min_chars=100,
            min_words=15,
            min_diversity=0.3,
        )
        
        for chunk in chunks:
            result = analyzer.analyze(chunk.page_content)
            
            if result.should_keep:
                process(chunk)
            else:
                log_rejection(chunk, result.rejection_reason)
    
    THESIS DOCUMENTATION:
    
    The analyze() method returns detailed metrics that can be
    aggregated for thesis documentation:
    
        results = [analyzer.analyze(c.text) for c in chunks]
        
        avg_diversity = mean(r.lexical_diversity for r in results)
        rejection_rate = sum(1 for r in results if not r.should_keep) / len(results)
    
    Attributes:
        min_chars: Minimum character count threshold
        min_words: Minimum word count threshold
        min_diversity: Minimum lexical diversity (TTR)
        min_entropy: Minimum Shannon entropy
        max_url_count: Maximum allowed URLs
        max_citation_count: Maximum allowed citations
        max_whitespace_ratio: Maximum whitespace ratio
        filter_bibliography: Enable bibliography filtering
        filter_transcript: Enable transcript filtering
        filter_boilerplate: Enable boilerplate filtering
    """
    
    def __init__(
        self,
        min_chars: int = 100,
        min_words: int = 15,
        min_diversity: float = 0.25,
        min_entropy: float = 2.0,
        max_url_count: int = 2,
        max_citation_count: int = 3,
        max_whitespace_ratio: float = 0.5,
        filter_bibliography: bool = True,
        filter_transcript: bool = True,
        filter_boilerplate: bool = True,
    ):
        """
        Initialize quality analyzer with thresholds.
        
        Args:
            min_chars: Minimum character count (default: 100)
            min_words: Minimum word count (default: 15)
            min_diversity: Minimum TTR (default: 0.25)
            min_entropy: Minimum entropy in bits (default: 2.0)
            max_url_count: Maximum URLs (default: 2)
            max_citation_count: Maximum citations (default: 3)
            max_whitespace_ratio: Maximum whitespace (default: 0.5)
            filter_bibliography: Enable bibliography filter (default: True)
            filter_transcript: Enable transcript filter (default: True)
            filter_boilerplate: Enable boilerplate filter (default: True)
        """
        self.min_chars = min_chars
        self.min_words = min_words
        self.min_diversity = min_diversity
        self.min_entropy = min_entropy
        self.max_url_count = max_url_count
        self.max_citation_count = max_citation_count
        self.max_whitespace_ratio = max_whitespace_ratio
        self.filter_bibliography = filter_bibliography
        self.filter_transcript = filter_transcript
        self.filter_boilerplate = filter_boilerplate
        
        # Statistics tracking
        self._analysis_count = 0
        self._rejection_counts: Dict[FilterReason, int] = {r: 0 for r in FilterReason}
    
    def configure(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameter names and values to update
            
        Example:
            analyzer.configure(min_chars=200, min_diversity=0.3)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def analyze(self, text: str) -> ChunkQualityResult:
        """
        Perform comprehensive quality analysis on text chunk.
        
        ANALYSIS STAGES:
        
        1. Compute basic statistics
        2. Check length thresholds
        3. Detect patterns (bibliography, transcript, boilerplate)
        4. Compute statistical metrics (diversity, entropy)
        5. Apply threshold filters
        6. Return detailed result
        
        The analysis short-circuits on first rejection for efficiency,
        but computes all metrics for accepted chunks.
        
        Args:
            text: Text chunk to analyze
            
        Returns:
            ChunkQualityResult with all metrics and decision
        """
        self._analysis_count += 1
        
        # Initialize result with basic statistics
        char_count = len(text)
        words = tokenize_simple(text)
        word_count = len(words)
        
        # URL and citation counts
        url_count = len(URL_PATTERN.findall(text))
        citation_count = len(CITATION_PATTERN.findall(text))
        
        # Pattern detection
        is_bibliography, bib_confidence = detect_bibliography(text)
        is_transcript, trans_confidence = detect_transcript(text)
        is_boilerplate, boiler_confidence = detect_boilerplate(text)
        
        # Statistical metrics
        lexical_diversity = calculate_lexical_diversity(text)
        entropy = calculate_entropy(text)
        whitespace_ratio = calculate_whitespace_ratio(text)
        
        # Build result object
        result = ChunkQualityResult(
            should_keep=True,  # Assume keep until proven otherwise
            char_count=char_count,
            word_count=word_count,
            lexical_diversity=lexical_diversity,
            entropy=entropy,
            url_count=url_count,
            citation_count=citation_count,
            is_bibliography=is_bibliography,
            is_transcript=is_transcript,
            confidence=1.0,
        )
        
        # Apply filters (order matters for efficiency)
        
        # 1. Length filters
        if char_count < self.min_chars:
            return self._reject(result, FilterReason.TOO_SHORT)
        
        if word_count < self.min_words:
            return self._reject(result, FilterReason.TOO_FEW_WORDS)
        
        # 2. Whitespace filter
        if whitespace_ratio > self.max_whitespace_ratio:
            return self._reject(result, FilterReason.WHITESPACE_DOMINANT)
        
        # 3. URL density filter
        if url_count > self.max_url_count:
            return self._reject(result, FilterReason.HIGH_URL_DENSITY)
        
        # 4. Citation density filter
        if citation_count > self.max_citation_count:
            return self._reject(result, FilterReason.HIGH_CITATION_DENSITY)
        
        # 5. Pattern-based filters
        if self.filter_bibliography and is_bibliography:
            result.confidence = 1.0 - bib_confidence
            return self._reject(result, FilterReason.BIBLIOGRAPHY)
        
        if self.filter_transcript and is_transcript:
            result.confidence = 1.0 - trans_confidence
            return self._reject(result, FilterReason.TRANSCRIPT)
        
        if self.filter_boilerplate and is_boilerplate:
            result.confidence = 1.0 - boiler_confidence
            return self._reject(result, FilterReason.BOILERPLATE)
        
        # 6. Statistical filters
        if lexical_diversity < self.min_diversity:
            return self._reject(result, FilterReason.LOW_DIVERSITY)
        
        if entropy < self.min_entropy:
            return self._reject(result, FilterReason.LOW_ENTROPY)
        
        # Passed all filters
        return result
    
    def _reject(
        self, 
        result: ChunkQualityResult, 
        reason: FilterReason
    ) -> ChunkQualityResult:
        """
        Mark result as rejected and update statistics.
        
        Args:
            result: Result object to update
            reason: Rejection reason
            
        Returns:
            Updated result object
        """
        result.should_keep = False
        result.rejection_reason = reason
        self._rejection_counts[reason] += 1
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get filtering statistics.
        
        Returns:
            Dictionary with analysis counts and rejection breakdown
        """
        total_rejected = sum(self._rejection_counts.values())
        
        return {
            "total_analyzed": self._analysis_count,
            "total_rejected": total_rejected,
            "total_kept": self._analysis_count - total_rejected,
            "rejection_rate": total_rejected / self._analysis_count if self._analysis_count > 0 else 0,
            "rejection_breakdown": {
                reason.value: count 
                for reason, count in self._rejection_counts.items() 
                if count > 0
            },
        }
    
    def reset_statistics(self) -> None:
        """Reset analysis statistics."""
        self._analysis_count = 0
        self._rejection_counts = {r: 0 for r in FilterReason}
    
    def print_statistics(self) -> None:
        """Print formatted statistics to console."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 50)
        print("CHUNK QUALITY ANALYZER STATISTICS")
        print("=" * 50)
        print(f"Total Analyzed:  {stats['total_analyzed']}")
        print(f"Total Kept:      {stats['total_kept']}")
        print(f"Total Rejected:  {stats['total_rejected']}")
        print(f"Rejection Rate:  {stats['rejection_rate']:.1%}")
        
        if stats['rejection_breakdown']:
            print("\nRejection Breakdown:")
            for reason, count in sorted(
                stats['rejection_breakdown'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                pct = count / stats['total_rejected'] * 100 if stats['total_rejected'] > 0 else 0
                print(f"  {reason}: {count} ({pct:.1f}%)")
        
        print("=" * 50)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def clean_and_normalize(text: str) -> str:
    """
    Clean and normalize text for quality analysis.
    
    OPERATIONS:
    
    1. Unicode normalization (NFC form)
    2. Whitespace normalization
    3. Control character removal
    4. Empty line consolidation
    
    This function prepares text for consistent quality assessment
    across documents with different encodings and formatting.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    import unicodedata
    
    # Unicode normalization (composed form)
    text = unicodedata.normalize('NFC', text)
    
    # Remove control characters (except newlines and tabs)
    text = ''.join(
        c for c in text 
        if not unicodedata.category(c).startswith('C') or c in '\n\t'
    )
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
    
    return text.strip()


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def filter_chunks(
    chunks: List[Any],
    text_accessor: callable = lambda x: x.page_content if hasattr(x, 'page_content') else str(x),
    analyzer: Optional[ChunkQualityAnalyzer] = None,
    return_rejected: bool = False,
) -> Dict[str, List[Any]]:
    """
    Filter a list of chunks using quality analysis.
    
    This function provides batch filtering with detailed statistics,
    useful for thesis documentation.
    
    Args:
        chunks: List of chunk objects
        text_accessor: Function to extract text from chunk object
        analyzer: ChunkQualityAnalyzer instance (created if None)
        return_rejected: Include rejected chunks in output
        
    Returns:
        Dictionary with:
        - 'kept': List of chunks that passed filtering
        - 'rejected': List of rejected chunks (if return_rejected=True)
        - 'statistics': Filtering statistics
        
    Example:
        from langchain.schema import Document
        
        chunks = [Document(page_content="..."), ...]
        result = filter_chunks(chunks)
        
        print(f"Kept {len(result['kept'])} of {len(chunks)} chunks")
        print(f"Rejection rate: {result['statistics']['rejection_rate']:.1%}")
    """
    if analyzer is None:
        analyzer = ChunkQualityAnalyzer()
    
    analyzer.reset_statistics()
    
    kept = []
    rejected = []
    
    for chunk in chunks:
        text = text_accessor(chunk)
        result = analyzer.analyze(text)
        
        if result.should_keep:
            kept.append(chunk)
        elif return_rejected:
            rejected.append(chunk)
    
    output = {
        'kept': kept,
        'statistics': analyzer.get_statistics(),
    }
    
    if return_rejected:
        output['rejected'] = rejected
    
    return output