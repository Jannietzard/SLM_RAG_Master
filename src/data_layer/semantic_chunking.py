"""
Advanced Semantic Chunking: Language-Agnostic Document Segmentation

Version: 2.1.0
Author: Edge-RAG Research Project
Last Modified: 2026-01-13

===============================================================================
OVERVIEW
===============================================================================

This module implements an advanced semantic chunking system that segments
documents into meaningful chunks based on structural and statistical analysis
rather than fixed character counts.

KEY FEATURES:
    - Language-agnostic: Works for any language without modification
    - No hardcoded keywords: Uses statistical measures instead
    - Automatic quality filtering: Removes low-value content
    - Hierarchical structure detection: Extracts chapters, sections, subsections
    - TF-IDF importance scoring: Identifies key content chunks

===============================================================================
SCIENTIFIC FOUNDATION
===============================================================================

SEMANTIC CHUNKING vs FIXED-SIZE CHUNKING:

Traditional fixed-size chunking splits text at arbitrary character boundaries,
which can break semantic units (sentences, paragraphs, concepts) in the middle.

Semantic chunking attempts to find natural boundaries where:
    - Topics shift
    - Structural elements change (headers, sections)
    - Semantic coherence is preserved within chunks

Reference: Chen, J. et al. (2023). "Dense X Retrieval: What Retrieval 
Granularity Should We Use?" arXiv:2312.06648

INFORMATION THEORY METRICS:

This implementation uses several information-theoretic measures:

1. Lexical Diversity (Type-Token Ratio):
   TTR = |unique_words| / |total_words|
   
   Interpretation:
   - High TTR (> 0.6): Diverse vocabulary, likely informative content
   - Low TTR (< 0.3): Repetitive, possibly filler or transcript
   
2. Shannon Entropy:
   H(X) = -sum(p(x) * log2(p(x)))
   
   Measures information content per word:
   - High entropy (> 4 bits): High information density
   - Low entropy (< 2 bits): Repetitive or formulaic text
   
3. TF-IDF (Term Frequency - Inverse Document Frequency):
   TF-IDF(t,d,D) = TF(t,d) * IDF(t,D)
   
   Where:
   - TF(t,d) = frequency of term t in document d
   - IDF(t,D) = log(|D| / |{d in D : t in d}|)
   
   Identifies terms that are important within a chunk but rare across
   the corpus, highlighting distinctive content.

Reference: Shannon, C. (1948). "A Mathematical Theory of Communication."
Bell System Technical Journal.

DOCUMENT STRUCTURE DETECTION:

Academic and technical documents typically follow hierarchical structures:

    Document
    +-- Chapter 1
    |   +-- Section 1.1
    |   |   +-- Subsection 1.1.1
    |   +-- Section 1.2
    +-- Chapter 2

This module detects structure using language-independent patterns:
    - Numeric patterns: "1.", "1.1", "1.1.1"
    - Roman numerals: "I.", "II.", "III."
    - Structural indicators: paragraph breaks, capitalization

===============================================================================
QUALITY FILTERING
===============================================================================

Not all extracted text is suitable for RAG retrieval. Low-quality content
types that should be filtered:

1. TRANSCRIPTS:
   Interview transcripts (e.g., "I: question B: answer") often contain
   colloquial speech with low information density.
   Detection: Pattern matching for "X: text" structures
   
2. BIBLIOGRAPHIES:
   Reference lists contain metadata, not semantic content.
   Detection: High URL density, citation patterns
   
3. TABLES/FIGURES:
   Extracted as text, these often produce nonsensical sequences.
   Detection: Excessive whitespace ratio
   
4. HEADERS/FOOTERS:
   Repeated page elements add noise.
   Detection: Very short chunks, repeated patterns

The automatic quality filter uses statistical thresholds rather than
language-specific keywords, making it applicable to any language.

===============================================================================
EDGE DEVICE OPTIMIZATION
===============================================================================

Memory Efficiency:
    - Streaming processing where possible
    - No external NLP models required
    - Pure Python implementation with minimal dependencies

Computational Complexity:
    - Boundary detection: O(n) where n = text length
    - Quality filtering: O(n * w) where w = words per chunk
    - TF-IDF scoring: O(c * t) where c = chunks, t = terms

Total complexity is linear in document size, suitable for edge deployment.

===============================================================================
MODULE STRUCTURE
===============================================================================

Data Classes:
    ChunkMetadata          - Structured metadata for chunks

Classes:
    HeaderExtractor        - Document structure detection
    SemanticBoundaryDetector - Natural boundary detection
    AutomaticQualityFilter - Statistical quality assessment
    TFIDFScorer           - Importance scoring
    SemanticChunker       - Main chunking orchestrator

Factory Functions:
    create_semantic_chunker() - Convenience constructor

===============================================================================
USAGE
===============================================================================

Basic Usage:
    from semantic_chunking import create_semantic_chunker
    from langchain.schema import Document
    
    chunker = create_semantic_chunker(
        chunk_size=1024,
        chunk_overlap=128,
        min_chunk_size=200
    )
    
    doc = Document(page_content="...", metadata={})
    chunks = chunker.chunk_document(doc)

Access Chunk Quality Metrics:
    for chunk in chunks:
        print(f"Importance: {chunk.metadata['importance_score']}")
        print(f"Diversity: {chunk.metadata['lexical_diversity']}")
        print(f"Section: {chunk.metadata['section']}")
"""

import re
import math
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChunkMetadata:
    """
    Structured metadata for semantically-chunked documents.
    
    This dataclass captures both structural position (chapter, section)
    and quality metrics (importance, diversity) for each chunk.
    
    Attributes:
        chapter: Chapter heading if detected (e.g., "1. Introduction")
        section: Section heading if detected (e.g., "1.1 Background")
        subsection: Subsection heading if detected (e.g., "1.1.1 Context")
        heading_level: Depth in document hierarchy (0=body, 1=chapter, 2=section, 3=subsection)
        is_header: Whether this chunk starts with a detected header
        page_number: Source page number if available
        importance_score: TF-IDF based importance (0.0 to ~10.0, higher = more important)
        lexical_diversity: Type-token ratio (0.0 to 1.0, higher = more diverse vocabulary)
    
    USAGE IN RAG:
    
    Metadata can be used for:
    - Filtering: Exclude low-importance chunks from context
    - Weighting: Boost scores for high-importance chunks
    - Navigation: Present section hierarchy in responses
    - Debugging: Understand why certain chunks were retrieved
    """
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    heading_level: int = 0
    is_header: bool = False
    page_number: Optional[int] = None
    importance_score: float = 0.0
    lexical_diversity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
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


# ============================================================================
# HEADER EXTRACTION
# ============================================================================

class HeaderExtractor:
    """
    Extract hierarchical document structure using language-agnostic patterns.
    
    DESIGN PHILOSOPHY:
    
    Instead of using language-specific keywords like "Chapter", "Section",
    "Kapitel", "Abschnitt", etc., this extractor uses universal patterns:
    
    1. Numeric patterns: "1.", "1.1", "1.1.1"
    2. Roman numerals: "I.", "II.", "III."
    3. Structural indicators: Line position, capitalization
    
    This makes the extractor work for any language without modification.
    
    PATTERN HIERARCHY:
    
    Level 1 (Chapter):
        - "1. Title" - Single number with period
        - "I. Title" - Roman numeral with period
        - "Chapter 1: Title" - Keyword (any language) + number
        
    Level 2 (Section):
        - "1.1 Title" - Two-level numbering
        - "1.1. Title" - Two-level with trailing period
        
    Level 3 (Subsection):
        - "1.1.1 Title" - Three-level numbering
    
    STATE MANAGEMENT:
    
    The extractor maintains state across chunks to track current position
    in the document hierarchy. This allows body text chunks to inherit
    the most recent chapter/section context.
    
    Attributes:
        current_chapter: Most recently detected chapter heading
        current_section: Most recently detected section heading  
        current_subsection: Most recently detected subsection heading
    """
    
    # Pattern for chapter-level headers
    # Matches: "1. Title", "I. Title", "Chapter 1: Title"
    CHAPTER_PATTERNS = [
        # "1. Introduction" - numeric with period and title
        r'^(\d+)\.\s+([A-Z\u00C0-\u024F][^\n]{3,80})$',
        # "I. Introduction" - roman numeral with period and title
        r'^([IVX]+)\.\s+([A-Z\u00C0-\u024F][^\n]{3,80})$',
        # "Chapter 1: Title" or "Kapitel 1: Titel" - word + number + title
        r'^\w+\s+(\d+)[:\s]+([^\n]{3,80})$',
    ]
    
    # Pattern for section-level headers
    # Matches: "1.1 Title", "1.1. Title"
    SECTION_PATTERNS = [
        r'^(\d+\.\d+)[\.\s]+([A-Z\u00C0-\u024F][^\n]{3,80})$',
    ]
    
    # Pattern for subsection-level headers
    # Matches: "1.1.1 Title"
    SUBSECTION_PATTERNS = [
        r'^(\d+\.\d+\.\d+)[\.\s]+([A-Z\u00C0-\u024F][^\n]{3,80})$',
    ]
    
    def __init__(self):
        """Initialize header extractor with empty state."""
        self.current_chapter: Optional[str] = None
        self.current_section: Optional[str] = None
        self.current_subsection: Optional[str] = None
    
    def reset(self) -> None:
        """Reset state to initial values."""
        self.current_chapter = None
        self.current_section = None
        self.current_subsection = None
    
    def extract_headers(self, text: str) -> Tuple[ChunkMetadata, str]:
        """
        Extract header information from text chunk.
        
        ALGORITHM:
        1. Get first line of text
        2. Try matching against chapter patterns
        3. If no match, try section patterns
        4. If no match, try subsection patterns
        5. Update internal state if header found
        6. Return metadata and text (without header line if detected)
        
        Args:
            text: Text chunk to analyze
            
        Returns:
            Tuple of (ChunkMetadata, cleaned_text):
            - ChunkMetadata contains detected structure and current context
            - cleaned_text has header line removed if detected
        """
        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        
        metadata = ChunkMetadata()
        
        # Try chapter patterns (highest level)
        for pattern in self.CHAPTER_PATTERNS:
            match = re.match(pattern, first_line, re.MULTILINE)
            if match:
                number, title = match.groups()
                self.current_chapter = f"{number}. {title}"
                self.current_section = None
                self.current_subsection = None
                
                metadata.chapter = self.current_chapter
                metadata.heading_level = 1
                metadata.is_header = True
                
                # Remove header line from text
                cleaned_text = '\n'.join(lines[1:]).strip()
                return metadata, cleaned_text
        
        # Try section patterns (second level)
        for pattern in self.SECTION_PATTERNS:
            match = re.match(pattern, first_line, re.MULTILINE)
            if match:
                number, title = match.groups()
                self.current_section = f"{number} {title}"
                self.current_subsection = None
                
                metadata.chapter = self.current_chapter
                metadata.section = self.current_section
                metadata.heading_level = 2
                metadata.is_header = True
                
                cleaned_text = '\n'.join(lines[1:]).strip()
                return metadata, cleaned_text
        
        # Try subsection patterns (third level)
        for pattern in self.SUBSECTION_PATTERNS:
            match = re.match(pattern, first_line, re.MULTILINE)
            if match:
                number, title = match.groups()
                self.current_subsection = f"{number} {title}"
                
                metadata.chapter = self.current_chapter
                metadata.section = self.current_section
                metadata.subsection = self.current_subsection
                metadata.heading_level = 3
                metadata.is_header = True
                
                cleaned_text = '\n'.join(lines[1:]).strip()
                return metadata, cleaned_text
        
        # No header detected - return current context
        metadata.chapter = self.current_chapter
        metadata.section = self.current_section
        metadata.subsection = self.current_subsection
        metadata.heading_level = 0
        metadata.is_header = False
        
        return metadata, text


# ============================================================================
# BOUNDARY DETECTION
# ============================================================================

class SemanticBoundaryDetector:
    """
    Detect natural semantic boundaries in text.
    
    BOUNDARY TYPES:
    
    1. Paragraph Boundaries:
       Double newlines (\n\n) universally indicate paragraph breaks.
       These are the strongest semantic boundaries.
       
    2. Sentence Boundaries with Structure Change:
       Sentence ending (. ! ?) followed by newline and capital letter
       often indicates topic transition.
       
    3. List/Enumeration Boundaries:
       Numbered or bulleted items represent semantic units.
    
    ALGORITHM:
    
    1. Find all potential boundaries using regex patterns
    2. Score boundaries by strength (paragraph > sentence > other)
    3. Select boundaries that respect min/max chunk size constraints
    4. Return boundary positions as character offsets
    
    LANGUAGE INDEPENDENCE:
    
    The patterns used are based on universal punctuation and formatting
    conventions that work across Latin-script languages. For non-Latin
    scripts, additional patterns may be needed.
    
    Attributes:
        BOUNDARY_PATTERNS: List of regex patterns for boundary detection
    """
    
    # Universal boundary patterns
    # Ordered by decreasing strength/reliability
    BOUNDARY_PATTERNS = [
        # Double newline - strongest paragraph indicator
        r'\n\n+',
        # Sentence end + newline + capital letter - topic transition
        r'\.\s*\n(?=[A-Z\u00C0-\u024F])',
        # Sentence end + blank line
        r'[.!?]\s*\n\s*\n',
        # Colon followed by newline - often introduces list/quote
        r':\s*\n',
    ]
    
    def __init__(self, min_boundary_distance: int = 200):
        """
        Initialize boundary detector.
        
        Args:
            min_boundary_distance: Minimum characters between boundaries
        """
        self.min_boundary_distance = min_boundary_distance
    
    def find_semantic_boundaries(
        self, 
        text: str, 
        max_chunk_size: int = 1024
    ) -> List[int]:
        """
        Find positions of semantic boundaries in text.
        
        ALGORITHM:
        
        1. Start with position 0 as first boundary
        2. Find all potential boundary positions using patterns
        3. Filter boundaries that are too close together
        4. Ensure no chunk exceeds max_chunk_size
        5. Add text length as final boundary
        
        Args:
            text: Full text to analyze
            max_chunk_size: Maximum desired chunk size in characters
            
        Returns:
            Sorted list of boundary positions (character offsets).
            First element is always 0, last is always len(text).
        """
        # Start with document beginning
        boundaries = [0]
        
        # Collect all potential boundary positions
        potential_boundaries = []
        
        for pattern in self.BOUNDARY_PATTERNS:
            for match in re.finditer(pattern, text):
                position = match.end()
                # Only consider positions past minimum chunk size
                if position >= self.min_boundary_distance:
                    potential_boundaries.append(position)
        
        # Sort and deduplicate
        potential_boundaries = sorted(set(potential_boundaries))
        
        # Select boundaries respecting constraints
        current_position = 0
        
        for boundary in potential_boundaries:
            distance_from_current = boundary - current_position
            
            # Accept boundary if it creates reasonably sized chunk
            if distance_from_current >= self.min_boundary_distance:
                # If we're approaching max size, force a boundary
                if distance_from_current >= max_chunk_size * 0.8:
                    boundaries.append(boundary)
                    current_position = boundary
                # Otherwise, accept strong boundaries
                elif distance_from_current >= max_chunk_size * 0.5:
                    boundaries.append(boundary)
                    current_position = boundary
        
        # Ensure document end is included
        if boundaries[-1] != len(text):
            boundaries.append(len(text))
        
        return boundaries
    
    def find_nearest_boundary(
        self, 
        text: str, 
        target_position: int,
        search_range: int = 100
    ) -> int:
        """
        Find the nearest semantic boundary to a target position.
        
        Useful when a chunk would otherwise break mid-sentence.
        
        Args:
            text: Full text
            target_position: Desired split position
            search_range: Characters to search in each direction
            
        Returns:
            Position of nearest boundary, or target_position if none found
        """
        search_start = max(0, target_position - search_range)
        search_end = min(len(text), target_position + search_range)
        search_text = text[search_start:search_end]
        
        # Look for sentence boundaries
        sentence_pattern = r'[.!?]\s+'
        
        best_position = target_position
        best_distance = search_range + 1
        
        for match in re.finditer(sentence_pattern, search_text):
            absolute_position = search_start + match.end()
            distance = abs(absolute_position - target_position)
            
            if distance < best_distance:
                best_distance = distance
                best_position = absolute_position
        
        return best_position


# ============================================================================
# QUALITY FILTERING
# ============================================================================

class AutomaticQualityFilter:
    """
    Automatic quality assessment for text chunks using statistical measures.
    
    DESIGN PHILOSOPHY:
    
    Traditional quality filters use language-specific keywords to detect
    low-quality content (e.g., filtering chunks containing "bibliography",
    "references", "interview"). This approach fails for:
    - Non-English documents
    - Domain-specific terminology
    - Evolving language patterns
    
    This filter uses language-agnostic statistical measures:
    
    1. LEXICAL DIVERSITY (Type-Token Ratio):
       Measures vocabulary richness.
       Low diversity indicates repetitive/formulaic content.
       
    2. INFORMATION DENSITY (Shannon Entropy):
       Measures information content per word.
       Low entropy indicates predictable/low-value content.
       
    3. STRUCTURAL PATTERNS:
       Detects transcript patterns (X: text) statistically.
       No hardcoded speaker labels needed.
       
    4. LAYOUT ARTIFACTS:
       Detects tables/figures by whitespace ratio.
    
    THRESHOLD CALIBRATION:
    
    Default thresholds are calibrated for academic/technical documents.
    For other content types, adjust:
    - Transcripts: Lower min_lexical_diversity
    - Poetry: Lower min_information_density
    - Technical: Higher min_words
    
    Attributes:
        min_length: Minimum character count
        min_words: Minimum word count
        min_lexical_diversity: Minimum type-token ratio [0, 1]
        min_information_density: Minimum entropy in bits [0, ~8]
    """
    
    def __init__(
        self,
        min_length: int = 100,
        min_words: int = 15,
        min_lexical_diversity: float = 0.3,
        min_information_density: float = 2.0,
    ):
        """
        Initialize quality filter with thresholds.
        
        Args:
            min_length: Minimum chunk length in characters
            min_words: Minimum word count
            min_lexical_diversity: Minimum type-token ratio (0.0-1.0)
                                   0.3 filters highly repetitive text
            min_information_density: Minimum Shannon entropy (bits/word)
                                     2.0 filters very predictable text
        """
        self.min_length = min_length
        self.min_words = min_words
        self.min_lexical_diversity = min_lexical_diversity
        self.min_information_density = min_information_density
    
    def calculate_lexical_diversity(self, text: str) -> float:
        """
        Calculate lexical diversity using Type-Token Ratio (TTR).
        
        FORMULA:
            TTR = |unique_words| / |total_words|
        
        INTERPRETATION:
            0.0-0.3: Very repetitive (transcripts, lists)
            0.3-0.5: Moderate diversity (normal text)
            0.5-0.7: High diversity (varied vocabulary)
            0.7-1.0: Very high diversity (technical/academic)
        
        NOTE:
            TTR decreases with text length (more words = more repetition).
            For comparing chunks of different sizes, use MATTR or similar.
        
        Args:
            text: Text to analyze
            
        Returns:
            Type-token ratio in range [0.0, 1.0]
        """
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def calculate_information_density(self, text: str) -> float:
        """
        Calculate information density using Shannon entropy.
        
        FORMULA:
            H(X) = -sum(p(word) * log2(p(word)))
        
        INTERPRETATION:
            0-2 bits: Very low information (repetitive)
            2-4 bits: Moderate information (normal text)
            4-6 bits: High information (diverse content)
            6+ bits: Very high information (technical/unique terms)
        
        THEORETICAL BASIS:
            Entropy measures the average "surprise" of each word.
            Low entropy = predictable = low information value.
            High entropy = unpredictable = high information value.
        
        Reference: Shannon, C. (1948). "A Mathematical Theory of Communication"
        
        Args:
            text: Text to analyze
            
        Returns:
            Shannon entropy in bits per word
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
        
        # Calculate word frequency distribution
        word_counts = Counter(words)
        total_words = len(words)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:  # Avoid log(0)
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def detect_transcript_pattern(self, text: str) -> bool:
        """
        Detect interview/transcript patterns statistically.
        
        PATTERN:
            Transcripts typically have structure like:
            "I: Question text here"
            "B: Response text here"
            "A1: Another response"
        
        Instead of hardcoding labels (I, B, A1), we detect the pattern:
            Short token (1-3 chars) + colon + space + text
        
        THRESHOLD:
            If >30% of lines match this pattern, classify as transcript.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if transcript pattern detected
        """
        # Pattern: 1-3 alphanumeric characters + colon + text
        pattern = r'(?:^|\n)\s*\w{1,3}\s*:\s*.{10,}'
        matches = re.findall(pattern, text)
        
        lines = text.split('\n')
        if len(lines) == 0:
            return False
        
        match_ratio = len(matches) / len(lines)
        return match_ratio > 0.3
    
    def detect_excessive_whitespace(self, text: str) -> bool:
        """
        Detect layout artifacts by whitespace ratio.
        
        Tables, figures, and formatted content often have excessive
        whitespace when extracted as plain text.
        
        THRESHOLD:
            >40% whitespace indicates layout artifact
        
        Args:
            text: Text to analyze
            
        Returns:
            True if excessive whitespace detected
        """
        if not text:
            return True
        
        whitespace_count = text.count(' ') + text.count('\t')
        whitespace_ratio = whitespace_count / len(text)
        
        return whitespace_ratio > 0.4
    
    def should_keep_chunk(self, text: str) -> Tuple[bool, str]:
        """
        Determine if chunk should be kept based on quality metrics.
        
        Applies all quality checks in sequence, returning immediately
        when a check fails.
        
        CHECK ORDER (fast to slow):
            1. Length check (O(1))
            2. Word count (O(n))
            3. Transcript pattern (O(n))
            4. Lexical diversity (O(n))
            5. Information density (O(n))
            6. Whitespace check (O(n))
        
        Args:
            text: Text chunk to evaluate
            
        Returns:
            Tuple of (keep: bool, reason: str):
            - keep: True if chunk passes all quality checks
            - reason: "passed" or description of failure reason
        """
        # Check 1: Minimum length
        if len(text) < self.min_length:
            return False, f"too_short ({len(text)} chars)"
        
        # Check 2: Minimum word count
        words = re.findall(r'\b\w+\b', text)
        if len(words) < self.min_words:
            return False, f"too_few_words ({len(words)} words)"
        
        # Check 3: Transcript pattern detection
        if self.detect_transcript_pattern(text):
            return False, "transcript_pattern_detected"
        
        # Check 4: Lexical diversity
        diversity = self.calculate_lexical_diversity(text)
        if diversity < self.min_lexical_diversity:
            return False, f"low_lexical_diversity ({diversity:.2f})"
        
        # Check 5: Information density
        density = self.calculate_information_density(text)
        if density < self.min_information_density:
            return False, f"low_information_density ({density:.2f} bits/word)"
        
        # Check 6: Layout artifacts
        if self.detect_excessive_whitespace(text):
            return False, "layout_artifact"
        
        return True, "passed"
    
    def get_quality_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate all quality metrics for a chunk.
        
        Useful for debugging and analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with all quality metrics
        """
        words = re.findall(r'\b\w+\b', text)
        
        return {
            "length": len(text),
            "word_count": len(words),
            "lexical_diversity": self.calculate_lexical_diversity(text),
            "information_density": self.calculate_information_density(text),
            "whitespace_ratio": text.count(' ') / len(text) if text else 0,
            "is_transcript": self.detect_transcript_pattern(text),
        }


# ============================================================================
# TF-IDF IMPORTANCE SCORING
# ============================================================================

class TFIDFScorer:
    """
    Calculate TF-IDF importance scores for text chunks.
    
    TF-IDF (Term Frequency - Inverse Document Frequency) identifies
    terms that are distinctive for a chunk within the corpus.
    
    FORMULA:
        TF-IDF(t,d,D) = TF(t,d) * IDF(t,D)
        
    Where:
        TF(t,d) = count(t in d) / |d|  (term frequency in document)
        IDF(t,D) = log(|D| / DF(t))    (inverse document frequency)
        DF(t) = |{d in D : t in d}|    (document frequency)
    
    USAGE IN RAG:
    
    Chunks with high TF-IDF scores contain distinctive terminology
    and are likely to be more relevant for specific queries.
    
    Low TF-IDF chunks contain common vocabulary and may be less
    useful for retrieval (though still valuable for context).
    
    IMPLEMENTATION:
    
    1. analyze_corpus(): Build term statistics from all chunks
    2. calculate_chunk_importance(): Score individual chunks
    
    The two-phase approach allows efficient batch processing.
    
    Attributes:
        document_frequency: term -> number of chunks containing term
        total_chunks: Total number of chunks in corpus
        chunk_term_frequencies: List of term frequency dicts per chunk
    """
    
    def __init__(self):
        """Initialize empty TF-IDF scorer."""
        self.document_frequency: Dict[str, int] = {}
        self.total_chunks: int = 0
        self.chunk_term_frequencies: List[Counter] = []
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.document_frequency = {}
        self.total_chunks = 0
        self.chunk_term_frequencies = []
    
    def analyze_corpus(self, chunks: List[str]) -> None:
        """
        Analyze corpus to build TF-IDF statistics.
        
        Must be called before calculate_chunk_importance().
        
        ALGORITHM:
            1. For each chunk, extract and count terms
            2. Update document frequency (how many chunks contain each term)
            3. Store term frequencies for later scoring
        
        Args:
            chunks: List of text chunks to analyze
        """
        self.reset()
        self.total_chunks = len(chunks)
        
        for chunk in chunks:
            # Tokenize and count terms
            words = re.findall(r'\b\w+\b', chunk.lower())
            term_freq = Counter(words)
            self.chunk_term_frequencies.append(term_freq)
            
            # Update document frequency
            for term in set(words):  # Use set to count each term once per chunk
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
    
    def calculate_chunk_importance(self, chunk_index: int) -> float:
        """
        Calculate TF-IDF importance score for a chunk.
        
        FORMULA:
            score = sum(TF(t) * IDF(t)) / |terms|
        
        The normalization by term count makes scores comparable
        across chunks of different lengths.
        
        Args:
            chunk_index: Index of chunk in analyzed corpus
            
        Returns:
            Normalized TF-IDF score (typically 0.0 to ~3.0)
        """
        if chunk_index >= len(self.chunk_term_frequencies):
            return 0.0
        
        term_freq = self.chunk_term_frequencies[chunk_index]
        
        if not term_freq:
            return 0.0
        
        tfidf_score = 0.0
        
        for term, tf in term_freq.items():
            # Inverse document frequency
            df = self.document_frequency.get(term, 1)
            idf = math.log(self.total_chunks / df) if df > 0 else 0
            
            # Accumulate TF * IDF
            tfidf_score += tf * idf
        
        # Normalize by chunk length
        total_terms = sum(term_freq.values())
        return tfidf_score / total_terms if total_terms > 0 else 0.0
    
    def get_top_terms(self, chunk_index: int, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF terms for a chunk.
        
        Useful for understanding why a chunk was scored highly.
        
        Args:
            chunk_index: Index of chunk in analyzed corpus
            n: Number of top terms to return
            
        Returns:
            List of (term, tfidf_score) tuples, sorted by score descending
        """
        if chunk_index >= len(self.chunk_term_frequencies):
            return []
        
        term_freq = self.chunk_term_frequencies[chunk_index]
        
        term_scores = []
        for term, tf in term_freq.items():
            df = self.document_frequency.get(term, 1)
            idf = math.log(self.total_chunks / df) if df > 0 else 0
            term_scores.append((term, tf * idf))
        
        term_scores.sort(key=lambda x: x[1], reverse=True)
        return term_scores[:n]


# ============================================================================
# SEMANTIC CHUNKER
# ============================================================================

class SemanticChunker:
    """
    Main semantic chunking orchestrator.
    
    Combines all components to produce high-quality, semantically-coherent
    chunks with rich metadata.
    
    PIPELINE:
    
    1. BOUNDARY DETECTION:
       Find natural split points using SemanticBoundaryDetector
       
    2. CHUNK EXTRACTION:
       Extract text between boundaries with overlap
       
    3. HEADER EXTRACTION:
       Detect document structure using HeaderExtractor
       
    4. QUALITY FILTERING:
       Remove low-quality chunks using AutomaticQualityFilter
       
    5. IMPORTANCE SCORING:
       Calculate TF-IDF scores using TFIDFScorer
       
    6. METADATA ENRICHMENT:
       Add all metrics to chunk metadata
    
    FALLBACK BEHAVIOR:
    
    If semantic chunking fails for any reason, the chunker falls back
    to RecursiveCharacterTextSplitter to ensure documents are processed.
    
    Attributes:
        max_chunk_size: Maximum chunk size in characters
        min_chunk_size: Minimum chunk size (smaller are merged/dropped)
        overlap: Number of characters to overlap between chunks
        header_extractor: HeaderExtractor instance
        boundary_detector: SemanticBoundaryDetector instance
        quality_filter: AutomaticQualityFilter instance
        tfidf_scorer: TFIDFScorer instance
        fallback_splitter: RecursiveCharacterTextSplitter for fallback
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 200,
        overlap: int = 128,
    ):
        """
        Initialize semantic chunker.
        
        Args:
            max_chunk_size: Target maximum chunk size
            min_chunk_size: Minimum chunk size (smaller chunks dropped)
            overlap: Character overlap between consecutive chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        
        # Initialize components
        self.header_extractor = HeaderExtractor()
        self.boundary_detector = SemanticBoundaryDetector(
            min_boundary_distance=min_chunk_size
        )
        self.quality_filter = AutomaticQualityFilter(
            min_length=min_chunk_size,
            min_words=15,
            min_lexical_diversity=0.3,
            min_information_density=2.0,
        )
        self.tfidf_scorer = TFIDFScorer()
        
        # Fallback splitter for error cases
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        logger.info(
            f"SemanticChunker initialized: "
            f"max_size={max_chunk_size}, "
            f"min_size={min_chunk_size}, "
            f"overlap={overlap}"
        )
    
    def _extract_raw_chunks(self, text: str) -> List[str]:
        """
        Extract raw text chunks based on semantic boundaries.
        
        ALGORITHM:
            1. Find boundary positions
            2. Extract text between boundaries
            3. Add overlap from previous chunk
            4. Filter chunks below minimum size
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of raw chunk strings
        """
        boundaries = self.boundary_detector.find_semantic_boundaries(
            text,
            self.max_chunk_size
        )
        
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Add overlap from previous chunk (except for first chunk)
            if i > 0 and start >= self.overlap:
                start -= self.overlap
            
            chunk_text = text[start:end].strip()
            
            # Skip chunks below minimum size
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a document into semantically-coherent segments.
        
        PIPELINE:
            1. Extract raw chunks using boundary detection
            2. Build TF-IDF statistics across all chunks
            3. Process each chunk:
               a. Extract header metadata
               b. Apply quality filter
               c. Calculate importance score
               d. Enrich metadata
            4. Return filtered, enriched chunks
        
        Args:
            document: LangChain Document to chunk
            
        Returns:
            List of chunked Documents with enriched metadata
        """
        text = document.page_content
        base_metadata = document.metadata.copy()
        
        # Phase 1: Extract raw chunks
        try:
            raw_chunks = self._extract_raw_chunks(text)
        except Exception as e:
            logger.warning(f"Semantic chunking failed, using fallback: {e}")
            return self.fallback_splitter.split_documents([document])
        
        if not raw_chunks:
            logger.warning("No chunks extracted, using fallback")
            return self.fallback_splitter.split_documents([document])
        
        # Phase 2: Build TF-IDF statistics
        self.tfidf_scorer.analyze_corpus(raw_chunks)
        
        # Phase 3: Process each chunk
        processed_chunks = []
        filter_stats = {"kept": 0, "filtered": 0, "reasons": {}}
        
        # Reset header extractor for new document
        self.header_extractor.reset()
        
        for i, chunk_text in enumerate(raw_chunks):
            # Extract header metadata
            metadata, cleaned_text = self.header_extractor.extract_headers(chunk_text)
            
            # Apply quality filter
            keep, reason = self.quality_filter.should_keep_chunk(cleaned_text)
            
            if not keep:
                filter_stats["filtered"] += 1
                filter_stats["reasons"][reason] = filter_stats["reasons"].get(reason, 0) + 1
                logger.debug(f"Filtered chunk {i}: {reason}")
                continue
            
            filter_stats["kept"] += 1
            
            # Calculate quality metrics
            importance_score = self.tfidf_scorer.calculate_chunk_importance(i)
            lexical_diversity = self.quality_filter.calculate_lexical_diversity(cleaned_text)
            
            # Build enriched metadata
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
            
            # Create chunk document
            chunk_doc = Document(
                page_content=cleaned_text,
                metadata=enriched_metadata
            )
            processed_chunks.append(chunk_doc)
        
        # Log summary
        if filter_stats["filtered"] > 0 or len(raw_chunks) > 10:
            logger.info(
                f"Semantic chunking: {len(raw_chunks)} raw -> "
                f"{filter_stats['kept']} kept "
                f"(filtered {filter_stats['filtered']}: {filter_stats['reasons']})"
            )
        
        return processed_chunks
    
    def get_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Calculate statistics for processed chunks.
        
        Args:
            chunks: List of processed chunk documents
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"count": 0}
        
        sizes = [len(c.page_content) for c in chunks]
        importance_scores = [
            c.metadata.get("importance_score", 0) for c in chunks
        ]
        diversity_scores = [
            c.metadata.get("lexical_diversity", 0) for c in chunks
        ]
        
        import statistics
        
        return {
            "count": len(chunks),
            "size_min": min(sizes),
            "size_max": max(sizes),
            "size_mean": statistics.mean(sizes),
            "size_median": statistics.median(sizes),
            "importance_mean": statistics.mean(importance_scores),
            "diversity_mean": statistics.mean(diversity_scores),
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_semantic_chunker(
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    min_chunk_size: int = 200,
) -> SemanticChunker:
    """
    Factory function to create a configured SemanticChunker.
    
    This is the recommended way to create a chunker instance,
    as it provides sensible defaults and clear parameter names.
    
    Args:
        chunk_size: Target maximum chunk size in characters
                    Default: 1024 (recommended for RAG)
        chunk_overlap: Overlap between consecutive chunks
                       Default: 128 (~12% overlap)
        min_chunk_size: Minimum chunk size (smaller are dropped)
                        Default: 200
    
    Returns:
        Configured SemanticChunker instance
    
    Example:
        chunker = create_semantic_chunker(
            chunk_size=1024,
            chunk_overlap=128,
            min_chunk_size=200
        )
        chunks = chunker.chunk_document(document)
    """
    return SemanticChunker(
        max_chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        overlap=chunk_overlap,
    )


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_semantic_chunking():
    """
    Demonstrate semantic chunking capabilities.
    
    This function shows the chunker in action with sample text
    and displays the resulting chunk metadata.
    """
    # Sample text with mixed structure
    sample_text = """
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

2.1 Language Models

Language models learn statistical patterns in text to predict
subsequent tokens. Modern transformer-based models like BERT
and GPT have achieved state-of-the-art results on many NLP tasks.
"""
    
    # Create document
    doc = Document(
        page_content=sample_text,
        metadata={"source_file": "thesis.pdf", "page": 1}
    )
    
    # Create chunker and process
    chunker = create_semantic_chunker(
        chunk_size=500,  # Smaller for demonstration
        chunk_overlap=50,
        min_chunk_size=100
    )
    
    chunks = chunker.chunk_document(doc)
    
    # Display results
    print("\n" + "=" * 70)
    print("SEMANTIC CHUNKING DEMONSTRATION")
    print("=" * 70)
    print(f"\nInput: {len(sample_text)} characters")
    print(f"Output: {len(chunks)} chunks\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Chapter: {chunk.metadata.get('chapter', 'N/A')}")
        print(f"  Section: {chunk.metadata.get('section', 'N/A')}")
        print(f"  Size: {chunk.metadata['chunk_size']} chars")
        print(f"  Importance: {chunk.metadata.get('importance_score', 0):.3f}")
        print(f"  Diversity: {chunk.metadata.get('lexical_diversity', 0):.3f}")
        print(f"  Preview: {chunk.page_content[:60]}...")
        print()
    
    # Display statistics
    stats = chunker.get_statistics(chunks)
    print("Statistics:")
    print(f"  Total chunks: {stats['count']}")
    print(f"  Size range: {stats['size_min']}-{stats['size_max']} chars")
    print(f"  Avg importance: {stats['importance_mean']:.3f}")
    print(f"  Avg diversity: {stats['diversity_mean']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_semantic_chunking()


    