"""
Advanced Semantic Chunking - Fully Automatic & Language-Agnostic.

KEINE manuellen Keywords!
KEINE Sprachfestlegung!
NUR: Dokument hochladen → automatische Analyse → intelligentes Chunking

Implementiert:
1. Automatic TF-IDF-based importance scoring (sprachunabhängig)
2. Statistical quality filtering (keine hardcoded keywords)
3. Header-based metadata extraction (multi-language patterns)
4. Linguistic diversity analysis (statt keyword matching)

Scientific Rationale:
- TF-IDF identifiziert automatisch wichtige Terme im Dokument-Kontext
- Statistical measures (lexical diversity, entropy) bewerten Chunk-Qualität
- Pattern-based header detection (language-independent numbering)
- Transcript detection durch statistische Muster, nicht Keywords
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter
import math

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Strukturierte Metadaten für Context-Aware Chunks."""
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    heading_level: int = 0
    is_header: bool = False
    page_number: Optional[int] = None
    importance_score: float = 0.0  # TF-IDF-basiert
    lexical_diversity: float = 0.0  # Automatisch berechnet


class HeaderExtractor:
    """
    Extrahiert hierarchische Überschriften - Language-Agnostic.
    
    Nutzt universelle Muster (Nummern, Römische Ziffern) statt Sprach-Keywords.
    """
    
    # Universal patterns (arbeiten für Deutsch, English, Französisch, etc.)
    CHAPTER_PATTERNS = [
        r'^(\d+)\.\s+([A-ZÄÖÜ\u00C0-\u017F][^\n]{3,80})$',     # "1. Title" (any language)
        r'^([IVX]+)\.\s+([A-ZÄÖÜ\u00C0-\u017F][^\n]{3,80})$',  # "I. Title"
        r'^\w+\s+(\d+)[:\s]+([^\n]{3,80})$',                    # "Kapitel/Chapter/Chapitre 1: Title"
    ]
    
    SECTION_PATTERNS = [
        r'^(\d+\.\d+)[\.\s]+([A-ZÄÖÜ\u00C0-\u017F][^\n]{3,80})$',  # "1.1 Title" or "1.1. Title"
    ]
    
    SUBSECTION_PATTERNS = [
        r'^(\d+\.\d+\.\d+)[\.\s]+([A-ZÄÖÜ\u00C0-\u017F][^\n]{3,80})$',  # "1.1.1 Title"
    ]
    
    def __init__(self):
        self.current_chapter = None
        self.current_section = None
        self.current_subsection = None
    
    def extract_headers(self, text: str) -> Tuple[ChunkMetadata, str]:
        """
        Extrahiere Header-Informationen aus Text (language-independent).
        
        Args:
            text: Text Chunk
            
        Returns:
            (metadata, cleaned_text): Metadata + Text ohne Header-Zeile
        """
        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        
        metadata = ChunkMetadata()
        
        # Check for chapter
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
                
                cleaned_text = '\n'.join(lines[1:]).strip()
                return metadata, cleaned_text
        
        # Check for section
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
        
        # Check for subsection
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
        
        # No header - use current context
        metadata.chapter = self.current_chapter
        metadata.section = self.current_section
        metadata.subsection = self.current_subsection
        metadata.heading_level = 0
        metadata.is_header = False
        
        return metadata, text


class SemanticBoundaryDetector:
    """
    Erkennt semantische Grenzen in Text - Language-Agnostic.
    
    Nutzt universelle Strukturmarker statt Sprach-spezifische Wörter.
    """
    
    # Universal boundary markers (funktionieren sprachübergreifend)
    BOUNDARY_PATTERNS = [
        r'\n\n+',                           # Double newline (universal paragraph break)
        r'\.\s*\n(?=[A-ZÄÖÜ\u00C0-\u017F])', # Sentence end + newline + capital letter
        r'[.!?]\s*\n\s*\n',                 # Sentence end + blank line
    ]
    
    def find_semantic_boundaries(self, text: str, max_chunk_size: int = 1024) -> List[int]:
        """
        Finde semantische Chunk-Grenzen (language-independent).
        
        Args:
            text: Volltext
            max_chunk_size: Maximale Chunk-Größe
            
        Returns:
            Liste von Boundary-Positionen
        """
        boundaries = [0]
        potential_boundaries = []
        
        # Find all potential boundaries
        for pattern in self.BOUNDARY_PATTERNS:
            for match in re.finditer(pattern, text):
                pos = match.end()
                if pos >= 200:  # Minimum chunk size
                    potential_boundaries.append(pos)
        
        # Sort and deduplicate
        potential_boundaries = sorted(set(potential_boundaries))
        
        # Select boundaries respecting max_chunk_size
        current_pos = 0
        for boundary in potential_boundaries:
            if boundary - current_pos >= max_chunk_size * 0.8:
                boundaries.append(boundary)
                current_pos = boundary
        
        boundaries.append(len(text))
        
        return boundaries


class AutomaticQualityFilter:
    """
    VOLLAUTOMATISCHER Quality Filter - KEINE hardcoded Keywords!
    
    Nutzt statistische Maße:
    1. Lexical Diversity (Type-Token Ratio)
    2. Information Density (Entropie)
    3. Structural Patterns (Transcript Detection)
    4. Length Statistics
    
    Funktioniert für JEDE Sprache ohne Anpassung!
    """
    
    def __init__(
        self,
        min_length: int = 100,
        min_words: int = 15,
        min_lexical_diversity: float = 0.3,  # Statt keyword ratio!
        min_information_density: float = 2.0, # Bits pro Wort
    ):
        """
        Initialisiere automatischen Filter.
        
        Args:
            min_length: Minimum Zeichen
            min_words: Minimum Wörter
            min_lexical_diversity: Minimum Type-Token Ratio (0-1)
            min_information_density: Minimum Shannon-Entropie pro Wort
        """
        self.min_length = min_length
        self.min_words = min_words
        self.min_lexical_diversity = min_lexical_diversity
        self.min_information_density = min_information_density
    
    def calculate_lexical_diversity(self, text: str) -> float:
        """
        Berechne Lexical Diversity (Type-Token Ratio).
        
        Type-Token Ratio = unique words / total words
        
        Hohe Ratio = viele verschiedene Wörter = informativ
        Niedrige Ratio = viele Wiederholungen = Smalltalk/Transcripts
        
        Returns:
            Float 0-1 (höher = diverser = besser)
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def calculate_information_density(self, text: str) -> float:
        """
        Berechne Information Density via Shannon Entropy.
        
        Entropy = -Σ p(word) * log2(p(word))
        
        Hohe Entropy = viele verschiedene Wörter = informativ
        Niedrige Entropy = repetitiv = Smalltalk
        
        Returns:
            Float (bits pro Wort, typisch 2-8)
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        # Word frequency distribution
        word_counts = Counter(words)
        total_words = len(words)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def detect_transcript_pattern(self, text: str) -> bool:
        """
        Erkenne Interview-Transkripte durch statistische Muster.
        
        KEINE hardcoded keywords wie "I:" oder "B4:"!
        Stattdessen: Erkenne Muster wie "X: text Y: text"
        
        Returns:
            True wenn Transkript erkannt
        """
        # Pattern: Short token followed by colon and text
        # Beispiel: "I: text" oder "B4: text" oder "A: text"
        pattern = r'(?:^|\n)\s*\w{1,3}\s*:\s*.{10,}'
        
        matches = re.findall(pattern, text)
        
        # Wenn >30% der Zeilen diesem Muster folgen → Transkript
        lines = text.split('\n')
        if len(lines) > 0:
            match_ratio = len(matches) / len(lines)
            return match_ratio > 0.3
        
        return False
    
    def detect_excessive_whitespace(self, text: str) -> bool:
        """
        Erkenne Layout-Artefakte (Tabellen, Formatierung).
        
        Zu viel Whitespace = Layout-Problem, kein Content.
        """
        if not text:
            return True
        
        whitespace_ratio = text.count(' ') / len(text)
        return whitespace_ratio > 0.4
    
    def should_keep_chunk(self, text: str) -> Tuple[bool, str]:
        """
        AUTOMATISCHE Entscheidung ohne Keywords!
        
        Returns:
            (keep: bool, reason: str)
        """
        # 1. Length check
        if len(text) < self.min_length:
            return False, f"too_short ({len(text)} chars)"
        
        # 2. Word count check
        words = re.findall(r'\b\w+\b', text)
        if len(words) < self.min_words:
            return False, f"too_few_words ({len(words)} words)"
        
        # 3. Transcript detection (automatic!)
        if self.detect_transcript_pattern(text):
            return False, "transcript_pattern_detected"
        
        # 4. Lexical Diversity check (automatic!)
        diversity = self.calculate_lexical_diversity(text)
        if diversity < self.min_lexical_diversity:
            return False, f"low_lexical_diversity ({diversity:.2f})"
        
        # 5. Information Density check (automatic!)
        density = self.calculate_information_density(text)
        if density < self.min_information_density:
            return False, f"low_information_density ({density:.2f} bits/word)"
        
        # 6. Whitespace check
        if self.detect_excessive_whitespace(text):
            return False, "layout_artifact"
        
        return True, "passed"


class TFIDFScorer:
    """
    Berechnet TF-IDF Scores für Chunks AUTOMATISCH.
    
    Identifiziert wichtige Chunks ohne manuelle Keywords!
    """
    
    def __init__(self):
        self.document_frequency = {}  # term → number of chunks containing it
        self.total_chunks = 0
        self.chunk_term_frequencies = []  # List of {term: frequency} dicts
    
    def analyze_corpus(self, chunks: List[str]) -> None:
        """
        Analysiere gesamtes Corpus für TF-IDF.
        
        Args:
            chunks: Alle Text-Chunks
        """
        self.total_chunks = len(chunks)
        self.chunk_term_frequencies = []
        
        # Calculate term frequencies per chunk
        for chunk in chunks:
            words = re.findall(r'\b\w+\b', chunk.lower())
            term_freq = Counter(words)
            self.chunk_term_frequencies.append(term_freq)
            
            # Update document frequency
            for term in set(words):
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
    
    def calculate_chunk_importance(self, chunk_index: int) -> float:
        """
        Berechne Importance Score für einen Chunk via TF-IDF.
        
        Args:
            chunk_index: Index des Chunks
            
        Returns:
            Importance score (höher = wichtiger)
        """
        if chunk_index >= len(self.chunk_term_frequencies):
            return 0.0
        
        term_freq = self.chunk_term_frequencies[chunk_index]
        
        # Calculate TF-IDF score for chunk
        tfidf_score = 0.0
        
        for term, tf in term_freq.items():
            # TF: Term frequency in chunk
            # IDF: log(total_chunks / chunks_containing_term)
            df = self.document_frequency.get(term, 1)
            idf = math.log(self.total_chunks / df) if df > 0 else 0
            
            tfidf_score += tf * idf
        
        # Normalize by chunk length
        total_terms = sum(term_freq.values())
        return tfidf_score / total_terms if total_terms > 0 else 0.0


class SemanticChunker:
    """
    VOLLAUTOMATISCHER Semantic Chunker.
    
    KEINE manuellen Keywords!
    KEINE Sprachfestlegung!
    NUR statistische und strukturelle Analyse!
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 200,
        overlap: int = 128,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        
        self.header_extractor = HeaderExtractor()
        self.boundary_detector = SemanticBoundaryDetector()
        self.quality_filter = AutomaticQualityFilter()
        self.tfidf_scorer = TFIDFScorer()
        
        # Fallback
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunke Dokument vollautomatisch.
        
        Args:
            document: LangChain Document (full page)
            
        Returns:
            Liste von semantisch gechunkten Documents mit Metadaten
        """
        text = document.page_content
        base_metadata = document.metadata.copy()
        
        # Try semantic chunking
        try:
            chunks = self._semantic_chunk(text)
        except Exception as e:
            logger.warning(f"Semantic chunking failed, using fallback: {e}")
            return self.fallback_splitter.split_documents([document])
        
        # FIRST PASS: Analyze corpus for TF-IDF
        self.tfidf_scorer.analyze_corpus(chunks)
        
        # SECOND PASS: Process chunks with all metadata
        processed_chunks = []
        filter_stats = {'kept': 0, 'filtered': 0, 'reasons': {}}
        
        for i, chunk_text in enumerate(chunks):
            # Extract header metadata
            metadata, cleaned_text = self.header_extractor.extract_headers(chunk_text)
            
            # AUTOMATIC quality filter (no keywords!)
            keep, reason = self.quality_filter.should_keep_chunk(cleaned_text)
            
            if not keep:
                filter_stats['filtered'] += 1
                filter_stats['reasons'][reason] = filter_stats['reasons'].get(reason, 0) + 1
                logger.debug(f"Filtered chunk {i}: {reason}")
                continue
            
            filter_stats['kept'] += 1
            
            # Calculate AUTOMATIC importance score
            importance_score = self.tfidf_scorer.calculate_chunk_importance(i)
            lexical_diversity = self.quality_filter.calculate_lexical_diversity(cleaned_text)
            
            # Create enriched document
            enriched_metadata = base_metadata.copy()
            enriched_metadata.update({
                'chunk_id': len(processed_chunks),
                'chunk_size': len(cleaned_text),
                'chapter': metadata.chapter,
                'section': metadata.section,
                'subsection': metadata.subsection,
                'heading_level': metadata.heading_level,
                'is_header': metadata.is_header,
                'chunking_method': 'semantic_automatic',
                'importance_score': round(importance_score, 4),  # TF-IDF based!
                'lexical_diversity': round(lexical_diversity, 4),  # Statistical!
            })
            
            chunk_doc = Document(
                page_content=cleaned_text,
                metadata=enriched_metadata
            )
            
            processed_chunks.append(chunk_doc)
        
        # Log statistics
        logger.info(
            f"Automatic semantic chunking: {len(chunks)} raw → {filter_stats['kept']} kept "
            f"(filtered {filter_stats['filtered']}: {filter_stats['reasons']})"
        )
        
        return processed_chunks
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Chunke Text basierend auf semantischen Grenzen.
        
        Args:
            text: Volltext
            
        Returns:
            Liste von Chunk-Texten
        """
        boundaries = self.boundary_detector.find_semantic_boundaries(
            text, 
            self.max_chunk_size
        )
        
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Add overlap from previous chunk
            if i > 0 and start >= self.overlap:
                start -= self.overlap
            
            chunk_text = text[start:end].strip()
            
            # Skip empty chunks
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            chunks.append(chunk_text)
        
        return chunks


def create_semantic_chunker(
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    min_chunk_size: int = 200,
) -> SemanticChunker:
    """
    Factory für vollautomatischen SemanticChunker.
    
    KEINE manuellen Keywords!
    KEINE Sprachfestlegung!
    
    Args:
        chunk_size: Max Chunk-Größe
        chunk_overlap: Overlap zwischen Chunks
        min_chunk_size: Min Chunk-Größe
        
    Returns:
        Konfigurierter SemanticChunker
    """
    return SemanticChunker(
        max_chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        overlap=chunk_overlap,
    )


def demonstrate_automatic_chunking():
    """Demonstriere AUTOMATIC Chunking ohne Keywords."""
    
    # Example texts (German + English mixed!)
    example_text = """
1. Introduction

In this thesis, we investigate the concept of knowledge management.

1.1 Problem Statement

The central research question is: How can knowledge be effectively 
transferred between generations?

23 I.: Are there situations where you specifically pass on knowledge?

24 B4: Yes, sure, through the group book, yes.

1.2 Theoretical Framework

A Community of Practice refers to a group of people who share a common 
interest or practice. The concept was developed by Wenger (1998) and 
describes informal learning communities. Organizations benefit from 
facilitating such communities through strategic knowledge management 
initiatives and collaborative platforms.

In summary, knowledge management is a complex topic requiring systematic 
approaches to organizational learning and development.

2. Methodology

The research is based on qualitative interviews with experts from various 
organizations. Analysis was conducted using thematic coding methods.
"""
    
    doc = Document(
        page_content=example_text,
        metadata={'source_file': 'example.pdf', 'page': 1}
    )
    
    # Chunk it (AUTOMATIC!)
    chunker = create_semantic_chunker()
    chunks = chunker.chunk_document(doc)
    
    print("\n" + "="*70)
    print("AUTOMATIC SEMANTIC CHUNKING (NO KEYWORDS!)")
    print("="*70)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Chapter: {chunk.metadata.get('chapter', 'N/A')}")
        print(f"  Section: {chunk.metadata.get('section', 'N/A')}")
        print(f"  Size: {chunk.metadata['chunk_size']} chars")
        print(f"  Importance (TF-IDF): {chunk.metadata.get('importance_score', 0):.3f}")
        print(f"  Lexical Diversity: {chunk.metadata.get('lexical_diversity', 0):.3f}")
        print(f"  Text: {chunk.page_content[:100]}...")
    
    print("\n" + "="*70)
    print("Filtering was FULLY AUTOMATIC:")
    print("- NO hardcoded keywords")
    print("- Statistical measures: lexical diversity, information density")
    print("- Pattern detection: transcripts, layout artifacts")
    print("- Works for ANY language!")
    print("="*70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_automatic_chunking()