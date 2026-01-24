"""
Document Ingestion Pipeline: PDF Processing and Text Chunking

Version: 2.1.0
Author: Edge-RAG Research Project
Last Modified: 2026-01-13

===============================================================================
OVERVIEW
===============================================================================

This module implements the document ingestion pipeline for the RAG system.
It handles the transformation of raw PDF documents into indexed, searchable
text chunks suitable for vector embedding and retrieval.

Pipeline Stages:
    1. PDF Loading: Extract text from PDF files
    2. Text Cleaning: Remove artifacts and normalize formatting
    3. Chunking: Split text into semantic units
    4. Metadata Enrichment: Add tracking information

===============================================================================
SCIENTIFIC FOUNDATION
===============================================================================

CHUNKING STRATEGIES:

The choice of chunking strategy significantly impacts retrieval quality.
This implementation supports multiple approaches:

1. Fixed-Size Chunking:
   - Split text at fixed character/token boundaries
   - Simple and predictable
   - May break semantic units mid-sentence
   
2. Recursive Character Splitting (Default):
   - Hierarchical splitting using separator priority
   - Attempts to preserve paragraph and sentence boundaries
   - Good balance of simplicity and semantic coherence
   
   Reference: LangChain RecursiveCharacterTextSplitter

3. Semantic Chunking (Optional):
   - Uses embedding similarity to find semantic boundaries
   - Better preservation of semantic units
   - Higher computational cost
   
   Reference: see semantic_chunking.py

CHUNK SIZE CONSIDERATIONS:

Chunk size affects both retrieval precision and recall:

- Too Small (< 256 chars):
  - High precision but low context
  - May miss relevant information split across chunks
  - More chunks to index and search
  
- Too Large (> 2048 chars):
  - More context but lower precision
  - May include irrelevant information
  - Approaches embedding model's context limit
  
- Optimal Range (512-1024 chars):
  - Balanced precision and context
  - Typically 1-3 paragraphs
  - Recommended for most use cases

OVERLAP RATIONALE:

Chunk overlap ensures information at boundaries is not lost:

    Chunk 1: [--------overlap]
    Chunk 2:         [overlap--------]

Typical overlap: 10-25% of chunk size
- Too little: Boundary information lost
- Too much: Storage overhead, duplicate results

Reference: Lewis et al. (2020). "Retrieval-Augmented Generation for 
Knowledge-Intensive NLP Tasks." NeurIPS 2020.

===============================================================================
PDF TEXT EXTRACTION
===============================================================================

PDF Challenges:
    - No standard text encoding
    - Layout information mixed with content
    - Hyphenation across line breaks
    - Headers/footers repeated on each page
    - Tables and figures as text artifacts

Text Cleaning Operations:
    1. Soft hyphen removal (Unicode U+00AD)
    2. Hyphenation rejoining (word-\nbreak -> wordbreak)
    3. URL removal (often not semantically useful)
    4. Page number removal
    5. Whitespace normalization

===============================================================================
EDGE DEVICE OPTIMIZATION
===============================================================================

Memory Efficiency:
    - Page-by-page processing (not loading entire PDF)
    - Generator patterns where possible
    - Explicit garbage collection hints

Error Resilience:
    - Per-page error handling (corrupted pages don't crash pipeline)
    - Graceful degradation with logging
    - Fallback strategies for problematic PDFs

===============================================================================
MODULE STRUCTURE
===============================================================================

Classes:
    ChunkingConfig           - Configuration dataclass
    RobustPDFLoader          - PDF loading with error handling
    DocumentIngestionPipeline - Main pipeline orchestrator

Functions:
    load_ingestion_config()  - Load config from YAML
    clean_pdf_text()         - Text cleaning utilities
    get_processing_stats()   - Retrieve saved statistics

===============================================================================
USAGE
===============================================================================

Basic Usage:
    config = load_ingestion_config(Path("config/settings.yaml"))
    pipeline = DocumentIngestionPipeline(
        chunking_config=config,
        document_path=Path("data/documents"),
    )
    documents = pipeline.process_documents()

With Custom Settings:
    config = ChunkingConfig(
        mode="standard",
        chunk_size=1024,
        chunk_overlap=128,
        enable_pdf_cleaning=True,
    )
    pipeline = DocumentIngestionPipeline(config, document_path)
    documents = pipeline.process_documents()
"""

import logging
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# PDF parsing library
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning(
        "PyPDF2 not available. Install with: pip install pypdf2"
    )

# Optional semantic chunking
try:
    from src.semantic_chunking import SemanticChunker, create_semantic_chunker
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Unicode soft hyphen character (invisible hyphenation point)
SOFT_HYPHEN = "\u00ad"

# Default separators for recursive text splitting
# Ordered by priority: paragraph breaks > line breaks > sentences > words
DEFAULT_SEPARATORS = [
    "\n\n",    # Paragraph breaks (highest priority)
    "\n",      # Line breaks
    ". ",      # Sentence boundaries
    "! ",      # Exclamation sentences
    "? ",      # Question sentences
    "; ",      # Semicolon clauses
    ", ",      # Comma clauses
    " ",       # Word boundaries
    "",        # Character level (last resort)
]


# ============================================================================
# TEXT CLEANING UTILITIES
# ============================================================================

def clean_pdf_text(text: str) -> str:
    """
    Clean PDF-extracted text by removing common artifacts.
    
    PDF EXTRACTION PROBLEMS:
    
    1. Soft Hyphens:
       PDFs often contain invisible soft hyphen characters (U+00AD) that
       indicate where words can be broken for line wrapping. These should
       be removed as they interfere with text matching.
       
    2. Line-Break Hyphenation:
       Words split across lines appear as "hyphen-\nated" and need to be
       rejoined as "hyphenated".
       
    3. URLs:
       URLs in academic documents often add noise without semantic value.
       They are removed to improve embedding quality.
       
    4. Page Numbers:
       Patterns like "39/104" or "Page 23 of 100" are layout artifacts.
       
    5. Whitespace:
       Multiple spaces and irregular line breaks are normalized.
       
    6. Spacing Bugs:
       Some PDF extractors produce "V or..." instead of "Vor..." for
       German text (capital letter followed by space).
    
    CLEANING ORDER:
    
    The operations are ordered to handle dependencies:
    1. Remove soft hyphens (must precede hyphenation fix)
    2. Fix line-break hyphenation
    3. Remove URLs
    4. Remove page numbers
    5. Normalize whitespace (after other operations)
    6. Fix spacing bugs
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text with artifacts removed
        
    Example:
        Input:  "Die Wis-\\nsenschaft unter-\\nsucht..."
        Output: "Die Wissenschaft untersucht..."
    """
    if not text:
        return ""
    
    # 1. Remove soft hyphens (invisible hyphenation markers)
    text = text.replace(SOFT_HYPHEN, "")
    
    # 2. Rejoin hyphenated words split across lines
    # Pattern: word fragment + hyphen + optional whitespace + newline + continuation
    # Example: "infor-\nmation" -> "information"
    text = re.sub(r"-\s*\n\s*", "", text)
    
    # 3. Remove HTTP/HTTPS URLs
    # URLs rarely contribute to semantic understanding
    text = re.sub(r"https?://\S+", "", text)
    
    # 4. Remove page number artifacts
    # Pattern: "39/104" (page X of Y)
    text = re.sub(r"\b\d{1,3}/\d{1,3}\b", "", text)
    # Pattern: "Page 23 of 100" or "Seite 23 von 100"
    text = re.sub(
        r"\b(page|seite)\s+\d+\s+(of|von)\s+\d+\b", 
        "", 
        text, 
        flags=re.IGNORECASE
    )
    
    # 5. Normalize line breaks
    # Single newlines are often layout artifacts; convert to spaces
    # Double newlines (paragraph breaks) are preserved
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    
    # 6. Normalize multiple spaces to single space
    text = re.sub(r" {2,}", " ", text)
    
    # 7. Fix PDF spacing bug for German text
    # Pattern: Capital letter + space + 2+ lowercase letters
    # Example: "V or..." -> "Vor..."
    # This is common in German PDFs with ligature issues
    text = re.sub(
        r"\b([A-Z\u00C0-\u00D6\u00D8-\u00DE])\s+([a-z\u00DF-\u00F6\u00F8-\u00FF]{2,})",
        r"\1\2",
        text
    )
    
    return text.strip()


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text.
    
    This is a rough estimation based on whitespace tokenization.
    For accurate counts, use the actual tokenizer of your embedding model.
    
    Rule of thumb: 1 token ~ 4 characters for English text
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Simple whitespace-based estimation
    words = text.split()
    return len(words)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ChunkingConfig:
    """
    Configuration for document chunking.
    
    This dataclass encapsulates all parameters for the chunking process,
    enabling reproducible experiments and easy configuration via YAML.
    
    Attributes:
        mode: Chunking strategy ("standard" or "semantic")
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks in characters
        min_chunk_size: Minimum chunk size (smaller chunks are merged or dropped)
        separators: List of separator strings for recursive splitting
        enable_pdf_cleaning: Whether to apply PDF text cleaning
    
    PARAMETER GUIDELINES:
    
    chunk_size:
        - 256-512: High precision, low context (short documents)
        - 512-1024: Balanced (recommended default)
        - 1024-2048: High context, lower precision (complex topics)
        
    chunk_overlap:
        - 10% of chunk_size: Minimal overhead
        - 15-20% of chunk_size: Recommended
        - 25%+ of chunk_size: High redundancy
        
    mode:
        - "standard": RecursiveCharacterTextSplitter (fast, reliable)
        - "semantic": SemanticChunker (slower, better boundaries)
    """
    mode: str = "standard"
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    separators: List[str] = field(default_factory=lambda: DEFAULT_SEPARATORS.copy())
    enable_pdf_cleaning: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.mode not in ("standard", "semantic"):
            raise ValueError(
                f"Invalid chunking mode: {self.mode}. "
                f"Must be 'standard' or 'semantic'."
            )
        
        if self.chunk_size < 100:
            raise ValueError(
                f"chunk_size too small: {self.chunk_size}. "
                f"Minimum recommended: 100 characters."
            )
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})."
            )
        
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative: {self.chunk_overlap}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "mode": self.mode,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "separators": self.separators,
            "enable_pdf_cleaning": self.enable_pdf_cleaning,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingConfig":
        """Create configuration from dictionary."""
        return cls(
            mode=data.get("mode", "standard"),
            chunk_size=data.get("chunk_size", 512),
            chunk_overlap=data.get("chunk_overlap", 128),
            min_chunk_size=data.get("min_chunk_size", 100),
            separators=data.get("separators", DEFAULT_SEPARATORS.copy()),
            enable_pdf_cleaning=data.get("enable_pdf_cleaning", True),
        )


# ============================================================================
# PDF LOADER
# ============================================================================

class RobustPDFLoader:
    """
    Production-grade PDF loader with per-page error handling.
    
    DESIGN RATIONALE:
    
    PDF documents can contain problematic pages that fail to extract:
    - Scanned images without OCR
    - Corrupted page data
    - Unsupported encoding
    - Complex vector graphics
    
    A naive implementation would crash on any problematic page,
    losing all successfully extracted content. This implementation
    uses per-page error handling to maximize content extraction.
    
    ERROR HANDLING STRATEGY:
    
    1. Attempt extraction for each page individually
    2. Log warning for failed pages
    3. Continue with remaining pages
    4. Report statistics at completion
    
    This "graceful degradation" approach is essential for production
    systems processing diverse document collections.
    
    USAGE:
    
        loader = RobustPDFLoader(
            file_path=Path("document.pdf"),
            enable_cleaning=True
        )
        documents = loader.load()
        # Returns list of Document objects, one per successfully extracted page
    
    Attributes:
        file_path: Path to PDF file
        enable_cleaning: Whether to apply text cleaning
    """
    
    def __init__(
        self,
        file_path: Path,
        enable_cleaning: bool = True,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize PDF loader.
        
        Args:
            file_path: Path to PDF file
            enable_cleaning: Apply text cleaning after extraction
            logger_instance: Custom logger (uses module logger if None)
            
        Raises:
            FileNotFoundError: If PDF file does not exist
            ImportError: If PyPDF2 is not available
        """
        self.file_path = Path(file_path)
        self.enable_cleaning = enable_cleaning
        self.logger = logger_instance or logger
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not PYPDF2_AVAILABLE:
            raise ImportError(
                "PyPDF2 is required for PDF loading. "
                "Install with: pip install pypdf2"
            )
    
    def load(self) -> List[Document]:
        """
        Load and extract text from PDF.
        
        ALGORITHM:
        
        1. Open PDF with PyPDF2 reader
        2. Iterate through pages
        3. For each page:
           a. Attempt text extraction
           b. Apply cleaning if enabled
           c. Skip empty pages
           d. Create Document with metadata
        4. Log summary statistics
        5. Return list of Documents
        
        Returns:
            List of LangChain Document objects, one per successfully 
            extracted page. Empty pages are not included.
            
        Raises:
            Exception: Only if PDF is completely unreadable
        """
        self.logger.info(f"Loading PDF: {self.file_path.name}")
        start_time = time.time()
        
        try:
            reader = PdfReader(str(self.file_path))
            total_pages = len(reader.pages)
            
            documents = []
            failed_pages = []
            empty_pages = []
            
            for page_num in range(total_pages):
                try:
                    # Extract text from page
                    page = reader.pages[page_num]
                    text = page.extract_text() or ""
                    
                    # Apply cleaning if enabled
                    if self.enable_cleaning:
                        text = clean_pdf_text(text)
                    
                    # Skip pages with insufficient content
                    if len(text.strip()) < 50:
                        empty_pages.append(page_num + 1)
                        continue
                    
                    # Create Document with comprehensive metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(self.file_path),
                            "source_file": self.file_path.name,
                            "page": page_num + 1,
                            "page_number": page_num + 1,
                            "total_pages": total_pages,
                            "char_count": len(text),
                            "extraction_method": "pypdf2",
                            "cleaned": self.enable_cleaning,
                        }
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    # Log error but continue with other pages
                    self.logger.warning(
                        f"Page {page_num + 1} extraction failed: {str(e)}"
                    )
                    failed_pages.append(page_num + 1)
            
            # Log summary
            elapsed = time.time() - start_time
            total_chars = sum(len(doc.page_content) for doc in documents)
            
            self.logger.info(
                f"PDF loaded: {self.file_path.name} | "
                f"{len(documents)}/{total_pages} pages extracted | "
                f"{total_chars:,} characters | "
                f"{elapsed:.2f}s"
            )
            
            if failed_pages:
                self.logger.warning(
                    f"Failed pages: {failed_pages[:10]}"
                    f"{'...' if len(failed_pages) > 10 else ''}"
                )
            
            if empty_pages and len(empty_pages) <= 5:
                self.logger.debug(f"Empty pages skipped: {empty_pages}")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"PDF loading failed completely: {str(e)}")
            raise


# ============================================================================
# DOCUMENT INGESTION PIPELINE
# ============================================================================

class DocumentIngestionPipeline:
    """
    Complete document ingestion pipeline for RAG systems.
    
    PIPELINE STAGES:
    
    1. Document Discovery:
       - Scan directory for PDF files
       - Filter by file pattern
    
    2. PDF Loading:
       - Extract text from each PDF
       - Apply text cleaning
       - Handle extraction errors gracefully
    
    3. Text Chunking:
       - Split documents into smaller units
       - Apply configured chunking strategy
       - Enrich with metadata
    
    4. Statistics Collection:
       - Track processing metrics
       - Save for reproducibility
    
    CHUNKING STRATEGIES:
    
    Standard (RecursiveCharacterTextSplitter):
        - Uses hierarchical separator list
        - Fast and deterministic
        - Good for most use cases
        
    Semantic (SemanticChunker):
        - Uses embedding similarity
        - Better boundary detection
        - Slower, requires embedding model
    
    USAGE EXAMPLE:
    
        # Load configuration
        config = load_ingestion_config(Path("config/settings.yaml"))
        
        # Create pipeline
        pipeline = DocumentIngestionPipeline(
            chunking_config=config,
            document_path=Path("data/documents"),
            enable_cleaning=True,
        )
        
        # Process documents
        chunks = pipeline.process_documents()
        
        # chunks is a list of LangChain Document objects ready for embedding
    
    THESIS DOCUMENTATION:
    
    For reproducibility, the pipeline saves processing metadata including:
    - Configuration parameters
    - Processing statistics
    - Timing information
    
    Retrieve with: get_processing_stats()
    
    Attributes:
        chunking_config: ChunkingConfig instance
        document_path: Path to document directory
        enable_cleaning: Whether to clean PDF text
        text_splitter: Initialized text splitter (standard mode)
        semantic_chunker: Initialized semantic chunker (semantic mode)
        stats: Processing statistics dictionary
    """
    
    def __init__(
        self,
        chunking_config: ChunkingConfig,
        document_path: Path,
        enable_cleaning: bool = True,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize document ingestion pipeline.

        Args:
            chunking_config: Chunking configuration
            document_path: Path to directory containing PDF files
            enable_cleaning: Apply PDF text cleaning
            logger_instance: Custom logger instance
            
        Raises:
            FileNotFoundError: If document_path does not exist
        """
        self.chunking_config = chunking_config
        self.document_path = Path(document_path)
        self.enable_cleaning = enable_cleaning
        self.logger = logger_instance or logger
        
        # Validate document path
        if not self.document_path.exists():
            raise FileNotFoundError(
                f"Document directory not found: {document_path}"
            )
        
        # Initialize appropriate chunker based on mode
        self.text_splitter = None
        self.semantic_chunker = None
        
        if chunking_config.mode == "semantic":
            if SEMANTIC_CHUNKING_AVAILABLE:
                self.logger.info("Initializing semantic chunking mode")
                self.semantic_chunker = create_semantic_chunker(
                    chunk_size=chunking_config.chunk_size,
                    chunk_overlap=chunking_config.chunk_overlap,
                    min_chunk_size=chunking_config.min_chunk_size,
                )
            else:
                self.logger.warning(
                    "Semantic chunking not available, falling back to standard"
                )
                chunking_config.mode = "standard"
        
        if chunking_config.mode == "standard":
            self.logger.info("Initializing standard chunking mode")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap,
                separators=chunking_config.separators,
                length_function=len,
                is_separator_regex=False,
            )
        
        # Initialize statistics tracking
        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "total_characters": 0,
            "processing_time_seconds": 0.0,
        }
        
        self.logger.info(
            f"DocumentIngestionPipeline initialized: "
            f"mode={chunking_config.mode}, "
            f"chunk_size={chunking_config.chunk_size}, "
            f"overlap={chunking_config.chunk_overlap}"
        )

    def discover_documents(self, pattern: str = "*.pdf") -> List[Path]:
        """
        Discover PDF files in document directory.
        
        Args:
            pattern: Glob pattern for file matching
            
        Returns:
            List of Path objects for discovered files
        """
        pdf_files = list(self.document_path.glob(pattern))
        
        if not pdf_files:
            self.logger.warning(
                f"No files matching '{pattern}' found in {self.document_path}"
            )
        else:
            self.logger.info(f"Discovered {len(pdf_files)} PDF files")
        
        return pdf_files

    def load_documents(self, pdf_files: List[Path]) -> List[Document]:
        """
        Load and extract text from PDF files.
        
        Uses RobustPDFLoader for per-page error handling.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of Document objects (one per page)
        """
        all_documents = []
        
        for pdf_path in pdf_files:
            try:
                loader = RobustPDFLoader(
                    file_path=pdf_path,
                    enable_cleaning=self.enable_cleaning,
                    logger_instance=self.logger,
                )
                
                documents = loader.load()
                all_documents.extend(documents)
                
                self.stats["files_processed"] += 1
                self.stats["total_pages"] += len(documents)
                
            except Exception as e:
                self.logger.error(
                    f"Failed to load PDF {pdf_path.name}: {str(e)}"
                )
                self.stats["files_failed"] += 1
        
        return all_documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Applies the configured chunking strategy (standard or semantic).
        Enriches chunks with metadata including chunk_id and position.
        
        Args:
            documents: List of Document objects (typically one per page)
            
        Returns:
            List of chunked Document objects
        """
        start_time = time.time()
        
        if self.chunking_config.mode == "semantic" and self.semantic_chunker:
            chunks = self._chunk_semantic(documents)
        else:
            chunks = self._chunk_standard(documents)
        
        # Enrich metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunking_mode"] = self.chunking_config.mode
        
        elapsed = time.time() - start_time
        
        # Update statistics
        self.stats["total_chunks"] = len(chunks)
        self.stats["total_characters"] = sum(
            len(c.page_content) for c in chunks
        )
        
        avg_chunk_size = (
            self.stats["total_characters"] / len(chunks) 
            if chunks else 0
        )
        
        self.logger.info(
            f"Chunking complete: "
            f"{len(documents)} pages -> {len(chunks)} chunks | "
            f"avg_size={avg_chunk_size:.0f} chars | "
            f"time={elapsed:.2f}s"
        )
        
        return chunks

    def _chunk_standard(self, documents: List[Document]) -> List[Document]:
        """
        Apply standard recursive character splitting.
        
        ALGORITHM (RecursiveCharacterTextSplitter):
        
        1. Try to split on highest-priority separator (e.g., "\n\n")
        2. If resulting chunks are too large, recurse with next separator
        3. Continue until chunks are within size limit
        4. Add overlap by including text from adjacent chunks
        
        This approach preserves semantic boundaries where possible
        while guaranteeing maximum chunk size.
        
        Args:
            documents: Input documents
            
        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)
        
        self.logger.debug(
            f"Standard chunking: {len(documents)} docs -> {len(chunks)} chunks"
        )
        
        return chunks

    def _chunk_semantic(self, documents: List[Document]) -> List[Document]:
        """
        Apply semantic chunking using embedding similarity.
        
        See semantic_chunking.py for implementation details.
        Falls back to standard chunking on error.
        
        Args:
            documents: Input documents
            
        Returns:
            List of chunked documents
        """
        all_chunks = []
        
        for doc in documents:
            try:
                doc_chunks = self.semantic_chunker.chunk_document(doc)
                all_chunks.extend(doc_chunks)
            except Exception as e:
                self.logger.warning(
                    f"Semantic chunking failed for page, using fallback: {e}"
                )
                # Fallback to standard chunking for this document
                fallback_chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(fallback_chunks)
        
        self.logger.debug(
            f"Semantic chunking: {len(documents)} docs -> {len(all_chunks)} chunks"
        )
        
        return all_chunks

    def process_documents(self, pattern: str = "*.pdf") -> List[Document]:
        """
        Execute complete ingestion pipeline.
        
        PIPELINE STAGES:
        1. Discover PDF files
        2. Load and extract text
        3. Chunk documents
        4. Save processing metadata
        
        Args:
            pattern: Glob pattern for file discovery
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        pipeline_start = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("DOCUMENT INGESTION PIPELINE - START")
        self.logger.info("=" * 70)
        
        # Stage 1: Discover documents
        pdf_files = self.discover_documents(pattern)
        
        if not pdf_files:
            self.logger.warning("No documents to process")
            return []
        
        # Stage 2: Load documents
        documents = self.load_documents(pdf_files)
        
        if not documents:
            self.logger.warning("No content extracted from documents")
            return []
        
        # Stage 3: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Stage 4: Save metadata
        pipeline_time = time.time() - pipeline_start
        self.stats["processing_time_seconds"] = pipeline_time
        self._save_processing_metadata()
        
        self.logger.info("=" * 70)
        self.logger.info("DOCUMENT INGESTION PIPELINE - COMPLETE")
        self.logger.info(f"Total time: {pipeline_time:.2f}s")
        self.logger.info(f"Output: {len(chunks)} chunks ready for embedding")
        self.logger.info("=" * 70)
        
        return chunks

    def _save_processing_metadata(self) -> None:
        """
        Save processing metadata for reproducibility.
        
        Saves to data/ingestion_metadata.json with:
        - Timestamp
        - Configuration parameters
        - Processing statistics
        """
        metadata = {
            "timestamp": time.time(),
            "pipeline_version": "2.1.0",
            "chunking_config": self.chunking_config.to_dict(),
            "pdf_cleaning_enabled": self.enable_cleaning,
            "statistics": self.stats.copy(),
        }
        
        metadata_path = Path("data/ingestion_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Processing metadata saved: {metadata_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_ingestion_config(config_path: Path) -> ChunkingConfig:
    """
    Load chunking configuration from YAML file.
    
    Expected YAML structure:
    
        chunking:
          mode: "standard"
          chunk_size: 512
          chunk_overlap: 128
          min_chunk_size: 100
          enable_pdf_cleaning: true
          separators:
            - "\\n\\n"
            - "\\n"
            - ". "
            - " "
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ChunkingConfig instance
        
    Raises:
        FileNotFoundError: If config file does not exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    chunking_cfg = config.get("chunking", {})
    
    if not chunking_cfg:
        logger.warning(
            "No 'chunking' section in config, using defaults"
        )
    
    return ChunkingConfig(
        mode=chunking_cfg.get("mode", "standard"),
        chunk_size=chunking_cfg.get("chunk_size", 512),
        chunk_overlap=chunking_cfg.get("chunk_overlap", 128),
        min_chunk_size=chunking_cfg.get("min_chunk_size", 100),
        separators=chunking_cfg.get("separators", DEFAULT_SEPARATORS.copy()),
        enable_pdf_cleaning=chunking_cfg.get("enable_pdf_cleaning", True),
    )


def get_processing_stats(
    metadata_path: Path = Path("data/ingestion_metadata.json")
) -> Optional[Dict[str, Any]]:
    """
    Load processing statistics from saved metadata.
    
    Useful for:
    - Thesis documentation
    - Debugging processing issues
    - Verifying reproducibility
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Dictionary with processing metadata, or None if not found
    """
    if not metadata_path.exists():
        logger.warning(f"Processing metadata not found: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load processing metadata: {e}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_chunks(chunks: List[Document]) -> Dict[str, Any]:
    """
    Validate chunked documents and compute statistics.
    
    Useful for quality assurance and thesis documentation.
    
    Args:
        chunks: List of chunked documents
        
    Returns:
        Dictionary containing validation results and statistics
    """
    if not chunks:
        return {
            "valid": False,
            "error": "No chunks provided",
            "count": 0,
        }
    
    sizes = [len(c.page_content) for c in chunks]
    
    # Check for empty chunks
    empty_chunks = [i for i, s in enumerate(sizes) if s == 0]
    
    # Check for very small chunks
    small_threshold = 50
    small_chunks = [i for i, s in enumerate(sizes) if 0 < s < small_threshold]
    
    # Compute statistics
    import statistics
    
    result = {
        "valid": len(empty_chunks) == 0,
        "count": len(chunks),
        "size_min": min(sizes),
        "size_max": max(sizes),
        "size_mean": statistics.mean(sizes),
        "size_median": statistics.median(sizes),
        "size_stdev": statistics.stdev(sizes) if len(sizes) > 1 else 0,
        "empty_chunks": empty_chunks,
        "small_chunks": small_chunks,
    }
    
    if empty_chunks:
        result["error"] = f"Found {len(empty_chunks)} empty chunks"
    
    return result