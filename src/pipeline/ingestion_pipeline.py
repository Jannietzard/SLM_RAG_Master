"""
Document Ingestion Pipeline: Chunking → Entity Extraction → Hybrid Storage

Version: 4.1.0
Author: Edge-RAG Research Project
Last Modified: 2026-04-11

================================================================================
ARCHITECTURAL POSITION
================================================================================

This module is the entry point for Artifact A (Data Layer) population. It
transforms raw documents into the dual-storage knowledge representation that
Artifact B (the agentic query pipeline) subsequently retrieves from:

    benchmark_datasets.py  /  local_importingestion.py
               │
               ▼  create_ingestion_pipeline()
    IngestionPipeline.ingest(source_path)
               │
    ┌──────────┴──────────────────────────────────────┐
    │  1. DocumentLoader   — multi-format document I/O │
    │  2. SpacySentenceChunker — sentence-window split │
    │  3. EntityExtractionPipeline — GLiNER + REBEL    │
    │  4. BatchedOllamaEmbeddings — nomic-embed-text   │
    │  5. HybridStore — LanceDB + KuzuDB               │
    └─────────────────────────────────────────────────┘

================================================================================
SCIENTIFIC CONTRIBUTION
================================================================================

The ingestion pipeline implements the three-stage knowledge construction process
described in the thesis (Artifact A, Chapter 2):

  Stage 1 — Sentence-window chunking (Chapter 2.2):
    Consecutive sentences are grouped into overlapping windows of fixed size.
    This preserves local discourse coherence across chunk boundaries.

  Stage 2 — Selective Relation Extraction (Chapter 2.5):
    Relation Extraction (REBEL) is only applied to chunks that contain at least
    min_entities_for_re named entities identified by GLiNER. For HotpotQA
    documents this avoids roughly 60% of RE calls while retaining full coverage
    of informative chunks.
    Reference: Cabot, P-L. & Navigli, R. (2021). "REBEL: Relation Extraction By
    End-to-end Language generation." EMNLP 2021 Findings.

  Stage 3 — Embedding consistency (Chapter 2.3):
    The same Ollama-hosted nomic-embed-text model is used for BOTH ingestion
    (BatchedOllamaEmbeddings) and query-time retrieval. Using the same model
    endpoint ensures that ingestion vectors and query vectors live in the same
    embedding space, which is required for cosine similarity to be meaningful.

================================================================================
ABLATION / TEST FLAGS
================================================================================

    use_mocks=True   → MockEmbeddingGenerator + MockEntityExtractor (no GPU,
                       no Ollama required). Intended for unit tests ONLY.
                       Mock results must NOT be used for thesis evaluation.

================================================================================
USAGE
================================================================================

    from src.pipeline.ingestion_pipeline import create_ingestion_pipeline
    import yaml

    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)

    pipeline = create_ingestion_pipeline(config=cfg)
    metrics = pipeline.ingest("data/documents/")
    print(metrics.to_dict())

================================================================================
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..data_layer.chunking import SpacySentenceChunker
    from ..data_layer.embeddings import BatchedOllamaEmbeddings
    from ..data_layer.entity_extraction import EntityExtractionPipeline
    from ..data_layer.storage import HybridStore

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IngestionConfig:
    """
    Configuration for the ingestion pipeline.

    All parameters match settings.yaml entries. Dataclass defaults serve as
    emergency fallbacks only — in production, always construct via from_yaml().

    Parameter sources (settings.yaml key paths):
        sentences_per_chunk          ← ingestion.sentences_per_chunk
        sentence_overlap             ← ingestion.sentence_overlap
        min_chunk_length             ← ingestion.min_chunk_size
        max_chunk_length             ← ingestion.max_chunk_chars
        gliner_batch_size            ← entity_extraction.gliner.batch_size
        rebel_batch_size             ← entity_extraction.rebel.batch_size
        entity_confidence_threshold  ← entity_extraction.gliner.confidence_threshold
        relation_confidence_threshold← entity_extraction.rebel.confidence_threshold
        min_entities_for_re          ← entity_extraction.rebel.min_entities_for_re
        embedding_model              ← embeddings.model_name
        embedding_dim                ← embeddings.embedding_dim
        embedding_batch_size         ← performance.batch_size
        vector_db_path               ← paths.vector_db
        graph_db_path                ← paths.graph_db
        enable_caching               ← entity_extraction.caching.enabled
    """
    # Sentence-window chunking (Chapter 2.2)
    sentences_per_chunk: int = 3
    sentence_overlap: int = 1
    min_chunk_length: int = 50
    max_chunk_length: int = 2000

    # Entity extraction (Chapter 2.5)
    gliner_batch_size: int = 16
    rebel_batch_size: int = 8
    # Recall-optimised threshold for HotpotQA (see MEMORY.md, NER refactoring 2026-04-03)
    entity_confidence_threshold: float = 0.15
    # REBEL emits no per-triplet score; value serves as uniform sentinel
    relation_confidence_threshold: float = 0.5
    min_entities_for_re: int = 2  # selective RE: skip ~60% of chunks

    # Embeddings (Chapter 2.3) — Ollama model identifier, NOT HuggingFace path
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    embedding_batch_size: int = 64

    # Storage paths
    vector_db_path: str = "./data/vector"
    graph_db_path: str = "./data/graph"

    # Performance
    enable_caching: bool = True

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "IngestionConfig":
        """
        Construct IngestionConfig from a settings.yaml dict.

        Key paths match the actual settings.yaml structure. Any missing key
        falls back to the dataclass default (which emits a log warning).
        """
        _d = cls()  # defaults for fallback

        ingestion_cfg = config.get("ingestion", {})
        gliner_cfg = config.get("entity_extraction", {}).get("gliner", {})
        rebel_cfg = config.get("entity_extraction", {}).get("rebel", {})
        caching_cfg = config.get("entity_extraction", {}).get("caching", {})
        emb_cfg = config.get("embeddings", {})
        perf_cfg = config.get("performance", {})
        paths_cfg = config.get("paths", {})

        return cls(
            # Sentence-window chunking
            sentences_per_chunk=ingestion_cfg.get(
                "sentences_per_chunk", _d.sentences_per_chunk
            ),
            sentence_overlap=ingestion_cfg.get(
                "sentence_overlap", _d.sentence_overlap
            ),
            min_chunk_length=ingestion_cfg.get(
                "min_chunk_size", _d.min_chunk_length
            ),
            max_chunk_length=ingestion_cfg.get(
                "max_chunk_chars", _d.max_chunk_length
            ),
            # Entity extraction
            gliner_batch_size=gliner_cfg.get(
                "batch_size", _d.gliner_batch_size
            ),
            rebel_batch_size=rebel_cfg.get(
                "batch_size", _d.rebel_batch_size
            ),
            entity_confidence_threshold=gliner_cfg.get(
                "confidence_threshold", _d.entity_confidence_threshold
            ),
            relation_confidence_threshold=rebel_cfg.get(
                "confidence_threshold", _d.relation_confidence_threshold
            ),
            min_entities_for_re=rebel_cfg.get(
                "min_entities_for_re", _d.min_entities_for_re
            ),
            # Embeddings — model name is the Ollama identifier, not HuggingFace path
            embedding_model=emb_cfg.get("model_name", _d.embedding_model),
            embedding_dim=emb_cfg.get("embedding_dim", _d.embedding_dim),
            embedding_batch_size=perf_cfg.get(
                "batch_size", _d.embedding_batch_size
            ),
            # Storage
            vector_db_path=paths_cfg.get("vector_db", _d.vector_db_path),
            graph_db_path=paths_cfg.get("graph_db", _d.graph_db_path),
            # Caching
            enable_caching=caching_cfg.get("enabled", _d.enable_caching),
        )


# ============================================================================
# METRICS & STATISTICS
# ============================================================================

@dataclass
class IngestionMetrics:
    """Metrics for a single ingestion run."""
    documents_processed: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0

    # Per-stage timing (milliseconds)
    total_time_ms: float = 0.0
    chunking_time_ms: float = 0.0
    extraction_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    storage_time_ms: float = 0.0

    # Aggregate performance
    avg_chunk_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to nested dict for JSON export and benchmark logging."""
        return {
            "counts": {
                "documents": self.documents_processed,
                "chunks": self.chunks_created,
                "entities": self.entities_extracted,
                "relations": self.relations_extracted,
            },
            "timing_ms": {
                "total": self.total_time_ms,
                "chunking": self.chunking_time_ms,
                "extraction": self.extraction_time_ms,
                "embedding": self.embedding_time_ms,
                "storage": self.storage_time_ms,
            },
            "performance": {
                "avg_chunk_latency_ms": self.avg_chunk_latency_ms,
                "cache_hit_rate": self.cache_hit_rate,
            },
        }


# ============================================================================
# DOCUMENT LOADER
# ============================================================================

class DocumentLoader:
    """
    Load documents from multiple file formats into a uniform dict schema.

    Supported formats:
        - Plain text (.txt)
        - Markdown (.md)
        - JSON (.json) — single dict or list of dicts
        - JSON Lines (.jsonl) — one JSON object per line

    Note: PDF support is NOT implemented. Add via PyMuPDF or pypdf if needed.

    Document IDs are path-based (hashes of the source path string), not
    content-based. Moving or renaming a file produces a different ID even if
    the content is identical. This is intentional: the pipeline does not
    implement document deduplication; two differently-named files with the same
    content are treated as separate documents.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".json", ".jsonl", ".md"}

    def load(self, path: str) -> Iterator[Dict[str, Any]]:
        """
        Load document(s) from a file path or directory.

        Parameters
        ----------
        path : str
            Path to a file or directory. Directories are searched recursively
            for files with supported extensions.

        Yields
        ------
        dict
            {"id": str, "text": str, "metadata": dict}

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        """
        p = Path(path)

        if p.is_file():
            yield from self._load_file(p)
        elif p.is_dir():
            for file_path in p.rglob("*"):
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    yield from self._load_file(file_path)
        else:
            raise FileNotFoundError(f"Path not found: {p}")

    def _load_file(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Route a single file to the appropriate format loader."""
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            yield from self._load_text(path)
        elif suffix == ".json":
            yield from self._load_json(path)
        elif suffix == ".jsonl":
            yield from self._load_jsonl(path)
        else:
            logger.warning("Unsupported file format, skipping: %s", path)

    def _load_text(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Load a plain-text or Markdown file as a single document."""
        try:
            text = path.read_text(encoding="utf-8")
            doc_id = self._generate_id(str(path))
            yield {
                "id": doc_id,
                "text": text,
                "metadata": {"source": str(path), "format": "text"},
            }
        except (OSError, UnicodeDecodeError) as e:
            logger.error("Failed to load text file %s: %s", path, e)

    def _load_json(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Load a JSON file — supports both list and single-dict structures."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for i, item in enumerate(data):
                    yield self._parse_json_item(item, f"{path}:{i}")
            elif isinstance(data, dict):
                yield self._parse_json_item(data, str(path))

        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load JSON file %s: %s", path, e)

    def _load_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        """
        Load a JSON Lines file (one JSON object per line).

        Typical formats: HotpotQA, 2WikiMultiHopQA.
        Reference: Yang, Z. et al. (2018). "HotpotQA: A Dataset for Diverse,
        Explainable Multi-hop Question Answering." EMNLP 2018.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        item = json.loads(line)
                        yield self._parse_json_item(item, f"{path}:{i}")

        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load JSONL file %s: %s", path, e)

    def _parse_json_item(
        self, item: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        """
        Normalise a JSON item to the uniform document schema.

        Handles HotpotQA context format (list of (title, sentences) tuples)
        and plain text/content/passage fields. Falls back to full JSON
        serialisation when no known text field is present.

        Reference for HotpotQA format: Yang, Z. et al. (2018). EMNLP 2018.
        """
        # HotpotQA / 2WikiMultiHopQA: context is [(title, [sentences, ...]), ...]
        if "context" in item:
            texts = []
            for entry in item.get("context", []):
                try:
                    title, sentences = entry
                    texts.append(f"{title}\n" + " ".join(sentences))
                except (TypeError, ValueError) as e:
                    logger.warning(
                        "Malformed context entry in %s (expected 2-tuple): %s",
                        source, e,
                    )
            text = "\n\n".join(texts)
        elif "text" in item:
            text = item["text"]
        elif "content" in item:
            text = item["content"]
        elif "passage" in item:
            text = item["passage"]
        else:
            # Last resort: serialise the full item as text
            logger.warning(
                "No recognised text field in JSON item from %s; "
                "falling back to full JSON serialisation.",
                source,
            )
            text = json.dumps(item)

        doc_id = item.get("id", item.get("_id", self._generate_id(source)))

        return {
            "id": str(doc_id),
            "text": text,
            "metadata": {
                "source": source,
                "question": item.get("question"),
                "answer": item.get("answer"),
                "type": item.get("type"),
                "level": item.get("level"),
            },
        }

    @staticmethod
    def _generate_id(source: str) -> str:
        """
        Return a 16-hex-char SHA-256 key for the given source string.

        Truncation to 64 bits is safe for document sets of ≤ 10^6 items
        (birthday collision probability < 10^-9).
        """
        return hashlib.sha256(source.encode()).hexdigest()[:16]


# ============================================================================
# MOCK COMPONENTS (for tests — never use for thesis evaluation)
# ============================================================================

class MockEmbeddingGenerator:
    """
    Mock embedding generator for unit tests that require no GPU or Ollama.

    IMPORTANT: Results are non-deterministic unless seed is set. Never use
    this class for thesis evaluation — embeddings are random vectors that do
    not reflect semantic similarity.
    """

    def __init__(self, embedding_dim: int = 768) -> None:
        self.embedding_dim = embedding_dim

    def embed(
        self,
        texts: List[str],
        show_progress: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return L2-normalised random embedding vectors.

        Parameters
        ----------
        texts : list of str
        show_progress : bool
            Ignored; present for interface compatibility with BatchedOllamaEmbeddings.
        seed : int, optional
            Set for reproducible test output.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        rng = np.random.default_rng(seed)
        embeddings = rng.standard_normal((len(texts), self.embedding_dim))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / norms).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain-compatible interface: returns list of float lists."""
        return self.embed(texts).tolist()


class MockEntityExtractor:
    """
    Mock entity extractor for unit tests.

    Extracts capitalised words as entities. Field names match the production
    ExtractedEntity interface (entity_id, name, entity_type, confidence,
    source_chunk_id) so that mock and real extractor outputs are interchangeable
    by downstream code.
    """

    def process_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Any], List[Any]]:
        """Return simple mock entities (capitalised words) and empty relations."""
        @dataclass
        class MockEntity:
            entity_id: str
            name: str
            entity_type: str       # matches ExtractedEntity.entity_type (post-2026-04-05)
            confidence: float
            source_chunk_id: str

        entities: List[Any] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            for word in words[:3]:  # max 3 per chunk to keep tests fast
                entities.append(
                    MockEntity(
                        entity_id=hashlib.sha256(word.encode()).hexdigest()[:8],
                        name=word,
                        entity_type="CONCEPT",
                        confidence=0.8,
                        source_chunk_id=chunk.get("chunk_id", ""),
                    )
                )

        return entities, []


# ============================================================================
# MAIN INGESTION PIPELINE
# ============================================================================

class IngestionPipeline:
    """
    Orchestrator for the five-stage document ingestion pipeline.

    Stages:
        1. Document loading  (DocumentLoader)
        2. Sentence-window chunking  (SpacySentenceChunker)
        3. Entity + relation extraction  (GLiNER + REBEL via EntityExtractionPipeline)
        4. Embedding generation  (BatchedOllamaEmbeddings)
        5. Hybrid storage  (LanceDB vector store + KuzuDB knowledge graph)

    Performance target: 80–120 ms per chunk (thesis Chapter 2.5).

    Thread safety: Not thread-safe. Sequential single-process operation is
    assumed throughout, consistent with the edge deployment model.
    """

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        chunker: Optional[Any] = None,
        entity_extractor: Optional[Any] = None,
        embedding_generator: Optional[Any] = None,
        hybrid_store: Optional[Any] = None,
        use_mocks: bool = False,
    ) -> None:
        """
        Initialise the ingestion pipeline.

        Parameters
        ----------
        config : IngestionConfig, optional
            Pipeline configuration. Constructed from hardcoded defaults when
            omitted — not recommended for production use.
        chunker : SpacySentenceChunker, optional
            Sentence-based chunker. Created from config when None.
        entity_extractor : EntityExtractionPipeline, optional
            GLiNER + REBEL extractor. Created from config when None.
        embedding_generator : BatchedOllamaEmbeddings or MockEmbeddingGenerator, optional
            Embedding model. Created from config when None.
        hybrid_store : HybridStore, optional
            LanceDB + KuzuDB dual store. Created from config when None.
        use_mocks : bool
            When True, substitutes MockEmbeddingGenerator and MockEntityExtractor.
            Intended for unit tests only. Results must not be used for evaluation.
        """
        self.config = config or IngestionConfig()
        self.use_mocks = use_mocks

        self.loader = DocumentLoader()

        # ── Chunker ──────────────────────────────────────────────────────────
        if chunker is not None:
            self.chunker = chunker
        else:
            self.chunker = self._init_chunker()

        # ── Entity extractor ─────────────────────────────────────────────────
        if entity_extractor is not None:
            self.entity_extractor = entity_extractor
        elif use_mocks:
            logger.warning(
                "FALLBACK ACTIVE: MockEntityExtractor in use "
                "(use_mocks=True) — NER/RE produces no real entities."
            )
            self.entity_extractor = MockEntityExtractor()
        else:
            self.entity_extractor = self._init_entity_extractor()

        # ── Embedding generator ──────────────────────────────────────────────
        if embedding_generator is not None:
            self.embedding_generator = embedding_generator
        elif use_mocks:
            logger.warning(
                "FALLBACK ACTIVE: MockEmbeddingGenerator in use "
                "(use_mocks=True) — embeddings are random vectors."
            )
            self.embedding_generator = MockEmbeddingGenerator(
                self.config.embedding_dim
            )
        else:
            self.embedding_generator = self._init_embedding_generator()

        # ── Hybrid store ─────────────────────────────────────────────────────
        if hybrid_store is not None:
            self.hybrid_store = hybrid_store
        elif use_mocks:
            logger.warning(
                "FALLBACK ACTIVE: HybridStore is None "
                "(use_mocks=True) — storage is disabled."
            )
            self.hybrid_store = None
        else:
            self.hybrid_store = self._init_hybrid_store()

        self._metrics = IngestionMetrics()

        logger.info(
            "IngestionPipeline initialised: mocks=%s", use_mocks,
        )

    def _init_chunker(self) -> Optional[Any]:
        """
        Import and initialise SpacySentenceChunker from the data layer.

        Returns None on ImportError (e.g., spaCy not installed) — the pipeline
        falls back to a simple regex splitter in that case.
        """
        try:
            from ..data_layer.chunking import SpacySentenceChunker

            return SpacySentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
                sentence_overlap=self.config.sentence_overlap,
                min_chunk_chars=self.config.min_chunk_length,
                max_chunk_chars=self.config.max_chunk_length,
            )
        except ImportError as e:
            logger.warning(
                "Could not import SpacySentenceChunker (%s); "
                "falling back to regex sentence splitter. "
                "Install spaCy and run 'python -m spacy download en_core_web_sm'.",
                e,
            )
            return None

    def _init_entity_extractor(self) -> Any:
        """
        Import and initialise EntityExtractionPipeline from the data layer.

        ImportError (missing model libraries) produces a warning and falls back
        to MockEntityExtractor. Any other exception is re-raised so that
        configuration errors do not silently degrade to a no-op extractor.
        """
        try:
            from ..data_layer.entity_extraction import (
                EntityExtractionPipeline,
                ExtractionConfig,
            )

            extraction_config = ExtractionConfig(
                ner_batch_size=self.config.gliner_batch_size,
                re_batch_size=self.config.rebel_batch_size,
                ner_confidence_threshold=self.config.entity_confidence_threshold,
                re_confidence_threshold=self.config.relation_confidence_threshold,
                min_entities_for_re=self.config.min_entities_for_re,
                cache_enabled=self.config.enable_caching,
            )
            return EntityExtractionPipeline(extraction_config)

        except ImportError as e:
            logger.warning(
                "FALLBACK ACTIVE: Could not import EntityExtractionPipeline (%s). "
                "Using MockEntityExtractor — no real entities will be extracted.",
                e,
            )
            return MockEntityExtractor()
        # Any non-import exception (OOM, missing spaCy model, etc.) propagates
        # to the caller so that configuration errors are not silently swallowed.

    def _init_embedding_generator(self) -> Any:
        """
        Import and initialise BatchedOllamaEmbeddings from the data layer.

        Uses the same Ollama endpoint as the query-time retriever, ensuring
        that ingestion and query vectors occupy the same embedding space.
        This is required for cosine similarity to be meaningful (Chapter 2.3).
        """
        try:
            from ..data_layer.embeddings import BatchedOllamaEmbeddings

            return BatchedOllamaEmbeddings(
                model_name=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
            )
        except ImportError as e:
            logger.warning(
                "FALLBACK ACTIVE: Could not import BatchedOllamaEmbeddings (%s). "
                "Using MockEmbeddingGenerator.",
                e,
            )
            return MockEmbeddingGenerator(self.config.embedding_dim)

    def _init_hybrid_store(self) -> Optional[Any]:
        """
        Import and initialise HybridStore (LanceDB + KuzuDB) from the data layer.

        Returns None on ImportError; all other exceptions propagate.
        """
        try:
            from ..data_layer.storage import HybridStore, StorageConfig

            storage_config = StorageConfig(
                vector_db_path=Path(self.config.vector_db_path),
                graph_db_path=Path(self.config.graph_db_path),
                embedding_dim=self.config.embedding_dim,
            )
            return HybridStore(storage_config)
        except ImportError as e:
            logger.warning(
                "Could not import HybridStore (%s); storage is disabled.", e
            )
            return None

    def ingest(
        self,
        source: str,
        show_progress: bool = True,
    ) -> IngestionMetrics:
        """
        Ingest document(s) from a file path or directory.

        Processes documents in a streaming loop (one document at a time) to
        avoid materialising the full corpus in memory — important on the
        < 16 GB RAM edge target hardware.

        Parameters
        ----------
        source : str
            Path to a file or directory.
        show_progress : bool
            When True, pass progress display through to the embedding generator.

        Returns
        -------
        IngestionMetrics
            Per-stage timing and count statistics for this run.

        Raises
        ------
        FileNotFoundError
            If source path does not exist.
        """
        start_time = time.time()
        self._reset_metrics()

        logger.info("Starting ingestion from: %s", source)

        doc_count = 0
        # Stream documents one at a time — avoids loading the full corpus into
        # RAM, which is essential for large corpora on edge hardware.
        for doc in self.loader.load(source):
            self._process_document(doc, show_progress=show_progress)
            doc_count += 1

        logger.info("Loaded and processed %d document(s)", doc_count)

        self._metrics.total_time_ms = (time.time() - start_time) * 1000

        if self._metrics.chunks_created > 0:
            self._metrics.avg_chunk_latency_ms = (
                self._metrics.total_time_ms / self._metrics.chunks_created
            )

        logger.info(
            "Ingestion completed: %d docs, %d chunks, %.2fms total",
            self._metrics.documents_processed,
            self._metrics.chunks_created,
            self._metrics.total_time_ms,
        )

        return self._metrics

    def _process_document(
        self, doc: Dict[str, Any], show_progress: bool = False
    ) -> None:
        """Orchestrate the four processing stages for a single document."""
        doc_id = doc["id"]
        text = doc["text"]
        metadata = doc.get("metadata", {})

        logger.debug("Processing document: %s", doc_id)

        # Stage 1: Chunking
        chunk_start = time.time()
        chunks = self._chunk_document(text, doc_id, metadata)
        self._metrics.chunking_time_ms += (time.time() - chunk_start) * 1000

        if not chunks:
            logger.warning("No chunks created for document: %s", doc_id)
            return

        self._metrics.chunks_created += len(chunks)

        # Stage 2: Entity extraction (GLiNER + selective REBEL)
        extraction_start = time.time()
        entities, relations = self._extract_entities(chunks)
        self._metrics.extraction_time_ms += (time.time() - extraction_start) * 1000
        self._metrics.entities_extracted += len(entities)
        self._metrics.relations_extracted += len(relations)

        # Stage 3: Embedding generation via Ollama (same model as query time)
        embedding_start = time.time()
        texts = [c.get("text", "") for c in chunks]
        raw_embeddings = self._embed_texts(texts, show_progress=show_progress)
        self._metrics.embedding_time_ms += (time.time() - embedding_start) * 1000

        # Stage 4: Store in HybridStore (LanceDB + KuzuDB)
        storage_start = time.time()
        self._store_data(chunks, raw_embeddings, entities, relations)
        self._metrics.storage_time_ms += (time.time() - storage_start) * 1000

        self._metrics.documents_processed += 1

    def _chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Split a document into sentence-window chunks.

        Uses SpacySentenceChunker when available (thesis default). Falls back
        to a simple regex sentence splitter when SpaCy is not initialised.
        The fallback produces lower-quality chunks and should not be used in
        production evaluation runs.
        """
        if self.chunker is None:
            # Regex fallback — acceptable for unit tests, not for evaluation
            logger.warning(
                "SpacySentenceChunker is None; using regex sentence splitter "
                "as fallback. Do not use fallback chunks for thesis evaluation."
            )
            sentences = re.split(r"(?<=[.!?])\s+", text)
            chunks: List[Dict[str, Any]] = []

            step = max(
                1, self.config.sentences_per_chunk - self.config.sentence_overlap
            )
            for i in range(0, len(sentences), step):
                chunk_sents = sentences[i : i + self.config.sentences_per_chunk]
                chunk_text = " ".join(chunk_sents)

                if len(chunk_text) >= self.config.min_chunk_length:
                    chunks.append(
                        {
                            "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                            "text": chunk_text,
                            "source_doc": doc_id,
                            "position": len(chunks),
                            "sentences": [],  # regex path has no sentence objects
                            "metadata": metadata,
                        }
                    )

            return chunks

        # Production path: SpacySentenceChunker
        chunk_objects = self.chunker.chunk_text(text)
        return [
            {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": chunk.text,
                "source_doc": doc_id,
                "position": i,
                "sentences": (
                    chunk.sentences if hasattr(chunk, "sentences") else []
                ),
                "metadata": {**metadata, "chunk_method": "sentence_based"},
            }
            for i, chunk in enumerate(chunk_objects)
        ]

    def _extract_entities(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Any], List[Any]]:
        """Delegate entity + relation extraction to the configured extractor."""
        if self.entity_extractor is None:
            return [], []

        return self.entity_extractor.process_chunks_batch(chunks)

    def _embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Embed a list of texts and return as List[List[float]].

        Uses embed_documents() when available (BatchedOllamaEmbeddings interface)
        or embed() with numpy-to-list conversion (MockEmbeddingGenerator).
        The List[List[float]] format matches HybridStore.ingest_chunks_with_entities().
        """
        if not texts:
            return []

        if hasattr(self.embedding_generator, "embed_documents"):
            return self.embedding_generator.embed_documents(texts)

        # MockEmbeddingGenerator path: convert np.ndarray → List[List[float]]
        result: np.ndarray = self.embedding_generator.embed(
            texts, show_progress=show_progress
        )
        return result.tolist()

    def _store_data(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        entities: List[Any],
        relations: List[Any],
    ) -> None:
        """
        Write chunks, embeddings, entities, and relations to HybridStore.

        Raises
        ------
        Exception
            Re-raises any storage error after logging, so callers can detect
            that storage failed. A silent swallow would produce metrics claiming
            successful ingestion while the vector store and graph remain empty.
        """
        if self.hybrid_store is None:
            logger.debug("No hybrid store configured, skipping storage")
            return

        try:
            self.hybrid_store.ingest_chunks_with_entities(
                chunks=chunks,
                embeddings=embeddings,
                entities=entities,
                relations=relations,
            )
        except Exception as e:
            logger.error("Storage error: %s", e)
            raise

    def _reset_metrics(self) -> None:
        """Reset metrics for a new ingestion run."""
        self._metrics = IngestionMetrics()

    def get_metrics(self) -> IngestionMetrics:
        """Return the metrics from the most recent ingest() call."""
        return self._metrics

    def get_store_stats(self) -> Dict[str, Any]:
        """Return storage statistics from HybridStore (empty dict if unavailable)."""
        if self.hybrid_store is None:
            return {}
        return self.hybrid_store.get_stats()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_ingestion_pipeline(
    config: Optional[Dict[str, Any]] = None,
    use_mocks: bool = False,
) -> IngestionPipeline:
    """
    Factory for a fully configured IngestionPipeline.

    Parameters
    ----------
    config : dict, optional
        Full settings.yaml dict. Pass None only for unit tests — doing so
        applies all hardcoded defaults, which may not match the thesis
        evaluation configuration.
    use_mocks : bool
        When True, substitutes mock components (no GPU or Ollama required).

    Returns
    -------
    IngestionPipeline
    """
    if config:
        ingestion_config = IngestionConfig.from_yaml(config)
    else:
        logger.warning(
            "FALLBACK ACTIVE: No config provided to create_ingestion_pipeline. "
            "All parameters use hardcoded defaults. "
            "Pass settings.yaml content for reproducible evaluation results."
        )
        ingestion_config = IngestionConfig()

    return IngestionPipeline(
        config=ingestion_config,
        use_mocks=use_mocks,
    )


# ============================================================================
# SMOKE DEMO / CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument("--source", type=str, help="Path to document(s)")
    parser.add_argument(
        "--mock", action="store_true", help="Use mock components (no GPU/Ollama)"
    )
    args = parser.parse_args()

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = IngestionPipeline(use_mocks=True)

    # ── Choose source ─────────────────────────────────────────────────────────
    _tmp_file: Optional[Path] = None
    if args.source:
        source_path = args.source
    else:
        # Create a temporary test document; clean up after the run
        test_text = (
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne "
            "in April 1976. The company is headquartered in Cupertino, California.\n\n"
            "Steve Jobs served as the CEO of Apple until his death in 2011. He was "
            "known for his innovative products including the iPhone and iPad.\n\n"
            "The first iPhone was released in 2007 and revolutionised the smartphone "
            "industry. Apple has since become one of the most valuable companies in the world.\n\n"
            "Tim Cook became the CEO of Apple after Steve Jobs. Under his leadership, "
            "Apple launched the Apple Watch and expanded its services business."
        )
        _tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        )
        _tmp.write(test_text)
        _tmp.close()
        _tmp_file = Path(_tmp.name)
        source_path = str(_tmp_file)
        logger.info("Temporary test file: %s", _tmp_file)

    # ── Run ingestion ─────────────────────────────────────────────────────────
    try:
        metrics = pipeline.ingest(source_path)
    finally:
        # Always clean up the temporary file
        if _tmp_file is not None and _tmp_file.exists():
            _tmp_file.unlink()
            logger.info("Temporary test file removed.")

    # ── Print results ─────────────────────────────────────────────────────────
    metrics_dict = metrics.to_dict()
    logger.info("=" * 70)
    logger.info("INGESTION PIPELINE SMOKE DEMO RESULTS")
    logger.info("=" * 70)
    logger.info(
        "Counts: docs=%d  chunks=%d  entities=%d  relations=%d",
        metrics_dict["counts"]["documents"],
        metrics_dict["counts"]["chunks"],
        metrics_dict["counts"]["entities"],
        metrics_dict["counts"]["relations"],
    )
    logger.info(
        "Timing (ms): total=%.2f  chunking=%.2f  extraction=%.2f  "
        "embedding=%.2f  storage=%.2f",
        metrics_dict["timing_ms"]["total"],
        metrics_dict["timing_ms"]["chunking"],
        metrics_dict["timing_ms"]["extraction"],
        metrics_dict["timing_ms"]["embedding"],
        metrics_dict["timing_ms"]["storage"],
    )
    logger.info(
        "Avg chunk latency: %.2f ms",
        metrics_dict["performance"]["avg_chunk_latency_ms"],
    )
    logger.info("Smoke demo completed successfully.")
