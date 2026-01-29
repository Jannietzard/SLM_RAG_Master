"""
Document Ingestion Pipeline: Chunking → Entity Extraction → Hybrid Storage

Version: 4.0.0 - MASTERTHESIS IMPLEMENTATION
Author: Edge-RAG Research Project

===============================================================================
IMPLEMENTATION GEMÄSS MASTERTHESIS ABSCHNITT 2
===============================================================================

Die Ingestion-Pipeline transformiert Rohdokumente in eine strukturierte
Wissensrepräsentation bestehend aus:
    1. Vector Store (LanceDB): Embedding-basierte semantische Suche
    2. Knowledge Graph (KuzuDB): Entity-Relation-basierte Traversierung

Pipeline-Stufen:
    1. Document Loading: PDF, TXT, JSON, JSONL Support
    2. Sentence-Based Chunking: 3-Satz-Fenster mit Überlappung
    3. Entity Extraction: GLiNER (NER) + REBEL (RE)
    4. Embedding Generation: nomic-embed-text-v1.5
    5. Hybrid Storage: Parallele Speicherung in LanceDB + KuzuDB

Latenz-Optimierungen (Abschnitt 2.5):
    - Batch-Processing: GLiNER (16), REBEL (8), Embeddings (32)
    - Entity Caching: LRU + SQLite für häufige Entitäten
    - Selective RE: Nur Chunks mit ≥2 Entities

Performance-Ziele:
    - Durchschnittliche Ingestion-Latenz: 80-120ms pro Chunk
    - Typisches HotpotQA-Dokument (10-15 Chunks): < 2 Sekunden

===============================================================================
"""

import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator, Generator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IngestionConfig:
    """
    Konfiguration für die Ingestion-Pipeline.
    
    Alle Parameter gemäß Masterthesis spezifiziert.
    """
    # Chunking (Abschnitt 2.2)
    sentences_per_chunk: int = 3
    sentence_overlap: int = 1
    min_chunk_length: int = 50
    max_chunk_length: int = 2000
    
    # Entity Extraction (Abschnitt 2.5)
    gliner_batch_size: int = 16
    rebel_batch_size: int = 8
    entity_confidence_threshold: float = 0.5
    relation_confidence_threshold: float = 0.7
    min_entities_for_re: int = 2  # Selective RE
    
    # Embeddings (Abschnitt 2.3)
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768
    embedding_batch_size: int = 32
    
    # Storage Paths
    vector_db_path: str = "./data/vector_db"
    graph_db_path: str = "./data/knowledge_graph"
    entity_cache_path: str = "./data/entity_cache.db"
    
    # Performance
    enable_caching: bool = True
    enable_parallel: bool = True
    num_workers: int = 4
    
    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> 'IngestionConfig':
        """Erstelle Config aus YAML-Dict."""
        return cls(
            # Chunking
            sentences_per_chunk=config.get("chunking", {}).get("sentence_chunking", {}).get("sentences_per_chunk", 3),
            sentence_overlap=config.get("chunking", {}).get("sentence_chunking", {}).get("sentence_overlap", 1),
            min_chunk_length=config.get("chunking", {}).get("sentence_chunking", {}).get("min_chunk_length", 50),
            max_chunk_length=config.get("chunking", {}).get("sentence_chunking", {}).get("max_chunk_length", 2000),
            
            # Entity Extraction
            gliner_batch_size=config.get("entity_extraction", {}).get("gliner", {}).get("batch_size", 16),
            rebel_batch_size=config.get("entity_extraction", {}).get("rebel", {}).get("batch_size", 8),
            entity_confidence_threshold=config.get("entity_extraction", {}).get("gliner", {}).get("confidence_threshold", 0.5),
            relation_confidence_threshold=config.get("entity_extraction", {}).get("rebel", {}).get("confidence_threshold", 0.7),
            min_entities_for_re=config.get("entity_extraction", {}).get("rebel", {}).get("min_entities_for_re", 2),
            
            # Embeddings
            embedding_model=config.get("embedding", {}).get("model", "nomic-ai/nomic-embed-text-v1.5"),
            embedding_dim=config.get("embedding", {}).get("dimension", 768),
            embedding_batch_size=config.get("embedding", {}).get("batch_size", 32),
            
            # Storage
            vector_db_path=config.get("storage", {}).get("vector_store", {}).get("path", "./data/vector_db"),
            graph_db_path=config.get("storage", {}).get("graph_store", {}).get("path", "./data/knowledge_graph"),
            entity_cache_path=config.get("caching", {}).get("entity_cache", {}).get("path", "./data/entity_cache.db"),
            
            # Performance
            enable_caching=config.get("caching", {}).get("enabled", True),
            enable_parallel=config.get("optimization", {}).get("parallel_processing", {}).get("enabled", True),
            num_workers=config.get("optimization", {}).get("parallel_processing", {}).get("num_workers", 4)
        )


# ============================================================================
# METRICS & STATISTICS
# ============================================================================

@dataclass
class IngestionMetrics:
    """Metriken für die Ingestion-Pipeline."""
    documents_processed: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    
    # Timing (ms)
    total_time_ms: float = 0.0
    chunking_time_ms: float = 0.0
    extraction_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    storage_time_ms: float = 0.0
    
    # Performance
    avg_chunk_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": {
                "documents": self.documents_processed,
                "chunks": self.chunks_created,
                "entities": self.entities_extracted,
                "relations": self.relations_extracted
            },
            "timing_ms": {
                "total": self.total_time_ms,
                "chunking": self.chunking_time_ms,
                "extraction": self.extraction_time_ms,
                "embedding": self.embedding_time_ms,
                "storage": self.storage_time_ms
            },
            "performance": {
                "avg_chunk_latency_ms": self.avg_chunk_latency_ms,
                "cache_hit_rate": self.cache_hit_rate
            }
        }


# ============================================================================
# DOCUMENT LOADER
# ============================================================================

class DocumentLoader:
    """
    Lädt Dokumente aus verschiedenen Formaten.
    
    Unterstützte Formate:
        - Plain Text (.txt)
        - JSON (.json)
        - JSON Lines (.jsonl) - z.B. HotpotQA-Format
        - Markdown (.md)
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.json', '.jsonl', '.md'}
    
    def load(self, path: str) -> Iterator[Dict[str, Any]]:
        """
        Lade Dokument(e) aus Pfad.
        
        Args:
            path: Pfad zu Datei oder Verzeichnis
        
        Yields:
            Dict mit 'id', 'text', 'metadata'
        """
        path = Path(path)
        
        if path.is_file():
            yield from self._load_file(path)
        elif path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    yield from self._load_file(file_path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    
    def _load_file(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Lade einzelne Datei."""
        suffix = path.suffix.lower()
        
        if suffix == '.txt' or suffix == '.md':
            yield from self._load_text(path)
        elif suffix == '.json':
            yield from self._load_json(path)
        elif suffix == '.jsonl':
            yield from self._load_jsonl(path)
        else:
            logger.warning(f"Unsupported file format: {path}")
    
    def _load_text(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Lade Plain Text Datei."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            doc_id = self._generate_id(str(path))
            yield {
                'id': doc_id,
                'text': text,
                'metadata': {
                    'source': str(path),
                    'format': 'text'
                }
            }
        except Exception as e:
            logger.error(f"Failed to load text file {path}: {e}")
    
    def _load_json(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Lade JSON Datei."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                for i, item in enumerate(data):
                    yield self._parse_json_item(item, f"{path}:{i}")
            elif isinstance(data, dict):
                yield self._parse_json_item(data, str(path))
                
        except Exception as e:
            logger.error(f"Failed to load JSON file {path}: {e}")
    
    def _load_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        """
        Lade JSON Lines Datei.
        
        Typisches Format für HotpotQA, 2WikiMultiHopQA:
        {"question": "...", "answer": "...", "context": [...]}
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        item = json.loads(line)
                        yield self._parse_json_item(item, f"{path}:{i}")
                        
        except Exception as e:
            logger.error(f"Failed to load JSONL file {path}: {e}")
    
    def _parse_json_item(self, item: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Parse JSON Item zu Dokument."""
        # HotpotQA/WikiMultiHop Format
        if 'context' in item:
            # context ist Liste von (title, sentences)
            texts = []
            for title, sentences in item.get('context', []):
                texts.append(f"{title}\n" + " ".join(sentences))
            text = "\n\n".join(texts)
        elif 'text' in item:
            text = item['text']
        elif 'content' in item:
            text = item['content']
        elif 'passage' in item:
            text = item['passage']
        else:
            # Fallback: Gesamtes Item als Text
            text = json.dumps(item)
        
        doc_id = item.get('id', item.get('_id', self._generate_id(source)))
        
        return {
            'id': str(doc_id),
            'text': text,
            'metadata': {
                'source': source,
                'question': item.get('question'),
                'answer': item.get('answer'),
                'type': item.get('type'),
                'level': item.get('level')
            }
        }
    
    @staticmethod
    def _generate_id(source: str) -> str:
        """Generiere deterministische ID."""
        return hashlib.sha256(source.encode()).hexdigest()[:16]


# ============================================================================
# EMBEDDING GENERATOR
# ============================================================================

class EmbeddingGenerator:
    """
    Generiere Embeddings mit nomic-embed-text-v1.5.
    
    Features:
        - Batch Processing (32 Texte pro Batch)
        - L2 Normalization für Cosine Similarity
        - Caching für häufige Texte
    """
    
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        batch_size: int = 32,
        device: str = "auto",
        enable_cache: bool = True
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.enable_cache = enable_cache
        
        # Lazy-loaded
        self._model = None
        
        # Simple cache
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 10000
        
        logger.info(f"EmbeddingGenerator configured: {model_name}")
    
    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Lade Embedding-Modell."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True
            )
            
            # Move to appropriate device
            if self.device == "auto":
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to('cuda')
            
            logger.info("Embedding model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generiere Embeddings für Texte.
        
        Args:
            texts: Liste von Texten
            show_progress: Zeige Progress-Bar
        
        Returns:
            numpy array (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Check cache
        if self.enable_cache:
            embeddings, uncached_indices, uncached_texts = self._check_cache(texts)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
            embeddings = [None] * len(texts)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            start_time = time.time()
            
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True  # L2 normalization
            )
            
            # Update cache and results
            for i, idx in enumerate(uncached_indices):
                embeddings[idx] = new_embeddings[i]
                
                if self.enable_cache:
                    self._update_cache(texts[idx], new_embeddings[i])
            
            latency = (time.time() - start_time) * 1000
            logger.debug(
                f"Generated {len(uncached_texts)} embeddings in {latency:.2f}ms "
                f"({latency/len(uncached_texts):.2f}ms/text)"
            )
        
        return np.array(embeddings)
    
    def _check_cache(
        self,
        texts: List[str]
    ) -> Tuple[List[Optional[np.ndarray]], List[int], List[str]]:
        """Check cache für Texte."""
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        return embeddings, uncached_indices, uncached_texts
    
    def _update_cache(self, text: str, embedding: np.ndarray):
        """Update Embedding-Cache."""
        if len(self._cache) >= self._cache_max_size:
            # Simple eviction: remove first entry
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        cache_key = self._get_cache_key(text)
        self._cache[cache_key] = embedding
    
    @staticmethod
    def _get_cache_key(text: str) -> str:
        """Generiere Cache-Key für Text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "size": len(self._cache),
            "max_size": self._cache_max_size
        }


# ============================================================================
# MOCK COMPONENTS (für Tests)
# ============================================================================

class MockEmbeddingGenerator:
    """Mock Embedding Generator für Tests ohne GPU."""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
    
    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Generiere Random Embeddings."""
        embeddings = np.random.randn(len(texts), self.embedding_dim)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {"size": 0, "max_size": 0}


class MockEntityExtractor:
    """Mock Entity Extractor für Tests."""
    
    def process_chunks_batch(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[Any]]:
        """Return empty entities and relations."""
        from dataclasses import dataclass
        
        @dataclass
        class MockEntity:
            entity_id: str
            name: str
            type: str
            confidence: float
            source_chunk_id: str
        
        @dataclass
        class MockRelation:
            subject: str
            relation_type: str
            object: str
            confidence: float
            source_chunk_ids: List[str]
        
        entities = []
        relations = []
        
        # Simple mock: extract capitalized words as entities
        import re
        for chunk in chunks:
            text = chunk.get('text', '')
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for word in words[:3]:  # Max 3 per chunk
                entities.append(MockEntity(
                    entity_id=hashlib.md5(word.encode()).hexdigest()[:8],
                    name=word,
                    type="CONCEPT",
                    confidence=0.8,
                    source_chunk_id=chunk.get('chunk_id', '')
                ))
        
        return entities, relations
    
    def get_stats(self) -> Dict[str, Any]:
        return {"entities": 0, "relations": 0}


# ============================================================================
# MAIN INGESTION PIPELINE
# ============================================================================

class IngestionPipeline:
    """
    Haupt-Ingestion-Pipeline für Dokumente.
    
    Orchestriert:
        1. Document Loading
        2. Sentence-Based Chunking
        3. Entity Extraction (GLiNER + REBEL)
        4. Embedding Generation
        5. Hybrid Storage (Vector + Graph)
    
    Performance-Ziel: 80-120ms pro Chunk (Masterthesis Abschnitt 2.5)
    """
    
    def __init__(
        self,
        config: IngestionConfig = None,
        chunker=None,
        entity_extractor=None,
        embedding_generator=None,
        hybrid_store=None,
        use_mocks: bool = False
    ):
        """
        Args:
            config: IngestionConfig
            chunker: SentenceBasedChunker Instanz
            entity_extractor: EntityExtractionPipeline Instanz
            embedding_generator: EmbeddingGenerator Instanz
            hybrid_store: HybridStore Instanz
            use_mocks: True für Mock-Komponenten (Tests)
        """
        self.config = config or IngestionConfig()
        self.use_mocks = use_mocks
        
        # Initialize components
        self.loader = DocumentLoader()
        
        # Chunker
        if chunker is not None:
            self.chunker = chunker
        else:
            self.chunker = self._init_chunker()
        
        # Entity Extractor
        if entity_extractor is not None:
            self.entity_extractor = entity_extractor
        elif use_mocks:
            self.entity_extractor = MockEntityExtractor()
        else:
            self.entity_extractor = self._init_entity_extractor()
        
        # Embedding Generator
        if embedding_generator is not None:
            self.embedding_generator = embedding_generator
        elif use_mocks:
            self.embedding_generator = MockEmbeddingGenerator(self.config.embedding_dim)
        else:
            self.embedding_generator = EmbeddingGenerator(
                model_name=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
                enable_cache=self.config.enable_caching
            )
        
        # Hybrid Store
        if hybrid_store is not None:
            self.hybrid_store = hybrid_store
        elif use_mocks:
            self.hybrid_store = None
        else:
            self.hybrid_store = self._init_hybrid_store()
        
        # Metrics
        self._metrics = IngestionMetrics()
        
        logger.info(
            f"IngestionPipeline initialized: "
            f"mocks={use_mocks}, parallel={self.config.enable_parallel}"
        )
    
    def _init_chunker(self):
        """Initialize Sentence-Based Chunker."""
        try:
            # FIX: Richtigen Klassennamen importieren
            from ..data_layer.chunking import SpacySentenceChunker
            
            # FIX: Klasse direkt mit Parametern initialisieren (kein Config-Objekt nötig)
            return SpacySentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
                sentence_overlap=self.config.sentence_overlap,
                min_chunk_chars=self.config.min_chunk_length,  # Achtung: Parameter heißt in chunking.py anders!
                max_chunk_chars=self.config.max_chunk_length
            )
        except ImportError as e:
            logger.warning(f"Could not import chunker: {e}")
            return None
    
    def _init_entity_extractor(self):
        """Initialize Entity Extraction Pipeline."""
        try:
            from ..data_layer.entity_extraction import EntityExtractionPipeline, ExtractionConfig
            
            extraction_config = ExtractionConfig(
                gliner_batch_size=self.config.gliner_batch_size,
                rebel_batch_size=self.config.rebel_batch_size,
                entity_confidence_threshold=self.config.entity_confidence_threshold,
                relation_confidence_threshold=self.config.relation_confidence_threshold,
                min_entities_for_re=self.config.min_entities_for_re,
                enable_caching=self.config.enable_caching
            )
            return EntityExtractionPipeline(extraction_config)
        except ImportError as e:
            logger.warning(f"Could not import entity extractor: {e}")
            return MockEntityExtractor()
    
    def _init_hybrid_store(self):
        """Initialize Hybrid Store."""
        try:
            from ..data_layer.storage import HybridStore, StorageConfig
            
            storage_config = StorageConfig(
                vector_db_path=self.config.vector_db_path,
                graph_db_path=self.config.graph_db_path,
                embedding_dim=self.config.embedding_dim
            )
            return HybridStore(storage_config)
        except ImportError as e:
            logger.warning(f"Could not import hybrid store: {e}")
            return None
    
    def ingest(
        self,
        source: str,
        show_progress: bool = True
    ) -> IngestionMetrics:
        """
        Ingestiere Dokument(e) aus Quelle.
        
        Args:
            source: Pfad zu Datei oder Verzeichnis
            show_progress: Zeige Fortschritt
        
        Returns:
            IngestionMetrics mit Statistiken
        """
        start_time = time.time()
        self._reset_metrics()
        
        logger.info(f"Starting ingestion from: {source}")
        
        # Load documents
        documents = list(self.loader.load(source))
        logger.info(f"Loaded {len(documents)} document(s)")
        
        # Process each document
        for doc in documents:
            self._process_document(doc)
        
        # Finalize metrics
        self._metrics.total_time_ms = (time.time() - start_time) * 1000
        
        if self._metrics.chunks_created > 0:
            self._metrics.avg_chunk_latency_ms = (
                self._metrics.total_time_ms / self._metrics.chunks_created
            )
        
        logger.info(
            f"Ingestion completed: {self._metrics.documents_processed} docs, "
            f"{self._metrics.chunks_created} chunks, "
            f"{self._metrics.total_time_ms:.2f}ms total"
        )
        
        return self._metrics
    
    def _process_document(self, doc: Dict[str, Any]):
        """Verarbeite einzelnes Dokument."""
        doc_id = doc['id']
        text = doc['text']
        metadata = doc.get('metadata', {})
        
        logger.debug(f"Processing document: {doc_id}")
        
        # 1. Chunking
        chunk_start = time.time()
        chunks = self._chunk_document(text, doc_id, metadata)
        self._metrics.chunking_time_ms += (time.time() - chunk_start) * 1000
        
        if not chunks:
            logger.warning(f"No chunks created for document: {doc_id}")
            return
        
        self._metrics.chunks_created += len(chunks)
        
        # 2. Entity Extraction
        extraction_start = time.time()
        entities, relations = self._extract_entities(chunks)
        self._metrics.extraction_time_ms += (time.time() - extraction_start) * 1000
        
        self._metrics.entities_extracted += len(entities)
        self._metrics.relations_extracted += len(relations)
        
        # 3. Embedding Generation
        embedding_start = time.time()
        texts = [c['text'] for c in chunks]
        embeddings = self.embedding_generator.embed(texts)
        self._metrics.embedding_time_ms += (time.time() - embedding_start) * 1000
        
        # 4. Storage
        storage_start = time.time()
        self._store_data(chunks, embeddings, entities, relations)
        self._metrics.storage_time_ms += (time.time() - storage_start) * 1000
        
        self._metrics.documents_processed += 1
    
    def _chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Erstelle Chunks aus Dokument-Text."""
        if self.chunker is None:
            # Fallback: Simple sentence splitting
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            
            for i in range(0, len(sentences), self.config.sentences_per_chunk - self.config.sentence_overlap):
                chunk_sents = sentences[i:i + self.config.sentences_per_chunk]
                chunk_text = " ".join(chunk_sents)
                
                if len(chunk_text) >= self.config.min_chunk_length:
                    chunks.append({
                        'chunk_id': f"{doc_id}_chunk_{len(chunks)}",
                        'text': chunk_text,
                        'source_doc': doc_id,
                        'position': len(chunks),
                        'metadata': metadata
                    })
            
            return chunks
        
        # Use actual chunker
        chunk_objects = self.chunker.chunk_text(text)
        
        chunks = []
        for i, chunk in enumerate(chunk_objects):
            chunks.append({
                'chunk_id': f"{doc_id}_chunk_{i}",
                'text': chunk.text,
                'source_doc': doc_id,
                'position': i,
                'sentences': chunk.sentences if hasattr(chunk, 'sentences') else [],
                'metadata': {**metadata, 'chunk_method': 'sentence_based'}
            })
        
        return chunks
    
    def _extract_entities(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[Any]]:
        """Extrahiere Entities und Relations."""
        if self.entity_extractor is None:
            return [], []
        
        return self.entity_extractor.process_chunks_batch(chunks)
    
    def _store_data(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        entities: List[Any],
        relations: List[Any]
    ):
        """Speichere Daten in Hybrid Store."""
        if self.hybrid_store is None:
            logger.debug("No hybrid store configured, skipping storage")
            return
        
        try:
            self.hybrid_store.ingest_chunks_with_entities(
                chunks=chunks,
                embeddings=embeddings,
                entities=entities,
                relations=relations
            )
        except Exception as e:
            logger.error(f"Storage error: {e}")
    
    def _reset_metrics(self):
        """Reset Metriken für neuen Durchlauf."""
        self._metrics = IngestionMetrics()
    
    def get_metrics(self) -> IngestionMetrics:
        """Return aktuelle Metriken."""
        return self._metrics
    
    def get_store_stats(self) -> Dict[str, Any]:
        """Return Store-Statistiken."""
        if self.hybrid_store is None:
            return {}
        return self.hybrid_store.get_stats()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_ingestion_pipeline(
    config: Dict[str, Any] = None,
    use_mocks: bool = False
) -> IngestionPipeline:
    """
    Factory für IngestionPipeline.
    
    Args:
        config: Konfiguration aus settings.yaml
        use_mocks: True für Mock-Komponenten (Tests)
    
    Returns:
        Konfigurierte IngestionPipeline
    """
    if config:
        ingestion_config = IngestionConfig.from_yaml(config)
    else:
        ingestion_config = IngestionConfig()
    
    return IngestionPipeline(
        config=ingestion_config,
        use_mocks=use_mocks
    )


# ============================================================================
# CLI / TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument("--source", type=str, help="Path to document(s)")
    parser.add_argument("--mock", action="store_true", help="Use mock components")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("INGESTION PIPELINE TEST")
    print("="*70)
    
    # Create pipeline with mocks
    pipeline = IngestionPipeline(use_mocks=True)
    
    # Test with sample text
    if args.source:
        metrics = pipeline.ingest(args.source)
    else:
        # Create test document
        test_text = """
        Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne 
        in April 1976. The company is headquartered in Cupertino, California.
        
        Steve Jobs served as the CEO of Apple until his death in 2011. He was 
        known for his innovative products including the iPhone and iPad.
        
        The first iPhone was released in 2007 and revolutionized the smartphone 
        industry. Apple has since become one of the most valuable companies in 
        the world.
        
        Tim Cook became the CEO of Apple after Steve Jobs. Under his leadership,
        Apple launched the Apple Watch and expanded its services business.
        """
        
# Save test file
    # Einfach als "test_doc.txt" im aktuellen Ordner speichern
    test_file = Path("test_doc.txt")
    
    test_file.write_text(test_text, encoding="utf-8")
    
    # .absolute() zeigt dir genau, wo die Datei jetzt liegt
    print(f"Test file created at: {test_file.absolute()}") 
    
    metrics = pipeline.ingest(str(test_file))
    
    print("\n" + "-"*70)
    print("INGESTION METRICS")
    print("-"*70)
    
    metrics_dict = metrics.to_dict()
    
    print(f"\nCounts:")
    print(f"  Documents: {metrics_dict['counts']['documents']}")
    print(f"  Chunks: {metrics_dict['counts']['chunks']}")
    print(f"  Entities: {metrics_dict['counts']['entities']}")
    print(f"  Relations: {metrics_dict['counts']['relations']}")
    
    print(f"\nTiming:")
    print(f"  Total: {metrics_dict['timing_ms']['total']:.2f} ms")
    print(f"  Chunking: {metrics_dict['timing_ms']['chunking']:.2f} ms")
    print(f"  Extraction: {metrics_dict['timing_ms']['extraction']:.2f} ms")
    print(f"  Embedding: {metrics_dict['timing_ms']['embedding']:.2f} ms")
    print(f"  Storage: {metrics_dict['timing_ms']['storage']:.2f} ms")
    
    print(f"\nPerformance:")
    print(f"  Avg Chunk Latency: {metrics_dict['performance']['avg_chunk_latency_ms']:.2f} ms")
    
    print("\n" + "="*70)
    print("INGESTION PIPELINE TEST COMPLETED")
    print("="*70)
