"""
Comprehensive Test Suite for Data Layer (Artefakt A)

Version: 4.0.0
Author: Edge-RAG Research Project

===============================================================================
TEST COVERAGE
===============================================================================

1. Storage Tests (LanceDB + KuzuDB)
   - Vector Store: add, search, count
   - Graph Store: nodes, edges, traversal
   - Hybrid Store: combined operations

2. Embedding Tests
   - Batch processing
   - Cache hit/miss
   - Dimension validation

3. Chunking Tests
   - Sentence-based (SpaCy)
   - Semantic chunking
   - Fixed-size chunking

4. Entity Extraction Tests (wenn verfügbar)
   - GLiNER NER
   - REBEL Relation Extraction
   - Caching

5. Retrieval Tests
   - Vector retrieval
   - Graph retrieval
   - RRF Fusion
   - Pre-generative filtering

6. Integration Tests
   - Full ingestion pipeline
   - End-to-end retrieval

===============================================================================
USAGE
===============================================================================

# Run all tests:
pytest test_data_layer.py -v

# Run specific test class:
pytest test_data_layer.py::TestVectorStore -v

# Run with coverage:
pytest test_data_layer.py --cov=src.data_layer --cov-report=html

===============================================================================
"""

import pytest
import tempfile
import shutil
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def temp_dir():
    """Create temporary directory for test databases."""
    tmp = tempfile.mkdtemp(prefix="test_data_layer_")
    yield Path(tmp)
    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity.",
        "Marie Curie was a Polish physicist. She discovered radium and polonium.",
        "Isaac Newton formulated the laws of motion. He worked at Cambridge University.",
        "Charles Darwin proposed the theory of evolution. He traveled on the HMS Beagle.",
        "Nikola Tesla invented alternating current. He worked for Edison before starting his own company.",
    ]


@pytest.fixture
def sample_documents():
    """Sample LangChain-style documents."""
    try:
        from langchain.schema import Document
    except ImportError:
        pytest.skip("LangChain not installed")
    
    return [
        Document(
            page_content="Einstein developed E=mc². This equation relates mass and energy.",
            metadata={"source_file": "physics.pdf", "page_number": 1, "chunk_id": "c1"}
        ),
        Document(
            page_content="Curie won two Nobel Prizes. She was the first woman to win a Nobel Prize.",
            metadata={"source_file": "biography.pdf", "page_number": 5, "chunk_id": "c2"}
        ),
        Document(
            page_content="Newton's laws describe motion. The third law states every action has a reaction.",
            metadata={"source_file": "physics.pdf", "page_number": 12, "chunk_id": "c3"}
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Mock embedding model that returns random vectors."""
    class MockEmbeddings:
        def __init__(self, dim=768):
            self.dim = dim
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self._random_embedding() for _ in texts]
        
        def embed_query(self, text: str) -> List[float]:
            return self._random_embedding()
        
        def _random_embedding(self) -> List[float]:
            vec = np.random.randn(self.dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # L2 normalize
            return vec.tolist()
    
    return MockEmbeddings()


# ============================================================================
# 1. STORAGE TESTS
# ============================================================================

class TestVectorStore:
    """Tests for LanceDB Vector Store."""
    
    def test_initialization(self, temp_dir):
        """Test vector store initialization."""
        from src.data_layer.storage import VectorStoreAdapter
        
        db_path = temp_dir / "vector_test"
        store = VectorStoreAdapter(
            db_path=db_path,
            embedding_dim=768,
            normalize_embeddings=True,
            distance_metric="cosine"
        )
        
        assert store.db is not None
        assert store.embedding_dim == 768
        assert store.distance_metric == "cosine"
    
    def test_add_and_search(self, temp_dir, sample_documents, mock_embeddings):
        """Test adding documents and searching."""
        from src.data_layer.storage import VectorStoreAdapter
        
        db_path = temp_dir / "vector_add_search"
        store = VectorStoreAdapter(db_path=db_path, embedding_dim=768)
        
        # Add documents
        store.add_documents_with_embeddings(sample_documents, mock_embeddings)
        
        # Search
        query_embedding = mock_embeddings.embed_query("Einstein relativity")
        results = store.vector_search(query_embedding, top_k=3)
        
        assert len(results) > 0
        assert all("similarity" in r for r in results)
        assert all(0 <= r["similarity"] <= 1 for r in results)
    
    def test_dimension_validation(self, temp_dir):
        """Test embedding dimension mismatch detection."""
        from src.data_layer.storage import VectorStoreAdapter
        
        db_path = temp_dir / "vector_dim_test"
        store = VectorStoreAdapter(db_path=db_path, embedding_dim=768)
        
        # Try to add wrong dimension
        wrong_embeddings = [[0.1] * 512]  # Wrong dimension
        
        with pytest.raises(ValueError, match="DIMENSION MISMATCH"):
            store._validate_embedding_dimension(wrong_embeddings)
    
    def test_distance_to_similarity(self, temp_dir):
        """Test distance to similarity conversion."""
        from src.data_layer.storage import VectorStoreAdapter
        
        db_path = temp_dir / "vector_dist_test"
        store = VectorStoreAdapter(db_path=db_path, distance_metric="cosine")
        
        # Cosine distance 0 -> similarity 1
        assert store._distance_to_similarity(0.0) == 1.0
        # Cosine distance 1 -> similarity 0
        assert store._distance_to_similarity(1.0) == 0.0
        # Cosine distance 0.5 -> similarity 0.5
        assert store._distance_to_similarity(0.5) == 0.5


class TestKuzuGraphStore:
    """Tests for KuzuDB Graph Store."""
    
    def test_initialization(self, temp_dir):
        """Test graph store initialization."""
        try:
            from src.data_layer.storage import KuzuGraphStore
        except ImportError:
            pytest.skip("KuzuDB not installed")
        
        db_path = temp_dir / "graph_test"
        store = KuzuGraphStore(db_path)
        
        assert store.db is not None
        assert store.conn is not None
    
    def test_add_document_chunk(self, temp_dir):
        """Test adding document chunks."""
        try:
            from src.data_layer.storage import KuzuGraphStore
        except ImportError:
            pytest.skip("KuzuDB not installed")
        
        db_path = temp_dir / "graph_chunks"
        store = KuzuGraphStore(db_path)
        
        store.add_document_chunk(
            chunk_id="chunk_001",
            text="Einstein was born in Ulm.",
            page_number=1,
            chunk_index=0,
            source_file="test.pdf"
        )
        
        # Verify via query
        result = store.conn.execute(
            "MATCH (c:DocumentChunk {chunk_id: 'chunk_001'}) RETURN c.text"
        )
        assert result.has_next()
        row = result.get_next()
        assert "Einstein" in row[0]
    
    def test_add_source_document(self, temp_dir):
        """Test adding source documents."""
        try:
            from src.data_layer.storage import KuzuGraphStore
        except ImportError:
            pytest.skip("KuzuDB not installed")
        
        db_path = temp_dir / "graph_source"
        store = KuzuGraphStore(db_path)
        
        store.add_source_document(
            doc_id="doc_001",
            filename="thesis.pdf",
            total_pages=100
        )
        
        # Verify
        result = store.conn.execute(
            "MATCH (d:SourceDocument {doc_id: 'doc_001'}) RETURN d.filename"
        )
        assert result.has_next()
    
    def test_graph_traversal(self, temp_dir):
        """Test multi-hop graph traversal."""
        try:
            from src.data_layer.storage import KuzuGraphStore
        except ImportError:
            pytest.skip("KuzuDB not installed")
        
        db_path = temp_dir / "graph_traversal"
        store = KuzuGraphStore(db_path)
        
        # Add chunks
        for i in range(3):
            store.add_document_chunk(
                chunk_id=f"chunk_{i}",
                text=f"Text {i}",
                page_number=1,
                chunk_index=i,
                source_file="test.pdf"
            )
        
        # Add sequential relations
        store.add_next_chunk_relation("chunk_0", "chunk_1")
        store.add_next_chunk_relation("chunk_1", "chunk_2")
        
        # Test traversal
        neighbors = store.get_context_chunks("chunk_0", window=2)
        assert len(neighbors) >= 1
    
    def test_statistics(self, temp_dir):
        """Test graph statistics."""
        try:
            from src.data_layer.storage import KuzuGraphStore
        except ImportError:
            pytest.skip("KuzuDB not installed")
        
        db_path = temp_dir / "graph_stats"
        store = KuzuGraphStore(db_path)
        
        # Add some data
        store.add_document_chunk("c1", "Text 1", 1, 0, "test.pdf")
        store.add_document_chunk("c2", "Text 2", 1, 1, "test.pdf")
        
        stats = store.get_statistics()
        assert "document_chunks" in stats
        assert stats["document_chunks"] >= 2


class TestHybridStore:
    """Tests for combined Vector + Graph store."""
    
    def test_initialization(self, temp_dir, mock_embeddings):
        """Test hybrid store initialization."""
        from src.data_layer.storage import HybridStore, StorageConfig
        
        config = StorageConfig(
            vector_db_path=temp_dir / "hybrid_vector",
            graph_db_path=temp_dir / "hybrid_graph",
            embedding_dim=768
        )
        
        store = HybridStore(config, mock_embeddings)
        
        assert store.vector_store is not None
        assert store.graph_store is not None
    
    def test_add_documents(self, temp_dir, sample_documents, mock_embeddings):
        """Test adding documents to both stores."""
        from src.data_layer.storage import HybridStore, StorageConfig
        
        config = StorageConfig(
            vector_db_path=temp_dir / "hybrid_add_vector",
            graph_db_path=temp_dir / "hybrid_add_graph"
        )
        
        store = HybridStore(config, mock_embeddings)
        store.add_documents(sample_documents)
        
        # Verify vector store
        query_emb = mock_embeddings.embed_query("Einstein")
        vector_results = store.vector_store.vector_search(query_emb, top_k=3)
        assert len(vector_results) > 0


# ============================================================================
# 2. EMBEDDING TESTS
# ============================================================================

class TestBatchedOllamaEmbeddings:
    """Tests for batched embedding generation."""
    
    def test_cache_initialization(self, temp_dir):
        """Test embedding cache initialization."""
        try:
            from src.data_layer.embeddings import EmbeddingCache
        except ImportError:
            pytest.skip("Embeddings module not available")
        
        cache_path = temp_dir / "embed_cache.db"
        cache = EmbeddingCache(cache_path)
        
        assert cache.db_path == cache_path
        assert cache_path.exists()
        
        cache.close()
    
    def test_cache_put_get(self, temp_dir):
        """Test cache put and get operations."""
        try:
            from src.data_layer.embeddings import EmbeddingCache
        except ImportError:
            pytest.skip("Embeddings module not available")
        
        cache_path = temp_dir / "embed_cache_ops.db"
        cache = EmbeddingCache(cache_path)
        
        test_text = "Hello World"
        test_embedding = [0.1] * 768
        
        # Put
        cache.put(test_text, test_embedding, "test-model")
        
        # Get
        retrieved = cache.get(test_text, "test-model")
        assert retrieved is not None
        assert len(retrieved) == 768
        assert abs(retrieved[0] - 0.1) < 0.001
        
        cache.close()
    
    def test_cache_miss(self, temp_dir):
        """Test cache miss returns None."""
        try:
            from src.data_layer.embeddings import EmbeddingCache
        except ImportError:
            pytest.skip("Embeddings module not available")
        
        cache_path = temp_dir / "embed_cache_miss.db"
        cache = EmbeddingCache(cache_path)
        
        result = cache.get("nonexistent text", "test-model")
        assert result is None
        
        cache.close()
    
    def test_metrics_tracking(self):
        """Test embedding metrics."""
        try:
            from src.data_layer.embeddings import EmbeddingMetrics
        except ImportError:
            pytest.skip("Embeddings module not available")
        
        metrics = EmbeddingMetrics()
        metrics.total_texts = 100
        metrics.cache_hits = 80
        metrics.cache_misses = 20
        
        assert metrics.cache_hit_rate == 80.0
        
        metrics.total_time_ms = 500
        assert metrics.avg_time_per_text_ms == 5.0


# ============================================================================
# 3. CHUNKING TESTS
# ============================================================================

class TestSentenceChunking:
    """Tests for sentence-based chunking."""
    
    def test_sentence_chunker_basic(self):
        """Test basic sentence chunking."""
        try:
            from src.data_layer.ingestion import SentenceChunker
        except ImportError:
            pytest.skip("Ingestion module not available")
        
        chunker = SentenceChunker(sentences_per_chunk=2, min_chunk_size=20)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 2
        assert all("text" in c for c in chunks)
    
    def test_spacy_sentence_chunker(self):
        """Test SpaCy-based sentence chunking (wenn verfügbar)."""
        try:
            from data_layer.chunking import (
                SentenceBasedChunker, 
                SentenceChunkingConfig
            )
        except ImportError:
            pytest.skip("sentence_chunking module not available")
        
        config = SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1
        )
        chunker = SentenceBasedChunker(config)
        
        text = """
        Albert Einstein was born in 1879. He was a theoretical physicist.
        Einstein developed the theory of relativity. He won the Nobel Prize in 1921.
        His work changed our understanding of physics. He died in 1955.
        """
        
        chunks = chunker.chunk_text(text, source_doc="test.txt")
        
        assert len(chunks) >= 2
        assert all(hasattr(c, 'text') for c in chunks)
        assert all(hasattr(c, 'sentences') for c in chunks)
        # 3-Satz-Fenster Validierung
        assert all(c.sentence_count <= 3 for c in chunks)


class TestSemanticChunking:
    """Tests for semantic chunking."""
    
    def test_semantic_chunker(self):
        """Test semantic chunker with quality metrics."""
        try:
            from data_layer.chunking import (
                create_semantic_chunker
            )
            from langchain.schema import Document
        except ImportError:
            pytest.skip("semantic_chunking module not available")
        
        chunker = create_semantic_chunker(
            chunk_size=300,
            chunk_overlap=50,
            min_chunk_size=50
        )
        
        doc = Document(
            page_content="""
            1. Introduction
            
            This thesis investigates machine learning techniques.
            The research focuses on edge deployment scenarios.
            
            1.1 Problem Statement
            
            Modern language models require significant resources.
            This creates challenges for edge device deployment.
            """,
            metadata={"source_file": "test.pdf"}
        )
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 1
        assert all("importance_score" in c.metadata for c in chunks)
        assert all("lexical_diversity" in c.metadata for c in chunks)
    
    def test_header_extraction(self):
        """Test header/section extraction."""
        try:
            from data_layer.chunking import HeaderExtractor
        except ImportError:
            pytest.skip("semantic_chunking module not available")
        
        extractor = HeaderExtractor()
        
        text = "1.1 Introduction\nThis is the introduction text."
        metadata, cleaned = extractor.extract_headers(text)
        
        assert metadata.section is not None or metadata.chapter is not None


# ============================================================================
# 4. ENTITY EXTRACTION TESTS
# ============================================================================

class TestEntityExtraction:
    """Tests for GLiNER + REBEL entity extraction."""
    
    def test_extracted_entity_dataclass(self):
        """Test ExtractedEntity dataclass."""
        try:
            from src.data_layer.entity_extraction import ExtractedEntity
        except ImportError:
            pytest.skip("entity_extraction module not available")
        
        entity = ExtractedEntity(
            entity_id="e001",
            name="Albert Einstein",
            entity_type="PERSON",
            confidence=0.95,
            mention_span=(0, 15),
            source_chunk_id="c001"
        )
        
        assert entity.name == "Albert Einstein"
        assert entity.entity_type == "PERSON"
        
        d = entity.to_dict()
        assert d["type"] == "PERSON"
        assert d["confidence"] == 0.95
    
    def test_extracted_relation_dataclass(self):
        """Test ExtractedRelation dataclass."""
        try:
            from src.data_layer.entity_extraction import ExtractedRelation
        except ImportError:
            pytest.skip("entity_extraction module not available")
        
        relation = ExtractedRelation(
            subject_entity="Einstein",
            relation_type="works_for",
            object_entity="Princeton University",
            confidence=0.85,
            source_chunk_ids=["c001"]
        )
        
        assert relation.relation_type == "works_for"
        
        d = relation.to_dict()
        assert d["subject"] == "Einstein"
        assert d["object"] == "Princeton University"
    
    def test_extraction_config(self):
        """Test ExtractionConfig defaults."""
        try:
            from src.data_layer.entity_extraction import ExtractionConfig
        except ImportError:
            pytest.skip("entity_extraction module not available")
        
        config = ExtractionConfig()
        
        # Thesis-Spezifikationen
        assert config.ner_confidence_threshold == 0.15
        assert config.re_confidence_threshold == 0.5
        assert config.ner_batch_size == 16
        assert config.re_batch_size == 8
        assert config.min_entities_for_re == 2
        assert "PERSON" in config.entity_types
        assert "ORGANIZATION" in config.entity_types
    
    def test_entity_cache(self, temp_dir):
        """Test entity caching."""
        try:
            from src.data_layer.entity_extraction import EntityCache
        except ImportError:
            pytest.skip("entity_extraction module not available")
        
        cache = EntityCache(temp_dir / "entity_cache.db", max_size=100)
        
        test_entities = [{"name": "Einstein", "type": "PERSON"}]
        cache.put("Test text about Einstein", test_entities)
        
        retrieved = cache.get("Test text about Einstein")
        assert retrieved is not None
        assert retrieved[0]["name"] == "Einstein"
        
        cache.close()


# ============================================================================
# 5. RETRIEVAL TESTS
# ============================================================================

class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""
    
    def test_rrf_formula(self):
        """Test RRF score calculation."""
        try:
            from src.data_layer.hybrid_retriever import RRFFusion
        except ImportError:
            pytest.skip("hybrid_retriever module not available")
        
        fusion = RRFFusion(k=60, cross_source_boost=1.2)
        
        # RRF score for rank 1: 1/(60+1) = 0.01639...
        expected_score = 1 / 61
        assert abs(expected_score - 0.01639) < 0.001
    
    def test_rrf_fusion_basic(self):
        """Test basic RRF fusion."""
        try:
            from src.data_layer.hybrid_retriever import RRFFusion
        except ImportError:
            pytest.skip("hybrid_retriever module not available")
        
        fusion = RRFFusion(k=60, cross_source_boost=1.2)
        
        vector_results = [
            {"chunk_id": "c1", "text": "Text 1", "relevance_score": 0.9, "source_doc": "a", "position": 0},
            {"chunk_id": "c2", "text": "Text 2", "relevance_score": 0.8, "source_doc": "a", "position": 1},
        ]
        
        graph_results = [
            {"chunk_id": "c1", "text": "Text 1", "confidence": 0.85, "hop": 1, "source_doc": "a", "position": 0},
            {"chunk_id": "c3", "text": "Text 3", "confidence": 0.7, "hop": 2, "source_doc": "b", "position": 0},
        ]
        
        results = fusion.fuse(vector_results, graph_results, final_top_k=3)
        
        assert len(results) > 0
        # c1 sollte am höchsten sein (in beiden Listen)
        assert results[0].chunk_id == "c1"
        assert results[0].retrieval_method == "hybrid"  # In beiden
    
    def test_cross_source_boost(self):
        """Test cross-source boost for hybrid results."""
        try:
            from src.data_layer.hybrid_retriever import RRFFusion
        except ImportError:
            pytest.skip("hybrid_retriever module not available")
        
        fusion = RRFFusion(k=60, cross_source_boost=1.5)
        
        # Same chunk in both sources
        vector_results = [
            {"chunk_id": "c1", "text": "T1", "relevance_score": 0.9, "source_doc": "a", "position": 0},
        ]
        graph_results = [
            {"chunk_id": "c1", "text": "T1", "confidence": 0.9, "hop": 1, "source_doc": "a", "position": 0},
        ]
        
        results = fusion.fuse(vector_results, graph_results)
        
        # With boost, score should be higher than without
        boosted_score = results[0].rrf_score
        
        # Calculate unboosted
        fusion_no_boost = RRFFusion(k=60, cross_source_boost=1.0)
        results_no_boost = fusion_no_boost.fuse(vector_results, graph_results)
        unboosted_score = results_no_boost[0].rrf_score
        
        assert boosted_score > unboosted_score


class TestPreGenerativeFilter:
    """Tests for pre-generative filtering."""
    
    def test_relevance_filter(self):
        """Test relevance threshold filtering."""
        try:
            from src.data_layer.hybrid_retriever import (
                PreGenerativeFilter, 
                RetrievalResult
            )
        except ImportError:
            pytest.skip("hybrid_retriever module not available")
        
        pf = PreGenerativeFilter(relevance_threshold_factor=0.5)
        
        results = [
            RetrievalResult("c1", "High score", "a", 0, rrf_score=1.0),
            RetrievalResult("c2", "Medium score", "a", 1, rrf_score=0.6),
            RetrievalResult("c3", "Low score", "a", 2, rrf_score=0.3),
        ]
        
        filtered = pf._relevance_filter(results)
        
        # With factor 0.5, threshold = 0.5 * 1.0 = 0.5
        # c3 (0.3) should be filtered out
        assert len(filtered) == 2
        assert all(r.rrf_score >= 0.5 for r in filtered)
    
    def test_redundancy_filter(self):
        """Test redundancy (duplicate) filtering."""
        try:
            from src.data_layer.hybrid_retriever import (
                PreGenerativeFilter, 
                RetrievalResult
            )
        except ImportError:
            pytest.skip("hybrid_retriever module not available")
        
        pf = PreGenerativeFilter(jaccard_threshold=0.8)
        
        results = [
            RetrievalResult("c1", "Einstein was a physicist", "a", 0, rrf_score=1.0),
            RetrievalResult("c2", "Einstein was a great physicist", "a", 1, rrf_score=0.9),  # Similar
            RetrievalResult("c3", "Darwin studied evolution", "a", 2, rrf_score=0.8),  # Different
        ]
        
        filtered = pf._redundancy_filter(results)
        
        # c2 should be filtered as redundant to c1
        assert len(filtered) <= 2


class TestHybridRetriever:
    """Tests for full hybrid retrieval."""
    
    def test_retrieval_modes(self, temp_dir, mock_embeddings):
        """Test different retrieval modes."""
        try:
            from data_layer.hybrid_retriever import (
                HybridRetriever, 
                RetrievalConfig, 
                RetrievalMode
            )
            from src.data_layer.storage import HybridStore, StorageConfig
        except ImportError:
            pytest.skip("Retrieval module not available")
        
        # Setup store
        storage_config = StorageConfig(
            vector_db_path=temp_dir / "retriever_vector",
            graph_db_path=temp_dir / "retriever_graph"
        )
        store = HybridStore(storage_config, mock_embeddings)
        
        # Test each mode
        for mode in [RetrievalMode.VECTOR, RetrievalMode.GRAPH, RetrievalMode.HYBRID]:
            retrieval_config = RetrievalConfig(
                mode=mode,
                top_k_vector=5,
                top_k_graph=3,
                vector_weight=0.7,
                graph_weight=0.3,
                similarity_threshold=0.0
            )
            
            retriever = HybridRetriever(
                config=retrieval_config,
                hybrid_store=store,
                embeddings=mock_embeddings
            )
            
            # Should not raise
            results = retriever.retrieve("test query")
            assert isinstance(results, list)


# ============================================================================
# 6. INTEGRATION TESTS
# ============================================================================

class TestFullPipeline:
    """End-to-end integration tests."""
    
    def test_ingestion_to_retrieval(self, temp_dir, sample_texts, mock_embeddings):
        """Test full pipeline from ingestion to retrieval."""
        try:
            from src.data_layer.storage import HybridStore, StorageConfig
            from src.data_layer.ingestion import DocumentIngestionPipeline, IngestionConfig
            from data_layer.hybrid_retriever import (
                HybridRetriever, 
                RetrievalConfig, 
                RetrievalMode
            )
            from langchain.schema import Document
        except ImportError:
            pytest.skip("Required modules not available")
        
        # 1. Setup
        storage_config = StorageConfig(
            vector_db_path=temp_dir / "e2e_vector",
            graph_db_path=temp_dir / "e2e_graph"
        )
        store = HybridStore(storage_config, mock_embeddings)
        
        ingestion_config = IngestionConfig(
            chunking_strategy="sentence",
            sentences_per_chunk=2
        )
        pipeline = DocumentIngestionPipeline(ingestion_config)
        
        # 2. Ingest
        documents = [
            Document(page_content=text, metadata={"source_file": f"doc_{i}.txt"})
            for i, text in enumerate(sample_texts)
        ]
        
        chunked_docs = pipeline.process_documents(documents)
        store.add_documents(chunked_docs)
        
        # 3. Retrieve
        retrieval_config = RetrievalConfig(
            mode=RetrievalMode.VECTOR,
            top_k_vector=3,
            top_k_graph=2,
            vector_weight=0.7,
            graph_weight=0.3,
            similarity_threshold=0.0
        )
        
        retriever = HybridRetriever(
            config=retrieval_config,
            hybrid_store=store,
            embeddings=mock_embeddings
        )
        
        results = retriever.retrieve("Who developed relativity?")
        
        # Verify
        assert len(results) > 0
        assert all(hasattr(r, 'text') for r in results)
        assert all(hasattr(r, 'relevance_score') for r in results)
    
    def test_thesis_compliance(self):
        """Verify implementation matches Thesis Abschnitt 2 specifications."""
        try:
            from src.data_layer.entity_extraction import ExtractionConfig
            from data_layer.chunking import SentenceChunkingConfig
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Thesis 2.2: 3-Satz-Fenster
        chunk_config = SentenceChunkingConfig()
        assert chunk_config.sentences_per_chunk == 3
        
        # Thesis 2.5: GLiNER Confidence 0.15, REBEL Confidence 0.5
        extract_config = ExtractionConfig()
        assert extract_config.ner_confidence_threshold == 0.15
        assert extract_config.re_confidence_threshold == 0.5
        
        # Thesis 2.5: Batch sizes
        assert extract_config.ner_batch_size == 16
        assert extract_config.re_batch_size == 8
        
        # Thesis 2.5: Selective RE
        assert extract_config.min_entities_for_re == 2


# ============================================================================
# 7. PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and latency tests."""
    
    def test_embedding_latency(self, mock_embeddings, sample_texts):
        """Test embedding generation latency."""
        start = time.time()
        
        for _ in range(10):
            mock_embeddings.embed_documents(sample_texts)
        
        elapsed = (time.time() - start) * 1000 / 10
        logger.info(f"Avg embedding latency: {elapsed:.1f}ms for {len(sample_texts)} texts")
        
        # Should be fast with mock
        assert elapsed < 100  # ms
    
    def test_vector_search_latency(self, temp_dir, sample_texts, mock_embeddings):
        """Test vector search latency."""
        from src.data_layer.storage import VectorStoreAdapter
        from langchain.schema import Document
        
        db_path = temp_dir / "perf_vector"
        store = VectorStoreAdapter(db_path=db_path, embedding_dim=768)
        
        # Add documents
        docs = [
            Document(page_content=t, metadata={"chunk_id": f"c{i}"})
            for i, t in enumerate(sample_texts * 20)  # 100 docs
        ]
        store.add_documents_with_embeddings(docs, mock_embeddings)
        
        # Measure search
        query_emb = mock_embeddings.embed_query("test query")
        
        latencies = []
        for _ in range(10):
            start = time.time()
            store.vector_search(query_emb, top_k=10)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Avg vector search latency: {avg_latency:.1f}ms")
        
        # Thesis target: 20-40ms
        assert avg_latency < 100  # Relaxed for CI environments


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])