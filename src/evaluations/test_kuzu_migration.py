"""
KuzuDB Migration Guide & Test Script - FIXED VERSION

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

===============================================================================
FIX: Korrigierte Import-Pfade (src.data_layer.* statt data_layer.*)
===============================================================================

Usage:
    Führe dieses Script aus dem Projekt-Root-Verzeichnis aus:
    python src/evaluations/test_kuzu_migration.py
    
    ODER kopiere diese Datei und ersetze die alte Version.
"""

import sys
from pathlib import Path

# ============================================================================
# FIX: Korrekter Python-Pfad (Projekt-Root, nicht Script-Verzeichnis)
# ============================================================================
# Alte Version: sys.path.insert(0, str(Path(__file__).parent))  # ❌ FALSCH
# Neue Version:
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/evaluations -> src -> ROOT
sys.path.insert(0, str(PROJECT_ROOT))


def test_kuzu_installation():
    """Test 1: KuzuDB Installation"""
    print("\n" + "="*60)
    print("TEST 1: KuzuDB Installation")
    print("="*60)
    
    try:
        import kuzu
        print(f"✓ KuzuDB installed: version {kuzu.__version__ if hasattr(kuzu, '__version__') else 'unknown'}")
        return True
    except ImportError:
        print("✗ KuzuDB NOT installed!")
        print("  Run: pip install kuzu")
        return False


def test_storage_import():
    """Test 2: Storage Module Import"""
    print("\n" + "="*60)
    print("TEST 2: Storage Module Import")
    print("="*60)
    
    try:
        # ✅ FIX: Korrekter Import-Pfad
        from src.data_layer.storage import (
            HybridStore, 
            StorageConfig, 
            KuzuGraphStore,
            VectorStoreAdapter,
            KUZU_AVAILABLE,
        )
        print(f"✓ Storage module imported successfully")
        print(f"  KUZU_AVAILABLE: {KUZU_AVAILABLE}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_retrieval_import():
    """Test 3: Retrieval Module Import"""
    print("\n" + "="*60)
    print("TEST 3: Retrieval Module Import")
    print("="*60)
    
    try:
        # ✅ FIX: Korrekter Import-Pfad
        from src.data_layer.retrieval import (
            HybridRetriever,
            RetrievalConfig,
            RetrievalMode,
        )
        print(f"✓ Retrieval module imported successfully")
        print(f"  Available modes: {[m.value for m in RetrievalMode]}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_kuzu_database_creation():
    """Test 4: KuzuDB Database Creation"""
    print("\n" + "="*60)
    print("TEST 4: KuzuDB Database Creation")
    print("="*60)
    
    import tempfile
    import shutil
    
    try:
        # ✅ FIX: Korrekter Import-Pfad
        from src.data_layer.storage import KuzuGraphStore
        
        # Create temporary database
        temp_dir = Path(tempfile.mkdtemp()) / "test_kuzu"
        
        print(f"  Creating test database at: {temp_dir}")
        
        graph_store = KuzuGraphStore(temp_dir)
        
        # Add test data
        print("  Adding test document chunk...")
        graph_store.add_document_chunk(
            chunk_id="test_chunk_1",
            text="This is a test chunk about machine learning.",
            page_number=1,
            chunk_index=0,
            source_file="test.pdf",
        )
        
        print("  Adding test source document...")
        graph_store.add_source_document(
            doc_id="test.pdf",
            filename="test.pdf",
            total_pages=10,
        )
        
        print("  Creating FROM_SOURCE relation...")
        graph_store.add_from_source_relation("test_chunk_1", "test.pdf")
        
        # Get statistics
        stats = graph_store.get_statistics()
        print(f"\n  Graph Statistics:")
        print(f"    Document Chunks: {stats.get('document_chunks', 0)}")
        print(f"    Source Documents: {stats.get('source_documents', 0)}")
        print(f"    FROM_SOURCE edges: {stats.get('from_source_edges', 0)}")
        
        # Cleanup
        shutil.rmtree(temp_dir.parent)
        
        print("\n✓ KuzuDB database creation successful!")
        return True
        
    except Exception as e:
        print(f"✗ Database creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_traversal():
    """Test 5: Graph Traversal with Cypher"""
    print("\n" + "="*60)
    print("TEST 5: Graph Traversal (Cypher)")
    print("="*60)
    
    import tempfile
    import shutil
    
    try:
        # ✅ FIX: Korrekter Import-Pfad
        from src.data_layer.storage import KuzuGraphStore
        
        temp_dir = Path(tempfile.mkdtemp()) / "test_traversal"
        graph_store = KuzuGraphStore(temp_dir)
        
        # Create a chain of chunks
        print("  Creating chunk chain: chunk_1 -> chunk_2 -> chunk_3")
        
        for i in range(1, 4):
            graph_store.add_document_chunk(
                chunk_id=f"chunk_{i}",
                text=f"Content of chunk {i}",
                page_number=1,
                chunk_index=i-1,
                source_file="test.pdf",
            )
        
        graph_store.add_next_chunk_relation("chunk_1", "chunk_2")
        graph_store.add_next_chunk_relation("chunk_2", "chunk_3")
        
        # Test traversal
        print("\n  Testing traversal from chunk_1 (max_hops=2)...")
        visited = graph_store.graph_traversal("chunk_1", max_hops=2)
        
        print(f"  Visited nodes: {visited}")
        
        # Test find_related_chunks (if method exists)
        if hasattr(graph_store, 'find_related_chunks'):
            print("\n  Testing find_related_chunks from chunk_1...")
            related = graph_store.find_related_chunks("chunk_1", max_hops=2)
            
            for r in related:
                print(f"    - {r['chunk_id']} (hops: {r['hops']})")
        
        # Cleanup
        shutil.rmtree(temp_dir.parent)
        
        expected_nodes = {"chunk_1", "chunk_2", "chunk_3"}
        found_nodes = set(visited.keys())
        
        if expected_nodes == found_nodes:
            print("\n✓ Graph traversal successful!")
            return True
        else:
            print(f"\n✗ Traversal incomplete: expected {expected_nodes}, got {found_nodes}")
            return False
        
    except Exception as e:
        print(f"✗ Traversal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_store_integration():
    """Test 6: Full HybridStore Integration"""
    print("\n" + "="*60)
    print("TEST 6: HybridStore Integration")
    print("="*60)
    
    import tempfile
    import shutil
    
    try:
        # ✅ FIX: Korrekter Import-Pfad
        from src.data_layer.storage import HybridStore, StorageConfig
        from langchain.schema import Document
        
        # Mock embeddings
        class MockEmbeddings:
            def embed_query(self, text):
                return [0.1] * 768
            
            def embed_documents(self, texts):
                return [[0.1] * 768 for _ in texts]
        
        temp_dir = Path(tempfile.mkdtemp())
        
        config = StorageConfig(
            vector_db_path=temp_dir / "vector_db",
            graph_db_path=temp_dir / "graph_db",
            embedding_dim=768,
            graph_backend="kuzu",
        )
        
        embeddings = MockEmbeddings()
        
        print("  Creating HybridStore...")
        store = HybridStore(config, embeddings)
        
        # Add documents
        print("  Adding test documents...")
        docs = [
            Document(
                page_content="Machine learning is transforming industries.",
                metadata={
                    "chunk_id": "0",
                    "source_file": "ml_paper.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                }
            ),
            Document(
                page_content="Deep learning uses neural networks.",
                metadata={
                    "chunk_id": "1",
                    "source_file": "ml_paper.pdf",
                    "page_number": 1,
                    "chunk_index": 1,
                }
            ),
        ]
        
        store.add_documents(docs)
        
        # Check graph stats
        stats = store.graph_store.get_statistics()
        print(f"\n  Graph after ingestion:")
        print(f"    Chunks: {stats.get('document_chunks', stats.get('nodes', 0))}")
        print(f"    Documents: {stats.get('source_documents', 0)}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n✓ HybridStore integration successful!")
        return True
        
    except Exception as e:
        print(f"✗ HybridStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """Test 7: Performance Comparison (Kuzu vs NetworkX)"""
    print("\n" + "="*60)
    print("TEST 7: Performance Comparison")
    print("="*60)
    
    import tempfile
    import shutil
    import time
    
    try:
        # ✅ FIX: Korrekter Import-Pfad
        from src.data_layer.storage import KuzuGraphStore, NetworkXGraphStore, KUZU_AVAILABLE, NETWORKX_AVAILABLE
        
        if not KUZU_AVAILABLE:
            print("  ⚠ KuzuDB not available, skipping comparison")
            return True
            
        if not NETWORKX_AVAILABLE:
            print("  ⚠ NetworkX not available, skipping comparison")
            return True
        
        # Create test graphs
        num_nodes = 100
        num_edges = 300
        
        temp_kuzu = Path(tempfile.mkdtemp()) / "kuzu"
        temp_nx = Path(tempfile.mkdtemp()) / "nx.graphml"
        
        kuzu_store = KuzuGraphStore(temp_kuzu)
        nx_store = NetworkXGraphStore(temp_nx)
        
        print(f"  Creating graphs with {num_nodes} nodes and {num_edges} edges...")
        
        # Add nodes to both
        for i in range(num_nodes):
            kuzu_store.add_document_chunk(
                chunk_id=f"chunk_{i}",
                text=f"Content {i}",
                page_number=i // 10,
                chunk_index=i,
                source_file="test.pdf",
            )
            nx_store.add_entity(f"chunk_{i}", "document_chunk", {"text": f"Content {i}"})
        
        # Add random edges
        import random
        random.seed(42)
        
        for _ in range(num_edges):
            i = random.randint(0, num_nodes - 2)
            j = random.randint(i + 1, num_nodes - 1)
            
            kuzu_store.add_next_chunk_relation(f"chunk_{i}", f"chunk_{j}")
            nx_store.add_relation(f"chunk_{i}", f"chunk_{j}", "next_chunk")
        
        # Benchmark traversal
        print("\n  Benchmarking traversal (100 queries, max_hops=3)...")
        
        num_queries = 100
        
        # KuzuDB
        start = time.time()
        for i in range(num_queries):
            kuzu_store.graph_traversal(f"chunk_{i % num_nodes}", max_hops=3)
        kuzu_time = (time.time() - start) * 1000
        
        # NetworkX
        start = time.time()
        for i in range(num_queries):
            nx_store.graph_traversal(f"chunk_{i % num_nodes}", max_hops=3)
        nx_time = (time.time() - start) * 1000
        
        print(f"\n  Results ({num_queries} traversals):")
        print(f"    KuzuDB:   {kuzu_time:.1f}ms ({kuzu_time/num_queries:.2f}ms/query)")
        print(f"    NetworkX: {nx_time:.1f}ms ({nx_time/num_queries:.2f}ms/query)")
        
        if kuzu_time > 0:
            print(f"    Ratio:    {nx_time/kuzu_time:.1f}x")
        
        # Cleanup
        shutil.rmtree(temp_kuzu.parent)
        shutil.rmtree(temp_nx.parent)
        
        print("\n✓ Performance comparison complete!")
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all migration tests."""
    print("\n" + "="*60)
    print("KUZU MIGRATION TEST SUITE (FIXED)")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    
    tests = [
        ("KuzuDB Installation", test_kuzu_installation),
        ("Storage Module Import", test_storage_import),
        ("Retrieval Module Import", test_retrieval_import),
        ("KuzuDB Database Creation", test_kuzu_database_creation),
        ("Graph Traversal", test_graph_traversal),
        ("HybridStore Integration", test_hybrid_store_integration),
        ("Performance Comparison", test_performance_comparison),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nYour KuzuDB setup is working correctly!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the failing tests before proceeding.")
        print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())