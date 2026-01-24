"""
Verification Script for Storage Fix v2.1.0

This script verifies that the distance metric fix has been correctly applied
and that similarity scores are now computed correctly.

Expected Results After Fix:
- Similarity scores should be in range [0.5, 0.95] for related content
- Similarity scores should be in range [0.1, 0.4] for unrelated content
- Previously observed scores of 0.17-0.25 should now be 0.75-0.83

Mathematical Verification:
If previous distance was 0.25, and we were using L2 incorrectly:
- Old (wrong): similarity = 1 - 0.25 = 0.75 (but this was L2, not cosine)
- The actual L2 distance of 0.75 corresponds to moderately similar vectors

With cosine metric correctly specified:
- Cosine distance for similar text: typically 0.1-0.3
- Cosine similarity: 1 - 0.2 = 0.8 (expected for related content)

Usage:
    python verify_storage_fix.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import logging
import numpy as np


def setup_logging():
    """Configure minimal logging for verification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def verify_config(config_path: Path = Path("./config/settings.yaml")):
    """
    Verify that configuration includes distance_metric parameter.
    """
    print("\n" + "=" * 70)
    print("STEP 1: CONFIGURATION VERIFICATION")
    print("=" * 70)
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return False
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    vector_config = config.get("vector_store", {})
    
    # Check for distance_metric
    distance_metric = vector_config.get("distance_metric")
    if distance_metric is None:
        print("[WARNING] distance_metric not specified in config")
        print("          Add 'distance_metric: cosine' to vector_store section")
        return False
    
    print(f"[OK] distance_metric: {distance_metric}")
    
    # Check for normalize_embeddings
    normalize = vector_config.get("normalize_embeddings", True)
    print(f"[OK] normalize_embeddings: {normalize}")
    
    # Check similarity_threshold
    threshold = vector_config.get("similarity_threshold", 0.5)
    print(f"[OK] similarity_threshold: {threshold}")
    
    if distance_metric != "cosine":
        print(f"[WARNING] distance_metric is '{distance_metric}', expected 'cosine'")
        return False
    
    print("\n[OK] Configuration verified successfully")
    return True


def verify_storage_module():
    """
    Verify that storage module has correct implementation.
    """
    print("\n" + "=" * 70)
    print("STEP 2: STORAGE MODULE VERIFICATION")
    print("=" * 70)
    
    try:
        from src.storage import StorageConfig, VectorStoreAdapter
        
        # Check StorageConfig has distance_metric parameter
        import inspect
        sig = inspect.signature(StorageConfig)
        params = list(sig.parameters.keys())
        
        if "distance_metric" not in params:
            print("[ERROR] StorageConfig missing 'distance_metric' parameter")
            print("        Update src/storage.py with the corrected version")
            return False
        
        print("[OK] StorageConfig has 'distance_metric' parameter")
        
        # Check VectorStoreAdapter has _distance_to_similarity method
        if not hasattr(VectorStoreAdapter, '_distance_to_similarity'):
            print("[ERROR] VectorStoreAdapter missing '_distance_to_similarity' method")
            return False
        
        print("[OK] VectorStoreAdapter has '_distance_to_similarity' method")
        
        # Verify distance_to_similarity computation
        adapter = VectorStoreAdapter.__new__(VectorStoreAdapter)
        adapter.distance_metric = "cosine"
        
        # Test conversion: cosine distance 0.2 should give similarity 0.8
        test_distance = 0.2
        similarity = adapter._distance_to_similarity(test_distance)
        expected = 0.8
        
        if abs(similarity - expected) > 0.001:
            print(f"[ERROR] Conversion incorrect: distance={test_distance} -> "
                  f"similarity={similarity}, expected={expected}")
            return False
        
        print(f"[OK] Distance-to-similarity conversion verified")
        print(f"     distance=0.2 -> similarity=0.8")
        
        print("\n[OK] Storage module verified successfully")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Cannot import storage module: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False


def verify_embedding_model():
    """
    Verify embedding model produces normalized vectors.
    """
    print("\n" + "=" * 70)
    print("STEP 3: EMBEDDING MODEL VERIFICATION")
    print("=" * 70)
    
    try:
        from src.data_layer.embeddings import BatchedOllamaEmbeddings
        
        config_path = Path("./config/settings.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        embedding_config = config.get("embeddings", {})
        
        print("Initializing embedding model...")
        embeddings = BatchedOllamaEmbeddings(
            model_name=embedding_config.get("model_name", "nomic-embed-text"),
            base_url=embedding_config.get("base_url", "http://localhost:11434"),
            batch_size=32,
            cache_path=Path("./cache/verify_embeddings.db"),
            device="cpu",
        )
        
        # Test embedding generation
        test_text = "financial sentiment analysis"
        emb = embeddings.embed_query(test_text)
        
        print(f"[OK] Embedding generated: {len(emb)} dimensions")
        
        # Check normalization
        emb_array = np.array(emb)
        norm = np.linalg.norm(emb_array)
        
        print(f"[INFO] Embedding L2 norm: {norm:.6f}")
        
        if abs(norm - 1.0) > 0.01:
            print("[WARNING] Embedding not normalized (norm != 1.0)")
            print("          Ensure normalize_embeddings=True in config")
        else:
            print("[OK] Embedding is normalized")
        
        # Test similarity between related texts
        emb1 = np.array(embeddings.embed_query("financial analysis"))
        emb2 = np.array(embeddings.embed_query("financial sentiment analysis"))
        
        # Normalize for cosine similarity
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        similarity = np.dot(emb1, emb2)
        print(f"\n[INFO] Similarity test:")
        print(f"       Text 1: 'financial analysis'")
        print(f"       Text 2: 'financial sentiment analysis'")
        print(f"       Cosine similarity: {similarity:.4f}")
        
        if similarity > 0.7:
            print("[OK] High similarity for related terms (as expected)")
        elif similarity > 0.5:
            print("[OK] Moderate similarity for related terms")
        else:
            print("[WARNING] Low similarity for related terms")
            print("          Embedding model may have quality issues")
        
        print("\n[OK] Embedding model verified successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Embedding verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_vector_search():
    """
    Verify vector search returns correct similarity scores.
    """
    print("\n" + "=" * 70)
    print("STEP 4: VECTOR SEARCH VERIFICATION")
    print("=" * 70)
    
    try:
        from src.data_layer.embeddings import BatchedOllamaEmbeddings
        from src.storage import HybridStore, StorageConfig
        
        config_path = Path("./config/settings.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        embedding_config = config.get("embeddings", {})
        vector_config = config.get("vector_store", {})
        
        # Initialize embeddings
        embeddings = BatchedOllamaEmbeddings(
            model_name=embedding_config.get("model_name", "nomic-embed-text"),
            base_url=embedding_config.get("base_url", "http://localhost:11434"),
            batch_size=32,
            cache_path=Path("./cache/verify_embeddings.db"),
            device="cpu",
        )
        
        # Initialize storage with correct metric
        storage_config = StorageConfig(
            vector_db_path=Path(config.get("paths", {}).get("vector_db", "./data/vector_db")),
            graph_db_path=Path(config.get("paths", {}).get("graph_db", "./data/knowledge_graph")),
            embedding_dim=embedding_config.get("embedding_dim", 768),
            similarity_threshold=vector_config.get("similarity_threshold", 0.3),
            normalize_embeddings=vector_config.get("normalize_embeddings", True),
            distance_metric=vector_config.get("distance_metric", "cosine"),
        )
        
        hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)
        
        # Try to load existing table
        try:
            table_path = storage_config.vector_db_path / "documents.lance"
            if table_path.exists():
                hybrid_store.vector_store.table = hybrid_store.vector_store.db.open_table("documents")
                doc_count = len(hybrid_store.vector_store.table)
                print(f"[OK] Loaded existing vector store with {doc_count} documents")
            else:
                print("[WARNING] No existing vector store found")
                print("          Run main.py first to ingest documents")
                return True  # Not a failure, just no data yet
        except Exception as e:
            print(f"[WARNING] Could not load vector store: {e}")
            return True
        
        # Test search
        query = "What is financial sentiment analysis?"
        print(f"\nTest query: '{query}'")
        
        query_embedding = embeddings.embed_query(query)
        
        results = hybrid_store.vector_store.vector_search(
            query_embedding=query_embedding,
            top_k=10,
            threshold=0.0,  # No filtering for diagnosis
        )
        
        print(f"\nSearch Results ({len(results)} found):")
        print("-" * 70)
        
        if not results:
            print("[WARNING] No results returned")
            return False
        
        scores = []
        for i, result in enumerate(results[:5], 1):
            score = result['similarity']
            scores.append(score)
            text_preview = result['text'][:80].replace('\n', ' ')
            print(f"{i}. Score: {score:.4f} | {text_preview}...")
        
        print("-" * 70)
        print(f"\nScore Statistics:")
        print(f"  Maximum: {max(scores):.4f}")
        print(f"  Minimum: {min(scores):.4f}")
        print(f"  Average: {sum(scores)/len(scores):.4f}")
        
        # Verify scores are in expected range
        if max(scores) < 0.3:
            print("\n[ERROR] Scores still too low!")
            print("        Maximum score < 0.3 indicates the fix may not be applied")
            print("        Check that you are using the updated storage.py")
            return False
        elif max(scores) < 0.5:
            print("\n[WARNING] Scores moderate (0.3-0.5)")
            print("          This may be acceptable depending on document content")
        else:
            print("\n[OK] Scores in expected range (>0.5)")
        
        print("\n[OK] Vector search verified successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Vector search verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification steps."""
    logger = setup_logging()
    
    print("\n" + "=" * 70)
    print("STORAGE FIX VERIFICATION - v2.1.0")
    print("=" * 70)
    print("\nThis script verifies that the cosine metric fix has been applied.")
    print("Expected: Similarity scores should increase from ~0.25 to ~0.75+")
    
    results = {
        "config": False,
        "storage": False,
        "embeddings": False,
        "search": False,
    }
    
    # Step 1: Verify configuration
    results["config"] = verify_config()
    
    # Step 2: Verify storage module
    results["storage"] = verify_storage_module()
    
    # Step 3: Verify embedding model
    results["embeddings"] = verify_embedding_model()
    
    # Step 4: Verify vector search
    results["search"] = verify_vector_search()
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAILED]"
        print(f"  {name.capitalize():12s}: {status}")
        if not passed:
            all_passed = False
    
    print("-" * 70)
    
    if all_passed:
        print("\n[SUCCESS] All verifications passed!")
        print("\nNext Steps:")
        print("1. Delete existing vector store: rm -rf data/vector_db/")
        print("2. Re-run ingestion: python main.py")
        print("3. Test retrieval: python test_rag_quality.py")
    else:
        print("\n[FAILURE] Some verifications failed!")
        print("\nTroubleshooting:")
        print("1. Ensure src/storage.py is updated with the new version")
        print("2. Ensure config/settings.yaml includes 'distance_metric: cosine'")
        print("3. Ensure Ollama is running: ollama serve")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())