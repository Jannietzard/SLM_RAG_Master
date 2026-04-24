"""
Embedding Dimension Diagnostic

Standalone diagnostic script — NOT a pytest module.

Systematically checks every place in the stack where embedding dimensions
play a role and reports any mismatch that could cause low retrieval scores
or runtime crashes.

Run with:
    python -X utf8 src/evaluations/ollama_performance_diagnostic.py
    python -X utf8 src/evaluations/ollama_performance_diagnostic.py --test-query
"""
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_layer.embeddings import BatchedOllamaEmbeddings
from src.data_layer.storage import HybridStore, StorageConfig


def diagnose_embedding_dimensions():
    """Systematically diagnose all embedding-dimension touchpoints in the stack."""

    print("\n" + "=" * 70)
    print("EMBEDDING DIMENSION DIAGNOSTIC")
    print("=" * 70)

    # 1. Load config
    print("\n1. CHECKING CONFIG...")
    config_path = Path("./config/settings.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    embedding_config = config.get("embeddings", {})

    config_dim = embedding_config.get("embedding_dim", "NOT SET!")
    config_model = embedding_config.get("model_name", "NOT SET!")

    print(f"   Config embedding_dim: {config_dim}")
    print(f"   Config model_name: {config_model}")

    if config_dim == "NOT SET!":
        print("   ERROR: embedding_dim not set in config!")

    # 2. Test actual embeddings via Ollama
    print("\n2. TESTING ACTUAL EMBEDDINGS...")

    try:
        embeddings = BatchedOllamaEmbeddings(
            model_name=config_model,
            base_url=embedding_config.get("base_url", "http://localhost:11434"),
            batch_size=32,
            cache_path=Path("./cache/test_embeddings.db"),
            device="cpu",
        )

        test_embedding = embeddings.embed_query("This is a test sentence.")
        actual_dim = len(test_embedding)

        print(f"   Actual embedding dimension: {actual_dim}")

        if config_dim != "NOT SET!" and actual_dim != config_dim:
            print(f"   MISMATCH! Config says {config_dim}, actual is {actual_dim}!")
        else:
            print("   ✓ Embedding dimension verified")

    except Exception as e:
        print(f"   ✗ ERROR generating embedding: {e}")
        return

    # 3. Check StorageConfig initialisation
    print("\n3. CHECKING STORAGE CONFIG...")

    # Simulate how main.py builds StorageConfig so we can compare dimensions.
    storage_config = StorageConfig(
        vector_db_path=Path("./data/vector"),
        graph_db_path=Path("./data/graph"),
        embedding_dim=embedding_config.get("embedding_dim", 768),
    )

    storage_dim = storage_config.embedding_dim
    print(f"   StorageConfig embedding_dim: {storage_dim}")

    if storage_dim != actual_dim:
        print("  CRITICAL MISMATCH!")
        print(f"      Embeddings produce: {actual_dim} dims")
        print(f"      Storage expects: {storage_dim} dims")
        print("   This causes LOW SCORES or CRASHES!")
    else:
        print("   Storage config matches embeddings")

    # 4. Check LanceDB table (if present)
    print("\n4. CHECKING LANCEDB TABLE...")

    table_path = Path("./data/vector/documents.lance")

    if table_path.exists():
        try:
            import lancedb
            db = lancedb.connect("./data/vector")
            table = db.open_table("documents")

            sample = table.to_pandas().head(1)

            if "vector" in sample.columns:
                stored_vec = sample["vector"].iloc[0]
                stored_dim = len(stored_vec)
                print(f"   Stored vector dimension: {stored_dim}")

                if stored_dim != actual_dim:
                    print("    CRITICAL MISMATCH!")
                    print(f"      Current embeddings: {actual_dim} dims")
                    print(f"      Stored vectors: {stored_dim} dims")
                    print("    You ingested with a DIFFERENT embedding model!")
                    print("    Need to CLEAR and RE-INGEST!")
                else:
                    print("    Stored vectors match current embeddings")
            else:
                print("    No 'vector' column found in table")
                print(f"   Available columns: {list(sample.columns)}")

        except Exception as e:
            print(f"    ERROR reading table: {e}")
    else:
        print(f"     No vector store found at {table_path}")
        print("   Run ingestion first to create one")

    # 5. Summary and recommendations
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    if config_dim == "NOT SET!":
        issues.append("Config missing embedding_dim → defaults to 384")

    if config_dim != "NOT SET!" and config_dim != actual_dim:
        issues.append(f"Config says {config_dim}, but model produces {actual_dim}")

    if storage_dim != actual_dim:
        issues.append(f"Storage expects {storage_dim}, embeddings produce {actual_dim}")

    if table_path.exists():
        try:
            import lancedb
            db = lancedb.connect("./data/vector")
            table = db.open_table("documents")
            sample = table.to_pandas().head(1)
            if "vector" in sample.columns:
                stored_vec = sample["vector"].iloc[0]
                stored_dim = len(stored_vec)
                if stored_dim != actual_dim:
                    issues.append(
                        f"Stored vectors are {stored_dim}D, "
                        f"current embeddings are {actual_dim}D"
                    )
        except Exception:
            pass   # dimension lookup failed; skip gracefully

    if issues:
        print("\n ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\n" + "-" * 70)
        print("FIXES:")
        print("-" * 70)

        print("\n1. UPDATE config/settings.yaml:")
        print("   embeddings:")
        print(f"     embedding_dim: {actual_dim}  # ← set to actual dimension")

        print("\n2. CLEAR vector store (dimension mismatch):")
        print("   rm -rf data/vector/")
        print("   rm -rf cache/embeddings.db")

        print("\n3. RE-INGEST with correct config:")
        print("   python -X utf8 local_importingestion.py")

        print("\n4. VERIFY dimensions match:")
        print("   python -X utf8 src/evaluations/ollama_performance_diagnostic.py")

    else:
        print("\n✓ NO DIMENSION ISSUES FOUND!")
        print("\nIf you still get low scores (< 0.20), the problem is elsewhere:")
        print("  - Query-document language mismatch?")
        print("  - Chunking too aggressive?")
        print("  - Threshold too high?")

    print("=" * 70 + "\n")


def test_query_with_diagnostics():
    """Run a single diagnostic query and print full score / embedding statistics."""

    print("\n" + "=" * 70)
    print("QUERY TEST WITH DIAGNOSTICS")
    print("=" * 70)

    with open("./config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    embedding_config = config.get("embeddings", {})
    perf_config = config.get("performance", {})

    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=perf_config.get("batch_size", 32),
        cache_path=Path("./cache/embeddings.db"),
        device="cpu",
    )

    storage_config = StorageConfig(
        vector_db_path=Path("./data/vector"),
        graph_db_path=Path("./data/graph"),
        embedding_dim=embedding_config.get("embedding_dim", 768),
    )

    hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)

    try:
        hybrid_store.vector_store.table = hybrid_store.vector_store.db.open_table("documents")
        print(f"\n Loaded vector store with {len(hybrid_store.vector_store.table)} documents")
    except Exception as e:
        print(f"\n ERROR: Could not load vector store: {e}")
        return

    query = "What is the main concept of the paper?"
    print(f"\nQuery: '{query}'")

    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding dim: {len(query_embedding)}")
    print(f"Query embedding sample (first 5 dims): {query_embedding[:5]}")
    print(f"Query embedding magnitude: {sum(x**2 for x in query_embedding)**0.5:.4f}")

    magnitude = sum(x**2 for x in query_embedding)**0.5
    is_normalized = abs(magnitude - 1.0) < 0.01
    print(f"Is normalized: {is_normalized} (magnitude={magnitude:.6f})")

    print("\nSearching vector store (threshold=0.0)...")
    results = hybrid_store.vector_store.vector_search(
        query_embedding=query_embedding,
        top_k=10,
        threshold=0.0,
    )

    print(f"\nFound {len(results)} results:")

    if results:
        for i, result in enumerate(results[:5], 1):
            score = result["similarity"]
            text_preview = result["text"][:80].replace("\n", " ")

            print(f"\n{i}. Score: {score:.6f}")
            print(f"   Text: {text_preview}...")

            doc_id = result["document_id"]
            print(f"   Doc ID: {doc_id}")

        scores = [r["similarity"] for r in results]
        print("\nScore Statistics:")
        print(f"  Max: {max(scores):.6f}")
        print(f"  Min: {min(scores):.6f}")
        print(f"  Avg: {sum(scores)/len(scores):.6f}")

        if max(scores) < 0.20:
            print("\n CRITICAL: Max score < 0.20!")
            print("   This indicates SEVERE embedding mismatch or corruption!")
            print("   Check embedding dimensions and re-ingest.")
    else:
        print("\n No results found!")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Embedding dimension diagnostic for Edge-RAG pipeline"
    )
    parser.add_argument(
        "--test-query", action="store_true",
        help="Run a single diagnostic query with score statistics",
    )
    args = parser.parse_args()

    if args.test_query:
        test_query_with_diagnostics()
    else:
        diagnose_embedding_dimensions()
