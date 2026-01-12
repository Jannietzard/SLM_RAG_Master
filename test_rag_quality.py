"""
RAG Quality Diagnosis & Testing Script (FIXED)

This script helps you:
1. Verify your config is loaded correctly
2. Test retrieval with different thresholds
3. Measure actual RAG quality
4. Compare vector vs hybrid retrieval

FIXED: Properly loads existing LanceDB tables
"""

import yaml
import logging
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings import BatchedOllamaEmbeddings
from src.storage import HybridStore, StorageConfig
from src.retrieval import HybridRetriever, RetrievalConfig, RetrievalMode


def load_and_verify_config(config_path: Path = Path("./config/settings.yaml")):
    """Load and print current config values."""
    print("\n" + "="*70)
    print("CONFIG VERIFICATION")
    print("="*70)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Check critical values
    print(f"\n✓ Embedding Dimension: {config['embeddings']['embedding_dim']}")
    print(f"✓ Chunk Size: {config['chunking']['chunk_size']}")
    print(f"✓ Chunk Overlap: {config['chunking']['chunk_overlap']}")
    print(f"✓ Similarity Threshold: {config['vector_store']['similarity_threshold']}")
    print(f"✓ Top K Vectors: {config['vector_store']['top_k_vectors']}")
    print(f"✓ Vector Weight: {config['rag']['vector_weight']}")
    print(f"✓ Graph Weight: {config['rag']['graph_weight']}")
    
    print("\n" + "="*70)
    
    # Check if threshold is too high
    threshold = config['vector_store']['similarity_threshold']
    if threshold >= 0.5:
        print("⚠️  WARNING: Similarity threshold is HIGH (>= 0.5)")
        print("   This may filter out too many results!")
        print("   Recommended: 0.25 or lower")
    else:
        print("✓ Similarity threshold looks good")
    
    print("="*70 + "\n")
    
    return config


def test_vector_search_raw(
    hybrid_store: HybridStore,
    embeddings: BatchedOllamaEmbeddings,
    query: str = "Was ist Wissensmanagement?"
):
    """Test raw vector search without threshold filtering."""
    print("\n" + "="*70)
    print("RAW VECTOR SEARCH TEST (No Threshold)")
    print("="*70)
    
    print(f"\nQuery: '{query}'")
    print("\nEmbedding query...")
    
    query_embedding = embeddings.embed_query(query)
    print(f"✓ Query embedded (dim: {len(query_embedding)})")
    
    # Raw search with threshold=0.0 to see ALL results
    print("\nSearching vector store (threshold=0.0)...")
    results = hybrid_store.vector_store.vector_search(
        query_embedding=query_embedding,
        top_k=20,  # Get more results
        threshold=0.0  # NO FILTERING
    )
    
    print(f"\n✓ Found {len(results)} results\n")
    
    # Display top 10 with scores
    print("Top Results with Scores:")
    print("-" * 70)
    
    for i, result in enumerate(results[:10], 1):
        score = result['similarity']
        text_preview = result['text'][:100].replace('\n', ' ')
        source = result['metadata'].get('source_file', 'unknown')
        
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Source: {source}")
        print(f"   Text: {text_preview}...")
    
    print("\n" + "="*70)
    
    # Analyze score distribution
    scores = [r['similarity'] for r in results]
    if scores:
        print("\nScore Statistics:")
        print(f"  Max Score: {max(scores):.4f}")
        print(f"  Min Score: {min(scores):.4f}")
        print(f"  Avg Score: {sum(scores)/len(scores):.4f}")
        
        # Count by threshold ranges
        ranges = [
            (0.6, 1.0, "Excellent"),
            (0.5, 0.6, "Good"),
            (0.4, 0.5, "Moderate"),
            (0.3, 0.4, "Fair"),
            (0.0, 0.3, "Poor"),
        ]
        
        print("\nScore Distribution:")
        for min_s, max_s, label in ranges:
            count = sum(1 for s in scores if min_s <= s < max_s)
            if count > 0:
                print(f"  {label:12s} ({min_s:.1f}-{max_s:.1f}): {count} results")
    
    print("="*70 + "\n")
    
    return results


def test_different_thresholds(
    hybrid_store: HybridStore,
    embeddings: BatchedOllamaEmbeddings,
    query: str = "Was ist Wissensmanagement?"
):
    """Test retrieval with different threshold values."""
    print("\n" + "="*70)
    print("THRESHOLD COMPARISON TEST")
    print("="*70)
    
    thresholds = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    query_embedding = embeddings.embed_query(query)
    
    print(f"\nQuery: '{query}'")
    print(f"\nTesting thresholds: {thresholds}")
    print("\n" + "-"*70)
    
    for threshold in thresholds:
        results = hybrid_store.vector_store.vector_search(
            query_embedding=query_embedding,
            top_k=10,
            threshold=threshold
        )
        
        if results:
            top_score = results[0]['similarity']
            print(f"Threshold {threshold:.2f}: {len(results):2d} results (top score: {top_score:.4f})")
        else:
            print(f"Threshold {threshold:.2f}:  0 results (all filtered out)")
    
    print("="*70 + "\n")


def test_german_queries(
    retriever: HybridRetriever,
    queries: list = None
):
    """Test with German queries (better for your German thesis)."""
    print("\n" + "="*70)
    print("GERMAN QUERY TEST")
    print("="*70)
    
    if queries is None:
        queries = [
            "Was ist Wissensmanagement?",
            "Welche Methoden werden diskutiert?",
            "Community of Practice Konzept",
            "Wie wird Wissen zwischen Generationen geteilt?",
            "Welche Forschungsfragen werden behandelt?",
        ]
    
    print(f"\nTesting {len(queries)} German queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 70)
        
        results = retriever.retrieve(query)
        
        if results:
            print(f"✓ {len(results)} results found")
            
            # Show top 3
            for j, result in enumerate(results[:3], 1):
                print(f"\n   {j}. Score: {result.relevance_score:.4f}")
                print(f"      Method: {result.retrieval_method}")
                text_preview = result.text[:120].replace('\n', ' ')
                print(f"      Text: {text_preview}...")
        else:
            print("✗ No results found!")
    
    print("\n" + "="*70 + "\n")


def calculate_rag_quality_metrics(
    retriever: HybridRetriever,
    test_queries: list,
    relevance_threshold: float = 0.4
):
    """Calculate RAG quality metrics."""
    print("\n" + "="*70)
    print("RAG QUALITY METRICS")
    print("="*70)
    
    total_queries = len(test_queries)
    queries_with_results = 0
    total_results = 0
    high_quality_results = 0
    all_scores = []
    
    print(f"\nTesting {total_queries} queries...")
    print(f"Relevance threshold: {relevance_threshold}\n")
    
    for query in test_queries:
        results = retriever.retrieve(query)
        
        if results:
            queries_with_results += 1
            total_results += len(results)
            
            for result in results:
                score = result.relevance_score
                all_scores.append(score)
                
                if score >= relevance_threshold:
                    high_quality_results += 1
    
    # Calculate metrics
    coverage = queries_with_results / total_queries if total_queries > 0 else 0
    avg_results_per_query = total_results / total_queries if total_queries > 0 else 0
    precision = high_quality_results / total_results if total_results > 0 else 0
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    print("="*70)
    print("RESULTS:")
    print("="*70)
    print(f"\nCoverage (queries with results): {coverage:.1%} ({queries_with_results}/{total_queries})")
    print(f"Avg results per query: {avg_results_per_query:.1f}")
    print(f"Precision@{relevance_threshold}: {precision:.1%}")
    print(f"Average relevance score: {avg_score:.4f}")
    
    if all_scores:
        print(f"\nScore Distribution:")
        print(f"  Max: {max(all_scores):.4f}")
        print(f"  Min: {min(all_scores):.4f}")
        print(f"  Median: {sorted(all_scores)[len(all_scores)//2]:.4f}")
    
    # Quality assessment
    print("\n" + "="*70)
    print("QUALITY ASSESSMENT:")
    print("="*70)
    
    if coverage < 0.5:
        print("✗ POOR: Less than 50% query coverage")
        print("   → Lower similarity_threshold in config")
    elif coverage < 0.8:
        print("⚠  MODERATE: 50-80% query coverage")
        print("   → Consider lowering threshold to 0.20")
    else:
        print("✓ GOOD: >80% query coverage")
    
    if avg_score < 0.3:
        print("✗ POOR: Average score < 0.3 (weak relevance)")
        print("   → Use German queries for German documents")
        print("   → Check document quality/chunking")
    elif avg_score < 0.5:
        print("⚠  MODERATE: Average score 0.3-0.5")
        print("   → Try more specific queries")
    else:
        print("✓ GOOD: Average score > 0.5")
    
    print("="*70 + "\n")
    
    return {
        "coverage": coverage,
        "avg_results_per_query": avg_results_per_query,
        "precision": precision,
        "avg_score": avg_score,
        "all_scores": all_scores,
    }


def main():
    """Run full diagnosis."""
    print("\n" + "="*70)
    print("RAG QUALITY DIAGNOSIS TOOL")
    print("="*70)
    
    # 1. Verify config
    config = load_and_verify_config()
    
    # 2. Initialize components
    print("\nInitializing components...")
    
    embedding_config = config.get("embeddings", {})
    perf_config = config.get("performance", {})
    
    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=perf_config.get("batch_size", 32),
        cache_path=Path(config.get("paths", {}).get("cache", "./cache")) / "embeddings.db",
        device=perf_config.get("device", "cpu"),
    )
    
    storage_config = StorageConfig(
        vector_db_path=Path(config.get("paths", {}).get("vector_db", "./data/vector_db")),
        graph_db_path=Path(config.get("paths", {}).get("graph_db", "./data/knowledge_graph")),
        embedding_dim=embedding_config.get("embedding_dim", 768),
    )
    
    hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)
    
    # FIXED: Try to open the table explicitly
    try:
        if hybrid_store.vector_store.table is None:
            # Try to open existing table
            table_path = storage_config.vector_db_path / "documents.lance"
            if table_path.exists():
                print("✓ Found existing vector store, loading...")
                hybrid_store.vector_store.table = hybrid_store.vector_store.db.open_table("documents")
                print(f"✓ Loaded table with {len(hybrid_store.vector_store.table)} documents")
            else:
                print("\n✗ ERROR: Vector store is empty!")
                print("  Please run 'python main.py' first to ingest documents.")
                print(f"  Looking for: {table_path}")
                return
    except Exception as e:
        print(f"\n✗ ERROR loading vector store: {e}")
        print("  Please run 'python main.py' first to ingest documents.")
        return
    
    print("✓ Components initialized\n")
    
    rag_config = config.get("rag", {})
    retrieval_config = RetrievalConfig(
        mode=RetrievalMode(rag_config.get("retrieval_mode", "hybrid")),
        top_k_vector=config['vector_store']['top_k_vectors'],
        top_k_graph=rag_config.get("top_k_entities", 5),
        vector_weight=rag_config.get("vector_weight", 0.7),
        graph_weight=rag_config.get("graph_weight", 0.3),
        similarity_threshold=config['vector_store']['similarity_threshold'],
    )
    
    retriever = HybridRetriever(
        config=retrieval_config,
        hybrid_store=hybrid_store,
        embeddings=embeddings,
    )
    
    print("✓ Retriever ready\n")
    
    # 3. Run tests
    test_vector_search_raw(hybrid_store, embeddings)
    
    test_different_thresholds(hybrid_store, embeddings)
    
    test_german_queries(retriever)
    
    # 4. Calculate metrics
    test_queries = [
        "Was ist Wissensmanagement?",
        "Welche Methoden werden diskutiert?",
        "Community of Practice",
        "Wie wird Wissen zwischen Generationen geteilt?",
        "Welche Forschungsfragen werden behandelt?",
        "Andragogik Definition",
        "Qualitative Forschungsmethoden",
    ]
    
    metrics = calculate_rag_quality_metrics(retriever, test_queries)
    
    # 5. Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70 + "\n")
    
    if metrics["coverage"] < 0.8:
        print("1. Lower similarity_threshold in config/settings.yaml")
        print("   Current threshold may be too strict")
        print("   Try: 0.20 or 0.15\n")
    
    if metrics["avg_score"] < 0.4:
        print("2. Use more specific German queries")
        print("   Example: 'Community of Practice Definition'\n")
        
        print("3. Consider different embedding model")
        print("   nomic-embed-text is primarily English-trained\n")
    
    if metrics["avg_results_per_query"] < 3:
        print("4. Increase top_k_vectors in config")
        print("   More candidates = better recall\n")
    
    if metrics["precision"] < 0.6:
        print("5. Consider reranking")
        print("   Add cross-encoder for fine-grained relevance\n")
    
    print("="*70)
    
    # 6. Summary for Thesis
    print("\nFOR YOUR THESIS - KEY METRICS:")
    print("="*70)
    print(f"Coverage:          {metrics['coverage']:.1%}")
    print(f"Avg Relevance:     {metrics['avg_score']:.3f}")
    print(f"Results per Query: {metrics['avg_results_per_query']:.1f}")
    print(f"Precision@0.4:     {metrics['precision']:.1%}")
    print("\nThese metrics show your RAG's retrieval quality.")
    print("Include them in your evaluation section!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Suppress verbose logs from other modules
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s"
    )
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()