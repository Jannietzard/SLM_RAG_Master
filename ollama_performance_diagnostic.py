"""
Ollama Performance Diagnostics für Edge-RAG.

Deine Embeddings sind 16x zu langsam (821ms vs erwartet 50ms).
Dieser Test identifiziert den Bottleneck.
"""
import time
import requests


def test_ollama_performance():
    """Test Ollama Embedding Performance."""
    base_url = "http://localhost:11434"
    model = "nomic-embed-text"
    
    print("="*70)
    print("OLLAMA PERFORMANCE DIAGNOSTIC")
    print("="*70)
    
    # Test 1: Single Embedding Latency
    print("\n1. Testing SINGLE embedding latency...")
    test_text = "This is a test sentence for embedding performance."
    
    start = time.time()
    response = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": test_text},
        timeout=10
    )
    single_latency = (time.time() - start) * 1000
    
    print(f"   Single embed: {single_latency:.1f}ms")
    if single_latency > 100:
        print("   ⚠️  WARNING: >100ms is slow! Ollama might be CPU-throttled.")
    else:
        print("   ✓ OK: Single embedding latency acceptable")
    
    # Test 2: Batch Embedding (32 texts)
    print("\n2. Testing BATCH embedding (32 texts)...")
    batch_texts = [f"Test sentence number {i}" for i in range(32)]
    
    start = time.time()
    response = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": batch_texts},
        timeout=30
    )
    batch_latency = (time.time() - start) * 1000
    per_text_latency = batch_latency / 32
    
    print(f"   Batch total: {batch_latency:.1f}ms")
    print(f"   Per text: {per_text_latency:.1f}ms")
    
    if per_text_latency > 100:
        print("   ⚠️  CRITICAL: Batching is not helping! Check Ollama config.")
    elif per_text_latency > 50:
        print("   ⚠️  WARNING: Still slow. Possible CPU bottleneck.")
    else:
        print("   ✓ OK: Batch performance good")
    
    # Test 3: Network Latency
    print("\n3. Testing NETWORK latency (ping)...")
    
    start = time.time()
    response = requests.get(f"{base_url}/api/tags", timeout=5)
    ping_latency = (time.time() - start) * 1000
    
    print(f"   Ping: {ping_latency:.1f}ms")
    if ping_latency > 50:
        print("   ⚠️  WARNING: Network latency high! Is Ollama remote?")
    else:
        print("   ✓ OK: Network latency acceptable")
    
    # Test 4: Model Info
    print("\n4. Checking MODEL configuration...")
    response = requests.post(
        f"{base_url}/api/show",
        json={"name": model},
        timeout=5
    )
    
    if response.status_code == 200:
        model_info = response.json()
        print(f"   Model: {model_info.get('model', 'N/A')}")
        
        # Check for GPU usage
        modelfile = model_info.get('modelfile', '')
        if 'gpu' in modelfile.lower() or 'cuda' in modelfile.lower():
            print("   ✓ GPU detected in modelfile")
        else:
            print("   ⚠️  WARNING: No GPU config detected - running on CPU")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    if single_latency > 200:
        print("1. ⚠️  Ollama is VERY slow. Check:")
        print("   - Is Ollama using CPU-only? (Try: ollama ps)")
        print("   - Is system under load? (Check CPU usage)")
        print("   - Try smaller model: ollama pull all-minilm")
    
    if ping_latency > 50:
        print("2. ⚠️  Network latency high:")
        print("   - Is Ollama remote? Should be localhost!")
        print("   - Check firewall/antivirus blocking")
    
    if per_text_latency > 100:
        print("3. ⚠️  Batch processing not effective:")
        print("   - Ollama version might not support efficient batching")
        print("   - Try: ollama serve with OLLAMA_NUM_PARALLEL=4")
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE:")
    print("  Single: 20-50ms")
    print("  Batch (32): 500-1500ms total (15-50ms per text)")
    print(f"\n  YOUR RESULTS:")
    print(f"  Single: {single_latency:.1f}ms")
    print(f"  Batch (32): {batch_latency:.1f}ms total ({per_text_latency:.1f}ms per text)")
    print("="*70)


if __name__ == "__main__":
    test_ollama_performance()