# test_ollama_api.py
import requests
import time

# Test 1: Einfacher Call (sollte schnell sein)
print("Test 1: Simple prompt...")
start = time.time()
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi3",
        "prompt": "What is 2+2? Answer in one word.",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 50}
    },
    timeout=60
)
print(f"  Time: {time.time()-start:.1f}s")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json().get('response', '')[:100]}")

# Test 2: Mit viel Context (wie der Verifier)
print("\nTest 2: Large context...")
large_context = "This is a test document. " * 500  # ~3000 words
start = time.time()
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi3",
        "prompt": f"Context:\n{large_context}\n\nQuestion: What is this about?\nAnswer:",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 100}
    },
    timeout=120
)
print(f"  Time: {time.time()-start:.1f}s")
print(f"  Status: {response.status_code}")
if response.status_code == 200:
    print(f"  Response: {response.json().get('response', '')[:100]}")
else:
    print(f"  Error: {response.text[:200]}")

print("\nDone!")