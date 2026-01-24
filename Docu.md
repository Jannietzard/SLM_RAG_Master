# GRAPH-AUGMENTED EDGE-RAG SYSTEM
## Comprehensive Technical Documentation

**Version:** 2.1.0  
**Author:** Jan Nietzard  
**Institution:** RWTH Aachen University  
**Last Updated:** 2026-01-13  
**Project Type:** Master Thesis Research Implementation

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Scientific Foundation](#3-scientific-foundation)
4. [Module Reference](#4-module-reference)
5. [Deployment Guide](#5-deployment-guide)
6. [Performance Benchmarks](#6-performance-benchmarks)
7. [Configuration Reference](#7-configuration-reference)
8. [Troubleshooting](#8-troubleshooting)
9. [Research Methodology](#9-research-methodology)
10. [Future Work](#10-future-work)

---



## 1. EXECUTIVE SUMMARY

### 1.1 Project Objective
Development of a locally executed, hybrid Retrieval-Augmented Generation (RAG) system optimized for consumer-grade hardware. The system integrates vector-based semantic search with graph-based structural retrieval to maximize information recall in multi-hop query scenarios. A core objective is to validate the feasibility of "Agentic Compensation" (RQ2) on devices with limited compute resources, ensuring total data privacy through offline execution.

### 1.2 Key Innovations
1.  **Hybrid Graph-Vector Architecture**
    * Combines dense vector embeddings (LanceDB) with structural knowledge graphs (NetworkX, transitioning to KuzuDB).
    * Designed to answer **RQ1** by quantifying the trade-offs between retrieval effectiveness (Recall@10) and system latency on constrained hardware.

2.  **Consumer Hardware Optimization**
    * Tailored for the $\le$ 16GB RAM / $\le$ 8 Core constraint defined in the research methodology.
    * Implements 4-bit model quantization (via Ollama) and memory-mapped storage to fit within the thermal and memory envelopes of standard laptops.

3.  **Neuro-Symbolic Verification (Plan-Retrieve-Verify)**
    * Implements an iterative reasoning loop to compensate for the lower reasoning capabilities of Small Language Models (SLMs).
    * Targets the **RQ2** latency threshold (< 3s end-to-end) for interactive offline utility.

### 1.3 Technical Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Embedding Model** | nomic-embed-text (768-dim) | High-performance open-source model optimized for RAG. |
| **Language Model** | Phi-3 (3.8B, 4-bit quantized) | SOTA performance-per-watt; fits entirely in VRAM/RAM. |
| **Vector Store** | LanceDB | Serverless, disk-based vector search; minimal overhead. |
| **Knowledge Graph** | NetworkX (Migration to KuzuDB planned) | Flexible graph modeling for structural retrieval testing. |
| **Inference Engine** | Ollama | Local API management for quantized GGUF models. |
| **Environment** | Python 3.10+ on Windows 11 | Native execution on consumer OS. |

### 1.4 Experimental Environment & Hardware Constraints

To satisfy the constraints defined in **RQ1** (*"consumer hardware $\le$ 16 GB RAM, $\le$ 8 CPU cores"*), the system is built, tested, and evaluated on a representative consumer laptop. This setup serves as the primary baseline for measuring memory footprint and query latency.

**Primary Testbed (Acer Swift SF314-57G):**

* **Processor:** Intel® Core™ i7-1065G7 (Ice Lake Architecture)
    * *Configuration:* 4 Physical Cores / 8 Logical Threads
    * *Clock Speed:* Base 1.30 GHz (Turbo Boost supported)
* **Memory (RAM):** 16.0 GB Total Physical Memory
    * *Constraint:* High utilization expected; system relies on efficient memory mapping to avoid excessive paging.
* **Operating System:** Microsoft Windows 11 Home (Build 26100)
    * *Execution:* Local environment (No cloud offloading).
* **Storage & I/O:** SSD-based storage (Required for low-latency LanceDB reads).

**Relevance to Research Questions:**
This hardware configuration represents the upper bound of the "resource-constrained" category. If the *Plan-Retrieve-Verify* loops (RQ2) exceed the 3-second latency threshold on this device, the architecture will be deemed "uneconomical for interactive applications" as per the thesis hypothesis.

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EDGE DEVICE                              │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         OLLAMA INFERENCE SERVER                    │    │
│  │  ┌──────────────┐      ┌──────────────┐           │    │
│  │  │ nomic-embed  │      │   Phi-3      │           │    │
│  │  │  (768-dim)   │      │  (3.8B, 4b)  │           │    │
│  │  └──────────────┘      └──────────────┘           │    │
│  │         ↑ HTTP (localhost:11434)                   │    │
│  └─────────┼──────────────────────────────────────────┘    │
│            │                                                │
│  ┌─────────┴──────────────────────────────────────────┐    │
│  │         PYTHON RAG PIPELINE                        │    │
│  │                                                     │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │   INGESTION LAYER                          │  │    │
│  │  │  • PDF Loader (PyPDF2)                     │  │    │
│  │  │  • Text Cleaner (Regex-based)              │  │    │
│  │  │  • Recursive Chunker (512 chars, 128 olap) │  │    │
│  │  │  • Semantic Chunker (Optional, TF-IDF)     │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  │                      ↓                             │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │   EMBEDDING LAYER                          │  │    │
│  │  │  • BatchedOllamaEmbeddings (32 batch)      │  │    │
│  │  │  • SQLite Cache (SHA256 indexed)           │  │    │
│  │  │  • L2 Normalization                        │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  │                      ↓                             │    │
│  │  ┌──────────────────┬────────────────────────┐   │    │
│  │  │ VECTOR STORE     │  KNOWLEDGE GRAPH       │   │    │
│  │  │  (LanceDB)       │  (NetworkX)            │   │    │
│  │  │                  │                        │   │    │
│  │  │ • IVF-FLAT Index │ • DiGraph Structure    │   │    │
│  │  │ • Cosine Metric  │ • BFS Traversal        │   │    │
│  │  │ • Memory-Mapped  │ • GraphML Persistence  │   │    │
│  │  └──────────────────┴────────────────────────┘   │    │
│  │                      ↓                             │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │   HYBRID RETRIEVAL LAYER                   │  │    │
│  │  │  • Vector Retriever (top-k ANN search)     │  │    │
│  │  │  • Graph Retriever (multi-hop traversal)   │  │    │
│  │  │  • Ensemble Fusion (weighted scores)       │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

#### INDEXING PHASE

```
PDF Documents
    ↓
[PDF Loader] → Extract text per page
    ↓
[Text Cleaner] → Remove hyphenation, URLs, page numbers
    ↓
[Chunker] → Split into 512-char chunks with 128-char overlap
    ↓
[Embedding Generator] → Batch processing (32 texts/batch)
    ↓                     Cache lookup (SHA256-indexed)
    ↓                     API call to Ollama (if cache miss)
    ↓
[Hybrid Store]
    ├── [Vector Store] → LanceDB with cosine similarity index
    └── [Graph Store] → NetworkX with document relationships
```

#### RETRIEVAL PHASE

```
User Query
    ↓
[Query Embedding] → Encode via nomic-embed-text
    ↓
┌───────────────┴───────────────┐
│                               │
[Vector Retriever]         [Graph Retriever]
│ 1. ANN Search              │ 1. Entity Extraction
│ 2. Cosine Similarity       │ 2. Node Matching
│ 3. Top-k Selection         │ 3. BFS Traversal
│ 4. Threshold Filter        │ 4. Hop-based Scoring
│                               │
└───────────────┬───────────────┘
                ↓
        [Ensemble Fusion]
        • Weighted Score Combination
        • Deduplication by document_id
        • Final Ranking
                ↓
        [Results] → Sorted by relevance score
```

### 2.3 Component Interaction Matrix

| Component | Depends On | Provides To | Communication Protocol |
|-----------|-----------|-------------|----------------------|
| Ollama Server | None (standalone) | Embedding Layer, LLM Layer | HTTP REST API (localhost) |
| Embedding Layer | Ollama Server | Storage Layer | In-process Python |
| Ingestion Pipeline | Embedding Layer | Storage Layer | In-process Python |
| Vector Store | Embedding Layer | Retrieval Layer | In-process Python |
| Graph Store | Ingestion Pipeline | Retrieval Layer | In-process Python |
| Retrieval Layer | Vector Store, Graph Store | Application Layer | In-process Python |

---

## 3. SCIENTIFIC FOUNDATION

### 3.1 Dense Retrieval Theory

**Mathematical Framework:**

Let D = {d₁, d₂, ..., dₙ} be a document corpus and q be a query.

**Embedding Function:**
```
f: Text → ℝᵈ
where d = 768 (nomic-embed-text dimensionality)
```

**Similarity Metric (Cosine Similarity):**
```
sim(q, dᵢ) = (f(q) · f(dᵢ)) / (||f(q)|| · ||f(dᵢ)||)

For L2-normalized vectors:
sim(q, dᵢ) = f(q) · f(dᵢ)  (dot product)
```

**Retrieval Function:**
```
retrieve(q, k) = arg-top-k_{dᵢ ∈ D} sim(q, dᵢ)
```

**Complexity Analysis:**
- Naive: O(n·d) for n documents
- With IVF-FLAT index: O(√n · d) expected
- With approximations: O(log n · d) for HNSW graphs

### 3.2 Knowledge Graph Retrieval

**Graph Definition:**
```
G = (V, E)
where:
  V = {document_chunks, source_files, entities}
  E = {(u,v,r) | u,v ∈ V, r ∈ RelationTypes}
```

**Relation Types:**
```
RelationTypes = {
  'from_source',    # chunk → source_file
  'references',     # chunk → chunk (citation)
  'mentions',       # chunk → entity
  'part_of',        # entity → parent_entity
  'follows'         # chunk → chunk (sequential)
}
```

**Traversal Algorithm (BFS with hop-based scoring):**
```
function graph_retrieve(query, max_hops):
    entities ← extract_entities(query)
    visited ← {}
    
    for entity in entities:
        node ← match_entity_to_graph(entity)
        queue ← [(node, 0)]
        
        while queue not empty:
            (current, hops) ← queue.dequeue()
            
            if hops < max_hops:
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        visited[neighbor] ← hops + 1
                        queue.enqueue((neighbor, hops + 1))
    
    # Score by inverse hop distance
    scores ← {v: 1 - (h / (max_hops + 1)) for v, h in visited.items()}
    return top_k(scores)
```

### 3.3 Hybrid Fusion Strategy

**Weighted Score Combination:**
```
Let:
  Sᵥ(q, d) = vector similarity score
  Sᵍ(q, d) = graph relevance score
  wᵥ, wᵍ = configurable weights

Hybrid Score:
  S_hybrid(q, d) = (wᵥ · Sᵥ(q, d) + wᵍ · Sᵍ(q, d)) / (wᵥ + wᵍ)
```

**Ablation Study Configurations:**
```
Experiment 1 (Vector-Only):   wᵥ = 1.0, wᵍ = 0.0
Experiment 2 (Graph-Only):    wᵥ = 0.0, wᵍ = 1.0
Experiment 3 (Balanced):      wᵥ = 0.5, wᵍ = 0.5
Experiment 4 (Vector-Heavy):  wᵥ = 0.7, wᵍ = 0.3
Experiment 5 (Graph-Heavy):   wᵥ = 0.3, wᵍ = 0.7
```

### 3.4 Evaluation Metrics

**Precision@k:**
```
P@k = (|{relevant documents in top-k}|) / k
```

**Recall@k:**
```
R@k = (|{relevant documents in top-k}|) / |{all relevant documents}|
```

**Mean Reciprocal Rank (MRR):**
```
MRR = (1/|Q|) · Σ (1 / rank(first_relevant_doc))
```

**Normalized Discounted Cumulative Gain (nDCG@k):**
```
DCG@k = Σᵢ₌₁ᵏ (2^(relᵢ) - 1) / log₂(i + 1)
nDCG@k = DCG@k / IDCG@k
```

---

## 4. MODULE REFERENCE

### 4.1 `main.py` - Orchestration Layer

**Purpose:** Entry point and pipeline orchestration

**Key Classes:**
- `EdgeRAGPipeline`: Main orchestrator implementing pipeline pattern

**Configuration Loading:**
```python
config = load_configuration(Path("./config/settings.yaml"))
```

**Initialization Sequence:**
```python
pipeline.setup()
  ├── initialize_embeddings()      # Ollama connection + test
  ├── initialize_ingestion()       # Document loader setup
  ├── initialize_storage()         # Vector + Graph stores
  └── initialize_retriever()       # Hybrid retriever
```

**Execution Flow:**
```python
documents = pipeline.run_ingestion_pipeline()
pipeline.run_storage_pipeline(documents)
results = pipeline.retrieve(query)
```

**Logging Configuration:**
```python
setup_logging(log_file=Path("./logs/edge_rag.log"))
  • Console: INFO level (UTF-8 encoded)
  • File: DEBUG level (UTF-8 encoded)
  • Format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 4.2 `embeddings.py` - High-Performance Embedding Layer

**Purpose:** Batched embedding generation with persistent caching

**Key Classes:**

#### `BatchedOllamaEmbeddings(Embeddings)`

Implements LangChain's Embeddings interface with performance optimizations.

**Constructor Parameters:**
```python
BatchedOllamaEmbeddings(
    model_name: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    batch_size: int = 32,
    cache_path: Path = Path("./cache/embeddings.db"),
    device: str = "cpu",
    timeout: int = 60
)
```

**Performance Characteristics:**
```
Sequential Processing (baseline):
  1000 texts × 50ms/call = 50,000ms = 50s

Batched Processing (batch_size=32):
  32 batches × 50ms/batch = 1,600ms = 1.6s
  Speedup: 31.25x

With 80% Cache Hit Rate:
  200 API calls (misses) × 50ms = 10,000ms
  800 cache lookups × 0.1ms = 80ms
  Total: 10,080ms = 10.08s
  Effective Speedup: 5x
```

**Methods:**
- `embed_documents(texts: List[str]) -> List[List[float]]`: Batch embedding
- `embed_query(text: str) -> List[float]`: Single query embedding
- `print_metrics()`: Display performance statistics
- `clear_cache()`: Reset cache (for ablation studies)

#### `EmbeddingCache`

SQLite-based persistent cache with content-addressable storage.

**Schema:**
```sql
CREATE TABLE embeddings (
    text_hash TEXT PRIMARY KEY,        -- SHA256(text)
    text_content TEXT NOT NULL,        -- Original text
    embedding BLOB NOT NULL,           -- JSON-encoded vector
    model_name TEXT NOT NULL,          -- Model identifier
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1     -- LRU tracking
);

CREATE INDEX idx_model_hash ON embeddings(model_name, text_hash);
```

**Cache Statistics:**
```python
cache.get_stats()
  → {
      'total_entries': 5432,
      'total_accesses': 12845,
      'size_mb': 127.4
    }
```

### 4.3 `ingestion.py` - Document Processing Pipeline

**Purpose:** PDF ingestion, text cleaning, and chunking

**Key Classes:**

#### `DocumentIngestionPipeline`

**Initialization:**
```python
pipeline = DocumentIngestionPipeline(
    chunking_config: ChunkingConfig,
    document_path: Path,
    enable_cleaning: bool = True
)
```

**Processing Stages:**
```
PDFs → RobustPDFLoader → Clean Text → Chunker → Documents
```

**Text Cleaning Operations:**
1. Remove soft hyphens (U+00AD)
2. Rejoin hyphenated words across line breaks
3. Remove URLs (https?://...)
4. Remove page number artifacts (39/104, Page 23 of 100)
5. Normalize whitespace
6. Fix PDF spacing bugs (German: "V or..." → "Vor...")

**Chunking Modes:**

**Standard (Recursive Character Splitter):**
```python
ChunkingConfig(
    mode="standard",
    chunk_size=512,
    chunk_overlap=128,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Semantic (TF-IDF based):**
```python
ChunkingConfig(
    mode="semantic",
    chunk_size=512,
    chunk_overlap=128,
    min_chunk_size=200
)
```

**Output:**
```python
Document(
    page_content="...",
    metadata={
        'source_file': 'paper.pdf',
        'page': 5,
        'chunk_id': 42,
        'chunk_size': 487,
        'chunking_mode': 'standard'
    }
)
```

#### `RobustPDFLoader`

Per-page error handling for corrupted PDFs.

**Features:**
- Graceful degradation (continues on page failures)
- Empty page detection (< 50 chars)
- Comprehensive logging
- Statistics reporting

### 4.4 `storage.py` - Hybrid Storage Backend

**Purpose:** Unified interface for vector and graph storage

**Key Classes:**

#### `StorageConfig`

```python
@dataclass
class StorageConfig:
    vector_db_path: Path
    graph_db_path: Path
    embedding_dim: int = None           # Auto-detected
    similarity_threshold: float = 0.3
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"     # "cosine" or "l2"
```

**CRITICAL: Distance Metric Specification**

The `distance_metric` parameter is essential for correct similarity computation:

```python
# Cosine Distance (LanceDB definition):
cosine_distance = 1 - cosine_similarity

# Conversion to similarity:
similarity = 1.0 - cosine_distance

# L2 (Euclidean) Distance:
l2_distance = ||q - d||₂

# Conversion to similarity:
similarity = 1.0 / (1.0 + l2_distance)
```

#### `VectorStoreAdapter`

LanceDB wrapper with explicit metric handling.

**Initialization:**
```python
vector_store = VectorStoreAdapter(
    db_path=Path("./data/vector_db"),
    embedding_dim=768,
    normalize_embeddings=True,
    distance_metric="cosine"
)
```

**Critical Implementation Detail:**
```python
# EXPLICIT METRIC SPECIFICATION (prevents L2 default)
results = (
    self.table
    .search(query_embedding)
    .metric(self.distance_metric)  # ← CRITICAL
    .limit(top_k)
    .to_list()
)
```

**Search API:**
```python
results = vector_store.vector_search(
    query_embedding=embedding,
    top_k=10,
    threshold=0.3
)
  → [
      {
        'text': '...',
        'similarity': 0.847,
        'document_id': 'chunk_42',
        'metadata': {...}
      },
      ...
    ]
```

#### `KnowledgeGraphStore`

NetworkX-based graph storage with GraphML persistence.

**Graph Structure:**
```python
G = nx.DiGraph()

# Nodes
G.add_node(
    'chunk_42',
    entity_type='document_chunk',
    source_file='paper.pdf'
)

# Edges
G.add_edge(
    'chunk_42',
    'paper.pdf',
    relation_type='from_source'
)
```

**Traversal API:**
```python
visited = graph_store.graph_traversal(
    start_entity='chunk_42',
    relation_types=['references', 'mentions'],
    max_hops=2
)
  → {'chunk_43': 1, 'chunk_44': 2, 'entity_x': 1}
```

#### `HybridStore`

Facade combining vector and graph stores.

**Unified API:**
```python
hybrid_store.add_documents(documents)
  • Generates embeddings
  • Stores in vector DB
  • Creates graph entities
  • Adds relations

hybrid_store.save()
  • Persists graph to GraphML
  • LanceDB auto-persists

hybrid_store.reset_all()
  • Clears both stores (for ablation studies)
```

### 4.5 `retrieval.py` - Hybrid Retrieval Engine

**Purpose:** Ensemble retrieval combining vector and graph methods

**Key Classes:**

#### `RetrievalConfig`

```python
@dataclass
class RetrievalConfig:
    mode: RetrievalMode              # VECTOR, GRAPH, or HYBRID
    top_k_vector: int
    top_k_graph: int
    vector_weight: float
    graph_weight: float
    similarity_threshold: float
```

#### `HybridRetriever`

**Initialization:**
```python
retriever = HybridRetriever(
    config=RetrievalConfig(
        mode=RetrievalMode.HYBRID,
        top_k_vector=10,
        top_k_graph=5,
        vector_weight=0.7,
        graph_weight=0.3,
        similarity_threshold=0.3
    ),
    hybrid_store=hybrid_store,
    embeddings=embeddings
)
```

**Retrieval Modes:**

**VECTOR Mode:**
```python
config.mode = RetrievalMode.VECTOR
config.vector_weight = 1.0
config.graph_weight = 0.0

# Uses only vector similarity search
results = retriever.retrieve("What is RAG?")
```

**GRAPH Mode:**
```python
config.mode = RetrievalMode.GRAPH
config.vector_weight = 0.0
config.graph_weight = 1.0

# Uses only graph traversal
results = retriever.retrieve("What is RAG?")
```

**HYBRID Mode:**
```python
config.mode = RetrievalMode.HYBRID
config.vector_weight = 0.7
config.graph_weight = 0.3

# Combines both methods
results = retriever.retrieve("What is RAG?")
```

**Result Format:**
```python
RetrievalResult(
    text="...",
    relevance_score=0.847,
    document_id="chunk_42",
    source_file="paper.pdf",
    retrieval_method="hybrid",
    metadata={
        'vector_score': 0.82,
        'graph_score': 0.91,
        'page': 5
    }
)
```

### 4.6 `semantic_chunking.py` - Advanced Chunking (Optional)

**Purpose:** TF-IDF based semantic boundary detection

**Key Components:**

**AutomaticQualityFilter:**
- Lexical diversity (Type-Token Ratio)
- Information density (Shannon entropy)
- Transcript pattern detection
- No hardcoded keywords (language-agnostic)

**TFIDFScorer:**
```python
scorer = TFIDFScorer()
scorer.analyze_corpus(chunks)
importance = scorer.calculate_chunk_importance(chunk_index)
```

**SemanticChunker:**
```python
chunker = SemanticChunker(
    max_chunk_size=1024,
    min_chunk_size=200,
    overlap=128
)
chunks = chunker.chunk_document(document)
```

**Quality Metrics Added:**
```python
metadata = {
    'importance_score': 0.73,      # TF-IDF based
    'lexical_diversity': 0.62,     # Type-Token Ratio
    'heading_level': 2,            # Detected from patterns
    'is_header': False
}
```

### 4.7 `preprocessing.py` - Content Quality Filtering

**Purpose:** Filter low-quality chunks before indexing

**Key Functions:**

**Simple Filter:**
```python
if should_skip_chunk(text, min_chars=100, min_diversity=0.25):
    continue  # Skip this chunk
```

**Comprehensive Analysis:**
```python
analyzer = ChunkQualityAnalyzer(
    min_chars=100,
    min_words=15,
    min_diversity=0.3,
    min_entropy=2.0,
    max_url_count=2,
    max_citation_count=3
)

result = analyzer.analyze(text)

if not result.should_keep:
    logger.info(f"Filtered: {result.rejection_reason}")
else:
    process_chunk(text)
```

**Filter Reasons:**
- `TOO_SHORT`: < 100 characters
- `TOO_FEW_WORDS`: < 15 words
- `BIBLIOGRAPHY`: Detected as reference section
- `TRANSCRIPT`: Detected as interview transcript
- `LOW_DIVERSITY`: Type-Token Ratio < 0.25
- `LOW_ENTROPY`: Shannon entropy < 2.0 bits/word
- `HIGH_URL_DENSITY`: > 2 URLs
- `HIGH_CITATION_DENSITY`: > 3 academic citations

---

## 5. DEPLOYMENT GUIDE

### 5.1 Local Deployment (Development)

**Prerequisites:**
```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    curl \
    git

# Ollama installation
curl -fsSL https://ollama.com/install.sh | sh
```

**Setup Steps:**
```bash
# 1. Clone repository
git clone <repository-url>
cd edge-rag

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start Ollama server
ollama serve &

# 5. Pull models
ollama pull nomic-embed-text
ollama pull phi3

# 6. Verify installation
python -c "import ollama; print(ollama.list())"

# 7. Place PDFs
mkdir -p data/documents
cp your_pdfs/*.pdf data/documents/

# 8. Run pipeline
python main.py
```

### 5.2 Docker Deployment (Production)

**Single-Container Deployment:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/documents data/vector_db logs cache

# Startup script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
```

```bash
# docker/entrypoint.sh
#!/bin/bash
set -e

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done

echo "Pulling models..."
ollama pull nomic-embed-text
ollama pull phi3

echo "Starting RAG system..."
python main.py

kill $OLLAMA_PID
```

**Build and Run:**
```bash
# Build image
docker build -t edge-rag:latest .

# Run container
docker run -d \
    --name edge-rag \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/cache:/app/cache \
    -p 8080:8080 \
    --memory=6g \
    --cpus=4 \
    edge-rag:latest

# View logs
docker logs -f edge-rag
```

### 5.3 Docker Compose Deployment (Multi-Container)

```yaml
# docker-compose.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: edge-rag-ollama
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '3.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - edge-rag-net

  rag-system:
    build: .
    container_name: