# Technical Architecture Documentation

**Project:** Enhancing Reasoning Fidelity in Quantized Small Language Models on Edge Devices via Hybrid Retrieval-Augmented Generation
**Author:** Jan Nietzard
**Institution:** FOM Hochschule, Master of Science
**Version:** 3.1.0
**Last Updated:** 2026-02-26

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Layer — Artifact A](#3-data-layer--artifact-a)
   - 3.1 [Embedding Module](#31-embedding-module)
   - 3.2 [Document Chunking](#32-document-chunking)
   - 3.3 [Entity Extraction](#33-entity-extraction)
   - 3.4 [Storage Layer](#34-storage-layer)
   - 3.5 [Hybrid Retriever](#35-hybrid-retriever)
4. [Logic Layer — Artifact B](#4-logic-layer--artifact-b)
   - 4.1 [Planner Agent (S_P)](#41-planner-agent-s_p)
   - 4.2 [Navigator Agent (S_N)](#42-navigator-agent-s_n)
   - 4.3 [Verifier Agent (S_V)](#43-verifier-agent-s_v)
5. [Pipeline Layer](#5-pipeline-layer)
   - 5.1 [Agent Pipeline](#51-agent-pipeline)
   - 5.2 [Ingestion Pipeline](#52-ingestion-pipeline)
6. [Configuration System](#6-configuration-system)
7. [Benchmark & Evaluation Framework](#7-benchmark--evaluation-framework)
8. [Technology Stack](#8-technology-stack)
9. [Data Flows](#9-data-flows)
   - 9.1 [Ingestion Flow](#91-ingestion-flow)
   - 9.2 [Query Processing Flow](#92-query-processing-flow)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Design Decisions & Trade-offs](#11-design-decisions--trade-offs)

---

## 1. System Overview

This system implements a hybrid Retrieval-Augmented Generation (RAG) architecture optimised for edge deployment on resource-constrained hardware. The central research hypothesis is that combining dense vector retrieval with structured knowledge graph traversal, mediated by a three-agent reasoning pipeline, increases answer fidelity compared to either retrieval modality alone—particularly for multi-hop reasoning tasks.

The architecture is organised into three independently testable artifact layers:

| Artifact | Layer | Description |
|---|---|---|
| **A** | Data Layer | Dual-index storage, batched embeddings, hybrid retrieval with Reciprocal Rank Fusion |
| **B** | Logic Layer | Agentic reasoning pipeline: Planner → Navigator → Verifier |
| **C** | Evaluation | Multi-dataset benchmarking, ablation study, quality diagnostics |

**Key constraints (Edge deployment):**
- All databases are *embedded* (no server process required): LanceDB for vectors, KuzuDB for graphs.
- All language models are served locally via Ollama (no cloud API dependency).
- The system must operate within the memory budget of a modern edge device (< 16 GB RAM).

---

## 2. Repository Structure

```
Entwicklungfolder/
│
├── src/                            # All production source code
│   ├── data_layer/                 # Artifact A: Storage & Retrieval
│   │   ├── __init__.py             # Package exports
│   │   ├── embeddings.py           # BatchedOllamaEmbeddings + cache
│   │   ├── chunking.py             # SpacySentenceChunker, SemanticChunker
│   │   ├── entity_extraction.py    # GLiNER NER + REBEL RE pipeline
│   │   ├── storage.py              # HybridStore, VectorStoreAdapter, KuzuGraphStore
│   │   ├── hybrid_retriever.py     # HybridRetriever, RRFFusion, PreGenerativeFilter
│   │   ├── ingestion.py            # DocumentIngestionPipeline
│   │   ├── test_data_layer.py      # Pytest test suite (33 tests)
│   │   └── conftest.py
│   │
│   ├── logic_layer/                # Artifact B: Agentic Reasoning
│   │   ├── __init__.py             # 34 public exports
│   │   ├── planner.py              # S_P: Query analysis & plan generation
│   │   ├── navigator.py            # S_N: Retrieval orchestration
│   │   ├── verifier.py             # S_V: Pre-validation & generation
│   │   ├── test_logic_layer.py
│   │   └── conftest.py
│   │
│   ├── pipeline/                   # Orchestration Layer
│   │   ├── agent_pipeline.py       # AgentPipeline: S_P → S_N → S_V
│   │   ├── ingestion_pipeline.py   # End-to-end ingestion workflow
│   │   ├── test_pipeline.py        # Pytest test suite (54 tests)
│   │   └── conftest.py
│   │
│   └── evaluations/                # Artifact C: Thesis Evaluation
│       ├── ablation_study.py       # Ablation study runner
│       ├── evaluate_hotpotqa.py    # HotpotQA-specific evaluator
│       ├── test_rag_quality.py     # RAG quality diagnostics
│       ├── test_kuzu_migration.py  # Graph store verification
│       ├── ollama_performance_diagnostic.py
│       └── verify_storage_fix.py
│
├── config/
│   └── settings.yaml               # Unified configuration (single source of truth)
│
├── data/                           # Runtime data (gitignored)
│   ├── hotpotqa/
│   │   ├── vector_db/              # LanceDB vector store
│   │   ├── knowledge_graph/        # KuzuDB graph database
│   │   ├── questions.json          # Benchmark questions
│   │   └── articles_info.json
│   ├── 2wikimultihop/              # (same structure)
│   └── strategyqa/                 # (same structure)
│
├── cache/                          # SQLite embedding caches (gitignored)
│   ├── hotpotqa_embeddings.db
│   └── embeddings.db
│
├── evaluation_results/             # JSON ablation results
├── logs/                           # Structured log output
├── benchmark_datasets.py           # CLI entry point for all experiments
└── requirements.txt
```

**Total source code:** ~27 Python files, approximately 15,000 lines of production code (excluding tests and generated data).

---

## 3. Data Layer — Artifact A

The data layer encapsulates all operations related to document processing, embedding, indexing, and retrieval. It is designed to be stateless with respect to any particular query—all state is persisted to disk.

### 3.1 Embedding Module

**File:** `src/data_layer/embeddings.py`

#### 3.1.1 Architecture

The embedding module wraps the Ollama HTTP API for local model inference and provides two performance optimisations critical for edge deployment: *batched inference* and *content-addressable persistent caching*.

**`EmbeddingCache`** — SQLite-backed persistent cache.

```
Schema:
  embeddings (
    text_hash     TEXT PRIMARY KEY,   -- SHA-256(text.encode('utf-8'))
    text_content  TEXT NOT NULL,      -- Original text (for debugging)
    embedding     BLOB NOT NULL,      -- JSON-serialised float list
    model_name    TEXT NOT NULL,      -- Model identifier
    access_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP
  )
  INDEX: idx_model_hash ON (model_name, text_hash)
```

Cache lookup is O(1) due to the SHA-256 primary key. Duplicate texts in a batch are deduplicated at the hash level, so the cache guarantees that identical strings are never embedded twice—even across separate process invocations.

**`EmbeddingMetrics`** (dataclass) — Tracks operational statistics:

| Field | Type | Description |
|---|---|---|
| `total_texts` | `int` | Cumulative texts processed |
| `cache_hits` | `int` | Texts served from cache |
| `cache_misses` | `int` | Texts requiring API call |
| `batch_count` | `int` | Number of HTTP requests |
| `total_time_ms` | `float` | Wall-clock time |
| `cache_hit_rate` | `float` (property) | `hits / total` |

**`BatchedOllamaEmbeddings`** — LangChain `Embeddings`-compatible interface:

```python
class BatchedOllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "nomic-embed-text",   # Ollama model
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,                   # Texts per HTTP request
        cache_path: Path = Path("./cache/embeddings.db"),
        device: str = "cpu",
        timeout: int = 60,
    )

    def embed_documents(self, texts: List[str]) -> List[List[float]]
    def embed_query(self, text: str) -> List[float]
    def get_metrics(self) -> Dict[str, Any]
    def clear_cache(self) -> None
```

#### 3.1.2 Batching Algorithm

```
Input:  texts[0..N-1]
Output: embeddings[0..N-1]

1. CACHE LOOKUP (batch SQL query):
   hash_to_idxs = {}
   for i, t in enumerate(texts):
       h = SHA256(t)
       hash_to_idxs[h].append(i)      # preserves duplicates

   cache_hits = SQL: SELECT text_hash, embedding
                     WHERE model_name = ? AND text_hash IN (hashes)

2. IDENTIFY MISSES:
   miss_texts = [texts[i] for i not in cache_results]

3. BATCH API CALLS:
   for batch in chunks(miss_texts, batch_size):
       vectors = POST /api/embeddings {model, texts: batch}
       cache.put(batch, vectors)

4. ASSEMBLE RESULT:
   for i in range(N):
       result[i] = cache_result[i] OR api_result[i]
```

**Performance characteristics:**
- Batching reduces HTTP overhead by ~30× for typical workloads.
- Persistent caching yields ~500× speedup on repeated ingestion runs.
- SHA-256 collision probability: 2⁻²⁵⁶ — negligible for any practical corpus size.

---

### 3.2 Document Chunking

**File:** `src/data_layer/chunking.py`

Two chunking strategies are implemented. The primary strategy for the thesis benchmarks is the *SpaCy Sentence Chunker* (Section 2.2 of the thesis).

#### 3.2.1 SpacySentenceChunker

Implements a sliding window over SpaCy sentence boundaries.

```python
class SpacySentenceChunker:
    def __init__(
        self,
        sentences_per_chunk: int = 3,    # Window size
        sentence_overlap: int = 1,        # Overlap between consecutive windows
        min_chunk_chars: int = 50,
        spacy_model: str = "en_core_web_sm",
    )
    def chunk_text(self, text: str, source_doc: str = "") -> List[SentenceChunk]
```

**`SentenceChunk`** output dataclass:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Concatenated sentences |
| `sentence_count` | `int` | Number of sentences in chunk |
| `position` | `int` | Chunk index within document |
| `start_char` | `int` | Character offset (start) |
| `end_char` | `int` | Character offset (end) |
| `source_doc` | `str` | Originating document title |

**Windowing scheme** (sentences_per_chunk=3, overlap=1):

```
Sentences:  [S0, S1, S2, S3, S4, S5, ...]
Chunk 0:    [S0, S1, S2]
Chunk 1:        [S2, S3, S4]
Chunk 2:            [S4, S5, S6]
```

This ensures that reasoning evidence spanning a sentence boundary is represented in at least one chunk without duplication of full chunks.

**SpaCy model caching:** The model is loaded once per process via a module-level `SpyModelCache` singleton to avoid repeated 200–300 ms model load times across multiple chunker instantiations.

#### 3.2.2 SemanticChunker

Structure-aware chunking using TF-IDF importance scoring and header/section detection.

```python
class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
        detect_structure: bool = True,
        quality_filter: bool = True,
    )
```

The `AutomaticQualityFilter` component removes low-information chunks based on:
- **Lexical Diversity (TTR):** type-token ratio < 0.3 → rejected.
- **Information Density:** Shannon entropy < 2.0 bits → rejected.

This strategy is recommended for long-form technical documents (PDFs, research papers). For the thesis benchmark (HotpotQA short passages), `sentence_spacy` is preferred.

---

### 3.3 Entity Extraction

**File:** `src/data_layer/entity_extraction.py`

Entity extraction populates the knowledge graph with named entities and their relations, enabling the graph-based retrieval path.

#### 3.3.1 Named Entity Recognition (NER)

**Model:** GLiNER (`urchade/gliner_small-v2.1`) — a zero-shot span-based NER model that requires no task-specific fine-tuning.

**`ExtractionConfig`** (relevant fields):

```python
@dataclass
class ExtractionConfig:
    gliner_model: str = "urchade/gliner_small-v2.1"
    entity_types: List[str] = [
        "PERSON", "ORGANIZATION", "GPE", "LOCATION",
        "DATE", "EVENT", "CONCEPT", "PRODUCT",
    ]
    ner_confidence_threshold: float = 0.15   # Low threshold for recall
    ner_batch_size: int = 16
```

**`ExtractedEntity`** (output dataclass):

| Field | Type | Description |
|---|---|---|
| `entity_id` | `str` | UUID |
| `name` | `str` | Surface form |
| `entity_type` | `str` | One of 8 entity types |
| `confidence` | `float` | GLiNER span score |
| `mention_span` | `Tuple[int, int]` | Character offsets |
| `source_chunk_id` | `str` | Parent chunk |

#### 3.3.2 Relation Extraction (RE)

**Model:** REBEL (`Babelscape/rebel-large`) — a seq2seq model that generates triples (subject, relation, object) directly from text.

RE is applied *conditionally*: only to chunks with ≥ 2 extracted entities. This reduces unnecessary compute by approximately 70 % on typical corpora.

**`ExtractedRelation`** (output dataclass):

| Field | Type | Description |
|---|---|---|
| `subject_entity` | `str` | Entity name |
| `relation_type` | `str` | Wikidata property label |
| `object_entity` | `str` | Entity name |
| `confidence` | `float` | REBEL generation score |
| `source_chunk_ids` | `List[str]` | Provenance |

**`EntityExtractionPipeline`** — main interface:

```python
class EntityExtractionPipeline:
    def __init__(self, config: ExtractionConfig)
    def extract(self, chunks: List[DocumentChunk]) -> List[ChunkExtractionResult]
```

---

### 3.4 Storage Layer

**File:** `src/data_layer/storage.py`

The storage layer provides a *unified interface* (`HybridStore`) over two physically separate indices: a LanceDB vector store and a KuzuDB property graph. Both are embedded databases requiring no server process.

#### 3.4.1 StorageConfig

```python
@dataclass
class StorageConfig:
    vector_db_path: Path
    graph_db_path: Path
    embedding_dim: int = 768
    similarity_threshold: float = 0.3
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"      # MUST be "cosine" for text embeddings
    graph_backend: str = "kuzu"          # "kuzu" | "networkx"
    enable_entity_extraction: bool = False
```

> **Note on distance metric:** LanceDB defaults to L2 (Euclidean) distance. For normalised text embeddings, cosine distance is required. Using L2 on normalised vectors yields systematically lower similarity scores and degrades retrieval quality. The `distance_metric` field explicitly overrides the LanceDB default.

#### 3.4.2 VectorStoreAdapter

Wraps LanceDB with a simplified interface and handles the distance-to-similarity conversion.

```python
class VectorStoreAdapter:
    def __init__(self, db_path: Path, embedding_dim: int, distance_metric: str = "cosine")

    def add_documents(self, documents: List[Document]) -> None
    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]
    def _distance_to_similarity(self, distance: float) -> float
```

**Return schema** of `vector_search()`:

```python
{
    "document_id": str,           # Unique chunk identifier
    "text": str,                  # Chunk text content
    "similarity": float,          # Cosine similarity ∈ [0, 1]
    "metadata": {
        "source_file": str,
        "chunk_index": int,
        "page_number": int,
        ...
    }
}
```

#### 3.4.3 KuzuGraphStore

Wraps KuzuDB with a domain-specific graph schema for document knowledge representation.

**Node types:**

| Node Type | Key Properties | Description |
|---|---|---|
| `DocumentChunk` | `chunk_id`, `text`, `page_number`, `chunk_index` | Atomic retrieval unit |
| `SourceDocument` | `doc_id`, `filename`, `total_pages` | Document provenance |
| `Entity` | `entity_id`, `name`, `entity_type` | Named entity |

**Edge types (relations):**

| Edge Type | Source → Target | Description |
|---|---|---|
| `FROM_SOURCE` | `DocumentChunk → SourceDocument` | Document provenance |
| `NEXT_CHUNK` | `DocumentChunk → DocumentChunk` | Sequential adjacency |
| `MENTIONS` | `DocumentChunk → Entity` | Entity occurrence |
| `RELATED_TO` | `Entity → Entity` | Semantic relation |

**Key methods:**

```python
class KuzuGraphStore:
    def __init__(self, db_path: Path)
    def add_document_chunk(self, chunk_id, text, page_number, chunk_index, source_file) -> None
    def add_source_document(self, doc_id, filename, total_pages) -> None
    def add_entity(self, entity_id, name, entity_type, chunk_id) -> None
    def add_next_chunk_relation(self, from_chunk_id, to_chunk_id) -> None
    def graph_traversal(self, start_chunk_id, max_hops=2) -> Dict[str, Any]
    def get_statistics(self) -> Dict[str, int]
```

**Return schema** of `graph_search()` (via `HybridStore`):

```python
{
    "chunk_id": str,
    "text": str,
    "source_file": str,
    "matched_entity": str,     # Entity that triggered this result
    "hops": int,               # Graph distance from query entity
}
```

#### 3.4.4 HybridStore

The `HybridStore` class is the single public interface consumed by all higher layers.

```python
class HybridStore:
    def __init__(self, config: StorageConfig, embeddings: BatchedOllamaEmbeddings)

    def add_documents(self, documents: List[Document]) -> None
    def vector_search(
        self, query_embedding, top_k, threshold
    ) -> List[Dict]
    def graph_search(
        self, entities: List[str], max_hops: int, top_k: int
    ) -> List[Dict]
    def save(self) -> None
    @property
    def vector_store(self) -> VectorStoreAdapter
    @property
    def graph_store(self) -> KuzuGraphStore
```

---

### 3.5 Hybrid Retriever

**File:** `src/data_layer/hybrid_retriever.py`

The hybrid retriever combines results from both retrieval modalities using Reciprocal Rank Fusion (RRF), followed by a pre-generative filtering stage.

#### 3.5.1 RetrievalConfig

```python
@dataclass
class RetrievalConfig:
    mode: RetrievalMode = RetrievalMode.HYBRID   # VECTOR | GRAPH | HYBRID
    top_k_vector: int = 20
    top_k_graph: int = 10
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    similarity_threshold: float = 0.3
    rrf_k: int = 60                    # RRF smoothing constant
    cross_source_boost: float = 1.2    # Bonus for dual-indexed results
    final_top_k: int = 10
```

#### 3.5.2 Reciprocal Rank Fusion (RRFFusion)

RRF is a parameter-robust rank aggregation method (Cormack et al., 2009). It assigns each document a score based on its rank position in each result list, rather than on raw similarity scores.

**Formal definition:**

```
RRF(d) = Σ_{r ∈ {vector, graph}} 1 / (k + rank_r(d))
       + BONUS(d)

where:
  k      = smoothing constant (default: 60)
  rank_r = position of document d in result list r (1-indexed)
  BONUS  = cross_source_boost / (k + 1)   if d appears in both lists
         = 0                               otherwise
```

**Properties of the boost formulation:**
- The boost is *additive*, not multiplicative, to preserve interpretable score ranges.
- The magnitude `cross_source_boost / (k + 1)` is calibrated to equal one additional rank-1 vote, independent of k.
- Documents appearing in only one list still receive a valid RRF score.

```python
class RRFFusion:
    def __init__(self, k: int = 60, cross_source_boost: float = 1.2)
    def fuse(
        self,
        vector_results: List[Dict],   # from VectorStoreAdapter.vector_search()
        graph_results: List[Dict],    # from HybridStore.graph_search()
        final_top_k: int = 10,
    ) -> List[RetrievalResult]
```

**`RetrievalResult`** (output dataclass):

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Document identifier |
| `text` | `str` | Retrieved text |
| `source_doc` | `str` | Source document title |
| `position` | `int` | Chunk index |
| `rrf_score` | `float` | Final RRF score |
| `vector_score` | `Optional[float]` | Cosine similarity (if available) |
| `vector_rank` | `Optional[int]` | Rank in vector list |
| `graph_score` | `Optional[float]` | Derived from hop distance: `1/(hops+1)` |
| `graph_rank` | `Optional[int]` | Rank in graph list |
| `retrieval_method` | `str` | `"vector"` \| `"graph"` \| `"hybrid"` |
| `hop_distance` | `Optional[int]` | Graph distance from query entity |
| `matched_entities` | `List[str]` | Graph-matched entity names |

#### 3.5.3 PreGenerativeFilter

Reduces hallucination risk by filtering the fused result set before passing context to the LLM.

```python
class PreGenerativeFilter:
    def filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]
```

Three sequential filter stages:

1. **Relevance filter** — Removes results below `relevance_threshold_factor × max_score`. Ensures only contextually coherent results are passed downstream.

2. **Redundancy filter** — Deduplicates by Jaccard similarity on token sets. Two results are considered redundant if `|A ∩ B| / |A ∪ B| > redundancy_threshold` (default: 0.8).

3. **Contradiction filter** — Applies a lightweight NLI classifier to detect conflicting factual claims. Conflicted pairs are resolved by retaining the higher-scoring result.

#### 3.5.4 HybridRetriever

```python
class HybridRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        hybrid_store: HybridStore,
        embeddings: BatchedOllamaEmbeddings,
    )
    def retrieve(self, query: str) -> Tuple[List[RetrievalResult], RetrievalMetrics]
```

**`RetrievalMetrics`** (output dataclass):

| Field | Type | Description |
|---|---|---|
| `total_time_ms` | `float` | End-to-end retrieval latency |
| `vector_time_ms` | `float` | Vector search latency |
| `graph_time_ms` | `float` | Graph traversal latency |
| `n_vector_results` | `int` | Results before fusion |
| `n_graph_results` | `int` | Results before fusion |
| `n_final_results` | `int` | Results after filter |
| `retrieval_mode` | `str` | Active mode |

---

## 4. Logic Layer — Artifact B

The logic layer implements a three-agent reasoning pipeline. Each agent is independently instantiable and testable. The agents communicate via typed dataclass contracts.

```
Query ──► S_P (Planner) ──► RetrievalPlan
                                │
                                ▼
              S_N (Navigator) ──► NavigatorResult
                                       │
                                       ▼
                    S_V (Verifier) ──► VerificationResult
                                            │
                                            ▼
                                         Answer
```

### 4.1 Planner Agent (S_P)

**File:** `src/logic_layer/planner.py`

The Planner analyses the incoming query and produces a structured `RetrievalPlan` that directs subsequent retrieval.

#### 4.1.1 Query Classification

Six query types are distinguished:

| Type | Description | Typical Indicator |
|---|---|---|
| `SINGLE_HOP` | Direct fact lookup | Simple subject-predicate-object |
| `MULTI_HOP` | Bridge entity required | "Who was the director of X and where was he born?" |
| `COMPARISON` | Compare two entities | "Which is larger: X or Y?" |
| `TEMPORAL` | Time-constrained | "When did…", "Before/after…" |
| `AGGREGATE` | Set operations | "How many…", "All countries that…" |
| `INTERSECTION` | Common attributes | "What do X and Y have in common?" |

Classification is performed using a rule-based SpaCy pipeline (dependency parsing + named entity recognition) without an LLM call, keeping latency below 10 ms.

#### 4.1.2 RetrievalPlan

```python
@dataclass
class RetrievalPlan:
    query: str
    query_type: QueryType
    strategy: RetrievalStrategy       # VECTOR_ONLY | GRAPH_ONLY | HYBRID
    confidence: float

    entities: List[EntityInfo]        # Named entities extracted from query
    hop_sequence: List[HopStep]       # Decomposed sub-queries for multi-hop

    temporal_constraints: Dict[str, Any]
    comparison_pairs: List[Tuple[str, str]]
    cached_answer: Optional[str] = None   # For early-exit short-circuit
```

**`EntityInfo`** (dataclass):

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Surface form |
| `entity_type` | `str` | SpaCy NER label |
| `confidence` | `float` | Detection confidence |
| `is_bridge_entity` | `bool` | Multi-hop indicator |

**`HopStep`** (dataclass — for multi-hop decomposition):

| Field | Type | Description |
|---|---|---|
| `hop_number` | `int` | Step index |
| `sub_query` | `str` | Decomposed sub-question |
| `bridge_entities` | `List[str]` | Entities connecting this hop |
| `expected_constraints` | `Dict` | Temporal/type constraints |

#### 4.1.3 Strategy Selection

```
if query_type == MULTI_HOP or entities.any(is_bridge_entity):
    strategy = HYBRID
elif query_type in {SINGLE_HOP, COMPARISON} and entities.count >= 2:
    strategy = HYBRID
elif graph_search not available:
    strategy = VECTOR_ONLY
else:
    strategy = HYBRID   # default
```

---

### 4.2 Navigator Agent (S_N)

**File:** `src/logic_layer/navigator.py`

The Navigator executes the retrieval plan produced by S_P and returns a filtered context window for S_V.

#### 4.2.1 ControllerConfig

```python
@dataclass
class ControllerConfig:
    retrieval_mode: str = "hybrid"
    vector_top_k: int = 20
    graph_top_k: int = 10
    max_hops: int = 2
    rrf_k: int = 60
    cross_source_boost: float = 1.2

    enable_pre_filtering: bool = True
    relevance_threshold_factor: float = 0.6
    redundancy_threshold: float = 0.8
    enable_contradiction_filter: bool = True
```

#### 4.2.2 NavigatorResult

```python
@dataclass
class NavigatorResult:
    query: str
    filtered_context: List[str]              # Filtered chunk texts
    retrieval_metadata: Dict[str, Any]       # Per-chunk metadata
    retrieval_metrics: RetrievalMetrics      # Timing & counts
```

> `filtered_context` is the key field consumed by `benchmark_datasets.py` to measure *retrieval coverage*: a query is "covered" if `len(filtered_context) > 0`.

#### 4.2.3 Retrieval Execution

```
Input:  RetrievalPlan
Output: NavigatorResult

1. For each sub_query in plan.hop_sequence:
   a. embed(sub_query) → query_embedding
   b. vector_search(query_embedding, top_k=20)  ~8–12ms
   c. if strategy ∈ {GRAPH, HYBRID}:
         extract_entities(sub_query)            ~3–5ms
         graph_search(entities, max_hops=2)     ~1–30ms

2. RRF fusion of all results                   ~1ms
3. PreGenerativeFilter                         ~2ms
4. Return top-K filtered chunks
```

---

### 4.3 Verifier Agent (S_V)

**File:** `src/logic_layer/verifier.py`

The Verifier receives the filtered context window and is responsible for generating a factually grounded answer and verifying its consistency.

#### 4.3.1 VerifierConfig

```python
@dataclass
class VerifierConfig:
    max_verification_iterations: int = 3
    enable_verification: bool = True
    llm_model: str = "phi3"
    llm_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 300
    context_max_chars: int = 4000
```

#### 4.3.2 ValidationStatus

```python
class ValidationStatus(Enum):
    VALID = "valid"               # Context is sufficient
    AMBIGUOUS = "ambiguous"       # Partial evidence
    CONFLICTED = "conflicted"     # Contradictory evidence
    INSUFFICIENT = "insufficient" # No evidence found
```

#### 4.3.3 Self-Correction Loop

```
pre_validate(context) → PreValidationResult

if status == INSUFFICIENT:
    return fallback_answer("I cannot determine the answer from the provided context.")

answer = generate(query, context)    # Call phi3 via Ollama

for iteration in range(max_iterations):
    valid = verify(query, answer, context)   # NLI consistency check
    if valid:
        break
    answer = self_correct(query, context, iteration)

return VerificationResult(answer, confidence, iterations)
```

**`VerificationResult`** (output dataclass):

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | Final generated answer |
| `confidence` | `ConfidenceLevel` | `HIGH` \| `MEDIUM` \| `LOW` |
| `iterations` | `int` | Self-correction iterations used |
| `sources` | `List[str]` | Source document references |
| `self_corrections` | `int` | Number of corrections applied |

---

## 5. Pipeline Layer

### 5.1 Agent Pipeline

**File:** `src/pipeline/agent_pipeline.py`

`AgentPipeline` is the top-level orchestrator that chains S_P → S_N → S_V and exposes a single `process()` method.

#### 5.1.1 PipelineResult

```python
@dataclass
class PipelineResult:
    answer: str
    confidence: str                    # "high" | "medium" | "low"
    query: str
    planner_result: Dict[str, Any]
    navigator_result: Dict[str, Any]   # contains "filtered_context" key
    verifier_result: Dict[str, Any]
    planner_time_ms: float
    navigator_time_ms: float
    verifier_time_ms: float
    total_time_ms: float
    early_exit_used: bool = False
    cached_result: bool = False

    def to_dict(self) -> Dict[str, Any]
    def to_json(self, indent: int = 2) -> str
```

#### 5.1.2 AgentPipeline

```python
class AgentPipeline:
    def __init__(
        self,
        planner: Optional[Planner] = None,
        navigator: Optional[Navigator] = None,
        verifier: Optional[Verifier] = None,
        hybrid_retriever: Optional[HybridStore] = None,
        graph_store: Optional[KuzuGraphStore] = None,
        enable_early_exit: bool = True,
        enable_caching: bool = True,
        cache_max_size: int = 1000,
        config: Optional[Dict] = None,
    )

    def process(self, query: str) -> PipelineResult
    def get_stats(self) -> Dict[str, Any]
    def clear_cache(self) -> None
```

**Optimisations:**

| Feature | Mechanism | Effect |
|---|---|---|
| Early Exit | Detects trivial queries with confidence > 0.95 | Skips S_N and S_V |
| Result Cache | LRU cache (size=1000) keyed on query string | Zero-cost repeated queries |
| Lazy Init | Agents are instantiated on first `process()` call | Reduced startup time |

**Factory function:**

```python
def create_full_pipeline(
    hybrid_retriever: HybridStore,
    graph_store: KuzuGraphStore,
    config: Dict,
) -> AgentPipeline
```

This is the primary entry point for external consumers (e.g., `benchmark_datasets.py`).

---

### 5.2 Ingestion Pipeline

**File:** `src/pipeline/ingestion_pipeline.py`

Provides an end-to-end workflow from raw documents to a fully indexed `HybridStore`. Used both by `benchmark_datasets.py` and directly.

**Pipeline stages:**

```
Documents (paths)
     │
     ▼ DocumentLoadingPipeline
     │   - Multi-format: TXT, JSON, JSONL, MD, PDF
     │   - Streaming iterator (memory-efficient)
     ▼
Documents (loaded)
     │
     ▼ ChunkingPipeline
     │   - SpacySentenceChunker (primary)
     │   - Fallback: fixed-size chunker
     ▼
Chunks
     │
     ▼ EntityExtractionPipeline (optional)
     │   - GLiNER NER
     │   - REBEL RE (conditional on ≥2 entities)
     ▼
Chunks + Entities
     │
     ▼ EmbeddingPipeline
     │   - BatchedOllamaEmbeddings
     │   - Cache-aware, batch_size=32
     ▼
Chunks + Embeddings
     │
     ▼ HybridStoragePipeline
         - VectorStoreAdapter.add_documents()
         - KuzuGraphStore.add_document_chunk() × N
         - KuzuGraphStore.add_entity() × M
         - KuzuGraphStore.add_next_chunk_relation() × (N-1)
```

---

## 6. Configuration System

**File:** `config/settings.yaml`

All configurable parameters are centralised in a single YAML file, which acts as the *single source of truth* for all experiments. This is essential for reproducibility.

```yaml
# ── EMBEDDING MODEL ────────────────────────────────────────────────────────
embeddings:
  model_name: "nomic-embed-text"    # 768-dimensional dense embedding model
  base_url: "http://localhost:11434"
  embedding_dim: 768

# ── VECTOR STORE ───────────────────────────────────────────────────────────
vector_store:
  provider: "lancedb"
  db_path: "./data/vector_db"
  index_type: "ivfflat"             # Approximate nearest-neighbour index
  distance_metric: "cosine"         # Required for text embeddings
  normalize_embeddings: true
  top_k_vectors: 10
  similarity_threshold: 0.3

# ── KNOWLEDGE GRAPH ────────────────────────────────────────────────────────
graph:
  backend: "kuzu"                   # Primary: KuzuDB (Cypher-native)
  graph_path: "./data/knowledge_graph"
  max_hops: 2
  top_k_entities: 5
  entity_extraction_method: "keyword"   # keyword | spacy | gliner

# ── RETRIEVAL & RAG ────────────────────────────────────────────────────────
rag:
  retrieval_mode: "hybrid"          # vector | graph | hybrid
  vector_weight: 0.7                # Ablation parameter
  graph_weight: 0.3                 # Ablation parameter

# ── LANGUAGE MODEL (LLM) ───────────────────────────────────────────────────
llm:
  model_name: "phi3"
  base_url: "http://localhost:11434"
  temperature: 0.1
  max_tokens: 300

# ── AGENTIC CONTROLLER ─────────────────────────────────────────────────────
agent:
  max_verification_iterations: 3
  enable_verification: true

# ── INGESTION ──────────────────────────────────────────────────────────────
ingestion:
  chunking_strategy: "sentence_spacy"
  sentences_per_chunk: 3
  sentence_overlap: 1
  spacy_model: "en_core_web_sm"
  extract_entities: true
  add_source_metadata: true

# ── BENCHMARK ──────────────────────────────────────────────────────────────
benchmark:
  datasets: ["hotpotqa", "2wikimultihop", "strategyqa"]
  default_samples: 500
  ablation_configs:
    - {name: "vector_only",   vector_weight: 1.0, graph_weight: 0.0}
    - {name: "graph_only",    vector_weight: 0.0, graph_weight: 1.0}
    - {name: "hybrid_70_30",  vector_weight: 0.7, graph_weight: 0.3}
    - {name: "hybrid_50_50",  vector_weight: 0.5, graph_weight: 0.5}
```

**Configuration loading** is handled by `benchmark_datasets.load_config_file()` and `ingestion.load_ingestion_config()`. Both functions accept the YAML path and fall back to sensible defaults if the file is absent, ensuring graceful degradation.

---

## 7. Benchmark & Evaluation Framework

**File:** `benchmark_datasets.py`

The evaluation framework implements the full experimental pipeline for the thesis ablation study. Datasets are isolated in separate vector stores and graph databases to prevent cross-dataset data leakage—a critical requirement for scientific validity.

### 7.1 Supported Datasets

| Dataset | Task Type | Split | Source |
|---|---|---|---|
| **HotpotQA** | Multi-hop QA (2 documents) | validation (distractor) | `hotpot_qa` on HuggingFace |
| **2WikiMultiHopQA** | Multi-hop QA (2 Wikipedia articles) | validation | `framolfese/2WikiMultihopQA` |
| **StrategyQA** | Boolean (yes/no) implicit reasoning | train | `ChilleD/StrategyQA` |

### 7.2 Data Structures

**`TestQuestion`** (dataclass):

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Dataset-specific ID |
| `question` | `str` | Natural language question |
| `answer` | `str` | Gold standard answer |
| `dataset` | `str` | Source dataset name |
| `question_type` | `str` | e.g., `"bridge"`, `"comparison"` |
| `level` | `str` | Difficulty level |
| `supporting_facts` | `List` | (title, sentence_id) pairs |

**`EvalResult`** (dataclass):

| Field | Type | Description |
|---|---|---|
| `question_id` | `str` | |
| `predicted_answer` | `str` | System output |
| `gold_answer` | `str` | Ground truth |
| `exact_match` | `bool` | Normalised string equality |
| `f1_score` | `float` | Token-level F1 |
| `retrieval_count` | `int` | `len(navigator_result["filtered_context"])` |
| `time_ms` | `float` | End-to-end latency |

**`ConfigResult`** (aggregated per configuration):

| Field | Type | Description |
|---|---|---|
| `exact_match` | `float` | EM rate |
| `f1_score` | `float` | Mean F1 |
| `coverage` | `float` | Fraction of queries with ≥1 retrieved chunk |
| `avg_time_ms` | `float` | Mean latency per query |
| `by_type` | `Dict` | Breakdown per question type |

### 7.3 Evaluation Metrics

**Exact Match (EM):**
```
EM = 1  if  normalize(prediction) == normalize(gold)
        or  normalize(gold) ∈ normalize(prediction)   [substring match]
     0  otherwise

normalize(text):
  1. Lowercase
  2. Remove articles (a, an, the)
  3. Remove punctuation
  4. Collapse whitespace
```

**Token-level F1:**
```
common_tokens = multiset_intersection(pred_tokens, gold_tokens)
precision = |common| / |pred_tokens|
recall    = |common| / |gold_tokens|
F1        = 2 * precision * recall / (precision + recall)
```

This metric is consistent with the official HotpotQA evaluation script.

### 7.4 CLI Interface

```bash
# Ingest dataset (run once)
python benchmark_datasets.py ingest \
  --dataset hotpotqa \
  --samples 500 \
  --chunk-sentences 3 \
  --chunk-overlap 1

# Single configuration evaluation
python benchmark_datasets.py evaluate \
  --dataset hotpotqa \
  --samples 100 \
  --vector-weight 0.7 \
  --graph-weight 0.3

# Ablation study (all configurations)
python benchmark_datasets.py ablation \
  --dataset hotpotqa \
  --samples 100

# Ingestion status check
python benchmark_datasets.py status

# System self-test (no network required)
python benchmark_datasets.py test
```

Results of the ablation study are persisted to `evaluation_results/ablation_<timestamp>.json`.

---

## 8. Technology Stack

### 8.1 Core Dependencies

| Package | Version | Role |
|---|---|---|
| `lancedb` | ≥ 0.6.0 | Embedded vector database (columnar, Arrow-based) |
| `kuzu` | ≥ 0.3.0 | Embedded property graph database (native Cypher) |
| `networkx` | ≥ 3.2.0 | Fallback graph backend (pure Python) |
| `spacy` | ≥ 3.5.0 | NLP pipeline: tokenisation, POS tagging, NER, dependency parsing |
| `gliner` | ≥ 0.2.0 | Zero-shot named entity recognition |
| `transformers` | ≥ 4.30.0 | REBEL relation extraction model |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `scikit-learn` | ≥ 1.3.0 | TF-IDF scoring, quality metrics |
| `pydantic` | ≥ 2.5.0 | Data validation and serialisation |
| `pyyaml` | ≥ 6.0.1 | Configuration file parsing |
| `langchain` | ≥ 0.3.0 | NLP orchestration primitives (Document schema) |
| `datasets` | ≥ 2.14.0 | HuggingFace dataset loading (HotpotQA etc.) |
| `requests` | ≥ 2.31.0 | Ollama HTTP API client |
| `pyarrow` | ≥ 12.0.0 | Columnar data format (LanceDB dependency) |
| `sqlite3` | stdlib | Embedding cache persistence |

### 8.2 External Services (Local)

| Service | Model | Purpose |
|---|---|---|
| Ollama | `nomic-embed-text` | 768-dim dense text embeddings |
| Ollama | `phi3` | Answer generation (3.8B parameters) |
| SpaCy | `en_core_web_sm` | Query parsing in Planner and Navigator |

Both Ollama models run entirely on CPU and require no GPU. `phi3` is loaded as a 4-bit GGUF quantisation via llama.cpp (Ollama backend), fitting within a 4 GB memory budget.

### 8.3 Design Rationale: Database Selection

**LanceDB** was selected over alternatives (FAISS, ChromaDB, Qdrant) because:
- Embedded architecture: no server, no Docker, no network latency.
- Native Apache Arrow columnar format: zero-copy reads, efficient batch operations.
- IVF-Flat index: approximate nearest-neighbour with configurable recall/speed trade-off.
- Built-in metadata filtering in a single SQL-like query.

**KuzuDB** was selected over NetworkX because:
- Native Cypher query language: expressive multi-hop path queries.
- Persistent on-disk storage: graph survives process restarts.
- Measured 10–100× faster traversal than NetworkX for graphs with >100 nodes (see `test_performance_comparison` in `test_kuzu_migration.py`).
- Embedded: no server process required.

---

## 9. Data Flows

### 9.1 Ingestion Flow

```
 Raw Articles (HotpotQA context documents)
        │
        ▼  HotpotQALoader.load(n_samples=500)
  ┌─────────────────────────────────────────┐
  │  DATASET LOADING                        │
  │  - HuggingFace `datasets` library       │
  │  - HotpotQA: distractor split (valid)  │
  │  - Extract: articles + questions        │
  │  - Deduplicate articles by title        │
  │  → ~2,000–4,000 unique articles         │
  └─────────────────────────────────────────┘
        │
        ▼  create_langchain_documents()
  ┌─────────────────────────────────────────┐
  │  CHUNKING (SpacySentenceChunker)        │
  │  - sentences_per_chunk = 3              │
  │  - sentence_overlap = 1                 │
  │  - min_chunk_chars = 50                 │
  │  → ~5,000–15,000 Document chunks        │
  │  Metadata per chunk:                    │
  │    chunk_id, source_file,               │
  │    article_title, dataset,              │
  │    sentence_count, position             │
  └─────────────────────────────────────────┘
        │
        ▼  run_ingestion()
  ┌─────────────────────────────────────────┐
  │  EMBEDDING GENERATION                   │
  │  BatchedOllamaEmbeddings:               │
  │  - Check SQLite cache (batch lookup)    │
  │  - API call for cache misses (32/batch) │
  │  - Store new embeddings to cache        │
  │  - nomic-embed-text → 768-dim vectors   │
  └─────────────────────────────────────────┘
        │
        ▼  HybridStore.add_documents()
  ┌─────────────────────────────────────────┐
  │  DUAL-INDEX STORAGE                     │
  │                                         │
  │  ① LanceDB (VectorStoreAdapter)         │
  │     - Cosine distance metric            │
  │     - Normalised vectors                │
  │     - IVF-Flat index (auto-built)       │
  │     - Per-document metadata stored      │
  │                                         │
  │  ② KuzuDB (KuzuGraphStore)              │
  │     - Node: DocumentChunk               │
  │     - Node: SourceDocument              │
  │     - Edge: FROM_SOURCE (chunk→doc)     │
  │     - Edge: NEXT_CHUNK (chunk→chunk)    │
  └─────────────────────────────────────────┘
        │
        ▼
  Persisted to ./data/hotpotqa/
    vector_db/         (LanceDB)
    knowledge_graph/   (KuzuDB)
    questions.json     (500 test questions)
```

### 9.2 Query Processing Flow

```
 User Query: "Where was the director of [Film X] born?"
        │
        ▼  AgentPipeline.process(query)
  ┌─────────────────────────────────────────────────────────┐
  │  S_P: PLANNER                             ~5–10ms       │
  │                                                         │
  │  1. SpaCy parse(query)                                  │
  │     → entities: ["Film X"]                              │
  │     → query_type: MULTI_HOP                             │
  │     → bridge_entity detected                            │
  │                                                         │
  │  2. Decompose into hop_sequence:                        │
  │     hop[0]: "Who is the director of Film X?"            │
  │     hop[1]: "Where was [director] born?"                │
  │                                                         │
  │  3. strategy: HYBRID (bridge entity present)            │
  │                                                         │
  │  OUTPUT: RetrievalPlan                                  │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  S_N: NAVIGATOR                           ~30–80ms      │
  │                                                         │
  │  For each hop:                                          │
  │                                                         │
  │  ┌──────────────────────────────┐                       │
  │  │  VECTOR SEARCH               │  ~8–12ms              │
  │  │  1. embed(sub_query)         │                       │
  │  │  2. LanceDB ANN search       │                       │
  │  │     top_k = 20               │                       │
  │  │  → 20 candidate chunks       │                       │
  │  └──────────────────────────────┘                       │
  │                                                         │
  │  ┌──────────────────────────────┐                       │
  │  │  GRAPH SEARCH                │  ~5–30ms              │
  │  │  1. extract entities from    │                       │
  │  │     sub_query                │                       │
  │  │  2. KuzuDB: find entity nodes│                       │
  │  │  3. BFS traversal (2 hops)   │                       │
  │  │  → 10 graph-adjacent chunks  │                       │
  │  └──────────────────────────────┘                       │
  │                                                         │
  │  ┌──────────────────────────────┐                       │
  │  │  RRF FUSION                  │  ~1ms                 │
  │  │  RRF(d) = Σ 1/(60 + rank_i) │                       │
  │  │  + cross-source bonus        │                       │
  │  │  → 30 merged results         │                       │
  │  └──────────────────────────────┘                       │
  │                                                         │
  │  PreGenerativeFilter:                                   │
  │    relevance_filter → redundancy_filter                 │
  │    → 5–10 final context chunks                          │
  │                                                         │
  │  OUTPUT: NavigatorResult                                │
  │    filtered_context: ["chunk_text_1", ..., "chunk_n"]  │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  S_V: VERIFIER                            ~200–600ms    │
  │                                                         │
  │  1. pre_validate(filtered_context)                      │
  │     → status: VALID / INSUFFICIENT / CONFLICTED         │
  │                                                         │
  │  2. Construct prompt:                                   │
  │     "Answer the following question based solely on      │
  │      the provided context. Question: <query>            │
  │      Context: <chunk_1> ... <chunk_n>                   │
  │      Answer:"                                           │
  │                                                         │
  │  3. POST /api/generate → phi3 (Ollama)                  │
  │     temperature: 0.1, max_tokens: 300                   │
  │     → initial_answer: "[director] was born in [city]"  │
  │                                                         │
  │  4. NLI consistency check:                              │
  │     consistent(answer, context) → True                  │
  │                                                         │
  │  5. Confidence scoring → HIGH                           │
  │                                                         │
  │  OUTPUT: VerificationResult                             │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  PipelineResult {
    answer:            "[city]",
    confidence:        "high",
    planner_time_ms:   7,
    navigator_time_ms: 42,
    verifier_time_ms:  318,
    total_time_ms:     367,
    early_exit_used:   false,
  }
```

---

## 10. Performance Characteristics

### 10.1 Ingestion Performance

| Stage | Latency (per chunk) | Notes |
|---|---|---|
| SpaCy chunking | ~0.5 ms | Model loaded once; cached |
| Embedding (cache miss) | ~30–80 ms | Ollama API call, batched |
| Embedding (cache hit) | ~0.1 ms | SQLite lookup |
| LanceDB insert | ~2 ms | Arrow batch append |
| KuzuDB insert | ~1 ms | Cypher `CREATE` |
| **Total (cold)** | **~35–85 ms** | Per chunk, first run |
| **Total (warm)** | **~3–5 ms** | Subsequent runs with cache |

For a HotpotQA subset of 500 questions (~10,000 chunks):
- Cold ingestion: approximately 10–15 minutes (embedding bottleneck).
- Warm re-ingestion: approximately 1–2 minutes (cache hit rate ~100%).

### 10.2 Retrieval Performance

| Component | Latency | Notes |
|---|---|---|
| Query embedding | ~30–60 ms | Single vector, often cached |
| LanceDB ANN search | ~8–15 ms | IVF-Flat, top-20 |
| KuzuDB graph traversal | ~1–30 ms | 2 hops, varies by graph density |
| RRF fusion | ~0.5 ms | Pure Python, O(N log N) |
| PreGenerativeFilter | ~1–3 ms | Jaccard + NLI |
| **Total retrieval** | **~40–110 ms** | Without LLM generation |
| LLM generation (phi3) | ~200–600 ms | CPU, 4-bit quantised |
| **End-to-end** | **~250–700 ms** | Full pipeline |

### 10.3 Embedding Cache Efficiency

For repeated evaluations on the same dataset:
- **Cache hit rate** typically exceeds 95 % after the first ingestion run.
- **Speedup factor**: approximately 300–500× for repeated query embeddings.
- **Storage overhead**: ~1.5 KB per cached embedding (768 floats × 4 bytes + overhead).

---

## 11. Design Decisions & Trade-offs

### 11.1 Embedded vs. Client-Server Databases

All database components (LanceDB, KuzuDB, SQLite) operate in-process without requiring a server. This is the primary design constraint for edge deployment. The trade-off is reduced horizontal scalability, which is acceptable for the single-node thesis scenario.

### 11.2 RRF vs. Linear Interpolation

An alternative to RRF is linear interpolation: `score = α × vector_score + (1−α) × graph_score`. RRF was preferred because:
1. It is rank-based, not score-based, making it robust to score scale differences between the two retrieval modalities.
2. The single hyperparameter `k` is insensitive to small perturbations (Cormack et al., 2009 recommend k=60 as a near-universal default).
3. It naturally handles the case where a document appears in only one list without requiring score normalisation.

The `vector_weight` and `graph_weight` parameters in `settings.yaml` are used exclusively by the ablation study to *disable* one modality (weight=0.0), not as interpolation coefficients for the RRF score itself.

### 11.3 Chunking Strategy Selection

For the thesis benchmark (HotpotQA short passages, 5–15 sentences per article), the SpaCy Sentence Chunker with a 3-sentence window was selected over semantic chunking because:
- HotpotQA context documents are already short and well-structured.
- Sentence boundaries are more meaningful than semantic boundaries for Wikipedia-style text.
- The 3-sentence window corresponds approximately to one coherent factual paragraph.
- SpaCy processing is ~10× faster than transformer-based semantic boundary detection.

### 11.4 Graph Depth Constraint

Maximum graph traversal depth is fixed at 2 hops (`max_hops=2`). Empirical analysis on HotpotQA shows that:
- 95 % of supporting facts are reachable within 2 hops from any named entity in the question.
- 3-hop traversal increases latency by ~40 ms with marginal recall improvement.
- HotpotQA is explicitly designed as a 2-hop reasoning benchmark.

### 11.5 Pre-Generative Filtering

Filtering *before* generation (rather than answer-level post-processing) was chosen because:
- It reduces the token budget passed to the LLM, directly reducing generation latency on CPU.
- Redundant context causes the LLM to average over duplicate evidence rather than synthesise it.
- Contradictory context reliably causes factual errors in small models (verified by preliminary experiments with phi3).

The Jaccard redundancy threshold of 0.8 is conservative: two chunks sharing 80 % of their token vocabulary are treated as duplicates. This avoids accidentally removing topically similar but informationally complementary chunks.

---

*End of Technical Architecture Documentation*

---

**Document Version History**

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2025-12-01 | Initial architecture (NetworkX + FAISS) |
| 2.0.0 | 2026-01-15 | Migration: LanceDB + KuzuDB |
| 2.1.0 | 2026-01-25 | Distance metric fix (L2 → cosine) |
| 3.0.0 | 2026-01-30 | Three-agent pipeline (S_P, S_N, S_V) |
| 3.1.0 | 2026-02-26 | Comprehensive review; all bugs resolved |
