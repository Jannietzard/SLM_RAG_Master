# Technical Architecture Documentation

**Project:** Enhancing Reasoning Fidelity in Quantized Small Language Models on Edge Devices via Hybrid Retrieval-Augmented Generation
**Author:** Jan Nietzard
**Institution:** FOM Hochschule, Master of Science
**Version:** 4.0.0
**Last Updated:** 2026-04-23

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
12. [Changes & Design Decisions](#12-changes--design-decisions)

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
│   │   ├── hybrid_retriever.py     # HybridRetriever, RRFFusion (vector+graph+BM25), ImprovedQueryEntityExtractor
│   │   ├── ingestion.py            # DocumentIngestionPipeline
│   │   └── conftest.py             # Adds project root to sys.path
│   │
│   ├── logic_layer/                # Artifact B: Agentic Reasoning
│   │   ├── __init__.py             # 34 public exports
│   │   ├── planner.py              # S_P: Query analysis & plan generation
│   │   ├── navigator.py            # S_N: Retrieval orchestration
│   │   ├── verifier.py             # S_V: Pre-validation & generation
│   │   ├── controller.py           # AgenticController: LangGraph state machine
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
│       └── ollama_performance_diagnostic.py
│
├── config/
│   └── settings.yaml               # Unified configuration (single source of truth)
│
├── data/                           # Runtime data (gitignored)
│   ├── hotpotqa/
│   │   ├── vector/                 # LanceDB vector store (directory)
│   │   ├── graph/                  # KuzuDB graph database (directory)
│   │   │   └── extraction_results.json  # GLiNER+REBEL extraction output (Colab)
│   │   ├── chunks_export.json      # All chunks with text + metadata (Phase 1 output)
│   │   ├── questions.json          # Benchmark questions with supporting_facts
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
├── diagnose.py                     # Layer-by-layer diagnostic (graph quality, vector scores)
├── diagnose_verbose.py             # Trace-mode diagnostic with per-call timing
├── diagnose_ingestion.py           # Ingestion consistency checker (chunks→LanceDB→KuzuDB→rank)
├── local_importingestion.py        # Phase 3 import: chunks_export.json + extraction_results.json → stores
├── test_system/
│   ├── graph_3d.py                 # Graph visualisation: matplotlib PNG + pyvis HTML
│   ├── graph_inspect.py            # Graph schema and statistics inspector
│   ├── test_chunking.py            # 29 chunking unit tests
│   ├── test_embeddings.py          # 34 embedding unit tests
│   ├── test_ner_quality.py         # NER quality evaluation (20-sentence gold set)
│   └── graph_preview.html          # Generated interactive visualisation (gitignored)
└── requirements.txt
```

**Total source code:** ~35 Python files, approximately 18,000 lines of production code (excluding tests and generated data).

**Decoupled ingestion architecture (3-phase):**

| Phase | Tool | Description |
|---|---|---|
| Phase 1 | `benchmark_datasets.py ingest` | Chunk articles → `chunks_export.json` |
| Phase 2 | Google Colab (GPU) | GLiNER + REBEL extraction → `extraction_results.json` |
| Phase 3 | `local_importingestion.py` | Import Phase 1+2 outputs → LanceDB + KuzuDB |

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

**SpaCy model caching:** The model is loaded once per process via a module-level `SpacyModelCache` singleton to avoid repeated 200–300 ms model load times across multiple chunker instantiations.

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
        # Lowercase natural-language labels — better zero-shot performance with GLiNER
        "person", "organization", "city", "country", "state",
        "location", "film", "movie", "album", "work of art",
        "landmark", "event", "award",
    ]
    ner_confidence_threshold: float = 0.15  # Recall-optimised for HotpotQA
    ner_batch_size: int = 16
    rebel_max_input_length: int = 256
    rebel_max_output_length: int = 256
    rebel_num_beams: int = 3
    device: str = "cpu"
```

> **Change history:** Prior to 2026-04-01, `ner_confidence_threshold=0.15` and entity types were UPPERCASE (`PERSON`, `ORGANIZATION`, etc.), causing hub contamination ("American" 898 chunks, "He" 737). From 2026-04-02: threshold raised to 0.5, lowercase types — unique entities reduced from 36,996 → 23,858 (−35%). Subsequently revised to `0.15` again for recall (Section 12.6): the entity types and normalisation now prevent noise better than the threshold alone.

**`ExtractedEntity`** (output dataclass):

| Field | Type | Description |
|---|---|---|
| `entity_id` | `str` | SHA-256 24-char hex ID |
| `name` | `str` | Surface form (normalised) |
| `entity_type` | `str` | One of 13 entity types |
| `confidence` | `float` | GLiNER span score |
| `mention_span` | `Tuple[int, int]` | Character offsets |
| `source_chunk_id` | `str` | Parent chunk |

> **Note on `to_dict()`:** The serialised key is `"entity_type"` (not `"type"`). Entity IDs were migrated from UUID (12-char MD5) to SHA-256 24-char hex in April 2026 — existing KuzuDB stores built before this change use the old format and require re-ingestion.

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

> **Note on `to_dict()`:** Serialised keys are `"subject_entity"`, `"relation_type"`, `"object_entity"` (not `"subject"`, `"relation"`, `"object"`). REBEL's `extract_batch` was renamed `extract_sequential` to reflect the seq2seq model's limitation.

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
    vector_top_k: int = 20          # canonical name (not top_k_vector)
    graph_top_k: int = 10           # canonical name (not top_k_graph)
    similarity_threshold: float = 0.3
    rrf_k: int = 60                 # RRF smoothing constant (Cormack et al., 2009)
    cross_source_boost: float = 1.2 # Bonus for dual-indexed results
    final_top_k: int = 10
    enable_bm25: bool = True        # 3rd RRF path; settings.yaml: rag.enable_bm25 (§12.29)
    bm25_top_k: int = 20            # BM25 candidates per query
    query_ner_confidence: float = 0.15       # Read from settings.yaml
    query_entity_types: List[str] = field(default_factory=list)  # Read from settings.yaml
    gliner_model_name: str = "urchade/gliner_small-v2.1"
```

> The legacy `vector_weight` / `graph_weight` fields were removed in the
> 2026-05-06 cleanup audit (see §12.30). They were never read by production
> code; weighted-fusion ablation is now done by switching `mode` to
> `RetrievalMode.VECTOR` or `RetrievalMode.GRAPH`.

#### 3.5.2 Reciprocal Rank Fusion (RRFFusion)

RRF is a parameter-robust rank aggregation method (Cormack et al., 2009). It assigns each document a score based on its rank position in each result list, rather than on raw similarity scores.

**Formal definition (extended for BM25, §12.29):**

```
RRF(d) = Σ_{r ∈ {vector, graph, bm25}} 1 / (k + rank_r(d))
       + BONUS(d)

where:
  k      = smoothing constant (default: 60)
  rank_r = position of document d in result list r (1-indexed)
  BONUS  = cross_source_boost / (k + 1)
           for every pair of lists in which d appears
         = 0 otherwise
```

**Properties of the boost formulation:**
- The boost is *additive*, not multiplicative, to preserve interpretable score ranges.
- The magnitude `cross_source_boost / (k + 1)` is calibrated to equal one additional rank-1 vote, independent of k.
- Documents appearing in only one list still receive a valid RRF score.
- A chunk surfacing in 2+ retrieval lanes (e.g. dense vector AND BM25) gets one bonus per pair, capping at one bonus per chunk regardless of how many pairs match.

```python
class RRFFusion:
    def __init__(self, k: int = 60, cross_source_boost: float = 1.2)
    def fuse(
        self,
        vector_results: List[Dict],            # from VectorStoreAdapter.vector_search()
        graph_results: List[Dict],             # from HybridStore.graph_search()
        final_top_k: int = 10,
        bm25_results: Optional[List[Dict]] = None,  # 3rd path (§12.29)
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
| `retrieval_method` | `str` | `"vector"` \| `"graph"` \| `"bm25"` \| `"hybrid"` |
| `hop_distance` | `Optional[int]` | Graph distance from query entity |
| `matched_entities` | `List[str]` | Graph-matched entity names |
| `bm25_score` | `Optional[float]` | BM25 score normalised to [0, 1] (§12.29) |
| `bm25_rank` | `Optional[int]` | Rank in BM25 list (§12.29) |

#### 3.5.3 Pre-Generative Filtering — Owned by the Navigator (§4.2.4)

> **Note (2026-05-06 cleanup audit, §12.30):** The standalone
> `PreGenerativeFilter` class formerly defined in `hybrid_retriever.py` was
> removed. Production code never invoked it; it duplicated logic the
> Navigator (S_N) had already owned since v3.4. Pre-generative filtering is
> a **Logic Layer** concern, not a Data Layer concern, and now lives in
> [`src/logic_layer/navigator.py`](src/logic_layer/navigator.py).

The Navigator's `navigate()` method runs the full filter chain after RRF
fusion. The chain has **six sequential stages** plus an optional
**Stage 2.5 cross-encoder reranker** (§12.29):

1. **Relevance filter** — Removes results below
   `relevance_threshold_factor × max_score`. Default factor lowered from
   `0.85 → 0.6` in the 2026-05-06 audit (§12.30) to stop the score-
   compression of `nomic-embed-text` from evicting bridge chunks.

2. **Redundancy filter** — Deduplicates by Jaccard similarity on token sets.
   Two chunks are redundant if `|A ∩ B| / |A ∪ B| > redundancy_threshold`
   (default `0.8`).

3. **Contradiction filter** — Numeric heuristic: two chunks discussing the
   same topic (high word overlap) but with strongly differing numeric values
   are treated as conflicting; the lower-RRF chunk is dropped. Threshold
   defaults: `overlap ≥ 0.3`, `ratio ≥ 2.0`, `min_value ≥ 100`. The
   `min_value` floor was raised from `10 → 100` in §12.25 to prevent
   day-of-month vs. year false positives.

4. **Entity-overlap pruning** — Drops chunks whose entity set is a strict
   subset of a higher-ranked chunk's entity set. Original contribution.

5. **Entity-mention filter** — Each chunk's text must literally contain at
   least one query entity. Multi-word entities match on the full phrase OR
   on individual tokens ≥ 8 characters (audit fix §12.28 Fix A); single-
   token entities require length ≥ 5. **Safety fallback:** if every chunk
   would be removed, all are returned (context is never empty).

6. **Context-shrinkage filter** — Per-chunk truncation to
   `max_chars_per_doc` (default raised `500 → 800` in §12.30) with
   sentence-boundary awareness.

**Stage 2.5 — Cross-encoder reranker (optional, §12.29):**
`cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top-K fused chunks by
`(query, chunk)` relevance and re-orders before Stage 3. Lazy-loaded;
22 MB model, ~30 ms per pair on CPU. Toggle via
`navigator.enable_reranker` (default `true` since the 2026-05-06 audit).

#### 3.5.4 ImprovedQueryEntityExtractor

Extracts entities from the query consistently with the ingestion-time extractor.

```python
class ImprovedQueryEntityExtractor:
    def __init__(self, gliner_model=None, spacy_model: str = "en_core_web_sm")
    def extract(self, query: str, confidence_threshold: float = 0.2) -> List[str]
```

**Key design decisions (as of 2026-04-02):**
- Loads `gliner_small-v2.1` **independently** via `_get_gliner_model()` when no model is supplied (process-level cache).
- Uses **the same entity types as ingestion**: `["person", "organization", "city", "country", "film", "movie", "work of art", "event"]`.
- **Threshold 0.2** (not 0.5): queries are short sentences, so GLiNER scores are systematically lower than on longer chunk texts.
- **Module-level cache** `_GLINER_MODEL_CACHE`: the model is loaded at most once per process (7.5 s cold start, <1 ms thereafter).

> **Critical bug fixed on 2026-04-02:** the default
> `StorageConfig(enable_entity_extraction=False)` caused
> `ImprovedQueryEntityExtractor` to receive `gliner_model=None` and silently
> fall back to SpaCy — even though GLiNER was installed. Symptom: the
> extracted entity was `"Were Scott Derrickson"` instead of
> `"Scott Derrickson"`. Fixed by independent loading in `_load_gliner()`.
> See section 12.7.

#### 3.5.5 HybridRetriever

```python
class HybridRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        hybrid_store: HybridStore,
        embeddings: BatchedOllamaEmbeddings,
    )
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        entity_hints: Optional[List[str]] = None,   # Pre-extracted entities (bypass GLiNER)
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]

    # Internal lanes (called from retrieve())
    def _build_bm25_index(self) -> None     # lazy, lazy-built once per process
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]
```

> **`entity_hints` parameter:** When provided, the retriever uses these entity names directly for graph search instead of running GLiNER on the sub-query. This is critical for iterative multi-hop retrieval (Section 12.18), where bridge entities are discovered at runtime and passed to the next hop. Without this, GLiNER fails on short sub-query fragments (3–5 words) that lack context for reliable NER.

> **Three-lane retrieval (§12.29):** `retrieve()` runs vector ANN, graph
> traversal, and BM25 in parallel and feeds all three result lists into
> `RRFFusion.fuse()`. The BM25 index is built lazily from the LanceDB
> chunks DataFrame on first use (~9 k chunks → ~50 ms one-shot cost).
> A chunk surfacing in 2+ lanes earns the cross-source boost defined in
> §3.5.2.

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
    original_query: str              # The unmodified input query
    query_type: QueryType
    strategy: RetrievalStrategy      # VECTOR_ONLY | GRAPH_ONLY | HYBRID
    confidence: float

    entities: List[EntityInfo]       # Named entities extracted from query (SpaCy)
    hop_sequence: List[HopStep]      # Decomposed sub-queries for multi-hop
    sub_queries: List[str]           # Flat list of sub-query strings

    temporal_constraints: Dict[str, Any]
    comparison_pairs: List[Tuple[str, str]]
    estimated_hops: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached_answer: Optional[str] = None   # For early-exit short-circuit
```

**`EntityInfo`** (dataclass):

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Surface form of the entity |
| `label` | `str` | SpaCy NER label (e.g. `PERSON`, `ORG`) |
| `confidence` | `float` | Detection confidence |
| `start_char` | `int` | Start character offset in query |
| `end_char` | `int` | End character offset in query |
| `is_bridge` | `bool` | Multi-hop indicator |

**`HopStep`** (dataclass — for multi-hop decomposition):

| Field | Type | Description |
|---|---|---|
| `step_id` | `int` | Step index (0-based) |
| `sub_query` | `str` | Decomposed sub-question |
| `target_entities` | `List[str]` | Entity names targeted in this hop |
| `depends_on` | `List[int]` | `step_id`s that must complete before this step |
| `is_bridge` | `bool` | Whether this step retrieves a bridge entity |

#### 4.1.3 Strategy Selection

```
if query_type == MULTI_HOP or entities.any(is_bridge=True):
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
    # LLM Settings — emergency fallbacks; live values from settings.yaml
    model_name: str = "qwen2:1.5b"            # settings.yaml: llm.model_name
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0                   # 0.0 = fully deterministic

    # Pipeline Settings
    max_verification_iterations: int = 2       # settings.yaml: agent.max_verification_iterations

    # Navigator Settings (pre-generative filtering, thesis section 3.3)
    relevance_threshold_factor: float = 0.85   # Raised from 0.6; see §12.16
    redundancy_threshold: float = 0.8
    max_context_chunks: int = 10
    rrf_k: int = 60                            # Cormack et al. (2009). SIGIR.
    top_k_per_subquery: int = 10
    max_chars_per_doc: int = 500
    corroboration_source_weight: float = 0.1
    corroboration_query_weight: float = 0.05
    contradiction_overlap_threshold: float = 0.3
    contradiction_ratio_threshold: float = 2.0
    contradiction_min_value: float = 10.0
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

SIMPLE (no bridge dependencies):
  For each sub_query in plan.hop_sequence:
    a. embed(sub_query) → query_embedding
    b. vector_search(query_embedding, top_k=20)          ~8–12ms
    c. if strategy ∈ {GRAPH, HYBRID}:
          entities = entity_hints or extract_entities(sub_query)  ~3–5ms
          graph_search(entities, max_hops=2)                      ~1–30ms
  → RRF fusion → PreGenerativeFilter → top-K chunks

ITERATIVE (bridge dependencies detected, see Section 12.18):
  Sort hop_sequence by step_id
  current_hints = plan.entities (SpaCy-extracted)
  accumulated_context = []

  for step in sorted_hops:
    sub_results = retrieve(step.sub_query, entity_hints=current_hints)
    accumulated_context += deduplicate(sub_results)

    if step.is_bridge:
      bridge_entities = _extract_bridge_entities(sub_results, exclude=query_tokens)
      if bridge_entities:
        current_hints = current_hints ∪ bridge_entities  (up to 3 new entities)

  → PreGenerativeFilter(accumulated_context) → top-K chunks
```

The iterative path activates when any `HopStep.depends_on` is non-empty. It resolves the *hidden bridge entity problem*: the answer to hop 0 becomes the graph search key for hop 1, enabling retrieval of the answer document even when the bridge entity was unknown at query time.

---

### 4.3 Verifier Agent (S_V)

**File:** `src/logic_layer/verifier.py`

The Verifier receives the filtered context window and is responsible for generating a factually grounded answer and verifying its consistency.

#### 4.3.1 VerifierConfig

```python
@dataclass
class VerifierConfig:
    model_name: str = "qwen2:1.5b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 200            # Increased from 50; prevents answer truncation
    max_iterations: int = 2          # Initial generation + 1 correction round
    max_context_chars: int = 2400    # Total chars fed to LLM (~600 tokens)
    max_docs: int = 6                # Maximum number of context documents
    max_chars_per_doc: int = 500     # Characters per individual document (raised from 400)
```

> **Important:** All `VerifierConfig` values are loaded from
> `config/settings.yaml` via `_verifier_config_from_cfg()` in
> `agent_pipeline.py`. The class-level defaults are emergency fallbacks
> only; production code paths should never override them inline.

#### 4.3.2 ValidationStatus

```python
class ValidationStatus(Enum):
    VALID = "valid"               # Context is sufficient
    AMBIGUOUS = "ambiguous"       # Partial evidence
    CONFLICTED = "conflicted"     # Contradictory evidence
    INSUFFICIENT = "insufficient" # No evidence found
```

#### 4.3.3 Self-Correction Loop

The self-correction loop lives **entirely inside
`Verifier.generate_and_verify()`** and is the scientific core contribution of
Artifact B.

```
# Verifier.generate_and_verify(query, context)

pre_validate(context) → PreValidationResult

if status == INSUFFICIENT:
    return fallback_answer("I cannot determine the answer from the provided context.")

# Step 1: initial answer generation
context_str = build_context(context, max_chars=max_context_chars,
                            max_docs=max_docs, max_chars_per_doc=max_chars_per_doc)
answer = call_llm(GENERATION_PROMPT, query, context_str)   # qwen2:1.5b via Ollama

# Step 2: up to (max_iterations - 1) correction rounds.
# With max_iterations=1 (default since §12.30) the loop body never runs;
# self-correction is opt-in for ablation by raising max_iterations to 2.
for iteration in range(1, max_iterations):
    is_valid, violations = verify(query, answer, context)  # NLI consistency check
    if is_valid:
        break
    # The correction prompt embeds the concrete violations as feedback.
    answer = call_llm(CORRECTION_PROMPT, query, context_str, violations)

return VerificationResult(answer, confidence, iterations_used)
```

> **Architectural decision (2026-03-31):** an earlier outer retry loop in
> `AgentPipeline.process()` was removed. That outer loop called
> `generate_and_verify()` multiple times with identical input — not a real
> self-correction. Self-correction now lives exclusively in the Verifier's
> inner loop, which feeds concrete violations back into the prompt
> (`CORRECTION_PROMPT`) on every iteration.
>
> **Update (2026-05-06, §12.30):** the default `max_iterations` was lowered
> from `2 → 1`. On a 1.5 B-parameter SLM the second pass typically injects
> hallucinations rather than correcting them (Madaan et al. 2023 Self-Refine
> reproduces only on GPT-3.5+). Raise to `2` for ablation comparisons.

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

The pipeline layer contains two orchestration implementations: the original `AgentPipeline` (simple sequential chain) and the newer `AgenticController` (LangGraph state machine with iterative multi-hop support). Both expose a compatible `process(query)` interface.

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

> **Architectural note:** `AgentPipeline.process()` invokes
> `verifier.generate_and_verify()` **exactly once**. There is no outer
> retry loop at the pipeline level. The self-correction mechanism (the
> thesis contribution) lives exclusively **inside the Verifier**.
> `max_verification_iterations` in `settings.yaml` controls the number of
> correction rounds *within* the Verifier, not the number of pipeline
> iterations.

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

### 5.3 AgenticController

**File:** `src/logic_layer/controller.py`

The `AgenticController` is a LangGraph-based state machine that replaces the linear `AgentPipeline` for production use. It models the S_P → S_N → S_V chain as a directed graph of nodes with typed state transitions.

#### 5.3.1 State Machine

```
START
  │
  ▼
_planner_node()          → Calls Planner; populates RetrievalPlan in state
  │
  ▼
_navigator_node()         → Inspects hop_sequence for bridge dependencies
  │                          ├─ has_bridge_deps=False → _simple_navigate()
  │                          └─ has_bridge_deps=True  → _iterative_navigator_node()
  ▼
_verifier_node()          → Calls Verifier with accumulated context
  │
  ▼
END
```

#### 5.3.2 Iterative Navigator

`_iterative_navigator_node(state, hop_sequence, entity_names, plan_dict, start_time)`:

```python
# Step 1: Sort hops by step_id
sorted_hops = sorted(hop_sequence, key=lambda h: h.get("step_id", 0))

# Step 2: Execute hops in dependency order
for step in sorted_hops:
    results = navigator.navigate_step(step.sub_query, entity_hints=current_hints)
    accumulated_context += deduplicate(results)   # by chunk_id

    # Step 3: After bridge hop, extract new entities
    if step.is_bridge:
        bridge_entities = _extract_bridge_entities(results, exclude=query_tokens)
        current_hints.extend(bridge_entities)     # max 3 new entities

# Step 4: Apply pre-generative filter to accumulated context
final_context = navigator.filter(accumulated_context)
```

#### 5.3.3 Bridge Entity Extraction

`_extract_bridge_entities(chunks, exclude)` — static method:
- Regex: capitalized multi-word phrases (`\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b`)
- Excludes: tokens already in the original query
- Returns up to 3 candidates (prioritised by frequency in retrieved chunks)
- **Fallback:** if extraction yields nothing, original `entity_hints` are retained unchanged

#### 5.3.4 Safety Constraints

| Constraint | Value | Reason |
|---|---|---|
| Max iterative hops | 3 | Prevents unbounded loops |
| Bridge entity limit | 3 per step | Prevents graph search fan-out |
| Fallback on empty bridge | retain prior hints | Graceful degradation to single-hop behaviour |

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
  top_k_vectors: 20                 # widened 10 -> 20 in §12.30
  similarity_threshold: 0.3

# ── KNOWLEDGE GRAPH ────────────────────────────────────────────────────────
graph:
  backend: "kuzu"                   # Primary: KuzuDB (Cypher-native)
  graph_path: "./data/{dataset}/graph"  # Per-dataset KuzuDB directory
  max_hops: 2
  top_k_entities: 10                # widened 5 -> 10 in §12.30
  entity_extraction_method: "keyword"   # keyword | spacy | gliner

# ── RETRIEVAL & RAG ────────────────────────────────────────────────────────
rag:
  retrieval_mode: "hybrid"          # vector | graph | hybrid
  rrf_k: 60                         # Cormack et al. 2009
  cross_source_boost: 1.2           # bonus for chunks present in 2+ lanes
  enable_bm25: true                 # 3rd RRF path (§12.29)
  bm25_top_k: 20                    # widened 10 -> 20 in §12.30
  # vector_weight / graph_weight removed in §12.30 — never read by code;
  # use retrieval_mode for modality-disable ablation.

# ── LANGUAGE MODEL (LLM) ───────────────────────────────────────────────────
llm:
  model_name: "qwen2:1.5b"
  base_url: "http://localhost:11434"
  temperature: 0.1
  max_tokens: 200              # raised from 50 to prevent answer truncation
  max_context_chars: 3500      # widened 2500 -> 3500 in §12.30 (qwen2:1.5b ctx = 4096 tokens)
  max_docs: 5                  # cap on chunks forwarded to S_V
  max_chars_per_doc: 800       # widened 500 -> 800 in §12.30 (answer-bearing sentences)

# ── AGENTIC CONTROLLER ─────────────────────────────────────────────────────
agent:
  max_verification_iterations: 1   # 1 = no self-correction (§12.30 default)
  enable_verification: true

# ── NAVIGATOR (S_N) ────────────────────────────────────────────────────────
navigator:
  relevance_threshold_factor: 0.6   # lowered 0.85 -> 0.6 in §12.30
  redundancy_threshold: 0.8         # Jaccard dedup threshold
  max_context_chunks: 8             # lowered 10 -> 8 in §12.30
  rrf_k: 60                         # cross-sub-query RRF smoothing
  top_k_per_subquery: 20            # widened 10 -> 20 in §12.30
  # Cross-encoder reranker (§12.29)
  enable_reranker: true             # toggled false -> true in §12.30
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  reranker_top_k: 10
  # Numeric contradiction filter (restored in §12.31 after audit revert)
  contradiction_overlap_threshold: 0.3
  contradiction_ratio_threshold: 2.0
  contradiction_min_value: 100.0    # see §12.25

# ── ENTITY EXTRACTION ──────────────────────────────────────────────────────
entity_extraction:
  gliner:
    model_name: "urchade/gliner_small-v2.1"
    confidence_threshold: 0.15          # Recall-optimised for HotpotQA
    query_ner_confidence: 0.15          # Query-time threshold (same model, shorter text)
    entity_types:
      - person
      - organization
      - city
      - country
      - state
      - location
      - film
      - movie
      - album
      - work of art
      - landmark
      - event
      - award
  rebel:
    max_input_length: 256
    max_output_length: 256
    num_beams: 3

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

## 6.x Evaluation Layer — Artifact C

**Package:** `src/evaluations/`

The evaluation layer (Artifact C) provides the shared metric functions and experiment runners used across all thesis experiments.

### 6.x.1 Module Overview

| Module | Purpose |
|---|---|
| `metrics.py` | Canonical EM and F1 implementations shared by all evaluators |
| `evaluate_hotpotqa.py` | End-to-end HotpotQA benchmark runner |
| `ablation_study.py` | Configurable ablation study (vector/graph weights, verifier on/off) |
| `ollama_performance_diagnostic.py` | Embedding-dimension and LLM latency diagnostics |

### 6.x.2 Metric Functions (`metrics.py`)

All three functions are the single canonical implementation imported via `src.evaluations`:

- **`normalize_answer(s)`** — lowercase, strip articles, punctuation, and whitespace; following the official HotpotQA evaluation script.
- **`compute_exact_match(pred, gold)`** — normalised string equality with word-boundary substring fallback (handles cases where the gold answer is a proper subset of the predicted span).
- **`compute_f1(pred, gold)`** — token-level F1 matching the official HotpotQA evaluator (precision × recall harmonic mean over unigram bag-of-words overlap).

### 6.x.3 Ablation Study (`ablation_study.py`)

`AblationStudy.run()` accepts a `pipeline_factory` callable and a list of `(name, vector_weight, graph_weight)` configurations. For each configuration it:
1. Instantiates a fresh pipeline via `pipeline_factory`.
2. Evaluates `samples_per_dataset` questions.
3. Aggregates per-question EM/F1 into a `ConfigurationResult`.
4. Optionally saves raw JSON, CSV summaries, Markdown report, and LaTeX tables.

Random seed `42` is set at module import (`random.seed(42)`, `numpy.random.seed(42)`) to ensure reproducible question sampling across runs.

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

# Ingest chunks only — skip embedding recomputation (faster on re-ingest)
python benchmark_datasets.py ingest \
  --dataset hotpotqa \
  --samples 500 \
  --chunks-only

# Single configuration evaluation
# Note: --vector-weight / --graph-weight kept for backwards-compatibility on
# this CLI; the weights themselves were removed from RetrievalConfig in
# §12.30. Use --mode {vector,graph,hybrid} instead.
python benchmark_datasets.py evaluate \
  --dataset hotpotqa \
  --samples 100 \
  --model qwen2:1.5b \
  --mode hybrid

# Component ablation (toggle individual pipeline stages)
python benchmark_datasets.py evaluate \
  --dataset hotpotqa \
  --samples 100 \
  --no-planner \     # disable S_P
  --no-verifier \    # disable S_V
  --iterations 1     # no self-correction

# Full ablation study (all retrieval-mode combinations)
python benchmark_datasets.py ablation \
  --dataset hotpotqa \
  --samples 100

# Component ablation (Planner, Verifier, iterations)
python benchmark_datasets.py ablation \
  --dataset hotpotqa \
  --samples 100 \
  --component-ablation

# Ingestion status check
python benchmark_datasets.py status

# System self-test (no network required)
python benchmark_datasets.py test
```

Results of the ablation study are persisted to `evaluation_results/ablation_<timestamp>.json`.

### 7.5 Diagnostic Tools

Three diagnostic scripts are available for investigating retrieval quality, ingestion consistency, and LLM behaviour.

#### `diagnose.py` — Layer-by-layer pipeline diagnostic

```bash
# Test all pipeline layers for one question
python -X utf8 diagnose.py --idx 5

# Graph quality analysis (hub contamination, answer coverage)
python -X utf8 diagnose.py --graph-quality 20

# Vector score analysis for N questions (no LLM required)
python -X utf8 diagnose.py --multi 50

# Test a specific layer only
python -X utf8 diagnose.py --layer retrieval
python -X utf8 diagnose.py --layer verifier --skip-llm
```

#### `diagnose_verbose.py` — Trace-mode diagnostic with per-call timing

```bash
# Full trace including retrieval call timings and chunk scores
python -X utf8 diagnose_verbose.py --idx 11 --trace-calls

# Skip LLM to focus on retrieval only
python -X utf8 diagnose_verbose.py --idx 12 --skip-llm
```

#### `diagnose_ingestion.py` — Ingestion consistency checker

Traces each question's supporting articles through the full ingestion pipeline: `chunks_export.json` → LanceDB → KuzuDB → retrieval rank.

```bash
# Check ingestion consistency for specific question indices
python -X utf8 diagnose_ingestion.py --indices 11,12

# Check a range with vector rank included (requires Ollama)
python -X utf8 diagnose_ingestion.py --indices 0-19 --vector

# Different dataset
python -X utf8 diagnose_ingestion.py --indices 0-9 --dataset 2wikimultihop
```

**Output per question:**
1. Answer-bearing chunks (chunks whose text contains the gold answer)
2. Supporting-fact article presence in each store (chunks_export → LanceDB → KuzuDB)
3. Retrieval rank per query entity (graph and vector)
4. Crowd-out analysis: how many competing sources contain the same entity

**Graph-quality output (`--graph-quality N`)** prints, per question:

| Column | Meaning |
|---|---|
| `KW-H` | Number of graph chunks via keyword extraction |
| `Top-Entity` | Which entity actually matched |
| `Hub` | `HUB` if a generic/pronoun entity matched |
| `GrAns` | ✓ if the gold answer appears in the graph chunks |
| `VecH` | Number of vector chunks |
| `VecAns` | ✓ if the gold answer appears in the vector chunks |

**Empirical results:**

*As of 2026-04-01 (threshold=0.15, UPPERCASE types, 15 questions):*

| Metrik | Wert |
|---|---|
| Hub-Kontamination (Keyword-Regex) | 0% |
| Graph Answer Coverage | 33% (5/15) |
| Vector Answer Coverage | 40% (6/15) |
| Union Coverage (Graph ODER Vector) | 60% (9/15) |
| GLiNER Answer Coverage | 13% (2/15) — schlechter als Keyword-Regex |

*As of 2026-04-02 (threshold=0.5, lowercase types, re-ingestion, 10 questions):*

| Metrik | Wert | vs. vorher |
|---|---|---|
| Hub-Kontamination | 10% | ↓ besser |
| Graph Answer Coverage | 20% (2/10) | ↓ schlechter (aber N zu klein) |
| Vector Answer Coverage | 50% (5/10) | ↑ besser |
| Unique Entities | 23.858 | −35% (von 36.996) |
| MENTIONS | 50.096 | −33% (von 74.741) |

> **Interpretation:** The drop in graph answer coverage is attributable to
> (a) the small sample (N=10 vs. N=15) and (b) the entity-name
> disambiguation problem (chunk 7 "Edward Davis Wood Jr." is not reachable
> via the query "Ed Wood"). The two runs are not statistically comparable.
> For a robust comparison: `python diagnose.py --graph-quality 50`.

The graph contributes exclusive value-add (GrAns=✓, VecAns=✗) for ≥ 3
questions, supporting the hybrid hypothesis.

**Graph-Visualisierung** (`test_system/graph_3d.py`):

```bash
# PNG für Thesis generieren (Hub-Entities gefiltert)
python test_system/graph_3d.py --top 80 --no-html

# Mit interaktiver HTML-Ansicht (pyvis)
python test_system/graph_3d.py --top 80
```

Erzeugt `test_system/graph_preview.png` mit matplotlib (dpi=200). Für Druckqualität: `dpi=300` setzen.

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
| `rank_bm25` | ≥ 0.2.2 | BM25 sparse retrieval (3rd RRF path, §12.29) |
| `sentence-transformers` | ≥ 2.2 | Cross-encoder reranker `ms-marco-MiniLM-L-6-v2` (Stage 2.5, §12.29) |
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
| Ollama | `qwen2:1.5b` | Answer generation (1.5B parameters) |
| SpaCy | `en_core_web_sm` | Query parsing in Planner and Navigator |

Both Ollama models run entirely on CPU and require no GPU. `qwen2:1.5b` is loaded as a 4-bit GGUF quantisation via llama.cpp (Ollama backend), fitting within a 2 GB memory budget.

### 8.3 Design Rationale: Database Selection

**LanceDB** was selected over alternatives (FAISS, ChromaDB, Qdrant) because:
- Embedded architecture: no server, no Docker, no network latency.
- Native Apache Arrow columnar format: zero-copy reads, efficient batch operations.
- IVF-Flat index: approximate nearest-neighbour with configurable recall/speed trade-off.
- Built-in metadata filtering in a single SQL-like query.

**KuzuDB** was selected over NetworkX because:
- Native Cypher query language: expressive multi-hop path queries.
- Persistent on-disk storage: graph survives process restarts.
- Measured 10–100× faster traversal than NetworkX for graphs with >100 nodes (verified during migration benchmarks, April 2026).
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
    vector/                 (LanceDB directory)
    graph/                  (KuzuDB directory)
      extraction_results.json  (Colab Phase 2 output — unchanged by Phase 3)
    chunks_export.json      (Phase 1 output — source of truth for chunks)
    questions.json          (500 test questions with supporting_facts)
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
  │  3. POST /api/generate → qwen2:1.5b (Ollama)             │
  │     temperature: 0.1, max_tokens: 200                   │
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
| BM25 retrieval | ~5–20 ms | rank_bm25 over the cached corpus DataFrame (§12.29) |
| RRF fusion | ~0.5 ms | Pure Python, O(N log N), 3 lanes |
| Cross-encoder reranker | ~80–150 ms | Top-K × ~30 ms/pair on CPU, optional (§12.29) |
| Navigator filter chain (S_N) | ~2–6 ms | Six sequential filters in `navigator.py` |
| **Total retrieval** | **~120–280 ms** | Without LLM generation; reranker dominates |
| LLM generation (qwen2:1.5b) | ~200 ms–62 s | CPU, 4-bit quantised; context size and KV-cache allocation dominate |
| **End-to-end** | **~250 ms–65 s** | Full pipeline; LLM is the dominant bottleneck on CPU |

> **LLM timeout note:** Ollama allocates the full KV-cache context window regardless of prompt length. On CPU, `qwen2:1.5b` runs at approximately 8–15 tokens/second. A 2,400-character context (~600 tokens) can produce 60+ second responses. The system applies a 60-second `timeout` to Ollama HTTP calls; exceeding it returns the partial output. In practice, responses for bridge-resolved multi-hop queries with full context fit within 45 seconds.

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

The legacy `vector_weight` / `graph_weight` parameters were removed from
`settings.yaml` and from `RetrievalConfig` in the 2026-05-06 cleanup audit
(§12.30) — they were never read by production code. Modality-disable
ablation is now performed by setting `rag.retrieval_mode` to `vector`,
`graph`, or `hybrid`.

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

**Filter ownership.** Pre-generative filtering is owned by the Navigator
(S_N), not by the Data Layer. The 2026-05-06 cleanup audit (§12.30)
removed the duplicate `PreGenerativeFilter` class that previously sat
in `hybrid_retriever.py` and was never invoked in production.

**Stage 2.5 — cross-encoder reranker.** The `cross-encoder/ms-marco-MiniLM-L-6-v2`
model re-orders the top-K fused chunks by `(query, chunk)` relevance
between RRF fusion and the relevance-threshold filter (§12.29). It is
opt-in via `navigator.enable_reranker` (default `true` after the audit).
Disabling it reverts to pure RRF order.

---

---

## 12. Changes & Design Decisions

This section documents significant changes from earlier versions, discarded approaches, and the rationale for decisions made. It serves as a decision log for the thesis.

---

### 12.1 LLM Configuration: max_tokens and Context Budget

**Problem:** The initial configuration set `max_tokens: 50`. This caused model responses to be truncated mid-sentence, as HotpotQA answers typically require 10–30 tokens, but the model's context-building (chain-of-thought reasoning) demanded a larger token budget.

**Symptom:** Ablation with 10 samples (2026-03-31) showed EM=20% for all three configurations (vector_only, hybrid, graph_only) — identical results despite different retrieval methods. Root cause: all answers were truncated before they could include the gold answer.

**Fix:** `max_tokens: 200` (4× increase). Three additional context-budget parameters were introduced to control the amount of text passed to the LLM:
- `max_context_chars: 2400` — Prevents out-of-context errors on small models with a 4k context window
- `max_docs: 6` — More chunks increase the chance of retrieving both supporting facts (HotpotQA requires 2)
- `max_chars_per_doc: 400` — Balances breadth (more docs) vs. depth (more text per doc)

**Discarded alternative:** `max_tokens: 500` — tested; produced longer responses with hallucinations ("According to the context, the answer is X, however it should be noted that..."). Small models tend toward redundant output when given a larger token budget.

---

### 12.2 Removal of the Outer Pipeline Retry Loop

**Problem:** An earlier version of `AgentPipeline.process()` contained an outer loop that called `verifier.generate_and_verify()` up to `max_verification_iterations` times. This loop was introduced under the assumption that repeating the call with identical input would yield better answers.

**Finding:** An LLM with an identical prompt and `temperature=0.1` (near-deterministic) produces nearly identical outputs on every iteration. The outer loop was therefore a costly no-op — multiple LLM calls with no content improvement.

**Fix (2026-03-31):** Outer loop removed. `AgentPipeline.process()` now calls `generate_and_verify()` exactly once. The actual self-correction mechanism is the inner loop inside the Verifier, which at each iteration passes **concrete violations as feedback** (CORRECTION_PROMPT with specific `violations`). This is the core thesis contribution.

**Configuration consolidation:** `max_verification_iterations` is configured exclusively in `settings.yaml` under `agent:` and read by `_verifier_config_from_cfg()`. There is no other location in the code where this value is set.

---

### 12.3 Graph Visualisation: pyvis → matplotlib

**Problem:** The original `test_system/graph_3d.py` used pyvis with `LIMIT 200` for MENTIONS and `LIMIT 100` for RELATED_TO edges. With 66,585 MENTIONS and 14,445 RELATED_TO edges in the HotpotQA graph, this represented 0.3% of the data — the visualisation appeared sparse and structureless.

**Fix:** Complete rewrite using matplotlib + networkx:
- Loads up to 2,000 RELATED_TO edges; filters hub entities (pronouns, generic terms)
- Selects the top-N entities by degree (configurable via `--top N`)
- Saves directly as PNG (dpi=200; use dpi=300 for thesis figures)
- Optionally still generates interactive pyvis HTML

**Findings from the analysis:**
- Keyword-regex extraction performs well (hub contamination: 0%)
- Graph answer coverage: 33% vs. vector: 40% — the graph provides genuine added value
- GLiNER answer coverage: 13% — worse than simple keyword-regex. No re-ingestion required.

---

### 12.4 Graph Quality: REBEL Relation Extraction

**Finding:** REBEL (`Babelscape/rebel-large`) extracts primarily Wikipedia infobox-style relations: `publication_date`, `country`, `date_of_birth`, `genre`, `performer`. These relation types are largely useless for HotpotQA bridge questions, which require narrative relations such as `directed_by`, `portrayed_by`, `held_position`.

**HotpotQA graph metrics:**

*As of 2026-04-01 (threshold=0.15):*
- 9,412 DocumentChunks, 36,996 unique entities, 74,741 MENTIONS, 17,221 RELATED_TO
- Hub entities: "American" (898 chunks), "He" (737), "United States" (699), "She" (283)

*As of 2026-04-02 (threshold=0.5, re-ingestion):*
- 9,412 DocumentChunks, 23,858 unique entities, 50,096 MENTIONS, 15,766 RELATED_TO
- Hub contamination significantly reduced (10% vs. previously high)

**Thesis decision:** No re-ingestion with an alternative RE model (GPU effort > 1h, uncertain gain). Instead:
1. The graph limitation is documented as a known limitation in the thesis
2. Union coverage (60%) serves as an argument for the hybrid approach: the graph retrieves 3 questions that vector misses
3. The graph's value lies in MENTIONS (66,585 edges), not in RELATED_TO

---

### 12.5 Benchmark Sample Size

**Problem:** The initial ablation with only 10 samples produced statistically meaningless results (EM=20% for all configurations — could be chance; N too small for confidence intervals).

**Minimum requirement for the thesis:** N≥50 for preliminary orientation; N≥200 for reliable confidence intervals (95% CI ≈ ±7pp at N=200, ±14pp at N=50).

**Recommended configuration for final thesis results:**
```bash
python benchmark_datasets.py ablation --dataset hotpotqa --samples 200
```

---

### 12.6 Entity Confidence Threshold: 0.15 → 0.5 (Re-Ingestion 2026-04-02)

**Problem:** With `ner_confidence_threshold=0.15`, generic tokens dominated the graph: "American" (898 chunks), "He" (737), "United States" (699), "She" (283). These entities were stored as hub nodes with hundreds of MENTIONS edges. During graph traversal, they were returned as "matched entities" despite providing no semantic value.

**Fix:** `entity_confidence_threshold=0.5` in `local_importingestion.py`. The filter is applied at import time (not at extraction time), so `extraction_results.json` remains unchanged and can be re-imported with a different threshold if needed.

**Data incident:** The first `--clear` run used `shutil.rmtree(base_path)` and deleted `data/hotpotqa/` entirely (including `chunks_export.json`, `questions.json`). `extraction_results.json` survived only because it was open in the IDE. Fixed: `--clear` now deletes only `vector_db/` and `knowledge_graph/`, never source JSON files.

**Result:** 36,996 → 23,858 unique entities (−35%), 74,741 → 50,096 MENTIONS (−33%).

---

### 12.7 GLiNER Query-Time Consistency (2026-04-02)

**Problem:** `ImprovedQueryEntityExtractor` received `gliner_model=None` because `StorageConfig(enable_entity_extraction=False)` is the default. Consequence: every query ran with the SpaCy fallback instead of GLiNER — even though GLiNER is installed and the graph was built with GLiNER.

**Symptom:** SpaCy extracted `"Were Scott Derrickson"` (verb + name) instead of `"Scott Derrickson"`. Additionally, SpaCy entity types (`PERSON`, `ORG`, `GPE`) did not match the graph entity types (`person`, `film`, `city`) — causing name mismatches in graph lookups.

**Fix:**
1. `_get_gliner_model()` — module-level cache; loads `gliner_small-v2.1` once per process
2. `_load_gliner()` uses the cache instead of calling `GLiNER.from_pretrained()` directly
3. Entity types in `ImprovedQueryEntityExtractor` aligned with ingestion types: `["person", "organization", "city", "country", "film", "movie", "work of art", "event"]`
4. Default threshold: `0.5 → 0.2` (queries are short → lower GLiNER scores than for long chunk texts)
5. `diagnose.py` Layer 3: regex extraction replaced by `ImprovedQueryEntityExtractor`

**Performance note:** GLiNER cold start: ~7.5s, cached: <1ms. One-time delay on the first call per process.

---

### 12.8 Fallback Visibility: ⚠ FALLBACK ACTIVE Warnings

**Motivation:** Fallbacks are always symptoms — they indicate that the primary implementation is not working. Silent fallbacks hide errors.

**Implementation (2026-04-02):** All known fallback paths now emit `logger.warning("⚠ FALLBACK ACTIVE: ...")`:

| File | Fallback path |
|---|---|
| `hybrid_retriever.py` | SpaCy/regex instead of GLiNER for query entities |
| `entity_extraction.py` | SpaCy/regex instead of GLiNER for batch extraction |
| `ingestion_pipeline.py` | MockEntityExtractor / MockEmbeddingGenerator (use_mocks=True) |
| `verifier.py` | Heuristic contradiction detection instead of NLI |
| `navigator.py` | Relative import fallback |
| `chunking.py` | LangChain not installed |

---

### 12.9 Known Limitations: Entity Name Disambiguation

**Finding (2026-04-02):** The system fails on queries where the query uses a short form of an entity name, but the graph stores it under the full name.

**Concrete example:**
- Query: "Were Scott Derrickson and **Ed Wood** of the same nationality?"
- GLiNER extracts: `"Ed Wood"` → graph lookup finds entity "Ed Wood" → FILM article (1994 Tim Burton film)
- Correct Chunk 7: `"Edward Davis Wood Jr. was an American filmmaker"` → attached to entity `"Edward Davis Wood Jr."` → **not found**

**Cause:** KuzuDB graph lookup uses **exact string matching**. No alias resolution, no coreference.

**Possible solutions (for thesis outlook):**

| Approach | Effort | Effect |
|---|---|---|
| Entity linking (e.g. BLINK, REL) | High | Normalises entities to Wikidata IDs |
| Alias table at ingestion | Medium | `"Ed Wood"` ↔ `"Edward Davis Wood Jr."` as alias node |
| Fuzzy graph lookup | Low | Allow Levenshtein distance in graph search |
| Sub-query reformulation | Low | "What is the nationality of Ed Wood?" instead of "Were Ed Wood of same nationality?" |

For the thesis, this limitation is treated as a documented known issue. It is visible in the union-coverage metric and motivates the hybrid approach.

---

### 12.10 Planner Sub-Query Rewriting for Comparison Queries (2026-04-02)

**Problem:** Comparison queries such as `"Were Scott Derrickson and Ed Wood of the same nationality?"` produce, without rewriting, sub-queries like `"Were Scott Derrickson of the same nationality as Ed Wood?"`. These complex formulations have low vector similarity to factual chunks.

**Fix:** `_ATTR_MAP` with 8 regex patterns in `src/logic_layer/planner.py` `_decompose_comparison()`:

| Pattern | Template |
|---|---|
| `same nationality` | `"What is the nationality of {entity}?"` |
| `same birthplace` | `"Where was {entity} born?"` |
| `same profession` | `"What is the profession of {entity}?"` |
| `same genre` | `"What genre is {entity}?"` |
| `same age` | `"When was {entity} born?"` |
| `same country` | `"What country is {entity} from?"` |
| `same religion` | `"What is the religion of {entity}?"` |
| `born in the same` | `"Where was {entity} born?"` |

**Result:** `"Were Scott Derrickson and Ed Wood of the same nationality?"` now produces:
1. `"What is the nationality of Scott Derrickson?"` → direct match ✓
2. `"What is the nationality of Ed Wood?"` → no match (disambiguation issue 12.9)
3. Original query as context fallback

---

### 12.11 Navigator Entity-Mention Filter (2026-04-02)

**Problem:** For sub-queries such as `"What is the nationality of Ed Wood?"`, `nomic-embed-text` assigns high similarity to arbitrary nationality articles. Example: `"British people, or Britons..."` scores 0.798, while the relevant Ed Wood chunk scores only 0.649 — below all other results.

**Fix:** `_entity_mention_filter()` as Filter 5 in `_navigate()` of the `Navigator` class (`src/logic_layer/navigator.py`):

- Multi-word entities (e.g. `"Scott Derrickson"`): check for the full phrase **or** individual tokens ≥5 characters as whole words (`\bscott\b`, `\bderrickson\b`)
- Single-token entities: checked only if ≥5 characters (avoids false positives from stopwords like `"Were"` or `"Wood"`)
- **Safety fallback:** if all chunks are filtered → retain all (no empty context)

**Result (question 0):**
- `"British people"` → removed ✓ (no "Scott", "Derrickson")
- `"The Oku people"` → removed ✓ (no match; previously retained due to "Were" false positive)
- `"Tyler Bates"` → retained ✓ (contains "Scott" in the full chunk text, as Tyler Bates collaborated with Scott Derrickson)
- 20 raw → 5 relevant chunks (previously 6 with 2 irrelevant)

---

### 12.12 Verifier `best_answer or ...` Bug Fix (2026-04-02)

**Problem:** In `src/logic_layer/verifier.py` the verifier used `best_answer or "[Error:...]"`. Python's `or` returns the fallback when `best_answer` is **falsy** — including empty string `""`. An LLM returning `""` would incorrectly be treated as an error.

**Fix:** `best_answer if best_answer is not None else "[Error: no valid answer generated]"`

---

### 12.13 Data Layer Code Reviews (April 2026)

All four data layer modules were comprehensively reviewed and refactored in April 2026. Key changes:

**`entity_extraction.py`:**
- Entity IDs: MD5 12-char → SHA-256 24-char (`_generate_entity_id` module-level function)
- `EntityCache`: `get()`/`put()` now require `model_name` param — scoped by `(text_hash, model_name)` to invalidate on model change; `get_batch()` added (1 SQL query vs N round-trips)
- `ExtractionConfig`: removed `selective_re` (never used); added `rebel_max_input_length`, `rebel_max_output_length`, `rebel_num_beams`, `device`
- `GLiNERExtractor`: SpaCy NLP loaded once in `__init__` (not per-chunk)
- `REBELExtractor.extract_batch` → renamed `extract_sequential` (documents model's seq2seq limitation)
- `ExtractedEntity.to_dict()`: key `"type"` → `"entity_type"`; `ExtractedRelation.to_dict()`: keys renamed to `"subject_entity"`, `"object_entity"`, `"relation_type"`
- **Re-ingestion required** if entity IDs changed (MD5→SHA-256): existing KuzuDB entries are invalid.

**`chunking.py`:**
- Removed 316 lines of print()-based test harness
- UUID chunk IDs → deterministic SHA-256 (`source_doc:position:text[:50]`)
- Dead code removed; `word_boundary_factor` threaded as parameter
- `except Exception` narrowed to `(ValueError, RuntimeError, AttributeError)`

**`embeddings.py`:**
- `embed_documents` Phase 1 now uses `cache.get_batch()` (1 SQL query instead of N individual calls)
- `embed_query` now increments `metrics.total_texts` (was missing)
- `print_metrics()` replaced `print()` with `logger.info()`
- `DEFAULT_BATCH_SIZE` 32→64; all 4 `except Exception` in `EmbeddingCache` → `except sqlite3.Error`
- `get_batch` now bulk-updates `access_count`; context manager added (`__enter__`/`__exit__`)

**`hybrid_retriever.py`:**
- `QueryEntityExtractor` class removed (dead code — was immediately overwritten)
- Key name bugs fixed: `_vector_only_results` uses `document_id`+`metadata.source_file`; `_graph_only_results` uses `source_file`, `hops`, `matched_entity`
- `threading.Lock` added to `_GLINER_MODEL_CACHE` (double-checked locking)
- `gliner_model_name` threaded through `RetrievalConfig` → `ImprovedQueryEntityExtractor` → `_get_gliner_model()`
- Alias params `top_k_vector`/`top_k_graph` removed; canonical names `vector_top_k`/`graph_top_k` used everywhere
- `top_k or ...` → `top_k if top_k is not None else ...` (falsy-zero bug fixed)

---

### 12.14 entity_hints Parameter: GLiNER Re-extraction on Short Sub-Queries (2026-04-11)

**Problem:** In iterative multi-hop retrieval, the second hop's sub-query is a short 3–5 word phrase like `"screenwriter of Evolution"`. GLiNER on such a short fragment produces unreliable or no entity extractions. The graph search then falls back to regex patterns which miss multi-word bridge entities.

**Solution:** `entity_hints: Optional[List[str]]` parameter added to `HybridRetriever.retrieve()`. When provided, the hints are used directly as graph search entity names, bypassing GLiNER. The `Navigator` passes `entity_hints=entity_names` (SpaCy-extracted entities from the full original query) when invoking the retriever, and `AgenticController` extends these with bridge entities discovered during iterative execution.

**Propagation path:**
```
AgenticController._iterative_navigator_node()
  └─► Navigator.navigate_step(sub_query, entity_hints=current_hints)
        └─► HybridRetriever.retrieve(sub_query, entity_hints=current_hints)
              └─► HybridStore.graph_search(entity_names=entity_hints)
```

---

### 12.15 Storage Fuzzy Name Matching (`_name_variants`) (2026-04-11)

**Problem:** KuzuDB graph lookup is exact string matching. Queries for `"David Weissman"` find no results if the graph stores `"David N. Weissman"`. Similarly, looking up `"End of Days"` fails if the entity was stored as `"End of Days (film)"`.

**Solution:** `_name_variants(name)` helper in `KuzuGraphStore.find_chunks_by_entity_multihop()`:
- For 2-token names where the first token is ≤ 3 characters (e.g. `"Ed Wood"`): also tries last-name-only (`"Wood"`)
- For all names: also tries individual tokens ≥ 4 characters as standalone search terms
- The loop tries each variant until hop-0 results are found; `entity_name` is updated to the effective variant for hop-2/3 queries

**Limitation:** This is a lightweight heuristic, not a full entity linking solution. It resolves common short-name/full-name mismatches but does not handle aliases (`"Ed Wood"` ↔ `"Edward Davis Wood Jr."`). True disambiguation requires an entity linking system (e.g., BLINK, REL) — documented as Known Issue 12.9.

---

### 12.16 Relevance Threshold Factor: 0.6 → 0.85 (2026-04-11)

**Problem:** `nomic-embed-text` exhibits score compression: all text-pair similarities fall in the range 0.739–0.786 regardless of semantic relevance. A threshold factor of `0.6 × max_score` ≈ `0.6 × 0.786` = 0.47 — well below every result — effectively disabled the relevance filter. Every chunk passed through, filling the context window with noise.

**Solution:** `relevance_threshold_factor: 0.6 → 0.85`. This keeps only results within 15% of the top score. For score range [0.74, 0.79], this filters chunks scoring below ~0.67. Combined with the entity-mention filter (12.11), context quality improved measurably.

**Configuration location:** `config/settings.yaml` — `agent.relevance_threshold_factor`. No hardcoding in code.

---

### 12.17 Comparison Decomposition: Removed 3rd Sub-Query (2026-04-11)

**Problem:** `_decompose_comparison()` in `planner.py` produced three sub-queries:
1. `"What is the nationality of Scott Derrickson?"`
2. `"What is the nationality of Ed Wood?"`
3. Original query as fallback: `"Were Scott Derrickson and Ed Wood of the same nationality?"`

The third sub-query is a rephrased version of the original question. It embeds as near-identical to the first two and retrieves duplicate chunks, wasting context budget without adding new evidence.

**Solution:** Removed the third sub-query. The comparison decomposition now produces exactly 2 sub-queries (one per entity). If neither entity yields retrievable chunks, the original query is used as the sole sub-query via the existing single-hop fallback path.

---

### 12.18 Iterative Multi-hop Implementation (2026-04-11)

**Problem:** `HopStep.depends_on` and `HopStep.is_bridge` were defined in the data structures but never evaluated. All sub-queries in `hop_sequence` were dispatched simultaneously, which defeats the purpose of bridge-entity reasoning: the bridge entity name is unknown until hop 0 completes.

**Concrete failure case (idx=11):** Question: *"Who was the screenwriter of the film 'Evolution', and also wrote the screenplay for the movie in which David Weissman was credited as a producer?"* — The bridge entity is the name of the film David Weissman produced, which must be retrieved first to find the relevant screenwriter.

**Concrete failure case (idx=12):** Question: *"The song 'Oh My God' appears on the soundtrack for which year was the film with Guns N' Roses' contribution released?"* — The bridge is the film title ("End of Days"), not stated in the query.

**Solution:** Three coordinated changes:

1. **Planner (`planner.py`) — Patterns C and D:**
   - Pattern C: detects `"for a/an/the [film|movie|show|...]"` → HopStep 0 retrieves the bridge film, HopStep 1 retrieves the answer
   - Pattern D: detects `"[role] with [qualifier] co-wrote/directed/..."` → bridge is the work, not the person

2. **Controller (`controller.py`) — `_iterative_navigator_node()`:**
   - Sorts hops by `step_id`; executes in order
   - After each `is_bridge=True` step: calls `_extract_bridge_entities()` on retrieved chunks
   - Bridge entities are added to `current_hints`; passed as `entity_hints` to subsequent steps
   - Context accumulated across all steps, deduplicated by `chunk_id`

3. **HybridRetriever / Navigator — `entity_hints` parameter (see 12.14)**

**Routing in `_navigator_node()`:**
```python
has_bridge_deps = any(h.get("depends_on") for h in hop_sequence_raw)
if has_bridge_deps and len(hop_sequence_raw) > 1:
    return _iterative_navigator_node(state, ...)
else:
    return _simple_navigate(state, ...)   # original behaviour
```

**Verified results:** Both idx=11 and idx=12 now retrieve the correct answer-bearing chunks after the iterative hop resolves the bridge entity.

---

### 12.19 Ingestion Diagnostic Tool: `diagnose_ingestion.py` (2026-04-11)

**Motivation:** When retrieval fails for a specific question, the root cause may be at any of three levels: (1) the article was never chunked, (2) the chunk was never indexed in LanceDB or KuzuDB, or (3) the chunk is present but ranked below the top-k cutoff. Previously, this required manually querying each store.

**Tool:** `diagnose_ingestion.py` provides a full trace from source article to retrieval rank for any set of question indices.

**Four diagnostic stages per question:**
1. Answer-bearing chunks: scans `chunks_export.json` for the gold answer string
2. Supporting article presence: checks each store (chunks_export → LanceDB → KuzuDB) per supporting fact
3. Retrieval rank: runs graph and vector search and reports rank of first chunk from the correct source
4. Crowd-out analysis: shows competing sources per entity (identifies why correct chunk is pushed down)

**Usage:**
```bash
python -X utf8 diagnose_ingestion.py --indices 11,12
python -X utf8 diagnose_ingestion.py --indices 0-19 --vector  # includes vector rank (slow)
python -X utf8 diagnose_ingestion.py --indices 0-9 --dataset 2wikimultihop
```

**Confirmed findings (idx=12):** "End of Days" chunk (id=262) present in all three stores; entity "End of Days" returns graph rank #1; "Oh My God" chunk (id=263) contains gold answer ("1999"). Root cause of failure: query entity was "Arnold Schwarzenegger" (crowd-out by other Schwarzenegger films) rather than "End of Days" — resolved by iterative multi-hop (12.18).

---

### 12.20 Shared Utility Module: `src/utils.py` (2026-04-24)

**Motivation:** `jaccard_similarity` was independently implemented in both `hybrid_retriever.py` (as a `@staticmethod`) and `navigator.py` (as an instance method). Dual implementations risk silent divergence when one copy is updated.

**Change:** Created `src/utils.py` as the canonical shared utilities module.

```python
# src/utils.py
def jaccard_similarity(text1: str, text2: str) -> float:
    """Word-set Jaccard similarity in [0.0, 1.0]."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    union = len(words1 | words2)
    return len(words1 & words2) / union if union > 0 else 0.0
```

**Consumers:**
- `src/data_layer/hybrid_retriever.py` — deduplication step in RRF fusion
- `src/logic_layer/navigator.py` — MMR diversity scoring in `_mmr_rerank()`

**Design note:** The function uses word-set overlap (not character n-grams) which is appropriate for multi-hop retrieval deduplication: two retrieval results are considered duplicates when they share the majority of their content words, regardless of minor surface-form variation.

---

### 12.21 Test Suite Audit II — Gap-Filling Invariants (2026-04-30)

**Motivation:** A second 12-step test health audit (2026-04-30) identified five structural gaps in the test suite where correctness invariants were not asserted: embeddings vector-space consistency (T-B), GLiNER compound-span extraction (T-C), AgentPipeline FIFO cache eviction (T-D), ingestion metadata isolation (T-E), and Verifier factual grounding (T-A).

**Changes:**

| ID | File | Class added | Invariant covered |
|---|---|---|---|
| T-A | `test_verifier_semantic.py` | `TestVerifierFactualCorrectness` | Wrong entity in answer → violated claim or LOW confidence |
| T-B | `test_embeddings.py` | `TestEmbedQueryDocumentsConsistency` | `embed_query` / `embed_documents` same dimension + cosine ≥ 0.99 for identical text |
| T-C | `test_gliner_boundary.py` | `TestGLiNERSpanBoundary` | Multi-token spans ("Eiffel Tower", "New York City") extracted as one entity |
| T-D | `test_pipeline.py` | `TestAgentPipelineFIFOCache` | FIFO eviction: oldest entry removed when `_cache_max_size` exceeded |
| T-E | `test_pipeline.py` | `TestIngestionMetadataIsolation` | `source_doc` metadata does not leak across `_chunk_document()` calls |

**R1 compliance:** All five new test classes use zero or ≤2 ML inference calls (T-C respects `EDGE_RAG_N_SAMPLES=2`). T-A, T-B, T-D, T-E are pure unit tests with no model calls.

**Additional R-1 (nightly):** `TestEmbeddingSemanticQuality` added to `test_embeddings.py` with `@pytest.mark.nightly @pytest.mark.llm` — tests ordinal ranking of similar vs. dissimilar text pairs via live `nomic-embed-text`, documenting the score-compression limitation (§4.4) quantitatively.

**Additional R-2 (nightly):** `TestGLiNERRecall.test_recall_by_entity_type` added to `test_gliner_boundary.py` — emits `UserWarning` for entity types with recall < 0.5, surfacing the per-type weaknesses documented in §12.6.

---

### 12.22 Keyword Entity Fallback with Dual Injection (2026-05-04)

**Motivation:** `nomic-embed-text` score compression causes ANN retrieval to fail for specific entity-named chunks when the query contains multiple genus-level entity names. In practice, for the query "Are both Dictyosperma and Huernia described as a genus?" the Dictyosperma chunk ranked at ANN position #71–80 (score ≈ 0.68 vs. Fokienia ≈ 0.84 at position #1) and never entered the top-10 ANN results. KuzuDB also had no "Dictyosperma" entity node because GLiNER's entity-type list (`person`, `organization`, …) does not include plant genera. The chunk was therefore completely invisible to both retrieval paths.

**Root cause chain:**
1. nomic-embed-text score compression → Dictyosperma chunk at ANN rank #71 (not in top-10)
2. GLiNER entity-type list excludes plant genera → 0 KuzuDB nodes for "Dictyosperma" → 0 graph results
3. Without any retrieval hit, the Navigator Relevance Filter (threshold = 0.85 × max_score) rejects the chunk even after entity-mention check

**Solution — `_keyword_entity_search()` with dual injection** in `src/data_layer/hybrid_retriever.py`:

1. **O(N) pandas keyword scan:** On first call, load the full LanceDB table into a pandas DataFrame (`_keyword_df_cache`). For each query entity (≥ 4 chars), case-insensitively scan the `text` column and collect matching chunks.
2. **Dual injection:** Each keyword-matched chunk is injected into BOTH `vector_results` (with synthetic `similarity=0.76`) AND `graph_results` (with `hops=0`). This ensures the RRF cross-source boost applies, yielding a combined score of ~0.050 — above the relevance threshold of ~0.043.
3. **Single-path injection would fail:** A chunk present only in `graph_results` yields a single-path RRF score ≈ 0.016, below the threshold. Both paths are required.

**RRF arithmetic for dual-injected chunk:**
- Vector path: rank 10/10 → `1/(60+10) ≈ 0.0143`; graph path: rank 1/1 → `1/(60+1) ≈ 0.0164`
- Cross-source boost: `1.2 / 61 ≈ 0.0197`; total ≈ `0.0143 + 0.0164 + 0.0197 = 0.050`
- Relevance threshold ≈ `0.85 × 0.059 ≈ 0.050` → dual-injected chunk is retained

**Scope guard:** The scan only runs when `entity_names` is non-empty and the LanceDB table is accessible. A `_keyword_df_cache` attribute avoids repeated full-table scans within a single pipeline call.

**Trade-off:** O(N) pandas scan is acceptable for the thesis corpus (≤ 10,000 chunks). For production scale (millions of chunks), a full-text inverted index (e.g., LanceDB FTS or Tantivy) would replace this approach.

---

### 12.23 Navigator Entity Hints Propagation Fix (2026-05-04)

**Motivation:** `navigator.navigate()` accepted an `entity_names` parameter, but diagnostic tooling called `navigate(plan, sub_queries)` without supplying it. Without explicit hints, the HybridRetriever ran GLiNER independently on each sub-query. For sub-query 2 ("Are both Huernia described as a genus?"), GLiNER extracted only "Huernia" — so the keyword entity fallback (§12.22) never searched for "Dictyosperma" in that sub-query's retrieval call. Dictyosperma appeared in sub-query 1's results only, giving it a Navigator-level RRF score of ≈ 0.016 — below the relevance threshold of 0.029.

**Fix:** When `entity_names` is `None` (not supplied by caller), `navigate()` now falls back to `retrieval_plan.entities` to build the entity hints list:

```python
if entity_names is not None:
    hints = entity_names
elif retrieval_plan is not None and getattr(retrieval_plan, "entities", None):
    hints = [e.text for e in retrieval_plan.entities if getattr(e, "text", None)]
else:
    hints = None
```

This guarantees all sub-queries share the full entity list from the Planner, so keyword searches run for all entities across all sub-queries regardless of which entity appears in the sub-query text.

**Effect:** With both §12.22 and this fix, `diagnose.py --idx 27` retrieves 10 raw → **2** filtered chunks (Huernia + Dictyosperma), and the pipeline answers "Yes." (gold: "yes").

---

### 12.25 Contradiction Filter: Raised `contradiction_min_value` from 10 → 100 (2026-05-04)

**Problem:** The pre-generative contradiction filter in `navigator.py` compares numbers across chunk pairs to detect factual conflicts. The number extraction regex (`\b\d{4}\b|\b\d+(?:\.\d+)?\b`) captures all integers, including day-of-month values (e.g., `18` from "born **18** November 1963"). With `contradiction_min_value=10`, the filter required only that both numbers exceed 10. This meant the pair `(18, 1992)` — a birth-day from one chunk and a year from another — triggered with a ratio of 110×, incorrectly evicting valid answer chunks.

**Root cause:** The filter was designed to detect genuine fact conflicts like "has 200 employees" vs. "has 2000 employees". Day-of-month values (1–31) and month numbers (1–12) are date components, not independent factual claim slots, and must not drive contradiction logic.

**Fix:** `contradiction_min_value` raised from `10.0` to `100.0` in `config/settings.yaml` and the `ControllerConfig` dataclass default. Values < 100 (days, months, small counts) are excluded from the ratio comparison. Year-scale numbers (≥ 100) and count-scale statistics (population, budget, staff count) are unaffected.

**Effect:** Answer chunks in biographical articles (which routinely contain both day-of-month and year values) are no longer incorrectly evicted. The Peter Schmeichel bio chunk (containing the IFFHS award fact) now survives all pre-generative filters.

**Verification:** New smoke-test case in `navigator.py` (`if __name__ == "__main__"`) verifies that the pair `(18, 1992)` does **not** trigger (false-positive guard) while the pair `(200, 2000)` on a shared-topic pair **does** trigger (correct contradiction).

---

### 12.26 Planner Pattern E: Relational Anchor Bridge Decomposition (2026-05-04)

**Problem:** Bridge questions with the structure "What [fact] about the [role] of [Entity]?" were classified as `MULTI_HOP` but produced only a single sub-query (the verbatim original). The existing split patterns split at `that/which/who` or `of the`, neither of which matches "of Kasper Schmeichel" (a proper noun, not the article "the"). With one sub-query, retrieval is anchored entirely to the surface entity (e.g., Kasper Schmeichel), and the bridge entity's article (Peter Schmeichel) may not rank high enough to survive filtering.

**Examples of the failure class:**
- "What was the **father of** Kasper Schmeichel voted to be?" → bridge = Peter Schmeichel
- "Where did the **wife of** John Lennon grow up?" → bridge = Yoko Ono
- "What award did the **founder of** Apple win?" → bridge = Steve Jobs

**Fix:** Added Pattern E to `PlanGenerator._decompose_multi_hop()` in `planner.py`. When all existing split patterns fail (`len(parts) == 1`) and the query matches `the [role] of [Entity]` (using a fixed vocabulary of ~25 role nouns via class constant `_RELATIONAL_ANCHOR_ROLES`), two sub-queries are generated:
- **Hop 0 (bridge):** `"Who is the {role} of {anchor entity}?"` — causes the Navigator to retrieve the anchor's article, which names the bridge entity, and to run keyword entity search for the anchor, surfacing the bridge's own article via cross-mention
- **Hop 1 (final):** The original query — direct retrieval for the downstream fact

Pattern E only fires as a fallback after Patterns C and D are checked, preserving existing decomposition paths.

**Effect:** For relational bridge questions, the Navigator now runs two sub-queries. The bridge resolution sub-query (hop 0) retrieves the anchor entity's article (containing the bridge entity name) and, via keyword entity fallback, may also surface the bridge entity's own article. Both appear in the RRF fusion pool with cross-query boosts.

---

### 12.27 Verifier Question-Relevance Context Reordering (2026-05-04)

**Problem:** Small LLMs (qwen2:1.5b, 1.5B parameters) exhibit a strong position bias: they preferentially extract the first plausible entity or fact they encounter in the context prompt. In multi-hop questions the final answer chunk often has a lower RRF rank than distractor chunks (those containing the bridge entity name), so the answer appears late in the formatted context and the LLM outputs the bridge entity rather than the requested fact.

**Root cause observed (idx=24):** Query "What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?" Peter Schmeichel biographical chunk (containing "voted the IFFHS World's Best Goalkeeper in 1992") had RRF score 0.0152 — ranked last after Kasper Schmeichel chunks. LLM saw "Peter Schmeichel" first and answered "Peter Schmeichel." instead of "World's Best Goalkeeper."

**Fix:** Added `Verifier._reorder_by_question_relevance(query, context)` called immediately before `_format_context`. The method:
1. Tokenises the query with `re.findall(r"\b\w{4,}\b", ...)` and removes a fixed stopword set (`_QR_STOPWORDS`)
2. Scores each chunk by counting how many query content tokens appear in the chunk text (case-insensitive substring match)
3. Stable-sorts descending — chunks most lexically aligned with the query float to the top; equal-score chunks retain their original RRF order

**Properties:**
- Zero-cost when context has ≤1 chunk
- Falls back to original order when query produces no content tokens (all stopwords)
- Does not alter ranking semantics — RRF scores are preserved and available for downstream processing; only the display order to the LLM is changed

**Effect:** The chunk "Peter Schmeichel was voted the IFFHS World's Best Goalkeeper in 1992" scores 4 query-word hits (schmeichel, voted, iffhs, 1992) and now appears first in the formatted context. The LLM reads the answer evidence before distractors.

**Tests:** `TestQuestionRelevanceReorder` in `test_system/test_verifier_semantic.py` (5 tests).

---

### 12.28 Entity-Mention Filter Token Threshold + Bridge Entity Extraction (2026-05-05)

**Root cause identified via gold-answer stage tracking (diagnose_verbose.py §12.28 additions):**

For the query "What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?", the entity mention filter was configured to fall back to individual token matching (≥5 chars) when a full multi-word phrase was not found. With entity_names=["Kasper Schmeichel"], the fallback token "kasper" (6 chars ≥ 5) matched:
- "Kasper Gus Ntjalka Williams" (unrelated Australian country singer)
- "Summer of '92 ... directed by Kasper Barfoed" (unrelated Danish film)

Both irrelevant chunks passed the filter, corrupting the bridge retrieval pool. The Gus Williams chunk occupied position #1 in filtered_context, which caused `_extract_bridge_entities` to extract "Kasper Gus Ntjalka Williams", "Gus Williams", "Central Australia" instead of "Peter Schmeichel".

**Fix A — Entity mention filter fallback threshold (navigator.py):**
Raised individual-token fallback threshold from ≥5 → ≥8 characters for **multi-word entities only**. Single-token entity threshold unchanged at ≥5.

Effect:
- "kasper" (6 chars) → excluded from fallback → Gus Williams and Barfoed articles no longer pass
- "schmeichel" (10 chars ≥ 8) → still included → Peter Schmeichel's article still passes via surname fallback
- Single-token entities (e.g. "France") → unchanged behavior

**Fix B — Bridge entity extraction: surname-anchor + all chunks (controller.py):**

Two changes to `_extract_bridge_entities`:
1. Uses **all** filtered chunks instead of `filtered_context[:2]`; the bridge entity (Peter Schmeichel's article) was consistently in the 3rd–4th position and was never examined.
2. Added **surname-anchor Pass 1**: for each known entity (e.g. "Kasper Schmeichel"), extract surname tokens (≥6 chars → "Schmeichel"), then run a unicode-aware regex `[FirstName] [OptionalMiddle]? [Surname]` over all chunks. This recovers "Peter Schmeichel" from "Peter Bolesław Schmeichel" where the middle name's special character ł broke the ASCII-only `_PROPER_NOUN_RE`.

**Fix C — Bridge extraction rejects `[About: X]` annotation artifacts (controller.py):**

The own-doc chunk alias annotation (§12.24) prepends `[About: Kasper Schmeichel]` to chunks whose article title is not verbatim in the text. The surname-anchor regex in Pass 1 captured "About:" as a first-name token (colon is not in the `[^\s,.()\[\]]` exclusion set), producing false bridge entities "About: Kasper" and "About: Schmeichel".

Fix: added `and ":" not in first` to the Pass 1 candidate filter. A colon in the matched first-name token is a reliable signal that the match is an annotation prefix, not a proper name.

**Combined effect (idx=24 trace):**
After Fix A, Hop 0's entity mention filter passes only 2 chunks (Kasper Schmeichel article + Peter Schmeichel article). Bridge extraction via Pass 1 finds "Peter Schmeichel" (Fix C eliminates the former "About: Kasper" / "About: Schmeichel" noise). Hop 1 runs with entity_names=["Kasper Schmeichel", "1992", "Peter Schmeichel"] → Peter Schmeichel's IFFHS chunk passes filter → gold answer in context position #1 after reorder (§12.27).

---

### 12.30 Cleanup Audit — Dead Code, Dependencies & Parameter Tuning (2026-05-06)

A "Radical Simplification" pass on the v4.4 codebase. Goals: (1) remove
unused/duplicated code, (2) install the two missing edge-feasible
retrieval primitives that §12.29 had wired in but the environment lacked,
(3) widen retrieval funnels and loosen over-aggressive filters per a
quantitative parameter-optimisation analysis. **No new agents, layers,
or filters were added.**

**A. Code & module deletions:**

| Item | Location | Reason |
|---|---|---|
| `PreGenerativeFilter` class (≈ 167 LoC) | `src/data_layer/hybrid_retriever.py` | Never called in production — the Navigator owned the filter chain since v3.4 |
| Public re-export `PreGenerativeFilter` | `src/data_layer/__init__.py` | Cleanup of public API |
| `_keyword_entity_search` method (≈ 100 LoC) | `src/data_layer/hybrid_retriever.py` | BM25 (§12.29) strictly subsumes substring scanning |
| `vector_weight`, `graph_weight` fields | `RetrievalConfig` + `settings.yaml` | Never read by production code; modality ablation now uses `mode` |
| Disabled NLI verifier YAML keys | `settings.yaml` (`enable_contradiction_detection`, `nli_model`, `nli_max_input_chars`, `contradiction_threshold`) | Disabled by default since v3.x; 270 MB model violates edge constraint |
| Stub flags `query_expansion_enabled`, `reranking_enabled` | `settings.yaml` (`rag:` block) | No consumer code |
| Unused import `_PROPER_NOUN_RE` | `src/logic_layer/navigator.py` | Lint warning, dead reference |
| `_main()` smoke demo | `src/data_layer/hybrid_retriever.py` (≈ 123 LoC) | Replaced by canonical `pytest test_system/test_data_layer.py` |

**B. Environment / dependencies:**

| Action | Outcome |
|---|---|
| `pip install rank_bm25` | Activates BM25 sparse retrieval (§12.29) — previously silently disabled |
| `pip install sentence-transformers` | Activates the cross-encoder reranker (§12.29) — previously silently disabled |

**C. Parameter optimisation table (`config/settings.yaml`):**

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `vector_store.top_k_vectors` | 10 | 20 | Bridge-answer chunks regularly land at rank 11–15 under nomic-embed score compression |
| `graph.top_k_entities` | 5 | 10 | Mirror the vector widening; KuzuDB queries are < 30 ms |
| `rag.bm25_top_k` | 10 | 20 | BM25 cost is O(N) but tiny; widening improves cross-source boost recall |
| `navigator.relevance_threshold_factor` | 0.85 | 0.6 | 0.85 × max evicted bridge chunks under compressed scores |
| `navigator.top_k_per_subquery` | 10 | 20 | Match upstream widening |
| `navigator.max_context_chunks` | 10 | 8 | Avoid double-trim with `max_docs=5`; leaves a 3-chunk safety margin |
| `navigator.enable_reranker` | `false` | `true` | Reranker is now installed and edge-feasible (22 MB, ~30 ms/pair) |
| `llm.max_chars_per_doc` | 500 | 800 | Answer-bearing sentences regularly exceed 500 chars |
| `llm.max_context_chars` | 2500 | 3500 | qwen2:1.5b has a 4096-token context window; previous cap used only ~20 % |
| `agent.max_verification_iterations` | 2 | 1 | Self-Refine (Madaan 2023) reproduces on GPT-3.5+; on 1.5 B SLMs the second pass injects hallucinations |
| `verifier.entity_coverage_threshold` | 0.5 | 0.34 | For 3-entity bridge queries the bridge entity is unknown at S_P time; 1/3 lets the gold chunk pass on the first hop |

**D. Localisation pass.** All German user-facing strings in
`diagnose.py`, `diagnose_verbose.py`, `test_system/diagnose.py`,
`local_importingestion.py`, and `benchmark_datasets.py` were translated
to English to satisfy the project rule "all artefacts in English".

**E. Test suite impact.** The audit deleted tests covering removed code
(`TestPreGenerativeFilter`, `TestContradictionFilterPassthrough`, the
`vector_weight`/`graph_weight` RetrievalConfig assertions). Total CI
suite: **496 passing** (excl. graph_inspect, nightly, llm, integration).

---

### 12.31 Partial Revert — Contradiction Filter Restored, Planner Thresholds Restored (2026-05-07)

The 2026-05-06 audit also attempted to delete the Navigator's numeric
contradiction filter (Filter 3) and to tighten the Planner's
entity-confidence thresholds. Both changes had to be partially reverted
within 24 hours.

**Reverted changes:**

| Reverted item | Reason |
|---|---|
| Deletion of `Navigator._contradiction_filter` | `diagnose_verbose.py` monkey-patches every Navigator filter for tracing; `getattr(navigator, "_contradiction_filter")` raised `AttributeError` and crashed every diagnostic run |
| Deletion of `contradiction_*` fields from `ControllerConfig` | Same reason; the diagnostic accesses them in the trace wrapper |
| `planner.min_entity_confidence: 0.7 → 0.85` | Caused the Planner to emit `[]` entities for queries where SpaCy NER returned nothing; Filter 5 then fell back to keep-all and the Verifier received 8 chunks instead of 5, exceeding the 60 s Ollama timeout on `idx=33` |
| `planner.regex_entity_confidence: 0.75 → 0.6` | Same root cause as above |
| `entity_extraction.gliner.confidence_threshold: 0.15 → 0.30` | Affects ingestion threshold too; not selectively applicable at query time only |
| New gating block in `Planner._extract_entities` (`if regex_entity_confidence >= min_entity_confidence`) | Logic change rather than a tuning change; reverted with the threshold revert |

**Net state after partial revert:**

| Audit item | Status |
|---|---|
| `pip install rank_bm25` | **Kept** |
| `pip install sentence-transformers` | **Kept** |
| All §12.30 code deletions (PreGenerativeFilter, `_keyword_entity_search`, weights, NLI YAML keys, stub flags, unused imports) | **Kept** |
| All §12.30 parameter-optimisation values (top-k = 20, relevance_factor = 0.6, max_chars_per_doc = 800, etc.) | **Kept** |
| `navigator._contradiction_filter()` method | **Restored** (kept after revert) |
| `ControllerConfig.contradiction_*` fields + YAML keys | **Restored** |
| `planner.min_entity_confidence` | **0.7** (reverted) |
| `planner.regex_entity_confidence` | **0.75** (reverted) |
| `entity_extraction.gliner.confidence_threshold` | **0.15** (reverted) |

Total CI suite after partial revert: **496 passing**, identical to the
pre-revert state (the contradiction-filter tests were re-instated together
with the method, and the Planner-confidence test in
`test_planner_semantic.py` was reverted to expect `0.7` again).

**Lesson.** Diagnostic tools are part of the system surface; deleting
methods or config fields that they depend on is a behavioural change, not
a clean-up. Future audits must either (a) preserve diagnostic-touched
APIs or (b) update the diagnostic in the same commit.

---

### 12.29 BM25 + Cross-Encoder Reranker + qwen3:4b Config (2026-05-06)

**Motivation:** Three targeted improvements to close the largest gaps vs. SOTA (§12 overall):

1. **BM25 sparse retrieval** fills the missing "sparse" leg of the Hybrid RAG architecture.
   Dense embeddings (nomic-embed-text) exhibit score compression (all pairs 0.74–0.78),
   making RRF ranking nearly arbitrary. BM25 is unaffected by this because it operates
   on raw term-frequency statistics, not semantic similarity. Queries like
   "IFFHS World's Best Goalkeeper 1992" score near-zero in nomic but score high in BM25.

   **Implementation (`src/data_layer/hybrid_retriever.py`):**
   - `RetrievalConfig`: added `enable_bm25: bool = True`, `bm25_top_k: int = 10`
   - `RetrievalResult`: added `bm25_score`, `bm25_rank` fields
   - `RRFFusion.fuse()`: added `bm25_results` optional parameter; BM25 ranks contribute
     1/(k+rank) to RRF scores; cross-source boost extended to any chunk in 2+ retrieval lists
   - `HybridRetriever._build_bm25_index()`: builds `BM25Okapi` lazily from the same
     `_keyword_df_cache` DataFrame used by `_keyword_entity_search()` — no extra DB scan
   - `HybridRetriever._bm25_search()`: returns top-k chunks in vector-result dict format
     with normalised similarity score for clean RRF integration
   - Activated in `retrieve()` as step 2b before graph retrieval
   - Dependency: `rank_bm25>=0.2.2` (pure Python, no GPU, ~50 ms index build)

   **Expected effect:** +8–12 EM points on HotpotQA. Exact-match queries (award names,
   titles) that embed poorly now surface via BM25's term overlap.

2. **Cross-encoder reranker** adds a 22 MB query-aware re-ranking step after RRF fusion
   and before the pre-generative filter chain (Stage 2.5 in the Navigator pipeline).

   **Implementation (`src/logic_layer/navigator.py`, `src/logic_layer/_config.py`):**
   - `ControllerConfig`: added `enable_reranker: bool = False`, `reranker_model`,
     `reranker_top_k: int = 10`
   - `Navigator.__init__`: added `self._reranker = None` with lazy-load sentinel
   - `Navigator._reranker_filter(results, query)`: re-scores top `reranker_top_k` results
     using `CrossEncoder(model).predict([(query, chunk)])`, sorts by cross-encoder score;
     on `ImportError` or load failure sets `self._reranker = False` to skip future attempts
   - Called in `navigate()` after `_rrf_fusion()`, uses `retrieval_plan.original_query`
   - Disabled by default (`enable_reranker: false` in settings.yaml); activate for
     non-edge deployment or for ablation comparison

   **Expected effect:** +5–10 EM on HotpotQA. Resolves the score-compression problem
   independently of BM25, prioritising the most query-relevant chunk at position 1.

3. **qwen3:4b as ablation/comparison model** — added to `available_models` in settings.yaml.
   Switch by setting `llm.model_name: "qwen3:4b"` (requires `ollama pull qwen3:4b`).
   Expected effect: +15–20 EM over qwen2:1.5b from improved chain-of-thought capability.

**Ablation study configurations enabled by §12.29:**

| Config | BM25 | Reranker | LLM | Expected EM |
|---|---|---|---|---|
| Baseline (v4.4) | ✗ | ✗ | qwen2:1.5b | ~25–30 |
| +BM25 | ✓ | ✗ | qwen2:1.5b | ~33–42 |
| +Reranker | ✗ | ✓ | qwen2:1.5b | ~30–40 |
| +Both | ✓ | ✓ | qwen2:1.5b | ~38–45 |
| +Both +qwen3:4b | ✓ | ✓ | qwen3:4b | ~50–60 |

**Note on KuzuDB relation edges (Tier 2.5):** The RELATED_TO edge infrastructure and REBEL
relation extraction pipeline are already implemented (`storage.py:_integrate_entities`,
`entity_extraction.py`). Family/role relations are extracted during ingestion via REBEL
and stored as RELATED_TO edges. Re-ingestion activates this for existing documents.

**Note on embedding model (Tier 1.2):** BGE-small / E5-small require re-ingestion and a
different embedding backend. Documented as config comments in `config/settings.yaml`.
Not changed as the active default to preserve the evaluation baseline.

---

*End of Technical Architecture Documentation*

---

## Test Suite Summary

CI run on **2026-05-07** after the §12.31 partial revert. Markers
`slow`, `nightly`, `llm`, and `integration` are deselected; tests that
require a live KuzuDB population (`test_graph_inspect.py`) are skipped.

| File | Tests | Status |
|---|---|---|
| `test_system/test_chunking.py` | 30 | ✓ |
| `test_system/test_embeddings.py` | 40 | ✓ |
| `test_system/test_data_layer.py` | 86 | ✓ (post-audit; was 90) |
| `test_system/test_logic_layer.py` | 106 | ✓ (post-audit; was 110) |
| `test_system/test_planner_semantic.py` | 27 | ✓ |
| `test_system/test_navigator_semantic.py` | 32 | ✓ (post-audit; was 38) |
| `test_system/test_verifier_semantic.py` | 39 | ✓ |
| `test_system/test_thesis_matrix.py` | 9 | ✓ |
| `test_system/test_thesis_matrix_ext.py` | 10 | ✓ |
| `test_system/test_pipeline.py` | 74 | ✓ |
| `test_system/test_missing_coverage.py` | 7 | ✓ |
| `test_system/test_config_robustness.py` | 12 | ✓ |
| `test_system/test_gliner_boundary.py` | 16 | ✓ |
| `test_system/test_graph_inspect.py` | — | skipped (graph not populated in CI) |
| **Total** | **496** | **✓ all green** |

Tests removed in the §12.30 audit (covering deleted code):
`TestPreGenerativeFilter` (2), `TestContradictionFilterPassthrough` (2),
the four `_contradiction_filter` tests in `test_logic_layer.py`, the six
`TestContradictionFilter` tests in `test_navigator_semantic.py`, and the
`vector_weight=0.7, graph_weight=0.3` parameterisation in
`test_data_layer.py::test_retrieval_modes`. The contradiction-filter
tests were **restored** in §12.31 together with the filter method itself.

---

**Document Version History**

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2025-12-01 | Initial architecture (NetworkX + FAISS) |
| 2.0.0 | 2026-01-15 | Migration: LanceDB + KuzuDB |
| 2.1.0 | 2026-01-25 | Distance metric fix (L2 → cosine) |
| 3.0.0 | 2026-01-30 | Three-agent pipeline (S_P, S_N, S_V) |
| 3.1.0 | 2026-02-26 | Comprehensive review; all bugs resolved |
| 3.1.1 | 2026-03-15 | LLM config: max_tokens=200; new context-budget fields |
| 3.1.2 | 2026-03-31 | Outer retry loop removed from AgentPipeline; self-correction in Verifier only |
| 3.1.3 | 2026-03-31 | Graph quality diagnostics (diagnose.py --graph-quality); graph visualisation (graph_3d.py) |
| 3.2.0 | 2026-04-01 | Full documentation revision; Section 12 (Changes & Alternatives) added |
| 3.3.0 | 2026-04-02 | Threshold 0.15→0.5 (re-ingestion); GLiNER query consistency fix; fallback warnings; entity disambiguation documented |
| 3.4.0 | 2026-04-02 | Planner sub-query rewriting (12.10); Navigator entity-mention filter (12.11); Verifier `or`-bug fix (12.12) |
| 4.0.0 | 2026-04-11 | Data layer reviews (12.13); entity_hints parameter (12.14); storage fuzzy matching (12.15); threshold 0.6→0.85 (12.16); comparison noise sub-query removed (12.17); iterative multi-hop implemented (12.18); diagnose_ingestion.py added (12.19); all dataclass field names corrected; AgenticController documented (§5.3); repository structure updated |
| 4.1.0 | 2026-05-04 | Keyword entity fallback + dual injection (12.22); Navigator entity hints propagation fix (12.23); test_data_layer.py: 74→90 tests; total CI suite: 501 passed |
| 4.2.0 | 2026-05-04 | Contradiction filter min_value 10→100 (12.25); Planner Pattern E relational-anchor bridge (12.26); own-doc chunk alias annotation (12.24) |
| 4.3.0 | 2026-05-04 | Verifier question-relevance context reordering (12.27); test_verifier_semantic.py: 36→39 tests; total CI suite: 500 passed |
| 4.4.0 | 2026-05-05 | Entity-mention filter fallback threshold ≥5→≥8 (12.28 Fix A); Bridge extraction surname-anchor + all-chunks (12.28 Fix B); Bridge extraction rejects `[About:]` artifacts (12.28 Fix C); diagnose_verbose.py gold-tracking |
| 4.5.0 | 2026-05-06 | BM25 sparse retrieval as third RRF path (12.29); Cross-encoder reranker Stage 2.5 in Navigator (12.29); qwen3:4b added to available_models |
| 4.6.0 | 2026-05-06 | Cleanup audit (12.30): `PreGenerativeFilter`, `_keyword_entity_search`, `vector_weight`/`graph_weight`, NLI verifier YAML, stub flags & smoke demos deleted; rank_bm25 + sentence-transformers installed; full parameter-optimisation pass (top-k 10→20, relevance_factor 0.85→0.6, max_chars_per_doc 500→800, max_iterations 2→1, reranker on by default); all German UI strings translated to English |
| 4.6.1 | 2026-05-07 | Partial revert (12.31): contradiction filter, `ControllerConfig.contradiction_*` fields and Planner thresholds restored after the audit broke `diagnose_verbose.py` and overflowed the LLM context budget on yes/no queries; CI suite back to 496 passing |
