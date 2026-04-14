# Technical Architecture Documentation

**Project:** Enhancing Reasoning Fidelity in Quantized Small Language Models on Edge Devices via Hybrid Retrieval-Augmented Generation
**Author:** Jan Nietzard
**Institution:** FOM Hochschule, Master of Science
**Version:** 4.0.0
**Last Updated:** 2026-04-11

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
12. [Änderungen & Alternativen](#12-änderungen--alternativen)

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
    vector_weight: float = 0.7      # used to disable modality (weight=0); NOT an RRF coefficient
    graph_weight: float = 0.3       # used to disable modality (weight=0); NOT an RRF coefficient
    similarity_threshold: float = 0.3
    rrf_k: int = 60                 # RRF smoothing constant (Cormack et al., 2009)
    cross_source_boost: float = 1.2 # Bonus for dual-indexed results
    final_top_k: int = 10
    query_ner_confidence: float = 0.15       # Read from settings.yaml
    query_entity_types: List[str] = field(default_factory=list)  # Read from settings.yaml
    gliner_model_name: str = "urchade/gliner_small-v2.1"
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

Six sequential filter stages:

1. **Relevance filter** — Removes results below `relevance_threshold_factor × max_score` (default factor: 0.85). Score compression in nomic-embed-text (all pairs score 0.74–0.80) means this threshold must be calibrated carefully — too low keeps all chunks, too high removes everything. The factor `0.85 × max` keeps only results close to the top score.

2. **Redundancy filter** — Deduplicates by Jaccard similarity on token sets. Two results are considered redundant if `|A ∩ B| / |A ∪ B| > redundancy_threshold` (default: 0.8). Preserves topically similar but informationally complementary chunks.

3. **Contradiction filter** — Applies a lightweight NLI classifier to detect conflicting factual claims. Conflicted pairs are resolved by retaining the higher-scoring result.

4. **Entity-overlap filter** — Retains only chunks that share at least one named entity with the query entity set. Reduces noise from topically adjacent but factually irrelevant chunks.

5. **Entity-mention filter** (`_entity_mention_filter` in `navigator.py`) — Checks that each chunk's *text* contains at least one query entity as a literal mention. Multi-word entities (e.g. `"Scott Derrickson"`) match if the full phrase appears OR individual tokens ≥ 5 characters appear as whole words (`\b...\b`). Single-token entities are only checked if length ≥ 5 (avoids false positives from short stopwords like `"Were"` or `"Wood"`). **Safety fallback:** if all chunks are filtered, all are returned (context is never empty).

6. **Context-shrinkage guard** — Ensures the final context set does not drop below a minimum size threshold. Prevents edge cases where aggressive upstream filters produce an empty context window.

#### 3.5.4 ImprovedQueryEntityExtractor

Extrahiert Entitäten aus der Query konsistent mit der Ingestion-Zeit-Extraktion.

```python
class ImprovedQueryEntityExtractor:
    def __init__(self, gliner_model=None, spacy_model: str = "en_core_web_sm")
    def extract(self, query: str, confidence_threshold: float = 0.2) -> List[str]
```

**Wichtige Designentscheidungen (Stand 2026-04-02):**
- Lädt `gliner_small-v2.1` **eigenständig** via `_get_gliner_model()` wenn kein Modell übergeben wird (Prozess-Level-Cache)
- Verwendet **identische Entity Types wie Ingestion**: `["person", "organization", "city", "country", "film", "movie", "work of art", "event"]`
- **Threshold 0.2** (nicht 0.5): Queries sind kurze Sätze → GLiNER-Scores systematisch niedriger als bei langen Chunk-Texten
- **Modul-Level-Cache** `_GLINER_MODEL_CACHE`: Modell wird pro Prozess nur einmal geladen (7.5s cold start, danach <1ms)

> **Kritisches Problem bis 2026-04-02:** `StorageConfig(enable_entity_extraction=False)` ist der Default. `ImprovedQueryEntityExtractor` bekam dadurch `gliner_model=None` und fiel auf SpaCy zurück — obwohl GLiNER installiert ist. Symptom: `"Were Scott Derrickson"` statt `"Scott Derrickson"` als extrahierte Entität. Behoben durch eigenständiges Laden in `_load_gliner()`. Siehe Abschnitt 12.7.

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
```

> **`entity_hints` parameter:** When provided, the retriever uses these entity names directly for graph search instead of running GLiNER on the sub-query. This is critical for iterative multi-hop retrieval (Section 12.18), where bridge entities are discovered at runtime and passed to the next hop. Without this, GLiNER fails on short sub-query fragments (3–5 words) that lack context for reliable NER.

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
    retrieval_mode: str = "hybrid"
    vector_top_k: int = 20
    graph_top_k: int = 10
    max_hops: int = 2
    rrf_k: int = 60
    cross_source_boost: float = 1.2

    enable_pre_filtering: bool = True
    relevance_threshold_factor: float = 0.85   # Raised from 0.6; see Section 12.16
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
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 200            # Increased from 50; prevents answer truncation
    max_iterations: int = 2          # Initial generation + 1 correction round
    max_context_chars: int = 2400    # Total chars fed to LLM (~600 tokens)
    max_docs: int = 6                # Maximum number of context documents
    max_chars_per_doc: int = 500     # Characters per individual document (raised from 400)
```

> **Wichtig:** Alle `VerifierConfig`-Werte werden durch `_verifier_config_from_cfg()` in `agent_pipeline.py` aus `config/settings.yaml` gelesen. Die Klassen-Defaults sind Fallback-Werte und sollten nie direkt im Code überschrieben werden.

#### 4.3.2 ValidationStatus

```python
class ValidationStatus(Enum):
    VALID = "valid"               # Context is sufficient
    AMBIGUOUS = "ambiguous"       # Partial evidence
    CONFLICTED = "conflicted"     # Contradictory evidence
    INSUFFICIENT = "insufficient" # No evidence found
```

#### 4.3.3 Self-Correction Loop

Der Self-Correction-Loop befindet sich **vollständig innerhalb von `Verifier.generate_and_verify()`** und ist der wissenschaftliche Kernbeitrag von Artifact B.

```
# Verifier.generate_and_verify(query, context)

pre_validate(context) → PreValidationResult

if status == INSUFFICIENT:
    return fallback_answer("I cannot determine the answer from the provided context.")

# Schritt 1: Initiale Antwortgenerierung
context_str = build_context(context, max_chars=max_context_chars,
                            max_docs=max_docs, max_chars_per_doc=max_chars_per_doc)
answer = call_llm(GENERATION_PROMPT, query, context_str)   # phi3 via Ollama

# Schritt 2: Bis zu (max_iterations - 1) Korrektur-Runden
# Bei max_iterations=2: maximal 1 Korrektur-Runde
for iteration in range(1, max_iterations):
    is_valid, violations = verify(query, answer, context)  # NLI consistency check
    if is_valid:
        break
    # Korrektur-Prompt enthält konkrete Verletzungen als Feedback
    answer = call_llm(CORRECTION_PROMPT, query, context_str, violations)

return VerificationResult(answer, confidence, iterations_used)
```

> **Architekturentscheidung (2026-03-31):** Ein früherer äußerer Wiederholungs-Loop in `AgentPipeline.process()` wurde entfernt. Dieser rief `generate_and_verify()` mehrfach mit identischem Input auf, was keine echte Selbstkorrektur darstellte. Die Selbstkorrektur findet ausschließlich im inneren Loop des Verifiers statt, der bei jeder Iteration konkrete Verletzungen als Feedback übergibt (CORRECTION_PROMPT).

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

> **Architekturhinweis:** `AgentPipeline.process()` ruft `verifier.generate_and_verify()` **genau einmal** auf. Es gibt keinen äußeren Wiederholungs-Loop auf Pipeline-Ebene. Der Self-Correction-Mechanismus (Thesis-Beitrag) findet ausschließlich **innerhalb des Verifiers** statt. `max_verification_iterations` in `settings.yaml` steuert die Anzahl der Korrektur-Runden *im Verifier*, nicht die Anzahl der Pipeline-Durchläufe.

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
  vector_top_k: 10
  similarity_threshold: 0.3

# ── KNOWLEDGE GRAPH ────────────────────────────────────────────────────────
graph:
  backend: "kuzu"                   # Primary: KuzuDB (Cypher-native)
  graph_path: "./data/{dataset}/graph"  # Per-dataset KuzuDB directory
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
  max_tokens: 200              # Erhöht von 50 auf 200; verhindert Antwort-Truncation
  max_context_chars: 2400      # Total chars for LLM context (~600 tokens)
  max_docs: 6                  # Maximum context documents
  max_chars_per_doc: 500       # Characters per document (raised from 400 to reduce key-sentence loss)

# ── AGENTIC CONTROLLER ─────────────────────────────────────────────────────
agent:
  max_verification_iterations: 2   # 1 initial answer + 1 correction round
  enable_verification: true
  relevance_threshold_factor: 0.85  # PreGenerativeFilter: keep within 15% of top score
  redundancy_threshold: 0.8         # Jaccard dedup threshold

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

# Nur Chunks ingestieren, ohne Embeddings neu zu berechnen (schneller bei Re-Ingest)
python benchmark_datasets.py ingest \
  --dataset hotpotqa \
  --samples 500 \
  --chunks-only

# Single configuration evaluation
python benchmark_datasets.py evaluate \
  --dataset hotpotqa \
  --samples 100 \
  --model phi3 \
  --vector-weight 0.7 \
  --graph-weight 0.3

# Evaluation mit deaktivierten Komponenten (Ablation der Pipeline-Stufen)
python benchmark_datasets.py evaluate \
  --dataset hotpotqa \
  --samples 100 \
  --no-planner \     # S_P deaktiviert
  --no-verifier \    # S_V deaktiviert
  --iterations 1     # Keine Selbstkorrektur

# Ablation study (alle Gewichtungskombinationen)
python benchmark_datasets.py ablation \
  --dataset hotpotqa \
  --samples 100

# Komponenten-Ablation (Planner, Verifier, Iterationen)
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

**Graph-Quality Output (--graph-quality N)** zeigt pro Frage:

| Spalte | Bedeutung |
|---|---|
| `KW-H` | Anzahl Graph-Chunks per Keyword-Extraktion |
| `Top-Entity` | Welche Entität tatsächlich gematcht hat |
| `Hub` | `HUB` wenn generische/Pronomen-Entität gematcht hat |
| `GrAns` | ✓ wenn Gold-Antwort in Graph-Chunks vorkommt |
| `VecH` | Anzahl Vector-Chunks |
| `VecAns` | ✓ wenn Gold-Antwort in Vector-Chunks vorkommt |

**Empirische Ergebnisse:**

*Stand 2026-04-01 (threshold=0.15, UPPERCASE types, 15 Fragen):*

| Metrik | Wert |
|---|---|
| Hub-Kontamination (Keyword-Regex) | 0% |
| Graph Answer Coverage | 33% (5/15) |
| Vector Answer Coverage | 40% (6/15) |
| Union Coverage (Graph ODER Vector) | 60% (9/15) |
| GLiNER Answer Coverage | 13% (2/15) — schlechter als Keyword-Regex |

*Stand 2026-04-02 (threshold=0.5, lowercase types, Re-Ingestion, 10 Fragen):*

| Metrik | Wert | vs. vorher |
|---|---|---|
| Hub-Kontamination | 10% | ↓ besser |
| Graph Answer Coverage | 20% (2/10) | ↓ schlechter (aber N zu klein) |
| Vector Answer Coverage | 50% (5/10) | ↑ besser |
| Unique Entities | 23.858 | −35% (von 36.996) |
| MENTIONS | 50.096 | −33% (von 74.741) |

> **Interpretation:** Der Rückgang der Graph Answer Coverage ist auf (a) zu kleine Stichprobe (N=10 vs. N=15) und (b) Entity-Name-Disambiguation-Problem zurückzuführen (Chunk 7 "Edward Davis Wood Jr." nicht via "Ed Wood"-Query erreichbar). Statistisch nicht vergleichbar. Für belastbaren Vergleich: `python diagnose.py --graph-quality 50`.

Der Graph fügt für ≥3 Fragen exklusiven Mehrwert hinzu (GrAns=✓, VecAns=✗), was die Hybrid-Hypothese stützt.

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
  │  3. POST /api/generate → phi3 (Ollama)                  │
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
| RRF fusion | ~0.5 ms | Pure Python, O(N log N) |
| PreGenerativeFilter | ~1–3 ms | Jaccard + NLI |
| **Total retrieval** | **~40–110 ms** | Without LLM generation |
| LLM generation (phi3) | ~200 ms–62 s | CPU, 4-bit quantised; context size and KV-cache allocation dominate |
| **End-to-end** | **~250 ms–65 s** | Full pipeline; LLM is the dominant bottleneck on CPU |

> **LLM timeout note:** Ollama allocates the full KV-cache context window regardless of prompt length. On CPU, phi3 runs at approximately 8 tokens/second. A 2,400-character context (~600 tokens) can produce 60+ second responses. The system applies a 60-second `timeout` to Ollama HTTP calls; exceeding it returns the partial output. In practice, responses for bridge-resolved multi-hop queries with full context fit within 45 seconds.

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

---

## 12. Änderungen & Alternativen

Dieser Abschnitt dokumentiert signifikante Änderungen gegenüber früheren Versionen, verworfene Ansätze und die Begründung für getroffene Entscheidungen. Er dient als Entscheidungsprotokoll für die Masterarbeit.

---

### 12.1 LLM-Konfiguration: max_tokens und Context-Budget

**Problem:** In der initialen Konfiguration war `max_tokens: 50` gesetzt. Das führte dazu, dass Antworten des phi3-Modells mitten im Satz abgeschnitten wurden, da HotpotQA-Antworten häufig 10–30 Tokens benötigen, aber der Kontext-Aufbau (CoT-Reasoning) des Modells mehr Token-Budget erforderte.

**Symptom:** Ablation mit 10 Samples (2026-03-31) zeigte EM=20% für alle drei Konfigurationen (vector_only, hybrid, graph_only) — identische Ergebnisse trotz unterschiedlicher Retrieval-Methoden. Ursache: Alle Antworten wurden truncated, bevor sie die Gold-Antwort enthielten.

**Lösung:** `max_tokens: 200` (Faktor 4 Erhöhung). Zusätzlich wurden drei neue Context-Budget-Parameter eingeführt, die die Menge an Text steuern die dem LLM übergeben wird:
- `max_context_chars: 2400` — Verhindert Out-of-Context-Fehler bei phi3 (4k Context Window)
- `max_docs: 6` — Mehr Chunks erhöhen die Chance, beide Supporting Facts zu treffen (HotpotQA braucht 2)
- `max_chars_per_doc: 400` — Balanciert Breite (mehr Docs) vs. Tiefe (mehr Text pro Doc)

**Verworfene Alternative:** `max_tokens: 500` — getestet, führt zu längeren Antworten mit Halluzinationen ("According to the context, the answer is X, however it should be noted that..."). phi3 neigt bei mehr Token-Budget zu redundantem Output.

---

### 12.2 Entfernung des äußeren Pipeline-Retry-Loops

**Problem:** In einer früheren Version enthielt `AgentPipeline.process()` einen äußeren Loop der `verifier.generate_and_verify()` bis zu `max_verification_iterations` mal aufrief. Dieser Loop wurde unter der Annahme eingebaut, dass Wiederholungen mit identischem Input zu besseren Antworten führen.

**Erkenntnis:** Ein LLM mit identischem Prompt und `temperature=0.1` (fast-deterministisch) produziert bei jeder Iteration nahezu identische Outputs. Der äußere Loop war damit faktisch eine teure No-Op-Operation (mehrfache LLM-Calls ohne inhaltliche Verbesserung).

**Lösung (2026-03-31):** Äußerer Loop entfernt. `AgentPipeline.process()` ruft `generate_and_verify()` jetzt genau einmal auf. Der echte Self-Correction-Mechanismus ist der innere Loop im Verifier, der bei jeder Iteration **konkrete Verletzungen als Feedback** übergibt (CORRECTION_PROMPT mit spezifischen `violations`). Das ist der tatsächliche Thesis-Beitrag.

**Konfigurationskonsolidierung:** `max_verification_iterations` wird ausschließlich in `settings.yaml` unter `agent:` konfiguriert und von `_verifier_config_from_cfg()` gelesen. Es gibt keine weitere Stelle im Code wo dieser Wert festgelegt wird.

---

### 12.3 Graph-Visualisierung: pyvis → matplotlib

**Problem:** Das ursprüngliche `test_system/graph_3d.py` verwendete pyvis mit `LIMIT 200` für MENTIONS und `LIMIT 100` für RELATED_TO. Bei 66.585 MENTIONS und 14.445 RELATED_TO-Kanten im HotpotQA-Graph entsprach das 0,3% der Daten — der Graph sah leer und unstrukturiert aus.

**Lösung:** Komplette Neuentwicklung mit matplotlib + networkx:
- Lädt bis zu 2000 RELATED_TO-Kanten, filtert Hub-Entitäten (Pronomen, generische Terme)
- Wählt Top-N Entitäten nach Verbindungsgrad aus (konfigurierbares `--top N`)
- Speichert direkt als PNG (dpi=200, für Thesis dpi=300)
- Erzeugt optional weiterhin interaktives pyvis-HTML

**Erkenntnisse durch Analyse:**
- Keyword-Regex-Extraktion funktioniert gut (Hub-Kontamination: 0%)
- Graph Answer Coverage: 33% vs. Vector: 40% — Graph hat echten Mehrwert
- GLiNER Answer Coverage: 13% — schlechter als einfaches Keyword-Regex. Kein Re-Ingest erforderlich.

---

### 12.4 Graph-Qualität: REBEL-Relation-Extraktion

**Befund:** REBEL (`Babelscape/rebel-large`) extrahiert hauptsächlich Wikipedia-Infobox-Style-Relationen:
`publication_date`, `country`, `date_of_birth`, `genre`, `performer`

Diese Relationstypen sind für HotpotQA-Bridge-Fragen weitgehend nutzlos. HotpotQA benötigt narrative Relationen wie `directed_by`, `portrayed_by`, `held_position`.

**Kennzahlen HotpotQA-Graph:**

*Stand 2026-04-01 (threshold=0.15):*
- 9.412 DocumentChunks, 36.996 unique Entities, 74.741 MENTIONS, 17.221 RELATED_TO
- Hub-Entitäten: "American" (898 Chunks), "He" (737), "United States" (699), "She" (283)

*Stand 2026-04-02 (threshold=0.5, Re-Ingestion):*
- 9.412 DocumentChunks, 23.858 unique Entities, 50.096 MENTIONS, 15.766 RELATED_TO
- Hub-Kontamination deutlich reduziert (10% vs. zuvor hoch)

**Entscheidung für die Masterarbeit:** Keine Re-Ingestion mit anderem RE-Modell (Zeitaufwand GPU > 1h, unsicherer Gewinn). Stattdessen:
1. Graph-Limitierung als dokumentierte Limitation in der Thesis
2. Union-Coverage (60%) als Argument für Hybrid-Ansatz: Graph findet 3 Fragen die Vector verfehlt
3. Der Mehrwert des Graphs liegt in MENTIONS (66.585 Kanten), nicht in RELATED_TO

---

### 12.5 Benchmark-Stichprobengröße

**Problem:** Erste Ablation mit nur 10 Samples lieferte statistisch bedeutungslose Ergebnisse (EM=20% für alle Configs — könnte Zufall sein, N zu klein für Konfidenzintervalle).

**Mindestanforderung für Masterarbeit:** N≥50 für erste orientierendem Ergebnisse, N≥200 für belastbare Konfidenzintervalle (95% CI ≈ ±7pp bei N=200, ±14pp bei N=50).

**Empfohlene Konfiguration für finale Thesis-Ergebnisse:**
```bash
python benchmark_datasets.py ablation --dataset hotpotqa --samples 200
```

---

### 12.6 Entity Confidence Threshold: 0.15 → 0.5 (Re-Ingestion 2026-04-02)

**Problem:** Bei `ner_confidence_threshold=0.15` dominierten generische Tokens den Graphen:
- "American" (898 Chunks), "He" (737), "United States" (699), "She" (283)
Diese Entitäten wurden als Hub-Nodes mit hunderten MENTIONS-Kanten im Graphen gespeichert. Bei Graph-Traversal wurden sie als "matched entity" zurückgegeben, obwohl sie keinen semantischen Mehrwert liefern.

**Lösung:** `entity_confidence_threshold=0.5` in `local_importingestion.py`. Der Filter wird beim Import angewendet (nicht bei der Extraktion), sodass `extraction_results.json` unverändert bleibt und bei Bedarf mit anderem Threshold neu importiert werden kann.

**Datenvorfall:** Der erste `--clear`-Lauf verwendete `shutil.rmtree(base_path)` und löschte `data/hotpotqa/` vollständig (inkl. `chunks_export.json`, `questions.json`). `extraction_results.json` überlebte nur weil es im IDE geöffnet war. Behoben: `--clear` löscht jetzt nur `vector_db/` und `knowledge_graph/`, niemals Source-JSON-Dateien.

**Ergebnis:** 36.996 → 23.858 unique Entities (−35%), 74.741 → 50.096 MENTIONS (−33%).

---

### 12.7 GLiNER Query-Zeit-Konsistenz (2026-04-02)

**Problem:** `ImprovedQueryEntityExtractor` erhielt `gliner_model=None` weil `StorageConfig(enable_entity_extraction=False)` der Default ist. Folge: Jede Query lief mit SpaCy-Fallback statt GLiNER — obwohl GLiNER installiert und der Graph mit GLiNER gebaut ist.

**Symptom:** SpaCy extrahierte `"Were Scott Derrickson"` (Verb + Name) statt `"Scott Derrickson"`. Außerdem: SpaCy-Entity-Types (`PERSON`, `ORG`, `GPE`) stimmen nicht mit Graph-Entity-Types (`person`, `film`, `city`) überein → Name-Mismatch bei Graph-Lookup.

**Lösung:**
1. `_get_gliner_model()` — Modul-Level-Cache, lädt `gliner_small-v2.1` einmal pro Prozess
2. `_load_gliner()` nutzt Cache statt `GLiNER.from_pretrained()` direkt aufzurufen
3. Entity Types in `ImprovedQueryEntityExtractor` auf Ingestion-Types angeglichen: `["person", "organization", "city", "country", "film", "movie", "work of art", "event"]`
4. Default-Threshold: `0.5 → 0.2` (Queries kurz → niedrigere GLiNER-Scores als bei langen Chunk-Texten)
5. `diagnose.py` Layer 3: Regex-Extraktion durch `ImprovedQueryEntityExtractor` ersetzt

**Performance-Hinweis:** GLiNER cold start: ~7.5s, cached: <1ms. Beim ersten Aufruf pro Prozess einmalige Verzögerung.

---

### 12.8 Fallback-Sichtbarkeit: ⚠ FALLBACK AKTIV Warnings

**Motivation:** Fallbacks sind immer Symptome — sie bedeuten, dass die primäre Implementierung nicht funktioniert. Stille Fallbacks verbergen Fehler.

**Implementierung (2026-04-02):** Alle bekannten Fallback-Pfade emittieren jetzt `logger.warning("⚠ FALLBACK AKTIV: ...")`:

| Datei | Fallback-Pfad |
|---|---|
| `hybrid_retriever.py` | SpaCy/Regex statt GLiNER für Query-Entities |
| `entity_extraction.py` | SpaCy/Regex statt GLiNER für Batch-Extraction |
| `ingestion_pipeline.py` | MockEntityExtractor / MockEmbeddingGenerator (use_mocks=True) |
| `verifier.py` | Heuristische Widerspruchserkennung statt NLI |
| `navigator.py` | Relativer Import-Fallback |
| `chunking.py` | LangChain nicht installiert |

---

### 12.9 Bekannte Limitierungen: Entity Name Disambiguation

**Befund (2026-04-02):** Das System versagt bei Fragen wo die Query eine Entität als Kurzname verwendet, der Graph sie aber unter dem vollständigen Namen gespeichert hat.

**Konkretes Beispiel:**
- Query: "Were Scott Derrickson and **Ed Wood** of the same nationality?"
- GLiNER extrahiert: `"Ed Wood"` → Graph-Lookup findet Entity "Ed Wood" → FILM-Artikel (1994 Tim-Burton-Film)
- Korrekte Chunk 7: `"Edward Davis Wood Jr. was an American filmmaker"` → hängt an Entity `"Edward Davis Wood Jr."` → wird **nicht** gefunden

**Ursache:** KuzuDB-Graph-Lookup ist **exaktes String-Matching**. Keine Alias-Auflösung, keine Coreference.

**Mögliche Lösungen (für Thesis-Ausblick):**

| Ansatz | Aufwand | Wirkung |
|---|---|---|
| Entity-Linking (z.B. BLINK, REL) | hoch | Normalisiert Entitäten auf Wikidata-IDs |
| Alias-Tabelle bei Ingestion | mittel | `"Ed Wood"` ↔ `"Edward Davis Wood Jr."` als Alias-Node |
| Fuzzy Graph-Lookup | niedrig | Levenshtein-Abstand bei Graph-Suche erlauben |
| Sub-Query-Reformulierung | niedrig | "What is the nationality of Ed Wood?" statt "Were Ed Wood of same nationality?" |

Für die Masterarbeit wird diese Limitierung als dokumentierte Known Issue behandelt. Sie ist in der Union-Coverage-Metrik sichtbar und motiviert den Hybrid-Ansatz.

---

### 12.10 Planner Sub-Query Rewriting für Comparison-Queries (2026-04-02)

**Problem:** Comparison-Queries wie `"Were Scott Derrickson and Ed Wood of the same nationality?"` erzeugen ohne Rewriting Sub-Queries wie `"Were Scott Derrickson of the same nationality as Ed Wood?"`. Diese komplex formulierten Sub-Queries haben geringe Vektor-Ähnlichkeit zu Fakten-Chunks.

**Lösung:** `_ATTR_MAP` mit 8 Regex-Patterns in `src/logic_layer/planner.py` `_decompose_comparison()`:

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

**Ergebnis:** `"Were Scott Derrickson and Ed Wood of the same nationality?"` erzeugt jetzt:
1. `"What is the nationality of Scott Derrickson?"` → direkter Treffer ✓
2. `"What is the nationality of Ed Wood?"` → kein Treffer (Disambiguierungs-Problem 12.9)
3. Original-Query als Kontext-Fallback

---

### 12.11 Navigator Entity-Mention Filter (2026-04-02)

**Problem:** Für Sub-Queries wie `"What is the nationality of Ed Wood?"` hat `nomic-embed-text` hohe Ähnlichkeit zu beliebigen Nationalitäts-Artikeln. Beispiel: `"British people, or Britons..."` Score 0.798, relevanter Ed-Wood-Chunk nur 0.649 — unterhalb aller Ergebnisse.

**Lösung:** `_entity_mention_filter()` als Filter 5 in `_navigate()` der `Navigator`-Klasse (`src/logic_layer/navigator.py`):

- Multi-Word Entities (z.B. `"Scott Derrickson"`): prüfe vollständige Phrase **oder** einzelne Tokens ≥5 Zeichen als ganze Wörter (`\bscott\b`, `\bderrickson\b`)
- Single-Token Entities: nur prüfen wenn ≥5 Zeichen (vermeidet False Positives durch `"Were"`, `"Wood"`)
- **Safety-Fallback:** wenn alle Chunks gefiltert werden → behalte alle (kein leerer Kontext)

**Ergebnis (Frage 0):**
- `"British people"` → entfernt ✓ (kein "Scott", "Derrickson")
- `"The Oku people"` → entfernt ✓ (kein Match, vorher durch "Were"-Entity falsch behalten)
- `"Tyler Bates"` → behalten ✓ (enthält "Scott" im vollen Chunk-Text, da Tyler Bates mit Scott Derrickson zusammenarbeitete)
- 20 raw → 5 relevante Chunks (vorher 6 mit 2 irrelevanten)

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

*End of Technical Architecture Documentation*

---

## Test Suite Summary

| File | Tests | Status |
|---|---|---|
| `src/data_layer/test_data_layer.py` | 33 | ✓ |
| `src/logic_layer/test_logic_layer.py` | ~50 | ✓ |
| `src/pipeline/test_pipeline.py` | 54 | ✓ |
| `test_system/test_chunking.py` | 29 | ✓ |
| `test_system/test_embeddings.py` | 34 | ✓ |
| **Total** | **~200** | **✓** |

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
