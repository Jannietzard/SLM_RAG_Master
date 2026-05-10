# Technical Architecture

**Project:** Enhancing Reasoning Fidelity in Quantized Small Language Models on Edge Devices via Hybrid Retrieval-Augmented Generation
**Author:** Jan Nietzard
**Institution:** FOM Hochschule, Master of Science

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
   - 4.4 [Agentic Controller](#44-agentic-controller)
5. [Pipeline Layer](#5-pipeline-layer)
   - 5.1 [Agent Pipeline](#51-agent-pipeline)
   - 5.2 [Ingestion Pipeline](#52-ingestion-pipeline)
6. [Configuration System](#6-configuration-system)
7. [Evaluation Framework — Artifact C](#7-evaluation-framework--artifact-c)
8. [Technology Stack](#8-technology-stack)
9. [Data Flows](#9-data-flows)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Non-Functional Requirements](#11-non-functional-requirements)
12. [Design Decisions & Trade-offs](#12-design-decisions--trade-offs)

---

## 1. System Overview

This system implements a hybrid Retrieval-Augmented Generation (RAG) architecture optimised for edge deployment on resource-constrained hardware. The central research hypothesis is that combining dense vector retrieval, sparse term-frequency retrieval, and structured knowledge graph traversal — mediated by a three-agent reasoning pipeline — increases answer fidelity over any single modality, particularly for multi-hop reasoning tasks.

The architecture is organised into three independently testable artifact layers:

| Artifact | Layer | Responsibility |
|---|---|---|
| **A** | Data Layer | Dual-index storage (vectors + graph), batched embeddings, hybrid retrieval with Reciprocal Rank Fusion (RRF), BM25 sparse retrieval, optional cross-encoder reranking |
| **B** | Logic Layer | Three-agent reasoning pipeline: Planner → Navigator → Verifier, with optional iterative multi-hop bridge resolution |
| **C** | Evaluation | Multi-dataset benchmarking (HotpotQA, 2WikiMultiHopQA, StrategyQA), ablation runner, diagnostic tooling |

**Edge-deployment constraints:**
- All databases are *embedded* (no server process): LanceDB for vectors, KuzuDB for the property graph, SQLite for the embedding cache.
- All language models are served locally via Ollama; no cloud API dependency.
- The system is designed to operate within < 16 GB of host RAM.
- Generation runs on CPU-only hardware using 4-bit GGUF quantisation (llama.cpp backend).

---

## 2. Repository Structure

```
Entwicklungfolder/
│
├── src/                            # Production source code
│   ├── data_layer/                 # Artifact A: storage, retrieval, ingestion
│   │   ├── __init__.py             # Public exports
│   │   ├── embeddings.py           # BatchedOllamaEmbeddings + SQLite cache
│   │   ├── chunking.py             # SpacySentenceChunker, SemanticChunker
│   │   ├── entity_extraction.py    # GLiNER NER + REBEL RE pipeline
│   │   ├── entity_types.py         # Canonical label maps (single source of truth)
│   │   ├── storage.py              # HybridStore, VectorStoreAdapter, KuzuGraphStore
│   │   ├── hybrid_retriever.py     # HybridRetriever, RRFFusion, BM25, query NER
│   │   ├── ingestion.py            # DocumentIngestionPipeline (data-layer side)
│   │   └── conftest.py             # Adds project root to sys.path
│   │
│   ├── logic_layer/                # Artifact B: agentic reasoning
│   │   ├── __init__.py             # Public exports
│   │   ├── _config.py              # ControllerConfig (shared dataclass)
│   │   ├── _settings.py            # YAML loader + shared regex constants
│   │   ├── planner.py              # S_P: query analysis & plan generation
│   │   ├── navigator.py            # S_N: hybrid orchestration + filter chain
│   │   ├── verifier.py             # S_V: pre-validation, generation, self-correction
│   │   ├── controller.py           # AgenticController: LangGraph or sequential fallback
│   │   └── conftest.py
│   │
│   ├── pipeline/                   # Orchestration layer
│   │   ├── __init__.py
│   │   ├── agent_pipeline.py       # AgentPipeline: thin S_P → S_N → S_V chain
│   │   ├── ingestion_pipeline.py   # End-to-end ingestion workflow
│   │   └── conftest.py
│   │
│   ├── evaluations/                # Artifact C: thesis evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py              # Canonical EM / F1 (HotpotQA-compatible)
│   │   ├── evaluate_hotpotqa.py    # HotpotQA benchmark runner
│   │   ├── ablation_study.py       # Ablation runner (configurable matrix)
│   │   ├── ollama_performance_diagnostic.py
│   │   └── test_rag_quality.py     # Standalone retrieval-quality probe
│   │
│   └── utils.py                    # Shared utilities (jaccard_similarity)
│
├── config/
│   └── settings.yaml               # Single source of truth for all parameters
│
├── prompts/                        # External prompt templates (review/audit)
│
├── data/                           # Runtime data (gitignored)
│   ├── hotpotqa/
│   │   ├── vector/                 # LanceDB directory
│   │   ├── graph/                  # KuzuDB directory
│   │   ├── chunks_export.json      # Phase-1 chunking output
│   │   ├── extraction_results.json # Phase-2 extraction output (Colab GPU)
│   │   └── questions.json
│   ├── 2wikimultihop/              # (same structure)
│   └── strategyqa/                 # (same structure)
│
├── cache/                          # SQLite embedding caches (gitignored)
├── evaluation_results/             # JSON ablation outputs
├── logs/                           # Structured log output
│
├── benchmark_datasets.py           # CLI entry point: ingest / evaluate / ablation
├── diagnose.py                     # Layer-by-layer pipeline diagnostic
├── diagnose_verbose.py             # Trace-mode diagnostic with gold tracking
├── diagnose_ingestion.py           # Ingestion consistency checker
├── local_importingestion.py        # Phase-3 import: chunks + extraction → stores
│
├── test_system/                    # Test suite + visualisation tooling
│   ├── conftest.py                 # Pytest fixtures, sys.path setup
│   ├── fixtures/                   # Gold NER and similar reference data
│   ├── test_*.py                   # Active test modules (see §11.4)
│   ├── diagnose_*.py               # GLiNER / SpaCy / NER diagnostics
│   ├── graph_inspect.py            # KuzuDB schema/statistics inspector
│   └── graph_3d.py                 # matplotlib + pyvis graph visualisation
│
├── pytest.ini                      # CI markers: slow, nightly, llm, integration
├── requirements.txt                # Range constraints
├── requirements_frozen.txt         # Pinned reproducibility set
└── REPRODUCE.md                    # Reproduction protocol
```

**Decoupled ingestion architecture (3-phase):**

| Phase | Tool | Output |
|---|---|---|
| 1 | `benchmark_datasets.py ingest` | `chunks_export.json` (chunks + metadata) |
| 2 | Google Colab (GPU) | `extraction_results.json` (GLiNER + REBEL output) |
| 3 | `local_importingestion.py` | LanceDB + KuzuDB populated from Phase-1 and Phase-2 outputs |

This decoupling allows GPU-bound entity extraction to run separately from CPU-only ingestion on the edge target.

---

## 3. Data Layer — Artifact A

The data layer encapsulates all operations on documents, embeddings, indices, and retrieval. It is stateless with respect to any particular query; persistent state lives entirely on disk.

### 3.1 Embedding Module

**File:** [src/data_layer/embeddings.py](src/data_layer/embeddings.py)

#### 3.1.1 Architecture

The module wraps the Ollama HTTP API and adds two performance-critical primitives: *batched inference* and *content-addressable persistent caching*.

**`EmbeddingCache`** — SQLite-backed (WAL mode) persistent cache:

```
embeddings (
  text_hash     TEXT PRIMARY KEY,   -- SHA-256(text.encode('utf-8'))
  text_content  TEXT NOT NULL,
  embedding     BLOB NOT NULL,      -- JSON-serialised float list
  model_name    TEXT NOT NULL,
  access_count  INTEGER DEFAULT 0,
  created_at    TIMESTAMP
)
INDEX idx_model_hash ON (model_name, text_hash)
```

Lookup is O(1) on the SHA-256 primary key. Identical input strings are deduplicated at hash level, guaranteeing single embedding per unique text — even across separate process invocations. The schema is keyed by `(model_name, text_hash)`, so changing the embedding model invalidates cached vectors automatically.

**`EmbeddingMetrics`** (dataclass) tracks `total_texts`, `cache_hits`, `cache_misses`, `batch_count`, and `total_time_ms`; the `cache_hit_rate` property is derived.

**`BatchedOllamaEmbeddings`** — LangChain `Embeddings`-compatible interface:

```python
class BatchedOllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 64,
        cache_path: Path = Path("./cache/embeddings.db"),
        device: str = "cpu",
        timeout: int = 60,
    ): ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...
    def get_metrics(self) -> Dict[str, Any]: ...
    def clear_cache(self) -> None: ...
```

#### 3.1.2 Batching Algorithm

```
Input:  texts[0..N-1]
1. CACHE LOOKUP (single SQL query):
     hashes = [SHA256(t) for t in texts]
     hits   = SELECT text_hash, embedding FROM embeddings
              WHERE model_name = ? AND text_hash IN (hashes)
2. IDENTIFY MISSES.
3. BATCH API CALLS in chunks of `batch_size`.
4. WRITE-BACK to cache, then assemble results in original order.
```

The factory `create_embeddings(cfg)` constructs the client from a `settings.yaml` dict.

---

### 3.2 Document Chunking

**File:** [src/data_layer/chunking.py](src/data_layer/chunking.py)

Two strategies are implemented; the thesis benchmarks use the SpaCy sentence chunker.

#### 3.2.1 SpacySentenceChunker

Sliding window over SpaCy sentence boundaries:

```python
class SpacySentenceChunker:
    def __init__(
        self,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 1,
        min_chunk_chars: int = 50,
        spacy_model: str = "en_core_web_sm",
    ): ...
    def chunk_text(self, text: str, source_doc: str = "") -> List[SentenceChunk]: ...
    def chunk_document(self, document: Document) -> List[Document]: ...
```

The output `SentenceChunk` carries `text`, `sentence_count`, `position`, `start_char`, `end_char`, and `source_doc`. With the default `sentences_per_chunk=3` and `sentence_overlap=1`, evidence spanning a sentence boundary is represented in at least one chunk without full-chunk duplication. Chunk IDs are deterministic SHA-256 hashes of `source_doc:position:text[:50]` — re-ingestion of the same source produces identical IDs and is therefore idempotent.

A module-level `SpacyModelCache` singleton avoids repeated 200–300 ms model load times when multiple chunker instances are created.

#### 3.2.2 SemanticChunker

Structure-aware chunking using TF-IDF importance scoring and header detection:

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
    ): ...
```

The `AutomaticQualityFilter` rejects chunks with type-token ratio < 0.3 or Shannon entropy < 2.0 bits. This strategy is intended for long-form technical documents; on short HotpotQA passages, the sentence chunker is preferred.

---

### 3.3 Entity Extraction

**File:** [src/data_layer/entity_extraction.py](src/data_layer/entity_extraction.py)

Entity extraction populates the knowledge graph with named entities and their relations, enabling the graph retrieval path.

#### 3.3.1 Named Entity Recognition

**Model:** GLiNER (`urchade/gliner_small-v2.1`) — zero-shot span-based NER, no task-specific fine-tuning required.

**`ExtractionConfig`** key fields:

```python
@dataclass
class ExtractionConfig:
    gliner_model: str = "urchade/gliner_small-v2.1"
    entity_types: List[str] = [
        # Lowercase natural-language labels yield better zero-shot performance.
        "person", "organization", "city", "country", "state",
        "location", "film", "movie", "album", "work of art",
        "landmark", "event", "award",
    ]
    ner_confidence_threshold: float = 0.15   # Recall-optimised for HotpotQA
    ner_batch_size: int = 16
    rebel_max_input_length: int = 256
    rebel_max_output_length: int = 256
    rebel_num_beams: int = 5
    device: str = "cpu"
```

**`ExtractedEntity`** carries `entity_id` (24-character SHA-256 hex), `name` (normalised surface form), `entity_type` (one of the canonical 13), `confidence`, `mention_span`, and `source_chunk_id`. The serialised key for the type field is `"entity_type"` (not `"type"`).

A fallback chain ensures availability: GLiNER → SpaCy NER → regex. Each fallback hop emits `logger.warning("FALLBACK ACTIVE: …")` so silent degradations are observable in production logs.

#### 3.3.2 Relation Extraction

**Model:** REBEL (`Babelscape/rebel-large`) — seq2seq generator that emits `(subject, relation, object)` triples directly from text. RE is applied *conditionally*: only to chunks containing ≥ 2 extracted entities, reducing compute by ~60–70 % on Wikipedia-style corpora.

**`ExtractedRelation`** carries `subject_entity`, `relation_type`, `object_entity`, `confidence`, and `source_chunk_ids`. Serialised keys mirror the field names.

#### 3.3.3 Pipeline Interface

```python
class EntityExtractionPipeline:
    def __init__(self, config: ExtractionConfig): ...
    def extract_chunk(self, text: str) -> ChunkExtractionResult: ...
    def extract_batch(self, texts: List[str]) -> List[ChunkExtractionResult]: ...
```

A two-tier `EntityCache` (in-memory LRU + SQLite persistent store) keys cached results on `(text_hash, model_name)` so a model change automatically invalidates entries.

#### 3.3.4 Canonical Label Maps

**File:** [src/data_layer/entity_types.py](src/data_layer/entity_types.py) — single source of truth for label normalisation.

| Map | Direction |
|---|---|
| `GLINER_LABEL_MAP` | lowercase GLiNER label → canonical type |
| `SPACY_LABEL_MAP` | uppercase SpaCy NER label → canonical type (preserves `GPE`) |
| `SPACY_LABEL_MAP_FLAT` | uppercase SpaCy NER label → canonical type (flattens `GPE → LOCATION`) |

Both ingestion-time and query-time entity normalisation route through `normalize_entity_name()` and these maps, ensuring identical canonicalisation across the two phases.

---

### 3.4 Storage Layer

**File:** [src/data_layer/storage.py](src/data_layer/storage.py)

The storage layer provides a unified `HybridStore` façade over two physically separate indices: a LanceDB vector store and a KuzuDB property graph. Both are embedded; no server process is required.

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
    graph_backend: str = "kuzu"          # "kuzu" only in current build
    enable_entity_extraction: bool = False
```

LanceDB defaults to L2 distance; on normalised text embeddings this yields systematically lower similarity scores than cosine. The `distance_metric` field explicitly overrides the LanceDB default — a precondition for retrieval correctness.

#### 3.4.2 VectorStoreAdapter

```python
class VectorStoreAdapter:
    def __init__(self, db_path: Path, embedding_dim: int, distance_metric: str = "cosine"): ...
    def add_documents(self, documents: List[Document]) -> None: ...
    def vector_search(
        self, query_embedding: List[float], top_k: int = 10, threshold: float = 0.0,
    ) -> List[Dict[str, Any]]: ...
```

`vector_search()` returns dicts with the schema:

```python
{
    "document_id": str,
    "text": str,
    "similarity": float,        # ∈ [0, 1] after distance→similarity conversion
    "metadata": {"source_file": str, "chunk_index": int, "page_number": int, ...},
}
```

#### 3.4.3 KuzuGraphStore

Wraps KuzuDB with a domain-specific schema:

| Node | Key properties |
|---|---|
| `DocumentChunk` | `chunk_id`, `text`, `page_number`, `chunk_index`, `source_file` |
| `SourceDocument` | `doc_id`, `filename`, `total_pages` |
| `Entity` | `entity_id`, `name`, `entity_type`, `confidence` |

| Edge | Source → Target |
|---|---|
| `FROM_SOURCE` | `DocumentChunk → SourceDocument` |
| `NEXT_CHUNK` | `DocumentChunk → DocumentChunk` (sequential adjacency) |
| `MENTIONS` | `DocumentChunk → Entity` |
| `RELATED_TO` | `Entity → Entity` (with `relation_type`, `confidence`) |

Public methods include `add_chunk_node`, `add_entity_node`, `add_relation`, `find_chunks_by_entity_multihop(entity_name, max_hops)`, and `get_statistics`. The multi-hop helper applies a lightweight name-variant heuristic: for two-token names whose first token is ≤ 3 characters (e.g., `"Ed Wood"`), it also queries the surname; for any name, it tries individual tokens of length ≥ 4. Full alias resolution (e.g., `"Ed Wood"` ↔ `"Edward Davis Wood Jr."`) requires an entity-linking system and is documented as a known limitation (§12.3).

#### 3.4.4 HybridStore

The unified façade consumed by all higher layers:

```python
class HybridStore:
    def __init__(self, config: StorageConfig, embeddings: BatchedOllamaEmbeddings): ...
    def add_documents(self, documents: List[Document]) -> None: ...
    def vector_search(self, query_embedding, top_k, threshold) -> List[Dict]: ...
    def graph_search(self, entities: List[str], max_hops: int, top_k: int) -> List[Dict]: ...
    def close(self) -> None: ...
    @property
    def vector_store(self) -> VectorStoreAdapter: ...
    @property
    def graph_store(self) -> KuzuGraphStore: ...
```

`graph_search()` results carry `chunk_id`, `text`, `source_file`, `matched_entity` (the entity that triggered the result), and `hops` (graph distance). `close()` releases both database handles cleanly — important for long-running batch evaluations.

---

### 3.5 Hybrid Retriever

**File:** [src/data_layer/hybrid_retriever.py](src/data_layer/hybrid_retriever.py)

The hybrid retriever combines three retrieval modalities — dense vector ANN, BM25 sparse, and graph traversal — using Reciprocal Rank Fusion (RRF). The Navigator is the primary consumer.

#### 3.5.1 RetrievalConfig

```python
@dataclass
class RetrievalConfig:
    mode: RetrievalMode = RetrievalMode.HYBRID    # VECTOR | GRAPH | HYBRID
    vector_top_k: int = 20
    graph_top_k: int = 10
    similarity_threshold: float = 0.3
    rrf_k: int = 60                               # Cormack et al. (2009)
    cross_source_boost: float = 1.2               # Bonus for multi-lane chunks
    final_top_k: int = 10
    enable_bm25: bool = True
    bm25_top_k: int = 20
    query_ner_confidence: float = 0.15
    query_entity_types: List[str] = field(default_factory=list)
    gliner_model_name: str = "urchade/gliner_small-v2.1"
```

Modality ablation is performed by switching `mode` to `VECTOR` or `GRAPH`; weighted-fusion knobs are intentionally absent because RRF is rank-based and requires no scalar weighting.

#### 3.5.2 Reciprocal Rank Fusion

`RRFFusion` implements the Cormack et al. (2009) formulation, extended for three retrieval lanes:

```
RRF(d) = Σ_{r ∈ {vector, graph, bm25}} 1 / (k + rank_r(d))  +  BONUS(d)

k     = 60  (standard, parameter-robust)
rank_r = position of d in result list r (1-indexed)
BONUS = cross_source_boost / (k + 1)  per pair of lanes containing d
      = 0  otherwise
```

The boost is *additive* (preserving interpretable score ranges) and calibrated to equal one additional rank-1 vote independently of `k`. A chunk surfacing in multiple lanes receives at most one cross-source bonus per pair of lanes that match.

```python
class RRFFusion:
    def __init__(self, k: int = 60, cross_source_boost: float = 1.2): ...
    def fuse(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
        final_top_k: int = 10,
        bm25_results: Optional[List[Dict]] = None,
    ) -> List[RetrievalResult]: ...
```

`RetrievalResult` (output dataclass) carries `chunk_id`, `text`, `source_doc`, `position`, `rrf_score`, the per-lane scores and ranks (`vector_score/rank`, `graph_score/rank`, `bm25_score/rank`), `retrieval_method` (`"vector" | "graph" | "bm25" | "hybrid"`), `hop_distance`, and `matched_entities`.

#### 3.5.3 BM25 Sparse Retrieval

BM25 (`rank_bm25`) is the third RRF lane. The index is built lazily from a cached pandas DataFrame of the LanceDB table on the first query (no extra disk scan). BM25 scores are normalised to `[0, 1]` for clean RRF integration. Because BM25 operates on raw term-frequency statistics, it is unaffected by the *score compression* exhibited by `nomic-embed-text` (where most text-pair similarities cluster in the 0.74–0.78 band). Exact-match queries (award names, titles, identifiers) that embed poorly therefore surface reliably via BM25.

#### 3.5.4 ImprovedQueryEntityExtractor

Extracts entities from queries consistently with ingestion-time extraction:

```python
class ImprovedQueryEntityExtractor:
    def __init__(self, gliner_model=None, spacy_model: str = "en_core_web_sm"): ...
    def extract(self, query: str, confidence_threshold: float = 0.2) -> List[str]: ...
```

Design properties:
- Loads `gliner_small-v2.1` independently via `_get_gliner_model()` (process-level cache with double-checked locking) when no model is supplied.
- Uses the **same entity types as ingestion** to guarantee consistent normalisation.
- Default threshold is **0.2** rather than 0.5: queries are short, so GLiNER scores are systematically lower than on chunk-length text.
- A module-level cache (`_GLINER_MODEL_CACHE`) keeps the cold-start cost (~7.5 s) to once per process; thereafter calls return in < 1 ms.
- Handles offline / air-gapped environments by retrying with `HF_HUB_OFFLINE=1` if the initial load fails on a network call.

#### 3.5.5 HybridRetriever

```python
class HybridRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        hybrid_store: HybridStore,
        embeddings: BatchedOllamaEmbeddings,
    ): ...
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        entity_hints: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]: ...
```

The `entity_hints` parameter is essential for iterative multi-hop retrieval: when supplied, the retriever uses the provided entity names directly for graph search, bypassing GLiNER on short sub-query fragments where NER is unreliable. Bridge entities discovered at runtime by the controller are passed through this parameter to subsequent hops.

`RetrievalMetrics` carries `total_time_ms`, `vector_time_ms`, `graph_time_ms`, `n_vector_results`, `n_graph_results`, `n_final_results`, and `retrieval_mode`.

> **Filter ownership.** Pre-generative filtering is a **Logic Layer** concern owned by the Navigator (§4.2). The data layer is intentionally filter-free: it returns ranked candidates, not curated context.

---

## 4. Logic Layer — Artifact B

The logic layer implements a three-agent reasoning pipeline. Each agent is independently instantiable and testable. Agents communicate via typed dataclass contracts.

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

**File:** [src/logic_layer/planner.py](src/logic_layer/planner.py)

The Planner analyses the incoming query and produces a structured `RetrievalPlan`. Classification is rule-based (SpaCy `Matcher` over dependency parses + NER), keeping latency below 10 ms.

#### 4.1.1 Query Classification

| Type | Description | Indicator example |
|---|---|---|
| `SINGLE_HOP` | Direct fact lookup | Simple subject–predicate–object |
| `MULTI_HOP` | Bridge entity required | "Who directed X and where was he born?" |
| `COMPARISON` | Comparison of two entities | "Which is larger: X or Y?" |
| `TEMPORAL` | Time-constrained reasoning | "When did…", "Before/after…" |
| `AGGREGATE` | Set/count operations | "How many…", "All countries that…" |
| `INTERSECTION` | Common attributes | "What do X and Y have in common?" |

#### 4.1.2 RetrievalPlan

```python
@dataclass
class RetrievalPlan:
    original_query: str
    query_type: QueryType
    strategy: RetrievalStrategy        # VECTOR_ONLY | GRAPH_ONLY | HYBRID
    confidence: float
    entities: List[EntityInfo]         # SpaCy-extracted entities
    hop_sequence: List[HopStep]
    sub_queries: List[str]
    temporal_constraints: Dict[str, Any]
    comparison_pairs: List[Tuple[str, str]]
    estimated_hops: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached_answer: Optional[str] = None   # Early-exit short-circuit
```

`EntityInfo` carries `text`, `label`, `confidence`, character offsets, and an `is_bridge` flag. `HopStep` carries `step_id`, `sub_query`, `target_entities`, `depends_on` (other `step_id`s that must complete first), and `is_bridge`.

#### 4.1.3 Sub-Query Decomposition

The `_decompose_comparison()` rewrites comparison queries into per-entity factual sub-queries via a fixed regex map (`_ATTR_MAP`):

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

Multi-hop decomposition follows pattern matchers C, D, and E:
- **Pattern C** detects `"for a/an/the [film|movie|show|…]"` constructions.
- **Pattern D** detects `"[role] with [qualifier] co-wrote/directed/…"` constructions where the bridge is a work, not a person.
- **Pattern E** is a relational-anchor fallback: when no other split matches, `the [role] of [Entity]` is decomposed into a bridge sub-query (`"Who is the {role} of {anchor}?"`) followed by the original query as the final hop. The role vocabulary is the class constant `_RELATIONAL_ANCHOR_ROLES` (~25 nouns).

#### 4.1.4 Strategy Selection

```
if query_type == MULTI_HOP or any(entity.is_bridge for entity in entities):
    strategy = HYBRID
elif query_type in {SINGLE_HOP, COMPARISON} and len(entities) >= 2:
    strategy = HYBRID
elif graph not available:
    strategy = VECTOR_ONLY
else:
    strategy = HYBRID
```

> **Known limitation.** SpaCy's `en_core_web_sm` is loaded at module import time using a hard-coded default. The `PlannerConfig.spacy_model` setting (read after import) cannot override this. The active default matches `settings.yaml`, so the limitation has no current effect; a deferred lazy-load refactor would lift it.

---

### 4.2 Navigator Agent (S_N)

**File:** [src/logic_layer/navigator.py](src/logic_layer/navigator.py)

The Navigator executes the plan produced by S_P, performs RRF fusion across sub-query result lists, and applies the pre-generative filter chain that produces the context window for S_V.

#### 4.2.1 ControllerConfig

`ControllerConfig` (defined in [_config.py](src/logic_layer/_config.py)) is the shared configuration dataclass that backs both the Navigator and the AgenticController:

```python
@dataclass
class ControllerConfig:
    # LLM settings (emergency fallbacks; live values from settings.yaml)
    model_name: str = "qwen2:1.5b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    # Pipeline
    max_verification_iterations: int = 1   # 1 = no self-correction (default)

    # Navigator: pre-generative filtering
    relevance_threshold_factor: float = 0.6
    redundancy_threshold: float = 0.8
    max_context_chunks: int = 8
    rrf_k: int = 60
    top_k_per_subquery: int = 20
    max_chars_per_doc: int = 800

    # Cross-source corroboration weights (RRF fusion)
    corroboration_source_weight: float = 0.1
    corroboration_query_weight: float = 0.05

    # Numeric contradiction filter
    contradiction_overlap_threshold: float = 0.3
    contradiction_ratio_threshold: float = 2.0
    contradiction_min_value: float = 100.0

    # Stage 2.5 cross-encoder reranker
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "ControllerConfig": ...
```

#### 4.2.2 NavigatorResult

```python
@dataclass
class NavigatorResult:
    filtered_context: List[str]      # Aligned with `scores`
    raw_context: List[str]           # Pre-filter chunks from RRF fusion
    scores: List[float]
    metadata: Dict[str, Any]         # Per-filter counts, provenance
```

`filtered_context` is the field consumed by `benchmark_datasets.py` to compute *retrieval coverage*: a query is "covered" iff `len(filtered_context) > 0`.

#### 4.2.3 Retrieval Execution

```
SIMPLE PATH (no bridge dependencies):
  For each sub_query in plan.hop_sequence:
    a. embed(sub_query) → q_emb
    b. vector_search(q_emb, top_k=20)
    c. bm25_search(sub_query, top_k=20)            (if enabled)
    d. graph_search(entity_hints, max_hops=2)
  → RRF fusion across sub-queries → filter chain → top-K chunks

ITERATIVE PATH (any HopStep.depends_on is non-empty):
  Sort hop_sequence by step_id
  current_hints      = plan.entities
  accumulated_chunks = []

  for step in sorted_hops:
    sub_results = retrieve(step.sub_query, entity_hints=current_hints)
    accumulated_chunks += deduplicate(sub_results)

    if step.is_bridge:
      bridge = _extract_bridge_entities(sub_results, exclude=query_tokens)
      current_hints = current_hints ∪ bridge      # capped at 3 new entities

  → filter chain on accumulated_chunks → top-K chunks
```

The iterative path resolves the *hidden bridge entity problem*: the answer to hop 0 becomes the graph-search key for hop 1, enabling retrieval of an answer document even when the bridge entity was unknown at query time.

#### 4.2.4 Pre-Generative Filter Chain

After RRF fusion, the Navigator runs **six sequential filters**, with an **optional Stage 2.5 cross-encoder reranker** between fusion and Stage 1:

| Stage | Filter | Mechanism |
|---|---|---|
| **2.5** | Cross-encoder reranker (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores top-K fused chunks by `(query, chunk)` relevance. Lazy-loaded; ~22 MB; ~30 ms per pair on CPU. Toggle via `navigator.enable_reranker`. |
| **1** | Relevance filter | Keep chunks with `rrf_score ≥ relevance_threshold_factor × max_score`. Default factor `0.6` is calibrated for `nomic-embed-text` score compression. |
| **2** | Redundancy filter | Drop pairs with Jaccard token-set similarity > `redundancy_threshold` (default `0.8`). |
| **3** | Contradiction filter | Numeric heuristic: chunks with high word overlap (`> 0.3`) but strongly differing numeric values (`max/min > 2.0`, both `≥ 100`) are flagged; the lower-RRF chunk is dropped. The `min_value=100` floor prevents day-of-month vs. year false positives. |
| **4** | Entity-overlap pruning | Drop chunks whose entity set is a strict subset of a higher-ranked chunk's entity set (original contribution). |
| **5** | Entity-mention filter | Each surviving chunk must literally contain at least one query entity. Multi-word entities match on the full phrase OR on individual tokens of length ≥ 8 characters; single-token entities require length ≥ 5. **Safety fallback:** if the filter would empty the context, all chunks are returned unmodified. |
| **6** | Context shrinkage | Per-chunk truncation to `max_chars_per_doc` (default 800) with sentence-boundary awareness. |

The filter chain is pure Python and adds < 6 ms to retrieval latency in typical operation. Each filter logs its before/after chunk count for traceability.

---

### 4.3 Verifier Agent (S_V)

**File:** [src/logic_layer/verifier.py](src/logic_layer/verifier.py)

The Verifier consumes the filtered context window, generates a grounded answer, and optionally self-corrects.

#### 4.3.1 VerifierConfig

All values are loaded from `config/settings.yaml` via `_verifier_config_from_cfg()` in `agent_pipeline.py`. Class-level defaults are emergency fallbacks only.

```python
@dataclass
class VerifierConfig:
    model_name: str = "qwen2:1.5b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 200
    max_iterations: int = 1            # 1 = single generation, no self-correction loop
    max_context_chars: int = 3500
    max_docs: int = 5
    max_chars_per_doc: int = 800
    # Pre-validation
    enable_entity_path_validation: bool = True
    enable_credibility_scoring: bool = True
    entity_coverage_threshold: float = 0.34
    # Credibility weights (must sum to 1.0)
    credibility_weight_cross_ref: float = 0.4
    credibility_weight_entity_freq: float = 0.3
    credibility_weight_provenance: float = 0.3
```

#### 4.3.2 ValidationStatus

```python
class ValidationStatus(Enum):
    VALID = "valid"
    AMBIGUOUS = "ambiguous"
    CONFLICTED = "conflicted"
    INSUFFICIENT = "insufficient"
```

#### 4.3.3 Pre-Generation Validation

Three checks run before generation:

1. **Entity-Path Validation.** Verifies that the retrieved chunks cover the query entities. With a KuzuDB graph store available, `find_chunks_by_entity_multihop()` is used; otherwise the check falls back to substring matching. The required coverage fraction is `entity_coverage_threshold` (default 0.34) — for 3-entity bridge queries, 1/3 lets the gold chunk pass on the first hop while the bridge entity is still unknown.
2. **Contradiction Detection.** A numeric-divergence heuristic over consecutive chunk pairs (O(n)) is the offline default. The NLI-based check (Reimers & Gurevych 2019 cross-encoder) is disabled by default to honour the edge constraint (~270 MB model download).
3. **Source Credibility Scoring.** Weighted combination of `cross_ref_score` (0.4), `entity_freq_score` (0.3), and `provenance_score` (0.3). The provenance signal is currently always derived from a baseline because the Navigator does not forward retrieval-source metadata to S_V — a documented limitation.

#### 4.3.4 Question-Relevance Reordering

Before formatting the prompt, `Verifier._reorder_by_question_relevance(query, context)` stable-sorts context chunks by query content-word overlap (substring count of tokens of length ≥ 4 after stopword filtering). This counters the position bias of small LLMs (qwen2:1.5b strongly favours the first plausible entity it sees) and ensures the most lexically aligned chunk appears first. RRF scores are preserved; only the LLM display order is changed.

#### 4.3.5 Generation and Self-Correction Loop

```
pre_validate(context) → PreValidationResult
if status == INSUFFICIENT:
    return fallback("I cannot determine the answer from the provided context.")

context_str = build_context(context, max_chars=max_context_chars,
                            max_docs=max_docs, max_chars_per_doc=max_chars_per_doc)
answer = call_llm(GENERATION_PROMPT, query, context_str)

# With max_iterations=1 (default) the loop body never runs; raise to ≥ 2
# for the self-correction ablation comparison.
for iteration in range(1, max_iterations):
    is_valid, violations = verify_claims(query, answer, context)
    if is_valid:
        break
    answer = call_llm(CORRECTION_PROMPT, query, context_str, violations)

return VerificationResult(answer, confidence, iterations_used)
```

The correction prompt embeds concrete claim-level violations as feedback (Madaan et al. 2023, *Self-Refine*). On a 1.5 B-parameter SLM the second pass tends to inject hallucinations rather than correct them, so the default is a single pass; the loop is opt-in for ablation.

#### 4.3.6 VerificationResult

```python
@dataclass
class VerificationResult:
    answer: str
    confidence: ConfidenceLevel        # HIGH | MEDIUM | LOW
    iterations: int
    sources: List[str]
    self_corrections: int
```

`ConfidenceLevel` is derived from the verified-claim ratio (`≥ 0.8 → HIGH`, `≥ 0.5 → MEDIUM`, otherwise `LOW`).

---

### 4.4 Agentic Controller

**File:** [src/logic_layer/controller.py](src/logic_layer/controller.py)

`AgenticController` orchestrates the S_P → S_N → S_V chain as a state machine. It supports two execution modes:

- **LangGraph mode** — `StateGraph` workflow used when `langgraph` is importable.
- **Sequential fallback** — `_run_simple_pipeline`, always available; used for the thesis evaluation.

Both modes produce an identical `AgentState` result. LangGraph is therefore an optional dependency.

#### 4.4.1 AgentState

```python
class AgentState(TypedDict):
    # Input
    query: str
    # Planner output
    retrieval_plan: NotRequired[Optional[Dict[str, Any]]]
    sub_queries: List[str]
    entities: List[str]
    query_type: str
    # Navigator output
    raw_context: List[str]
    context: List[str]
    retrieval_scores: List[float]
    retrieval_metadata: Dict[str, Any]
    # Verifier output
    answer: str
    iterations: int
    verified_claims: List[str]
    violated_claims: List[str]
    all_verified: bool
    pre_validation: NotRequired[Optional[Dict[str, Any]]]
    # Metadata
    total_time_ms: float
    errors: List[str]
    stage_timings: Dict[str, float]
```

#### 4.4.2 State Machine

```
START
  ▼
_planner_node()    → populates RetrievalPlan in state
  ▼
_navigator_node()  → inspects hop_sequence for bridge dependencies
  │                  ├─ has_bridge_deps=False → _simple_navigate()
  │                  └─ has_bridge_deps=True  → _iterative_navigator_node()
  ▼
_verifier_node()   → calls Verifier with accumulated context
  ▼
END
```

#### 4.4.3 Iterative Navigator Node

Activates when any `HopStep.depends_on` is non-empty:

1. Sort hops by `step_id`.
2. Execute hops in dependency order via `Navigator.navigate_step(step.sub_query, entity_hints=current_hints)`.
3. After each `is_bridge=True` step, call `_extract_bridge_entities(results, exclude=query_tokens)` and append the discovered entities to `current_hints` (capped at 3 per step).
4. After all hops, run the pre-generative filter chain on the accumulated context.

#### 4.4.4 Bridge Entity Extraction

`_extract_bridge_entities(chunks, exclude)`:
- **Pass 1 (surname-anchor):** for each known entity (e.g., `"Kasper Schmeichel"`), extract surname tokens of length ≥ 6 (`"Schmeichel"`) and run a Unicode-aware regex `[FirstName] [OptionalMiddle]? [Surname]` over all chunks — recovers names containing diacritics that would otherwise break ASCII-only patterns. Candidates whose first token contains `:` are rejected (filters out `[About: …]` annotation artifacts).
- **Pass 2 (general):** capitalised multi-word phrases (`\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b`), excluding tokens already in the original query.
- Returns up to 3 candidates ranked by frequency in the retrieved chunks.
- **Fallback:** if extraction yields nothing, the original `entity_hints` are retained unchanged.

#### 4.4.5 Safety Constraints

| Constraint | Value | Reason |
|---|---|---|
| Maximum iterative hops | 3 | Bounds graph fan-out and total latency |
| Bridge entities added per step | 3 | Prevents combinatorial explosion in graph search |
| Empty-bridge fallback | retain prior hints | Graceful degradation to single-hop behaviour |

---

## 5. Pipeline Layer

The pipeline layer provides two orchestrators with compatible `process(query)` interfaces: `AgentPipeline` (a thin sequential chain used by the benchmark and ingestion CLIs) and `AgenticController` (the LangGraph state machine, §4.4). Either is acceptable as the production entry point; `AgentPipeline` is the simpler choice when iterative multi-hop is not required.

### 5.1 Agent Pipeline

**File:** [src/pipeline/agent_pipeline.py](src/pipeline/agent_pipeline.py)

`AgentPipeline` chains S_P → S_N → S_V and exposes a single `process()` method.

#### 5.1.1 AgentPipelineConfig

```python
@dataclass
class AgentPipelineConfig:
    enable_planner: bool = True            # ablation toggle
    enable_verifier: bool = True           # ablation toggle
    enable_caching: bool = True
    cache_max_size: int = 1000             # FIFO eviction

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "AgentPipelineConfig": ...
```

#### 5.1.2 PipelineResult

```python
@dataclass
class PipelineResult:
    answer: str
    confidence: str                        # "high" | "medium" | "low"
    query: str
    planner_result: Dict[str, Any]
    navigator_result: Dict[str, Any]       # contains "filtered_context"
    verifier_result: Dict[str, Any]
    planner_time_ms: float
    navigator_time_ms: float
    verifier_time_ms: float
    total_time_ms: float
    cached_result: bool = False
    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self, indent: int = 2) -> str: ...
```

#### 5.1.3 AgentPipeline

```python
class AgentPipeline:
    def __init__(
        self,
        planner: Optional[Planner] = None,
        navigator: Optional[Navigator] = None,
        verifier: Optional[Verifier] = None,
        hybrid_retriever: Optional[HybridStore] = None,
        graph_store: Optional[KuzuGraphStore] = None,
        enable_caching: bool = True,
        cache_max_size: int = 1000,
        config: Optional[Dict] = None,
    ): ...
    def process(self, query: str) -> PipelineResult: ...
    def get_stats(self) -> Dict[str, Any]: ...
    def clear_cache(self) -> None: ...
```

| Optimisation | Mechanism | Effect |
|---|---|---|
| Result cache | SHA-256-keyed FIFO cache (size 1000) on the normalised query string | Zero-cost repeated queries during evaluation |
| Lazy initialisation | Agents are constructed on first `process()` call | Reduced startup time |

`AgentPipeline.process()` calls `verifier.generate_and_verify()` **exactly once**; there is no outer pipeline-level retry loop. The self-correction mechanism is the inner loop inside the Verifier (§4.3.5), controlled by `agent.max_verification_iterations` in `settings.yaml`.

#### 5.1.4 Factory

```python
def create_full_pipeline(
    hybrid_retriever: HybridStore,
    graph_store: KuzuGraphStore,
    config: Dict,
) -> AgentPipeline: ...
```

This is the primary entry point used by `benchmark_datasets.py`.

---

### 5.2 Ingestion Pipeline

**File:** [src/pipeline/ingestion_pipeline.py](src/pipeline/ingestion_pipeline.py)

End-to-end workflow from raw documents to a fully indexed `HybridStore`.

```
Documents (paths)
     │
     ▼ DocumentLoader
     │   - Multi-format: TXT, JSON, JSONL, MD, PDF
     │   - Streaming iterator (memory-efficient)
     ▼
Documents (loaded)
     │
     ▼ Chunking (SpacySentenceChunker primary; fixed-size fallback)
     ▼
Chunks
     │
     ▼ EntityExtractionPipeline (optional)
     │   - GLiNER NER → REBEL RE (conditional on ≥ 2 entities)
     ▼
Chunks + Entities
     │
     ▼ EmbeddingPipeline (BatchedOllamaEmbeddings, batch_size = 32–64)
     ▼
Chunks + Embeddings
     │
     ▼ HybridStorage
         - VectorStoreAdapter.add_documents()
         - KuzuGraphStore.add_chunk_node × N
         - KuzuGraphStore.add_entity_node × M
         - KuzuGraphStore.add_relation × R
```

```python
class IngestionPipeline:
    def __init__(self, config: IngestionConfig): ...
    def ingest(self, source_path: str) -> IngestionMetrics: ...
    def ingest_texts(self, texts: List[str], metadatas: List[Dict]) -> IngestionMetrics: ...

def create_ingestion_pipeline(config: Dict[str, Any], use_mocks: bool = False) -> IngestionPipeline: ...
```

`IngestionMetrics` carries `chunks_added`, `entities_added`, `relations_added`, and `total_time_ms`.

---

## 6. Configuration System

**File:** [config/settings.yaml](config/settings.yaml)

All configurable parameters live in a single YAML file, the *single source of truth*. Every Python module reads it via `src/logic_layer/_settings.py` or directly via `yaml.safe_load()`; no values are hard-coded in production code paths.

```yaml
embeddings:
  model_name: "nomic-embed-text"
  base_url: "http://localhost:11434"
  embedding_dim: 768
  cache_path: "./cache/embeddings.db"

vector_store:
  provider: "lancedb"
  db_path: "./data/vector"
  distance_metric: "cosine"          # MUST be "cosine" for text embeddings
  normalize_embeddings: true
  top_k_vectors: 20
  similarity_threshold: 0.3
  overfetch_factor: 3
  graph_text_max_chars: 500

graph:
  enabled: true
  backend: "kuzu"                    # only "kuzu" supported
  graph_path: "./data/graph"
  max_hops: 2
  top_k_entities: 10
  expand_context: true
  entity_extraction_method: "keyword"  # keyword | spacy | gliner (query-time)
  relation_types: ["from_source", "next_chunk", "mentions", "related_to"]

rag:
  retrieval_mode: "hybrid"           # vector | graph | hybrid
  rrf_k: 60
  cross_source_boost: 1.2
  enable_bm25: true
  bm25_top_k: 20

llm:
  model_name: "qwen2:1.5b"
  base_url: "http://localhost:11434"
  temperature: 0.0                   # fully deterministic for reproducible thesis runs
  max_tokens: 200
  timeout: 60
  max_context_chars: 3500
  max_docs: 5
  max_chars_per_doc: 800

navigator:
  relevance_threshold_factor: 0.6
  redundancy_threshold: 0.8
  max_context_chunks: 8
  rrf_k: 60
  top_k_per_subquery: 20
  corroboration_source_weight: 0.1
  corroboration_query_weight: 0.05
  enable_reranker: true
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  reranker_top_k: 10
  contradiction_overlap_threshold: 0.3
  contradiction_ratio_threshold: 2.0
  contradiction_min_value: 100.0

planner:
  min_entity_confidence: 0.7
  max_entities: 10
  enable_bridge_detection: true
  enable_temporal_parsing: true
  classifier_spacy_weight: 1.5
  classifier_entity_boost: 0.5
  classifier_confidence_base: 0.6
  classifier_confidence_scale: 0.15
  classifier_confidence_cap: 0.95
  classifier_fallback_confidence: 0.8
  entity_density_threshold: 2
  noun_density_threshold: 4
  regex_entity_confidence: 0.75

agent:
  max_verification_iterations: 1     # 1 = no self-correction (default)
  enable_planner: true
  enable_verifier: true
  enable_caching: true
  cache_max_size: 1000

verifier:
  enable_entity_path_validation: true
  enable_credibility_scoring: true
  entity_coverage_threshold: 0.34
  min_credibility_score: 0.5
  credibility_weight_cross_ref: 0.4
  credibility_weight_entity_freq: 0.3
  credibility_weight_provenance: 0.3
  credibility_cross_ref_max: 3.0
  credibility_provenance_baseline: 0.5
  spacy_max_chars: 500
  confidence_high_threshold: 0.8
  confidence_medium_threshold: 0.5
  min_claim_chars: 15
  max_entities_to_verify: 5
  max_key_phrases: 10

entity_extraction:
  gliner:
    model_name: "urchade/gliner_small-v2.1"
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
    confidence_threshold: 0.15       # recall-optimised for HotpotQA
    batch_size: 16
  rebel:
    model_name: "Babelscape/rebel-large"
    confidence_threshold: 0.5        # uniform sentinel — REBEL emits no per-triplet score
    min_entities_for_re: 2
    max_input_length: 256
    max_output_length: 256
    num_beams: 5
  caching:
    enabled: true
    cache_path: "./data/entity_cache.db"
    lru_cache_size: 10000

ingestion:
  chunking_strategy: "sentence_spacy"
  sentences_per_chunk: 3
  sentence_overlap: 1
  spacy_model: "en_core_web_sm"
  extract_entities: true
  add_source_metadata: true

available_models:
  - { name: "qwen2:1.5b",  params_b: 1.5, expected_latency_s: 10, safe_60s: true }
  - { name: "gemma2:2b",   params_b: 2.0, expected_latency_s: 25, safe_60s: true }
  - { name: "llama3.2:3b", params_b: 3.0, expected_latency_s: 45, safe_60s: true }
  - { name: "phi3",        params_b: 3.8, expected_latency_s: 55, safe_60s: false }
  - { name: "qwen3:4b",    params_b: 4.0, expected_latency_s: 35, safe_60s: true }

paths:    { root: "./", data: "./data", documents: "./data/documents",
            vector_db: "./data/vector", graph_db: "./data/graph",
            logs: "./logs", cache: "./cache" }

logging:  { level: "INFO", format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file: "./logs/edge_rag.log" }

performance:
  batch_size: 64
  num_workers: 2
  device: "cpu"
  cache_embeddings: true
  max_cache_size_mb: 512
```

Configuration loading is handled by `_load_settings()` (which logs a `WARNING` on missing keys but returns a usable dict for graceful degradation) and by the `from_yaml()` classmethods on `ControllerConfig`, `VerifierConfig`, and `AgentPipelineConfig`. All three contain dataclass-level defaults that match the values above; production code constructs them via `from_yaml()` so that `settings.yaml` is the unique source of truth.

---

## 7. Evaluation Framework — Artifact C

**Package:** [src/evaluations/](src/evaluations/) and [benchmark_datasets.py](benchmark_datasets.py)

The evaluation layer provides the shared metric functions, dataset loaders, and experiment runners for the thesis evaluation. Datasets are isolated in separate vector stores and graph databases (per-dataset directory under `./data/`) to prevent cross-dataset leakage.

### 7.1 Supported Datasets

| Dataset | Task | Split | Source |
|---|---|---|---|
| **HotpotQA** | Multi-hop QA (2 documents) | validation (distractor) | `hotpot_qa` (HuggingFace) |
| **2WikiMultiHopQA** | Multi-hop QA (2 Wikipedia articles) | validation | `framolfese/2WikiMultihopQA` |
| **StrategyQA** | Boolean implicit-reasoning QA | train | `ChilleD/StrategyQA` |

### 7.2 Metric Functions

**File:** [src/evaluations/metrics.py](src/evaluations/metrics.py) — canonical implementations consumed by every evaluator.

- **`normalize_answer(text)`** — lowercase, strip articles (`a`, `an`, `the`), remove punctuation, collapse whitespace. Matches the official HotpotQA normaliser.
- **`compute_exact_match(pred, gold)`** — normalised string equality OR word-boundary substring fallback (handles cases where the gold answer is a strict subset of the predicted span).
- **`compute_f1(pred, gold)`** — token-level F1 over multiset (occurrence-counted, not set) intersection of tokens; precision × recall harmonic mean.

```
EM = 1 if normalize(pred) == normalize(gold)
        OR normalize(gold) ∈ normalize(pred)
     0 otherwise

common  = multiset_intersection(pred_tokens, gold_tokens)
prec    = |common| / |pred_tokens|
rec     = |common| / |gold_tokens|
F1      = 2 * prec * rec / (prec + rec)
```

### 7.3 HotpotQA Evaluator

**File:** [src/evaluations/evaluate_hotpotqa.py](src/evaluations/evaluate_hotpotqa.py)

```python
class HotpotQAEvaluator:
    def evaluate(self, n_samples: int) -> EvalSummary: ...
    def run_comparison(self, mode_list: List[str]) -> Dict[str, EvalSummary]: ...
```

`EvalResult` carries `question_id`, `question`, `gold_answer`, `predicted_answer`, `exact_match`, `f1_score`, `total_time_ms`. `EvalSummary` aggregates `exact_match_rate`, `avg_f1`, `avg_time_ms`, plus per-type and per-level breakdowns.

### 7.4 Ablation Study

**File:** [src/evaluations/ablation_study.py](src/evaluations/ablation_study.py)

```python
class AblationStudy:
    def run(self, datasets: List[str], samples: int) -> Dict[str, Dict[str, float]]: ...
```

For each configuration the runner instantiates a fresh pipeline via the supplied `pipeline_factory`, evaluates `samples_per_dataset` questions, aggregates per-question EM/F1, and emits `config.json`, `raw_results.json`, `summary.csv`, `per_question.csv`, `report.md`, and `latex_tables.tex`. Random seed `42` is set at module import (`random.seed`, `numpy.random.seed`) for reproducible question sampling.

### 7.5 CLI Interface

The top-level driver is `benchmark_datasets.py`:

```bash
# Ingest one dataset (Phase 1: chunking only)
python benchmark_datasets.py ingest --dataset hotpotqa --samples 500 \
       --chunk-sentences 3 --chunk-overlap 1

# Ingest chunks only (skip embedding; faster on re-ingest)
python benchmark_datasets.py ingest --dataset hotpotqa --samples 500 --chunks-only

# Single-configuration evaluation
python benchmark_datasets.py evaluate --dataset hotpotqa --samples 100 \
       --model qwen2:1.5b --mode hybrid

# Component ablation (toggle individual stages)
python benchmark_datasets.py evaluate --dataset hotpotqa --samples 100 \
       --no-planner --no-verifier --iterations 1

# Full ablation matrix (all retrieval modes)
python benchmark_datasets.py ablation --dataset hotpotqa --samples 100

# Component ablation (Planner / Verifier / iteration counts)
python benchmark_datasets.py ablation --dataset hotpotqa --samples 100 --component-ablation

# Status / self-test
python benchmark_datasets.py status
python benchmark_datasets.py test
```

Results are persisted to `evaluation_results/ablation_<timestamp>.json`.

### 7.6 Diagnostic Tooling

| Tool | Purpose |
|---|---|
| [diagnose.py](diagnose.py) | Layer-by-layer pipeline diagnostic for a single question; supports `--layer`, `--skip-llm`, `--graph-quality`, `--multi`. |
| [diagnose_verbose.py](diagnose_verbose.py) | Trace-mode diagnostic with monkey-patched filter hooks; tracks per-stage gold-answer survival. |
| [diagnose_ingestion.py](diagnose_ingestion.py) | Source-to-rank trace per question across `chunks_export.json` → LanceDB → KuzuDB → retrieval rank. |
| [src/evaluations/ollama_performance_diagnostic.py](src/evaluations/ollama_performance_diagnostic.py) | Embedding-dimension and Ollama endpoint health checks. |
| [src/evaluations/test_rag_quality.py](src/evaluations/test_rag_quality.py) | Standalone retrieval-quality probe (config verification, threshold sweep). |
| [test_system/graph_inspect.py](test_system/graph_inspect.py) | KuzuDB schema and statistics inspector. |
| [test_system/graph_3d.py](test_system/graph_3d.py) | matplotlib + pyvis knowledge-graph visualisation. |

All diagnostic CLIs require `python -X utf8` on Windows to render Unicode correctly.

---

## 8. Technology Stack

### 8.1 Core Dependencies

| Package | Constraint | Role |
|---|---|---|
| `lancedb` | ≥ 0.6 | Embedded vector store (Apache Arrow, IVF-Flat ANN) |
| `kuzu` | ≥ 0.3 | Embedded property graph (native Cypher) |
| `spacy` | ≥ 3.5 | Tokenisation, POS tagging, NER, dependency parsing |
| `gliner` | ≥ 0.2 | Zero-shot NER |
| `transformers` | ≥ 4.30 | REBEL relation extraction |
| `rank_bm25` | ≥ 0.2.2 | BM25 sparse retrieval (third RRF lane) |
| `sentence-transformers` | ≥ 2.2 | Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scikit-learn` | ≥ 1.3 | TF-IDF, quality metrics |
| `pydantic` | ≥ 2.5 | Data validation |
| `pyyaml` | ≥ 6.0 | Configuration parsing |
| `langchain` | 0.3.x | `Document` schema and `Embeddings` interface |
| `langgraph` | ≥ 1.0 (optional) | State-machine orchestration in `AgenticController` |
| `datasets` | ≥ 2.14 | HuggingFace dataset loading |
| `requests` | ≥ 2.31 | Ollama HTTP client |
| `pyarrow` | ≥ 12.0 | LanceDB columnar layer |
| `sqlite3` | stdlib | Embedding cache persistence |

`requirements_frozen.txt` pins every transitive dependency for thesis reproducibility.

### 8.2 External Services (Local)

| Service | Model | Purpose |
|---|---|---|
| Ollama | `nomic-embed-text` | 768-dim dense text embeddings |
| Ollama | `qwen2:1.5b` | Answer generation (1.5 B params, 4-bit GGUF) |
| Ollama (alternates) | `gemma2:2b`, `llama3.2:3b`, `phi3`, `qwen3:4b` | Ablation comparison models |
| SpaCy | `en_core_web_sm` | Query parsing in S_P; sentence segmentation in chunking |

All models run on CPU. `qwen2:1.5b` fits within ~2 GB memory; the full system stays under the 16 GB RAM constraint.

### 8.3 Database Selection Rationale

**LanceDB** was selected over FAISS, ChromaDB, and Qdrant for: embedded architecture (no server, no Docker), native Apache Arrow columnar format (zero-copy reads), IVF-Flat ANN with configurable recall/speed trade-off, and built-in metadata filtering in a single SQL-like query.

**KuzuDB** was selected over NetworkX for: native Cypher (expressive multi-hop path queries), persistent on-disk storage, and 10–100× faster traversal than NetworkX for graphs with > 100 nodes (verified during migration benchmarks). It is embedded — no server process required.

---

## 9. Data Flows

### 9.1 Ingestion Flow

```
 Raw articles (HotpotQA context documents)
        │
        ▼  HotpotQALoader.load(n_samples=500)
  ┌─────────────────────────────────────────┐
  │  DATASET LOADING                        │
  │  - HuggingFace `datasets` library       │
  │  - Distractor split (validation)        │
  │  - Deduplicate articles by title        │
  │  → ~2,000–4,000 unique articles         │
  └─────────────────────────────────────────┘
        │
        ▼  create_langchain_documents()
  ┌─────────────────────────────────────────┐
  │  CHUNKING (SpacySentenceChunker)        │
  │  - sentences_per_chunk = 3              │
  │  - sentence_overlap   = 1               │
  │  - min_chunk_chars    = 50              │
  │  → ~5,000–15,000 Document chunks        │
  │  Metadata per chunk: chunk_id,          │
  │  source_file, article_title, dataset,   │
  │  sentence_count, position               │
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  EMBEDDING                              │
  │  BatchedOllamaEmbeddings:               │
  │  - Single batched cache lookup          │
  │  - API call for misses (32–64/batch)    │
  │  - Write-back to cache                  │
  │  - nomic-embed-text → 768-dim vectors   │
  └─────────────────────────────────────────┘
        │
        ▼  HybridStore.add_documents()
  ┌─────────────────────────────────────────┐
  │  DUAL-INDEX STORAGE                     │
  │  ① LanceDB (cosine, normalised, IVF)    │
  │  ② KuzuDB (DocumentChunk, Entity,       │
  │     SourceDocument; FROM_SOURCE,        │
  │     NEXT_CHUNK, MENTIONS, RELATED_TO)   │
  └─────────────────────────────────────────┘
        │
        ▼
  Persisted to ./data/<dataset>/{vector,graph}/
```

### 9.2 Query Processing Flow

```
 User query: "Where was the director of [Film X] born?"
        │
        ▼  AgentPipeline.process(query)
  ┌─────────────────────────────────────────────────────────┐
  │  S_P: PLANNER                             ~5–10 ms      │
  │  1. SpaCy parse(query)                                  │
  │     → entities: ["Film X"], type: MULTI_HOP, bridge ✓   │
  │  2. Decompose into hop_sequence                         │
  │     hop[0]: "Who is the director of Film X?"            │
  │     hop[1]: "Where was [director] born?"                │
  │  3. strategy: HYBRID                                    │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  S_N: NAVIGATOR                          ~120–280 ms    │
  │  Per hop:                                               │
  │   - VECTOR  : embed → LanceDB ANN, top-20  ~30–60 ms    │
  │   - BM25    : term-frequency, top-20       ~5–20 ms     │
  │   - GRAPH   : KuzuDB BFS up to 2 hops      ~1–30 ms     │
  │   - RRF FUSION (3 lanes + cross-source)    ~1 ms        │
  │   - CROSS-ENCODER RERANK (optional)        ~80–150 ms   │
  │   - 6-stage filter chain                   ~2–6 ms      │
  │  Output: NavigatorResult.filtered_context (5–8 chunks)  │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  S_V: VERIFIER                            ~200 ms–60 s  │
  │  1. pre_validate(filtered_context)                      │
  │  2. _reorder_by_question_relevance(query, context)      │
  │  3. build_context (max_context_chars=3500, max_docs=5)  │
  │  4. POST /api/generate → qwen2:1.5b (Ollama)            │
  │     temperature=0.0, max_tokens=200                     │
  │  5. claim verification + (optional) self-correction     │
  │  6. confidence scoring → HIGH | MEDIUM | LOW            │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  PipelineResult { answer, confidence, per-stage timings,
                    cached_result }
```

---

## 10. Performance Characteristics

### 10.1 Ingestion

| Stage | Latency (per chunk) | Notes |
|---|---|---|
| SpaCy chunking | ~0.5 ms | Module-level model cache |
| Embedding (cache miss) | ~30–80 ms | Ollama API, batched |
| Embedding (cache hit) | ~0.1 ms | SQLite primary-key lookup |
| LanceDB insert | ~2 ms | Arrow batch append |
| KuzuDB insert | ~1 ms | Cypher `CREATE` |
| **Total cold** | **~35–85 ms** | First run |
| **Total warm** | **~3–5 ms** | Re-ingestion with cache |

For a HotpotQA subset of 500 questions (~10,000 chunks): cold ~10–15 min (embedding-bound); warm re-ingest ~1–2 min (cache hit rate ~100 %).

### 10.2 Retrieval

| Component | Latency | Notes |
|---|---|---|
| Query embedding | ~30–60 ms | Single vector, often cached |
| LanceDB ANN search | ~8–15 ms | IVF-Flat, top-20 |
| KuzuDB graph traversal | ~1–30 ms | 2 hops; varies by graph density |
| BM25 retrieval | ~5–20 ms | `rank_bm25` over cached corpus DataFrame |
| RRF fusion | ~0.5 ms | Pure Python, O(N log N), 3 lanes |
| Cross-encoder reranker | ~80–150 ms | Optional; top-K × ~30 ms/pair on CPU |
| Filter chain | ~2–6 ms | Six sequential stages |
| **Retrieval subtotal** | **~120–280 ms** | Without LLM generation |
| LLM generation (qwen2:1.5b) | ~200 ms – 60 s | CPU 4-bit; KV-cache and context dominate |
| **End-to-end** | **~250 ms – 65 s** | LLM is the bottleneck on CPU |

> **LLM timeout note.** Ollama allocates the full KV-cache regardless of prompt length. On CPU `qwen2:1.5b` runs at ~8–15 tokens/s; a 3,500-character context (~875 tokens) routinely produces 30–55 s responses. The system enforces a 60 s `timeout` on Ollama HTTP calls; exceeding it returns the partial output.

### 10.3 Cache Efficiency

For repeated evaluations on the same dataset:
- Embedding-cache hit rate exceeds 95 % after the first ingestion run.
- Speed-up factor: ~300–500× for repeated query embeddings.
- Storage overhead: ~1.5 KB per cached embedding (768 floats × 4 bytes + metadata).
- Pipeline result-cache (FIFO, 1000 entries) eliminates duplicate end-to-end work for repeated queries during the ablation runs.

---

## 11. Non-Functional Requirements

### 11.1 Reproducibility

- **Configuration as data.** All tunable parameters live in [config/settings.yaml](config/settings.yaml); no hard-coded values in production code paths.
- **Deterministic generation.** `llm.temperature = 0.0` is the default; reruns of the evaluation produce identical answers given identical retrieval results.
- **Pinned dependencies.** `requirements_frozen.txt` records the exact transitive dependency set used for thesis evaluation; `requirements.txt` carries range constraints for general use.
- **Seeded sampling.** `random.seed(42)` and `numpy.random.seed(42)` are set at `ablation_study` import time.
- **Idempotent ingestion.** Chunk IDs are deterministic SHA-256 hashes of `source_doc:position:text[:50]`; re-ingestion of the same source yields identical IDs.
- **Decoupled GPU phase.** Phase 2 entity extraction runs on Colab GPU and emits `extraction_results.json`; the local edge target re-imports this file in Phase 3 without re-running GPU-bound models.

### 11.2 Error Handling and Observability

- **Logger hierarchy.** Every module obtains its logger via `logging.getLogger(__name__)`; the format and level are set in `settings.yaml → logging`.
- **Narrow exception clauses.** Storage operations catch a defined `_STORAGE_ERRORS` tuple (`OSError`, `IOError`, `RuntimeError`, `ValueError`, `TypeError`, `KeyError`); programming errors (`AttributeError`, `NameError`, `IndexError`) are intentionally not silenced.
- **Visible fallbacks.** Every fallback path emits `logger.warning("FALLBACK ACTIVE: …")` (GLiNER → SpaCy/regex; LangGraph absent → sequential controller; LangChain absent → minimal `Document` shim; Ollama unreachable → wrapped `RuntimeError`). Silent degradation is treated as a bug.
- **Timeout discipline.** Ollama HTTP calls carry `timeout=60`; exceeding this returns partial output with a warning rather than blocking the evaluator.
- **Per-stage timing.** `PipelineResult` and `AgentState` carry per-stage latencies. `RetrievalMetrics` carries lane-level counts. These flow into the ablation report for performance regression detection.
- **Diagnostic tooling.** `diagnose.py`, `diagnose_verbose.py`, and `diagnose_ingestion.py` are part of the system surface: they monkey-patch Navigator filters to track gold-answer survival per stage. Any change that breaks them is treated as a behavioural change.

### 11.3 Scalability

The system is single-node by design (edge deployment). Within that envelope:

- **Horizontal data growth.** LanceDB and KuzuDB scale to millions of records on a single node; the embedding cache scales linearly in disk space (~1.5 KB per record).
- **Batch throughput.** `BatchedOllamaEmbeddings` scales request count as `ceil(N / batch_size)`; cache hits reduce HTTP traffic by orders of magnitude on re-evaluation.
- **Filter cost.** The Navigator filter chain is O(N) in the number of fused chunks; with `top_k_per_subquery = 20` and a typical 2-hop plan, N ≤ ~60.
- **Known bottleneck.** LLM generation on CPU is the dominant latency (~80 % of end-to-end); this is intrinsic to the edge constraint and is the subject of the thesis ablation (model-size vs. accuracy trade-off via `available_models`).
- **Production-scale gap.** BM25 (`rank_bm25`) is O(N) at query time. For true production scale (millions of chunks), a dedicated FTS index (LanceDB FTS or Tantivy) would be the next step.

### 11.4 Test Coverage

The CI suite is run via `pytest` with `pytest.ini` configuring markers `slow`, `nightly`, `llm`, and `integration` (deselected by default to keep CI under five minutes). Tests requiring a populated KuzuDB (`test_graph_inspect.py`) are skipped in CI.

| File | Tests | Scope |
|---|---|---|
| `test_system/test_chunking.py` | 30 | SpaCy chunking correctness |
| `test_system/test_embeddings.py` | 40 | Batch embeddings, caching, cross-process invariants |
| `test_system/test_data_layer.py` | ~86 | Storage, retrieval, RRF fusion |
| `test_system/test_logic_layer.py` | ~106 | Planner, Navigator, Verifier unit tests |
| `test_system/test_planner_semantic.py` | 27 | Query classification semantics |
| `test_system/test_navigator_semantic.py` | ~32 | Filter-chain semantics |
| `test_system/test_verifier_semantic.py` | 39 | Pre-validation, generation, confidence semantics |
| `test_system/test_pipeline.py` | 74 | End-to-end pipeline (FIFO cache, statelessness) |
| `test_system/test_thesis_matrix.py` | 9 | Thesis evaluation matrix |
| `test_system/test_thesis_matrix_ext.py` | 10 | Extended thesis matrix |
| `test_system/test_missing_coverage.py` | 7 | Coverage-gap probes |
| `test_system/test_config_robustness.py` | 12 | Config-loader edge cases |
| `test_system/test_gliner_boundary.py` | 16 | GLiNER span/boundary cases |
| `test_system/test_graph_inspect.py` | — | Skipped in CI (graph not populated) |
| **Total CI** | **~496** | All green at the documented configuration |

Five structural invariants are explicitly asserted across the suite: Verifier factual grounding (T-A), embeddings vector-space consistency (T-B), GLiNER compound-span extraction (T-C), `AgentPipeline` FIFO cache eviction (T-D), and ingestion metadata isolation (T-E).

### 11.5 Security and Data Handling

- **Local-only execution.** The system makes outbound HTTP calls only to `localhost:11434` (Ollama). No telemetry, no cloud API.
- **No credential storage.** No secrets, API keys, or authentication tokens are read or persisted.
- **Sandboxed inputs.** All ingested text is treated as untrusted: it never reaches an `eval()` or shell. Cypher statements use parameterised arguments (no string concatenation into queries).
- **Reproducible artefacts.** Generated databases (`./data/<dataset>/{vector,graph}`), caches (`./cache/`), and evaluation outputs (`./evaluation_results/`) are gitignored to prevent accidental check-in of derived data.
- **Offline operation.** `HF_HUB_OFFLINE=1` retry logic ensures GLiNER loads from the local Hugging Face cache when network calls fail — required for air-gapped edge deployment.

---

## 12. Design Decisions & Trade-offs

### 12.1 Embedded vs. Client-Server Databases

All persistence components (LanceDB, KuzuDB, SQLite) operate in-process. This is the primary edge-deployment constraint. The trade-off is reduced horizontal scalability — acceptable for the single-node thesis scenario.

### 12.2 Reciprocal Rank Fusion vs. Linear Interpolation

RRF was chosen over linear interpolation (`α · vec + (1−α) · graph`) because:

1. It is rank-based, not score-based — robust to the score-distribution differences between dense, sparse, and graph retrieval (notably the score compression of `nomic-embed-text`).
2. The single hyperparameter `k` is parameter-robust: Cormack et al. (2009) recommend `k = 60` as a near-universal default.
3. It naturally handles documents present in only one list without score normalisation.

Modality-disable ablation is performed via `rag.retrieval_mode ∈ {vector, graph, hybrid}` rather than scalar weights.

### 12.3 Entity Name Disambiguation

KuzuDB graph lookup uses exact string matching, augmented by the `_name_variants` heuristic (last-name fallback for short first names; individual tokens of length ≥ 4). This resolves common short/full-name mismatches but does not resolve aliases such as `"Ed Wood"` ↔ `"Edward Davis Wood Jr."`. True disambiguation requires an entity-linking system (BLINK, REL); this is documented as a known limitation. The hybrid approach mitigates the impact: queries that fail graph lookup may still succeed via dense or BM25 retrieval.

### 12.4 Chunking Strategy Selection

For HotpotQA's short, well-structured passages (5–15 sentences per article), the SpaCy 3-sentence sliding window was selected over semantic chunking because:

- Wikipedia-style sentence boundaries are more meaningful than TF-IDF importance scores at this granularity.
- A 3-sentence window approximates one coherent factual paragraph.
- SpaCy processing is ~10× faster than transformer-based semantic boundary detection.

The semantic chunker is retained for long-form technical documents where structure detection matters.

### 12.5 Graph Depth Constraint

`max_hops = 2` is fixed empirically:
- 95 % of HotpotQA supporting facts are reachable within 2 hops from any named entity in the question.
- 3-hop traversal increases latency by ~40 ms with marginal recall improvement.
- HotpotQA is explicitly designed as a 2-hop reasoning benchmark.

### 12.6 Pre-Generative Filtering Owned by the Logic Layer

Filtering before generation (rather than answer-level post-processing) was chosen because:

- It reduces the LLM token budget — a direct latency win on CPU.
- Redundant context causes small models to average over duplicate evidence rather than synthesise.
- Contradictory context reliably causes factual errors in models below 3 B parameters.

The filter chain lives in the **Navigator (S_N)**, not in the data layer. This places curation under the agent that owns retrieval policy. The Jaccard redundancy threshold of 0.8 is conservative: two chunks must share 80 % of their token vocabulary to be treated as duplicates.

### 12.7 Cross-Encoder Reranker as Stage 2.5

The cross-encoder (`ms-marco-MiniLM-L-6-v2`) is inserted between RRF fusion and the relevance filter rather than as a post-filter step:

- Reranking before relevance filtering means the relevance threshold is applied to the higher-quality post-rerank scores, not to raw fusion ranks.
- The 22 MB model and ~30 ms/pair latency are edge-feasible at the typical `reranker_top_k = 10`.
- Disabling the reranker reverts cleanly to pure RRF order — useful as an ablation comparison.

### 12.8 Single-Pass Generation by Default

`agent.max_verification_iterations = 1` means the Verifier generates once and skips the self-correction loop. The Self-Refine paradigm (Madaan et al. 2023) reproduces reliably on GPT-3.5+ but on a 1.5 B-parameter SLM the second pass typically injects hallucinations rather than correcting them. The loop is opt-in (`max_iterations ≥ 2`) for ablation comparison.

### 12.9 Question-Relevance Reordering

Small LLMs exhibit strong position bias — they preferentially extract the first plausible entity they encounter. The Verifier therefore stable-sorts context chunks by query content-word overlap before formatting the prompt. RRF scores are preserved for downstream processing; only the LLM display order is changed. Empirically, this prevents the LLM from "answering the bridge entity" instead of the requested fact in multi-hop bridge queries.

### 12.10 Iterative Multi-Hop with `entity_hints`

Bridge queries cannot be answered by dispatching all sub-queries simultaneously: the bridge entity is unknown until hop 0 completes. The system:

1. Detects bridge dependencies via `HopStep.depends_on`.
2. Executes hops in dependency order.
3. After each `is_bridge=True` step, extracts new entities from the retrieved chunks (surname-anchored regex with `[About: …]` artifact rejection).
4. Passes the extended hint set to subsequent hops via `HybridRetriever.retrieve(..., entity_hints=...)`, bypassing query-time GLiNER on short, low-context sub-query fragments.

Safety constraints (max 3 iterative hops, max 3 bridge entities per step, retain prior hints on empty extraction) bound the worst-case fan-out.

---

*End of Technical Architecture Documentation.*
