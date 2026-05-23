# Technical Architecture

**Project:** Enhancing Reasoning Fidelity in Quantized Small Language Models:
Agentic Verification for Hybrid Retrieval-Augmented Generation on
Resource-Constrained Devices
**Author:** Jan Nietzard
**Institution:** RWTH, Master of Science
**Document version:** 5.2 — paper-release (2026-05-15), refreshed 2026-05-23 to reflect the query-side NER normalization layer (§3.5), the entity-free definite-description and classifier-abstention decompositions (§4.1), the graded entity-mention filter with a survivor floor (§4.2), the IDF-specificity / structural-coverage / sentence-aware context budgeting in the Verifier (§4.3), the rank-aware single-pool bridge scorer (§4.5), and the random/seeded/range question-selection modes of the evaluator (§7.1).

> This document describes the system as it stands at the paper-release
> milestone. Earlier versions tracked an incremental change-log; that
> log has been removed because it does not describe the artifact being
> evaluated. Where a design choice deviates from a textbook approach, the
> deviation is justified in §11 (Design Decisions) with academic
> citations only — no internal version markers.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Layer — Artifact A](#3-data-layer--artifact-a)
   - 3.1 [Embeddings](#31-embeddings)
   - 3.2 [Chunking](#32-chunking)
   - 3.3 [Entity & Relation Extraction](#33-entity--relation-extraction)
   - 3.4 [Storage: Vectors + Graph](#34-storage-vectors--graph)
   - 3.5 [Hybrid Retriever](#35-hybrid-retriever)
   - 3.6 [Graph Quality](#36-graph-quality)
4. [Logic Layer — Artifact B](#4-logic-layer--artifact-b)
   - 4.1 [Planner (S_P)](#41-planner-s_p)
   - 4.2 [Navigator (S_N)](#42-navigator-s_n)
   - 4.3 [Verifier (S_V)](#43-verifier-s_v)
   - 4.4 [Pipeline Orchestrator (`AgentPipeline`)](#44-pipeline-orchestrator-agentpipeline)
   - 4.5 [Static Helper Namespace (`AgenticController`)](#45-static-helper-namespace-agenticcontroller)
5. [Ingestion: Decoupled Three-Phase Architecture](#5-ingestion-decoupled-three-phase-architecture)
6. [Configuration System](#6-configuration-system)
7. [Evaluation Framework — Artifact C](#7-evaluation-framework--artifact-c)
8. [Technology Stack](#8-technology-stack)
9. [Data Flows](#9-data-flows)
10. [Non-Functional Requirements](#10-non-functional-requirements)
11. [Design Decisions and Academic Grounding](#11-design-decisions-and-academic-grounding)
12. [Known Limitations and Future Work](#12-known-limitations-and-future-work)

---

## 1. System Overview

This system implements a hybrid Retrieval-Augmented Generation (RAG)
architecture optimised for deployment on edge devices with strict CPU and
memory budgets. The central research hypothesis is that combining three
retrieval modalities — dense vector search, sparse term-frequency
retrieval, and structured knowledge-graph traversal — mediated by a
three-agent reasoning pipeline (Planner → Navigator → Verifier) increases
answer fidelity over any single modality, particularly for multi-hop
reasoning tasks that require resolving an unnamed intermediate referent.

The architecture is organised into three independently testable artifact
layers.

| Artifact | Layer | Responsibility |
|---|---|---|
| **A** | Data | Dual-index storage (LanceDB vectors + KuzuDB property graph), batched embeddings with persistent cache, hybrid retrieval via Reciprocal Rank Fusion of vector / BM25 / graph paths, optional cross-encoder reranking. |
| **B** | Logic | Three-agent reasoning pipeline: query classification and decomposition (S_P), pre-generative retrieval orchestration and filtering (S_N), pre-validation followed by quantised-SLM answer generation with optional self-correction (S_V). |
| **C** | Evaluation | Multi-dataset benchmarking (HotpotQA, 2WikiMultiHopQA, StrategyQA), retrieval-only and end-to-end modes, per-pattern diagnostic JSONL, ablation suites for components and hyperparameters. |

**Edge-deployment constraints (binding throughout the design):**

The term *edge device* in this work refers to a single-host commodity
machine without server-grade GPU acceleration and without cloud
dependency at inference time. The concrete operating envelope used as
the design target throughout this thesis is:

| Resource | Budget | Rationale |
|---|---|---|
| Host RAM | ≤ 16 GB total; system targets ≤ 8 GB working set | Covers laptops, mini-PCs (Intel NUC / Mac mini class), and industrial single-board computers (Jetson Orin Nano 8 GB, Raspberry Pi 5 8 GB). |
| Compute | CPU-only at inference; x86-64 or ARM64; no CUDA / ROCm requirement | GPU is available only for the GPU-bound extraction phase (Phase 2, run offline in Colab); production inference must run without it. |
| Disk | ≤ 10 GB for code + databases + caches per dataset | Fits a typical embedded device storage profile. |
| Per-query latency (target) | < 60 s end-to-end on the reference hardware | Matches the budget profiled in `latency_memory_profile.py`. |
| External network | Not required at evaluation time | Ollama runs locally on `localhost:11434`; no cloud API calls. |
| Reference hardware (development & evaluation) | Intel-class CPU, 16 GB RAM, Windows 11 / Linux | Hardware on which the reported numbers were produced. |

These limits are not aspirational — they constrain every component
design choice in §3 (embedded stores; no GPU at runtime; quantised
generation), §4 (small generation models, single-pass orchestration,
bounded self-correction), and §10 (memory profiling).

- Every persistence backend is *embedded*: no separate database server.
  LanceDB stores vectors as Apache Arrow files; KuzuDB stores the graph
  as memory-mapped column store files; SQLite stores the embedding cache.
- All language models are served locally via Ollama; no cloud API
  dependency at evaluation time.
- Generation uses 4-bit GGUF quantisation (llama.cpp backend) via Ollama.

### 1.1 Position relative to prior work

Three recent lines of work are the closest neighbours; each is cited
throughout this document where its mechanism is reused.

**vs. IRCoT** (Trivedi et al. 2023, ACL; arXiv:2212.10509) —
*Interleaving Retrieval with Chain-of-Thought.* IRCoT interleaves
free-form CoT reasoning with retrieval steps on large LLMs (GPT-3,
Flan-T5 XXL). This work adopts the *iterative bridge-grounding* idea
(step-N retrieved entities feed step-N+1 query, see §4.4 and
`AgentPipeline._iterative_navigate`) but replaces free-form CoT with a
**structured Planner that emits a finite, parse-derived hop graph**
(Patterns E/G/H/K/L) executable by a 1.5B-parameter quantised SLM.
Free-form CoT is not robust at this model size; the dependency-parse
backbone supplies the reasoning skeleton externally.

**vs. HippoRAG** (Gutiérrez et al. 2024, NeurIPS; arXiv:2405.14831) —
*Neurobiologically inspired long-term memory.* HippoRAG runs personalised
PageRank over a passage graph at query time using a GPT-3.5 OpenIE
extractor. This work shares the typed-vs-untyped relation-weighting
intuition (cooccurrence edges down-weighted, semantic edges full
weight; §12.36) and the use of named-entity bridges as retrieval
anchors, but (i) extracts relations *offline* with GLiNER + REBEL
(open-source, edge-feasible), (ii) replaces personalised PageRank with
**bounded multi-hop Cypher traversal capped at hop-3** (§3.5), and
(iii) does not require a cloud LLM at extraction or query time.

**vs. Self-Refine** (Madaan et al. 2023, NeurIPS) — *Iterative
refinement with self-feedback.* Self-Refine has the same LLM critique
and revise its own output. This work uses the same *single feedback
loop* in the Verifier (§4.3) but replaces self-feedback with
**deterministic entity-presence verification** against retrieved
context (an external signal, not the model's own opinion). The
self-correction round count is bounded to ≤ 1 by default; the
ablation in §11.9 / `agentic_ablation.py` quantifies its marginal
contribution.

**The combined contribution** of this work is: a *parse-grounded,
edge-feasible* instantiation of the IRCoT / HippoRAG / Self-Refine
family in which (a) each agentic step is implementable without a
large LLM, (b) every retrieval modality and every reasoning hop fits
within the edge envelope defined above, and (c) the multi-modal
retrieval is weighted by an empirically-justified RRF schedule
(§3.5, §12.35 / I-1) rather than uniform fusion. The end-to-end
evaluation in Artifact C tests the hypothesis that this combination
exceeds vector-only and graph-only retrieval on bridge-heavy
multi-hop QA at the edge-device scale.

---

## 2. Repository Structure

```
Entwicklungfolder/
│
├── src/                              # Production source
│   ├── data_layer/                   # Artifact A
│   │   ├── __init__.py               # Public exports
│   │   ├── embeddings.py             # BatchedOllamaEmbeddings + SQLite cache
│   │   ├── chunking.py               # SpacySentenceChunker (primary), 4 alt
│   │   ├── coreference.py            # Optional Phase-1 pronoun resolver
│   │   ├── entity_extraction.py      # GLiNER NER + REBEL RE (local fallback)
│   │   ├── entity_types.py           # OntoNotes-5 / GLiNER label map
│   │   ├── svo_extraction.py         # SpaCy dep-parse Subject-Verb-Object triples
│   │   ├── storage.py                # HybridStore, VectorStoreAdapter, KuzuGraphStore
│   │   ├── hybrid_retriever.py       # HybridRetriever, RRFFusion, BM25
│   │   ├── graph_quality.py          # canonical_form, cleanup, cooccurrence,
│   │   │                             # entity-linking, baseline + invariants
│   │   ├── ingestion.py              # DocumentIngestionPipeline (data-layer)
│   │   └── conftest.py
│   │
│   ├── logic_layer/                  # Artifact B
│   │   ├── __init__.py               # Public exports
│   │   ├── _config.py                # ControllerConfig (Navigator config)
│   │   ├── _settings.py              # YAML loader + shared regex constants
│   │   ├── planner.py                # S_P: classifier + entity extractor + decomposer
│   │   ├── navigator.py              # S_N: hybrid retrieval orchestration + filters
│   │   ├── verifier.py               # S_V: pre-validation + generation + correction
│   │   ├── controller.py             # Static helpers for bridge entities + query rewriting
│   │   └── conftest.py
│   │
│   ├── pipeline/                     # Orchestration
│   │   ├── __init__.py
│   │   ├── agent_pipeline.py         # AgentPipeline: production S_P → S_N → S_V driver
│   │   ├── ingestion_pipeline.py     # End-to-end ingestion workflow
│   │   └── conftest.py
│   │
│   ├── thesis_evaluations/           # Artifact C
│   │   ├── __init__.py
│   │   ├── benchmark_datasets.py     # CLI: ingest / evaluate / ablation
│   │   ├── agentic_ablation.py       # Component-wise ablation (LLM-only → full)
│   │   ├── chunking_ablation.py      # Chunking-hyperparameter sweep
│   │   ├── quantization_sweep.py     # Cross-model evaluation
│   │   ├── latency_memory_profile.py # Per-stage timing + peak RSS
│   │   └── thesis_results_aggregator.py
│   │
│   └── utils.py
│
├── config/
│   └── settings.yaml                 # Single source of truth (~628 lines)
│
├── data/                             # Runtime data (gitignored)
│   └── hotpotqa/                     # Per-dataset: vector/, graph/,
│                                     # chunks_export.json, extraction_results.json,
│                                     # questions.json
│
├── cache/                            # SQLite embedding caches (gitignored)
├── evaluation_results/               # Per-question JSONL + summary tables
├── logs/                             # Structured log output
│
├── colab_extraction.py               # Phase 2 (GPU): GLiNER + REBEL on Drive-mounted chunks
├── local_importingestion.py          # Phase 3: import Phase-1+2 outputs into stores
├── diagnose.py                       # Layer-by-layer diagnostic tool
├── diagnose_verbose.py               # Full pipeline trace with gold-tracking hooks
├── diagnose_ingestion.py             # Ingestion-side consistency probe
├── diagnose_graph_baseline.py        # Read-only graph-quality reporter
│
├── test_system/                      # Test suite (~450 tests collected)
│   ├── conftest.py
│   ├── fixtures/                     # Gold NER, etc.
│   ├── test_*.py                     # Module-scoped tests
│   ├── diagnose_*.py                 # Component-level diagnostic helpers
│   ├── graph_inspect.py              # KuzuDB schema/statistics inspector
│   └── graph_3d.py
│
├── TECHNICAL_ARCHITECTURE.md         # This document
├── REPRODUCE.md                      # Reproduction protocol
├── pytest.ini                        # CI markers: slow / nightly / llm / integration
├── requirements.txt                  # Library constraints
└── requirements_frozen.txt           # Pinned reproducibility set
```

---

## 3. Data Layer — Artifact A

The data layer is stateless with respect to any individual query. All
persistent state lives on disk: LanceDB directories, KuzuDB column
stores, SQLite cache files. Modules in this layer are independently
testable and do not import the logic layer.

### 3.1 Embeddings

**File:** [`src/data_layer/embeddings.py`](src/data_layer/embeddings.py)

A LangChain-compatible client around Ollama's `/api/embed` endpoint,
augmented with two performance-critical primitives:
content-addressable persistent caching and request batching.

**Cache schema (SQLite, `cache/<dataset>_embeddings.db`):**

```
embeddings (
  text_hash    TEXT PRIMARY KEY,    -- SHA-256(text)
  text_content TEXT NOT NULL,
  embedding    BLOB NOT NULL,       -- JSON-serialised float list
  model_name   TEXT NOT NULL,
  access_count INTEGER DEFAULT 0,
  created_at   TIMESTAMP
)
INDEX idx_model_hash ON (model_name, text_hash)
```

The composite key on `(model_name, text_hash)` automatically invalidates
the cache when the embedding model changes; the SHA-256 key collapses
duplicate inputs to a single embedding.

**Public client:**

```python
class BatchedOllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url:   str = "http://localhost:11434",
        batch_size: int = 64,
        cache_path: Path = Path("./cache/embeddings.db"),
        device:     str = "cpu",
        timeout:    int = 60,
    ): ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...
    def get_metrics(self) -> Dict[str, Any]: ...
    def clear_cache(self) -> None: ...
```

**Batching pipeline** (single `embed_documents(texts)` call):

1. Compute SHA-256 hashes for all inputs; issue **one** SQL batch lookup
   for cache hits.
2. Identify cache misses (delta of input list vs. lookup hits).
3. Issue Ollama API calls in chunks of `batch_size`; on partial failure
   the batch is retried at `batch_size // 2` until single-text.
4. Write back fresh embeddings to the cache.
5. Assemble outputs in original input order.

The `EmbeddingMetrics` dataclass tracks `total_texts`, `cache_hits`,
`cache_misses`, `batch_count`, `total_time_ms`; the derived properties
`cache_hit_rate` (percent) and `avg_time_per_text_ms` are reported in
every evaluation summary (see §7).

The factory `create_embeddings(cfg)` constructs the client from a
`settings.yaml` dict so production and tests share one construction path.

### 3.2 Chunking

**File:** [`src/data_layer/chunking.py`](src/data_layer/chunking.py)

The chunking module exposes five chunker classes; the production
pipeline uses **`SpacySentenceChunker`** with a 3-sentence sliding
window and 1-sentence overlap. The alternative strategies
(`SemanticChunker`, `RecursiveChunker`, `FixedSizeChunker`,
`SentenceChunker`) are retained for the chunking ablation in §7.

```python
class SpacySentenceChunker:
    def __init__(
        self,
        sentences_per_chunk: int = 3,        # settings: ingestion.sentences_per_chunk
        sentence_overlap:    int = 1,        # settings: ingestion.sentence_overlap
        min_chunk_chars:     int = 50,
        max_chunk_chars:     int = 2000,
        spacy_model:         str = "en_core_web_sm",
        entity_aware:        bool = False,    # reserved; not yet implemented
    ): ...
    def chunk_text(self, text: str, source_doc: str = "") -> List[SentenceChunk]: ...
```

Each `SentenceChunk` carries `text`, `sentence_count`, `position`,
`start_char`, `end_char`, and `source_doc`. **Chunk IDs are
deterministic SHA-256 hashes of `source_doc:position:text[:50]`** so
re-ingesting the same source produces byte-identical IDs and is
therefore idempotent at the graph-node level.

A module-level `SpacyModelCache` singleton amortises the SpaCy load
cost across multiple chunker instances.

### 3.3 Entity & Relation Extraction

**File:** [`src/data_layer/entity_extraction.py`](src/data_layer/entity_extraction.py)
**GPU phase:** `colab_extraction.py` (production); local module is the
fallback path.

#### Named Entity Recognition

**Model:** GLiNER (`urchade/gliner_small-v2.1`) — zero-shot span-based
NER (Zaratiana et al. 2023).

The thesis evaluation uses a **fixed OntoNotes-5-aligned prompt set**:
9 GLiNER prompts collapsing to 8 canonical types via the
`GLINER_LABEL_MAP` in [`src/data_layer/entity_types.py`](src/data_layer/entity_types.py).

| Prompt | Canonical type |
|---|---|
| person | PERSON |
| organization | ORGANIZATION |
| location, city, country | LOCATION / GPE |
| date | DATE |
| event | EVENT |
| work of art | WORK_OF_ART |
| product | PRODUCT |

The multi-prompt expansion (city + country alongside location) gives
GLiNER higher recall than a single abstract label (Zaratiana et al.
2023 §4.2); all prompts collapse to OntoNotes-5 canonical types
(Weischedel et al. 2013, LDC2013T19) so the graph schema and downstream
filters operate on a single label set.

#### Relation Extraction

**Model:** REBEL (`Babelscape/rebel-large`) — seq2seq triplet generation
(Cabot & Navigli 2021, EMNLP 2021 Findings).

The Colab phase invokes REBEL with `num_beams=5` and
**log-probability-calibrated per-triplet confidence** (mean softmax of
the triplet's decoded tokens). The prior implementation emitted a
constant 0.5 sentinel; the calibrated variant is what is consumed by
the storage layer's `_triple_frequency_confidence` (see §3.4).

#### Subject-Verb-Object extraction

[`src/data_layer/svo_extraction.py`](src/data_layer/svo_extraction.py)
extracts narrative `(subject, verb, object)` triples from the SpaCy
dependency parse. SVO complements REBEL by recovering narrative
predicates ("X directed Y", "X founded Y") that REBEL's Wikipedia-
infobox training does not cover. Both endpoints must resolve to a known
entity via `canonical_form` (no free-text endpoints).

### 3.4 Storage: Vectors + Graph

**File:** [`src/data_layer/storage.py`](src/data_layer/storage.py)

The storage layer is a facade (`HybridStore`) over two embedded
databases plus an optional embedding cache.

#### LanceDB vector store (`VectorStoreAdapter`)

ANN search via HNSW (Malkov & Yashunin 2018) on dense embeddings.
Distances are converted to similarities in `[0, 1]` so downstream
consumers apply a uniform threshold:

- cosine: `sim = max(0, 1 - dist)`
- L2:     `sim = 1 / (1 + dist)`

`vector_search` over-fetches by `overfetch_factor` (default 3) before
threshold filtering, ensuring threshold-pruning does not silently drop
top-k.

#### KuzuDB property graph (`KuzuGraphStore`)

Native Cypher-based multi-hop traversal (Feng et al. 2023, CIDR 2023).
Columnar / vectorised execution and out-of-core processing via
memory-mapped files keep the graph operable on edge hardware.

**Schema:**

```
NODE TABLE DocumentChunk (chunk_id PK, text, page_number, chunk_index, source_file)
NODE TABLE SourceDocument (doc_id PK, filename, total_pages)
NODE TABLE Entity (entity_id PK, name, type, confidence)

REL TABLE FROM_SOURCE (DocumentChunk → SourceDocument)
REL TABLE NEXT_CHUNK   (DocumentChunk → DocumentChunk)
REL TABLE MENTIONS     (DocumentChunk → Entity)
REL TABLE RELATED_TO   (Entity → Entity)
                       attributes: relation_type, confidence, source_chunks
```

#### Multi-hop entity search

`find_chunks_by_entity_multihop(entity_name, max_results, enable_hop3, max_hops)`
returns chunks reachable from an entity through 0–3 hops of `RELATED_TO`
edges. The hop ladder is gated:

- **Hop 0** — direct `MENTIONS`. Three-stage cascade: exact name match →
  `CONTAINS $name` → `name IN $sub-phrases` (substring-aware alias).
- **Hop 2** — one `RELATED_TO` bridge. Capped fan-out from the matched
  entity (`_HUB_FANOUT_CAP = 5`) and hub-target exclusion via a cached
  set of high-degree entities (mention-degree > `HUB_MENTION_CAP`,
  default 280 ≈ 3 % of HotpotQA corpus).
- **Hop 3** — two `RELATED_TO` bridges. Disabled by default
  (`graph.enable_hop3: false`); when enabled, excludes `cooccurs` edges
  because two-hop noise compounds.

#### Triple-frequency confidence

Each retrieved bridge chunk carries a confidence
`(hop_distance, triple_confidence)` where `triple_confidence` is
**not** REBEL's per-triplet log-prob (which is per-edge, not per-bridge)
but a corpus-support score (DeepDive, Niu et al. 2012; Knowledge Vault,
Dong et al. 2014):

```
triple_confidence = relation_type_weight ×
                    min(1.0, log(1 + n_supporting_chunks) / log(10))
```

`relation_type_weight` distinguishes three relation origins:

| Origin | Detection | Weight |
|---|---|---|
| `cooccurs` (statistical co-mention) | explicit `relation_type == "cooccurs"` | **0.25** |
| SVO (dependency-parse verb lemma) | single-token verb lemma | **0.6** |
| REBEL (Wikidata-style predicate) | multi-token predicate or `_`/space in the string | **1.0** |

Origin classification is performed by `KuzuGraphStore._classified_weight()`.
The result is that any semantic REBEL bridge with even one supporting
chunk (conf ≥ 0.30) outranks any cooccurs bridge regardless of
support (max conf ≤ 0.25 with the 0.25 weight).

### 3.5 Hybrid Retriever

**File:** [`src/data_layer/hybrid_retriever.py`](src/data_layer/hybrid_retriever.py)

`HybridRetriever.retrieve(query)` orchestrates three retrieval paths and
fuses them via Reciprocal Rank Fusion (Cormack et al. 2009).

```
        ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
query → │  Dense vector  │    │   BM25 sparse  │    │ Graph multi-hop│
        │  (LanceDB)     │    │  (rank_bm25)   │    │  (KuzuDB)      │
        └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
                │ rank_v                │ rank_b              │ rank_g
                └────────────┬──────────┴──────────┬──────────┘
                             ▼                     ▼
                       ┌─────────────────────────────────┐
                       │  RRFFusion (Cormack 2009)       │
                       │  RRF(d) = Σ w_i / (k + rank_i)  │
                       └──────────────┬──────────────────┘
                                      ▼
                       Top-K fused results to Navigator (S_N)
```

**Per-path weights** (`vector_weight`, `bm25_weight`, `graph_weight`)
default to `1.0` (vanilla RRF). Unequal weights enable per-source
ablation without code changes.

The retriever also exposes:

- An entity-aware hop-2 reranker: per-chunk metadata records which
  sub-query the chunk ranked best for (`_best_sub_query`); the
  cross-encoder reranker (see §4.2) scores against that sub-query,
  not against the surface query, so a hop-2 chunk semantically distant
  from the surface form is not demoted.
- Optional cross-encoder reranker stage (`enable_reranker: true`) using
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (Reimers & Gurevych 2019).

#### Query-side entity extraction and normalization

Query-time entities (used both as graph-search anchors and as
entity-mention filter tokens in §4.2) are produced by
`ImprovedQueryEntityExtractor`, which runs the same GLiNER model as
ingestion (with SpaCy then regex fallbacks). Because the two consumers
trust these spans directly, a deterministic, query-side-only
normalization layer repairs span boundaries and gates non-discriminative
spans **before** they reach retrieval:

- **Span-boundary normalization.** (a) A leading auxiliary/copula verb
  absorbed into a span by the tagger is stripped (`_strip_leading_function_word`,
  closed class `is/are/was/were/do/does/did` only — wh-words and articles
  are deliberately *not* stripped, since legitimate titles begin with
  them, e.g. "Who Framed Roger Rabbit", and article handling is already
  type-aware in the shared normaliser). (b) Overlapping fragments of one
  span are merged to the maximal span using GLiNER character offsets
  (`_dedup_overlapping_spans`), so "Hook-Handed Man" survives intact
  rather than fragmenting. (c) A 4-digit year *interior* to a span
  (alphabetic tokens on both sides) is split off as a separate temporal
  constraint (`_strip_embedded_year`); leading/trailing years are kept,
  as they are often part of a title.
- **Discriminativeness gate** (`_is_junk_entity`). A span is rejected as
  an anchor/filter token if it is a stop-listed token, a generic question
  stem, or a *pure temporal/measure phrase* — all tokens are digits,
  measure nouns, or quantity adjectives with no capitalised proper-noun
  token (e.g. "7 consecutive seasons", "25 laps"). Such phrases match no
  graph node and, used as a filter token, retain noise.

Both passes are query-side only; the ingestion path is untouched, so
query/ingestion entity-ID consistency is preserved.

### 3.6 Graph Quality

**File:** [`src/data_layer/graph_quality.py`](src/data_layer/graph_quality.py)

Operations on the populated KuzuDB graph used during ingestion and on
diagnostic invocations:

| Function | Role |
|---|---|
| `canonical_form(name)` | NFKC normalisation + lowercase + suffix strip ("Jr.", "Sr."); merge key for duplicate detection. |
| `compute_graph_baseline(store)` | Reports `chunks`, `entities`, `mentions`, `relations`, density, isolated-entity rate, duplicate clusters, top hubs. |
| `assert_graph_invariants(metrics, strict)` | Checks `isolated_rate < 5 %`, `duplicate_rate < 2 %`, `relations_per_chunk ≥ 5`. |
| `build_cooccurrence_edges(store, results, name_to_id)` | One `RELATED_TO {relation_type='cooccurs'}` per entity pair sharing a chunk. |
| `cleanup_graph(...)` | Stop-list / orphan / hub / duplicate cleanup. |
| `link_entities_by_embedding(...)` | Embedding-based alias resolution within type buckets. Disabled by default — see §3.6.1. |
| `_redirect_entity_edges_bulk(store, redirect_map)` | Bulk re-pointing of all MENTIONS / RELATED_TO edges from a set of old entities to their merge targets in two batched CREATE queries (one per edge type). Replaces the per-edge MERGE used previously, which scaled with entity degree on KuzuDB. |
| `drop_subsumed_cooccurrence_edges(...)` | Removes cooccurs edges that already have a semantic relation. |
| `drop_isolated_entities(...)` | Drops entities with `MENTIONS` but zero `RELATED_TO` post-linking. |

#### 3.6.1 Embedding-based entity linking — empirical evaluation

`link_entities_by_embedding` clusters entities within a type bucket
(PERSON, ORG, LOCATION, GPE, …) by L2-normalised cosine similarity over
their nomic-embed-text vectors and merges clusters whose similarity
exceeds a threshold (`--linking-threshold`, default `0.92`). The linker
also supports per-bucket resume via `done_buckets` + `on_bucket_done`,
so a mid-phase crash does not lose finished buckets.

A threshold analysis over a 2000-entity sample per type bucket measured
the following merge rates with nomic-embed-text:

| Type | n | threshold 0.98 | threshold 0.99 | Largest cluster (0.98) |
|---|---|---|---|---|
| PERSON | 2 000 | 90.7 % | 90.0 % | 1 395 |
| LOCATION | 2 000 | 73.7 % | 72.5 % | 719 |
| GPE | 2 000 | 93.9 % | 93.3 % | 1 201 |

Sample cluster at threshold 0.98 (PERSON): `Uncle Albert` absorbs
`Julia Calvo`, `Mark Pickerel`, `David Ramsey`, `Vijay Yesudas`, and
1391 others — unrelated individuals. Threshold 0.99 yields nearly
identical merge rates and cluster sizes, indicating the embedding
model has effectively no discriminative range above 0.95 on entity
names. This score-compression behaviour is consistent with the
characterisation of nomic-embed-text in §10.

**Decision.** Embedding-based entity linking is disabled by default
(`--no-entity-linking`) for paper-release ingest. Alias resolution
reduces to exact-match deduplication via `canonical_form`. The
bulk-redirect + per-bucket-checkpoint infrastructure is retained as a
zero-cost engineering scaffold should a discriminative replacement
embedder be substituted in future work.

---

## 4. Logic Layer — Artifact B

The logic layer implements the three-agent reasoning architecture. Each
agent has a single public entry point and minimal cross-agent state.

### 4.1 Planner (S_P)

**File:** [`src/logic_layer/planner.py`](src/logic_layer/planner.py)

The Planner transforms a natural-language question into a structured
`RetrievalPlan` consisting of: a classified query type, an ordered hop
sequence of sub-queries, the named entities relevant to retrieval, and
any temporal or comparative constraints.

#### Three stages

1. **Query classification** — rule-based via SpaCy Matcher and
   compiled regex patterns over closed-class English function words
   (Honnibal & Montani 2017). Output labels: `SINGLE_HOP`, `MULTI_HOP`,
   `COMPARISON`, `TEMPORAL`, `AGGREGATE`, `INTERSECTION`.

   Two **deterministic pre-empts** run before the four-phase scoring
   classifier when closed-class English markers identify the
   construction unambiguously:
   - Distributive predication with floating "both"
     (`<aux> X and Y both <P>`) → `COMPARISON`. Quirk et al. 1985 §10.49.
   - Anaphoric introduction with "another" (`X and another <N> that …`)
     → `MULTI_HOP`. Karttunen 1976.

   Every plan records the pre-empt (if any) on the
   `RetrievalPlan.classifier_preempt` field so per-question
   diagnostics can audit false-positive rates.

2. **Entity extraction** — SpaCy NER restricted to OntoNotes-5 labels
   (Weischedel et al. 2013); per-label confidence estimate via
   `_LABEL_CONFIDENCE` mapping. Bridge-entity detection (for
   `MULTI_HOP` queries) marks entities that appear in
   bridge-connector context (`relcl` dep label or in-between two anchor
   entities). Generic NORP demonyms, DATE values, and other
   class-referring labels are excluded from bridge candidacy because
   they would steer retrieval toward high-degree hub nodes rather than
   specific bridging referents (West & Leskovec 2012).

3. **Plan generation** — Hop-sequence decomposition. Two generalisable
   mechanisms plus a baseline:

   **Mechanism A — Dependency-parse decomposers.** Four English
   constructions are recognised structurally via SpaCy dependency
   labels. Each recogniser is gated by a parse-confidence check
   requiring the relevant anchor to overlap a detected NER span.

   - **Pattern G** — Relative-clause bridge (Quirk et al. 1985 §17.7-15).
     "The [noun] in which [Entity] …" or "the [role] who [verb]
     [Entity]". Keys on the `relcl` dep label; two forms cover the
     relative-pronoun-subject case (form1) and the relative-pronoun-
     object case (form2).
   - **Pattern H** — Chained attribution (Levin 1993). Passive ROOT
     with `agent` by-phrase whose object is an indefinite pronoun,
     plus an `acl` clause on the subject anchored to a named entity.
     The attribution-clause head must come from a small closed class
     of derivation/depiction verbs (`_ATTRIBUTION_ACL_VERBS`).
   - **Pattern E** — Relational-noun + of-PP complement (Partee 1995;
     Barker 1995). A noun whose dependency structure contains a
     `prep("of")` child whose `pobj` is a named entity. **Generalises
     to any noun** for which the parser produces this structure; no
     role enumeration. Implemented by `_find_relational_noun_bridge`.
   - **Pattern F** — Passive-agent voice transformation
     (Bresnan 1982; Quirk et al. 1985 §3.65-71). A verb with
     `auxpass + nsubjpass + agent` children. Past-participle →
     infinitive transformation uses SpaCy's morphological lemmatiser,
     so the recogniser is **vocabulary-independent**. Implemented by
     `_find_passive_agent_bridge`. **Subject guards:** Pattern F is
     skipped when the extracted passive subject is itself an
     interrogative noun phrase (`_PASSIVE_F_INTERROGATIVE_SUBJ_RE`:
     leading what/which/whose) or a bare pronoun
     (`_PASSIVE_F_BARE_PRONOUN_SUBJ`: that/this/it/who/…). The template
     "Who {verb} {subject}?" with such a subject yields a
     self-referential sub-query ("Who hold what government position?")
     or a context-free one ("Who form that?"); both guards fall through
     to the connector-split baseline instead.

   **Mechanism B — Closed-class lexical pre-empts.** As above
   (Patterns I and J), routed before the scoring classifier.

   **Baseline — Connector-split decomposition.** The methodology-
   described baseline (Khattab et al. 2022, DSP). The query is split
   at bridge connectors ("that", "which", "who", "of the") and
   fragments are re-ordered so the bridge sub-query precedes the
   final sub-query. A 2-hop cap collapses spurious 3-part splits
   whose middle parts contain no named entity.

   **Mechanism C — Entity-free definite-description bridge.** When a
   `MULTI_HOP` query names no anchor entity (NER returns nothing
   seedable), the bridge referent is often given by *description*
   rather than by name ("the only player … to have a 0.300 batting
   average for 7 consecutive seasons"). `_find_entity_free_description`
   selects the longest object/complement noun phrase (pobj/dobj/attr
   head) that is entity-free and carries ≥ 2 description-modifier
   dependents (a *definite description*; Russell 1905; Strawson 1950)
   and emits its subtree text as the hop-0 retrieval query. This
   replaces what was previously a silent degrade-to-single-hop.

   **Classifier-abstention override (A4).** The classifier returns
   `SINGLE_HOP` with exactly `classifier_fallback_confidence` (0.5)
   *only* when no pattern scored (its documented no-signal sentinel;
   any real match yields ≥ 0.6). In that abstention case — and only
   then — `_attribute_over_entity_signal` consults the dependency parse:
   if the question asks for a wh-determined attribute of a thing related
   to a named entity (e.g. "what class of instrument does X play?"),
   the type is re-routed to `MULTI_HOP` so the entity-seeded 2-hop
   decomposition runs. The gate cannot override a classification that
   had positive evidence, so it cannot regress confident single-hop
   questions.

   **Never-collapse contract.** `_decompose_multi_hop` enforces a logged
   invariant: a `MULTI_HOP` classification must never silently emit a
   single sub-query; the only permitted single-sub-query output is the
   explicitly-marked degrade path below.

   **Failure modes**, explicit and surfaced in `matched_pattern`:
   - `structural_descriptive_2hop`: Mechanism C fired — entity-free
     definite-description bridge.
   - `fallback_generic_2hop`: classified `MULTI_HOP`, no mechanism
     applied, seed entity available. Emits "Who or what is X?" as
     hop-0 and the original query as hop-1.
   - `fallback_degraded_to_single_hop`: classified `MULTI_HOP`, no
     mechanism applied, no entity *and* no usable description. Logged at
     WARNING — the only sanctioned single-sub-query output.

#### Per-pattern diagnostics

Every `RetrievalPlan` records `matched_pattern` on the plan dataclass
and surfaces it through `to_dict()` into the per-question evaluation
JSONL. This enables hit-rate and SF-F1-conditional analysis per
pattern without parsing logs:

```bash
# Per-pattern hit count
jq -r '.matched_pattern' results.jsonl | sort | uniq -c | sort -rn

# Mean SF-F1 conditional on pattern
jq -r '"\(.matched_pattern)\t\(.sf_f1)"' results.jsonl | \
  awk -F'\t' '{s[$1]+=$2; n[$1]++} END {for (k in s) print k, s[k]/n[k], n[k]}'
```

#### Comparison decomposer

`_decompose_comparison` routes through (in order):

1. Pattern I — boolean conjunction `<aux> X and Y both P` → parallel
   yes/no.
2. Select-between-two — disjunction of two NER entities joined by
   "or"; the disjunction is detected via NER span positions, not
   surface regex.
3. `_ATTR_MAP` rewrites — closed-class English attribute nouns
   (nationality, birthplace, profession, genre, age, country,
   religion) → per-entity factual-lookup templates ("What is the
   nationality of X?").
4. Generic per-entity predicate template — used when no
   attribute-rewrite applies.

### 4.2 Navigator (S_N)

**File:** [`src/logic_layer/navigator.py`](src/logic_layer/navigator.py)

The Navigator executes the `RetrievalPlan` produced by S_P and delivers
a filtered, ranked context list to S_V.

```
                 ┌─────────────────────────────────────────────┐
RetrievalPlan ─→ │           NAVIGATOR (S_N)                    │
sub-queries ───→ │                                              │
                 │  1. Hybrid retrieval per sub-query           │
                 │  2. Reciprocal Rank Fusion (cross-sub-query) │
                 │     + cross-source corroboration boost       │
                 │  3. Pre-generative filter chain:             │
                 │     - Relevance filter                       │
                 │     - Redundancy (Jaccard) filter            │
                 │     - Contradiction filter (numeric heur.)   │
                 │     - Entity-overlap pruning                 │
                 │     - Entity-mention filter (top-K immune)   │
                 │     - Context shrinkage                      │
                 │  4. Cross-encoder reranking (Stage 2.5)      │
                 └──────────────┬───────────────────────────────┘
                                ▼
                  filtered_context, retrieval_methods → S_V
```

**Filter chain ordering rationale.** The six filters are ordered by
*increasing computational cost and decreasing reversibility*, so cheap
deterministic operations narrow the candidate set before expensive or
lossy ones run.

1. **Relevance filter** (lexical overlap) — cheapest, deterministic;
   removes manifest off-topic chunks first so subsequent filters
   operate on a relevance-aligned pool.
2. **Redundancy (Jaccard) filter** — pairwise n-gram comparison;
   removes near-duplicates *before* entity reasoning so duplicate
   chunks do not double-vote downstream.
3. **Contradiction filter** (numeric heuristic, context-aware) —
   removes pairs with mutually-exclusive numeric claims (year vs.
   count classification per §12.32). Runs after redundancy so a
   single duplicate cluster cannot dominate the contradiction
   evidence count.
4. **Entity-overlap pruning** — drops chunks whose entity set is a
   strict subset of a higher-ranked chunk's entity set; preserves
   broader-coverage chunks.
5. **Entity-mention filter** (tiered, top-K RRF immune, survivor floor)
   — assigns each chunk a tier: tier 0 mentions a *specific* query
   entity (multi-word, or a distinctive single token ≥ 8 chars), tier 1
   mentions only a generic entity or has strong query content-word
   overlap, tier 2 mentions neither. Tier-2 chunks are dropped, with two
   safeguards: the **top-2 RRF chunks are immune** (the implicit-bridge
   carve-out), and a **survivor floor** guarantees that when a full
   candidate set (≥ 5 chunks) was supplied the filter never reduces it
   below 5 — if matching kept fewer, it tops up with the highest-RRF
   dropped chunks. The floor engages only for full sets, so small inputs
   are still filtered normally (it does not force-keep noise in a 2–3
   chunk set). When the query yields *no* usable entities the filter no
   longer silently passes everything: it logs a structural warning and
   sets `_entity_filter_skipped`, then relies on RRF/reranker ranking —
   making the no-gating case observable rather than silent.
6. **Context shrinkage** — last, because it commits to a final budget;
   any filter that runs after this would operate on an already-truncated
   pool.

The cross-encoder reranker (Stage 2.5) runs *after* the chain because
it is the only step with a per-pair neural cost (~2 ms × K on CPU); it
must operate on a small, pre-cleaned pool to stay within the latency
budget defined in §1. The ordering is empirically defended by
`agentic_ablation.py` row-3 / row-4 deltas.

**Cross-encoder reranker (Stage 2.5).** When `enable_reranker: true`,
the top-K candidates after RRF are rescored with
`cross-encoder/ms-marco-MiniLM-L-6-v2` (Reimers & Gurevych 2019). The
key implementation detail: **the reranker scores
`(_best_sub_query, chunk)`, not `(surface_query, chunk)`**, so bridge
chunks semantically distant from the surface form are not demoted.

**Top-K RRF immunity.** The entity-mention filter keeps the top-2 RRF
chunks regardless of entity-mention status. This guards against
over-aggressive entity filtering when the answer-bearing chunk is an
implicit bridge target whose surface form does not contain the planned
entity name.

**Retrieval provenance.** The Navigator emits per-chunk metadata
including which retrieval method(s) produced each chunk
(`vector`/`graph`/`bm25`/`hybrid`). This metadata is used by the
Verifier's credibility score (see §4.3) — graph-retrieved chunks
receive a higher provenance weight than vector-only chunks.

**Iterative multi-hop.** When the plan's `hop_sequence` has dependent
hops (`depends_on` non-empty), the Navigator runs hops **sequentially**
rather than in parallel: bridge entities resolved at hop *i* are
injected into the sub-query of hop *i+1* via
`AgenticController._rewrite_hop_query_with_bridges` (see §4.5). This
implements the IRCoT (Trivedi et al. 2023) and HippoRAG
(Gutiérrez et al. 2024) pattern of feeding step-N retrieved entities
into step-(N+1) queries.

### 4.3 Verifier (S_V)

**File:** [`src/logic_layer/verifier.py`](src/logic_layer/verifier.py)

S_V is the final stage of the pipeline and implements two pre-generation
validation checks, four query-type-specialised prompt templates, a
quantised-SLM generation call, and an optional self-correction loop.

#### Pre-generation validation (default ON)

1. **Entity-Path Validation** — verifies that retrieved chunks cover
   the query entities. When a graph store is available,
   `find_chunks_by_entity_multihop` is used; otherwise falls back to
   substring matching. Coverage below
   `entity_coverage_threshold` (default 0.34) flags the plan as
   `INSUFFICIENT_EVIDENCE`, triggering a specialised prompt.
2. **Source Credibility Scoring** — weighted combination of three
   signals (Dong et al. 2014 KDD multi-source fusion):
   - Cross-reference corroboration (weight 0.4): the chunk shares a
     key phrase or an entity name token with another chunk.
   - Named-entity frequency (weight 0.3): chunk's SpaCy NER density,
     normalised.
   - Retrieval provenance (weight 0.3): graph-retrieved chunks score
     1.0; vector/BM25-only chunks score `provenance_baseline` = 0.5.

   The weights are documented as a deliberate inspection-time choice;
   the total contribution is bounded above by the chunk-eviction rate
   at `min_credibility_score`. The thesis reports one ablation row
   (without credibility filtering) rather than a weight sweep.

An **ablation-only** check (default OFF, `enable_contradiction_detection: false`):
Verifier-side NLI contradiction detection. The Navigator's
numeric-divergence filter already runs on the same context and is
enabled by default; the Verifier-side NLI check requires a 270 MB
cross-encoder download and is therefore retained only as a
research-mode toggle for ablation studies.

#### Context ordering and budgeting

Before the prompt is built, the surviving context is re-ordered and
truncated so the answer-bearing chunk reaches the small LLM early and
intact (small models attend poorly to later/long context;
Liu et al. 2024 "lost in the middle").

1. **Relevance re-ordering** (`_reorder_by_question_relevance`). Chunks
   are stable-sorted by a score combining three terms:
   - *IDF-weighted query-term overlap* — a query term occurring in many
     candidate chunks (a generic category word like "magazines") is
     down-weighted; a rare term (the specific entity) is decisive
     (inverse document frequency over the candidate set; Spärck Jones
     1972; Robertson 2004). IDF is applied only with ≥ 4 candidates;
     below that, document frequency is degenerate and the score falls
     back to length-normalised hit count.
   - *sqrt(word-count) length normalisation* — a short direct-answer
     chunk is not penalised against a long topic chunk that accumulates
     hits from sheer length (standard TF length normalisation).
   - *structural-coverage floor* — a chunk naming a distinctive query
     entity (multi-word, or a single token ≥ 8 chars) receives a score
     floor, so a required entity's article (a comparison conjunct, or a
     bridge target) cannot be demoted below the context cap by keyword
     sparsity.
2. **Sentence-aware per-doc truncation** (`_truncate_sentence_aware`).
   When a chunk exceeds `max_chars_per_doc`, the most query-relevant
   *sentences* are kept (in original order, preserving local coherence)
   rather than a blind head-truncation — so an answer-bearing sentence
   in the tail of a chunk survives. Falls back to head-truncation when
   no query is supplied or the chunk has no sentence structure.

#### Prompt selection

S_V selects one of four prompt templates based on `query_type` and
pre-validation status:

| Prompt | When |
|---|---|
| `ANSWER_PROMPT` | Single-hop / temporal / aggregate / default. |
| `BRIDGE_PROMPT` | `query_type in {"multi_hop", "bridge"}` and `hop_sequence` non-empty. The prompt includes a reasoning scaffold built by `_build_bridge_chain`. Non-final steps render as the directive "→ identify the intermediate result" rather than a pre-filled bridge entity: an entity merely *appearing* in context does not mean it *answers* the sub-query, and injecting it was observed to poison the SLM (it copied a wrong-but-grounded value, or over-abstained). The final step renders as "→ derive the final answer" (never a literal placeholder a small model could echo). |
| `COMPARISON_PROMPT` | `query_type == "comparison"`. |
| `INSUFFICIENT_EVIDENCE_PROMPT` | Pre-validation reports insufficient context. |

`CORRECTION_PROMPT` is used in iteration ≥ 2 of the self-correction
loop when prior iterations produced violated claims.

#### Self-correction loop

Up to `max_iterations` rounds (default 2). Each round:

1. Call the SLM with the selected prompt.
2. Extract atomic claims via SpaCy sentence splitting + meta-statement
   filtering.
3. Verify each claim against the graph store and/or retrieved context.
4. If all claims pass → return early.
5. If any claim violated → re-prompt with `CORRECTION_PROMPT` listing
   the violated claims.

The implementation follows Self-Refine (Madaan et al. 2023). Claim
verification is a conservative entity-presence proxy, not logical
entailment (Kryscinski et al. 2020 framing).

**Disclaimer override.** When the SLM returns an epistemic disclaimer
("I don't know", "no information", etc.), the verifier forces a
violated-claim entry and `LOW` confidence so the orchestrator does
not report a non-answer as `all_verified=True`.

**No-entity claim grounding.** Claims with no extractable proper noun
fall through to a token-grounding check whenever the claim is short
(≤ 6 tokens) or contains a numeric token. This catches hallucinated
short factual answers ("9 million inhabitants" vs the context's
"1.5 million inhabitants") that previously auto-verified.

#### Iteration history

`VerificationResult.iteration_history` records per-iteration answer,
claims, latency, and error flag. Strings are truncated to 200
characters with a `...[truncated]` marker so 500-question JSONL files
remain `jq`/pandas-tractable.

### 4.4 Pipeline Orchestrator (`AgentPipeline`)

**File:** [`src/pipeline/agent_pipeline.py`](src/pipeline/agent_pipeline.py)

`AgentPipeline` is the **single production orchestrator** of the
three-agent pipeline. All evaluation scripts call `pipeline.process(query)`
and consume `PipelineResult` dataclasses.

```python
class AgentPipeline:
    def __init__(
        self,
        planner:          Optional[Planner]     = None,
        navigator:        Optional[Navigator]   = None,
        verifier:         Optional[Verifier]    = None,
        hybrid_retriever: Optional[Any]         = None,
        graph_store:      Optional[Any]         = None,
        config:           Optional[Dict[str, Any]] = None,
    ): ...
    def process(self, query: str) -> PipelineResult: ...
```

Responsibilities:

- Construct each agent lazily on first `process()` call (or accept
  pre-built instances for tests).
- Chain S_P → S_N → S_V in fixed sequence.
- Forward the planner's `query_type` and `bridge_entities` (resolved
  by iterative-navigate) into the verifier so BRIDGE / COMPARISON
  prompt selection actually fires at evaluation time.
- Forward per-chunk graph-provenance flags
  (`chunk_is_graph_based`) into the verifier so the credibility
  score uses a real signal instead of a constant baseline.
- Maintain a FIFO query-result cache for repeated benchmark queries
  (`enable_caching: true`, `cache_max_size: 100` by default).
- Track aggregate statistics via Welford's incremental mean
  (Welford 1962).

The factory `create_full_pipeline(hybrid_retriever, graph_store, config)`
constructs the pipeline used by the evaluation harness. It attaches
the `BatchedOllamaEmbeddings` instance as `pipeline._embeddings` so
the evaluation summary can print cache hit-rate / batch-count /
average per-text latency at the end of every run.

### 4.5 Static Helper Namespace (`AgenticController`)

**File:** [`src/logic_layer/controller.py`](src/logic_layer/controller.py)

`AgenticController` is a stateless container of utility helpers
consumed by `AgentPipeline._iterative_navigate`. It contains:

- `_extract_bridge_entities(chunks, exclude, query)` — bridge-entity
  extraction from retrieved chunk text. **Pass 0** (GPE queries only)
  applies a location-context regex ("in the city of X", "capital of X")
  and returns early. The former surname-anchor (Pass 1) and
  general-proper-noun (Pass 2) passes are **merged into one scored
  candidate pool** rather than a priority-ordered cascade: the old
  early-return on the first surname match let a low-precision
  reconstruction preempt a stronger general candidate (e.g. a spurious
  "Salisbury Gardens" blocking the real "Thomas Mawson"). Candidates from
  both generators now compete on the same scoring function and the
  strongest wins. Chunks arrive in RRF rank order, so the list index is
  the chunk's retrieval rank. A substring-aware exclusion drops contiguous
  sub-phrases of an excluded compound entity.
- `_score_bridge_candidate(candidate, chunk, query, expected_type, chunk_rank)` —
  query-keyword proximity + expected-type bonus − position penalty, with
  type/length features gated on positive proximity, the whole **multiplied
  by a reciprocal chunk-rank prior `1/(1+rank)`** (Cormack et al. 2009;
  the same primitive RRF uses) so an entity from a top-ranked chunk
  outranks one from a low-ranked noise chunk. The local score is clamped
  non-negative before the rank multiply. A returned score of 0 means
  "not found / no query proximity".
  **Abstention floor:** if the best candidate scores ≤ 0,
  `_extract_bridge_entities` returns `[]` — a confidently-wrong bridge
  would misdirect hop-2 retrieval (and reranker hints), which is worse
  than none; hop-2 then falls back to its un-rewritten sub-query.
- `_detect_expected_type(query)` — interrogative-word → expected entity
  type (who → PERSON; where → GPE; when → DATE).
- `_rewrite_hop_query_with_bridges(sub_query, bridges)` — IRCoT-style
  hop-query rewriting that appends resolved bridge entities to the
  next-hop sub-query.

This module has **no `__init__`**, no orchestrator logic, no
runtime state. It is a namespace, not an agent. The production
orchestrator is `AgentPipeline`.

---

## 5. Ingestion: Decoupled Three-Phase Architecture

Phase boundaries are chosen so the GPU-bound entity-extraction step
runs separately from the CPU-only ingestion target.

| Phase | Tool | Hardware | Input | Output |
|---|---|---|---|---|
| **1** | `python -m src.thesis_evaluations.benchmark_datasets ingest --chunks-only` | CPU (local) | Raw dataset corpus | `chunks_export.json` (chunk text + metadata) |
| **2** | `colab_extraction.py` | GPU (Colab T4) | `chunks_export.json` | `extraction_results.json` (GLiNER entities + REBEL relations + per-triplet log-prob confidence) |
| **3** | `python local_importingestion.py` | CPU (local) | Both JSONs | LanceDB + KuzuDB populated |

### Phase 3 sub-phases

| Step | Description |
|---|---|
| 3a | LanceDB ingest (vectors only) |
| 3b | KuzuDB ingest: DocumentChunk + SourceDocument + FROM_SOURCE + NEXT_CHUNK + Entity + MENTIONS + RELATED_TO (REBEL) + SVO narrative relations. Uses `add_entities_bulk` / `add_mentions_relations_bulk` (batched CREATE after Python-side dedup) instead of per-edge MERGE — MERGE on KuzuDB MENTIONS adjacency scaled with entity degree and dominated wall-clock time before this change. |
| 3c | Co-occurrence edges (`RELATED_TO {relation_type='cooccurs'}` between every entity pair sharing a chunk) |
| 3c.5 | Subsumptive cleanup: drop cooccurs edges between pairs that already have a semantic edge |
| 3d | Cleanup: stop-list / orphan / hub / duplicate merge |
| 3d.5 | Embedding-based entity linking (alias resolution beyond canonical_form). **Disabled by default** for paper-release ingest; see §3.6.1. When enabled, uses `_redirect_entity_edges_bulk` (batched CREATE) and persists per-bucket completion to the checkpoint so a mid-phase crash skips already-linked buckets on resume. |
| 3e | Baseline metrics + invariant checks |
| 3f | Post-link isolated-entity drop |

Each phase writes a checkpoint to
`data/<dataset>/graph/.import_checkpoint.json`; `--resume` re-runs only
unfinished phases. Phase 3d.5 additionally stores a `done_buckets`
list so partial progress survives crashes within the phase.

---

## 6. Configuration System

**File:** [`config/settings.yaml`](config/settings.yaml) — single source of truth.

| Top-level block | Owns parameters for |
|---|---|
| `embeddings` | Ollama embedding model, base URL, dimension, cache path. |
| `chunking` | Strategy selection. |
| `vector_store` | LanceDB path, distance metric, normalisation, overfetch factor. |
| `graph` | KuzuDB path, traversal config (`max_hops`, `top_k_entities`, `enable_hop3`, `hub_mention_cap`). |
| `rag` | Retrieval mode (`vector` / `graph` / `hybrid`), per-source weights, RRF constant `k`, BM25 toggle. |
| `navigator` | Filter thresholds, reranker, contradiction filter, corroboration weights. |
| `planner` | Classifier weights, entity-density thresholds, classification confidence calibration. |
| `verifier` | Pre-validation flags, credibility weights, claim-extraction limits, `max_iterations`, fallback confidence. |
| `llm` | Active model, base URL, context budget (`max_docs`, `max_chars_per_doc`, `max_context_chars`), timeout. |
| `available_models` | Cross-model ablation roster (Ollama tag → metadata). |
| `agent` | Pipeline-level flags (`enable_planner`, `enable_verifier`, `enable_caching`, `cache_max_size`). |
| `entity_extraction` | GLiNER prompts, threshold, REBEL config — the values pinned for Phase 2. |
| `ingestion` | Chunker selection, `sentences_per_chunk`, `sentence_overlap`, spaCy model. |
| `quantization`, `paths`, `logging`, `performance`, `benchmark` | Operational. |

Every configurable threshold referenced in this document is loadable
from `settings.yaml` via the corresponding `from_yaml(cfg)` factory.
Dataclass defaults exist only as documented emergency fallbacks for
unit-test scenarios.

---

## 7. Evaluation Framework — Artifact C

**Directory:** [`src/thesis_evaluations/`](src/thesis_evaluations/)

### 7.1 Benchmark runner

[`benchmark_datasets.py`](src/thesis_evaluations/benchmark_datasets.py)
exposes `ingest`, `evaluate`, and `ablation` sub-commands.

**Question selection (`evaluate`).** The evaluator draws a **random
sample by default** (`--samples N`, default 20) so repeated runs probe
different questions and a gain is shown to be robust rather than tuned to
a fixed prefix. The random seed is auto-generated and logged
(`Question sample seed: N  (re-run with --seed N to reproduce)`), and
`--seed N` fixes the draw for exact reproduction or for comparing two
code versions on the same questions. For deterministic selection,
`--range START-END` takes `questions[START:END]` in index order
(separators `-`, `_`, `:`): `--range 0-20` reproduces the pre-2026-05
"first 20" behaviour; `--range 10-30` runs a defined band or shards a
large run. `--range` overrides `--samples` when both are given.

The evaluation summary block reports:

- Exact Match, F1, average wall-clock per query, coverage.
- Supporting-fact F1, SF-Recall (`all_gold_retrieved` rate).
- Pipeline / LLM failure decomposition.
- Per-question-type breakdown (bridge / comparison / compound).
- **Embedding cache metrics** (cache hit rate, batch count, average
  per-text latency).
- Per-question JSONL with `matched_pattern`, `classifier_preempt`,
  `verifier_iterations`, `all_verified`, `confidence`, and all retrieval
  metrics for downstream `jq`/pandas analysis.

### 7.2 Tier-1 ablation scripts

| Script | Role |
|---|---|
| `agentic_ablation.py` | Five-row decomposition: LLM-only → +Retrieval → +Planner → +Verifier → +SelfCorrect. Each row isolates one contribution. |
| `quantization_sweep.py` | Cross-model EM/F1/SF-F1 sweep across Ollama tags. |
| `latency_memory_profile.py` | Per-stage wall-clock + peak RSS distribution. |
| `chunking_ablation.py` | Per-`(sentences_per_chunk, sentence_overlap)` cell: re-chunks, ingests a per-config vector store (graph held constant), runs retrieval-only eval, writes summary.md. |

### 7.3 Diagnostic tooling

| Tool | Use |
|---|---|
| `diagnose.py` | Layer-by-layer dump for a single question (Planner output, Navigator filter chain, Verifier output). |
| `diagnose_verbose.py` | Full pipeline trace with per-stage gold-paragraph tracking. Honours `--skip-llm` for fast non-LLM passes. |
| `diagnose_ingestion.py` | Phase-3 consistency probe (counts vs. checkpoint vs. graph baseline). |
| `diagnose_graph_baseline.py` | Read-only graph-quality reporter. |

---

## 8. Technology Stack

### 8.1 Core dependencies

| Library | Role |
|---|---|
| **lancedb** | Embedded vector store (Apache Arrow columnar storage; HNSW). |
| **kuzu** | Embedded property graph (columnar, vectorised Cypher). |
| **sqlite3** | Persistent embedding cache. |
| **spacy** | Tokenisation, sentence splitting, NER (en_core_web_sm). |
| **gliner** | Zero-shot span-based NER (Phase 2 entity extraction; local fallback). |
| **transformers** | REBEL relation extraction (Phase 2 GPU; not used at query time). |
| **sentence-transformers** | Cross-encoder reranker (Stage 2.5 in Navigator). |
| **rank_bm25** | BM25 sparse retrieval. |
| **requests** | Ollama HTTP client. |
| **langchain-core** | `Embeddings` interface compatibility. |
| **coreferee** *(optional)* | Pronoun coreference for Phase-1 chunking. |
| **pytest** | Test runner; markers in `pytest.ini` separate slow/nightly/llm/integration tests. |

### 8.2 External services (local)

| Service | Endpoint | Models in production roster |
|---|---|---|
| Ollama | `http://localhost:11434` | `qwen2:1.5b` (primary), `qwen2.5:3b`, `gemma2:2b`, `llama3.2:3b`, `phi3`, `nomic-embed-text` (embeddings) |

### 8.3 Database selection rationale

- **LanceDB** — Embedded, Apache Arrow columnar, supports HNSW and
  IVF-Flat indexes. No server process.
- **KuzuDB** — Native Cypher, memory-mapped, fast for the 1–3 hop
  traversals this thesis requires. Out-of-core for graphs > RAM (Feng
  et al. 2023).
- **SQLite** — Universal, zero-configuration cache substrate.

---

## 9. Data Flows

### 9.1 Ingestion flow (Phase 1 → 3)

```
   Raw corpus (HotpotQA articles)
              │
              ▼
   Phase 1: SpacySentenceChunker + (optional) coreference
              │
              ▼
   chunks_export.json  ─── upload ───▶  Google Colab (GPU)
                                            │
                                            ▼
              Phase 2: GLiNER NER + REBEL RE
                       (log-prob calibrated confidence)
                                            │
                                            ▼
                                   extraction_results.json
                                            │
              ┌─────────────────────────────┘
              ▼
   Phase 3: local_importingestion.py
              │
              ├─→ 3a Vector ingest (LanceDB)
              └─→ 3b-3f Graph ingest (KuzuDB):
                       chunks + entities + MENTIONS +
                       RELATED_TO (REBEL + SVO) +
                       cooccurrence + cleanup +
                       embedding-based entity linking
```

### 9.2 Query flow

```
   user query
       │
       ▼
   ┌───────────────────────────────────────────────────────────┐
   │           AGENT PIPELINE (process)                         │
   │                                                            │
   │   1. S_P planner.plan(query)                              │
   │        ↳ RetrievalPlan(query_type, sub_queries,           │
   │          entities, hop_sequence, matched_pattern,         │
   │          classifier_preempt)                              │
   │                                                            │
   │   2. S_N navigator.navigate(plan, sub_queries)            │
   │        a. For each sub-query: HybridRetriever.retrieve    │
   │             ├── dense (LanceDB)                            │
   │             ├── BM25 (rank_bm25)                           │
   │             └── graph multi-hop (KuzuDB)                   │
   │        b. RRFFusion(vector, bm25, graph)                   │
   │        c. Cross-encoder rerank (optional)                  │
   │        d. 6-stage pre-generative filter chain              │
   │        ↳ filtered_context + retrieval_methods              │
   │                                                            │
   │   3. (multi-hop only) Iterative navigate:                  │
   │        for hop in plan.hop_sequence:                       │
   │          bridges = AgenticController._extract_bridge      │
   │                       _entities(chunks, exclude, query)    │
   │          next_hop.sub_query = _rewrite_hop_query           │
   │                       _with_bridges(sub_query, bridges)    │
   │                                                            │
   │   4. S_V verifier.generate_and_verify(query, context,     │
   │        entities, hop_sequence, query_type,                │
   │        bridge_entities, chunk_is_graph_based)              │
   │        ↳ pre-validation                                    │
   │        ↳ prompt selection (ANSWER / BRIDGE /              │
   │           COMPARISON / INSUFFICIENT_EVIDENCE)              │
   │        ↳ self-correction loop (up to max_iterations)      │
   │                                                            │
   └────────────────────┬───────────────────────────────────────┘
                        ▼
              PipelineResult(answer, confidence, planner_result,
                             navigator_result, verifier_result,
                             planner_time_ms, navigator_time_ms,
                             verifier_time_ms, total_time_ms,
                             cached_result)
```

---

## 10. Non-Functional Requirements

### 10.1 Reproducibility

- All hyperparameters are sourced from `config/settings.yaml`. No
  hard-coded design-time values remain in production code; dataclass
  defaults are emergency fallbacks only.
- Chunk IDs are deterministic SHA-256 hashes of source + position +
  text prefix, so re-ingestion is idempotent.
- Phase-2 (Colab) writes a checkpoint hashed over
  `(chunks_hash, config_hash)`; resumes are valid only when both hashes
  match.
- Phase-3 writes per-step checkpoints under
  `data/<dataset>/graph/.import_checkpoint.json`; `--resume` re-runs
  only unfinished steps.
- `requirements_frozen.txt` pins library versions; the active Ollama
  model is recorded in every evaluation's summary block.

### 10.2 Observability

- Every agent logs at `INFO` for stage transitions and `DEBUG` for
  per-pattern decisions.
- The Verifier records `iteration_history` (per-iteration claims,
  verification status, latency).
- `RetrievalPlan` records `matched_pattern` and `classifier_preempt`
  so per-question JSONL supports per-pattern aggregation.
- `EmbeddingMetrics` accumulates cache hit-rate / batch-count /
  per-text latency; reported in every evaluation summary.
- `_install_retriever_title_capture` in the eval harness records, for
  each filtered chunk, the source-document title that produced it —
  enabling supporting-fact tracking against gold labels.

### 10.3 Test coverage

The test suite collects **~450 tests** across the layers. Key modules:

| Module | Test file |
|---|---|
| Storage / retriever | `test_data_layer.py`, `test_graph_inspect.py` (nightly) |
| Embeddings | `test_embeddings.py` |
| Chunking | `test_chunking.py` |
| GLiNER bounds | `test_gliner_boundary.py` (nightly) |
| Planner | `test_planner_semantic.py`, `test_logic_layer.py` |
| Navigator | `test_navigator_semantic.py` |
| Verifier | `test_verifier_semantic.py` |
| Pipeline | `test_pipeline.py` |
| Thesis cleanup | `test_thesis_cleanup.py` — regression guards that the paper-cleanup pass is intact (no dataset-revealing strings in source, no removed-pattern markers, etc.) |

Markers in `pytest.ini` separate slow / nightly / llm / integration
tests; CI runs only the fast subset.

### 10.4 Security and data handling

- No outbound API calls at evaluation time: Ollama runs locally; the
  embedding model and SLMs are local.
- No PII processing in the thesis corpus (HotpotQA articles).
- The Colab phase uses Drive-mounted volumes; checkpoint and
  `extraction_results.json` are written atomically.

---

## 11. Design Decisions and Academic Grounding

### 11.1 Embedded databases over client-server

All three stores (LanceDB, KuzuDB, SQLite) are embedded. The edge
deployment target precludes a separate database server process; the
embedded choice also removes a class of failure modes (network
unavailability, schema migration, version drift).

### 11.2 Reciprocal Rank Fusion over linear interpolation

RRF (Cormack et al. 2009) is robust to mis-calibrated per-source
scores. Dense cosine similarity and BM25 are on incompatible scales;
linear interpolation requires per-corpus tuning. RRF only consumes
ranks, so adding a third source (graph) requires no recalibration.

### 11.3 Graph depth cap at 2 hops by default

Hop-3 retrieval is gated behind `graph.enable_hop3` because two-hop
noise compounds: a `cooccurs` edge followed by another `cooccurs`
edge approximates "anything-to-anything within the corpus". With
`enable_hop3: true`, cooccurs edges are excluded from hop-3 paths.

### 11.4 Hub suppression in graph retrieval

Entities mentioned in > 3 % of the corpus are excluded as bridge
targets. The 3 % threshold parallels TF-IDF's IDF cap (Salton & Buckley
1988): a term in too many documents loses discriminative power. The
result on HotpotQA: ~3 entities qualify as hubs ("United States",
"The Young Ones"-style multi-topic pages, "September 9 2013"-style
dates); their suppression prevents spurious cross-topic bridges
(West & Leskovec 2012 on hub-avoidance in graph IR; GraphRAG, Edge
et al. 2024).

### 11.5 Triple-frequency confidence over REBEL's per-triplet score

REBEL's per-triplet log-prob (now correctly calibrated in Phase 2)
ranks individual *edges*, not *bridges*. Bridge quality is a property
of the entity-pair plus the number of supporting chunks: more chunks
mentioning the pair = stronger corpus support. The
`_triple_frequency_confidence` score follows the corpus-support
inference of DeepDive (Niu et al. 2012) and Knowledge Vault (Dong
et al. 2014).

### 11.6 Cooccurs edges down-weighted, not deleted

Approximately 85 % of entity-pairs in the HotpotQA graph have only
cooccurs edges (no REBEL/SVO semantic relation). Deleting cooccurs
edges eliminates bridge connectivity for those pairs. The system keeps
them but weights them at 0.25 (vs 1.0 for REBEL) so any semantic
bridge with even one supporting chunk outranks any cooccurs bridge.
Following the PMI tradition (Church & Hanks 1990): co-mention is a
weak but non-zero signal — down-weight, do not delete.

### 11.7 Reranker scores against the best sub-query, not the surface query

Multi-hop bridge chunks are often semantically distant from the
surface question (the answer chunk is "Bob Seger" while the surface
is "what is the stage name?"). The cross-encoder reranker therefore
scores `(_best_sub_query, chunk)` rather than `(surface_query, chunk)`,
where `_best_sub_query` is the sub-query for which the chunk had the
highest per-sub-query RRF rank.

### 11.8 Planner pattern classification: dependency parse + surface heuristics

The Planner uses **two complementary mechanisms** for query type
classification.

**Primary mechanism — structural patterns (Patterns E, G, H, K, L):**
Named patterns use SpaCy dependency-parse labels (relcl, acl, agent,
auxpass, prep("of"), pobj) and closed-class English markers (both,
another) to fire structural decomposition. These cite linguistic
literature: Quirk et al. 1985; Bresnan 1982; Karttunen 1976/1977;
Partee 1995; Barker 1995; Levin 1993.

**Secondary mechanism — `MULTI_HOP_PATTERNS` (regex pre-screen):**
Before structural patterns fire, a set of surface-level regex patterns
(`MULTI_HOP_PATTERNS`) screens whether the query *may* be multi-hop.
This set includes closed-class syntactic markers ("of a/the X
that/who", possessive chains) as well as a curated list of attribution
verbs (starring, featuring, directed by, written by, composed by) and
relational nouns (father/mother/son/daughter of, creator/founder of)
that consistently signal an unresolved nominal bridge in English (Quirk
et al. 1985 §17.7-15). The pre-screen is conservative: a false
positive escalates to structural parsing, which may still emit a
single-hop plan; a false negative degrades to the generic 2-hop
fallback seeded on detected entities.

The combined approach trades theoretical elegance for empirical
robustness on HotpotQA bridge questions: structural patterns provide
defensible linguistic grounding; the surface pre-screen recovers
questions where the bridge is lexically marked but syntactically opaque
to the dependency parser (e.g., passive nominalizations, fragments).

### 11.9 Single-pass generation by default, self-correction as ablation

`max_iterations` defaults to 2 to enable one self-correction round
(Self-Refine, Madaan et al. 2023). The contribution of iteration 2 is
defended via the `agentic_ablation.py` row-5 vs row-4 comparison.

### 11.10 Single execution path (`AgentPipeline`); helpers as a namespace

An earlier iteration carried two orchestrator implementations: a
LangGraph StateGraph and a sequential fallback. Both produced identical
AgentState dicts but maintained twice the surface area. The LangGraph
mode and the sequential fallback have been removed. **`AgentPipeline`
in `src/pipeline/` is the only production orchestrator.**
`AgenticController` in `src/logic_layer/` is a stateless namespace of
bridge-handling helpers consumed by `AgentPipeline._iterative_navigate`.

### 11.11 Query-side NER normalization as a pre-consumer contract

GLiNER spans are consumed directly as graph anchors and filter tokens,
so boundary defects (an absorbed leading auxiliary, a fragmented
hyphenated name, an embedded year) propagate into retrieval. A
deterministic normalization layer repairs boundaries and gates
non-discriminative spans before use (§3.5). The rules are deliberately
narrow and closed-class (auxiliaries only, interior years only,
offset-overlap merges only) so they have no known false positives on
real entity names — wh-words and articles, which legitimately begin
titles, are not touched. This is a query-side-only contract; the
ingestion path is untouched, preserving entity-ID consistency.

### 11.12 Definite descriptions and classifier abstention as decomposition signals

Two decomposition mechanisms key on *general linguistic notions*, not
enumerated surface constructions (consistent with §11.8's structural
philosophy). (i) When a multi-hop query names no entity, the bridge
referent is given by a *definite description* — a heavily-modified noun
phrase denoting an entity by its properties (Russell 1905; Strawson
1950); the longest such entity-free phrase becomes the hop-0 query.
(ii) The classifier's no-signal sentinel (`SINGLE_HOP` at exactly the
fallback confidence) is treated as *abstention*, and only then is the
dependency parse consulted as a tie-breaker. Gating on the exact
sentinel — not a tuned threshold — guarantees the override cannot
regress any classification that had positive evidence.

### 11.13 IDF specificity and structural coverage in context ordering

The Verifier's context re-ordering combines IDF-weighted term overlap
(Spärck Jones 1972; Robertson 2004), sqrt length normalisation, and a
structural-coverage floor. IDF counters the failure where generic
category words ("magazines", "athletes") shared by many candidate
chunks dominate ranking over the rare entity term that actually
identifies the gold chunk; it is guarded to ≥ 4 candidates because
document frequency over fewer is degenerate. The coverage floor is a
*hard guarantee* (not a tuned weight) that a required entity's article
survives the context cap — restricted to distinctive entities so a
common single-word name does not over-fire.

### 11.14 Reciprocal-rank prior and abstention in bridge extraction

The bridge-entity scorer multiplies local relevance by a reciprocal
chunk-rank prior `1/(1+rank)` (Cormack et al. 2009): the retriever's own
rank is the strongest available prior for which chunk holds the bridge
referent, so an entity from a top-ranked chunk is preferred over one
from a low-ranked noise chunk. The three passes are merged into one
confidence-scored pool rather than a priority-ordered cascade, so a
low-precision heuristic can no longer short-circuit a stronger
candidate. An abstention floor (best score ≤ 0 → return none) converts a
confidently-wrong bridge — which would misdirect hop-2 retrieval — into
a neutral no-op, after which hop-2 runs on its un-rewritten sub-query.

### 11.15 Survivor floor over hard entity-mention dropping

The entity-mention filter (§4.2) is lossy by design. A survivor floor
guarantees that, when a full candidate set was retrieved, the filter
never strands the Verifier with too little context to tolerate a single
retrieval error (observed degenerate case: 10 → 2). It generalises the
top-2 RRF immunity into a floor and engages only for full candidate
sets, so it does not force-keep noise in already-small inputs. The
no-usable-entity case is made observable (logged + flagged) rather than
silently passing all chunks unfiltered.

---

## 12. Known Limitations and Future Work

### 12.1 Out of scope for the thesis evaluation

- Replacing GLiNER with a fine-tuned NER model.
- Training a learned reranker on the target corpus.
- Streaming generation output to the user during evaluation.
- Multi-GPU inference.

### 12.2 Documented limitations

1. **REBEL relation coverage.** REBEL is trained on Wikipedia
   infoboxes and emits Wikidata-style predicates ("date_of_birth").
   It misses narrative predicates ("X founded Y") that SVO recovers
   from the dependency parse, but neither captures arbitrary
   open-information predicates. The thesis discusses this as a recall
   ceiling on the graph-retrieval side.

2. **Pre-validation contradiction detection is disabled by default.**
   The Verifier's NLI-based contradiction check requires a 270 MB
   model download incompatible with the edge-deployment constraint.
   The Navigator's numeric-divergence filter on the same context is
   enabled instead. The Verifier-side check remains as an opt-in
   ablation only.

3. **Pre-empts are not in the four-phase scoring classifier.**
   Patterns I (boolean conjunction) and J (anaphoric "another") run
   as deterministic pre-empts before the SpaCy-Matcher scoring
   pipeline because the function-word markers ("both", "another")
   are unambiguous English structural signals and the scoring
   pipeline would otherwise misclassify them. Every pre-empt firing
   is logged via `RetrievalPlan.classifier_preempt` so its
   false-positive rate is auditable per question.

4. **Chunking ablation is one-dimensional.** The
   `chunking_ablation.py` script varies `(sentences_per_chunk,
   sentence_overlap)` while holding the graph (Phase-2 NER+RE output
   and KuzuDB store) constant. The result therefore measures the
   vector-retrieval component's sensitivity to chunking, not the full
   end-to-end system. This framing is explicit in the script's
   output summary.

5. **Coreference impact is not measured empirically.** Pre-chunking
   coreference resolution is enabled by default (when the resolver is
   installed) on the qualitative argument that pronoun-dropped
   mentions are unrecoverable downstream. The magnitude of the
   effect on graph density is dataset- and resolver-dependent and is
   not part of this thesis's quantitative evaluation.

6. **Embedding-based entity linking disabled.** As documented in
   §3.6.1, the empirical probe on the post-ingest entity set showed
   nomic-embed-text producing merge rates of 90–94 % on
   PERSON / LOCATION / GPE at every threshold tested up to 0.99, with
   single clusters absorbing more than one thousand unrelated
   entities. The linker was therefore disabled for the paper-release
   ingest and alias resolution reduces to `canonical_form`
   exact-match deduplication. Aliases that differ in surface form
   beyond canonical normalisation (e.g. "VCU" vs "Virginia
   Commonwealth University", "Ed Wood" vs "Edward Davis Wood Jr.")
   remain separate graph nodes. A discriminative in-type linker
   (e.g. SapBERT, BLINK) is identified as future work.

7. **Phase 3d.5 latency at scale is engineering-mitigated but
   experimentally unverified on real data.** The bulk-redirect helper
   `_redirect_entity_edges_bulk` and per-bucket checkpointing replace
   the per-edge MERGE pattern that previously made Phase 3d.5 a
   multi-hour bottleneck (measured 14–16 h on a ~46 000-entity
   HotpotQA graph). Because the phase is now disabled by default per
   limitation 6, the expected order-of-magnitude speedup of the new
   path is not reflected in any end-to-end ingest measurement
   reported in this thesis. Unit tests pin correctness on synthetic
   graphs.

---

## References (in-document citations)

- Barker, C. (1995). *Possessive Descriptions.* CSLI Publications.
- Bresnan, J. (1982). "The Passive in Lexical Theory." In Bresnan ed.,
  *The Mental Representation of Grammatical Relations*, MIT Press.
- Cabot, P. L. H., & Navigli, R. (2021). "REBEL: Relation Extraction
  by End-to-end Language generation." EMNLP 2021 Findings.
- Church, K. W., & Hanks, P. (1990). "Word Association Norms, Mutual
  Information and Lexicography." *Computational Linguistics* 16(1).
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
  "Reciprocal Rank Fusion outperforms Condorcet and individual rank
  learning methods." SIGIR '09.
- Dong, X. L., et al. (2014). "Knowledge Vault." KDD.
- Edge, D., et al. (2024). "From Local to Global: A Graph RAG
  Approach to Query-Focused Summarization." arXiv:2404.16130.
- Feng, X., et al. (2023). "Kùzu Graph Database Management System."
  CIDR 2023.
- Gutiérrez, B. J., et al. (2024). "HippoRAG: Neurobiologically
  Inspired Long-Term Memory for Large Language Models." NeurIPS;
  arXiv:2405.14831.
- Honnibal, M., & Montani, I. (2017). "spaCy 2." arXiv:1802.04016.
- Karttunen, L. (1976). "Discourse Referents." In McCawley ed.,
  *Syntax and Semantics 7*, Academic Press.
- Karttunen, L. (1977). "Syntax and Semantics of Questions."
  *Linguistics and Philosophy* 1(1).
- Khattab, O., et al. (2022). "Demonstrate-Search-Predict."
  arXiv:2212.14024.
- Kryscinski, W., et al. (2020). "Evaluating the Factual Consistency
  of Abstractive Text Summarization." EMNLP 2020.
- Levin, B. (1993). *English Verb Classes and Alternations.* University
  of Chicago Press.
- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks." NeurIPS 2020.
- Liu, N. F., et al. (2024). "Lost in the Middle: How Language Models
  Use Long Contexts." *TACL* 12; arXiv:2307.03172.
- Madaan, A., et al. (2023). "Self-Refine." NeurIPS 2023;
  arXiv:2303.17651.
- Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust
  approximate nearest neighbor search using hierarchical navigable
  small world graphs." IEEE TPAMI.
- Niu, F., et al. (2012). "Elementary: Large-scale Knowledge-Base
  Construction." *AI Magazine* 33(3).
- Partee, B. (1995). "Lexical Semantics and Compositionality." In
  Gleitman & Liberman eds., *An Invitation to Cognitive Science:
  Language*, MIT Press.
- Quirk, R., Greenbaum, S., Leech, G., & Svartvik, J. (1985). *A
  Comprehensive Grammar of the English Language.* Longman.
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT." EMNLP 2019;
  arXiv:1908.10084.
- Robertson, S. (2004). "Understanding inverse document frequency: on
  theoretical arguments for IDF." *Journal of Documentation* 60(5).
- Russell, B. (1905). "On Denoting." *Mind* 14(56).
- Salton, G., & Buckley, C. (1988). "Term-weighting approaches in
  automatic text retrieval." *Information Processing & Management*
  24(5).
- Spärck Jones, K. (1972). "A statistical interpretation of term
  specificity and its application in retrieval." *Journal of
  Documentation* 28(1).
- Strawson, P. F. (1950). "On Referring." *Mind* 59(235).
- Trivedi, H., et al. (2023). "Interleaving Retrieval with
  Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step
  Questions" (IRCoT). ACL; arXiv:2212.10509.
- Weischedel, R., et al. (2013). "OntoNotes Release 5.0." LDC2013T19.
- Welford, B. P. (1962). "Note on a method for calculating corrected
  sums of squares and products." *Technometrics* 4(3).
- West, R., & Leskovec, J. (2012). "Human Wayfinding in Information
  Networks." WWW.
- Yang, Z., et al. (2018). "HotpotQA: A Dataset for Diverse,
  Explainable Multi-hop Question Answering." EMNLP 2018. *Evaluation
  benchmark — no system component is dataset-specific.*
- Zaratiana, U., et al. (2023). "GLiNER: Generalist Model for Named
  Entity Recognition using Bidirectional Transformer."
  arXiv:2311.08526.
