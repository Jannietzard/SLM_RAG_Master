# Test suite — evidence for the thesis

665 tests pass on `python -X utf8 -m pytest test_system/ -q` (11 skipped:
`graph_inspect` requires a populated KuzuDB; `nightly`-marked tests are
deselected from the CI default). Each test file pins one or more design
decisions documented in `TECHNICAL_ARCHITECTURE.md`. The mapping below
lets a reviewer locate the test evidence for any thesis section.

| Test file | Thesis section / design decision | What it pins |
|---|---|---|
| `test_chunking.py` | §3.2 Chunking | `SpacySentenceChunker` window/overlap; deterministic SHA-256 chunk IDs (T-E invariant); `ChunkQualityFilter` thresholds. |
| `test_embeddings.py` | §3.1 Embeddings | `embed_query` vs `embed_documents` dim + cosine-≥0.99 identity (T-B invariant); `ConnectionError → RuntimeError` wrap; SQLite cache hit semantics. |
| `test_gliner_boundary.py` *(nightly)* | §3.3 Entity & Relation Extraction | GLiNER span-boundary correctness on compound entities ("Eiffel Tower" stays one span — T-C invariant). |
| `test_data_layer.py` | §3.4 Storage; §3.5 Hybrid Retriever; §11.16.4 settings-wiring | `HybridStore` + `KuzuGraphStore` close/lock semantics (F6); B1 query-side NER normalization (`_strip_leading_function_word`, year-strip, span-dedup); `ImprovedQueryEntityExtractor._is_junk_entity` discriminativeness gate. |
| `test_logic_layer.py` | §4.1–4.5 Logic Layer end-to-end | Configuration loaders; planner/navigator/verifier interface contracts; entity-mention filter safety fallback (§11.15 survivor floor); navigator filter ordering. |
| `test_planner_semantic.py` | §4.1 Planner; §11.8 pattern classification; §11.16.3 Phase-3.6 router | Query-type classification, Pattern E/F/G(+L)/H decomposition, Pattern I/J pre-empts, Phase-3.6 structural-comparison router (with bridge-cue precision guard), well-formedness invariant on emitted sub-queries (Item 4). |
| `test_navigator_semantic.py` | §4.2 Navigator; §11.13 IDF specificity; §11.15 survivor floor; §11.16.2 fair-cap | RRF fusion + cross-source boost; the 6-stage filter chain; `_fair_cap_by_subquery` per-anchor fairness merge for parallel decompositions (single-hop no-op assertion). |
| `test_verifier_semantic.py` | §4.3 Verifier; §11.13 IDF/structural-coverage; §11.16.1 cap-by-RRF-first | Pre-validation, entity-path validation, credibility scoring; `_reorder_by_question_relevance` IDF + length-norm + structural-coverage floor; **membership-invariant test for the cap-by-RRF-first contract** (a high-RRF answer chunk cannot be evicted by the lexical-overlap reorder); F2 sentence-aware truncation. |
| `test_pipeline.py` | §4.4 AgentPipeline | FIFO cache (T-D invariant: FIFO not LRU); lazy agent construction; per-stage timing surfaced on `PipelineResult`; `_close_pipeline` lock-release in ablation loop. |
| `test_thesis_matrix.py`, `test_thesis_matrix_ext.py` | §11 Design Decisions (capability matrix) | Coverage of §11.1–11.16 — every documented design decision has at least one test pinning the observable behaviour. |
| `test_missing_coverage.py` | §10.3 Test coverage (gap-filling) | Edge cases the higher-level tests miss; small surface-area guards. |
| `test_config_robustness.py` | §11.16.5 `_REQUIRED_SETTINGS` reproducibility guard | Config-loader robustness: missing keys produce a WARNING, not a crash; defaults match documented values; the 35-key validator runs. |
| `test_thesis_cleanup.py` | §11.10 single execution path; §11.16 paper-cleanup regression | Regression guards that paper-cleanup is intact (no dataset-revealing strings in source, removed-pattern markers absent, LangGraph references absent). |
| `test_bootstrap.py` | §7 Evaluation Layer (significance testing) | Paired-bootstrap CI/p-value semantics for thesis ablation tables. |
| `test_graph_inspect.py` *(skipped in CI)* | §3.4 Graph quality | Live-graph invariants (`isolated_rate`, `duplicate_rate`, `relations_per_chunk`); requires a populated KuzuDB. |

## Markers (`pytest.ini`)
- `slow` — long-running unit tests (deselected by default; run with `-m slow`).
- `nightly` — model-loading or live-resource tests (GLiNER weights, KuzuDB
  populated); not run in CI default.
- `llm` — tests that hit Ollama; deselected unless explicitly requested.
- `integration` — multi-component tests that don't fit unit scope.

## Test invariants (named guarantees pinned across files)
- **T-A** — `verifier.py`: answer with entity absent from context → violated_claims or LOW confidence.
- **T-B** — `embeddings.py`: `embed_query`/`embed_documents` same dim, cosine ≥ 0.99 for identical text.
- **T-C** — `entity_extraction`: compound spans ("Eiffel Tower") extracted as one entity.
- **T-D** — `agent_pipeline.AgentPipeline._cache` is FIFO (not LRU).
- **T-E** — `ingestion`: source_doc metadata isolated per `_chunk_document()` call; chunk IDs globally unique.

Each invariant has a deterministic test that fails loudly if the contract drifts.
