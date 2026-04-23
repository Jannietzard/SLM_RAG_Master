ARCHITECTURAL COHESION ANALYSIS: src/pipeline
STEP 0 — DIRECTORY PROFILE
src/pipeline is the integration and orchestration layer of the Edge-RAG system. It does not implement any algorithm or storage mechanism itself; its mandate is to wire the data layer (src/data_layer) and logic layer (src/logic_layer) into two complete, end-to-end executable pipelines: one for corpus ingestion (Artifact A) and one for query answering (Artifact B). The abstraction it provides to the outside world is simple: two entry points (create_ingestion_pipeline() / create_full_pipeline()) that return self-contained pipeline objects requiring only a query string or a source path to execute.

Public interfaces consumed by other layers: AgentPipeline, create_full_pipeline, PipelineResult, IngestionPipeline, create_ingestion_pipeline, IngestionConfig, IngestionMetrics, DocumentLoader. Internal implementation details that must not leak: lazy-init helpers (_lazy_init_agents, _init_chunker, _init_entity_extractor, etc.), configuration bridge functions (_verifier_config_from_cfg, _navigator_config_from_cfg, _planner_config_from_cfg), and all Mock* classes.

File	Lines	Role
agent_pipeline.py	915	Production — query pipeline
ingestion_pipeline.py	1,104	Production — ingestion pipeline
__init__.py	49	Public API surface
conftest.py	13	Test infrastructure
test_pipeline.py	775	Tests — co-located in production package
Total production code: ~2,068 lines. Test code: ~775 lines. Ratio ≈ 2.7 : 1. The ratio is reasonable, but the ratio is misleading because ingestion_pipeline.py itself contains ~130 lines of mock test infrastructure (MockEmbeddingGenerator, MockEntityExtractor), inflating the apparent production count.

STEP 1 — INTERFACE COHERENCE
1.1 __init__.py exports

The export list covers all 13 public symbols and is internally consistent — every name in __all__ has an import at the top. However, two symbols are problematic for a production API:

MockEmbeddingGenerator — a test stub that emits non-deterministic random vectors
MockEntityExtractor — a test stub that extracts capitalized words as entities
Both classes carry prominent "NEVER USE FOR THESIS EVALUATION" warnings in their docstrings. Exporting them from the package API means that any caller who does from src.pipeline import MockEmbeddingGenerator will not get an import error — there is no language-level enforcement that these classes are test-only.

1.2 External consumers

Symbol	Consumers
create_full_pipeline	benchmark_datasets.py:127, diagnose.py:453, test_system/diagnose.py:434, src/evaluations/evaluate_hotpotqa.py:185
AgentPipeline	benchmark_datasets.py, test_system/test_logic_layer.py
PipelineResult	test_system/test_logic_layer.py
BatchProcessor	test_system/test_logic_layer.py
create_pipeline (deprecated)	src/pipeline/test_pipeline.py
IngestionPipeline	local_importingestion.py, src/data_layer/ingestion.py
create_ingestion_pipeline	local_importingestion.py
MockEmbeddingGenerator, MockEntityExtractor	src/pipeline/test_pipeline.py only
DocumentLoader	src/pipeline/test_pipeline.py only
1.3 API bloat

MockEmbeddingGenerator and MockEntityExtractor are consumed only by test_pipeline.py. Exporting them as part of the package public API is a leakage of test infrastructure.
create_pipeline is explicitly deprecated (v4.1.0) but remains in __all__. The deprecation docstring acknowledges three live callsites in benchmark_datasets.py that have not yet been migrated.
1.4 Import stability

All production consumers (benchmark_datasets.py, diagnose.py, evaluate_hotpotqa.py) import directly from the internal module, bypassing __init__:


# actual usage in benchmark_datasets.py:127
from src.pipeline.agent_pipeline import AgentPipeline, create_full_pipeline
This makes __init__.py largely decorative for the actual consumers. The module docstrings in both production files also show the internal path (from src.pipeline.agent_pipeline import create_full_pipeline), not the package path, which trains new contributors toward the internal pattern. The package-level import (from src.pipeline import create_full_pipeline) is the correct interface but is not demonstrated or enforced anywhere.

1.5 Type consistency

Return types are consistent: process() always returns PipelineResult, ingest() always returns IngestionMetrics. Config factory functions all accept Dict[str, Any] and return typed dataclass instances. One inconsistency: AgentPipeline.__init__ takes hybrid_retriever: Optional[Any] and graph_store: Optional[Any] instead of the concrete HybridRetriever / KuzuGraphStore types guarded via TYPE_CHECKING — this is an intentional design choice (avoids heavy imports at construction time) but weakens static analysis.

STEP 2 — INTERNAL DEPENDENCIES AND DATA FLOW
2.1 Dependency graph


__init__.py
  ├── agent_pipeline.py
  │     └── (lazy) logic_layer: Planner, Navigator, Verifier
  └── ingestion_pipeline.py
        └── (lazy) data_layer: SpacySentenceChunker, BatchedOllamaEmbeddings,
                                EntityExtractionPipeline, HybridStore
agent_pipeline.py and ingestion_pipeline.py have no dependency on each other — clean vertical separation between query path and ingestion path.

2.2 Circular dependencies

None. All cross-package imports are guarded with either if TYPE_CHECKING: (static-only) or deferred into methods (from ..logic_layer.planner import Planner). This is the correct pattern for avoiding startup-time circular imports on an edge device.

2.3 Data flow — ingestion


source path
  → DocumentLoader.load() → Iterator[{id, text, metadata}]
    → _chunk_document() → List[chunk_dicts]
      → _extract_entities() → (entities, relations)
        → _embed_texts() → List[List[float]]
          → _store_data() → HybridStore (LanceDB + KuzuDB)
            → IngestionMetrics
2.4 Data flow — query


query: str
  → (cache check)
  → S_P: Planner.plan() → RetrievalPlan
    → S_N: Navigator.navigate() → NavigatorResult
      → S_V: Verifier.generate_and_verify() → VerificationResult
        → PipelineResult
          → (cache insert)
2.5 Layering

The layering is clean: agent_pipeline.py operates exclusively at the orchestration level (no direct storage calls); ingestion_pipeline.py operates at the data-movement level (calls data layer directly). There is no leakage of low-level concerns into the top-level pipeline logic.

STEP 3 — REDUNDANCY AND DUPLICATION
3.1 Duplicated initialization pattern

ingestion_pipeline.py contains four _init_* methods, all following the same structure:


def _init_<component>(self):
    try:
        from ..data_layer.<module> import <Class>
        return <Class>(<config_params>)
    except ImportError as e:
        logger.warning("FALLBACK ACTIVE: ...")
        return <MockOrNone>
This pattern is repeated across _init_chunker, _init_entity_extractor, _init_embedding_generator, _init_hybrid_store. The shape is identical; only the import path, class name, and fallback differ. The duplication is acceptable because each has a distinct fallback strategy (None, Mock, Mock, None), but a factory-method table would make this significantly more readable.

3.2 Duplicated agent construction logic

create_full_pipeline() manually constructs Planner, Navigator, and Verifier and then passes them into AgentPipeline.__init__. _lazy_init_agents() duplicates the same construction logic:


# create_full_pipeline() lines ~744–755
planner = create_planner(config)
nav_config = _navigator_config_from_cfg(config)
navigator = Navigator(nav_config)
verifier = Verifier(config=_verifier_config_from_cfg(config), ...)

# _lazy_init_agents() lines ~344–368 — same three constructions
self.planner = Planner(config=_planner_config_from_cfg(self.config))
self.navigator = Navigator(_navigator_config_from_cfg(self.config))
self.verifier = Verifier(config=_verifier_config_from_cfg(self.config), ...)
The only difference is that create_full_pipeline() calls create_planner(config) (which reads settings.yaml) while _lazy_init_agents() calls Planner(config=_planner_config_from_cfg(self.config)). The two code paths are not guaranteed to produce identical results.

3.3 Duplicated config reading

No duplication of _load_settings() within the directory itself — both pipelines delegate to their respective from_yaml() methods. No local _load_settings() in either file.

3.4 SHA-256 ID generation

DocumentLoader._generate_id() implements hashlib.sha256(source.encode()).hexdigest()[:16], which is the same pattern used in data_layer/chunking.py and data_layer/embeddings.py. This is acceptable at this layer boundary — the pipeline should not import internal data layer utilities — but the [:16] truncation and the "collision probability" comment are repeated verbatim across three files.

STEP 4 — SEPARATION OF CONCERNS
File	Single-sentence responsibility
agent_pipeline.py	Chains S_P → S_N → S_V with caching, timing, and ablation controls
ingestion_pipeline.py	Streams raw documents through chunking, extraction, embedding, and storage
__init__.py	Declares the public API surface of the integration layer
conftest.py	Inserts the project root onto sys.path for test discovery
test_pipeline.py	Verifies functional contracts of both pipeline classes
4.2 Files too large

Both agent_pipeline.py (915 lines) and ingestion_pipeline.py (1,104 lines) exceed 800 lines. In agent_pipeline.py, BatchProcessor and the three config-bridge helpers are natural split candidates. In ingestion_pipeline.py, DocumentLoader, MockEmbeddingGenerator, MockEntityExtractor, and IngestionConfig are all standalone enough to justify extraction into separate sub-modules.

4.3 Files too small

conftest.py at 13 lines is not premature abstraction — conftest files are necessarily small. No issue here.

4.4 Abstraction level mixing

ingestion_pipeline.py mixes four abstraction levels in a single file:

Configuration (IngestionConfig) — declarative data
I/O (DocumentLoader) — file format handling
Mock test infrastructure (MockEmbeddingGenerator, MockEntityExtractor) — test stubs
Orchestration (IngestionPipeline, create_ingestion_pipeline) — production pipeline
The mock classes especially do not belong in a production orchestration file.

4.5 Test co-location

test_pipeline.py lives inside src/pipeline/, making it part of the production Python package. This mirrors the problem previously found in src/logic_layer/test_logic_layer.py (already resolved by moving to test_system/). For academic publication the src/ tree should contain only production code; test files belong in test_system/.

STEP 5 — ERROR HANDLING CONSISTENCY
5.1 Strategy

The two files adopt different strategies at pipeline boundaries:

Location	Strategy
IngestionPipeline._store_data()	Re-raises after logging — correct
IngestionPipeline._init_entity_extractor()	ImportError → fallback mock; all other exceptions propagate — documented, correct
IngestionPipeline._init_chunker()	ImportError → None (silent fallback chunker) — a chunker returning None is accepted; the pipeline then uses the regex fallback
AgentPipeline.process()	No try/except — any exception from Planner, Navigator, or Verifier propagates unhandled to the caller
BatchProcessor.process_batch()	except Exception per-query — correct for a batch harness
5.2 Custom exceptions

No custom exception types are defined or used in this directory. Errors propagate as ValueError (from process() query validation), FileNotFoundError (from DocumentLoader.load()), or bare Exception (from BatchProcessor). Given that this is the integration layer (not a library), the absence of custom exceptions is acceptable.

5.3 Error propagation

A gap exists in AgentPipeline.process(): if the LLM is unavailable, Verifier.generate_and_verify() will raise an OllamaConnectionError or similar, which will propagate unhandled through process() to the caller (likely benchmark_datasets.py). There is no structured error-result path (e.g., returning a PipelineResult with confidence="error") for inference-time failures. The BatchProcessor covers this with a per-query try/except, but callers that call process() directly have no protection.

5.4 Systemic pattern

No systemic error suppression. The one borderline case — _init_chunker() returning None and falling back to a regex splitter — is documented inline and limited to the fallback path.

STEP 6 — CONFIGURATION CONSISTENCY
6.1 Systemic hardcoding pattern

Both pipelines correctly delegate configuration to from_yaml() in their respective config dataclasses. One notable hardcoding remains in agent_pipeline.py:


# agent_pipeline.py:758-759 — inside create_full_pipeline()
enable_caching: bool = agent_cfg.get("enable_caching", True)
cache_max_size: int = agent_cfg.get("cache_max_size", 1000)
These two keys are read directly in create_full_pipeline() instead of via AgentPipelineConfig.from_yaml() — there is no AgentPipelineConfig dataclass analogous to IngestionConfig. The cache parameters, ablation flags, and cache_max_size are inline dict.get() calls scattered across __init__ and create_full_pipeline().

6.2 Inconsistent config access pattern

IngestionPipeline uses a proper IngestionConfig dataclass with a from_yaml() class method — all parameters in one place, documented with settings.yaml key paths. AgentPipeline has no equivalent AgentPipelineConfig; config is read ad-hoc in three places (__init__, _lazy_init_agents, create_full_pipeline). This is the most significant configuration inconsistency in the directory.

6.3 Single point of loading

Both pipelines expect the caller to pass the pre-loaded settings.yaml dict — neither file calls _load_settings() itself. This is correct: loading is the caller's responsibility.

6.4 Conflicting defaults

IngestionConfig.embedding_batch_size defaults to 64 (matching settings.yaml → performance.batch_size). No conflicting defaults found.

STEP 7 — NAMING AND CONVENTION CONSISTENCY
7.1 Naming conventions

snake_case throughout for functions and variables; PascalCase for classes — consistent.

7.2 Terminology

"chunk" is used consistently in both files and matches src/data_layer. "sub_queries" in agent_pipeline matches navigator.py. No terminology drift detected.

7.3 Docstring style inconsistency

Both agent_pipeline.py and ingestion_pipeline.py use NumPy-style docstrings throughout:


Parameters
----------
query : str
    ...
Returns
-------
PipelineResult
The logic layer (src/logic_layer/) was standardized to Google-style in the previous review cycle (Args:, Returns:). The pipeline layer is therefore inconsistent with the layer it directly depends on. This inconsistency is systemic: every __init__, process(), ingest(), load(), and from_yaml() method uses NumPy style.

7.4 Logging

Module-level logger = logging.getLogger(__name__) in both files — consistent. %-style format strings throughout — consistent with the rest of the codebase.

STEP 8 — PUBLICATION READINESS
8.1 Purpose clarity

The module docstrings in both production files are exemplary — they state the architectural position, the scientific contribution, ablation controls, and a usage example. Reading agent_pipeline.py:1–75 or ingestion_pipeline.py:1–77 provides complete orientation without needing to look at any other file. The __init__.py docstring is thin (8 lines) but sufficient for a reader who has read the module docstrings.

8.2 README

No README.md for this directory. Given the quality of the module docstrings, a README is not strictly needed, but a short ARCHITECTURE.md note connecting this directory to the thesis chapters (Chapter 2 for ingestion, Section 3.1 for agent pipeline) would aid academic readers.

8.3 Academic references

Madaan et al. (2023) "Self-Refine" NeurIPS — cited correctly in agent_pipeline.py
Welford (1962) "Note on a method for calculating corrected sums" Technometrics — cited correctly
Cabot & Navigli (2021) "REBEL" EMNLP — cited correctly in ingestion_pipeline.py
Yang et al. (2018) "HotpotQA" EMNLP — cited correctly in DocumentLoader._load_jsonl()
All four citations are complete (author, year, title, venue). No missing DOIs or incomplete references found.

8.4 Non-publishable files

test_pipeline.py inside src/pipeline/ is a non-publishable file embedded in the production package tree. A reader browsing the src/ directory would encounter test code without realizing they left the production artifact.

8.5 Test coverage

The test file provides good functional coverage:

PipelineResult — 6 tests (structure, serialization, caching flags) ✓
AgentPipeline — 10 tests (init, caching, process, Enum serialization) ✓
BatchProcessor — 7 tests (batch, error containment, EM eval) ✓
IngestionConfig — 5 tests (from_yaml key paths, fallback defaults) ✓
DocumentLoader — 12 tests (all formats, directory recursion, HotpotQA format) ✓
IngestionPipeline — 7 tests (end-to-end ingest, metric reset, fallback chunker) ✓
Notable gaps: No tests for the Welford online mean computation, no test for _update_cache FIFO eviction, no test for get_store_stats().

8.6 Reproducibility

Both pipelines emit "FALLBACK ACTIVE" warnings whenever constructed without a settings.yaml dict. The create_pipeline() deprecation notice explicitly documents that its results are non-reproducible and names the three benchmark callsites that must be migrated. These are the right mechanisms, but the migration is incomplete.

STEP 9 — ARCHITECTURAL VERDICT
9.1 Cohesion score: 7/10

The two pipeline modules belong together and serve a clear unified mandate. The cohesion penalty comes from: mock test stubs embedded in the production file, test_pipeline.py co-located with production code, and the absence of an AgentPipelineConfig dataclass creating asymmetry between the two halves of the directory.

9.2 Coupling assessment

Coupling to src/logic_layer and src/data_layer is appropriate and intentional — this is the integration layer. All cross-layer imports use lazy deferral (TYPE_CHECKING, deferred method-level imports), so startup cost is minimized. The coupling to benchmark_datasets.py is inverted (consumer, not dependency) — clean.

9.3 Mandate fulfillment

The pipeline layer fully fulfills its mandate as the integration and entry-point layer. Both pipelines are wired correctly, the data flow is linear and traceable, and the ablation controls are properly implemented. The academic documentation within the files is of publication quality.

9.4 Restructuring advice

If redesigning from scratch, the directory would contain:


src/pipeline/
  __init__.py             (public API, no change)
  agent_pipeline.py       (AgentPipeline only, ~400 lines)
  agent_config.py         (AgentPipelineConfig dataclass, analogous to IngestionConfig)
  ingestion_pipeline.py   (IngestionPipeline + IngestionConfig, ~600 lines)
  document_loader.py      (DocumentLoader class, standalone)
Mock classes would live in test_system/helpers/mock_components.py, not in production files.

9.5 Overall grade: 2 (gut)

Both pipeline classes are well-designed, well-documented, and correctly wired. The architecture is sound. The deductions are for: test stubs in the production API (MockEmbeddingGenerator, MockEntityExtractor); a test file co-located inside the production package tree; the missing AgentPipelineConfig dataclass causing asymmetric configuration patterns; and the deprecated create_pipeline() function remaining live with three unresolved callsites.

9.6 Publication verdict: Conditionally Accepted

Must change before publication: items 1–5 in the action list below. Items 6–8 are strongly recommended.