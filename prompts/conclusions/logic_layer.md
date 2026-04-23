ARCHITECTURAL COHESION ANALYSIS: src/logic_layer
STEP 0 — DIRECTORY PROFILE
src/logic_layer implements Artifact B of the thesis: the agent-based query processing pipeline. Its architectural mandate is to transform a raw user query into a verified answer through three sequenced agents (S_P → S_N → S_V), with the controller as their orchestrator. To the rest of the codebase it provides a single entry point: create_controller() + controller.run(query). All retrieval is delegated to the data layer via an injected HybridRetriever; the logic layer never touches LanceDB or KuzuDB directly.

Public interfaces consumed externally: AgenticController, create_controller, Planner, Navigator, Verifier, and their associated config/result types. Internal implementation details that should not leak: the six filter methods in Navigator, _extract_bridge_entities, _iterative_navigator_node, the three _*_node methods in AgenticController, and the component subclasses QueryClassifier, EntityExtractor, PlanGenerator.

Size: 7,472 total lines across 7 files. Production code (controller, navigator, planner, verifier, __init__, conftest): ~4,541 lines. Test code (test_logic_layer.py): 1,931 lines. Ratio ≈ 2.4:1. Two production files exceed 800 lines (planner.py: 1,834; verifier.py: 1,679). The test file at 1,931 lines is the largest single file in the directory.

STEP 1 — INTERFACE COHERENCE
1.1 — What __init__.py exports

The export list is too permissive. Three internal component classes are listed with the comment "for advanced use":


# Component classes (for advanced use)
QueryClassifier,
EntityExtractor,
PlanGenerator,
These are sub-components of Planner — callers who construct Planner via create_planner() should never need to reach inside it. Exporting them invites coupling to implementation details. They belong in the private namespace.

ControllerConfig is exported from the navigator import block — a naming mismatch that will confuse readers (ControllerConfig has no logical home in navigator.py).

ConfidenceLevel is defined in verifier.py, imported in tests (from src.logic_layer.verifier import ConfidenceLevel — 5 occurrences in test_logic_layer.py) but is absent from __init__.py. This is an API gap.

1.2 — External consumers

Symbol	External consumers
AgenticController / create_controller	diagnose_verbose.py, test_logic_layer.py
Planner / create_planner	diagnose.py, diagnose_verbose.py, test_logic_layer.py, test_planner_semantic.py, src/pipeline/test_pipeline.py
Navigator / ControllerConfig	diagnose.py, diagnose_verbose.py, test_logic_layer.py, src/pipeline/test_pipeline.py
Verifier / create_verifier	diagnose.py, diagnose_verbose.py, test_logic_layer.py, test_verifier_semantic.py, src/pipeline/test_pipeline.py
1.3 — API bloat

QueryClassifier, EntityExtractor, PlanGenerator are never imported from outside the package — confirmed by grepping all Python files. They should be removed from __all__.

1.4 — API surface stability — CRITICAL

Every external consumer bypasses __init__.py entirely:


# diagnose.py — direct submodule imports
from src.logic_layer.planner import create_planner
from src.logic_layer.navigator import Navigator, ControllerConfig
from src.logic_layer.verifier import create_verifier

# test_logic_layer.py — direct submodule imports (95 occurrences)
from src.logic_layer.planner import create_planner, Planner
from src.logic_layer.verifier import VerifierConfig, ConfidenceLevel
The __init__.py API therefore provides zero stability guarantee in practice — any consumer can reach directly into a submodule and be broken by a move or rename. If ControllerConfig moves from navigator.py to controller.py, every direct import breaks.

1.5 — Type consistency

Return types are consistent: Planner.plan() → RetrievalPlan, Navigator.navigate() → NavigatorResult, Verifier.generate_and_verify() → VerificationResult. All are dataclasses with typed fields. AgentState is a TypedDict — a different pattern for the controller output, but appropriate given LangGraph compatibility. No type inconsistencies at the boundary.

STEP 2 — INTERNAL DEPENDENCIES AND DATA FLOW
2.1 — Dependency graph


planner.py          (no internal imports)
    ↑
navigator.py        imports: planner.RetrievalPlan
    ↑
controller.py       imports: navigator.{ControllerConfig, Navigator, NavigatorResult}
                             planner.{Planner, QueryType, RetrievalPlan, ...}
                             verifier.{Verifier, create_verifier}
    ↑
verifier.py         (no internal imports)

__init__.py         imports all four

conftest.py         (no internal imports)

test_logic_layer.py imports directly into all four submodules
2.2 — Circular dependencies: None.

2.3 — Data flow


query (str)
  → planner.Planner.plan()        → RetrievalPlan (sub_queries, entities, hop_sequence)
  → navigator.Navigator.navigate() → NavigatorResult (filtered_context, scores, metadata)
  → verifier.Verifier.generate_and_verify() → VerificationResult (answer, claims, iterations)
  → AgentState (all intermediate data + timings + errors)
2.4 — Misplaced files

ControllerConfig is defined in navigator.py (line 99) but is the configuration class for the AgenticController. Navigator only reads a subset of its fields (relevance_threshold_factor, redundancy_threshold, etc.). The full class includes model_name, base_url, max_verification_iterations — fields that are meaningless to the Navigator. This is the single most significant architectural misplacement in the directory.

2.5 — Layering

The dependency order (planner → navigator → controller, with verifier standalone) is clean and acyclic. The only tangle is the ControllerConfig misplacement, which forces controller.py to import its own configuration from navigator.py.

STEP 3 — REDUNDANCY AND DUPLICATION
3.1 — CRITICAL: _load_settings() triplicated

The function is copied verbatim across three files with only the error message differing:

File	Line	Error message suffix
planner.py	106	PlannerConfig dataclass defaults
verifier.py	154	VerifierConfig dataclass defaults
controller.py	95	ControllerConfig dataclass defaults
All three bodies are structurally identical (20 lines: Path resolution, yaml.safe_load, YAMLError catch, missing-file warning). Any change to the loading logic (e.g., adding environment variable override, switching from YAML to TOML) must be applied in three places.

3.2 — Proper-noun regex quadruplicated

The pattern [A-Z][a-z]+(?:\s+[A-Z][a-z]+)+ (multi-word capitalized phrase matching) appears in four independent locations:

File	Location	Purpose
planner.py:722	EntityExtractor.ENTITY_PATTERNS	Regex entity fallback
navigator.py:716	_entity_overlap_pruning() local extract_entities()	Entity proxy for pruning
controller.py:400	_extract_bridge_entities()	Bridge entity discovery
verifier.py:1054	_MULTI_PROPER_NOUN_RE class constant	Claim verification
Minor variant in navigator.py ([A-Z][a-zA-Z]+ instead of [A-Z][a-z]+) — a subtle behavioral difference with no comment explaining the divergence.

3.3 — Configuration loading: Three independent file loads. See 3.1.

3.4 — Shared constants: The proper-noun regex (3.2). No other problematic shared constants.

3.5 — Error handling: The except Exception + log + degrade pattern in controller.py nodes is consistent, not duplicated redundantly. Navigator's per-subquery broad catch follows the same justified pattern.

STEP 4 — SEPARATION OF CONCERNS
4.1 — Responsibilities

File	Responsibility
planner.py	Transforms a raw query into a structured RetrievalPlan via rule-based classification, SpaCy NER, and hop sequencing.
navigator.py	Executes a RetrievalPlan against a HybridRetriever and delivers filtered, ranked evidence chunks.
verifier.py	Validates retrieved evidence for contradictions and entity coverage, generates an answer via a quantised LLM, and iteratively self-corrects.
controller.py	Orchestrates the S_P → S_N → S_V pipeline, manages AgentState, and exposes the public run() interface.
__init__.py	Defines the stable public API surface for the logic layer.
conftest.py	Provides sys.path configuration for pytest discovery of this package.
test_logic_layer.py	Covers unit and integration behaviour of all four agent files.
4.2 — Oversized files

planner.py at 1,834 lines is large but justified: it contains QueryClassifier (254 lines with 25 regex patterns), EntityExtractor (260 lines), PlanGenerator (420 lines), PlannerConfig (85 lines), and 5 data structures. These are genuinely distinct components that could each be their own file. For publication, refactoring into _planner_classifier.py, _planner_extractor.py, _planner_generator.py would improve navigability without changing the public API.

verifier.py at 1,679 lines is similarly dense but coherent: pre-validation (450 lines), generation/self-correction loop (350 lines), and PreGenerationValidator (500 lines) are tightly coupled by design.

test_logic_layer.py at 1,931 lines is the largest file and is co-located with production code (see 4.5).

4.3 — Undersized files

conftest.py at 13 lines is minimal but correct.

4.4 — Abstraction levels

No mixing: no direct DB operations in this layer, no LLM calls in the data layer. The injected RetrieverProtocol in navigator.py isolates the data layer properly.

4.5 — Test co-location

test_logic_layer.py lives inside src/logic_layer/, alongside the production modules it tests. The test_system/ directory holds test_planner_semantic.py, test_verifier_semantic.py, and test_data_layer.py. The split is inconsistent: structural tests sit inside the package, semantic tests sit in test_system/. For publication, all test code should be outside the production package (test_system/test_logic_layer.py), matching the pattern recently established for the data layer.

STEP 5 — ERROR HANDLING CONSISTENCY
5.1 — Strategy

Consistent and intentional: broad except Exception at pipeline stage boundaries in controller.py (nodes: _planner_node, _navigator_node, _verifier_node), narrow exceptions in helper methods. Each broad catch is documented with a rationale comment. This is the correct pattern for a pipeline that must not abort on partial failures.

5.2 — Custom exception types

None used. Acceptable: the pipeline uses degraded output (empty context, error-prefixed answer, errors accumulation in AgentState) rather than exception propagation.

5.3 — Error propagation at boundaries

Correct: errors are accumulated in AgentState.errors as strings and surfaced to callers. The empty-context case in _iterative_navigator_node correctly appends a diagnostic message rather than silently returning empty.

5.4 — Systemic error suppression

No silent swallowing. Every except block logs at logger.error() with exc_info=True and appends to AgentState.errors. The navigation-layer per-subquery broad catch also appends to result.metadata["retrieval_errors"]. Clean pattern.

STEP 6 — CONFIGURATION CONSISTENCY
6.1 — Systemic pattern

The _load_settings() duplication is the defining configuration problem. It is not merely a style issue — three independent file loads means three places to break, three different error messages that will confuse debugging, and no opportunity for a shared override mechanism.

6.2 — Configuration reading pattern

All three factory functions (create_planner, create_verifier, create_controller) follow the same pattern:

Accept optional cfg: Dict[str, Any]
Auto-load from settings.yaml if cfg is None
Construct config dataclass via XxxConfig.from_yaml(cfg)
Apply keyword overrides
This is consistent and well-designed. The problem is step 2 being independently implemented in three places.

6.3 — Single point of configuration loading

There is none. The correct fix is a shared _settings.py (or _load_settings at the package level in __init__.py or a new _utils.py) that all three factories call.

6.4 — Conflicting defaults

No conflicts found. model_name defaults are both "qwen2:1.5b" in ControllerConfig and VerifierConfig. max_chars_per_doc is 500 in both. max_verification_iterations / max_iterations defaults are 2 in both. The numeric defaults are in sync, which is a positive sign of intentional coordination.

STEP 7 — NAMING AND CONVENTION CONSISTENCY
7.1 — Naming conventions

Consistent: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for module-level constants and class-level tuple constants (COMPARISON_PATTERNS, MULTI_HOP_PATTERNS).

7.2 — Terminology

Consistent across all files: "chunks", "entities", "hop", "sub_queries", "context". One minor inconsistency: log messages use "sub_query" as a dict key but free-text strings vary between "sub-query" (hyphenated) and "sub_query" (underscored). Trivial.

7.3 — Docstring format

Minor inconsistency: from_yaml() methods across all config classes use NumPy-style docstrings (section headings with underlines: Parameters\n----------), while other methods use Google-style (Args:, Returns:). The _iterative_navigator_node and _extract_bridge_entities use full Google-style. The inconsistency is within-file, not cross-file. Standardizing to one format (Google-style, as the majority) would take ~1 hour.

7.4 — Logging

logger = logging.getLogger(__name__) is consistently defined at module level before the LangGraph import block in all files. Log-prefix convention has a meaningful inconsistency: the controller's node methods log [S_P], [S_N], [S_V] for pipeline-level visibility, while navigator.py's internal methods log [Navigator]. When both run inside the same pipeline execution, the log stream is readable but the naming convention diverges. Recommend: controller keeps [S_P], [S_N], [S_V] for stage transitions; navigator.py should keep [Navigator] for its own internal operations (it is correct as-is).

STEP 8 — PUBLICATION READINESS
8.1 — Clarity from __init__.py and module docstrings

__init__.py has a clean 7-line docstring with a three-agent summary and factory usage example. Module docstrings in all four production files are exemplary: ASCII architecture diagrams, complete thesis section references, explicit algorithm descriptions, and review history blocks. A reader can understand each file's role without running the code.

8.2 — Internal architecture README

TECHNICAL_ARCHITECTURE.md exists at the project root and is referenced throughout. No README.md inside src/logic_layer/ — acceptable given the comprehensive module docstrings.

8.3 — Academic references

Coverage is thorough and consistent:

File	References
planner.py	Yang et al. 2018 (HotpotQA), Honnibal & Montani 2017 (SpaCy), Weischedel et al. 2013 (OntoNotes), Khattab et al. 2022 (DSP)
navigator.py	Cormack et al. 2009 (RRF, with DOI), Jaccard 1901
verifier.py	Madaan et al. 2023 (Self-Refine), Bowman et al. 2015 (NLI), Reimers & Gurevych 2019 (SBERT), Kryscinski et al. 2020 (Factual Consistency)
controller.py	Madaan et al. 2023 (Self-Refine), Trivedi et al. 2022 (Interleaving Retrieval + CoT)
One gap: navigator.py contains five original contributions (contradiction filter, entity-overlap pruning, entity-mention filter, context-shrinkage, cross-source corroboration boost) with thesis table references (thesis Table 4.2, thesis Table 4.3) but the table numbers are provisional. Confirm before submission.

8.4 — Non-publishable files

test_logic_layer.py inside src/logic_layer/ is not a publication concern per se (it is production-quality test code) but its co-location with production modules contradicts the Python package convention that src/ contains only importable production code.

8.5 — Test coverage

Component	Test Location	Coverage Assessment
Planner (S_P)	test_logic_layer.py + test_planner_semantic.py	Good — 40+ tests, classification edge cases, entity extraction, sub-query rewriting
Navigator (S_N)	test_logic_layer.py only	Gap — no dedicated test_navigator_semantic.py; the 6 filter methods (contradiction, redundancy, entity-mention, entity-overlap, shrinkage, RRF) have only structural tests
Verifier (S_V)	test_logic_layer.py + test_verifier_semantic.py	Good — 27 semantic tests
Controller	test_logic_layer.py (end-to-end, no Ollama)	Adequate for publication; full integration requires Ollama
The Navigator's filter pipeline is the highest-value untested path: each filter has a measurable EM impact (contradiction +1.4, entity-mention +2.1, entity-overlap +0.8) and has no semantic-level test suite.

8.6 — Reproducibility

The create_controller() factory reads from settings.yaml, enabling full reproduction. Review history blocks in all four files document the review date, finding counts, and the reviewer tool — excellent for thesis artifact accountability. The LangGraph fallback is documented and the thesis evaluation mode (simple pipeline) is clearly noted.

STEP 9 — ARCHITECTURAL VERDICT
9.1 — Cohesion score: 7 / 10

The three agents and controller form a tight, well-motivated unit. The dependency graph is clean and acyclic. Deductions: ControllerConfig misplacement (−1), _load_settings() triplication (−1), test_logic_layer.py co-location and bypassed __init__.py (−1).

9.2 — Coupling assessment

Low external coupling: the data layer is accessed only through two injected objects (HybridRetriever and graph_store), both typed as Any to avoid cross-layer imports. No direct LanceDB/KuzuDB references. This is correct for the edge deployment context.

Moderate internal coupling: controller.py is tightly coupled to all three agent files (by design — it orchestrates them). The import of ControllerConfig from navigator.py creates an unnecessary indirect coupling.

9.3 — Mandate fulfillment

Yes. The directory implements the S_P → S_N → S_V pipeline as specified in TECHNICAL_ARCHITECTURE.md sections 3.2–3.4. The iterative multi-hop controller (pending implementation in the MEMORY notes) is now implemented in _iterative_navigator_node. The self-correction loop matches the Madaan et al. Self-Refine specification. All thesis-claimed original contributions (6 filters, sub-query rewriting, bridge entity discovery) are present and documented.

9.4 — Restructuring advice

From scratch, I would: (1) introduce a _utils.py with load_settings() and the shared proper-noun regex; (2) move ControllerConfig into controller.py; (3) split planner.py into _planner_classifier.py + _planner_extractor.py + _planner_generator.py with a planner.py façade; (4) place all tests in test_system/. The public API via __init__.py would remain unchanged. None of this restructuring is necessary for the current scope — it is guidance for a v2.0 refactor.

9.5 — Overall grade: 1–2 (Sehr gut)

Strengths: clean acyclic architecture, complete academic references, intentional fallback strategy, thorough module documentation, thesis-aligned design. Weaknesses: structural issues (ControllerConfig placement, settings duplication, co-located tests) and bypassed public API prevent a pure 1.

9.6 — Publication verdict: Conditionally Accepted

Three changes are required before submission: (1) move test_logic_layer.py out of the production package, (2) centralize _load_settings(), (3) move ControllerConfig to controller.py. One addition is strongly recommended: test_system/test_navigator_semantic.py to cover the 6 filter methods whose EM impact is directly cited in the thesis.

STEP 10 — CONSOLIDATED ACTION LIST
No	Priority	Category	Affected Files	Description	Est. Effort
1	CRITICAL	Duplication	planner.py, verifier.py, controller.py	_load_settings() duplicated verbatim in 3 files — extract to src/logic_layer/_settings.py and replace all three with from ._settings import _load_settings	30 min
2	IMPORTANT	Architecture	navigator.py, controller.py, __init__.py	ControllerConfig lives in navigator.py but belongs in controller.py — move the class, update the import in navigator.py to from .controller import ControllerConfig, update __init__.py import block	45 min
3	IMPORTANT	Publication	src/logic_layer/test_logic_layer.py, test_system/	Test file co-located in production package — move to test_system/test_logic_layer.py; create/extend test_system/conftest.py if needed (same pattern as data layer migration)	20 min
4	IMPORTANT	API Surface	__init__.py	Remove internal component classes from __all__: QueryClassifier, EntityExtractor, PlanGenerator; add missing ConfidenceLevel	10 min
5	IMPORTANT	API Stability	diagnose.py, diagnose_verbose.py	All top-level scripts bypass __init__.py — migrate to stable package-level imports (from src.logic_layer import AgenticController)	30 min
6	RECOMMENDED	Test Coverage	test_system/	Navigator has no semantic test file — create test_system/test_navigator_semantic.py covering the 6 filter methods with reproducible input/output fixtures	2 h
7	RECOMMENDED	Duplication	navigator.py, controller.py, verifier.py, planner.py	Proper-noun regex [A-Z][a-z]+(?:\s+[A-Z][a-z]+)+ appears 4× with one silent variant in navigator.py — define as module constant in _settings.py or _utils.py, import where needed	20 min
8	RECOMMENDED	Convention	planner.py, verifier.py, controller.py, navigator.py	from_yaml() methods use NumPy-style docstrings; all other methods use Google-style — standardize to Google-style (Args: / Returns:) across all four files	45 min
9	RECOMMENDED	Publication	__init__.py	Version string __version__ = '2.0.0' is inconsistent with data layer's 4.0.0 — align with a project-wide semantic version or document the independent versioning policy	5 min
10	RECOMMENDED	Publication	navigator.py	Five original contributions reference thesis Table 4.2 / thesis Table 4.3 with provisional numbers — confirm or replace with final numbers before submission	15 min