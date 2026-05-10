You are a senior software architect and academic code reviewer. I am submitting a Master's thesis
at RWTH Aachen titled "Enhancing Reasoning Fidelity in Quantized Small Language Models on
Edge Devices via Hybrid Retrieval-Augmented Generation."

The system is ~15,000 lines of Python across ~27 files, organised into three artifact layers:
- Artifact A (Data Layer): embeddings, chunking, entity extraction, storage, hybrid retrieval
- Artifact B (Logic Layer): Planner (S_P), Navigator (S_N), Verifier (S_V) agents
- Artifact C (Evaluation): ablation study, benchmark runners, diagnostics

Perform a COMPLETE publication-readiness audit. Work through every section below systematically.
For each finding, state: the FILE(s) affected, the SEVERITY (critical / major / minor / cosmetic),
the specific CODE or PATTERN you identified, and a concrete FIX with code.

═══════════════════════════════════════════════════════════════════════════════
SECTION 1 — STRUCTURAL INTEGRITY & SEPARATION OF CONCERNS
═══════════════════════════════════════════════════════════════════════════════

Analyse each file and answer:

1.1 LAYER BOUNDARY VIOLATIONS
    - Does any file in data_layer/ import from logic_layer/ or pipeline/?
    - Does any file in logic_layer/ import from pipeline/ or evaluations/?
    - Does pipeline/ reach into internal (non-__init__.py-exported) symbols of other layers?
    - Are there circular imports? Trace the full import graph.
    → The dependency direction MUST be: evaluations → pipeline → logic_layer → data_layer
    → Any reverse arrow is a critical violation.

1.2 SINGLE RESPONSIBILITY
    - Does any single file handle more than one architectural concern?
      (e.g., a file that does both entity extraction AND storage operations)
    - Are there God-classes with >500 lines or >10 public methods?
    - Does any class mix I/O (file/network) with pure computation?

1.3 MODULE INTERFACE CLARITY
    - Does each package __init__.py export a clean, minimal public API?
    - Are internal helpers prefixed with _ or kept out of __init__.py?
    - Can each layer be instantiated and tested with only its own __init__.py imports?

═══════════════════════════════════════════════════════════════════════════════
SECTION 2 — CODE DUPLICATION & OVERLAP ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

2.1 EXACT DUPLICATES
    - Find any functions or methods that appear in more than one file with identical
      or near-identical logic (>80% token overlap).
    - Pay special attention to:
      • Jaccard similarity computation (navigator.py vs elsewhere)
      • Embedding normalisation logic
      • Entity extraction calls from multiple entry points
      • Configuration loading / YAML parsing repeated in multiple files
      • Logging setup boilerplate

2.2 SEMANTIC DUPLICATES (same purpose, different implementation)
    - Are there two different ways to compute text similarity?
    - Are there two different ways to load/parse the config?
    - Are there two different ways to call the Ollama API?
    - Are there two different chunking invocations with hardcoded parameters
      that should share a single config-driven path?

2.3 COPY-PASTE PATTERNS
    - Find any try/except blocks that are copy-pasted with minor variations.
    - Find any dataclass definitions that duplicate fields from another dataclass.
    - Find any test fixtures that duplicate setup logic across test files.

For each duplicate found:
→ State which file should OWN the canonical implementation.
→ State which files should import from it.
→ Provide the refactored import statement.

═══════════════════════════════════════════════════════════════════════════════
SECTION 3 — CONFIGURATION COHERENCE
═══════════════════════════════════════════════════════════════════════════════

3.1 SINGLE SOURCE OF TRUTH
    - Is config/settings.yaml truly the ONLY place where parameters are defined?
    - Search for ALL hardcoded numeric values in source files:
      • similarity thresholds, top_k values, batch sizes, timeout values
      • model names ("nomic-embed-text", "phi3")
      • API URLs ("http://localhost:11434")
    - Each hardcoded value is a potential configuration drift bug.

3.2 CONFIG VALIDATION
    - Is settings.yaml validated against a schema (Pydantic model or JSON Schema)?
    - What happens if a required key is missing? Does the system crash with a
      clear error or silently use a wrong default?

3.3 ENVIRONMENT SENSITIVITY
    - Can someone reproduce results on a different machine without editing source code?
    - Are file paths absolute or relative? Are they configurable?

═══════════════════════════════════════════════════════════════════════════════
SECTION 4 — TEST AUDIT (COVERAGE, QUALITY, GAPS)
═══════════════════════════════════════════════════════════════════════════════

4.1 QUANTITATIVE COVERAGE
    Count and report:
    - Total number of test functions across all test files.
    - Number of public functions/methods in production code.
    - Estimated line coverage (identify untested files/classes).
    - Tests per layer: data_layer, logic_layer, pipeline, evaluations.

4.2 TEST QUALITY CLASSIFICATION
    Classify every existing test into one of:
    a) STRUCTURAL — tests that a class exists or has attributes (low value)
    b) BEHAVIOURAL — tests that verify correct output for given input (high value)
    c) INTEGRATION — tests that cross layer boundaries (highest value)
    d) REGRESSION — tests tied to a specific thesis claim or bug fix
    e) NEGATIVE — tests that verify correct error handling

    Report the ratio. A healthy research codebase needs:
    - ≥60% behavioural tests
    - ≥15% integration tests
    - ≥10% negative/edge-case tests
    - <15% structural-only tests

4.3 MISSING TESTS — THE REQUIRED TEST MATRIX
    Check that the following tests EXIST. For each missing one, write the test:

    DATA LAYER (Artifact A):
    □ Embedding dimension consistency (embed_query vs embed_documents same dim)
    □ Cache hit returns identical vector to cache miss (bit-exact)
    □ Chunking: empty string → no crash, returns empty list
    □ Chunking: single sentence → returns one chunk
    □ Chunking: overlap correctness (chunk N's last sentence == chunk N+1's first sentence)
    □ Entity extraction: text with no entities → empty list, no crash
    □ Entity extraction: known entity → correct type and confidence > threshold
    □ Vector store: add → search returns the added document
    □ Vector store: search with threshold filters low-relevance results
    □ Graph store: add node → node exists, add edge → edge traversable
    □ Graph store: multi-hop traversal returns correct path
    □ RRF fusion: known scores → mathematically correct fused score
    □ RRF fusion: single-source input → score equals single-source score
    □ HybridRetriever: vector-only mode ignores graph results
    □ HybridRetriever: graph-only mode ignores vector results
    □ Ingestion pipeline: full document → all stores populated correctly

    LOGIC LAYER (Artifact B):
    □ Planner: each QueryType triggers correct RetrievalStrategy
    □ Planner: multi-hop query → hop_sequence has >1 step
    □ Planner: comparison query → sub-queries contain individual entities
    □ Navigator: empty retrieval plan → graceful empty result
    □ Navigator: relevance filter removes below-threshold results
    □ Navigator: redundancy filter removes Jaccard > 0.8 duplicates
    □ Navigator: contradiction filter removes conflicting numeric chunks
    □ Verifier: all claims verified → HIGH confidence
    □ Verifier: <50% claims verified → LOW confidence
    □ Verifier: self-correction changes answer when first attempt is wrong
    □ Verifier: max_iterations respected (no infinite loop)

    PIPELINE LAYER:
    □ End-to-end: simple factual question → correct answer
    □ End-to-end: multi-hop question → answer references bridge entity
    □ Early exit: trivial question → skips Navigator and Verifier
    □ Cache: same question twice → second call returns cached result
    □ Cache: cache clear → next call recomputes
    □ Ablation: planner disabled → system still returns an answer
    □ Ablation: verifier disabled → system still returns an answer
    □ Timing: all timing fields in PipelineResult are > 0

    CROSS-LAYER INTEGRATION:
    □ Ingestion → Retrieval: ingest 5 docs, retrieve by content → finds them
    □ Planner → Navigator → Verifier: mock-free full chain with small test corpus
    □ Config change → Behaviour change: changing top_k in config changes result count
    □ Error propagation: Ollama down → clear error message, no silent failure

    THESIS COMPLIANCE:
    □ 3-sentence chunking window (SentenceChunkingConfig.sentences_per_chunk == 3)
    □ GLiNER confidence threshold matches thesis (0.15 for NER, 0.5 for RE)
    □ RRF k-parameter matches thesis formula
    □ Confidence levels: exactly 3 levels (HIGH ≥ 0.8, MEDIUM ≥ 0.5, LOW < 0.5)
    □ Self-correction loop is INSIDE Verifier, NOT in AgentPipeline
    □ No outer retry loop in AgentPipeline.process()

4.4 TEST ISOLATION
    - Do any tests depend on external services (Ollama, network)?
      → These must be mockable for CI.
    - Do any tests leave side effects (files on disk, database entries)?
      → These must use tmp directories with cleanup.
    - Do tests run in a deterministic order, or do they have hidden dependencies?

═══════════════════════════════════════════════════════════════════════════════
SECTION 5 — REPRODUCIBILITY CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

5.1 ONE-COMMAND REPRODUCTION
    Can a reviewer do this and reproduce thesis results?
If not, document every missing step.

5.2 RANDOM SEED CONTROL
    - Is there a global random seed set for numpy, random, torch (if used)?
    - Does the LLM call use temperature=0 for deterministic output?
    - Are results reproducible across runs? If not, what is the variance?

5.3 DEPENDENCY PINNING
    - Are ALL dependencies in requirements.txt pinned to exact versions (==)?
    - Using >= ranges in research code is a reproducibility risk.

5.4 DATA VERSIONING
    - Are benchmark datasets versioned? (HotpotQA version, split, date downloaded)
    - Are evaluation results stored with the config that produced them?

═══════════════════════════════════════════════════════════════════════════════
SECTION 6 — ERROR HANDLING & ROBUSTNESS
═══════════════════════════════════════════════════════════════════════════════

6.1 FAILURE MODE ANALYSIS
    What happens when:
    - Ollama is not running? → Expected: clear error, not a traceback
    - LanceDB directory is missing? → Expected: auto-create or clear error
    - KuzuDB is corrupted? → Expected: fallback to vector-only
    - Network timeout on embedding call? → Expected: retry with backoff
    - Empty query string? → Expected: no crash
    - Query in non-English language? → Expected: graceful degradation

6.2 SILENT FAILURES
    - Are there bare `except: pass` or `except Exception: pass` blocks?
    - Are there logging.debug() calls that should be logging.warning()?
    - Does any function return None where it should raise?

6.3 RESOURCE CLEANUP
    - Are database connections properly closed?
    - Are temp files cleaned up?
    - Is there a graceful shutdown path?

═══════════════════════════════════════════════════════════════════════════════
SECTION 7 — DOCUMENTATION-TO-CODE ALIGNMENT
═══════════════════════════════════════════════════════════════════════════════

7.1 ARCHITECTURE DOC vs REALITY
    - Does TECHNICAL_ARCHITECTURE.md match the current code?
    - Are there classes/methods described in the doc that don't exist in code?
    - Are there classes/methods in code that aren't documented?
    - Do the dataclass field names in the doc match the actual field names?

7.2 DOCSTRINGS
    - Does every public class and public method have a docstring?
    - Do docstrings match the actual parameters and return types?
    - Are there TODO/FIXME/HACK comments that need resolution before submission?

7.3 TYPE HINTS
    - Are all public function signatures fully type-hinted?
    - Do type hints match actual runtime types? (e.g., Optional[] where None is possible)

═══════════════════════════════════════════════════════════════════════════════
SECTION 8 — CODE QUALITY & ACADEMIC STANDARDS
═══════════════════════════════════════════════════════════════════════════════

8.1 NAMING CONVENTIONS
    - Are all names self-explanatory? No single-letter variables outside loops.
    - Are German/English mixed in names? Pick one and be consistent.
    - Do class names use PascalCase, functions snake_case, constants UPPER_CASE?

8.2 DEAD CODE
    - Are there commented-out blocks of code?
    - Are there functions that are never called from anywhere?
    - Are there imports that are unused?
    - Are there entire files that serve no purpose?

8.3 COMPLEXITY
    - Are there functions longer than 50 lines? They should be decomposed.
    - Are there functions with cyclomatic complexity > 10? Simplify.
    - Are there deeply nested if/else chains (>3 levels)?

8.4 SECURITY (for publication)
    - Are there hardcoded API keys, tokens, or passwords?
    - Are there file paths that reveal personal directory structures?
    - Is .gitignore comprehensive? (No cache files, no data files, no credentials)

═══════════════════════════════════════════════════════════════════════════════
SECTION 9 — SYSTEM COHERENCE (THE "WORKS AS ONE" CHECK)
═══════════════════════════════════════════════════════════════════════════════

9.1 DATA FLOW INTEGRITY
    Trace the COMPLETE data flow for a multi-hop query and verify:
    - Query string enters AgentPipeline.process()
    - Planner produces RetrievalPlan → verify all fields populated
    - Navigator receives plan → verify it uses the plan's strategy, not hardcoded
    - Navigator calls HybridRetriever → verify retriever respects config thresholds
    - HybridRetriever calls both vector_search AND graph_search → verify RRF fusion
    - Navigator filters results → verify filters match thesis description
    - Verifier receives context → verify context format matches what LLM expects
    - Verifier calls Ollama → verify prompt template matches thesis Section 3.4
    - Verifier returns VerificationResult → verify confidence calculation is correct
    - PipelineResult is assembled → verify all timing fields filled

9.2 ABLATION SWITCH INTEGRITY
    For each ablation configuration in the thesis:
    - enable_planner=False → verify system works (uses default plan)
    - enable_verifier=False → verify system works (skips verification)
    - strategy=VECTOR_ONLY → verify graph is never queried
    - strategy=GRAPH_ONLY → verify vector store is never queried
    - strategy=HYBRID → verify both stores are queried

9.3 METRIC CONSISTENCY
    - Are the same metrics (EM, F1, latency) computed identically across
      all evaluation scripts?
    - Is there ONE canonical implementation of each metric?

═══════════════════════════════════════════════════════════════════════════════
SECTION 10 — FINAL OUTPUT
═══════════════════════════════════════════════════════════════════════════════

Produce a structured report with:

1. EXECUTIVE SUMMARY
   - Overall readiness score: RED / YELLOW / GREEN
   - Number of critical / major / minor / cosmetic issues

2. PRIORITISED FIX LIST
   - Ordered by: critical first, then by effort (quick wins first)
   - Each item: file, line, issue, fix (with code)

3. MISSING TESTS
   - Complete list of tests from Section 4.3 that do not exist
   - For each: write the complete pytest function

4. REFACTORING PLAN
   - Which duplicates to consolidate and where
   - Which hardcoded values to move to config
   - Which files to split or merge

5. PRE-SUBMISSION CHECKLIST
   □ All tests pass: pytest -v --tb=short
   □ No hardcoded paths or credentials
   □ requirements.txt has pinned versions
   □ README has reproduction instructions
   □ .gitignore excludes all generated data
   □ No TODO/FIXME in submitted code
   □ All docstrings present and accurate
   □ Architecture doc matches code
   □ Evaluation results are reproducible
   □ Code runs on a clean machine with only requirements.txt