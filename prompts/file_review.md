================================================================================
FILE REVIEW — COMPLETE ANALYSIS WITH SELF-VERIFICATION
================================================================================

FILE TO ANALYZE: {{src/data_layer/hybrid_retriever.py}}
TODAY'S DATE:    {{TODAY}}
LAST REVIEWED:   {{"NEVER"}}
REVIEW TRIGGER:  {{"Initial review"}}
PREVIOUS FINDINGS (if any):
{{PASTE_PREVIOUS_ACTION_LIST — or "N/A" if first review}}

---

ROLE AND CONTEXT:

You act as a senior professor of Computer Science and Software Engineering
with research focus in distributed systems and information retrieval. You
have 20+ years of experience building production-grade systems and regularly
evaluate theses at doctoral and master level. You have published extensively
and have served as reviewer for major venues (ACL, EMNLP, NeurIPS, SIGIR).

You are reviewing the file specified above within the project "Edge-RAG" —
a master thesis implementation for Hybrid Retrieval-Augmented Generation on
resource-constrained edge devices. The system architecture is structured as
follows:

- Data Layer (Artifact A): Embeddings, Chunking, Entity Extraction,
  Storage, Hybrid Retrieval
- Logic Layer (Artifact B): Agentic Pipeline S_P -> S_N -> S_V
  (Planner, Navigator, Verifier)
- Pipeline Layer: Orchestration of Ingestion and Query Processing
- Configuration: config/settings.yaml as Single Source of Truth

Constraints: All models run locally via Ollama, all databases are embedded
(LanceDB, KuzuDB), target hardware < 16 GB RAM. No cloud access during
operation. The codebase is intended for publication as a companion artifact
to an academic paper on GitHub, reviewed by academic peers.

If a previous review exists (see header), verify which prior findings have
been addressed and which remain open.

---

TASK:

This prompt has two phases. Execute both sequentially in a single response.

PHASE 1 (Steps 0-10): Full systematic analysis of the file.
PHASE 2 (Steps 11-12): Self-verification and final consolidation.

Do not skip any step. If a step yields no findings, state "No findings".

---

================================================================================
PHASE 1 — ANALYSIS
================================================================================

STEP 0 — FILE PROFILE

Write an introductory paragraph (3-5 sentences) addressing:
- What role does this file fulfill within the overall system?
- What problem does it address and why is it necessary?
- In which artifact and architectural layer is it situated?
- Which modules depend on it or consume its interfaces?

---

STEP 1 — HARDCODED VALUES AND SETTINGS COMPLIANCE

Examine the entire file line by line for hardcoded values.

1.1 Every number, string, threshold, URL, and file path that appears
    directly in source code rather than from config/settings.yaml.

1.2 For each: genuine constant (mathematical, protocol-mandated) or
    configurable parameter belonging in settings.yaml?

1.3 Check default values in function signatures and dataclasses for
    consistency with settings.yaml entries.

1.4 Assess whether defaults serve as reasonable emergency fallbacks or
    mask missing configuration.

Output as table:
Line | Hardcoded Value | Classification (Constant/Configurable) | Recommendation

---

STEP 2 — FUNCTION INVENTORY

Complete inventory of all classes, methods, and functions. For each:

2.1 Name and signature (parameters, return type)
2.2 Brief description (1-2 sentences)
2.3 Call sites: Which modules invoke this? Name concrete files.
2.4 Usage status: Active / Unclear / Dead Code
2.5 If dead code: Recommend removal with justification.

---

STEP 3 — FALLBACK STRATEGIES

Identify all try/except blocks, default values, and fallback mechanisms.
For each:

3.1 What happens in the error case?
3.2 Is a message emitted via logger.warning or logger.error? Silent
    fallbacks are CRITICAL.
3.3 Designed for exceptions only, or can it trigger in regular operation?
3.4 Exception type sufficiently specific, or overly broad?
3.5 Exception re-raised or silently consumed?

Rating: CORRECT / IMPROVABLE / CRITICAL

---

STEP 4 — IMPORT ANALYSIS AND SIMPLIFICATION

4.1 List all imports; verify each is actually used.
4.2 Identify redundant imports.
4.3 Heavy dependencies that could be lazy imports?
4.4 Circular import risks?
4.5 General simplification opportunities?

---

STEP 5 — LOGIC REVIEW

Read the entire code top to bottom:

5.1 Correctness: Logical errors?
5.2 Edge cases: Empty inputs, None, empty strings, network errors, timeouts.
5.3 Thread safety (if relevant).
5.4 Type consistency: Return values match type hints?
5.5 Runtime complexity: Avoidable O(n^2)?
5.6 Memory: Large structures held unnecessarily? (Target: < 16 GB RAM)
5.7 Testability: Dependencies injectable?
5.8 Idempotency: Safe to call multiple times?

---

STEP 6 — SEMANTIC CORRECTNESS TEST SPECIFICATION

This step asks: "Does this file produce correct outputs that improve the
pipeline result?" — not just "does it run without errors".

For each primary public function or class:

6.1 CONTRIBUTION STATEMENT
    One sentence: What is the measurable difference between data before
    and after this function?

6.2 CONCRETE TEST INPUT
    A specific, realistic input from HotpotQA or equivalent. Real values,
    not placeholders.

6.3 EXPECTED OUTPUT
    What a correct output looks like — required content, forbidden content,
    structural and semantic properties.

6.4 FAILURE MODES
    3-5 ways the function could produce structurally correct but
    semantically wrong output. For each, the assertion that catches it.

6.5 STAGE DELTA VERIFICATION
    What information exists AFTER this stage that did not exist BEFORE?
    What breaks downstream if this stage is skipped?

6.6 EXECUTABLE TEST SKELETON
    A complete pytest test function (not pseudocode) that asserts semantic
    correctness using real data. Runnable without live LLM for S_P/S_N.

---

STEP 7 — PUBLICATION READINESS

7.1 Module-level documentation with purpose, architecture position,
    and scientific rationale?
7.2 Academic references at point of use? Format: Author(s) (Year).
    "Title." Venue/arXiv ID.
7.3 Reproducibility: Non-determinism fixed or configurable?
7.4 No debug artifacts (commented-out code, print(), TODO/FIXME/HACK)?
7.5 PEP 8 naming conventions, self-documenting names?
7.6 Full type annotations including return types?
7.7 Single responsibility, clean separation of concerns?
7.8 No absolute paths, API keys, credentials?
7.9 Inline comments explain WHY, not WHAT?
7.10 Implementation matches thesis methodology?

For each: PASS / NEEDS REVISION (with instructions) / FAIL (with fix)

---

STEP 8 — EXPERT ASSESSMENT

8.1 Code quality grade (German academic scale 1-6, with justification).
8.2 Architectural placement: Clean integration?
8.3 Concrete strengths.
8.4 Concrete weaknesses — directly stated.
8.5 Alternative approaches you would prefer and why.
8.6 Scientific rigor sufficient for thesis and publication?
8.7 Publication verdict: Accept as-is, or what must change?
8.8 Three most urgent changes in priority order.

---

STEP 9 — ACTION LIST (PRELIMINARY)

Produce a prioritized action list of all findings from Steps 1-8:

No | Priority (CRITICAL / IMPORTANT / RECOMMENDED) | Category | Description | Estimated Effort

Valid categories: Hardcoded Values, Dead Code, Fallback Strategy,
Performance, Logic Error, Architecture, Documentation, Settings
Compliance, Memory, Type Safety, Publication Readiness, Reproducibility,
Academic References, Semantic Correctness

---

STEP 10 — FILE HEADER UPDATE (PRELIMINARY)

The module-level docstring of every reviewed file must contain a review
metadata block. Produce the EXACT code change that inserts or updates
this block in the file's docstring:

    Review History:
        Last Reviewed: {{TODAY}}
        Review Result: [count from Step 9, e.g. "1 CRITICAL, 3 IMPORTANT, 2 RECOMMENDED"]
        Reviewer: Code Review Prompt v2.1
        Next Review: [condition, e.g. "After implementing action items"]

If a previous review block exists, preserve it:

        ---
        Previous Review: [old date]
        Previous Result: [old counts]
        Changes Since: [summary]

If the file has no module-level docstring at all, this is a CRITICAL
finding. Provide the complete docstring including module description
from Step 0 and the review metadata block.


================================================================================
PHASE 2 — SELF-VERIFICATION AND CONSOLIDATION
================================================================================

You have now completed the full analysis. Before finalizing, re-read the
file ONE MORE TIME from top to bottom and verify your own work. This phase
catches findings you missed and corrects any errors in Phase 1.

---

STEP 11 — SELF-VERIFICATION

Re-read the file. Compare every line against your Phase 1 findings.

11.1 MISSED FINDINGS: Identify anything Phase 1 missed:
     - Hardcoded values not in the Step 1 table
     - Functions not in the Step 2 inventory
     - try/except blocks not covered in Step 3
     - Unused imports not flagged in Step 4
     - Logic issues or edge cases not raised in Step 5
     - Semantic correctness gaps not in Step 6
     - Publication issues not in Step 7

     For each new finding, provide full detail as in the original step.
     If nothing was missed: "No additional findings. Phase 1 is complete."

11.2 FALSE POSITIVES: Are any Phase 1 findings incorrect?
     - Flagged as issues but actually correct?
     - Severity ratings too high or too low?
     - Recommendations impractical or contradictory?

     For each correction: state original finding, the error, corrected
     assessment.

11.3 CROSS-FILE CONSISTENCY:
     - Does this file's interface match what callers expect?
     - Latent contract violations that surface only at runtime?
     - Logic duplicated elsewhere in the codebase?

11.4 SEMANTIC TEST REVIEW:
     - Are Step 6 test inputs realistic and representative?
     - Are expected outputs specific enough to catch wrong-but-valid results?
     - Are failure modes comprehensive?
     - Is the test skeleton syntactically valid and runnable?

---

STEP 12 — FINAL CONSOLIDATED OUTPUT

This step produces the DEFINITIVE outputs that supersede all preliminary
outputs from Phase 1. It is the only section the user needs to act on.

12.1 FINAL ACTION LIST

     Merge Phase 1 findings (Step 9) with all corrections and additions
     from Step 11. Remove false positives, adjust severity ratings, add
     missed findings.

     No | Priority (CRITICAL / IMPORTANT / RECOMMENDED) | Category |
     Description | Estimated Effort

     This table is the single source of truth. It supersedes Step 9.

12.2 FINAL FILE HEADER UPDATE

     Update the review metadata block from Step 10 with corrected finding
     counts from 12.1. Provide the exact, ready-to-apply code change.

12.3 FINAL SEMANTIC TEST

     Provide the final, corrected version of the pytest test skeleton(s)
     from Step 6.6, incorporating any fixes from Step 11.4. This must be
     copy-paste-runnable.

12.4 REVIEW CLOSURE

     State explicitly:
     "Review of {{FILEPATH}} completed on {{TODAY}}.
      Findings: [X] CRITICAL, [Y] IMPORTANT, [Z] RECOMMENDED.
      File header update provided in 12.2.
      Semantic test provided in 12.3.
      No further passes required."


================================================================================
GENERAL REQUIREMENTS
================================================================================

- Thoroughness: Complete every step. No step may be silently omitted.
- Precision: Reference line numbers, function names, concrete values.
  No vague formulations.
- Constructiveness: Every finding must have a concrete improvement
  suggestion.
- Honesty: Name deficiencies without diplomatic softening.
- Edge context: Account for < 16 GB RAM constraint.
- Settings conformity: Configurable parameters must come from
  settings.yaml. Code defaults are emergency fallbacks only.
- Publication standard: File must be suitable for a public GitHub
  repository accompanying a peer-reviewed paper.
- Previous review: If prior findings exist in the header, state which
  are addressed and which remain open.
- File header: The review MUST produce a code change updating the
  file's docstring with review metadata (Step 12.2).
- Self-correction: Phase 2 exists because Phase 1 analysis is
  imperfect. Do not skip or abbreviate Phase 2. The final outputs
  in Step 12 are the only deliverables that matter.