# ARCHITECTURAL COHESION ANALYSIS: DIRECTORY-LEVEL REVIEW

**DIRECTORY TO ANALYZE:** {{DIRECTORY_PATH}}  
**ARTIFACT DESIGNATION:** {{ARTIFACT_NAME}} (e.g., "Artifact A — Data Layer")  
**FILES IN THIS DIRECTORY:** {{LIST_ALL_FILES_WITH_LINE_COUNTS}}

---

## PREREQUISITE
Individual file reviews (Pass 1 + Pass 2) have already been conducted for each file in this directory. The results of those reviews are provided below as context. This directory-level review does **NOT** repeat file-level analysis. Instead, it evaluates the directory as an architectural unit.

**INDIVIDUAL FILE REVIEW RESULTS:** {{PASTE_ALL_FILE_REVIEW_RESULTS_HERE}}

---

## ROLE AND CONTEXT
You act as a **senior professor of Computer Science and Software Engineering**. You have already reviewed each file in this directory individually. You now step back and evaluate the directory as a whole: Does it fulfill its architectural mandate? Do the files form a coherent, well-bounded module?

**Project context:** "Edge-RAG" — master thesis implementation for Hybrid Retrieval-Augmented Generation on edge devices (< 16 GB RAM). All models local via Ollama, all databases embedded (LanceDB, KuzuDB). Configuration centralized in `config/settings.yaml`. Codebase intended for publication as an artifact accompanying an academic paper.

---

## TASK
Conduct a directory-level architectural review. The unit of analysis is the directory as a whole, not individual files. Work through every step below.

### STEP 0 — DIRECTORY PROFILE
Write a concise summary (5-8 sentences) covering:
* What is the architectural mandate of this directory within the system?
* What abstraction does it provide to the rest of the codebase?
* What are the public interfaces (classes, functions) that other layers consume?
* What are the internal implementation details that should not leak outside?
* How many lines of code, how many files, what is the ratio of production code to test code?

### STEP 1 — INTERFACE COHERENCE
Examine the `__init__.py` exports and the public API surface of this directory.
1.1 What does `__init__.py` export? Is the export list intentional and minimal?
1.2 For each public class/function: List the external consumers.
1.3 Are there public symbols that are never imported from outside? (API bloat)
1.4 Is the API surface stable? (e.g., `from src.layer import X` vs. reaching into specific files).
1.5 Are return types and parameter types consistent across the public API?

### STEP 2 — INTERNAL DEPENDENCIES AND DATA FLOW
Map the internal dependency graph of the files within this directory.
2.1 Draw/Describe the dependency graph: Which file imports from which other file?
2.2 Are there circular dependencies between files?
2.3 What is the data flow through the directory during standard operations (e.g., ingestion)?
2.4 Are there isolated files that don't belong here?
2.5 Is there a clear layering (utilities vs. orchestration) or are dependencies tangled?

### STEP 3 — REDUNDANCY AND DUPLICATION
3.1 Are there functions across different files that implement similar logic?
3.2 Are there redundant data structures (dataclasses, TypedDicts) representing the same concept?
3.3 Are configuration lookups (`settings.yaml`) duplicated instead of centralized?
3.4 Are there shared constants/magic numbers that should be defined once?
3.5 Is error handling logic (retry/fallback) duplicated?

### STEP 4 — SEPARATION OF CONCERNS
4.1 State each file's responsibility in one sentence (avoid using the word "and").
4.2 Are there files too large (> 800 lines)?
4.3 Are there files too small (premature abstraction)?
4.4 Does the directory mix abstraction levels (low-level DB vs. high-level retrieval)?
4.5 Is test code co-located or properly separated for publication?

### STEP 5 — ERROR HANDLING CONSISTENCY
5.1 Is there a consistent error handling strategy across all files?
5.2 Are custom exception types used consistently?
5.3 Is error propagation correct at file boundaries?
5.4 Aggregate fallback findings: Is there a systemic pattern of error suppression?

### STEP 6 — CONFIGURATION CONSISTENCY
6.1 Aggregate hardcoded value findings: Is there a systemic pattern?
6.2 Do all files read configuration the same way?
6.3 Is there a single point of configuration loading?
6.4 Are there conflicting defaults (e.g., different `batch_size`) across files?

### STEP 7 — NAMING AND CONVENTION CONSISTENCY
7.1 Are naming conventions (snake_case, PascalCase, etc.) consistent?
7.2 Is terminology consistent? (e.g., "chunk" vs. "segment")
7.3 Are docstring formats (Google/NumPy style) consistent?
7.4 Is the logging format and naming consistent?

### STEP 8 — PUBLICATION READINESS (DIRECTORY LEVEL)
8.1 Is the purpose clear from reading only `__init__.py` and module docstrings?
8.2 Is there a README explaining the internal architecture?
8.3 Are academic references consistent and complete across files?
8.4 Are there any non-publishable files (debug/scratch files)?
8.5 Is test coverage adequate for the critical paths?
8.6 Is the functionality reproducible from documentation alone?

### STEP 9 — ARCHITECTURAL VERDICT
9.1 **Cohesion score (1-10):** How well do these files form a unified module?
9.2 **Coupling assessment:** How tightly is this directory tied to other parts?
9.3 Does it fulfill its mandate from the technical architecture docs?
9.4 **Restructuring advice:** What would you change from scratch?
9.5 **Overall grade:** (German academic scale 1-6) with justification.
9.6 **Publication verdict:** Accepted or Rejected? What must change?

### STEP 10 — CONSOLIDATED ACTION LIST
Produce a final table:

| No | Priority | Category | Affected Files | Description | Est. Effort |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | CRITICAL | Architectural Cohesion | ... | ... | ... |

---

## GENERAL REQUIREMENTS
* Evaluate the directory as a unit. Do not repeat file-level findings unless they have cross-file implications.
* Reference specific files and functions.
* Recommendations must be actionable.
* The final action list supersedes individual file lists for cross-cutting issues.