# Reproduction Guide

**Thesis:** Enhancing Reasoning Fidelity in Quantized Small Language Models on Edge Devices via Hybrid Retrieval-Augmented Generation
**Author:** Jan Nietzard
**System version:** 5.4 (2026-05-25) | **Python:** 3.12.3 | **OS tested:** Windows 11

This document is the reproducibility contract. Following it on a clean
machine recreates the environment and the inputs that produced the thesis
numbers.

> **Artifact-integrity claim — no hidden constants.** Every parameter
> that affects evaluation results lives in `config/settings.yaml`. No
> tuning constant is hardcoded in a production code path: where a value
> is implementation detail (a regex, a closed-class linguistic list with
> a citation), it is documented in `TECHNICAL_ARCHITECTURE.md §11`. The
> startup validator `_settings._validate_settings()` checks 35 required
> keys (`_REQUIRED_SETTINGS`) and emits a WARNING if any is missing —
> silent dataclass-default fallback is treated as a reproducibility risk
> and was the failure mode the 2026-05-24 audit caught
> (`vector_store.top_k_vectors` was silently 10 instead of the documented
> 20). Reproducing the thesis numbers therefore requires only
> `config/settings.yaml` + `requirements_frozen.txt` + the SHA-verified
> input artifacts; no patching of source is necessary or expected.
>
> **Random seed for the headline run.** The 500-question headline
> evaluation uses **`--range 0-500`** — a *deterministic* slice (first
> 500 questions of `data/hotpotqa/questions.json` in stored order), not
> a random sample. There is no seed to set; the slice is byte-stable for
> any user who passes the SHA-256 verification in Step 3. If a *random*
> sample is required (e.g. for an ablation that re-runs a subset under
> a different config), use **`--samples 500 --seed 42`** — the seed is
> auto-logged by the evaluator and the same `--seed` value reproduces
> the same question set exactly.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12.3 | Other 3.12.x likely fine; earlier minors untested. |
| Ollama | ≥ 0.4 | <https://ollama.com> — local LLM + embedding server. |
| RAM | ≥ 8 GB | 16 GB recommended; the system targets a < 16 GB budget. |
| Disk | ≥ 10 GB | Databases + HotpotQA corpus + caches. |

---

## Step 1 — Install Python dependencies

```bash
# Exact pinned versions used during the thesis evaluation.
pip install -r requirements_frozen.txt

# spaCy English model (required for chunking, NER, dependency parsing).
python -m spacy download en_core_web_sm
```

> `requirements_frozen.txt` is the **reproducibility contract** — exact
> pinned versions. `requirements.txt` (relaxed ranges) is for development
> only. Installing from `requirements.txt` may resolve different versions
> and is **not** guaranteed to reproduce the reported numbers.

---

## Step 2 — Start Ollama and pull models

```bash
# Terminal 1 — start the server, keep it open.
ollama serve

# Terminal 2 — pull the models.
ollama pull qwen2:1.5b        # primary SLM for answer generation (S_V)
ollama pull nomic-embed-text  # embedding model (Artifact A)

ollama list                   # verify both are present
```

The system expects Ollama at `http://localhost:11434` (configurable via
`config/settings.yaml → llm.base_url` and `embeddings.base_url`).

---

## Step 3 — Verify the input artifacts

The pipeline depends on three per-dataset input artifacts:

```
data/hotpotqa/chunks_export.json            (Phase-1 chunking output)
data/hotpotqa/graph/extraction_results.json (Phase-2 GLiNER + REBEL output)
data/hotpotqa/questions.json                (benchmark questions)
```

A SHA-256 manifest of the thesis inputs is committed at `data/SHA256.txt`.
Verify your local copies match it before ingesting (any mismatch means
your inputs differ from the thesis inputs and downstream numbers will not
be comparable):

```powershell
# Windows PowerShell — recompute and diff against the manifest.
Get-Content data/SHA256.txt | ForEach-Object {
  $expected, $path = $_ -split '\s+', 2
  $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
  if ($actual -ne $expected.ToLower()) { Write-Host "MISMATCH $path" }
}
```

```bash
# Unix — equivalent one-liner.
sha256sum -c data/SHA256.txt
```

If you regenerate the artifacts (re-chunk or re-extract), refresh the
manifest by re-hashing every line entry and committing the new
`data/SHA256.txt`.

---

## Step 4 — Ingest the corpus (decoupled three-phase architecture)

Ingestion is split into three phases so the GPU-bound extraction runs
separately from the CPU-only edge target.

| Phase | Command / tool | Hardware |
|---|---|---|
| 1 | `python -m src.thesis_evaluations.benchmark_datasets ingest --chunks-only` | CPU |
| 2 | `colab_extraction.py` (run in Google Colab on a GPU) | GPU |
| 3 | `python local_importingestion.py …` | CPU |

For reproduction, **Phases 1 and 2 are already done** — their outputs are
the artifacts verified in Step 3. You only run **Phase 3**:

```powershell
python -X utf8 local_importingestion.py `
    --chunks data/hotpotqa/chunks_export.json `
    --extractions data/hotpotqa/graph/extraction_results.json `
    --dataset hotpotqa `
    --clear `
    --no-entity-linking `
    --hub-threshold-ratio 0.03 `
    --cooccurrence-min-confidence 0.5
```

> **Note:** `--no-entity-linking` disables embedding-based alias resolution,
> which is the paper-release default. An empirical probe showed
> nomic-embed-text produces 90–94 % merge rates at every tested threshold
> (see §3.6.1 of TECHNICAL_ARCHITECTURE.md); the linker is therefore
> disabled and alias resolution reduces to `canonical_form` exact-match
> deduplication.

This populates `data/hotpotqa/vector/` (LanceDB) and
`data/hotpotqa/graph/` (KuzuDB). Phase 3 takes ~20–40 min on CPU; the
embedding step is accelerated by `cache/hotpotqa_embeddings.db` if a
healthy cache is present.

> **If the embedding cache is corrupted** ("database disk image is
> malformed" repeated on every batch), delete it and re-run — SQLite
> recreates a fresh one and the embeddings are recomputed identically:
> `del cache\embeddings.db cache\hotpotqa_embeddings.db`

After ingestion, confirm graph health:

```bash
python -X utf8 diagnose_graph_baseline.py --dataset hotpotqa
```

Expected: `isolated_rate < 5%`, `duplicate_rate < 2%`,
`relations_per_chunk ≥ 5`.

---

## Step 5 — Run the evaluations

All evaluation entry points are under `src/thesis_evaluations/`.

### Headline benchmark (HotpotQA, 500 questions)

```powershell
python -X utf8 -m src.thesis_evaluations.benchmark_datasets evaluate `
    --dataset hotpotqa --range 0-500 --retrieval-only        # fast: SF metrics only
python -X utf8 -m src.thesis_evaluations.benchmark_datasets evaluate `
    --dataset hotpotqa --range 0-500                          # full: + Soft-EM / EM / F1
```

Writes a per-question JSONL to `evaluation_results/hotpotqa_<model>_<ts>.jsonl`
and prints a summary. The summary now reports four complementary
correctness verdicts (added 2026-05-22):

- **`Exact Match`** — strict EM (HotpotQA-normalised string equality).
- **`Answer F1`** — token-overlap F1.
- **`Soft-EM (F1 ≥ θ)`** — token-F1 ≥ `benchmark.answer_f1_threshold`
  (default `0.6`, configurable via settings). Headline answer-correctness
  metric — strict EM systematically under-counts answers like
  `"Teach the Controversy"` vs gold `'"Teach the Controversy" campaign'`
  (F1=0.8, Soft-EM=True). Both EM and Soft-EM are reported.
- **`SoftEM | all-gold-retrieved`** — Soft-EM conditional on the
  Navigator having delivered all gold paragraphs. Isolates the SLM
  ceiling from pipeline retrieval quality.

The JSONL also records two **delivery-loss** instrumentation fields
(added 2026-05-23) that separate retrieval-stage from delivery-stage gold
loss:

- **`all_gold_retrieved`** — gold present after the Navigator's filter
  chain (≤ `max_context_chunks=8`).
- **`gold_in_final_context`** — gold present after the Verifier's
  `max_docs=5` cap (the LLM-visible window). The gap between the two is
  "delivery loss" — gold retrieved but cut before the model saw it.

### Tier-1 ablation suite

```powershell
# Component ablation: LLM-only -> +Retrieval -> +Planner -> +Verifier -> +SelfCorrect
python -X utf8 -m src.thesis_evaluations.agentic_ablation `
    --dataset hotpotqa --samples 200 --model qwen2:1.5b

# Retrieval-mode ablation (vector / graph / hybrid) — quantifies the
# graph's marginal contribution to recall (the +33pp super-additivity
# result for hybrid is the headline justification for the architecture).
python -X utf8 -m src.thesis_evaluations.benchmark_datasets ablation `
    --dataset hotpotqa --samples 100 --retrieval-only

# Cross-model quantization sweep
python -X utf8 -m src.thesis_evaluations.quantization_sweep `
    --dataset hotpotqa --samples 100 `
    --models "qwen2:1.5b,qwen2.5:3b,llama3.2:3b,phi3"

# Latency / peak-memory profile
python -X utf8 -m src.thesis_evaluations.latency_memory_profile `
    --dataset hotpotqa --samples 50 --budget-seconds 60

# Chunking-hyperparameter ablation (retrieval-only)
python -X utf8 -m src.thesis_evaluations.chunking_ablation `
    --dataset hotpotqa --samples 100 --configs "3:1,5:1,7:1"
```

### Aggregate everything for the thesis manuscript

```bash
python -X utf8 -m src.thesis_evaluations.thesis_results_aggregator
```

Reads the latest Tier-1 output directories and writes a single bundle
under `evaluation_results/thesis_final_<ts>/`:

- `table_quantization.tex`, `table_ablation.tex`, `table_latency.tex`
- `table_ablation_significance.tex` — paired-bootstrap 95 % CIs and
  p-values on each ablation-component delta.
- `significance_report.md` — plain-text companion (EM / F1 / SF-F1).
- `figure_*.png` — Pareto front, ablation waterfall, stage breakdown.
- `coverage_report.md` — confirms every thesis claim has data.

### Statistical significance directly

The paired-bootstrap module can also be invoked standalone on any two
per-question JSONL files:

```powershell
python -X utf8 -m src.thesis_evaluations.bootstrap `
    evaluation_results/.../row4_verifier.jsonl `
    evaluation_results/.../row5_self_correct.jsonl `
    --metric EM
```

It prints the delta, its 95 % CI, the bootstrap p-value, and a
significance verdict.

### Single-query diagnostics

```bash
python -X utf8 diagnose.py --idx 0
python -X utf8 diagnose_verbose.py --idx 0           # full per-stage trace
python -X utf8 diagnose_verbose.py --idx 0 --skip-llm # skip the Ollama call
```

The verbose diagnostic uses the **same Soft-EM verdict** as the benchmark
(reads `benchmark.answer_f1_threshold` from settings), so per-trace
correctness labels and aggregate numbers stay consistent.

---

## Step 6 — Verify the installation

The test suite runs without Ollama (LLM-dependent tests are marked and
deselected). The 665-test guardrail covers the data layer, logic layer,
pipeline, and the agentic-controller stateless helpers:

```bash
python -X utf8 -m pytest test_system/ -q
```

Expected: `665 passed`, 11 skipped (graph-inspect + nightly markers).

---

## Reproducibility checklist (before submission)

- [ ] `pip install -r requirements_frozen.txt` succeeds on a clean venv.
- [ ] SHA-256 manifest verification (Step 3) reports no mismatches.
- [ ] Settings validator on startup emits **no** `_REQUIRED_SETTINGS`
      WARNING — every guarded key is present in `config/settings.yaml`.
- [ ] Phase-3 ingest completes; `diagnose_graph_baseline.py` reports the
      invariants within threshold.
- [ ] The headline benchmark + Tier-1 ablations have been run with a
      fixed random seed (use `--range 0-500` for deterministic slicing;
      `--samples N --seed X` for reproducible random sampling).
- [ ] `thesis_results_aggregator` reports `ALL CLAIMS COVERED`.
- [ ] `requirements_frozen.txt` and `data/SHA256.txt` are committed.

---

## Dataset provenance

**HotpotQA** — fullwiki dev set v1.1 (Yang et al. 2018, EMNLP).
The per-question artifacts in `data/hotpotqa/` (`chunks_export.json`,
`questions.json`) are derived from this set; their SHA-256 checksums are
recorded in `data/SHA256.txt`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: KuzuDB not available` | kuzu not installed | Install from `requirements_frozen.txt`. |
| `ConnectionError: localhost:11434` | Ollama not running | `ollama serve`. |
| `Model 'qwen2:1.5b' not found` | Model not pulled | `ollama pull qwen2:1.5b`. |
| `IO exception: Could not set lock on file ... graph_KuzuDB` | A previous pipeline still holds the KuzuDB exclusive lock | Stop the other process (or wait for it to release); ablation runs auto-close pipelines between configs. |
| `database disk image is malformed` (repeated) | Corrupted embedding cache | `del cache\*.db`, re-run. |
| Unicode errors on Windows | Missing UTF-8 flag | Always run with `python -X utf8`. |
| GLiNER first-run slow | One-time model download (~250 MB) | Wait; cached afterwards. |
| `_validate_settings: required key 'X.Y' absent` WARNING | Key missing from `config/settings.yaml`; system fell back to a dataclass default | Add the key to `settings.yaml`. Reproducibility is not guaranteed otherwise. |
| SHA-256 verification (Step 3) reports MISMATCH | Local inputs differ from the manifest | Obtain the manifest-matching artifacts, or regenerate and re-run all evals. |
