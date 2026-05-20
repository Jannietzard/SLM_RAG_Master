# Reproduction Guide

**Thesis:** Enhancing Reasoning Fidelity in Quantized Small Language Models on Edge Devices via Hybrid Retrieval-Augmented Generation
**Author:** Jan Nietzard | FOM Hochschule
**System version:** 5.0 | **Python:** 3.12.3 | **OS tested:** Windows 11

This document is the reproducibility contract. Following it on a clean
machine recreates the environment and the inputs that produced the thesis
numbers.

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
> pinned versions, captured 2026-05-18 from the evaluation virtualenv.
> `requirements.txt` (relaxed ranges) is for development only. Installing
> from `requirements.txt` may resolve different versions and is **not**
> guaranteed to reproduce the reported numbers.

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

Before ingesting, verify your copies match the manifest used for the
thesis numbers:

```bash
python -X utf8 verify_artifacts.py verify
```

This checks every artifact against `data/SHA256.txt` (committed to the
repository). Exit code 0 = all match; exit code 1 = a mismatch, meaning
your inputs differ from the thesis inputs and downstream numbers will not
be comparable.

If you regenerate the artifacts (re-chunk or re-extract), refresh the
manifest:

```bash
python -X utf8 verify_artifacts.py generate
```

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

```bash
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
> which is the paper-release default. An empirical probe showed nomic-embed-text
> produces 90–94 % merge rates at every tested threshold (see §3.6.1 of
> TECHNICAL_ARCHITECTURE.md); the linker is therefore disabled and alias
> resolution reduces to `canonical_form` exact-match deduplication.

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

### Headline benchmark (HotpotQA)

```bash
python -X utf8 -m src.thesis_evaluations.benchmark_datasets evaluate `
    --dataset hotpotqa --samples 500
```

Writes a per-question JSONL to `evaluation_results/hotpotqa_<model>_<ts>.jsonl`
and prints a summary including EM / F1 / SF-F1 / SF-Recall, the
pipeline-vs-LLM failure decomposition, and embedding-cache metrics.

### Tier-1 ablation suite

```bash
# Component ablation: LLM-only -> +Retrieval -> +Planner -> +Verifier -> +SelfCorrect
python -X utf8 -m src.thesis_evaluations.agentic_ablation `
    --dataset hotpotqa --samples 200 --model qwen2:1.5b

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

```bash
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

---

## Step 6 — Verify the installation

The test suite runs without Ollama (LLM-dependent tests are marked and
deselected):

```bash
python -X utf8 -m pytest test_system/ -q
```

All tests pass.

---

## Reproducibility checklist (before submission)

- [ ] `pip install -r requirements_frozen.txt` succeeds on a clean venv.
- [ ] `python verify_artifacts.py verify` exits 0.
- [ ] Phase-3 ingest completes; `diagnose_graph_baseline.py` reports the
      invariants within threshold.
- [ ] The headline benchmark + Tier-1 ablations have been run.
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
| `database disk image is malformed` (repeated) | Corrupted embedding cache | `del cache\*.db`, re-run. |
| Unicode errors on Windows | Missing UTF-8 flag | Always run with `python -X utf8`. |
| GLiNER first-run slow | One-time model download (~250 MB) | Wait; cached afterwards. |
| `verify_artifacts.py verify` reports MISMATCH | Local inputs differ from the manifest | Obtain the manifest-matching artifacts, or regenerate and re-run all evals. |
