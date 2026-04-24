# Reproduction Guide

**Thesis:** Enhancing Reasoning Fidelity in Quantized Small Language Models on Edge Devices via Hybrid Retrieval-Augmented Generation
**Author:** Jan Nietzard | FOM Hochschule
**System version:** 4.0.0 | **Python:** 3.11 | **OS tested:** Windows 11

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | Earlier versions untested |
| Ollama | ≥ 0.4 | [https://ollama.com](https://ollama.com) |
| RAM | ≥ 8 GB | 16 GB recommended for larger models |
| Disk | ≥ 10 GB | For databases and HotpotQA corpus |

---

## Step 1 — Install Python Dependencies

```bash
# Exact pinned versions used during thesis evaluation
pip install -r requirements_frozen.txt

# Install spaCy language model (required for chunking + NER)
python -m spacy download en_core_web_sm
```

> Use `requirements.txt` for development (relaxed ranges).
> Use `requirements_frozen.txt` for reproducible results.

---

## Step 2 — Start Ollama and Pull Models

```bash
# Start the Ollama server (keep this terminal open)
ollama serve

# In a second terminal — pull the two required models
ollama pull qwen2:1.5b       # LLM for answer generation (S_V)
ollama pull nomic-embed-text  # Embedding model (Artifact A)

# Verify both are available
ollama list
```

The system expects Ollama at `http://localhost:11434` (configurable via `config/settings.yaml → llm.base_url`).

---

## Step 3 — Configure the System

Review `config/settings.yaml`. The thesis evaluation used these key settings:

```yaml
llm:
  model_name: "qwen2:1.5b"
  temperature: 0.0       # fully deterministic
  timeout: 60

embeddings:
  model_name: "nomic-embed-text"

entity_extraction:
  gliner:
    confidence_threshold: 0.15   # recall-optimised for HotpotQA

ingestion:
  chunking_strategy: "sentence_spacy"
  sentences_per_chunk: 3
  sentence_overlap: 1
```

No source-code changes are needed to reproduce thesis results.

---

## Step 4 — Ingest the Evaluation Corpus

```bash
# Ingest HotpotQA supporting documents into vector + graph stores
python -X utf8 local_importingestion.py
```

This populates:
- `./data/vector/` — LanceDB vector store
- `./data/graph/` — KuzuDB knowledge graph
- `./cache/` — SQLite embedding and entity caches

Ingestion of the full HotpotQA dev corpus (~90k documents) takes approximately 3–6 hours on CPU.
A pre-ingested dataset snapshot can be provided by the author on request.

---

## Step 5 — Run Evaluations

### Main Evaluation (HotpotQA dev set)

```bash
python -X utf8 src/evaluations/evaluate_hotpotqa.py --samples 500
```

Results are written to `evaluation_results/hotpotqa_<timestamp>/`.

### Ablation Study (Table 4.x in thesis)

```bash
python -X utf8 src/evaluations/ablation_study.py --samples 100 --datasets hotpotqa
```

Results are written to `evaluation_results/ablation_<timestamp>/`.

### Diagnostic Run (single query)

```bash
python -X utf8 diagnose.py --idx 0
python -X utf8 diagnose.py --idx 0 --skip-llm   # skip Ollama call
```

---

## Verification

Run the test suite to verify the installation (does not require Ollama):

```bash
python -X utf8 -m pytest src/ test_system/ --ignore=src/evaluations/test_rag_quality.py -q
# Expected: ~380+ passed, 0 failures
```

Verify ingestion completed before running graph-dependent tests:

```bash
python -X utf8 -c "
import pathlib, sys
graph = pathlib.Path('data/knowledge_graph_kuzu')
vec   = pathlib.Path('data/vector')
ok = graph.exists() and vec.exists()
print('Ingestion OK' if ok else 'ERROR: run local_importingestion.py first')
sys.exit(0 if ok else 1)
"
```

**Dataset:** HotpotQA fullwiki dev set v1.1
Download: `http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json`

Generate and record the SHA-256 checksum before submission:

```bash
# Linux / macOS
sha256sum hotpot_dev_fullwiki_v1.json

# Windows PowerShell
Get-FileHash hotpot_dev_fullwiki_v1.json -Algorithm SHA256 | Select-Object Hash
```

SHA-256: *(fill in from the command above before final submission)*

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: KuzuDB not available` | kuzu not installed | `pip install kuzu==0.11.3` |
| `ConnectionError: localhost:11434` | Ollama not running | Run `ollama serve` |
| `Model 'qwen2:1.5b' not found` | Model not pulled | `ollama pull qwen2:1.5b` |
| Unicode errors on Windows | Missing UTF-8 flag | Use `python -X utf8 script.py` |
| GLiNER first-run slow | Model download (~250 MB) | Wait; cached after first run |

---

## Expected Results (Thesis Table 4.x)

| Configuration | EM | F1 | Avg. Latency |
|---|---|---|---|
| Vector-only baseline | TBD | TBD | TBD |
| Graph-only | TBD | TBD | TBD |
| Hybrid 70/30 | TBD | TBD | TBD |
| Full system (qwen2:1.5b) | TBD | TBD | TBD |

*(Fill in with final thesis numbers before submission)*
