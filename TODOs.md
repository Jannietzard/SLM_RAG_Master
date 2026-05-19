# TODOs — Edge-RAG Thesis

Working task list. Newest priority block at the top.

---

## 1. Quantization sweep — defend the "Quantized" claim in the thesis title

**Why this block exists:** the thesis title is *"Enhancing Reasoning Fidelity
in **Quantized** Small Language Models on Edge Devices …"*. The word
"Quantized" obligates a measured result about quantization. Right now every
model is run at Ollama's default Q4_K_M and there is no comparison — so the
quantization claim is unsupported. This block produces the
quantization×accuracy×latency×memory table that becomes the "Quantized"
chapter.

Background facts (so the tasks make sense):
- Ollama's default model tags (`qwen2:1.5b`, `qwen2.5:3b`, `llama3.2:3b`,
  `phi3`) are **already Q4_K_M (~4-bit)**. The system is already running
  quantized models.
- Quantization stores weights in fewer bits → smaller file → fits in less
  RAM → fewer bytes RAM→CPU per token → **faster generation**. The cost is
  weight-precision loss → possible accuracy drop.
- Quantization does NOT change the context-window length (that is an
  architecture property), and does NOT make the model "smarter."
- The `#`-commented models in `test_prompt_idk.py` failed the 60 s budget
  because they are larger, not because of quantization. Quantizing them
  harder is the lever that could make them budget-viable.

### Tasks

- [ ] **1.1 — Pick the sweep grid.** Headline experiment: one model at three
      bit-widths. Recommended: `qwen2.5:3b` at Q8_0 / Q4_K_M / Q3_K_M.
      Rationale: a 3B model is large enough that the quantization effect on
      latency/memory is visible, small enough to stay near the edge budget.

- [ ] **1.2 — Pull the explicit-quantization Ollama tags.** Default tags are
      Q4_K_M; explicit-bit-width tags must be pulled by name. Check
      `ollama.com/library/qwen2.5/tags` for the exact tag strings, then:
      ```
      ollama pull qwen2.5:3b-instruct-q8_0
      ollama pull qwen2.5:3b-instruct-q4_K_M
      ollama pull qwen2.5:3b-instruct-q3_K_M
      ollama list   # confirm all three are present
      ```

- [ ] **1.3 — Verify input artifacts before any eval run.**
      ```
      python -X utf8 verify_artifacts.py verify
      ```
      Must exit 0. If not, the graph/inputs differ from the thesis inputs.

- [ ] **1.4 — Run the quantization sweep.** `quantization_sweep.py` already
      accepts `--models` as a comma-separated list and treats each entry as a
      separate sweep cell, so explicit quant tags work directly:
      ```
      python -X utf8 -m src.thesis_evaluations.quantization_sweep `
          --dataset hotpotqa --samples 100 `
          --models "qwen2.5:3b-instruct-q8_0,qwen2.5:3b-instruct-q4_K_M,qwen2.5:3b-instruct-q3_K_M"
      ```
      Output: `evaluation_results/quantization_sweep_<ts>/` with per-model
      JSONL + summary.json. Records EM / F1 / SF-F1 / latency per cell.

- [ ] **1.5 — Capture peak memory per quantization level.** The sweep records
      latency; run `latency_memory_profile.py` per quant tag to get peak RSS,
      OR confirm `quantization_sweep.py` already captures peak memory (check
      its summary.json fields). The "Edge Devices" claim needs the <16 GB
      RAM number per quantization level.

- [ ] **1.6 — Build the quantization table.** Run the aggregator; it emits
      `table_quantization.tex` from the sweep output:
      ```
      python -X utf8 -m src.thesis_evaluations.thesis_results_aggregator
      ```
      Target table shape — one row per bit-width:
      | Quantization | EM | F1 | p95 latency | Peak RAM | Within 60 s / 16 GB budget? |

- [ ] **1.7 — (Stretch) Edge-viability of a 7B model via aggressive
      quantization.** `mistral:7b` failed the 60 s budget at Q4. Pull
      `mistral:7b` at Q3_K_M (and/or Q2_K) and run a single benchmark pass.
      If it now passes the budget, that is a genuinely publishable result:
      *"aggressive quantization makes a 7B model edge-viable, at a measured
      accuracy cost of X."* If it still fails, report that honestly — a
      negative result is also a result.

- [ ] **1.8 — Write the "Quantized" methodology paragraph.** Once 1.4–1.6 are
      done, write the thesis text: state the quantization scheme (GGUF /
      llama.cpp `K_M` mixed-precision), the bit-widths evaluated, and the
      observed accuracy↔latency↔memory trade-off. The defensible claim is the
      sweet-spot finding (likely "Q4_K_M retains most of Q8's accuracy while
      meeting the edge budget"), backed by the 1.6 table.

**Definition of done for this block:** `table_quantization.tex` exists with
≥3 bit-width rows for at least one model, each row carrying EM/F1, p95
latency, and peak RAM; the thesis "Quantized" paragraph is written and cites
that table.

---

## 2. Fix Phase 3d.5 entity-linking performance (CODE — do after current ingest finishes)

**Why this block exists:** the 2026-05-18/19 ingestion measured Phase 3d.5
(embedding-based entity linking) at **~14–16 hours wall-clock**. The
embedding step is fast (5–13 min per type bucket); the slow part is the
graph-mutation step `_redirect_entity_edges` in `graph_quality.py`, which
re-points every `MENTIONS` / `RELATED_TO` edge of every merged entity using
**one auto-committed KuzuDB Cypher statement per edge** — i.e. one fsync per
statement, tens of thousands of them, catastrophically slow on Windows.

Measured linking durations: PERSON 2h27m, LOCATION 3h38m, DATE 3h32m.

The co-occurrence phase (3c) already solved the identical problem via
`add_related_to_relations_bulk()` with explicit `batch_begin()` /
`batch_commit()` transactions (500 edges/commit) — which is why 3c is fast.
Phase 3d.5 never received the same treatment.

**Thesis relevance:** a 14-hour entity-linking step contradicts the
"edge-deployable" claim and a reviewer would flag it. The fix makes the
phase edge-defensible AND is itself a reportable engineering result
("entity-linking edge re-pointing uses transactional batching").

### Tasks

- [ ] **2.1 — DO NOT touch the code while the current ingest is running.**
      Phase 3d.5 writes its `3d5` checkpoint only when the whole phase
      completes; let the in-progress run finish first.

- [ ] **2.2 — Wrap `_redirect_entity_edges` edge-rewrites in a transaction.**
      In `src/data_layer/graph_quality.py`, wrap the per-entity edge
      re-create + delete loop in `graph_store.batch_begin()` /
      `batch_commit()` (the same primitives `build_cooccurrence_edges` uses).
      Expected speedup: 10–30×, i.e. ~14 h → ~20–40 min.

- [ ] **2.3 — Add a regression test** asserting the batched path produces the
      same merged-graph state as the unbatched path on a small synthetic
      graph (correctness must be unchanged — only speed).

- [ ] **2.4 — Run the data-layer test suite** to confirm no regression.

**Definition of done:** `_redirect_entity_edges` uses transactional
batching; a test pins merge-correctness; the next full re-ingest's Phase
3d.5 completes in minutes, not hours.

---

## 3. Pending eval runs and writing (walltime + thesis text — NO code left)

1. Verifier — B3 (the one open item from the Verifier audit)
The Verifier audit closed B1, B2, B4–B10. B3 was never done. It needs:

Run agentic_ablation.py (~200 samples), compare row 4 (+Verifier) vs row 5 (+SelfCorrect).
Decide whether max_iterations=2 is justified or should drop to 1.
This is an eval run, not code — and the new bootstrap significance table will quantify it automatically.
2. Eval runs that everything else waits on
Across multiple sessions I flagged these as "walltime, not code, infrastructure is ready":

Re-ingest the graph — you hit the corrupted embedding cache; the fix (del cache\*.db, re-run the ingest command) was given but I never saw it confirmed complete.
500-question headline benchmark run — produces the headline numbers + per-pattern JSONL.
chunking_ablation.py run — the #4 ablation table (script written, never executed).
Tier-1 ablation suite — agentic_ablation.py, quantization_sweep.py, latency_memory_profile.py.
3. Component #8 — Entity extraction
Marked "out of scope / Colab-side" in the original 10-component audit. Effectively done (your new Colab with log-prob confidence), just needs the methodology paragraph written about the ~28% REBEL drop rate.

4. Per-pattern analysis (planner)
The matched_pattern instrumentation (P1) is in place, but the actual analysis — "which patterns fire, do they help, delete the dead ones" — can only happen after a 500-question eval run. That's a downstream task.


Check how much graph visible: - graph gernell einmal durchtesten mehrwert in resquests etc
70-80 testen 
fastes SLM / How do i quantisise? 
import laufen lassen