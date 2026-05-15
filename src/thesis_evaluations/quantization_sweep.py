"""
Quantization × model-size sweep — Tier 1A of the thesis evaluation.

Supports the title's "Quantized Small Language Models" claim by running the
full pipeline against several Ollama model variants and producing a side-by-
side comparison of accuracy, latency, peak memory, and retrieval quality.

What this script does
─────────────────────
For each (model, quantization) cell in --models:
    1. Loads the pipeline once with that model.
    2. Runs `n_samples` HotpotQA (or other --dataset) questions through it.
    3. Captures EM, F1, SF-F1, SF-recall, LLM-error rate, latency, peak RAM.
    4. Writes a per-cell JSONL (per-question) and a summary row.
After all cells:
    5. Aggregates rows into a Markdown table and a CSV file.
    6. Computes a "quantization cost" row: ΔEM/ΔSize between Q4 ↔ Q8 of the
       same model.

Why these metrics
─────────────────
- EM / F1                : final answer quality (what the user sees).
- SF-F1 / SF-recall      : pipeline retrieval quality (independent of LLM).
- EM | retrieval-ok      : model accuracy CONDITIONED on correct retrieval —
                           isolates LLM capability from pipeline noise.
- Avg latency / Peak RAM : edge-device feasibility (60s budget, RAM cap).
- LLM-error rate         : how often the model times out (sub-claim about
                           model stability under the 60s budget).

Suggested model set (from settings.yaml available_models):
    qwen2:1.5b              (Q4_0 default, baseline)
    qwen2.5:3b              (Q4_0, larger model same quantization)
    qwen2:1.5b-instruct-q8_0 (Q8_0 if you've pulled it — shows quant cost)
    phi3                    (3.8B, ablation comparison)

Usage:
    python -X utf8 -m src.thesis_evaluations.quantization_sweep \\
        --dataset hotpotqa \\
        --samples 100 \\
        --models qwen2:1.5b,qwen2.5:3b,phi3 \\
        --output ./evaluation_results/quantization_sweep

Output:
    evaluation_results/quantization_sweep_{ts}/
        {model}.jsonl              per-question records
        summary.csv                one row per model (machine-readable)
        summary.md                 same as a Markdown table (thesis-ready)
        peak_memory.csv            per-stage peak RAM samples (one row per Q)

Thesis mapping
──────────────
This script's summary.md goes into Chapter 5 §5.X "Quantization Impact" of
the thesis. The Pareto front (latency vs. EM) becomes Figure 5.X via
thesis_results_aggregator.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the project root importable when this file is run with `python -m ...`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    StoreManager,
    create_pipeline,
    evaluate_dataset,
    load_config_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Peak-memory measurement helper (best-effort; falls back gracefully)
# ---------------------------------------------------------------------------
try:
    import psutil  # type: ignore
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    logger.warning(
        "psutil not installed — peak-memory column will be NaN. "
        "Install with: pip install psutil"
    )


def _current_rss_mb() -> Optional[float]:
    """Resident set size of the current process in MB, or None if unavailable.

    RSS is the right metric for edge-device claims because it counts every
    page actually in physical memory — model weights, KV cache, retriever
    buffers all together."""
    if not _PSUTIL_OK:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def run_one_model(
    model_name: str,
    dataset: str,
    questions: list,
    config: Dict[str, Any],
    store_manager: StoreManager,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Run the pipeline with one model and return summary metrics.

    The pipeline is built fresh for each model (Ollama keeps weights warm
    in its own process, so this is not a measurement-fairness concern as
    long as we don't share Python state across runs)."""
    logger.info("=" * 70)
    logger.info("MODEL: %s", model_name)
    logger.info("=" * 70)

    pre_rss = _current_rss_mb()
    pipeline = create_pipeline(
        dataset, config, store_manager,
        vector_weight=0.5, graph_weight=0.5,
        model_name=model_name,
    )
    post_rss = _current_rss_mb()
    load_rss_delta = (
        (post_rss - pre_rss) if (pre_rss is not None and post_rss is not None) else None
    )

    jsonl_path = output_dir / f"{model_name.replace(':', '-').replace('/', '_')}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    start = time.time()
    try:
        result = evaluate_dataset(
            dataset, questions, pipeline,
            config_name=model_name,
            vector_weight=0.5, graph_weight=0.5,
            jsonl_out=jsonl_path,
            retrieval_only=False,
        )
    except Exception as exc:
        logger.error("Pipeline failed on model %s: %s", model_name, exc)
        result = None
    elapsed = time.time() - start

    peak_rss = _current_rss_mb()

    # Free the pipeline so the next model's RSS measurement isn't polluted.
    del pipeline
    gc.collect()

    if result is None:
        return None

    summary = {
        "model": model_name,
        "n_questions": result.n_questions,
        "em": result.exact_match,
        "f1": result.f1_score,
        "sf_f1": result.avg_sf_f1,
        "sf_recall": result.sf_recall_rate,
        "em_given_retrieval_ok": result.retrieval_only_em,
        "llm_error_rate": result.llm_error_rate,
        "pipeline_failed_rate": result.pipeline_failed_rate,
        "pipeline_ok_llm_failed_rate": result.pipeline_ok_llm_failed_rate,
        "pipeline_ok_llm_wrong_rate": result.pipeline_ok_llm_wrong_rate,
        "pipeline_ok_llm_ok_rate": result.pipeline_ok_llm_ok_rate,
        "avg_time_ms": result.avg_time_ms,
        "total_elapsed_s": elapsed,
        "load_rss_delta_mb": load_rss_delta,
        "peak_rss_mb": peak_rss,
        "jsonl_path": str(jsonl_path),
    }
    return summary


# ---------------------------------------------------------------------------
# Aggregation & output
# ---------------------------------------------------------------------------

def write_summary(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Emit summary.csv + summary.md from the per-model summary rows."""
    if not rows:
        logger.warning("No rows to write — all models failed.")
        return

    # CSV
    csv_path = output_dir / "summary.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Summary CSV: %s", csv_path)

    # Markdown
    md_path = output_dir / "summary.md"
    cols = [
        ("model", "Model"),
        ("em", "EM"),
        ("f1", "F1"),
        ("sf_f1", "SF-F1"),
        ("sf_recall", "SF-Recall"),
        ("em_given_retrieval_ok", "EM|retr.ok"),
        ("llm_error_rate", "LLM-err"),
        ("avg_time_ms", "Latency (ms)"),
        ("peak_rss_mb", "Peak RSS (MB)"),
    ]
    lines = ["# Quantization Sweep Results", ""]
    lines.append("| " + " | ".join(name for _, name in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells: List[str] = []
        for key, _ in cols:
            v = r.get(key)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                if "rate" in key or key in {"em", "f1", "sf_f1", "sf_recall",
                                            "em_given_retrieval_ok",
                                            "llm_error_rate"}:
                    cells.append(f"{v * 100:.1f}%" if v <= 1.0 else f"{v:.3f}")
                else:
                    cells.append(f"{v:.0f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Column legend:**")
    lines.append("- EM/F1: final answer correctness (LLM output).")
    lines.append("- SF-F1: supporting-fact F1 — did retrieval find the right paragraphs?")
    lines.append("- SF-Recall: % of questions where ALL gold supporting paragraphs were retrieved.")
    lines.append("- EM|retr.ok: EM among questions where retrieval succeeded "
                 "(isolates LLM capability).")
    lines.append("- LLM-err: % of timeouts/API errors (model stability under 60s budget).")
    lines.append("- Peak RSS: peak resident-set memory (edge-device feasibility).")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary Markdown: %s", md_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default="hotpotqa",
                        help="Dataset name (default: hotpotqa)")
    parser.add_argument("--samples", "-n", type=int, default=100,
                        help="Number of questions per model (default: 100)")
    parser.add_argument(
        "--models", "-m", type=str,
        default="qwen2:1.5b,qwen2.5:3b,phi3",
        help="Comma-separated list of Ollama model names. Each must be pulled "
             "via `ollama pull <name>`. Default: qwen2:1.5b,qwen2.5:3b,phi3",
    )
    parser.add_argument("--output", "-o", type=str,
                        default="./evaluation_results/quantization_sweep",
                        help="Output directory base (timestamp appended).")
    args = parser.parse_args()

    config = load_config_file()
    store_manager = StoreManager(Path("./data"))
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return

    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions loaded for %s", args.dataset)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    logger.info("Models to sweep: %s", models)
    logger.info("Questions per model: %d", len(questions))

    rows: List[Dict[str, Any]] = []
    for model in models:
        try:
            row = run_one_model(
                model, args.dataset, questions, config, store_manager, output_dir
            )
            if row:
                rows.append(row)
                logger.info("  → EM=%.1f%% F1=%.3f SF-F1=%.3f Latency=%.0fms PeakRSS=%s",
                            row["em"] * 100, row["f1"], row["sf_f1"],
                            row["avg_time_ms"],
                            f"{row['peak_rss_mb']:.0f}MB" if row["peak_rss_mb"] else "N/A")
        except Exception as exc:
            logger.error("Model %s crashed: %s", model, exc)

    write_summary(rows, output_dir)

    # Also dump the raw rows to JSON for thesis_results_aggregator.
    (output_dir / "summary.json").write_text(
        json.dumps({"timestamp": ts, "dataset": args.dataset,
                    "n_samples": len(questions), "rows": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Done. Inspect: %s/summary.md", output_dir)


if __name__ == "__main__":
    main()
