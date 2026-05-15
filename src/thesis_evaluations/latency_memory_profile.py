"""
Latency / peak-memory profile — Tier 1C of the thesis evaluation.

Supports the title's "Resource-Constrained Devices" claim by instrumenting
every query with wall-clock per stage (S_P, S_N, S_V) and peak resident
memory. Produces a per-query breakdown plus aggregate distributions.

What this script does
─────────────────────
For each question:
    1. Snapshots RSS before the query.
    2. Runs pipeline.process(question), capturing per-stage timings from
       the PipelineResult (planner_time_ms / navigator_time_ms / verifier_time_ms).
    3. Polls RSS during the query in a background thread to capture peak.
    4. Computes %-of-budget figures vs. the 60s edge-device timeout.

For aggregation:
    - Per-stage mean / median / p95 / max latency.
    - Peak RSS mean / max.
    - Fraction of queries within the 60s budget (timeout rate proxy).
    - Pareto-friendly dump of (latency, EM) pairs for plotting.

Why these metrics
─────────────────
- Per-stage latency        : which stage dominates the cost? (S_V usually).
- p95 / max latency        : tail behaviour matters more than mean on edge
                             devices — a single slow query exceeds budget.
- Peak RSS                 : edge devices have 4-8 GB RAM budgets; this is
                             the binding constraint, not disk size.
- Within-budget rate       : direct edge-feasibility metric.

Usage:
    python -X utf8 -m src.thesis_evaluations.latency_memory_profile \\
        --dataset hotpotqa \\
        --samples 50 \\
        --model qwen2:1.5b \\
        --budget-seconds 60

Output:
    evaluation_results/latency_memory_{ts}/
        per_query.jsonl       — one record per question (full breakdown)
        per_stage.csv         — pivot table: query × stage × ms
        summary.md            — aggregate distributions + budget compliance
        summary.json          — same as machine-readable

Thesis mapping
──────────────
The summary.md table is Chapter 5 §5.X "Resource Profile on Edge Hardware".
The Pareto-front plot (latency vs. EM) is the headline figure for the
edge-feasibility claim. The p95/max latency numbers go into the discussion
of why qwen2:1.5b dominates on a 60s budget.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    StoreManager,
    _classify_llm_error,
    _gold_titles_from_supporting_facts,
    _install_retriever_title_capture,
    _retrieved_titles_for_chunks,
    compute_exact_match,
    compute_f1,
    create_pipeline,
    load_config_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Peak-memory poller (background thread)
# ---------------------------------------------------------------------------

try:
    import psutil  # type: ignore
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


class _RSSPoller:
    """Polls Process.memory_info().rss in a background thread, recording
    the maximum observed during its lifetime.

    Why a thread? Python-level instrumentation of long-running LLM calls
    can't sample memory mid-call any other way. The 50ms interval is fine
    for the ~5-50s queries we measure."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_mb: float = 0.0

    def _poll(self) -> None:
        proc = psutil.Process(os.getpid()) if _PSUTIL_OK else None
        while not self._stop.is_set():
            if proc is not None:
                try:
                    rss_mb = proc.memory_info().rss / (1024 * 1024)
                    if rss_mb > self.peak_mb:
                        self.peak_mb = rss_mb
                except Exception:
                    pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if not _PSUTIL_OK:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if self._thread is None:
            return 0.0
        self._stop.set()
        self._thread.join(timeout=0.5)
        return self.peak_mb


# ---------------------------------------------------------------------------
# Per-query profiler
# ---------------------------------------------------------------------------

def profile_one_query(pipeline, q, budget_seconds: float) -> Dict[str, Any]:
    """Run one query through the pipeline and capture latency / memory."""
    poller = _RSSPoller()
    poller.start()
    pre_rss = (
        psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        if _PSUTIL_OK else None
    )

    start = time.time()
    try:
        result = pipeline.process(q.question)
        elapsed = time.time() - start
        error = None
    except Exception as exc:
        result = None
        elapsed = time.time() - start
        error = str(exc)

    peak_rss = poller.stop()

    rec: Dict[str, Any] = {
        "question_id": q.id,
        "question": q.question,
        "gold_answer": q.answer,
        "question_type": q.question_type,
        "total_time_ms": elapsed * 1000.0,
        "within_budget": elapsed <= budget_seconds,
        "pre_rss_mb": pre_rss,
        "peak_rss_mb": peak_rss if peak_rss > 0 else None,
        "delta_rss_mb": (peak_rss - pre_rss) if (pre_rss and peak_rss > 0) else None,
        "error": error,
    }

    if result is None:
        rec.update({
            "planner_time_ms": 0.0,
            "navigator_time_ms": 0.0,
            "verifier_time_ms": 0.0,
            "predicted_answer": "",
            "exact_match": False,
            "f1_score": 0.0,
            "llm_error": True,
            "llm_error_type": "exception",
        })
        return rec

    pred = getattr(result, "answer", "") or ""
    em = compute_exact_match(pred, q.answer)
    f1 = compute_f1(pred, q.answer)
    llm_err, llm_err_type = _classify_llm_error(pred)

    # Retrieval quality
    nav = getattr(result, "navigator_result", {}) or {}
    filtered = nav.get("filtered_context", []) if isinstance(nav, dict) else []
    gold_titles = _gold_titles_from_supporting_facts(q.supporting_facts)
    retrieved_titles = _retrieved_titles_for_chunks(filtered)
    all_gold = bool(gold_titles) and set(gold_titles).issubset(set(retrieved_titles))

    rec.update({
        "planner_time_ms": float(getattr(result, "planner_time_ms", 0.0) or 0.0),
        "navigator_time_ms": float(getattr(result, "navigator_time_ms", 0.0) or 0.0),
        "verifier_time_ms": float(getattr(result, "verifier_time_ms", 0.0) or 0.0),
        "predicted_answer": pred,
        "exact_match": em,
        "f1_score": f1,
        "llm_error": llm_err,
        "llm_error_type": llm_err_type,
        "retrieval_count": len(filtered),
        "all_gold_retrieved": all_gold,
    })
    return rec


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _stats(values: List[float]) -> Dict[str, float]:
    """mean / median / p95 / max for a non-empty list."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    vs = sorted(values)
    p95_idx = max(0, int(0.95 * (len(vs) - 1)))
    return {
        "mean": statistics.mean(vs),
        "median": statistics.median(vs),
        "p95": vs[p95_idx],
        "max": vs[-1],
    }


def aggregate(records: List[Dict[str, Any]], budget_seconds: float) -> Dict[str, Any]:
    if not records:
        return {}
    n = len(records)
    total_ms = [r["total_time_ms"] for r in records if r.get("total_time_ms") is not None]
    planner_ms = [r["planner_time_ms"] for r in records if r.get("planner_time_ms") is not None]
    nav_ms = [r["navigator_time_ms"] for r in records if r.get("navigator_time_ms") is not None]
    ver_ms = [r["verifier_time_ms"] for r in records if r.get("verifier_time_ms") is not None]
    peak_rss = [r["peak_rss_mb"] for r in records if r.get("peak_rss_mb")]
    delta_rss = [r["delta_rss_mb"] for r in records if r.get("delta_rss_mb")]

    within_budget = sum(1 for r in records if r.get("within_budget")) / n
    em_rate = sum(1 for r in records if r.get("exact_match")) / n
    llm_err_rate = sum(1 for r in records if r.get("llm_error")) / n
    all_gold_rate = sum(1 for r in records if r.get("all_gold_retrieved")) / n

    return {
        "n_queries": n,
        "budget_seconds": budget_seconds,
        "within_budget_rate": within_budget,
        "exact_match_rate": em_rate,
        "llm_error_rate": llm_err_rate,
        "all_gold_retrieved_rate": all_gold_rate,
        "total_ms": _stats(total_ms),
        "planner_ms": _stats(planner_ms),
        "navigator_ms": _stats(nav_ms),
        "verifier_ms": _stats(ver_ms),
        "peak_rss_mb": _stats(peak_rss),
        "delta_rss_mb": _stats(delta_rss),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_outputs(records: List[Dict[str, Any]], agg: Dict[str, Any],
                  output_dir: Path) -> None:
    # Per-query JSONL
    jsonl_path = output_dir / "per_query.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-stage CSV (one row per query)
    csv_path = output_dir / "per_stage.csv"
    fieldnames = [
        "question_id", "question_type",
        "planner_time_ms", "navigator_time_ms", "verifier_time_ms",
        "total_time_ms", "within_budget",
        "peak_rss_mb", "delta_rss_mb",
        "exact_match", "f1_score", "llm_error", "all_gold_retrieved",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Markdown summary
    md = ["# Latency / Memory Profile", ""]
    md.append(f"- Queries: **{agg['n_queries']}**")
    md.append(f"- Budget: **{agg['budget_seconds']:.0f}s**")
    md.append(f"- Within-budget rate: **{agg['within_budget_rate']*100:.1f}%**")
    md.append(f"- EM rate: **{agg['exact_match_rate']*100:.1f}%**")
    md.append(f"- LLM error rate: **{agg['llm_error_rate']*100:.1f}%**")
    md.append(f"- All-gold-retrieved rate: **{agg['all_gold_retrieved_rate']*100:.1f}%**")
    md.append("")
    md.append("## Per-stage latency (ms)")
    md.append("")
    md.append("| Stage | Mean | Median | P95 | Max |")
    md.append("|---|---|---|---|---|")
    for stage_key, stage_name in [
        ("planner_ms", "S_P (Planner)"),
        ("navigator_ms", "S_N (Navigator)"),
        ("verifier_ms", "S_V (Verifier)"),
        ("total_ms", "**Total**"),
    ]:
        s = agg.get(stage_key, {})
        md.append(
            f"| {stage_name} | {s.get('mean', 0):.0f} | {s.get('median', 0):.0f} "
            f"| {s.get('p95', 0):.0f} | {s.get('max', 0):.0f} |"
        )
    md.append("")
    md.append("## Peak resident memory (MB)")
    md.append("")
    md.append("| Metric | Mean | Median | P95 | Max |")
    md.append("|---|---|---|---|---|")
    s = agg.get("peak_rss_mb", {})
    md.append(
        f"| Peak RSS | {s.get('mean', 0):.0f} | {s.get('median', 0):.0f} "
        f"| {s.get('p95', 0):.0f} | {s.get('max', 0):.0f} |"
    )
    s = agg.get("delta_rss_mb", {})
    md.append(
        f"| ΔRSS / query | {s.get('mean', 0):.0f} | {s.get('median', 0):.0f} "
        f"| {s.get('p95', 0):.0f} | {s.get('max', 0):.0f} |"
    )
    md.append("")
    md.append("**Reading the table:**")
    md.append("- p95 latency is the headline edge-feasibility metric — if "
              "p95 < budget, the system is reliable on edge hardware.")
    md.append("- The dominant stage (almost always S_V) is the optimization target.")
    md.append("- Peak RSS bounds the minimum hardware spec; ΔRSS / query "
              "indicates working-set churn (KV cache, GLiNER, etc.).")

    (output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Per-query records: %s", jsonl_path)
    logger.info("Per-stage CSV:    %s", csv_path)
    logger.info("Summary MD:       %s", output_dir / "summary.md")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default="hotpotqa")
    parser.add_argument("--samples", "-n", type=int, default=50)
    parser.add_argument("--model", "-m", default=None,
                        help="LLM model name (default: from settings.yaml).")
    parser.add_argument("--budget-seconds", type=float, default=60.0,
                        help="Edge-device latency budget for within-budget %% "
                             "computation. Default: 60s (matches Ollama timeout).")
    parser.add_argument("--output", "-o", type=str,
                        default="./evaluation_results/latency_memory")
    args = parser.parse_args()

    if not _PSUTIL_OK:
        logger.warning(
            "psutil not installed — peak-memory columns will be empty. "
            "Install with: pip install psutil"
        )

    config = load_config_file()
    store_manager = StoreManager(Path("./data"))
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return

    model_name = args.model or config.get("llm", {}).get("model_name", "qwen2:1.5b")
    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions loaded.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model: %s | Questions: %d | Budget: %.0fs | Output: %s",
                model_name, len(questions), args.budget_seconds, output_dir)

    pipeline = create_pipeline(
        args.dataset, config, store_manager,
        vector_weight=0.5, graph_weight=0.5,
        model_name=model_name,
    )
    # Capture source titles for SF-retrieval tracking.
    original_retrieve = _install_retriever_title_capture(pipeline)

    from tqdm import tqdm
    records: List[Dict[str, Any]] = []
    try:
        for q in tqdm(questions, desc="Profiling", unit="q"):
            try:
                records.append(profile_one_query(pipeline, q, args.budget_seconds))
            except Exception as exc:
                logger.warning("Q%s crashed: %s", q.id, exc)
    finally:
        # Restore retriever and clean up the pipeline so the process exits cleanly.
        if original_retrieve is not None:
            for attr in ("retriever", "_retriever"):
                cand = getattr(pipeline, attr, None)
                if cand is not None and hasattr(cand, "retrieve"):
                    cand.retrieve = original_retrieve
                    break
        del pipeline
        gc.collect()

    agg = aggregate(records, args.budget_seconds)
    write_outputs(records, agg, output_dir)
    logger.info("Done. Inspect: %s/summary.md", output_dir)


if __name__ == "__main__":
    main()
