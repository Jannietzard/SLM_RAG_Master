"""
Agentic-verification ablation — Tier 1B of the thesis evaluation.

This is THE central table of the thesis. It quantifies what the title's
"Agentic Verification" contributes by removing one component at a time and
measuring the EM/F1/SF-F1 delta.

Five rows are produced (LLM-only → Full system) so the reader sees the
marginal contribution of each layer:

    Row | Planner | Verifier | Iter | Name              | What it isolates
    ----|---------|----------|------|-------------------|-----------------
     1  |   off   |   off    |  0   | LLM-only          | parametric baseline (no RAG)
     2  |   off   |  gen     |  1   | RAG (no agent)    | retrieval contribution
     3  |   on    |  gen     |  1   | +Planner          | query decomposition value
     4  |   on    | gen+val  |  1   | +Verifier         | pre-validation value
     5  |   on    | gen+val  |  2   | +SelfCorrect      | self-correction loop

Why this matters for the thesis
───────────────────────────────
Reviewers will ask: "Is the gain coming from retrieval or from your agentic
components?" The five-row decomposition answers that:
    ΔEM(row2 − row1)  = retrieval contribution.
    ΔEM(row3 − row2)  = planner contribution.
    ΔEM(row4 − row3)  = verifier pre-validation contribution.
    ΔEM(row5 − row4)  = self-correction contribution.
Each of those deltas is a separate claim defensible in isolation.

Note: row 1 (LLM-only) is run differently from the rest. The agent pipeline
always involves retrieval. To get a true parametric baseline, this script
asks the LLM directly with NO context — see `_run_llm_only_row()`.

Usage:
    python -X utf8 -m src.thesis_evaluations.agentic_ablation \\
        --dataset hotpotqa \\
        --samples 100 \\
        --model qwen2:1.5b

Output:
    evaluation_results/agentic_ablation_{ts}/
        row1_llm_only.jsonl
        row2_rag_no_agent.jsonl
        row3_planner.jsonl
        row4_verifier.jsonl
        row5_self_correct.jsonl
        summary.csv / summary.md / summary.json

Thesis mapping
──────────────
The summary.md table is Chapter 6 §6.X "Agentic Verification: Component
Contribution". The deltas between rows become Figure 6.X (waterfall plot)
via thesis_results_aggregator.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    EvalResult,
    StoreManager,
    _classify_llm_error,
    _gold_titles_from_supporting_facts,
    compute_exact_match,
    compute_f1,
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
# Ablation definitions
# ---------------------------------------------------------------------------

ABLATION_ROWS = [
    # name, label, enable_planner, enable_verifier, max_iterations, llm_only, enable_pre_validation
    # Fix 2b (2026-05-26): added enable_pre_validation column so row3 ("+Planner")
    # and row4 ("+Verifier") produce distinct configurations. Previously both
    # rows had identical settings (Δ=0pp was structural, not measurement).
    # Now row3 = Planner + LLM generation but NO pre-validation checks; row4 =
    # row3 + the three pre-validation filters (entity-path / contradiction /
    # credibility). This makes the +Verifier row a real ablation.
    ("row1_llm_only",      "LLM-only (no retrieval)",  False, False, 1, True,  False),
    ("row2_rag_no_agent",  "RAG (no agent)",           False, True,  1, False, False),
    ("row3_planner",       "+Planner (no pre-val)",    True,  True,  1, False, False),
    ("row4_verifier",      "+Verifier (1 iter)",       True,  True,  1, False, True),
    ("row5_self_correct",  "+SelfCorrect (2 iter)",    True,  True,  2, False, True),
]


# ---------------------------------------------------------------------------
# LLM-only baseline runner (no retrieval at all)
# ---------------------------------------------------------------------------

def _llm_only_prompt(question: str) -> str:
    """Direct factual-QA prompt with no context.

    Matches the Verifier's prompt style (short answer rule) so the LLM-only
    row competes fairly with the RAG rows — the only difference is whether
    context is present, not how the LLM is instructed."""
    return (
        "You are a factual QA assistant. Answer based on your knowledge.\n\n"
        "Rules:\n"
        "- Give the shortest possible answer: a name, place, date, number, or yes/no.\n"
        "- Do NOT explain or add sentences beyond the direct answer.\n"
        "- If you don't know the answer: reply with \"I don't know.\"\n\n"
        f"Question: {question}\n\n"
        "Answer (as short as possible):"
    )


def run_llm_only_row(
    model_name: str,
    config: Dict[str, Any],
    questions: list,
    jsonl_out: Path,
) -> Dict[str, Any]:
    """Query the LLM directly with no context — the parametric baseline.

    Uses the Verifier's LLM-call infrastructure so the comparison is fair
    (same Ollama client, same timeout, same temperature). We bypass the
    Planner/Navigator/Verifier orchestration entirely."""
    from src.logic_layer.verifier import Verifier, VerifierConfig

    llm_cfg = config.get("llm", {})
    verifier_cfg = VerifierConfig(
        model_name=model_name,
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 200),
        timeout=llm_cfg.get("timeout", 60),
    )
    verifier = Verifier(config=verifier_cfg)

    if jsonl_out.exists():
        jsonl_out.unlink()
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    em_count = 0
    f1_sum = 0.0
    llm_err_count = 0
    time_sum_ms = 0.0
    n = 0

    from tqdm import tqdm
    for q in tqdm(questions, desc="LLM-only baseline", unit="q"):
        prompt = _llm_only_prompt(q.question)
        try:
            answer, latency_ms = verifier._call_llm(prompt)
        except Exception as exc:
            answer, latency_ms = f"[Error: {exc}]", 0.0

        em = compute_exact_match(answer, q.answer)
        f1 = compute_f1(answer, q.answer)
        llm_err, llm_err_type = _classify_llm_error(answer)

        em_count += 1 if em else 0
        f1_sum += f1
        llm_err_count += 1 if llm_err else 0
        time_sum_ms += latency_ms
        n += 1

        gold_titles = _gold_titles_from_supporting_facts(q.supporting_facts)
        rec = asdict(EvalResult(
            question_id=q.id,
            question=q.question,
            gold_answer=q.answer,
            predicted_answer=answer,
            exact_match=em,
            f1_score=f1,
            retrieval_count=0,
            time_ms=latency_ms,
            dataset=q.dataset,
            question_type=q.question_type,
            gold_titles=gold_titles,
            retrieved_titles=[],
            retrieval_recall=0.0,
            retrieval_precision=0.0,
            sf_f1=0.0,
            all_gold_retrieved=False,
            llm_error=llm_err,
            llm_error_type=llm_err_type,
            pipeline_succeeded_llm_failed=False,
            planner_query_type="(skipped)",
            hop_count=0,
            n_entities=0,
            verifier_iterations=0,
            all_verified=False,
            confidence="n/a",
        ))
        with open(jsonl_out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if n == 0:
        return {"row_name": "row1_llm_only", "label": "LLM-only", "n_questions": 0}

    return {
        "row_name": "row1_llm_only",
        "label": "LLM-only (no retrieval)",
        "n_questions": n,
        "em": em_count / n,
        "f1": f1_sum / n,
        "sf_f1": 0.0,  # no retrieval
        "sf_recall": 0.0,
        "em_given_retrieval_ok": 0.0,
        "llm_error_rate": llm_err_count / n,
        "avg_time_ms": time_sum_ms / n,
    }


# ---------------------------------------------------------------------------
# Standard ablation rows (rows 2–5)
# ---------------------------------------------------------------------------

def run_pipeline_row(
    row_name: str,
    label: str,
    enable_planner: bool,
    enable_verifier: bool,
    max_iterations: int,
    dataset: str,
    model_name: str,
    questions: list,
    config: Dict[str, Any],
    store_manager: StoreManager,
    output_dir: Path,
    enable_pre_validation: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run one ablation row through the full evaluate_dataset() pipeline."""
    logger.info("=" * 70)
    logger.info("ROW: %s -- %s", row_name, label)
    logger.info("  planner=%s verifier=%s iter=%d pre_val=%s",
                enable_planner, enable_verifier, max_iterations,
                enable_pre_validation)
    logger.info("=" * 70)

    pipeline = create_pipeline(
        dataset, config, store_manager,
        vector_weight=0.5, graph_weight=0.5,
        model_name=model_name,
        enable_planner=enable_planner,
        enable_verifier=enable_verifier,
        max_iterations=max_iterations,
        enable_pre_validation=enable_pre_validation,
    )

    jsonl_path = output_dir / f"{row_name}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    try:
        result = evaluate_dataset(
            dataset, questions, pipeline,
            config_name=row_name,
            vector_weight=0.5, graph_weight=0.5,
            jsonl_out=jsonl_path,
            retrieval_only=False,
        )
    except Exception as exc:
        logger.error("Row %s failed: %s", row_name, exc)
        result = None
    finally:
        del pipeline
        gc.collect()

    if result is None:
        return None

    return {
        "row_name": row_name,
        "label": label,
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
    }


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers (2026-05-27)
# ---------------------------------------------------------------------------
# Granularity = one ABLATION row. A row whose per-row JSONL already holds
# `n_questions` lines is treated as DONE and skipped on resume; its metrics
# are recomputed from the JSONL so the summary still includes it. This lets
# you run a few rows, stop (Ctrl+C), and re-run with --resume to continue
# from the first incomplete row. Partial rows are NOT resumed mid-question:
# they are re-run from scratch (the row runner deletes the partial JSONL),
# because the slow agentic rows write append-only and a clean restart avoids
# duplicate-question records.

def _jsonl_line_count(path: Path) -> int:
    """Number of JSONL records in `path` (0 if absent/unreadable)."""
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _aggregate_row_from_jsonl(
    jsonl_path: Path, row_name: str, label: str, is_llm_only: bool
) -> Optional[Dict[str, Any]]:
    """
    Rebuild a row's summary dict from a completed per-row JSONL, so a
    checkpoint-skipped row still appears in the aggregate summary without
    re-running it. Mirrors the metric set produced by run_pipeline_row /
    run_llm_only_row.
    """
    records: List[Dict[str, Any]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Checkpoint read failed for %s: %s", jsonl_path, exc)
        return None
    n = len(records)
    if n == 0:
        return None

    def _rate(pred) -> float:
        return sum(1 for r in records if pred(r)) / n

    em = _rate(lambda r: r.get("exact_match"))
    f1 = sum(r.get("f1_score", 0.0) for r in records) / n
    sf_f1 = sum(r.get("sf_f1", 0.0) for r in records) / n
    sf_recall = _rate(lambda r: r.get("all_gold_retrieved"))
    llm_err = _rate(lambda r: r.get("llm_error"))
    avg_time = sum(r.get("time_ms", 0.0) for r in records) / n
    retr_ok = [r for r in records if r.get("all_gold_retrieved")]
    em_retr_ok = (
        sum(1 for r in retr_ok if r.get("exact_match")) / len(retr_ok)
        if retr_ok else 0.0
    )

    row: Dict[str, Any] = {
        "row_name": row_name,
        "label": label,
        "n_questions": n,
        "em": em,
        "f1": f1,
        "sf_f1": sf_f1,
        "sf_recall": sf_recall,
        "em_given_retrieval_ok": em_retr_ok,
        "llm_error_rate": llm_err,
        "avg_time_ms": avg_time,
    }
    if not is_llm_only:
        # Pipeline-stage breakdown (matches run_pipeline_row's extra fields).
        row["pipeline_failed_rate"] = _rate(
            lambda r: not r.get("all_gold_retrieved")
        )
        row["pipeline_ok_llm_failed_rate"] = _rate(
            lambda r: bool(r.get("pipeline_succeeded_llm_failed"))
        )
        row["pipeline_ok_llm_wrong_rate"] = _rate(
            lambda r: r.get("all_gold_retrieved")
            and not r.get("llm_error")
            and not r.get("exact_match")
        )
        row["pipeline_ok_llm_ok_rate"] = _rate(
            lambda r: r.get("all_gold_retrieved")
            and not r.get("llm_error")
            and r.get("exact_match")
        )
    return row


# ---------------------------------------------------------------------------
# Summary writer with marginal-delta computation
# ---------------------------------------------------------------------------

def write_summary(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Emit CSV + Markdown summary with marginal-contribution deltas."""
    if not rows:
        logger.warning("No rows to summarise.")
        return

    csv_path = output_dir / "summary.csv"
    # Union of all row keys in first-seen order. Different row builders
    # (run_llm_only_row vs run_pipeline_row) emit different field sets;
    # the LLM-only baseline has no pipeline_*_rate fields, the pipeline
    # rows do. DictWriter writes empty for any missing key on a per-row
    # basis, so the resulting CSV is sparse but correct.
    fieldnames: List[str] = []
    seen: set = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Summary CSV: %s", csv_path)

    # Markdown table with deltas
    cols = [
        ("label", "Configuration"),
        ("em", "EM"),
        ("f1", "F1"),
        ("sf_f1", "SF-F1"),
        ("sf_recall", "SF-Recall"),
        ("em_given_retrieval_ok", "EM|retr.ok"),
        ("llm_error_rate", "LLM-err"),
        ("avg_time_ms", "Latency (ms)"),
    ]
    lines = ["# Agentic Verification — Ablation Results", ""]
    lines.append("| " + " | ".join(name for _, name in cols) + " | ΔEM |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    prev_em = None
    for r in rows:
        cells: List[str] = []
        for key, _ in cols:
            v = r.get(key)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                if key == "avg_time_ms":
                    cells.append(f"{v:.0f}")
                elif key in {"em", "f1", "sf_f1", "sf_recall",
                             "em_given_retrieval_ok", "llm_error_rate"}:
                    cells.append(f"{v * 100:.1f}%" if v <= 1.0 else f"{v:.3f}")
                else:
                    cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        # Delta column: EM gain over previous row
        if prev_em is None or r.get("em") is None:
            delta = "—"
        else:
            d = (r["em"] - prev_em) * 100
            delta = f"{d:+.1f}pp"
        prev_em = r.get("em")
        lines.append("| " + " | ".join(cells) + f" | {delta} |")
    lines.append("")
    lines.append("**Reading the table:**")
    lines.append("- Each row adds one component on top of the previous.")
    lines.append("- The **ΔEM** column shows the marginal contribution of that component.")
    lines.append("- Row 2 − Row 1 = retrieval gain.")
    lines.append("- Row 3 − Row 2 = planner gain (query decomposition).")
    lines.append("- Row 4 − Row 3 = verifier pre-validation gain.")
    lines.append("- Row 5 − Row 4 = self-correction loop gain.")

    md_path = output_dir / "summary.md"
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
    parser.add_argument("--dataset", "-d", default="hotpotqa")
    parser.add_argument("--samples", "-n", type=int, default=100)
    parser.add_argument("--model", "-m", default=None,
                        help="LLM model name (default: from settings.yaml).")
    parser.add_argument("--output", "-o", type=str,
                        default="./evaluation_results/agentic_ablation")
    parser.add_argument("--skip-llm-only", action="store_true",
                        help="Skip row 1 (parametric baseline). Useful for "
                             "fast re-runs of rows 2–5.")
    parser.add_argument(
        "--resume", nargs="?", const="__latest__", default=None, metavar="DIR",
        help="Resume an interrupted run. Pass a run directory to continue it, "
             "or use --resume with no value to auto-continue the most recent "
             "{output}_{dataset}_* directory. Rows whose JSONL already has all "
             "N questions are skipped; the first incomplete row is re-run.",
    )
    args = parser.parse_args()

    config = load_config_file()
    store_manager = StoreManager(Path("./data"))
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return

    model_name = args.model or config.get("llm", {}).get("model_name", "qwen2:1.5b")
    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions loaded for %s", args.dataset)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Resume logic: reuse an existing run directory instead of starting a fresh
    # timestamped one. --resume DIR uses that dir; --resume (no value) finds the
    # most recent {output}_{dataset}_* dir; absent -> new timestamped dir.
    if args.resume:
        if args.resume == "__latest__":
            base = Path(args.output)
            pattern = f"{base.name}_{args.dataset}_*"
            candidates = sorted(
                base.parent.glob(pattern),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
            )
            if candidates:
                output_dir = candidates[-1]
                logger.info("Resuming most recent run: %s", output_dir)
            else:
                output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
                logger.info("No prior run to resume; starting fresh: %s", output_dir)
        else:
            output_dir = Path(args.resume)
            if not output_dir.exists():
                logger.error("--resume directory does not exist: %s", output_dir)
                return
            logger.info("Resuming run: %s", output_dir)
    else:
        output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model: %s | Questions: %d | Output: %s",
                model_name, len(questions), output_dir)

    rows: List[Dict[str, Any]] = []
    n_expected = len(questions)
    for row_name, label, p, v, iters, is_llm_only, pre_val in ABLATION_ROWS:
        # Checkpoint: skip a row whose JSONL already holds all N questions.
        # Its metrics are recomputed from the JSONL so the summary is complete.
        row_jsonl = output_dir / f"{row_name}.jsonl"
        done = _jsonl_line_count(row_jsonl)
        if done >= n_expected:
            cached = _aggregate_row_from_jsonl(row_jsonl, row_name, label, is_llm_only)
            if cached:
                rows.append(cached)
                logger.info(
                    "CHECKPOINT: %s already complete (%d/%d) — skipped, "
                    "metrics loaded from JSONL (EM=%.1f%%)",
                    row_name, done, n_expected, cached["em"] * 100,
                )
                continue
            logger.warning(
                "CHECKPOINT: %s JSONL complete but unreadable — re-running",
                row_name,
            )
        elif done > 0:
            logger.info(
                "CHECKPOINT: %s partial (%d/%d) — re-running this row from "
                "scratch (mid-row resume not supported)",
                row_name, done, n_expected,
            )

        if is_llm_only:
            if args.skip_llm_only:
                logger.info("Skipping %s (--skip-llm-only)", row_name)
                continue
            try:
                row = run_llm_only_row(
                    model_name, config, questions,
                    jsonl_out=row_jsonl,
                )
                if row:
                    rows.append(row)
            except Exception as exc:
                logger.error("Row %s crashed: %s", row_name, exc)
            continue

        try:
            row = run_pipeline_row(
                row_name, label,
                enable_planner=p, enable_verifier=v, max_iterations=iters,
                dataset=args.dataset, model_name=model_name,
                questions=questions, config=config,
                store_manager=store_manager, output_dir=output_dir,
                enable_pre_validation=pre_val,
            )
            if row:
                rows.append(row)
                logger.info("  → EM=%.1f%% F1=%.3f SF-F1=%.3f",
                            row["em"] * 100, row["f1"], row["sf_f1"])
        except Exception as exc:
            logger.error("Row %s crashed: %s", row_name, exc)

    write_summary(rows, output_dir)

    (output_dir / "summary.json").write_text(
        json.dumps({"timestamp": ts, "dataset": args.dataset,
                    "model": model_name,
                    "n_samples": len(questions), "rows": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Done. Inspect: %s/summary.md", output_dir)
    # Auto-refresh the canonical thesis bundle for THIS dataset. Safe-wrapped:
    # a bundling failure must never break the eval that just produced the data.
    try:
        from .thesis_results_aggregator import update_bundle
        update_bundle(dataset=args.dataset)
    except Exception:
        logger.debug("Auto-bundle skipped", exc_info=True)


if __name__ == "__main__":
    main()
