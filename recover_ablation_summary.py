"""
Recovery script: rebuild summary.csv + summary.md from the per-row JSONL
traces of a completed agentic_ablation run that crashed during the final
CSV write (see agentic_ablation.py write_summary union-fieldnames fix).

Usage:
    python -X utf8 recover_ablation_summary.py <run_dir>

Example:
    python -X utf8 recover_ablation_summary.py \\
        evaluation_results/agentic_ablation_hotpotqa_20260526_114535
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

ROW_ORDER = [
    ("row1_llm_only",    "LLM-only (no retrieval)"),
    ("row2_rag_no_agent","Hybrid retrieval + LLM (no agent)"),
    ("row3_planner",     "+ Planner (query decomposition)"),
    ("row4_verifier",    "+ Verifier (pre-validation)"),
    ("row5_self_correct","+ Self-correction loop"),
]

def aggregate(jsonl_path: Path, row_name: str, label: str) -> Dict[str, Any]:
    if not jsonl_path.exists():
        return {}
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n = len(records)
    if n == 0:
        return {"row_name": row_name, "label": label, "n_questions": 0}

    em = sum(1 for r in records if r.get("exact_match")) / n
    f1 = sum(r.get("f1_score", 0.0) for r in records) / n
    sf_f1 = sum(r.get("sf_f1", 0.0) for r in records) / n
    sf_recall = sum(1 for r in records if r.get("all_gold_retrieved")) / n
    llm_err = sum(1 for r in records if r.get("llm_error")) / n
    avg_time = sum(r.get("time_ms", 0.0) for r in records) / n

    # EM | retrieval ok = EM among questions where retrieval succeeded
    retr_ok = [r for r in records if r.get("all_gold_retrieved")]
    em_retr_ok = (sum(1 for r in retr_ok if r.get("exact_match")) / len(retr_ok)
                  if retr_ok else 0.0)

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

    # Pipeline-stage breakdown only available for pipeline rows
    if row_name != "row1_llm_only":
        pipeline_failed = sum(
            1 for r in records
            if r.get("retrieval_count", 0) == 0 or not r.get("all_gold_retrieved")
        ) / n
        pipeline_ok_llm_failed = sum(
            1 for r in records
            if r.get("pipeline_succeeded_llm_failed")
        ) / n
        pipeline_ok_llm_wrong = sum(
            1 for r in records
            if r.get("all_gold_retrieved")
            and not r.get("llm_error")
            and not r.get("exact_match")
        ) / n
        pipeline_ok_llm_ok = sum(
            1 for r in records
            if r.get("all_gold_retrieved")
            and not r.get("llm_error")
            and r.get("exact_match")
        ) / n
        row["pipeline_failed_rate"] = pipeline_failed
        row["pipeline_ok_llm_failed_rate"] = pipeline_ok_llm_failed
        row["pipeline_ok_llm_wrong_rate"] = pipeline_ok_llm_wrong
        row["pipeline_ok_llm_ok_rate"] = pipeline_ok_llm_ok

    return row


def write_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    csv_path = out_dir / "summary.csv"
    fieldnames: List[str] = []
    seen: set = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    cols = [
        ("label", "Configuration"), ("em", "EM"), ("f1", "F1"),
        ("sf_f1", "SF-F1"), ("sf_recall", "SF-Recall"),
        ("em_given_retrieval_ok", "EM|retr.ok"),
        ("llm_error_rate", "LLM-err"), ("avg_time_ms", "Latency (ms)"),
    ]
    lines = ["# Agentic Verification -- Ablation Results (recovered)", ""]
    lines.append("| " + " | ".join(n for _, n in cols) + " | ΔEM |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    prev_em = None
    for r in rows:
        cells: List[str] = []
        for k, _ in cols:
            v = r.get(k)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                if k == "avg_time_ms":
                    cells.append(f"{v:.0f}")
                elif k in {"em", "f1", "sf_f1", "sf_recall",
                           "em_given_retrieval_ok", "llm_error_rate"}:
                    cells.append(f"{v*100:.1f}%" if v <= 1.0 else f"{v:.3f}")
                else:
                    cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        d = "-" if prev_em is None or r.get("em") is None else f"{(r['em']-prev_em)*100:+.1f}pp"
        prev_em = r.get("em")
        lines.append("| " + " | ".join(cells) + f" | {d} |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: recover_ablation_summary.py <run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}")
        sys.exit(1)
    rows = []
    for row_name, label in ROW_ORDER:
        row = aggregate(run_dir / f"{row_name}.jsonl", row_name, label)
        if row:
            rows.append(row)
            print(f"  {row_name}: n={row.get('n_questions', 0)} "
                  f"EM={row.get('em', 0)*100:.1f}% F1={row.get('f1', 0):.3f}")
    if rows:
        write_summary(rows, run_dir)
