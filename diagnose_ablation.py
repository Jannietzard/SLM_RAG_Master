"""
Diagnostic harness for agentic_ablation runs.

Reads per-row JSONL traces and produces a precise breakdown of:
  - Per-row metric drift (EM, SF-Recall, EM|retr.ok)
  - Per-question transitions between adjacent rows (gained / lost / unchanged)
  - Planner routing distribution (query_type, matched_pattern, hop_count)
  - Failure-mode classification for lost cases (bridge confusion, context
    dilution, format perturbation, abstention, retrieval-miss)
  - Verifier action breakdown (did pre-validation fire? did self-correction
    iterate? did the answer change between iterations?)

Usage:
    python -X utf8 diagnose_ablation.py <run_dir>

Example:
    python -X utf8 diagnose_ablation.py \\
        evaluation_results/agentic_ablation_hotpotqa_20260526_152616

Writes a `diagnostic_report.md` into the run directory.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

ROW_ORDER = [
    "row1_llm_only",
    "row2_rag_no_agent",
    "row3_planner",
    "row4_verifier",
    "row5_self_correct",
]


def load_row(run_dir: Path, name: str) -> Dict[str, Dict[str, Any]]:
    """Load a row's JSONL by question_id."""
    path = run_dir / f"{name}.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["question_id"]] = rec
    return out


def classify_failure(rec: Dict[str, Any], baseline_rec: Optional[Dict[str, Any]] = None) -> str:
    """
    Pick the dominant failure category for a wrong-answer record.

    Categories:
        retrieval_miss      : gold paragraph was not retrieved at all
        gold_cut_by_cap     : retrieved but gold not in final LLM context
        llm_error           : LLM timeout / API error
        format_yn           : open-ended question received yes/no answer
        format_abstention   : LLM said "I don't know" or similar
        bridge_confusion    : answer is an entity that appears in baseline as bridge
        context_dilution    : retrieved alternative chunks confused the LLM
        unclassified        : none of the above
    """
    pred = (rec.get("predicted_answer") or "").strip().lower()
    gold = (rec.get("gold_answer") or "").strip().lower()
    if rec.get("llm_error"):
        return "llm_error"
    if not rec.get("all_gold_retrieved"):
        return "retrieval_miss"
    if not rec.get("gold_in_final_context", True):
        return "gold_cut_by_cap"
    # Format issues
    if pred in {"yes", "no", "yes.", "no.", "yes, they are.", "no, they are not.",
                "yes, they were.", "no, they were not."} and gold not in {"yes", "no"}:
        return "format_yn"
    if any(s in pred for s in ["i don't know", "i do not know", "cannot determine",
                                "not enough information", "unknown"]):
        return "format_abstention"
    # Bridge confusion: r3 answer is a sub-string of the question (e.g. picked
    # the bridge entity rather than the final answer)
    qtxt = (rec.get("question") or "").lower()
    if pred and pred in qtxt and pred != gold:
        return "bridge_confusion"
    # If baseline got it right, classify as context_dilution (more chunks +
    # decomposition changed the LLM's pick despite gold being available)
    if baseline_rec is not None and baseline_rec.get("exact_match"):
        return "context_dilution"
    return "unclassified"


def transition_table(prev: Dict, curr: Dict) -> Dict[str, List[str]]:
    """Categorise per-question transitions: gained/lost/stayed-correct/stayed-wrong."""
    out = {"stayed_correct": [], "stayed_wrong": [], "gained": [], "lost": []}
    qids = set(prev) & set(curr)
    for q in qids:
        a, b = prev[q].get("exact_match"), curr[q].get("exact_match")
        if a and b:        out["stayed_correct"].append(q)
        elif not a and not b: out["stayed_wrong"].append(q)
        elif not a and b:  out["gained"].append(q)
        else:              out["lost"].append(q)
    return out


def safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarise_row(name: str, recs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not recs:
        return {"row": name, "n": 0}
    n = len(recs)
    return {
        "row": name,
        "n": n,
        "em": sum(1 for r in recs.values() if r.get("exact_match")) / n,
        "sf_recall": sum(1 for r in recs.values() if r.get("all_gold_retrieved")) / n,
        "em_given_retr_ok": (
            sum(1 for r in recs.values() if r.get("all_gold_retrieved") and r.get("exact_match"))
            / max(1, sum(1 for r in recs.values() if r.get("all_gold_retrieved")))
        ),
        "avg_retrieval_count": safe_mean([r.get("retrieval_count", 0) for r in recs.values()]),
        "max_retrieval_count": max((r.get("retrieval_count", 0) for r in recs.values()), default=0),
        "avg_verifier_iters": safe_mean([r.get("verifier_iterations", 0) for r in recs.values()]),
        "max_verifier_iters": max((r.get("verifier_iterations", 0) for r in recs.values()), default=0),
        "pre_val_preempt_rate": sum(
            1 for r in recs.values() if r.get("classifier_preempt") is not None
        ) / n,
        "all_verified_rate": sum(1 for r in recs.values() if r.get("all_verified")) / n,
    }


def planner_routing_breakdown(recs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    qtype = Counter(r.get("planner_query_type", "unknown") for r in recs.values())
    pattern = Counter((r.get("matched_pattern") or "none") for r in recs.values())
    hop = Counter(r.get("hop_count", 0) for r in recs.values())
    return {"query_type": dict(qtype), "matched_pattern": dict(pattern), "hop_count": dict(hop)}


def render_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Agentic-Ablation Diagnostic Report", ""]
    lines.append(f"Source: `{report['run_dir']}`")
    lines.append("")

    # 1. Per-row summary
    lines.append("## 1. Per-row summary")
    lines.append("")
    cols = ["row", "n", "em", "sf_recall", "em_given_retr_ok",
            "avg_retrieval_count", "max_retrieval_count",
            "avg_verifier_iters", "max_verifier_iters",
            "pre_val_preempt_rate", "all_verified_rate"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for s in report["summary"]:
        cells = []
        for k in cols:
            v = s.get(k)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                if k in {"em", "sf_recall", "em_given_retr_ok",
                         "pre_val_preempt_rate", "all_verified_rate"}:
                    cells.append(f"{v*100:.1f}%" if v <= 1.0 else f"{v:.3f}")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # 2. Adjacent-row transitions
    lines.append("## 2. Per-question transitions (gained / lost / unchanged)")
    lines.append("")
    for t in report["transitions"]:
        lines.append(f"### {t['from']}  ->  {t['to']}")
        lines.append("")
        lines.append(f"- stayed correct:   **{len(t['stayed_correct'])}**")
        lines.append(f"- stayed wrong:     **{len(t['stayed_wrong'])}**")
        lines.append(f"- GAINED (wrong->right):  **{len(t['gained'])}**")
        lines.append(f"- LOST (right->wrong):    **{len(t['lost'])}**")
        net = len(t['gained']) - len(t['lost'])
        lines.append(f"- net delta:        **{net:+d}** ({net*2:+d}pp)")
        lines.append("")

    # 3. Planner routing distribution (where Planner is enabled, i.e. row3+)
    if report.get("planner_routing"):
        lines.append("## 3. Planner routing distribution")
        lines.append("")
        for row_name, info in report["planner_routing"].items():
            lines.append(f"### {row_name}")
            lines.append("")
            lines.append(f"- query_type breakdown: `{info['query_type']}`")
            lines.append(f"- matched_pattern:      `{info['matched_pattern']}`")
            lines.append(f"- hop_count:            `{info['hop_count']}`")
            lines.append("")

    # 4. Failure-mode classification for LOST cases per transition
    lines.append("## 4. Failure modes for LOST cases (right -> wrong)")
    lines.append("")
    for t in report["transitions"]:
        if not t["lost_failures"]:
            continue
        lines.append(f"### {t['from']}  ->  {t['to']}  ({len(t['lost'])} lost)")
        lines.append("")
        cat_counts = Counter(f["category"] for f in t["lost_failures"])
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}**: {count}")
        lines.append("")
        lines.append("#### Per-case detail")
        lines.append("")
        lines.append("| Category | Question (truncated) | Gold | Predicted |")
        lines.append("|---|---|---|---|")
        for f in t["lost_failures"]:
            q = (f["question"] or "")[:60].replace("|", "\\|")
            g = (f["gold"] or "")[:40].replace("|", "\\|")
            p = (f["predicted"] or "")[:60].replace("|", "\\|")
            lines.append(f"| {f['category']} | {q} | {g} | {p} |")
        lines.append("")

    # 5. Verifier-action breakdown
    lines.append("## 5. Verifier action breakdown")
    lines.append("")
    lines.append("Did pre-validation actually fire? Did self-correction iterate?")
    lines.append("If pre_val_preempt_rate is 0.0 in row4, pre-validation is dormant.")
    lines.append("If avg_verifier_iters is 1.0 in row5, self-correction never iterated past round 1.")
    lines.append("")
    for s in report["summary"]:
        if s["row"] in ("row4_verifier", "row5_self_correct"):
            lines.append(f"- **{s['row']}**: pre_val_preempt={s['pre_val_preempt_rate']*100:.1f}% "
                         f"| all_verified={s['all_verified_rate']*100:.1f}% "
                         f"| verifier_iters avg={s['avg_verifier_iters']:.2f} "
                         f"max={s['max_verifier_iters']}")
    lines.append("")

    # 6. Self-correction effect (row4 -> row5)
    if report.get("self_correction_flips"):
        flips = report["self_correction_flips"]
        lines.append("## 6. Self-correction effect (row4 -> row5)")
        lines.append("")
        lines.append(f"- questions where answer changed AT ALL: **{flips['answer_changed']}**")
        lines.append(f"- of those: wrong->right (fix):  **{flips['fixed']}**")
        lines.append(f"- of those: right->wrong (break): **{flips['broke']}**")
        lines.append(f"- of those: wrong->wrong (churn): **{flips['churn']}**")
        lines.append("")

    return "\n".join(lines)


def main(run_dir: Path) -> None:
    rows = {name: load_row(run_dir, name) for name in ROW_ORDER}
    present_rows = [n for n in ROW_ORDER if rows[n]]
    if not present_rows:
        print(f"No row JSONLs found in {run_dir}")
        return

    summary = [summarise_row(name, rows[name]) for name in present_rows]

    transitions = []
    for prev_name, curr_name in zip(present_rows, present_rows[1:]):
        t = transition_table(rows[prev_name], rows[curr_name])
        # Classify each lost case
        lost_failures = []
        for q in t["lost"]:
            cat = classify_failure(rows[curr_name][q], baseline_rec=rows[prev_name].get(q))
            lost_failures.append({
                "qid": q,
                "category": cat,
                "question": rows[curr_name][q].get("question"),
                "gold": rows[curr_name][q].get("gold_answer"),
                "predicted": rows[curr_name][q].get("predicted_answer"),
            })
        transitions.append({
            "from": prev_name, "to": curr_name,
            **t,
            "lost_failures": lost_failures,
        })

    planner_routing = {
        n: planner_routing_breakdown(rows[n])
        for n in present_rows
        if n in {"row3_planner", "row4_verifier", "row5_self_correct"}
    }

    # Self-correction effect: row4 vs row5 answer-changed counts
    sc_flips = None
    if "row4_verifier" in rows and "row5_self_correct" in rows:
        r4 = rows["row4_verifier"]
        r5 = rows["row5_self_correct"]
        common = set(r4) & set(r5)
        ans_changed = 0; fixed = 0; broke = 0; churn = 0
        for q in common:
            p4 = (r4[q].get("predicted_answer") or "").strip().lower()
            p5 = (r5[q].get("predicted_answer") or "").strip().lower()
            if p4 != p5:
                ans_changed += 1
                a4 = r4[q].get("exact_match"); a5 = r5[q].get("exact_match")
                if not a4 and a5: fixed += 1
                elif a4 and not a5: broke += 1
                else: churn += 1
        sc_flips = {"answer_changed": ans_changed, "fixed": fixed,
                    "broke": broke, "churn": churn}

    report = {
        "run_dir": str(run_dir),
        "summary": summary,
        "transitions": transitions,
        "planner_routing": planner_routing,
        "self_correction_flips": sc_flips,
    }

    out_path = run_dir / "diagnostic_report.md"
    out_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Diagnostic report written: {out_path}")

    # Print key headlines to stdout
    print()
    print("=" * 70)
    print("KEY HEADLINES")
    print("=" * 70)
    for s in summary:
        print(f"  {s['row']:<20} n={s['n']:>3}  "
              f"EM={s.get('em', 0)*100:>5.1f}%  "
              f"SF-R={s.get('sf_recall', 0)*100:>5.1f}%  "
              f"avg_chunks={s.get('avg_retrieval_count', 0):>5.1f}  "
              f"max_chunks={s.get('max_retrieval_count', 0):>3}  "
              f"v_iters_avg={s.get('avg_verifier_iters', 0):.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: diagnose_ablation.py <run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}")
        sys.exit(1)
    main(run_dir)
