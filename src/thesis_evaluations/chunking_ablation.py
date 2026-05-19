"""
Chunking ablation — Option-A measurement for thesis component #4.

What it does
------------
For each chunking configuration (sentences_per_chunk × sentence_overlap),
this script:

  1. Re-chunks the dataset's source articles via SpacySentenceChunker with
     the configured window/overlap.
  2. Re-ingests the resulting documents into a **dedicated, per-config
     vector store** (LanceDB). The KuzuDB graph store is reused from the
     existing ingestion — only the chunking-affected layer (vector store)
     is rebuilt per config.
  3. Runs the pipeline in **retrieval-only mode** on the dataset's
     questions, measuring supporting-fact F1, SF-Recall, and EM|retrieval.
  4. Writes a per-config JSONL of question-level metrics plus a summary
     table.

The output answers a single thesis-section question: how sensitive is
retrieval recall to the chunking window size and overlap?

Reproducibility
---------------
The script is fully deterministic given:
  - the same source articles (data/<dataset>/chunks_export.json input
    side: the raw articles, not the pre-chunked file),
  - the same Ollama embedding model + version,
  - the same retrieval config (rrf_k, top_k, etc.).

It does NOT re-run Phase 2 (GLiNER NER + REBEL RE) because those operate
per-chunk and would invalidate the existing extraction_results.json. To
keep the graph fixed across all configs, MENTIONS/RELATED_TO edges from
the old chunking are reused. **The graph is therefore HELD CONSTANT in
this ablation**; only the vector-retrieval side of the hybrid system
varies. This is the only fair comparison given a fixed Phase-2 output —
see the methodology paragraph at the bottom of this docstring for the
honest framing required in the thesis text.

Usage
-----
    python -X utf8 -m src.thesis_evaluations.chunking_ablation \
        --dataset hotpotqa \
        --samples 100 \
        --configs "3:1,5:1,7:1,3:0,5:2"

    --configs is a comma-separated list of "<sentences>:<overlap>" pairs.
    Default: "3:1,5:1,7:1" (the three primary cells).

Outputs
-------
    evaluation_results/chunking_ablation_<timestamp>/
        config_s3_o1.jsonl     # per-question metrics for window=3, overlap=1
        config_s5_o1.jsonl
        ...
        summary.csv
        summary.md

Honest reporting
----------------
The methodology section of the thesis MUST state:
  (a) The graph-side of the hybrid retrieval is held constant across
      configs (the Phase-2 NER+RE outputs are reused).
  (b) The ablation therefore measures the vector-retrieval contribution
      of chunking, not the full system end-to-end response.
  (c) Configurations are not exhaustive; they probe one dimension
      (window size, overlap) of the chunking hyperparameter space.

References
----------
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-
  Intensive NLP Tasks." NeurIPS 2020. (RAG framing.)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    LOADERS,
    StoreManager,
    create_langchain_documents,
    create_pipeline,
    evaluate_dataset,
    load_config_file,
    run_ingestion,
)


# ────────────────────────────────────────────────────────────────────────────
# PATH-OVERRIDING STORE MANAGER
# ────────────────────────────────────────────────────────────────────────────

class _AblationStoreManager(StoreManager):
    """StoreManager subclass that returns a custom vector-store path while
    keeping all other dataset paths (graph, questions, articles_info)
    pointing at the production locations.

    Used to direct `create_pipeline` at the per-config vector store this
    ablation builds, without disturbing the rest of the dataset layout.
    """

    def __init__(self, vector_override: Path):
        super().__init__()
        self._vector_override = Path(vector_override)

    def get_paths(self, dataset: str) -> Dict[str, Path]:
        paths = super().get_paths(dataset)
        paths["vector"] = self._vector_override
        return paths

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ────────────────────────────────────────────────────────────────────────────
# CONFIG GRID
# ────────────────────────────────────────────────────────────────────────────

# Default ablation grid. The three primary cells (3-1, 5-1, 7-1) probe
# window size at constant overlap. The two extra cells (3-0, 5-2) probe
# overlap at constant window. Five configurations × ~50-100 questions
# fits in a single afternoon of wall-clock on edge hardware.
DEFAULT_CONFIGS: List[Tuple[int, int]] = [
    (3, 1),  # current production default — the baseline of this ablation
    (5, 1),
    (7, 1),
    (3, 0),
    (5, 2),
]


def parse_configs(spec: str) -> List[Tuple[int, int]]:
    """Parse "3:1,5:1,7:1" → [(3,1), (5,1), (7,1)] with validation."""
    out: List[Tuple[int, int]] = []
    for cell in spec.split(","):
        cell = cell.strip()
        if not cell:
            continue
        try:
            s, o = cell.split(":")
            s_int, o_int = int(s), int(o)
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Bad config cell {cell!r}; expected 'sentences:overlap'"
            ) from exc
        if s_int < 1:
            raise ValueError(f"sentences_per_chunk must be >= 1; got {s_int}")
        if o_int < 0:
            raise ValueError(f"sentence_overlap must be >= 0; got {o_int}")
        if o_int >= s_int:
            raise ValueError(
                f"overlap ({o_int}) must be < window ({s_int}); "
                f"otherwise no progress is made between chunks"
            )
        out.append((s_int, o_int))
    if not out:
        raise ValueError("No valid configs supplied")
    return out


# ────────────────────────────────────────────────────────────────────────────
# PER-CONFIG RUN
# ────────────────────────────────────────────────────────────────────────────

def run_one_config(
    sentences: int,
    overlap: int,
    dataset: str,
    config: Dict,
    store_manager: StoreManager,
    questions: List,
    jsonl_path: Path,
    ablation_workdir: Path,
    apply_coreference: bool,
) -> Dict[str, float]:
    """Run a single chunking configuration end-to-end.

    Returns a dict of headline metrics so the caller can build the summary
    table without re-reading the JSONL.
    """
    tag = f"s{sentences}_o{overlap}"
    logger.info("=" * 70)
    logger.info(f"CONFIG: sentences_per_chunk={sentences}, sentence_overlap={overlap}")
    logger.info("=" * 70)

    # ── Step 1: load source articles via the dataset's loader ─────────────
    # LOADERS[dataset].load() returns (articles, questions); we discard the
    # latter since we'll reuse the previously-saved questions.json.
    if dataset not in LOADERS:
        raise RuntimeError(f"Unknown dataset: {dataset}")
    loader = LOADERS[dataset]
    articles, _ = loader.load(n_samples=None)  # load all articles
    if not articles:
        raise RuntimeError(
            f"No articles found for {dataset}. Ensure the dataset's source "
            f"corpus is present at the loader's expected location."
        )
    logger.info(f"Loaded {len(articles)} source articles")

    # ── Step 2: re-chunk with this config ─────────────────────────────────
    documents = create_langchain_documents(
        articles,
        chunk_sentences=sentences,
        sentence_overlap=overlap,
        apply_coreference=apply_coreference,
    )
    logger.info(f"Created {len(documents)} chunks (window={sentences}, overlap={overlap})")

    # ── Step 3: re-ingest into a dedicated per-config vector store ────────
    vector_path = ablation_workdir / tag / "vector"
    # KuzuDB graph store: reused unchanged from the dataset's main ingestion.
    # Only the vector store is rebuilt per config — see module docstring.
    graph_path = store_manager.get_paths(dataset)["graph"]
    if vector_path.exists():
        logger.info(f"Clearing previous vector store at {vector_path}")
        shutil.rmtree(vector_path)
    vector_path.parent.mkdir(parents=True, exist_ok=True)

    t_ingest = time.time()
    run_ingestion(documents, vector_path, graph_path, config, dataset)
    ingest_seconds = time.time() - t_ingest
    logger.info(f"Ingest took {ingest_seconds:.0f}s")

    # ── Step 4: build a pipeline pointing at the per-config vector store ──
    # Use an _AblationStoreManager subclass that overrides only the vector
    # path; graph store and other dataset files come from the production
    # location, so the graph side of the hybrid retrieval is held constant.
    ablation_store = _AblationStoreManager(vector_override=vector_path)

    pipeline = create_pipeline(
        dataset, config, ablation_store,
        vector_weight=1.0,
        graph_weight=1.0,
        model_name="qwen2:1.5b",  # any model — retrieval-only mode skips it
        enable_planner=True,
        enable_verifier=False,    # retrieval-only
        max_iterations=1,
    )

    # ── Step 5: evaluate (retrieval-only) ─────────────────────────────────
    t_eval = time.time()
    result = evaluate_dataset(
        dataset=dataset,
        questions=questions,
        pipeline=pipeline,
        config_name=tag,
        vector_weight=1.0,
        graph_weight=1.0,
        jsonl_out=jsonl_path,
        retrieval_only=True,
    )
    eval_seconds = time.time() - t_eval
    logger.info(f"Eval took {eval_seconds:.0f}s")

    return {
        "sentences": sentences,
        "overlap": overlap,
        "n_chunks": len(documents),
        "n_questions": len(questions),
        "sf_f1": result.avg_sf_f1,
        "sf_recall_rate": result.sf_recall_rate,
        "ingest_seconds": ingest_seconds,
        "eval_seconds": eval_seconds,
    }


# ────────────────────────────────────────────────────────────────────────────
# SUMMARY WRITER
# ────────────────────────────────────────────────────────────────────────────

def write_summary(
    results: List[Dict[str, float]],
    out_dir: Path,
    baseline_tag: str = "s3_o1",
) -> None:
    """Write summary.csv and summary.md.

    summary.md is the table that goes into the thesis methodology section.
    All deltas are relative to the baseline (default: s3_o1, the production
    default).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sentences", "overlap", "n_chunks", "n_questions",
            "sf_f1", "sf_recall_rate", "ingest_seconds", "eval_seconds",
        ])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Wrote {csv_path}")

    # Markdown table — find baseline
    baseline = next(
        (r for r in results if f"s{r['sentences']}_o{r['overlap']}" == baseline_tag),
        results[0],
    )

    lines = ["# Chunking Ablation Results", ""]
    lines.append(f"Baseline: window={baseline['sentences']} sentences, overlap={baseline['overlap']}.")
    lines.append("")
    lines.append(
        "| Window | Overlap | Chunks | SF-F1 | SF-Recall | Δ SF-F1 | Δ SF-Recall |"
    )
    lines.append(
        "|-------:|--------:|-------:|------:|----------:|--------:|------------:|"
    )
    for r in results:
        d_f1 = r["sf_f1"] - baseline["sf_f1"]
        d_rec = r["sf_recall_rate"] - baseline["sf_recall_rate"]
        is_baseline = (r["sentences"] == baseline["sentences"]
                       and r["overlap"] == baseline["overlap"])
        marker = " *(baseline)*" if is_baseline else ""
        lines.append(
            f"| {r['sentences']} | {r['overlap']} | {r['n_chunks']:,} | "
            f"{r['sf_f1']:.3f} | {r['sf_recall_rate']:.2%} | "
            f"{d_f1:+.3f}{marker} | {d_rec:+.2%} |"
        )
    lines.append("")
    lines.append("Δ columns: difference from baseline. Positive = improvement.")
    lines.append("")
    lines.append("## Methodology footnote")
    lines.append("")
    lines.append(
        "This ablation varies one hyperparameter pair (sentences-per-chunk, "
        "sentence-overlap) of the SpacySentenceChunker while holding the rest "
        "of the system constant. The knowledge-graph component (Phase 2 NER+RE "
        "outputs and the KuzuDB store) is reused from the production "
        "ingestion across all configurations; only the vector-retrieval layer "
        "(LanceDB) is rebuilt per config. Evaluation is retrieval-only: the "
        "Verifier (S_V) is disabled and SF-F1 / SF-Recall against gold "
        "supporting-paragraph titles is the headline metric."
    )

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {md_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunking ablation (Option A) — measures retrieval "
                    "recall sensitivity to chunk window / overlap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default="hotpotqa",
                        help="Dataset name (default: hotpotqa)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of questions to evaluate per config "
                             "(default: 100; use 500 for the final run).")
    parser.add_argument("--configs", default="3:1,5:1,7:1",
                        help="Comma-separated 'sentences:overlap' cells "
                             "(default: '3:1,5:1,7:1'). Use 'all' for the "
                             "full grid 3:1,5:1,7:1,3:0,5:2.")
    parser.add_argument("--no-coreference", action="store_true",
                        help="Disable coreference resolution before chunking. "
                             "Holds coref off across all configs so only the "
                             "chunking dimension varies.")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="Working directory for per-config vector stores. "
                             "Default: evaluation_results/chunking_ablation_<ts>/")
    args = parser.parse_args()

    if args.configs.strip().lower() == "all":
        configs = DEFAULT_CONFIGS
    else:
        configs = parse_configs(args.configs)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.workdir or (
        _PROJECT_ROOT / "evaluation_results" / f"chunking_ablation_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # ── Setup ─────────────────────────────────────────────────────────────
    config = load_config_file()
    store_manager = StoreManager()

    if not store_manager.dataset_exists(args.dataset):
        logger.error(f"Dataset not ingested: {args.dataset}")
        logger.error(
            "Run `python -m src.thesis_evaluations.benchmark_datasets ingest "
            f"--dataset {args.dataset}` first."
        )
        sys.exit(1)

    questions = store_manager.load_questions(args.dataset)
    if args.samples:
        questions = questions[: args.samples]
    logger.info(f"Loaded {len(questions)} questions for {args.dataset}")

    apply_coref = not args.no_coreference

    # ── Run each config ───────────────────────────────────────────────────
    results: List[Dict[str, float]] = []
    for sentences, overlap in configs:
        tag = f"s{sentences}_o{overlap}"
        jsonl_path = out_dir / f"config_{tag}.jsonl"
        try:
            r = run_one_config(
                sentences=sentences,
                overlap=overlap,
                dataset=args.dataset,
                config=config,
                store_manager=store_manager,
                questions=questions,
                jsonl_path=jsonl_path,
                ablation_workdir=out_dir,
                apply_coreference=apply_coref,
            )
            results.append(r)
        except Exception as exc:
            logger.exception(f"Config {tag} failed: {exc}")
            results.append({
                "sentences": sentences,
                "overlap": overlap,
                "n_chunks": 0,
                "n_questions": len(questions),
                "sf_f1": float("nan"),
                "sf_recall_rate": float("nan"),
                "ingest_seconds": 0.0,
                "eval_seconds": 0.0,
            })

    # ── Write summary ─────────────────────────────────────────────────────
    write_summary(results, out_dir)
    logger.info("Done.")
    logger.info(f"See {out_dir}/summary.md for the table.")


if __name__ == "__main__":
    main()
