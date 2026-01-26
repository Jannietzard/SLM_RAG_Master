"""
Ablation Study Module - Wissenschaftliche Evaluation für Masterarbeit

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

===============================================================================
OVERVIEW
===============================================================================

Dieses Modul führt systematische Ablationsstudien durch mit:

1. KLARE FORTSCHRITTSANZEIGE
   - Aktueller Stand (x/y Fragen, z/n Configs)
   - Geschätzte Restzeit (ETA)
   - Live-Updates pro Frage

2. DETAILLIERTE METRIKEN
   - Pro Frage: EM, F1, Retrieval-Count, Latenz
   - Pro Iteration im Agentic Loop: Claims verified/violated
   - Aggregiert: Mean, Std, Min, Max

3. WISSENSCHAFTLICHE OUTPUTS
   - JSON mit allen Rohdaten
   - CSV für statistische Analyse
   - Markdown-Report für Thesis
   - Latex-Tabellen (optional)

4. REPRODUZIERBARKEIT
   - Seed-basierte Sampling
   - Vollständige Config-Dokumentation
   - Timestamps und Versionen

===============================================================================
USAGE
===============================================================================

# Als eigenständiges Skript:
python ablation_study.py --samples 10 --datasets strategyqa

# Von benchmark_datasets.py:
from ablation_study import AblationStudy
study = AblationStudy(config)
results = study.run(datasets, samples=10)

===============================================================================
OUTPUT STRUCTURE
===============================================================================

results/
├── ablation_20260126_003000/
│   ├── config.json              # Vollständige Konfiguration
│   ├── raw_results.json         # Alle Rohdaten
│   ├── summary.csv              # Aggregierte Ergebnisse
│   ├── per_question.csv         # Ergebnisse pro Frage
│   ├── iteration_analysis.csv   # Agentic Loop Analyse
│   ├── report.md                # Markdown-Report
│   └── latex_tables.tex         # LaTeX für Thesis

===============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import statistics

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress Unified Planning credits
try:
    import unified_planning as up
    up.shortcuts.get_environment().credits_stream = None
except:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AblationConfig:
    """Konfiguration für Ablationsstudie."""
    # Study Settings
    name: str = "hybrid_retrieval_ablation"
    seed: int = 42
    
    # Ablation Configurations (vector_weight, graph_weight)
    configurations: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("vector_only", 1.0, 0.0),
        ("hybrid_80_20", 0.8, 0.2),
        ("hybrid_70_30", 0.7, 0.3),
        ("hybrid_50_50", 0.5, 0.5),
        ("hybrid_30_70", 0.3, 0.7),
        ("graph_only", 0.0, 1.0),
    ])
    
    # Output Settings
    results_dir: Path = field(default_factory=lambda: Path("results"))
    save_raw_results: bool = True
    generate_latex: bool = True
    
    # Logging
    verbose: bool = False
    log_every_n: int = 1  # Log every N questions


@dataclass
class QuestionResult:
    """Ergebnis für eine einzelne Frage."""
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    
    # Metriken
    exact_match: bool
    f1_score: float
    
    # Retrieval Info
    retrieval_count: int
    vector_results: int
    graph_results: int
    
    # Timing
    total_time_ms: float
    retrieval_time_ms: float
    llm_time_ms: float
    
    # Agentic Loop Details
    iterations_used: int
    iteration_details: List[Dict] = field(default_factory=list)
    all_verified: bool = False
    
    # Metadata
    dataset: str = ""
    config_name: str = ""
    question_type: str = ""


@dataclass
class ConfigurationResult:
    """Aggregierte Ergebnisse für eine Konfiguration."""
    config_name: str
    vector_weight: float
    graph_weight: float
    dataset: str
    
    # Sample Info
    n_questions: int
    n_success: int  # Fragen ohne Fehler
    
    # Aggregierte Metriken
    exact_match_mean: float
    exact_match_std: float
    f1_mean: float
    f1_std: float
    
    # Timing
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    
    # Agentic Loop Stats
    avg_iterations: float
    verification_rate: float  # % der Antworten vollständig verifiziert
    
    # Raw Results (für detaillierte Analyse)
    question_results: List[QuestionResult] = field(default_factory=list)


# =============================================================================
# PROGRESS DISPLAY
# =============================================================================

class ProgressTracker:
    """Übersichtliche Fortschrittsanzeige."""
    
    def __init__(self, total_questions: int, total_configs: int):
        self.total_questions = total_questions
        self.total_configs = total_configs
        self.current_question = 0
        self.current_config = 0
        self.start_time = time.time()
        self.question_times = []
        
    def start_config(self, config_name: str, config_num: int):
        """Neue Konfiguration starten."""
        self.current_config = config_num
        self.current_question = 0
        self.config_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"  CONFIG {config_num}/{self.total_configs}: {config_name}")
        print(f"{'='*70}")
    
    def update(self, question_num: int, question_id: str, result: Optional[QuestionResult] = None):
        """Update Fortschritt für eine Frage."""
        self.current_question = question_num
        elapsed = time.time() - self.start_time
        
        if result:
            self.question_times.append(result.total_time_ms / 1000)
        
        # Calculate ETA
        if self.question_times:
            avg_time = statistics.mean(self.question_times[-10:])  # Last 10 questions
            remaining_questions = (
                (self.total_configs - self.current_config) * self.total_questions +
                (self.total_questions - self.current_question)
            )
            eta_seconds = remaining_questions * avg_time
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "calculating..."
        
        # Progress bar
        progress = self.current_question / self.total_questions
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Status line
        status = f"  [{bar}] {self.current_question}/{self.total_questions}"
        
        if result:
            em = "✓" if result.exact_match else "✗"
            status += f" | {em} EM | F1={result.f1_score:.3f} | {result.total_time_ms/1000:.1f}s"
            status += f" | iter={result.iterations_used}"
        
        status += f" | ETA: {eta_str}"
        
        # Print (overwrite line)
        print(f"\r{status}", end="", flush=True)
    
    def finish_config(self, result: ConfigurationResult):
        """Konfiguration abgeschlossen."""
        print()  # New line after progress bar
        print(f"  ────────────────────────────────────────────────────────")
        print(f"  Results: EM={result.exact_match_mean*100:.1f}% | F1={result.f1_mean:.3f} | "
              f"Avg Time={result.avg_time_ms/1000:.1f}s | Avg Iter={result.avg_iterations:.1f}")
        print(f"  Verification Rate: {result.verification_rate*100:.1f}%")
    
    def finish_study(self, total_time: float):
        """Studie abgeschlossen."""
        print(f"\n{'='*70}")
        print(f"  ABLATION STUDY COMPLETE")
        print(f"  Total Time: {timedelta(seconds=int(total_time))}")
        print(f"{'='*70}\n")


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalisiere Antwort für Vergleich."""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Berechne Exact Match."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Berechne F1 Score (Token-Level)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    common = set(pred_tokens) & set(gold_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# ABLATION STUDY CLASS
# =============================================================================

class AblationStudy:
    """
    Wissenschaftliche Ablationsstudie für Hybrid Retrieval.
    
    Führt systematische Evaluation durch und generiert
    publikationsreife Ergebnisse.
    """
    
    def __init__(
        self,
        config: AblationConfig = None,
        pipeline_config: Dict = None,
    ):
        """
        Initialize Ablation Study.
        
        Args:
            config: AblationConfig für Studie
            pipeline_config: Config für RAG Pipeline (settings.yaml)
        """
        self.config = config or AblationConfig()
        self.pipeline_config = pipeline_config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results: Dict[str, Dict[str, ConfigurationResult]] = defaultdict(dict)
        
        # Create results directory
        self.run_dir = self._create_run_directory()
        
        self.logger.info(f"AblationStudy initialized. Results: {self.run_dir}")
    
    def _create_run_directory(self) -> Path:
        """Erstelle eindeutiges Verzeichnis für diesen Run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.results_dir / f"ablation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _save_config(self):
        """Speichere Konfiguration für Reproduzierbarkeit."""
        config_data = {
            "ablation_config": asdict(self.config),
            "pipeline_config": self.pipeline_config,
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
        }
        
        # Convert Path objects to strings
        config_data["ablation_config"]["results_dir"] = str(self.config.results_dir)
        
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def run(
        self,
        datasets: Dict[str, Tuple[List, List]],  # {name: (questions, articles)}
        samples_per_dataset: int = 10,
        pipeline_factory = None,  # Function to create pipeline
    ) -> Dict[str, Dict[str, ConfigurationResult]]:
        """
        Führe Ablationsstudie durch.
        
        Args:
            datasets: Dict mit {name: (questions, articles)}
            samples_per_dataset: Anzahl Fragen pro Dataset
            pipeline_factory: Factory-Funktion für Pipeline
            
        Returns:
            Nested Dict: {dataset: {config: ConfigurationResult}}
        """
        self._save_config()
        
        total_configs = len(self.config.configurations)
        total_questions = samples_per_dataset * len(datasets)
        
        progress = ProgressTracker(
            total_questions=samples_per_dataset,
            total_configs=total_configs * len(datasets)
        )
        
        study_start = time.time()
        global_config_num = 0
        
        print("\n" + "=" * 70)
        print("  ABLATION STUDY")
        print("=" * 70)
        print(f"  Datasets: {list(datasets.keys())}")
        print(f"  Samples per dataset: {samples_per_dataset}")
        print(f"  Configurations: {total_configs}")
        print(f"  Total evaluations: {total_configs * total_questions}")
        print("=" * 70)
        
        # Iterate over datasets
        for dataset_name, (questions, articles) in datasets.items():
            print(f"\n{'─'*70}")
            print(f"  DATASET: {dataset_name.upper()}")
            print(f"{'─'*70}")
            
            # Sample questions
            import random
            random.seed(self.config.seed)
            sampled_questions = random.sample(
                questions, 
                min(samples_per_dataset, len(questions))
            )
            
            # Iterate over configurations
            for config_name, vector_w, graph_w in self.config.configurations:
                global_config_num += 1
                
                progress.start_config(
                    f"{config_name} (v={vector_w}, g={graph_w})",
                    global_config_num
                )
                
                # Create pipeline with this configuration
                if pipeline_factory:
                    try:
                        pipeline = pipeline_factory(
                            vector_weight=vector_w,
                            graph_weight=graph_w,
                            dataset_name=dataset_name,
                        )
                    except Exception as e:
                        self.logger.error(f"Pipeline creation failed: {e}")
                        continue
                else:
                    self.logger.warning("No pipeline_factory provided, using dummy")
                    pipeline = None
                
                # Evaluate all questions
                question_results = []
                
                for q_num, question in enumerate(sampled_questions, 1):
                    progress.update(q_num, question.id)
                    
                    try:
                        result = self._evaluate_question(
                            question=question,
                            pipeline=pipeline,
                            config_name=config_name,
                            dataset_name=dataset_name,
                        )
                        question_results.append(result)
                        progress.update(q_num, question.id, result)
                        
                    except Exception as e:
                        self.logger.error(f"Question {question.id} failed: {e}")
                        # Create error result
                        result = QuestionResult(
                            question_id=question.id,
                            question=question.question,
                            gold_answer=question.answer,
                            predicted_answer=f"[ERROR: {str(e)[:50]}]",
                            exact_match=False,
                            f1_score=0.0,
                            retrieval_count=0,
                            vector_results=0,
                            graph_results=0,
                            total_time_ms=0,
                            retrieval_time_ms=0,
                            llm_time_ms=0,
                            iterations_used=0,
                            dataset=dataset_name,
                            config_name=config_name,
                        )
                        question_results.append(result)
                        progress.update(q_num, question.id, result)
                
                # Aggregate results
                config_result = self._aggregate_results(
                    question_results=question_results,
                    config_name=config_name,
                    vector_weight=vector_w,
                    graph_weight=graph_w,
                    dataset=dataset_name,
                )
                
                self.results[dataset_name][config_name] = config_result
                progress.finish_config(config_result)
                
                # Save intermediate results
                self._save_intermediate_results()
        
        total_time = time.time() - study_start
        progress.finish_study(total_time)
        
        # Generate final outputs
        self._generate_outputs()
        
        return self.results
    
    def _evaluate_question(
        self,
        question,
        pipeline,
        config_name: str,
        dataset_name: str,
    ) -> QuestionResult:
        """Evaluiere eine einzelne Frage."""
        
        start_time = time.time()
        
        # Default values
        predicted_answer = ""
        retrieval_count = 0
        vector_results = 0
        graph_results = 0
        retrieval_time_ms = 0
        llm_time_ms = 0
        iterations_used = 1
        iteration_details = []
        all_verified = False
        
        if pipeline is not None:
            try:
                # Call pipeline
                result = pipeline.query(question.question)
                
                # Extract results (adapt to your pipeline's return type)
                if hasattr(result, 'answer'):
                    predicted_answer = result.answer
                elif isinstance(result, dict):
                    predicted_answer = result.get('answer', str(result))
                else:
                    predicted_answer = str(result)
                
                # Extract metrics if available
                if hasattr(result, 'iterations'):
                    iterations_used = result.iterations
                if hasattr(result, 'all_verified'):
                    all_verified = result.all_verified
                if hasattr(result, 'iteration_history'):
                    iteration_details = result.iteration_history
                if hasattr(result, 'context_docs'):
                    retrieval_count = len(result.context_docs)
                if hasattr(result, 'total_time_ms'):
                    llm_time_ms = result.total_time_ms
                    
            except Exception as e:
                predicted_answer = f"[ERROR: {str(e)[:100]}]"
        else:
            predicted_answer = "[NO PIPELINE]"
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Compute metrics
        exact_match = compute_exact_match(predicted_answer, question.answer)
        f1_score = compute_f1(predicted_answer, question.answer)
        
        return QuestionResult(
            question_id=question.id,
            question=question.question,
            gold_answer=question.answer,
            predicted_answer=predicted_answer,
            exact_match=exact_match,
            f1_score=f1_score,
            retrieval_count=retrieval_count,
            vector_results=vector_results,
            graph_results=graph_results,
            total_time_ms=total_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            llm_time_ms=llm_time_ms,
            iterations_used=iterations_used,
            iteration_details=iteration_details,
            all_verified=all_verified,
            dataset=dataset_name,
            config_name=config_name,
            question_type=getattr(question, 'question_type', 'unknown'),
        )
    
    def _aggregate_results(
        self,
        question_results: List[QuestionResult],
        config_name: str,
        vector_weight: float,
        graph_weight: float,
        dataset: str,
    ) -> ConfigurationResult:
        """Aggregiere Ergebnisse für eine Konfiguration."""
        
        if not question_results:
            return ConfigurationResult(
                config_name=config_name,
                vector_weight=vector_weight,
                graph_weight=graph_weight,
                dataset=dataset,
                n_questions=0,
                n_success=0,
                exact_match_mean=0,
                exact_match_std=0,
                f1_mean=0,
                f1_std=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                avg_iterations=0,
                verification_rate=0,
            )
        
        # Filter successful results
        successful = [r for r in question_results if not r.predicted_answer.startswith("[ERROR")]
        
        # Extract metrics
        em_scores = [float(r.exact_match) for r in question_results]
        f1_scores = [r.f1_score for r in question_results]
        times = [r.total_time_ms for r in question_results]
        iterations = [r.iterations_used for r in question_results]
        verified = [r.all_verified for r in question_results]
        
        return ConfigurationResult(
            config_name=config_name,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            dataset=dataset,
            n_questions=len(question_results),
            n_success=len(successful),
            exact_match_mean=statistics.mean(em_scores) if em_scores else 0,
            exact_match_std=statistics.stdev(em_scores) if len(em_scores) > 1 else 0,
            f1_mean=statistics.mean(f1_scores) if f1_scores else 0,
            f1_std=statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0,
            avg_time_ms=statistics.mean(times) if times else 0,
            min_time_ms=min(times) if times else 0,
            max_time_ms=max(times) if times else 0,
            avg_iterations=statistics.mean(iterations) if iterations else 0,
            verification_rate=sum(verified) / len(verified) if verified else 0,
            question_results=question_results,
        )
    
    def _save_intermediate_results(self):
        """Speichere Zwischenergebnisse."""
        # Convert to serializable format
        data = {}
        for dataset, configs in self.results.items():
            data[dataset] = {}
            for config_name, result in configs.items():
                data[dataset][config_name] = {
                    "config_name": result.config_name,
                    "vector_weight": result.vector_weight,
                    "graph_weight": result.graph_weight,
                    "n_questions": result.n_questions,
                    "exact_match_mean": result.exact_match_mean,
                    "f1_mean": result.f1_mean,
                    "avg_time_ms": result.avg_time_ms,
                    "avg_iterations": result.avg_iterations,
                    "verification_rate": result.verification_rate,
                }
        
        with open(self.run_dir / "intermediate_results.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def _generate_outputs(self):
        """Generiere alle Output-Dateien."""
        self._save_raw_results()
        self._save_summary_csv()
        self._save_per_question_csv()
        self._save_iteration_analysis()
        self._generate_markdown_report()
        
        if self.config.generate_latex:
            self._generate_latex_tables()
        
        print(f"\n  Results saved to: {self.run_dir}")
    
    def _save_raw_results(self):
        """Speichere alle Rohdaten als JSON."""
        data = {}
        for dataset, configs in self.results.items():
            data[dataset] = {}
            for config_name, result in configs.items():
                data[dataset][config_name] = {
                    **asdict(result),
                    "question_results": [asdict(q) for q in result.question_results]
                }
        
        with open(self.run_dir / "raw_results.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_summary_csv(self):
        """Speichere aggregierte Ergebnisse als CSV."""
        rows = []
        for dataset, configs in self.results.items():
            for config_name, result in configs.items():
                rows.append({
                    "dataset": dataset,
                    "config": config_name,
                    "vector_weight": result.vector_weight,
                    "graph_weight": result.graph_weight,
                    "n": result.n_questions,
                    "EM_mean": f"{result.exact_match_mean:.3f}",
                    "EM_std": f"{result.exact_match_std:.3f}",
                    "F1_mean": f"{result.f1_mean:.3f}",
                    "F1_std": f"{result.f1_std:.3f}",
                    "time_ms_avg": f"{result.avg_time_ms:.0f}",
                    "iterations_avg": f"{result.avg_iterations:.2f}",
                    "verified_rate": f"{result.verification_rate:.3f}",
                })
        
        # Write CSV
        if rows:
            import csv
            with open(self.run_dir / "summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def _save_per_question_csv(self):
        """Speichere Ergebnisse pro Frage."""
        rows = []
        for dataset, configs in self.results.items():
            for config_name, result in configs.items():
                for q in result.question_results:
                    rows.append({
                        "dataset": dataset,
                        "config": config_name,
                        "question_id": q.question_id,
                        "exact_match": int(q.exact_match),
                        "f1_score": f"{q.f1_score:.3f}",
                        "time_ms": f"{q.total_time_ms:.0f}",
                        "iterations": q.iterations_used,
                        "verified": int(q.all_verified),
                        "retrieval_count": q.retrieval_count,
                    })
        
        if rows:
            import csv
            with open(self.run_dir / "per_question.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def _save_iteration_analysis(self):
        """Analysiere Agentic Loop Iterationen."""
        rows = []
        for dataset, configs in self.results.items():
            for config_name, result in configs.items():
                for q in result.question_results:
                    for i, iter_detail in enumerate(q.iteration_details, 1):
                        rows.append({
                            "dataset": dataset,
                            "config": config_name,
                            "question_id": q.question_id,
                            "iteration": i,
                            "verified_claims": len(iter_detail.get("verified", [])),
                            "violated_claims": len(iter_detail.get("violated", [])),
                            "llm_latency_ms": iter_detail.get("llm_latency_ms", 0),
                        })
        
        if rows:
            import csv
            with open(self.run_dir / "iteration_analysis.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def _generate_markdown_report(self):
        """Generiere Markdown-Report für Thesis."""
        report = []
        report.append("# Ablation Study Results\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Study:** {self.config.name}\n")
        report.append(f"**Seed:** {self.config.seed}\n\n")
        
        report.append("## Summary\n\n")
        
        for dataset, configs in self.results.items():
            report.append(f"### Dataset: {dataset}\n\n")
            report.append("| Configuration | Vector | Graph | EM (%) | F1 | Avg Time (s) | Avg Iter | Verified (%) |\n")
            report.append("|---------------|--------|-------|--------|-----|--------------|----------|-------------|\n")
            
            for config_name, result in configs.items():
                report.append(
                    f"| {config_name} | {result.vector_weight} | {result.graph_weight} | "
                    f"{result.exact_match_mean*100:.1f} | {result.f1_mean:.3f} | "
                    f"{result.avg_time_ms/1000:.1f} | {result.avg_iterations:.1f} | "
                    f"{result.verification_rate*100:.1f} |\n"
                )
            report.append("\n")
        
        # Best configurations
        report.append("## Best Configurations\n\n")
        for dataset, configs in self.results.items():
            if configs:
                best = max(configs.values(), key=lambda x: x.f1_mean)
                report.append(f"- **{dataset}**: {best.config_name} (F1={best.f1_mean:.3f})\n")
        
        with open(self.run_dir / "report.md", "w") as f:
            f.write("".join(report))
    
    def _generate_latex_tables(self):
        """Generiere LaTeX-Tabellen für Thesis."""
        latex = []
        latex.append("% Ablation Study Results - Auto-generated\n")
        latex.append("% Include with: \\input{latex_tables.tex}\n\n")
        
        for dataset, configs in self.results.items():
            latex.append(f"% Dataset: {dataset}\n")
            latex.append("\\begin{table}[h]\n")
            latex.append("\\centering\n")
            latex.append(f"\\caption{{Ablation results for {dataset}}}\n")
            latex.append("\\begin{tabular}{lcccccc}\n")
            latex.append("\\toprule\n")
            latex.append("Configuration & $w_v$ & $w_g$ & EM (\\%) & F1 & Time (s) & Iter \\\\\n")
            latex.append("\\midrule\n")
            
            for config_name, result in configs.items():
                latex.append(
                    f"{config_name.replace('_', '\\_')} & "
                    f"{result.vector_weight} & {result.graph_weight} & "
                    f"{result.exact_match_mean*100:.1f} & {result.f1_mean:.3f} & "
                    f"{result.avg_time_ms/1000:.1f} & {result.avg_iterations:.1f} \\\\\n"
                )
            
            latex.append("\\bottomrule\n")
            latex.append("\\end{tabular}\n")
            latex.append(f"\\label{{tab:ablation_{dataset}}}\n")
            latex.append("\\end{table}\n\n")
        
        with open(self.run_dir / "latex_tables.tex", "w") as f:
            f.write("".join(latex))


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="Scientific Ablation Study for Hybrid Retrieval"
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Number of samples per dataset (default: 10)"
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["strategyqa"],
        help="Datasets to evaluate (default: strategyqa)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for results (default: results/)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("\n" + "=" * 70)
    print("  ABLATION STUDY - Hybrid Retrieval Evaluation")
    print("=" * 70)
    print(f"  Datasets: {args.datasets}")
    print(f"  Samples: {args.samples}")
    print(f"  Seed: {args.seed}")
    print(f"  Results: {args.results_dir}")
    print("=" * 70 + "\n")
    
    # Create config
    config = AblationConfig(
        seed=args.seed,
        results_dir=args.results_dir,
        verbose=args.verbose,
    )
    
    # Initialize study
    study = AblationStudy(config=config)
    
    # Load datasets (placeholder - integrate with your benchmark_datasets.py)
    print("Loading datasets...")
    print("NOTE: Run from benchmark_datasets.py for full integration")
    print("      python benchmark_datasets.py ablation --samples 10")
    
    # Example usage:
    # from benchmark_datasets import load_dataset_questions
    # datasets = {name: load_dataset_questions(name) for name in args.datasets}
    # results = study.run(datasets, samples_per_dataset=args.samples)


if __name__ == "__main__":
    main()