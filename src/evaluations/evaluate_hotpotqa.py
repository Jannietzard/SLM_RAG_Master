"""
evaluate_hotpotqa.py - Benchmark Evaluation on HotpotQA

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

This script:
1. Loads test questions from ingestion
2. Runs your Agentic RAG pipeline
3. Computes standard metrics (Exact Match, F1, etc.)
4. Saves results for thesis

Usage:
    python evaluate_hotpotqa.py                    # Run full evaluation
    python evaluate_hotpotqa.py --samples 50      # Quick test (50 questions)
    python evaluate_hotpotqa.py --mode vector     # Test vector-only
    python evaluate_hotpotqa.py --mode hybrid     # Test hybrid
    python evaluate_hotpotqa.py --compare         # Compare all modes
"""

import argparse
import json
import logging
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

import yaml


# ============================================================================
# SETUP
# ============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # Reduce noise from other modules
    logging.getLogger("src.data_layer").setLevel(logging.WARNING)
    logging.getLogger("src.logic_layer").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


# ============================================================================
# METRICS
# ============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def compute_exact_match(prediction: str, gold: str) -> bool:
    """Compute Exact Match (EM) score."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    
    # Exact match
    if pred_norm == gold_norm:
        return True
    
    # Gold answer contained in prediction
    if gold_norm in pred_norm:
        return True
    
    return False


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gold_tokens)
    
    if not common:
        return 0.0
    
    # Count occurrences for proper F1
    num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EvalResult:
    """Single evaluation result."""
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    exact_match: bool
    f1_score: float
    retrieval_count: int
    total_time_ms: float
    question_type: str
    level: str


@dataclass
class EvalSummary:
    """Aggregated evaluation metrics."""
    timestamp: str
    config_name: str
    total_questions: int
    exact_match_rate: float
    avg_f1: float
    avg_time_ms: float
    coverage: float  # % with retrieved docs
    
    # By question type
    em_by_type: Dict[str, float] = field(default_factory=dict)
    f1_by_type: Dict[str, float] = field(default_factory=dict)
    
    # By difficulty level
    em_by_level: Dict[str, float] = field(default_factory=dict)
    f1_by_level: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# EVALUATION
# ============================================================================

class HotpotQAEvaluator:
    """Evaluate your RAG system on HotpotQA."""
    
    def __init__(
        self,
        config_path: Path = Path("./config/settings.yaml"),
        logger: logging.Logger = None
    ):
        self.config_path = config_path
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline = None
        self.config = None
        
    def setup_pipeline(self, retrieval_mode: str = "hybrid") -> None:
        """Initialize the RAG pipeline."""
        self.logger.info(f"Setting up pipeline (mode={retrieval_mode})...")
        
        # Load and modify config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Set retrieval mode
        self.config["rag"]["retrieval_mode"] = retrieval_mode
        
        # Import and initialize
        from main_agentic import AgenticRAGPipeline, setup_logging as setup_main_logging
        
        main_logger = setup_main_logging("WARNING")  # Reduce noise
        self.pipeline = AgenticRAGPipeline(self.config, main_logger)
        self.pipeline.setup()
        
        self.logger.info("Pipeline ready")
    
    def load_questions(
        self,
        questions_path: Path = Path("./data/hotpotqa_test_questions.json"),
        n_samples: int = None
    ) -> List[Dict]:
        """Load test questions."""
        if not questions_path.exists():
            self.logger.error(f"Questions file not found: {questions_path}")
            self.logger.error("Run 'python ingest_hotpotqa.py' first!")
            sys.exit(1)
        
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        if n_samples:
            questions = questions[:n_samples]
        
        self.logger.info(f"Loaded {len(questions)} test questions")
        return questions
    
    def evaluate_single(self, question: Dict) -> EvalResult:
        """Evaluate a single question."""
        q_text = question["question"]
        gold = question["answer"]
        
        # Run pipeline
        start_time = time.time()
        result = self.pipeline.query(q_text)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Extract answer
        predicted = result.answer
        
        # Compute metrics
        em = compute_exact_match(predicted, gold)
        f1 = compute_f1(predicted, gold)
        
        return EvalResult(
            question_id=question["id"],
            question=q_text,
            gold_answer=gold,
            predicted_answer=predicted,
            exact_match=em,
            f1_score=f1,
            retrieval_count=len(result.context_docs),
            total_time_ms=elapsed_ms,
            question_type=question.get("question_type", "unknown"),
            level=question.get("level", "unknown"),
        )
    
    def run_evaluation(
        self,
        questions: List[Dict],
        save_results: bool = True
    ) -> EvalSummary:
        """Run full evaluation."""
        results = []
        
        self.logger.info(f"\nEvaluating {len(questions)} questions...\n")
        
        for i, q in enumerate(questions):
            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                self.logger.info(f"Progress: {i+1}/{len(questions)}")
            
            try:
                result = self.evaluate_single(q)
                results.append(result)
                
                # Show sample results
                if i < 3:
                    self.logger.info(f"  Q: {q['question'][:50]}...")
                    self.logger.info(f"  Gold: {q['answer']}")
                    self.logger.info(f"  Pred: {result.predicted_answer[:100]}...")
                    self.logger.info(f"  EM={result.exact_match}, F1={result.f1_score:.3f}")
                    self.logger.info("")
                    
            except Exception as e:
                self.logger.warning(f"Error on question {i}: {e}")
                continue
        
        # Compute summary
        summary = self._compute_summary(results)
        
        # Save results
        if save_results:
            self._save_results(results, summary)
        
        return summary
    
    def _compute_summary(self, results: List[EvalResult]) -> EvalSummary:
        """Compute aggregated metrics."""
        if not results:
            return EvalSummary(
                timestamp=datetime.now().isoformat(),
                config_name="unknown",
                total_questions=0,
                exact_match_rate=0,
                avg_f1=0,
                avg_time_ms=0,
                coverage=0,
            )
        
        # Overall metrics
        em_rate = sum(1 for r in results if r.exact_match) / len(results)
        avg_f1 = sum(r.f1_score for r in results) / len(results)
        avg_time = sum(r.total_time_ms for r in results) / len(results)
        coverage = sum(1 for r in results if r.retrieval_count > 0) / len(results)
        
        # By question type
        em_by_type = {}
        f1_by_type = {}
        for qtype in set(r.question_type for r in results):
            type_results = [r for r in results if r.question_type == qtype]
            em_by_type[qtype] = sum(1 for r in type_results if r.exact_match) / len(type_results)
            f1_by_type[qtype] = sum(r.f1_score for r in type_results) / len(type_results)
        
        # By level
        em_by_level = {}
        f1_by_level = {}
        for level in set(r.level for r in results):
            level_results = [r for r in results if r.level == level]
            em_by_level[level] = sum(1 for r in level_results if r.exact_match) / len(level_results)
            f1_by_level[level] = sum(r.f1_score for r in level_results) / len(level_results)
        
        return EvalSummary(
            timestamp=datetime.now().isoformat(),
            config_name=self.config.get("rag", {}).get("retrieval_mode", "unknown"),
            total_questions=len(results),
            exact_match_rate=em_rate,
            avg_f1=avg_f1,
            avg_time_ms=avg_time,
            coverage=coverage,
            em_by_type=em_by_type,
            f1_by_type=f1_by_type,
            em_by_level=em_by_level,
            f1_by_level=f1_by_level,
        )
    
    def _save_results(
        self,
        results: List[EvalResult],
        summary: EvalSummary
    ) -> None:
        """Save evaluation results."""
        output_dir = Path("./evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = summary.config_name
        
        # Save detailed results
        results_path = output_dir / f"hotpotqa_{mode}_{timestamp}_details.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_path = output_dir / f"hotpotqa_{mode}_{timestamp}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2)
        
        self.logger.info(f"\nResults saved to {output_dir}/")
    
    def print_summary(self, summary: EvalSummary) -> None:
        """Print formatted summary."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\nConfig: {summary.config_name}")
        print(f"Questions: {summary.total_questions}")
        print(f"Timestamp: {summary.timestamp}")
        
        print("\n--- OVERALL METRICS ---")
        print(f"  Exact Match:  {summary.exact_match_rate:.2%}")
        print(f"  F1 Score:     {summary.avg_f1:.3f}")
        print(f"  Coverage:     {summary.coverage:.2%}")
        print(f"  Avg Time:     {summary.avg_time_ms:.0f}ms")
        
        if summary.em_by_type:
            print("\n--- BY QUESTION TYPE ---")
            for qtype, em in summary.em_by_type.items():
                f1 = summary.f1_by_type.get(qtype, 0)
                print(f"  {qtype:12s}: EM={em:.2%}, F1={f1:.3f}")
        
        if summary.em_by_level:
            print("\n--- BY DIFFICULTY ---")
            for level, em in summary.em_by_level.items():
                f1 = summary.f1_by_level.get(level, 0)
                print(f"  {level:12s}: EM={em:.2%}, F1={f1:.3f}")
        
        print("="*70 + "\n")


# ============================================================================
# COMPARISON MODE
# ============================================================================

def run_comparison(
    evaluator: HotpotQAEvaluator,
    questions: List[Dict],
    modes: List[str] = ["vector", "hybrid"]
) -> Dict[str, EvalSummary]:
    """Compare different retrieval modes."""
    results = {}
    
    for mode in modes:
        print(f"\n{'='*70}")
        print(f"EVALUATING MODE: {mode.upper()}")
        print(f"{'='*70}\n")
        
        evaluator.setup_pipeline(retrieval_mode=mode)
        summary = evaluator.run_evaluation(questions)
        evaluator.print_summary(summary)
        results[mode] = summary
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"\n{'Mode':<12} {'EM':>10} {'F1':>10} {'Time (ms)':>12} {'Coverage':>10}")
    print("-"*60)
    
    for mode, summary in results.items():
        print(f"{mode:<12} {summary.exact_match_rate:>10.2%} {summary.avg_f1:>10.3f} "
              f"{summary.avg_time_ms:>12.0f} {summary.coverage:>10.2%}")
    
    print("="*70 + "\n")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system on HotpotQA"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["vector", "graph", "hybrid"],
        help="Retrieval mode (default: hybrid)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare vector vs hybrid modes"
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("./data/hotpotqa_test_questions.json"),
        help="Path to test questions JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("INFO")
    
    print("\n" + "="*70)
    print("HOTPOTQA BENCHMARK EVALUATION")
    print("="*70)
    print(f"Mode: {'COMPARISON' if args.compare else args.mode.upper()}")
    print(f"Samples: {args.samples or 'ALL'}")
    print("="*70 + "\n")
    
    # Create evaluator
    evaluator = HotpotQAEvaluator(logger=logger)
    
    # Load questions
    questions = evaluator.load_questions(
        questions_path=args.questions_file,
        n_samples=args.samples
    )
    
    if args.compare:
        # Compare modes
        run_comparison(evaluator, questions, modes=["vector", "hybrid"])
    else:
        # Single mode evaluation
        evaluator.setup_pipeline(retrieval_mode=args.mode)
        summary = evaluator.run_evaluation(questions)
        evaluator.print_summary(summary)
    
    print("\nEvaluation complete!")
    print("Results saved to ./evaluation_results/")


if __name__ == "__main__":
    main()