"""
benchmark_datasets.py - Multi-Dataset Benchmark System with Separate Stores

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

WICHTIG - Wissenschaftlich korrekte Evaluation:
═══════════════════════════════════════════════════════════════════════
Jedes Dataset hat seinen EIGENEN Vector Store + Knowledge Graph.
Bei Evaluation wird NUR der entsprechende Store verwendet.
→ Kein Cross-Dataset Data Leakage!
═══════════════════════════════════════════════════════════════════════

Datenstruktur:
    data/
    ├── hotpotqa/
    │   ├── vector_db/
    │   ├── knowledge_graph
    │   └── questions.json
    ├── 2wikimultihop/
    │   ├── vector_db/
    │   ├── knowledge_graph
    │   └── questions.json
    └── strategyqa/
        ├── vector_db/
        ├── knowledge_graph
        └── questions.json

Usage:
    # Install dependencies
    pip install datasets

    # Ingest einzelnes Dataset
    python benchmark_datasets.py ingest --dataset hotpotqa --samples 500
    
    # Ingest alle Datasets (separate Stores)
    python benchmark_datasets.py ingest --dataset all --samples 500
    
    # Evaluate einzelnes Dataset
    python benchmark_datasets.py evaluate --dataset hotpotqa --samples 100
    
    # Volle Ablation Study (alle Datasets, alle Configs)
    python benchmark_datasets.py ablation --samples 100
    
    # Nur bestimmte Datasets in Ablation
    python benchmark_datasets.py ablation --dataset hotpotqa,strategyqa --samples 100
"""

import argparse
import json
import logging
import sys
import time
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from abc import ABC, abstractmethod

import yaml
try:
    from src.data_layer.ingestion import (
        DocumentIngestionPipeline, 
        IngestionConfig,
        load_ingestion_config,
        create_pipeline,
    )
    INGESTION_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Ingestion module not available: {e}")
    INGESTION_AVAILABLE = False

# Nach den anderen Imports:
try:
    from ablation_study import AblationStudy, AblationConfig
    ABLATION_MODULE_AVAILABLE = True
except ImportError:
    try:
        from src.evaluations.ablation_study import AblationStudy, AblationConfig
        ABLATION_MODULE_AVAILABLE = True
    except ImportError:
        ABLATION_MODULE_AVAILABLE = False
        print("[WARNING] Ablation module not found. Using legacy ablation.")

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(level: str = "INFO", quiet_modules: bool = True) -> logging.Logger:
    """Setup logging with optional quiet mode for sub-modules."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if quiet_modules:
        logging.getLogger("src.data_layer").setLevel(logging.WARNING)
        logging.getLogger("src.logic_layer").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# CONSTANTS
# ============================================================================

AVAILABLE_DATASETS = ["hotpotqa", "2wikimultihop", "strategyqa"]

# Ablation configurations: (name, vector_weight, graph_weight)
ABLATION_CONFIGS = [
    ("vector_only", 1.0, 0.0),
    ("hybrid_80_20", 0.8, 0.2),
    ("hybrid_70_30", 0.7, 0.3),
    ("hybrid_50_50", 0.5, 0.5),
    ("hybrid_30_70", 0.3, 0.7),
    ("graph_only", 0.0, 1.0),
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TestQuestion:
    """Universal test question format."""
    id: str
    question: str
    answer: str
    dataset: str
    question_type: str = "unknown"
    level: str = "unknown"
    supporting_facts: List = field(default_factory=list)


@dataclass
class Article:
    """Universal article/document format."""
    id: str
    title: str
    text: str
    sentences: List[str]
    dataset: str


@dataclass
class EvalResult:
    """Single question evaluation result."""
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    exact_match: bool
    f1_score: float
    retrieval_count: int
    time_ms: float
    dataset: str
    question_type: str


@dataclass
class ConfigResult:
    """Results for one configuration on one dataset."""
    dataset: str
    config_name: str
    vector_weight: float
    graph_weight: float
    n_questions: int
    exact_match: float
    f1_score: float
    avg_time_ms: float
    coverage: float
    by_type: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class AblationResults:
    """Complete ablation study results."""
    timestamp: str
    datasets: List[str]
    configs: List[str]
    results: Dict[str, List[ConfigResult]]  # dataset -> [ConfigResult, ...]
    
    def to_dict(self) -> Dict:
        output = {
            "timestamp": self.timestamp,
            "datasets": self.datasets,
            "configs": self.configs,
            "results": {}
        }
        for ds, results in self.results.items():
            output["results"][ds] = [asdict(r) for r in results]
        return output


# ============================================================================
# DATASET LOADERS
# ============================================================================

class DatasetLoader(ABC):
    """Abstract base for dataset loaders."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        pass


class HotpotQALoader(DatasetLoader):
    """HotpotQA: Multi-hop reasoning over Wikipedia."""
    
    @property
    def name(self) -> str:
        return "hotpotqa"
    
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        from datasets import load_dataset
        
        logger.info("Loading HotpotQA from HuggingFace...")
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        
        articles_dict = {}
        questions = []
        
        for idx, item in enumerate(ds):
            if idx % 100 == 0 and idx > 0:
                logger.info(f"  Processing {idx}/{len(ds)}...")
            
            # Question
            q = TestQuestion(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                dataset="hotpotqa",
                question_type=item["type"],  # 'bridge' or 'comparison'
                level=item["level"],         # 'easy', 'medium', 'hard'
                supporting_facts=list(zip(
                    item["supporting_facts"]["title"],
                    item["supporting_facts"]["sent_id"]
                )),
            )
            questions.append(q)
            
            # Articles from context
            for title, sentences in zip(
                item["context"]["title"],
                item["context"]["sentences"]
            ):
                if title not in articles_dict:
                    articles_dict[title] = Article(
                        id=f"hotpotqa_{len(articles_dict)}",
                        title=title,
                        text=" ".join(sentences),
                        sentences=list(sentences),
                        dataset="hotpotqa",
                    )
        
        articles = list(articles_dict.values())
        logger.info(f"  HotpotQA: {len(articles)} articles, {len(questions)} questions")
        
        return articles, questions


class WikiMultiHopLoader(DatasetLoader):
    """
    2WikiMultiHopQA: Requires 2 Wikipedia articles.
    
    FIX: framolfese/2WikiMultihopQA hat die GLEICHE Struktur wie HotpotQA:
         context = {'title': [...], 'sentences': [[...], ...]}
    """
    
    @property
    def name(self) -> str:
        return "2wikimultihop"
    
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        from datasets import load_dataset
        
        logger.info("Loading 2WikiMultiHopQA from HuggingFace...")
        
        try:
            ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
        except Exception as e:
            logger.error(f"2WikiMultiHopQA not available: {e}")
            logger.info("  Try: pip install datasets --upgrade")
            return [], []
        
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        
        articles_dict = {}
        questions = []
        
        for idx, item in enumerate(ds):
            if idx % 100 == 0 and idx > 0:
                logger.info(f"  Processing {idx}/{len(ds)}...")
            
            # Question
            q = TestQuestion(
                id=item.get("id", f"2wiki_{idx}"),
                question=item["question"],
                answer=item["answer"],
                dataset="2wikimultihop",
                question_type=item.get("type", "unknown"),
                supporting_facts=list(zip(
                    item.get("supporting_facts", {}).get("title", []),
                    item.get("supporting_facts", {}).get("sent_id", [])
                )) if item.get("supporting_facts") else [],
            )
            questions.append(q)
            
            # ═══════════════════════════════════════════════════════════════
            # FIX: Context ist ein DICT mit 'title' und 'sentences' Listen
            # (Gleiche Struktur wie HotpotQA!)
            # ═══════════════════════════════════════════════════════════════
            context = item.get("context", {})
            
            # Prüfe ob context ein Dict ist (framolfese Format)
            if isinstance(context, dict):
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])
                
                for title, sentences in zip(titles, sentences_list):
                    if title and title not in articles_dict:
                        # sentences ist eine Liste von Strings
                        if isinstance(sentences, list):
                            text = " ".join(str(s) for s in sentences)
                            sent_list = [str(s) for s in sentences]
                        else:
                            text = str(sentences)
                            sent_list = [text]
                        
                        articles_dict[title] = Article(
                            id=f"2wiki_{len(articles_dict)}",
                            title=title,
                            text=text,
                            sentences=sent_list,
                            dataset="2wikimultihop",
                        )
            
            # Fallback: Falls context eine Liste ist (altes Format)
            elif isinstance(context, list):
                for ctx in context:
                    try:
                        if isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
                            title = str(ctx[0])
                            sentences = ctx[1] if isinstance(ctx[1], list) else [str(ctx[1])]
                        else:
                            continue
                        
                        if title and title not in articles_dict:
                            articles_dict[title] = Article(
                                id=f"2wiki_{len(articles_dict)}",
                                title=title,
                                text=" ".join(str(s) for s in sentences),
                                sentences=[str(s) for s in sentences],
                                dataset="2wikimultihop",
                            )
                    except Exception:
                        continue
        
        articles = list(articles_dict.values())
        logger.info(f"  2WikiMultiHop: {len(articles)} articles, {len(questions)} questions")
        
        return articles, questions

class StrategyQALoader(DatasetLoader):
    """
    StrategyQA: Yes/No questions with implicit reasoning.
    
    FIX: Verschiedene HuggingFace Versionen haben unterschiedliche Felder.
         - wics/strategy-qa: Nur question + answer (keine facts)
         - ChilleD/StrategyQA: Hat facts
         - voidful/StrategyQA: Hat evidence paragraphs (aber broken)
    
    Lösung: Nutze ChilleD/StrategyQA für facts, oder generiere Dummy-Facts.
    """
    
    @property
    def name(self) -> str:
        return "strategyqa"
    
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        from datasets import load_dataset
        
        logger.info("Loading StrategyQA from HuggingFace...")
        
        # Versuche verschiedene Quellen
        ds = None
        source = None
        
        # Option 1: ChilleD hat facts
        try:
            ds = load_dataset("ChilleD/StrategyQA", split="train")
            source = "ChilleD/StrategyQA"
            logger.info(f"  Loaded from {source}")
        except Exception:
            pass
        
        # Option 2: wics (kein facts, aber stabil)
        if ds is None:
            try:
                ds = load_dataset("wics/strategy-qa", "strategyQA", split="test")
                source = "wics/strategy-qa"
                logger.info(f"  Loaded from {source}")
            except Exception:
                pass
        
        # Option 3: metaeval
        if ds is None:
            try:
                ds = load_dataset("metaeval/strategy-qa", split="test")
                source = "metaeval/strategy-qa"
                logger.info(f"  Loaded from {source}")
            except Exception as e:
                logger.error(f"StrategyQA not available: {e}")
                return [], []
        
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        
        articles = []
        questions = []
        
        # Debug: Zeige verfügbare Felder
        if len(ds) > 0:
            sample = ds[0]
            logger.info(f"  Available fields: {list(sample.keys())}")
        
        for idx, item in enumerate(ds):
            # ═══════════════════════════════════════════════════════════════
            # Answer: boolean/int -> yes/no
            # ═══════════════════════════════════════════════════════════════
            raw_answer = item.get("answer", item.get("label", False))
            if isinstance(raw_answer, bool):
                answer = "yes" if raw_answer else "no"
            elif isinstance(raw_answer, int):
                answer = "yes" if raw_answer == 1 else "no"
            elif isinstance(raw_answer, str):
                answer = raw_answer.lower().strip()
                if answer not in ["yes", "no"]:
                    answer = "yes" if answer in ["true", "1"] else "no"
            else:
                answer = "yes" if raw_answer else "no"
            
            q = TestQuestion(
                id=f"strategyqa_{idx}",
                question=item["question"],
                answer=answer,
                dataset="strategyqa",
                question_type="boolean",
            )
            questions.append(q)
            
            # ═══════════════════════════════════════════════════════════════
            # Facts/Evidence extrahieren (verschiedene Feldnamen möglich)
            # ═══════════════════════════════════════════════════════════════
            facts = None
            
            # Versuche verschiedene Feldnamen
            for field_name in ["facts", "evidence", "paragraphs", "decomposition"]:
                if field_name in item and item[field_name]:
                    facts = item[field_name]
                    break
            
            if facts:
                # Facts können verschiedene Formate haben
                if isinstance(facts, list):
                    for i, fact in enumerate(facts):
                        fact_text = None
                        
                        if isinstance(fact, str) and len(fact.strip()) > 10:
                            fact_text = fact.strip()
                        elif isinstance(fact, dict):
                            # Manche Datasets haben {'content': '...'} Format
                            fact_text = fact.get("content", fact.get("text", ""))
                        elif isinstance(fact, list) and len(fact) > 0:
                            # Nested list
                            fact_text = " ".join(str(f) for f in fact)
                        
                        if fact_text and len(fact_text) > 10:
                            articles.append(Article(
                                id=f"strategyqa_fact_{idx}_{i}",
                                title=f"Fact_{idx}_{i}",
                                text=fact_text,
                                sentences=[fact_text],
                                dataset="strategyqa",
                            ))
                
                elif isinstance(facts, str) and len(facts) > 20:
                    # Einzelner String (z.B. decomposition)
                    articles.append(Article(
                        id=f"strategyqa_fact_{idx}_0",
                        title=f"Reasoning_{idx}",
                        text=facts,
                        sentences=[facts],
                        dataset="strategyqa",
                    ))
        
        logger.info(f"  StrategyQA: {len(articles)} facts, {len(questions)} questions")
        
        # Warnung wenn keine Facts gefunden
        if len(articles) == 0 and len(questions) > 0:
            logger.warning("  ⚠ No facts/evidence found in this StrategyQA version!")
            logger.warning("  ⚠ StrategyQA requires external knowledge or web search.")
            logger.warning("  ⚠ Consider using a different evaluation approach for this dataset.")
        
        return articles, questions


# Loader registry
LOADERS: Dict[str, DatasetLoader] = {
    "hotpotqa": HotpotQALoader(),
    "2wikimultihop": WikiMultiHopLoader(),
    "strategyqa": StrategyQALoader(),
}


# ============================================================================
# STORE MANAGER - SEPARATE STORES PER DATASET
# ============================================================================

class StoreManager:
    """
    Manages separate vector stores and graphs per dataset.
    
    Structure:
        data/
        ├── hotpotqa/
        │   ├── vector_db/
        │   ├── knowledge_graph
        │   └── questions.json
        ├── 2wikimultihop/
        │   └── ...
        └── strategyqa/
            └── ...
    """
    
    def __init__(self, base_path: Path = Path("./data")):
        self.base_path = base_path
    
    def get_paths(self, dataset: str) -> Dict[str, Path]:
        """Get all paths for a dataset."""
        ds_path = self.base_path / dataset
        return {
            "root": ds_path,
            "vector_db": ds_path / "vector_db",
            "knowledge_graph": ds_path / "knowledge_graph",
            "questions": ds_path / "questions.json",
            "articles_info": ds_path / "articles_info.json",
        }
    
    def ensure_dirs(self, dataset: str) -> None:
        """Create directories for dataset."""
        paths = self.get_paths(dataset)
        paths["root"].mkdir(parents=True, exist_ok=True)
    
    def clear_dataset(self, dataset: str) -> None:
        """Clear all data for a dataset."""
        paths = self.get_paths(dataset)
        if paths["root"].exists():
            shutil.rmtree(paths["root"])
            logger.info(f"Cleared: {paths['root']}")
    
    def clear_all(self) -> None:
        """Clear all datasets."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            logger.info(f"Cleared all data: {self.base_path}")
    
    def save_questions(self, questions: List[TestQuestion], dataset: str) -> None:
        """Save test questions."""
        self.ensure_dirs(dataset)
        paths = self.get_paths(dataset)
        
        with open(paths["questions"], 'w', encoding='utf-8') as f:
            json.dump([asdict(q) for q in questions], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(questions)} questions to {paths['questions']}")
    
    def load_questions(self, dataset: str) -> List[TestQuestion]:
        """Load test questions."""
        paths = self.get_paths(dataset)
        
        if not paths["questions"].exists():
            logger.error(f"Questions not found: {paths['questions']}")
            logger.error(f"Run: python benchmark_datasets.py ingest --dataset {dataset}")
            return []
        
        with open(paths["questions"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [TestQuestion(**q) for q in data]
    
    def save_articles_info(self, articles: List[Article], dataset: str) -> None:
        """Save article metadata (not full text)."""
        self.ensure_dirs(dataset)
        paths = self.get_paths(dataset)
        
        info = {
            "count": len(articles),
            "dataset": dataset,
            "titles": [a.title for a in articles[:100]],  # Sample
        }
        
        with open(paths["articles_info"], 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    
    def dataset_exists(self, dataset: str) -> bool:
        """Check if dataset is ingested."""
        paths = self.get_paths(dataset)
        return paths["vector_db"].exists() and paths["questions"].exists()
    
    def get_status(self) -> Dict[str, bool]:
        """Get ingestion status for all datasets."""
        return {ds: self.dataset_exists(ds) for ds in AVAILABLE_DATASETS}


# ============================================================================
# DOCUMENT CREATION & INGESTION
# ============================================================================
def create_langchain_documents(
    articles: List[Article], 
    chunk_sentences: int = 3,
    chunking_strategy: str = "sentence",
    config: Optional[Any]= None,
) -> List:
    '''
    Convert articles to LangChain Document objects with configurable chunking.
    
    Args:
        articles: List of Article objects
        chunk_sentences: Sentences per chunk (for sentence strategy)
        chunking_strategy: "sentence", "semantic", "fixed", "recursive"
        config: Optional IngestionConfig for full control
        
    Returns:
        List of LangChain Document objects
    '''
    if not INGESTION_AVAILABLE:
        # Fallback to old implementation
        return _create_langchain_documents_legacy(articles, chunk_sentences)
    
    # Use new ingestion pipeline
    if config is None:
        config = IngestionConfig(
            chunking_strategy=chunking_strategy,
            sentences_per_chunk=chunk_sentences,
            min_chunk_size=50,
            extract_entities=True,
        )
    
    pipeline = DocumentIngestionPipeline(config)
    documents, _ = pipeline.process_articles(articles)
    
    logger.info(
        f"Created {len(documents)} chunks using '{chunking_strategy}' strategy"
    )
    
    return documents


def _create_langchain_documents_legacy(articles: List[Article], chunk_sentences: int = 3) -> List:
    '''Legacy fallback wenn ingestion.py nicht verfügbar.'''
    from langchain.schema import Document
    
    documents = []
    chunk_id = 0
    
    for article in articles:
        sentences = article.sentences
        
        for i in range(0, len(sentences), chunk_sentences):
            chunk_sents = sentences[i:i + chunk_sentences]
            chunk_text = " ".join(chunk_sents)
            
            if len(chunk_text.strip()) < 50:
                continue
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": chunk_id,
                    "source_file": f"{article.dataset}_{article.title}",
                    "article_title": article.title,
                    "dataset": article.dataset,
                    "sentence_start": i,
                    "sentence_end": i + len(chunk_sents),
                }
            )
            documents.append(doc)
            chunk_id += 1
    
    return documents



def run_ingestion(
    documents,
    vector_path: Path,
    graph_path: Path,
    config: Dict,
    dataset_name: str,
) -> None:
    """Ingest documents into vector store and knowledge graph."""
    from src.data_layer.embeddings import BatchedOllamaEmbeddings
    from src.data_layer.storage import HybridStore, StorageConfig
    
    logger.info(f"Ingesting {len(documents)} documents for {dataset_name}...")
    
    # Create directories
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize embeddings
    embedding_config = config.get("embeddings", {})
    perf_config = config.get("performance", {})
    
    cache_path = Path(f"./cache/{dataset_name}_embeddings.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=perf_config.get("batch_size", 32),
        cache_path=cache_path,
        device=perf_config.get("device", "cpu"),
    )
    
    # Initialize storage
    vector_config = config.get("vector_store", {})
    
    storage_config = StorageConfig(
        vector_db_path=vector_path,
        graph_db_path=graph_path,
        embedding_dim=embedding_config.get("embedding_dim", 768),
        similarity_threshold=vector_config.get("similarity_threshold", 0.3),
        normalize_embeddings=vector_config.get("normalize_embeddings", True),
        distance_metric=vector_config.get("distance_metric", "cosine"),
    )
    
    hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)
    
    # Ingest in batches with progress
    start_time = time.time()
    batch_size = 100
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        hybrid_store.add_documents(batch)
        
        progress = min(i + batch_size, len(documents))
        logger.info(f"  Progress: {progress}/{len(documents)} ({progress*100//len(documents)}%)")
    
    hybrid_store.save()
    
    elapsed = time.time() - start_time
    logger.info(f"  Ingestion complete: {elapsed:.1f}s")
    


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def compute_exact_match(prediction: str, gold: str) -> bool:
    """Compute exact match (EM)."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    
    # Exact match
    if pred_norm == gold_norm:
        return True
    
    # Gold contained in prediction (for longer answers)
    if gold_norm and gold_norm in pred_norm:
        return True
    
    # For yes/no questions
    if gold_norm in ["yes", "no"]:
        if gold_norm in pred_norm.split()[:5]:  # Check first 5 words
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
    
    # Count with multiplicity
    num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# PIPELINE INITIALIZATION - KEY: USES DATASET-SPECIFIC STORE
# ============================================================================

def create_pipeline(
    dataset: str,
    config: Dict,
    store_manager: StoreManager,
    vector_weight: float = 0.7,
    graph_weight: float = 0.3,
):
    """
    Create pipeline for specific dataset with given weights.
    
    WICHTIG: Nutzt NUR den Store dieses Datasets!
    """
    from main_agentic import AgenticRAGPipeline
    
    # Get paths for THIS dataset only
    paths = store_manager.get_paths(dataset)
    
    # Update config with dataset-specific paths
    config = config.copy()
    config["paths"] = config.get("paths", {}).copy()
    config["paths"]["vector_db"] = str(paths["vector_db"])
    config["paths"]["graph_db"] = str(paths["knowledge_graph"])
    
    # Set retrieval weights
    config["rag"] = config.get("rag", {}).copy()
    config["rag"]["vector_weight"] = vector_weight
    config["rag"]["graph_weight"] = graph_weight
    
    # Determine retrieval mode
    if graph_weight == 0:
        config["rag"]["retrieval_mode"] = "vector"
    elif vector_weight == 0:
        config["rag"]["retrieval_mode"] = "graph"
    else:
        config["rag"]["retrieval_mode"] = "hybrid"
    
    # Create pipeline (with reduced logging)
    quiet_logger = logging.getLogger("pipeline")
    quiet_logger.setLevel(logging.WARNING)
    
    pipeline = AgenticRAGPipeline(config, quiet_logger)
    pipeline.setup()
    
    return pipeline


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

def evaluate_dataset(
    dataset: str,
    questions: List[TestQuestion],
    pipeline,
    config_name: str,
    vector_weight: float,
    graph_weight: float,
) -> ConfigResult:
    """Evaluate a dataset with given configuration."""
    
    results = []
    
    for i, q in enumerate(questions):
        # Progress every 20 questions
        if (i + 1) % 20 == 0:
            logger.info(f"    Evaluated {i+1}/{len(questions)}...")
        
        try:
            start = time.time()
            result = pipeline.query(q.question)
            elapsed = (time.time() - start) * 1000
            
            em = compute_exact_match(result.answer, q.answer)
            f1 = compute_f1(result.answer, q.answer)
            
            results.append(EvalResult(
                question_id=q.id,
                question=q.question,
                gold_answer=q.answer,
                predicted_answer=result.answer,
                exact_match=em,
                f1_score=f1,
                retrieval_count=len(result.context_docs),
                time_ms=elapsed,
                dataset=q.dataset,
                question_type=q.question_type,
            ))
            
        except Exception as e:
            logger.warning(f"    Error on Q{i}: {str(e)[:50]}")
    
    if not results:
        return None
    
    # Aggregate metrics
    em_rate = sum(1 for r in results if r.exact_match) / len(results)
    avg_f1 = sum(r.f1_score for r in results) / len(results)
    avg_time = sum(r.time_ms for r in results) / len(results)
    coverage = sum(1 for r in results if r.retrieval_count > 0) / len(results)
    
    # By question type
    by_type = {}
    for qtype in set(r.question_type for r in results):
        type_results = [r for r in results if r.question_type == qtype]
        by_type[qtype] = {
            "count": len(type_results),
            "exact_match": sum(1 for r in type_results if r.exact_match) / len(type_results),
            "f1": sum(r.f1_score for r in type_results) / len(type_results),
        }
    
    return ConfigResult(
        dataset=dataset,
        config_name=config_name,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        n_questions=len(results),
        exact_match=em_rate,
        f1_score=avg_f1,
        avg_time_ms=avg_time,
        coverage=coverage,
        by_type=by_type,
    )


# ============================================================================
# COMMANDS
# ============================================================================

def cmd_ingest(args, config: Dict, store_manager: StoreManager):
    """Ingest command - load datasets and create stores."""
    
    # Determine which datasets to ingest
    if args.dataset == "all":
        datasets = AVAILABLE_DATASETS
    else:
        datasets = [d.strip() for d in args.dataset.split(",")]
    
    logger.info("="*70)
    logger.info("BENCHMARK INGESTION")
    logger.info("="*70)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Samples per dataset: {args.samples}")
    logger.info(f"Chunk size: {args.chunk_sentences} sentences")
    logger.info("="*70)
    
    for dataset in datasets:
        logger.info(f"\n{'─'*70}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'─'*70}")
        
        # Check if loader exists
        if dataset not in LOADERS:
            logger.error(f"Unknown dataset: {dataset}")
            logger.info(f"Available: {AVAILABLE_DATASETS}")
            continue
        
        # Clear existing data if requested
        if args.clear:
            store_manager.clear_dataset(dataset)
        
        # Skip if already exists and not clearing
        if store_manager.dataset_exists(dataset) and not args.clear:
            logger.info(f"  Already exists. Use --clear to re-ingest.")
            continue
        
        # Load dataset
        loader = LOADERS[dataset]
        articles, questions = loader.load(n_samples=args.samples)
        
        if not articles and not questions:
            logger.warning(f"  No data loaded for {dataset}")
            continue
        
        # Save questions
        store_manager.save_questions(questions, dataset)
        store_manager.save_articles_info(articles, dataset)
        
        # Create documents
        if articles:
            documents = create_langchain_documents(articles, args.chunk_sentences)
            logger.info(f"  Created {len(documents)} document chunks")
            
            # Run ingestion
            paths = store_manager.get_paths(dataset)
            run_ingestion(
                documents,
                paths["vector_db"],
                paths["knowledge_graph"],
                config,
                dataset,
            )
        else:
            logger.warning(f"  No articles to ingest for {dataset}")
    
    # Print status
    logger.info(f"\n{'='*70}")
    logger.info("INGESTION STATUS")
    logger.info("="*70)
    
    status = store_manager.get_status()
    for ds, exists in status.items():
        mark = "✓" if exists else "✗"
        logger.info(f"  {mark} {ds}")
    
    logger.info("="*70)


def cmd_evaluate(args, config: Dict, store_manager: StoreManager):
    """Evaluate command - run evaluation on single dataset."""
    
    dataset = args.dataset
    
    # Check if dataset exists
    if not store_manager.dataset_exists(dataset):
        logger.error(f"Dataset not ingested: {dataset}")
        logger.error(f"Run: python benchmark_datasets.py ingest --dataset {dataset}")
        return
    
    logger.info("="*70)
    logger.info(f"EVALUATION: {dataset.upper()}")
    logger.info("="*70)
    
    # Load questions
    questions = store_manager.load_questions(dataset)
    if not questions:
        return
    
    if args.samples:
        questions = questions[:args.samples]
    
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"Config: vector={args.vector_weight}, graph={args.graph_weight}")
    
    # Create pipeline FOR THIS DATASET
    pipeline = create_pipeline(
        dataset, config, store_manager,
        vector_weight=args.vector_weight,
        graph_weight=args.graph_weight,
    )
    try:
    # Run evaluation
        config_name = f"v{args.vector_weight}_g{args.graph_weight}"
        result = evaluate_dataset(
            dataset, questions, pipeline,
            config_name, args.vector_weight, args.graph_weight,
        )
        
        # Print results
        logger.info(f"\n{'─'*70}")
        logger.info("RESULTS")
        logger.info(f"{'─'*70}")
        logger.info(f"  Exact Match:  {result.exact_match:.2%}")
        logger.info(f"  F1 Score:     {result.f1_score:.3f}")
        logger.info(f"  Coverage:     {result.coverage:.2%}")
        logger.info(f"  Avg Time:     {result.avg_time_ms:.0f}ms")
    
        if result.by_type:
            logger.info(f"\n  By Question Type:")
            for qtype, stats in result.by_type.items():
                logger.info(f"    {qtype}: EM={stats['exact_match']:.2%}, F1={stats['f1']:.3f}")
        
        logger.info("="*70)
    finally:
        # Cleanup
        if hasattr(pipeline, 'hybrid_store') and pipeline.hybrid_store is not None:
            if hasattr(pipeline.hybrid_store, 'graph_db') and pipeline.hybrid_store.graph_db is not None:
                try:
                    pipeline.hybrid_store.graph_db.close()
                except:
                    pass
        del pipeline
        import gc
        gc.collect()

def cmd_ablation(args, config: Dict, store_manager) -> None:
    """
    Führe wissenschaftliche Ablationsstudie durch.
    
    Nutzt das neue AblationStudy-Modul für:
    - Klare Fortschrittsanzeige
    - Detaillierte Metriken
    - Wissenschaftliche Outputs (JSON, CSV, Markdown, LaTeX)
    """
    
    # Suppress noisy outputs
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    try:
        import unified_planning as up
        up.shortcuts.get_environment().credits_stream = None
    except:
        pass
    
    # Check if new module available
    if not ABLATION_MODULE_AVAILABLE:
        logger.warning("Using legacy ablation (new module not found)")
        return cmd_ablation_legacy(args, config, store_manager)
    
    # Parse arguments
    datasets_to_run = args.datasets if hasattr(args, 'datasets') and args.datasets else AVAILABLE_DATASETS
    samples = args.samples if hasattr(args, 'samples') else 10
    
    logger.info("=" * 70)
    logger.info("ABLATION STUDY (Scientific Mode)")
    logger.info("=" * 70)
    
    # Create ablation config
    ablation_config = AblationConfig(
        name="hybrid_retrieval_ablation",
        seed=getattr(args, 'seed', 42),
        results_dir=Path("results"),
    )
    
    # Initialize study
    study = AblationStudy(
        config=ablation_config,
        pipeline_config=config,
    )
    
    # Load datasets
    datasets = {}
    for dataset_name in datasets_to_run:
        logger.info(f"Loading {dataset_name}...")
        
        # Check if ingested
        if not store_manager.dataset_exists(dataset_name):
            logger.warning(f"  {dataset_name} not ingested, skipping")
            logger.warning(f"  Run: python benchmark_datasets.py ingest --dataset {dataset_name}")
            continue
        
        # Load questions
        questions = store_manager.load_questions(dataset_name)
        if not questions:
            logger.warning(f"  No questions found for {dataset_name}")
            continue
        
        # For ablation we don't need articles, just questions
        datasets[dataset_name] = (questions, [])
        logger.info(f"  Loaded {len(questions)} questions")
    
    if not datasets:
        logger.error("No datasets available! Run ingest first.")
        return
    
    # Create pipeline factory
    def create_pipeline(vector_weight: float, graph_weight: float, dataset_name: str):
        """Factory to create pipeline with specific weights."""
        
        # Get paths for this dataset
        paths = store_manager.get_paths(dataset_name)
        
        # Update config with weights
        pipeline_config = config.copy()
        pipeline_config["retrieval"] = pipeline_config.get("retrieval", {})
        pipeline_config["retrieval"]["vector_weight"] = vector_weight
        pipeline_config["retrieval"]["graph_weight"] = graph_weight
        
        # Create pipeline (adapt to your AgenticRAGPipeline)
        try:
            from main_agentic import AgenticRAGPipeline
            
            pipeline = AgenticRAGPipeline(pipeline_config, logging.getLogger(__name__))
            
            # Override paths for this dataset
            pipeline.config["paths"]["vector_db"] = str(paths["vector_db"])
            pipeline.config["paths"]["graph_db"] = str(paths["knowledge_graph"])
            
            # Setup pipeline
            pipeline.setup()
            
            # Set retrieval weights
            if hasattr(pipeline, 'retriever') and pipeline.retriever:
                pipeline.retriever.vector_weight = vector_weight
                pipeline.retriever.graph_weight = graph_weight
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Pipeline creation failed: {e}")
            raise
    
    # Run study
    results = study.run(
        datasets=datasets,
        samples_per_dataset=samples,
        pipeline_factory=create_pipeline,
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    
    for dataset_name, configs in results.items():
        print(f"\n  Dataset: {dataset_name}")
        print(f"  {'Config':<15} {'EM':<10} {'F1':<10} {'Time':<10} {'Iter':<8} {'Verified':<10}")
        print(f"  {'-'*63}")
        
        for config_name, result in configs.items():
            print(
                f"  {config_name:<15} "
                f"{result.exact_match_mean*100:>6.1f}%   "
                f"{result.f1_mean:>6.3f}    "
                f"{result.avg_time_ms/1000:>6.1f}s   "
                f"{result.avg_iterations:>5.1f}    "
                f"{result.verification_rate*100:>6.1f}%"
            )
        
        # Best config
        if configs:
            best = max(configs.values(), key=lambda x: x.f1_mean)
            print(f"\n  Best: {best.config_name} (F1={best.f1_mean:.3f})")
    
    print(f"\n  Results saved to: {study.run_dir}")
    print("=" * 70 + "\n")


def print_ablation_table(results: AblationResults):
    """Print formatted ablation results table."""
    
    print("\n" + "="*90)
    print("ABLATION STUDY RESULTS")
    print("="*90)
    print(f"Timestamp: {results.timestamp}")
    print(f"Datasets: {results.datasets}")
    
    # Header
    print("\n" + "─"*90)
    header = f"{'Dataset':<15}"
    for cfg in results.configs:
        header += f"{cfg:>12}"
    print(header)
    print("─"*90)
    
    # Results per dataset
    for dataset in results.datasets:
        ds_results = results.results.get(dataset, [])
        
        if not ds_results:
            print(f"{dataset:<15} (no results)")
            continue
        
        # Exact Match row
        row_em = f"{dataset:<15}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            if r:
                row_em += f"{r.exact_match:>11.1%} "
            else:
                row_em += f"{'N/A':>12}"
        print(row_em)
        
        # F1 row
        row_f1 = f"{'  (F1)':<15}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            if r:
                row_f1 += f"{r.f1_score:>11.3f} "
            else:
                row_f1 += f"{'':>12}"
        print(row_f1)
        print()
    
    print("="*90)
    
    # Find best config per dataset
    print("\nBEST CONFIGURATION PER DATASET:")
    print("─"*50)
    
    for dataset in results.datasets:
        ds_results = results.results.get(dataset, [])
        if ds_results:
            best = max(ds_results, key=lambda r: r.f1_score)
            print(f"  {dataset:<15}: {best.config_name} (F1={best.f1_score:.3f})")
    
    print("="*90 + "\n")


def save_ablation_results(results: AblationResults):
    """Save ablation results to JSON and CSV."""
    
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON
    output_path = output_dir / f"ablation_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Save CSV for Excel/LaTeX
    csv_path = output_dir / f"ablation_{timestamp}.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("dataset,config,vector_weight,graph_weight,n_questions,exact_match,f1_score,avg_time_ms\n")
        
        for dataset, ds_results in results.results.items():
            for r in ds_results:
                f.write(f"{r.dataset},{r.config_name},{r.vector_weight},{r.graph_weight},"
                       f"{r.n_questions},{r.exact_match:.4f},{r.f1_score:.4f},{r.avg_time_ms:.1f}\n")
    
    logger.info(f"CSV saved to: {csv_path}")


def cmd_status(args, config: Dict, store_manager: StoreManager):
    """Show status of all datasets."""
    
    print("\n" + "="*50)
    print("DATASET STATUS")
    print("="*50)
    
    status = store_manager.get_status()
    
    for dataset in AVAILABLE_DATASETS:
        exists = status.get(dataset, False)
        mark = "✓" if exists else "✗"
        
        if exists:
            paths = store_manager.get_paths(dataset)
            questions = store_manager.load_questions(dataset)
            n_questions = len(questions) if questions else 0
            print(f"  {mark} {dataset:<15} ({n_questions} questions)")
        else:
            print(f"  {mark} {dataset:<15} (not ingested)")
    
    print("="*50)
    print("\nTo ingest: python benchmark_datasets.py ingest --dataset <name>")
    print("="*50 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def load_config_file(config_path: Path = Path("./config/settings.yaml")) -> Dict:
    """Load configuration from YAML."""
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    logger.warning(f"Config not found: {config_path}, using defaults")
    return {
        "embeddings": {
            "model_name": "nomic-embed-text",
            "base_url": "http://localhost:11434",
            "embedding_dim": 768,
        },
        "vector_store": {
            "similarity_threshold": 0.3,
            "distance_metric": "cosine",
            "normalize_embeddings": True,
        },
        "rag": {
            "retrieval_mode": "hybrid",
            "vector_weight": 0.7,
            "graph_weight": 0.3,
        },
        "performance": {
            "batch_size": 32,
            "device": "cpu",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Benchmark System for RAG Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest HotpotQA
  python benchmark_datasets.py ingest --dataset hotpotqa --samples 500
  
  # Ingest all datasets (separate stores)
  python benchmark_datasets.py ingest --dataset all --samples 300
  
  # Evaluate single dataset
  python benchmark_datasets.py evaluate --dataset hotpotqa --samples 100
  
  # Full ablation study
  python benchmark_datasets.py ablation --samples 100
  
  # Check status
  python benchmark_datasets.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    
    # === EVALUATE ===
    eval_p = subparsers.add_parser("evaluate", help="Evaluate single dataset")
    eval_p.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Dataset to evaluate"
    )
    eval_p.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Number of questions (default: 100)"
    )
    eval_p.add_argument(
        "--vector-weight",
        type=float,
        default=0.7,
        help="Vector retrieval weight (default: 0.7)"
    )
    eval_p.add_argument(
        "--graph-weight",
        type=float,
        default=0.3,
        help="Graph retrieval weight (default: 0.3)"
    )
    
    # === ABLATION ===
    ablation_p = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_p.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        help="Dataset(s): hotpotqa, 2wikimultihop, strategyqa, or 'all'"
    )
    ablation_p.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Questions per dataset (default: 100)"
    )

    
    # === INGEST ===
    ingest_p = subparsers.add_parser("ingest", help="Ingest dataset(s)")
    ingest_p.add_argument(
        "--dataset", "-d",
        type=str,
        default="hotpotqa",
        help="Dataset: hotpotqa, 2wikimultihop, strategyqa, or 'all'"
    )
    ingest_p.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Samples per dataset (default: 500)"
    )
    ingest_p.add_argument(
        "--chunk-sentences",
        type=int,
        default=3,
        help="Sentences per chunk (default: 4)"
    )
    ingest_p.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingesting"
    )


    ingest_p.add_argument(
        "--chunking-strategy",
        type=str,
        choices=["sentence", "semantic", "fixed", "recursive"],
        default="sentence",
        help="Chunking strategy (default: sentence)"
    )
    ingest_p.add_argument(
        "--chunk-overlap",
        type=int,
        default=1,  # ← NEU: 1 Satz Überlappung
        help="Sentence overlap between chunks (default: 1)"
    )
    ingest_p.add_argument(
        "--min-chunk-size",
        type=int,
        default=50,  # ← NEU: Filtert kurze Chunks
        help="Minimum chunk size in characters (default: 50)"
    )
    
    # =============================================================================
    # OPTIONAL: FÜGE DIESE CLI-ARGUMENTE ZUM PARSER HINZU
    # =============================================================================


    # In der argparse-Sektion für 'ablation' subcommand:

    ablation_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    ablation_p.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip LaTeX table generation"
    )

    ablation_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with debug info"
    )


    # === STATUS ===
    subparsers.add_parser("status", help="Show ingestion status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    


    # Load config
    config = load_config_file()
    
    # Create store manager
    store_manager = StoreManager(Path("./data"))
    
    # Execute command
    if args.command == "ingest":
        cmd_ingest(args, config, store_manager)
    elif args.command == "evaluate":
        cmd_evaluate(args, config, store_manager)
    elif args.command == "ablation":
        cmd_ablation(args, config, store_manager)
    elif args.command == "status":
        cmd_status(args, config, store_manager)



if __name__ == "__main__":
    main()
