"""
benchmark_datasets.py - Multi-Dataset Benchmark System (UPDATED)

Version: 4.1.0 - Angepasst an neue Projektstruktur
Author: Edge-RAG Research Project
Last Modified: 2026-01-30

WICHTIG - Wissenschaftlich korrekte Evaluation:
═══════════════════════════════════════════════════════════════════════
Jedes Dataset hat seinen EIGENEN Vector Store + Knowledge Graph.
Bei Evaluation wird NUR der entsprechende Store verwendet.
→ Kein Cross-Dataset Data Leakage!
═══════════════════════════════════════════════════════════════════════

ÄNDERUNGEN v4.1.0:
- ✅ Entfernt: main_agentic.py (existiert nicht mehr)
- ✅ Verwendet: src.pipeline.agent_pipeline.AgentPipeline
- ✅ Verwendet: src.data_layer.chunking.SpacySentenceChunker
- ✅ Verwendet: src.data_layer.ingestion.DocumentIngestionPipeline (fallback)
- ✅ Korrekte Import-Pfade für alle Module

Usage:
    # Ingest einzelnes Dataset
    python benchmark_datasets.py ingest --dataset hotpotqa --samples 500
    
    # Ingest alle Datasets (separate Stores)
    python benchmark_datasets.py ingest --dataset all --samples 500
    
    # Evaluate einzelnes Dataset
    python benchmark_datasets.py evaluate --dataset hotpotqa --samples 100
    
    # Volle Ablation Study
    python benchmark_datasets.py ablation --samples 100
    
    # Self-Test
    python benchmark_datasets.py test
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
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # No-op fallback so code runs without tqdm installed
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable
        def __iter__(self):
            return iter(self._iterable) if self._iterable is not None else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kwargs):
            pass

# ============================================================================
# IMPORTS MIT FALLBACK-LOGIK
# ============================================================================

# LangChain (Core)
try:
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Mock Document class
    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

# Chunking (SpacySentenceChunker ist primär)
CHUNKING_AVAILABLE = False
SpacySentenceChunker = None

try:
    from src.data_layer.chunking import SpacySentenceChunker, create_sentence_chunker
    CHUNKING_AVAILABLE = True
except ImportError:
    pass

# Ingestion Pipeline (optional fallback)
INGESTION_PIPELINE_AVAILABLE = False
DocumentIngestionPipeline = None
IngestionConfig = None

try:
    from src.data_layer.ingestion import DocumentIngestionPipeline, IngestionConfig
    INGESTION_PIPELINE_AVAILABLE = True
except ImportError:
    pass

# Storage
STORAGE_AVAILABLE = False
try:
    from src.data_layer.storage import HybridStore, StorageConfig
    from src.data_layer.embeddings import BatchedOllamaEmbeddings
    from src.data_layer.hybrid_retriever import HybridRetriever, RetrievalConfig, RetrievalMode
    STORAGE_AVAILABLE = True
except ImportError:
    pass

# Pipeline (NEUE STRUKTUR)
PIPELINE_AVAILABLE = False
AgentPipeline = None

try:
    from src.pipeline.agent_pipeline import AgentPipeline, create_full_pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    pass

# Ablation Study (optional)
ABLATION_MODULE_AVAILABLE = False
try:
    from src.evaluations.ablation_study import AblationStudy, AblationConfig
    ABLATION_MODULE_AVAILABLE = True
except ImportError:
    pass

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

# Print availability status
logger.info(f"Module availability:")
logger.info(f"  LangChain:          {LANGCHAIN_AVAILABLE}")
logger.info(f"  Chunking:           {CHUNKING_AVAILABLE}")
logger.info(f"  IngestionPipeline:  {INGESTION_PIPELINE_AVAILABLE}")
logger.info(f"  Storage:            {STORAGE_AVAILABLE}")
logger.info(f"  AgentPipeline:      {PIPELINE_AVAILABLE}")
logger.info(f"  AblationModule:     {ABLATION_MODULE_AVAILABLE}")

# ============================================================================
# CONSTANTS
# ============================================================================

AVAILABLE_DATASETS = ["hotpotqa", "2wikimultihop", "strategyqa"]

ABLATION_CONFIGS = [
    ("vector_only", 1.0, 0.0),
    #("hybrid_80_20", 0.8, 0.2),
    #("hybrid_70_30", 0.7, 0.3),
    ("hybrid_50_50", 0.5, 0.5),
    #("hybrid_30_70", 0.3, 0.7),
    ("graph_only", 0.0, 1.0),
]

# Component ablation: (name, enable_planner, enable_verifier, max_iterations)
# Verwendet mit --component-ablation Flag (Hybrid 50/50 Gewichte als Baseline)
COMPONENT_CONFIGS = [
    ("full",        True,  True,  1),
    ("no_planner",  False, True,  1),
    ("no_verifier", True,  False, 1),
    ("iter_2",      True,  True,  2),
    ("iter_3",      True,  True,  3),
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
    results: Dict[str, List[ConfigResult]]
    
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
            
            q = TestQuestion(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                dataset="hotpotqa",
                question_type=item["type"],
                level=item["level"],
                supporting_facts=list(zip(
                    item["supporting_facts"]["title"],
                    item["supporting_facts"]["sent_id"]
                )),
            )
            questions.append(q)
            
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
    """2WikiMultiHopQA: Requires 2 Wikipedia articles."""
    
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
            return [], []
        
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        
        articles_dict = {}
        questions = []
        
        for idx, item in enumerate(ds):
            if idx % 100 == 0 and idx > 0:
                logger.info(f"  Processing {idx}/{len(ds)}...")
            
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
            
            context = item.get("context", {})
            
            if isinstance(context, dict):
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])
                
                for title, sentences in zip(titles, sentences_list):
                    if title and title not in articles_dict:
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
        
        articles = list(articles_dict.values())
        logger.info(f"  2WikiMultiHop: {len(articles)} articles, {len(questions)} questions")
        
        return articles, questions

class StrategyQALoader(DatasetLoader):
    """StrategyQA: Yes/No questions with implicit reasoning."""
    
    @property
    def name(self) -> str:
        return "strategyqa"
    
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        from datasets import load_dataset
        
        logger.info("Loading StrategyQA from HuggingFace...")
        
        ds = None
        try:
            ds = load_dataset("ChilleD/StrategyQA", split="train")
        except:
            try:
                ds = load_dataset("wics/strategy-qa", "strategyQA", split="test")
            except Exception as e:
                logger.error(f"StrategyQA not available: {e}")
                return [], []
        
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        
        articles = []
        questions = []
        
        for idx, item in enumerate(ds):
            raw_answer = item.get("answer", item.get("label", False))
            if isinstance(raw_answer, bool):
                answer = "yes" if raw_answer else "no"
            elif isinstance(raw_answer, int):
                answer = "yes" if raw_answer == 1 else "no"
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
            
            facts = None
            for field_name in ["facts", "evidence", "paragraphs", "decomposition"]:
                if field_name in item and item[field_name]:
                    facts = item[field_name]
                    break
            
            if facts and isinstance(facts, list):
                for i, fact in enumerate(facts):
                    if isinstance(fact, str) and len(fact.strip()) > 10:
                        articles.append(Article(
                            id=f"strategyqa_fact_{idx}_{i}",
                            title=f"Fact_{idx}_{i}",
                            text=fact.strip(),
                            sentences=[fact.strip()],
                            dataset="strategyqa",
                        ))
        
        logger.info(f"  StrategyQA: {len(articles)} facts, {len(questions)} questions")
        
        if len(articles) == 0:
            logger.warning("  ⚠ No facts found - StrategyQA requires external knowledge!")
        
        return articles, questions

LOADERS: Dict[str, DatasetLoader] = {
    "hotpotqa": HotpotQALoader(),
    "2wikimultihop": WikiMultiHopLoader(),
    "strategyqa": StrategyQALoader(),
}

# ============================================================================
# STORE MANAGER
# ============================================================================

class StoreManager:
    """Manages separate vector stores and graphs per dataset."""
    
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
            return []
        
        with open(paths["questions"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [TestQuestion(**q) for q in data]
    
    def save_articles_info(self, articles: List[Article], dataset: str) -> None:
        """Save article metadata."""
        self.ensure_dirs(dataset)
        paths = self.get_paths(dataset)
        
        info = {
            "count": len(articles),
            "dataset": dataset,
            "titles": [a.title for a in articles[:100]],
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
    sentence_overlap: int = 1,
) -> List[Document]:
    """
    Convert articles to LangChain Documents using SpacySentenceChunker.
    
    NEUE IMPLEMENTIERUNG v4.1.0:
    - Primär: SpacySentenceChunker (gemäß Masterthesis 2.2)
    - Fallback: Simple sentence grouping
    """
    
    # Methode 1: SpacySentenceChunker (primär)
    if CHUNKING_AVAILABLE and SpacySentenceChunker is not None:
        logger.info("Using SpacySentenceChunker (3-sentence window)")
        
        try:
            chunker = create_sentence_chunker(
                sentences_per_chunk=chunk_sentences,
                sentence_overlap=sentence_overlap,
                min_chunk_chars=50,
            )
            
            all_documents = []
            chunk_id = 0
            
            for article in articles:
                # Chunk this article's text
                article_text = article.text
                chunk_results = chunker.chunk_text(
                    article_text,
                    source_doc=article.title
                )
                
                # Convert to LangChain Documents
                for chunk in chunk_results:
                    doc = Document(
                        page_content=chunk.text,
                        metadata={
                            "chunk_id": chunk_id,
                            "source_file": f"{article.dataset}_{article.title}",
                            "article_title": article.title,
                            "dataset": article.dataset,
                            "sentence_count": chunk.sentence_count,
                            "position": chunk.position,
                        }
                    )
                    all_documents.append(doc)
                    chunk_id += 1
            
            logger.info(f"Created {len(all_documents)} chunks using SpaCy chunker")
            return all_documents
            
        except Exception as e:
            logger.warning(f"SpaCy chunker failed: {e} - using fallback")
    
    # Methode 2: Fallback (immer verfügbar)
    logger.info("Using fallback sentence grouping")
    return _create_documents_fallback(articles, chunk_sentences)

def _create_documents_fallback(articles: List[Article], chunk_sentences: int = 3) -> List[Document]:
    """Fallback chunking without SpaCy."""
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
    documents: List[Document],
    vector_path: Path,
    graph_path: Path,
    config: Dict,
    dataset_name: str,
) -> None:
    """Ingest documents into vector store and knowledge graph."""
    
    if not STORAGE_AVAILABLE:
        logger.error("Storage module not available!")
        logger.error("Install: pip install lancedb kuzu")
        return
    
    logger.info(f"Ingesting {len(documents)} documents for {dataset_name}...")
    
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
        enable_entity_extraction=True,   # GLiNER + REBEL → Entity-Nodes in KuzuDB
    )

    hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)

    # Ingest in batches
    start_time = time.time()
    batch_size = 100
    n_batches = (len(documents) + batch_size - 1) // batch_size

    with tqdm(total=len(documents), desc=f"Ingesting {dataset_name}", unit="doc",
              disable=not TQDM_AVAILABLE) as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_start = time.time()
            hybrid_store.add_documents(batch)
            batch_elapsed = time.time() - batch_start
            pbar.update(len(batch))
            done = i + len(batch)
            pct = 100 * done / len(documents)
            elapsed_total = time.time() - start_time
            remaining = (elapsed_total / done) * (len(documents) - done) if done > 0 else 0
            logger.info(
                f"  [{pct:5.1f}%] Batch {i // batch_size + 1}/{n_batches} "
                f"| {batch_elapsed:.0f}s/batch "
                f"| verbleibend ~{remaining/3600:.1f}h"
            )
            pbar.set_postfix(batch=f"{i // batch_size + 1}/{n_batches}")
    
    hybrid_store.save()
    
    elapsed = time.time() - start_time
    logger.info(f"  Ingestion complete: {elapsed:.1f}s")

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def compute_exact_match(prediction: str, gold: str) -> bool:
    """Compute exact match (EM)."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    
    if pred_norm == gold_norm:
        return True
    
    if gold_norm and gold_norm in pred_norm:
        return True
    
    if gold_norm in ["yes", "no"]:
        if gold_norm in pred_norm.split()[:5]:
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
    
    num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

# ============================================================================
# PIPELINE INITIALIZATION
# ============================================================================

def create_pipeline(
    dataset: str,
    config: Dict,
    store_manager: StoreManager,
    vector_weight: float = 0.7,
    graph_weight: float = 0.3,
    model_name: str = None,
    enable_planner: bool = True,
    enable_verifier: bool = True,
    max_iterations: int = 1,
):
    """
    Create pipeline for specific dataset.
    
    NEUE IMPLEMENTIERUNG v4.1.0:
    - Verwendet: src.pipeline.agent_pipeline.AgentPipeline
    - Verwendet: create_full_pipeline factory
    """
    
    if not PIPELINE_AVAILABLE:
        logger.error("Pipeline module not available!")
        logger.error("Install: Check src/pipeline/agent_pipeline.py")
        raise ImportError("AgentPipeline not found")
    
    # Get paths for this dataset
    paths = store_manager.get_paths(dataset)
    
    # Update config
    pipeline_config = config.copy()
    pipeline_config["paths"] = pipeline_config.get("paths", {}).copy()
    pipeline_config["paths"]["vector_db"] = str(paths["vector_db"])
    pipeline_config["paths"]["graph_db"] = str(paths["knowledge_graph"])

    # Model override (--model flag) — all model names live in settings.yaml
    if model_name is not None:
        pipeline_config["llm"] = pipeline_config.get("llm", {}).copy()
        pipeline_config["llm"]["model_name"] = model_name

    # Component ablation flags
    pipeline_config["agent"] = pipeline_config.get("agent", {}).copy()
    pipeline_config["agent"]["enable_planner"] = enable_planner
    pipeline_config["agent"]["enable_verifier"] = enable_verifier
    pipeline_config["agent"]["max_verification_iterations"] = max_iterations

    # Set retrieval weights
    pipeline_config["rag"] = pipeline_config.get("rag", {}).copy()
    pipeline_config["rag"]["vector_weight"] = vector_weight
    pipeline_config["rag"]["graph_weight"] = graph_weight
    
    if graph_weight == 0:
        pipeline_config["rag"]["retrieval_mode"] = "vector"
    elif vector_weight == 0:
        pipeline_config["rag"]["retrieval_mode"] = "graph"
    else:
        pipeline_config["rag"]["retrieval_mode"] = "hybrid"
    
    # Initialize storage components
    if not STORAGE_AVAILABLE:
        raise ImportError("Storage module required for pipeline")
    
    embedding_config = pipeline_config.get("embeddings", {})
    cache_path = Path(f"./cache/{dataset}_embeddings.db")
    
    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=32,
        cache_path=cache_path,
    )
    
    vector_config = pipeline_config.get("vector_store", {})
    storage_config = StorageConfig(
        vector_db_path=paths["vector_db"],
        graph_db_path=paths["knowledge_graph"],
        embedding_dim=embedding_config.get("embedding_dim", 768),
        similarity_threshold=vector_config.get("similarity_threshold", 0.3),
        normalize_embeddings=vector_config.get("normalize_embeddings", True),
        distance_metric=vector_config.get("distance_metric", "cosine"),
    )
    
    hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)

    # Determine retrieval mode
    if graph_weight == 0:
        retrieval_mode = RetrievalMode.VECTOR
    elif vector_weight == 0:
        retrieval_mode = RetrievalMode.GRAPH
    else:
        retrieval_mode = RetrievalMode.HYBRID

    # Wrap HybridStore in HybridRetriever (Navigator needs .retrieve())
    retrieval_config = RetrievalConfig(
        mode=retrieval_mode,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        similarity_threshold=vector_config.get("similarity_threshold", 0.3),
    )
    retriever = HybridRetriever(
        hybrid_store=hybrid_store,
        embeddings=embeddings,
        config=retrieval_config,
    )

    # Create pipeline using factory
    pipeline = create_full_pipeline(
        hybrid_retriever=retriever,
        graph_store=hybrid_store.graph_store,
        config=pipeline_config,
    )
    
    logger.info(f"Pipeline created for {dataset} (v={vector_weight}, g={graph_weight})")
    
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
    """Evaluate dataset with given configuration."""
    
    results = []

    for q in tqdm(questions, desc=f"Evaluating {dataset} [{config_name}]", unit="q"):
        try:
            start = time.time()
            result = pipeline.process(q.question)
            elapsed = (time.time() - start) * 1000
            
            em = compute_exact_match(result.answer, q.answer)
            f1 = compute_f1(result.answer, q.answer)
            
            retrieval_count = 0
            if hasattr(result, 'navigator_result'):
                nav_result = result.navigator_result
                if isinstance(nav_result, dict):
                    retrieval_count = len(nav_result.get('filtered_context', []))
            
            results.append(EvalResult(
                question_id=q.id,
                question=q.question,
                gold_answer=q.answer,
                predicted_answer=result.answer,
                exact_match=em,
                f1_score=f1,
                retrieval_count=retrieval_count,
                time_ms=elapsed,
                dataset=q.dataset,
                question_type=q.question_type,
            ))
            
        except Exception as e:
            logger.warning(f"    Error on Q{q.id}: {str(e)[:50]}")
    
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
    """Ingest command."""
    
    if args.dataset == "all":
        datasets = AVAILABLE_DATASETS
    else:
        datasets = [d.strip() for d in args.dataset.split(",")]
    
    logger.info("="*70)
    logger.info("BENCHMARK INGESTION")
    logger.info("="*70)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Samples per dataset: {args.samples}")
    logger.info(f"Chunking: {args.chunk_sentences} sentences, overlap={args.chunk_overlap}")
    logger.info("="*70)
    
    for dataset in datasets:
        logger.info(f"\n{'─'*70}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'─'*70}")
        
        if dataset not in LOADERS:
            logger.error(f"Unknown dataset: {dataset}")
            continue
        
        if args.clear:
            store_manager.clear_dataset(dataset)
        
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
        
        # Create documents and ingest
        if articles:
            documents = create_langchain_documents(
                articles,
                chunk_sentences=args.chunk_sentences,
                sentence_overlap=args.chunk_overlap,
            )
            logger.info(f"  Created {len(documents)} document chunks")

            # --chunks-only: JSON exportieren und stoppen (Decoupled Ingestion Part 1)
            if getattr(args, "chunks_only", False):
                out_dir = Path("./data") / dataset
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "chunks_export.json"
                chunks_data = [
                    {"text": doc.page_content, "metadata": {k: str(v) for k, v in doc.metadata.items()}}
                    for doc in documents
                ]
                with open(out_path, "w", encoding="utf-8") as f:
                    import json as _json
                    _json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                logger.info(f"  [chunks-only] Exportiert: {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
                logger.info(f"  [chunks-only] Naechster Schritt: Datei in Google Colab hochladen.")
                continue

            paths = store_manager.get_paths(dataset)
            run_ingestion(
                documents,
                paths["vector_db"],
                paths["knowledge_graph"],
                config,
                dataset,
            )
    
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
    """Evaluate command."""
    
    dataset = args.dataset
    
    if not store_manager.dataset_exists(dataset):
        logger.error(f"Dataset not ingested: {dataset}")
        logger.error(f"Run: python benchmark_datasets.py ingest --dataset {dataset}")
        return
    
    logger.info("="*70)
    logger.info(f"EVALUATION: {dataset.upper()}")
    logger.info("="*70)
    
    questions = store_manager.load_questions(dataset)
    if not questions:
        return
    
    if args.samples:
        questions = questions[:args.samples]
    
    model_name = getattr(args, "model", None) or config.get("llm", {}).get("model_name", "phi3")
    enable_planner = not getattr(args, "no_planner", False)
    enable_verifier = not getattr(args, "no_verifier", False)
    max_iterations = getattr(args, "iterations", 1) or 1
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"Config: vector={args.vector_weight}, graph={args.graph_weight}, model={model_name}")
    logger.info(f"Components: planner={enable_planner}, verifier={enable_verifier}, iter={max_iterations}")

    # Create pipeline
    pipeline = create_pipeline(
        dataset, config, store_manager,
        vector_weight=args.vector_weight,
        graph_weight=args.graph_weight,
        model_name=model_name,
        enable_planner=enable_planner,
        enable_verifier=enable_verifier,
        max_iterations=max_iterations,
    )

    try:
        config_name = f"v{args.vector_weight}_g{args.graph_weight}_{model_name}"
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
        del pipeline
        import gc
        gc.collect()

def cmd_ablation(args, config: Dict, store_manager: StoreManager):
    """Ablation study command."""
    
    logger.info("="*70)
    logger.info("ABLATION STUDY")
    logger.info("="*70)
    
    # Parse datasets
    if args.dataset == "all":
        datasets = AVAILABLE_DATASETS
    else:
        datasets = [d.strip() for d in args.dataset.split(",")]
    
    # Verify all datasets are ingested
    for dataset in datasets:
        if not store_manager.dataset_exists(dataset):
            logger.error(f"Dataset not ingested: {dataset}")
            logger.error(f"Run: python benchmark_datasets.py ingest --dataset {dataset}")
            return
    
    model_name = getattr(args, "model", None) or config.get("llm", {}).get("model_name", "phi3")
    do_component = getattr(args, "component_ablation", False)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Samples per dataset: {args.samples}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Retrieval configs: {len(ABLATION_CONFIGS)}")
    logger.info(f"Component ablation: {do_component}")
    logger.info("="*70)

    # Run ablation
    all_results: Dict[str, List] = {}
    used_run_names: List[str] = []

    for dataset in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'='*70}")

        questions = store_manager.load_questions(dataset)
        if args.samples:
            questions = questions[:args.samples]

        logger.info(f"Questions: {len(questions)}")

        dataset_results = []

        # ── Retrieval-weight ablation ──────────────────────────────────────
        for cfg_name, vector_weight, graph_weight in ABLATION_CONFIGS:
            run_name = f"{cfg_name}_{model_name}"
            logger.info(f"\n  [Retrieval] {run_name} (v={vector_weight}, g={graph_weight})")

            try:
                pipeline = create_pipeline(
                    dataset, config, store_manager,
                    vector_weight=vector_weight,
                    graph_weight=graph_weight,
                    model_name=model_name,
                )

                result = evaluate_dataset(
                    dataset, questions, pipeline,
                    run_name, vector_weight, graph_weight,
                )

                if result:
                    dataset_results.append(result)
                    if run_name not in used_run_names:
                        used_run_names.append(run_name)
                    logger.info(f"    EM: {result.exact_match:.2%}, F1: {result.f1_score:.3f}, "
                                f"Latency: {result.avg_time_ms:.0f}ms")

                del pipeline
                import gc
                gc.collect()

            except Exception as e:
                logger.error(f"    Failed: {e}")

        # ── Component ablation (optional) ─────────────────────────────────
        if do_component:
            logger.info(f"\n  {'─'*60}")
            logger.info(f"  COMPONENT ABLATION (hybrid 50/50)")
            logger.info(f"  {'─'*60}")
            for comp_name, enable_p, enable_v, max_iter in COMPONENT_CONFIGS:
                run_name = f"comp_{comp_name}_{model_name}"
                logger.info(f"\n  [Component] {run_name} "
                            f"(planner={enable_p}, verifier={enable_v}, iter={max_iter})")

                try:
                    pipeline = create_pipeline(
                        dataset, config, store_manager,
                        vector_weight=0.5, graph_weight=0.5,
                        model_name=model_name,
                        enable_planner=enable_p,
                        enable_verifier=enable_v,
                        max_iterations=max_iter,
                    )

                    result = evaluate_dataset(
                        dataset, questions, pipeline,
                        run_name, 0.5, 0.5,
                    )

                    if result:
                        dataset_results.append(result)
                        if run_name not in used_run_names:
                            used_run_names.append(run_name)
                        logger.info(f"    EM: {result.exact_match:.2%}, F1: {result.f1_score:.3f}, "
                                    f"Latency: {result.avg_time_ms:.0f}ms")

                    del pipeline
                    gc.collect()

                except Exception as e:
                    logger.error(f"    Failed: {e}")

        all_results[dataset] = dataset_results

    # Save results
    ablation_results = AblationResults(
        timestamp=datetime.now().isoformat(),
        datasets=datasets,
        configs=used_run_names,
        results=all_results,
    )
    
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ablation_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_results.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print_ablation_table(ablation_results)

def _compute_mur(ds_results: List) -> Dict[str, Optional[float]]:
    """
    MUR = ΔF1 / ΔLatency(s)  —  Marginal Utility Ratio.

    Baseline = erster Eintrag (vector_only oder full).
    Positiv = Verbesserung pro Sekunde Mehrlatenz.
    None = Latenz-Unterschied zu klein für zuverlässige Messung.
    """
    if not ds_results:
        return {}
    baseline = ds_results[0]
    mur: Dict[str, Optional[float]] = {baseline.config_name: None}
    for r in ds_results[1:]:
        delta_f1 = r.f1_score - baseline.f1_score
        delta_s = (r.avg_time_ms - baseline.avg_time_ms) / 1000.0
        if abs(delta_s) < 0.001:          # < 1ms Unterschied → nicht messbar
            mur[r.config_name] = None
        elif delta_s > 0:
            mur[r.config_name] = delta_f1 / delta_s
        else:                              # Schneller als Baseline
            mur[r.config_name] = float("inf") if delta_f1 > 0 else delta_f1 / delta_s
    return mur


def print_ablation_table(results: AblationResults):
    """Print formatted ablation results with MUR metric."""

    col_w = 14
    n_cols = len(results.configs)
    total_w = 16 + col_w * n_cols

    print("\n" + "="*total_w)
    print("ABLATION STUDY RESULTS")
    print("="*total_w)

    # Header
    header = f"{'Dataset':<16}"
    for cfg in results.configs:
        short = cfg[:col_w-1]
        header += f"{short:>{col_w}}"
    print(header)
    print("─"*total_w)

    for dataset in results.datasets:
        ds_results = results.results.get(dataset, [])

        if not ds_results:
            print(f"{dataset:<16} (no results)")
            continue

        mur_scores = _compute_mur(ds_results)

        # EM row
        row = f"{dataset + ' (EM)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.exact_match:>{col_w}.1%}" if r else f"{'N/A':>{col_w}}"
        print(row)

        # F1 row
        row = f"{'  (F1)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.f1_score:>{col_w}.3f}" if r else f"{'':>{col_w}}"
        print(row)

        # Latency row
        row = f"{'  (ms)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.avg_time_ms:>{col_w}.0f}" if r else f"{'':>{col_w}}"
        print(row)

        # MUR row  (ΔF1 / ΔLatency_s)
        row = f"{'  (MUR)':<16}"
        for cfg_name in results.configs:
            mur_val = mur_scores.get(cfg_name)
            if mur_val is None:
                row += f"{'—':>{col_w}}"
            elif mur_val == float("inf"):
                row += f"{'∞':>{col_w}}"
            else:
                row += f"{mur_val:>{col_w}.3f}"
        print(row)
        print()

    print("="*total_w)

    print("\nBEST CONFIGURATION PER DATASET (by F1):")
    for dataset in results.datasets:
        ds_results = results.results.get(dataset, [])
        if ds_results:
            best = max(ds_results, key=lambda r: r.f1_score)
            mur_val = _compute_mur(ds_results).get(best.config_name)
            mur_str = f", MUR={mur_val:.3f}" if mur_val is not None and mur_val != float("inf") else ""
            print(f"  {dataset:<15}: {best.config_name}  (F1={best.f1_score:.3f}{mur_str})")

    print("="*total_w + "\n")
    print("MUR = ΔF1 / ΔLatency(s) vs. baseline (first config).")
    print("Higher MUR = better accuracy-per-second trade-off.\n")

def cmd_status(args, config: Dict, store_manager: StoreManager):
    """Show status command."""
    
    print("\n" + "="*50)
    print("DATASET STATUS")
    print("="*50)
    
    status = store_manager.get_status()
    
    for dataset in AVAILABLE_DATASETS:
        exists = status.get(dataset, False)
        mark = "✓" if exists else "✗"
        
        if exists:
            questions = store_manager.load_questions(dataset)
            n_questions = len(questions) if questions else 0
            print(f"  {mark} {dataset:<15} ({n_questions} questions)")
        else:
            print(f"  {mark} {dataset:<15} (not ingested)")
    
    print("="*50)
    print("\nTo ingest: python benchmark_datasets.py ingest --dataset <name>")
    print("="*50 + "\n")

def cmd_test(args, config: Dict, store_manager: StoreManager):
    """Self-test command."""
    
    print("\n" + "="*70)
    print("🧪 BENCHMARK_DATASETS.PY - SELF-TEST")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Module imports
    print("\nTest 1: Module Availability")
    print(f"  LangChain:     {'✓' if LANGCHAIN_AVAILABLE else '✗'}")
    print(f"  Chunking:      {'✓' if CHUNKING_AVAILABLE else '✗'}")
    print(f"  Storage:       {'✓' if STORAGE_AVAILABLE else '✗'}")
    print(f"  AgentPipeline: {'✓' if PIPELINE_AVAILABLE else '✗'}")
    
    # Test 2: Data classes
    print("\nTest 2: Data Classes")
    try:
        q = TestQuestion("t1", "Q?", "A", "test")
        a = Article("a1", "Title", "Text", ["S1"], "test")
        print("  ✓ TestQuestion, Article")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        tests_failed += 1
    
    # Test 3: Metrics
    print("\nTest 3: Evaluation Metrics")
    try:
        em = compute_exact_match("Paris", "paris")
        f1 = compute_f1("Paris France", "Paris")
        assert em == True
        assert 0 < f1 < 1
        print("  ✓ compute_exact_match, compute_f1")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        tests_failed += 1
    
    # Test 4: Loaders
    print("\nTest 4: Dataset Loaders")
    try:
        assert "hotpotqa" in LOADERS
        assert "2wikimultihop" in LOADERS
        assert "strategyqa" in LOADERS
        print("  ✓ All loaders registered")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("="*70 + "\n")
    
    return 0 if tests_failed == 0 else 1

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
        description="Multi-Dataset Benchmark System (v4.1.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # INGEST
    ingest_p = subparsers.add_parser("ingest", help="Ingest dataset(s)")
    ingest_p.add_argument("--dataset", "-d", type=str, default="hotpotqa")
    ingest_p.add_argument("--samples", "-n", type=int, default=500)
    ingest_p.add_argument("--chunk-sentences", type=int, default=3)
    ingest_p.add_argument("--chunk-overlap", type=int, default=1)
    ingest_p.add_argument("--clear", action="store_true")
    ingest_p.add_argument("--chunks-only", action="store_true",
                          help="Nur Chunks erstellen und als JSON exportieren, keine Ingestion")
    
    # EVALUATE
    eval_p = subparsers.add_parser("evaluate", help="Evaluate single dataset")
    eval_p.add_argument("--dataset", "-d", type=str, required=True)
    eval_p.add_argument("--samples", "-n", type=int, default=100)
    eval_p.add_argument("--vector-weight", type=float, default=0.7)
    eval_p.add_argument("--graph-weight", type=float, default=0.3)
    eval_p.add_argument("--model", "-m", type=str, default=None,
                        help="Modellname (z.B. phi3, llama3.2:3b). Default: aus settings.yaml")
    eval_p.add_argument("--no-planner", action="store_true",
                        help="S_P überspringen (Ablation: kein Planner)")
    eval_p.add_argument("--no-verifier", action="store_true",
                        help="S_V überspringen (Ablation: kein Verifier)")
    eval_p.add_argument("--iterations", type=int, default=1,
                        help="Anzahl Verifier-Iterationen (1/2/3). Default: 1")

    # ABLATION
    ablation_p = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_p.add_argument("--dataset", "-d", type=str, default="all")
    ablation_p.add_argument("--samples", "-n", type=int, default=100)
    ablation_p.add_argument("--model", "-m", type=str, default=None,
                            help="Modellname (z.B. phi3, llama3.2:3b). Default: aus settings.yaml")
    ablation_p.add_argument("--component-ablation", action="store_true",
                            help="Zusätzlich Planner/Verifier/Iterations-Ablation durchführen")
    
    # STATUS
    subparsers.add_parser("status", help="Show ingestion status")
    
    # TEST
    test_p = subparsers.add_parser("test", help="Run self-tests")
    test_p.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config = load_config_file()
    store_manager = StoreManager(Path("./data"))
    
    if args.command == "ingest":
        cmd_ingest(args, config, store_manager)
    elif args.command == "evaluate":
        cmd_evaluate(args, config, store_manager)
    elif args.command == "ablation":
        cmd_ablation(args, config, store_manager)
    elif args.command == "status":
        cmd_status(args, config, store_manager)
    elif args.command == "test":
        cmd_test(args, config, store_manager)

if __name__ == "__main__":
    main()