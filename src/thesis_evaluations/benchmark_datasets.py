"""
benchmark_datasets.py - Multi-Dataset Benchmark System (UPDATED)

Version: 4.1.0 - Adapted to new project structure
Author: Edge-RAG Research Project
Last Modified: 2026-01-30

IMPORTANT - Scientifically correct evaluation:
═══════════════════════════════════════════════════════════════════════
Each dataset has its OWN vector store + knowledge graph.
During evaluation ONLY the corresponding store is used.
→ No cross-dataset data leakage!
═══════════════════════════════════════════════════════════════════════

Changes v4.1.0:
- ✅ Removed: main_agentic.py (no longer exists)
- ✅ Uses: src.pipeline.agent_pipeline.AgentPipeline
- ✅ Uses: src.data_layer.chunking.SpacySentenceChunker
- ✅ Uses: src.data_layer.ingestion.DocumentIngestionPipeline (fallback)
- ✅ Correct import paths for all modules

Usage:
    # Ingest single dataset
    python benchmark_datasets.py ingest --dataset hotpotqa --samples 500

    # Ingest all datasets (separate stores)
    python benchmark_datasets.py ingest --dataset all --samples 500

    # Evaluate single dataset
    python benchmark_datasets.py evaluate --dataset hotpotqa --samples 100

    # Full ablation study
    python benchmark_datasets.py ablation --samples 100

    # Self-test
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
# IMPORTS WITH FALLBACK LOGIC
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

# Chunking (SpacySentenceChunker is primary)
CHUNKING_AVAILABLE = False
SpacySentenceChunker = None

try:
    from src.data_layer.chunking import SpacySentenceChunker, create_sentence_chunker
    CHUNKING_AVAILABLE = True
except ImportError:
    pass

# Ingestion Pipeline (optional fallback — used when primary path unavailable)
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

# Pipeline (new structure)
PIPELINE_AVAILABLE = False
AgentPipeline = None

try:
    from src.pipeline import AgentPipeline, create_full_pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    pass

# Ablation Study module (optional)
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
# Used with --component-ablation flag (hybrid 50/50 weights as baseline)
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
    """Single question evaluation result.

    Separates pipeline correctness (retrieval) from model correctness (LLM
    answer) so the thesis can argue: "of N questions where retrieval found
    all gold paragraphs, X% were also answered correctly — the remaining
    gap is model capacity, not pipeline architecture."
    """
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

    # Retrieval quality (independent of LLM)
    gold_titles: List[str] = field(default_factory=list)
    retrieved_titles: List[str] = field(default_factory=list)
    retrieval_recall: float = 0.0
    retrieval_precision: float = 0.0
    sf_f1: float = 0.0
    all_gold_retrieved: bool = False

    # Failure-mode separation
    llm_error: bool = False
    llm_error_type: str = ""
    pipeline_succeeded_llm_failed: bool = False

    # Planner diagnostics
    planner_query_type: str = ""
    hop_count: int = 0
    n_entities: int = 0

    # Verifier diagnostics
    verifier_iterations: int = 0
    all_verified: bool = False
    confidence: str = ""

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

    # Retrieval-level aggregates (pipeline correctness)
    avg_sf_f1: float = 0.0
    sf_recall_rate: float = 0.0          # fraction of Qs where all gold retrieved
    retrieval_only_em: float = 0.0       # EM among Qs where all gold retrieved
    llm_error_rate: float = 0.0
    pipeline_failed_rate: float = 0.0    # gold not fully retrieved
    pipeline_ok_llm_failed_rate: float = 0.0
    pipeline_ok_llm_wrong_rate: float = 0.0
    pipeline_ok_llm_ok_rate: float = 0.0

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
    """Manages separate vector stores and knowledge graphs per dataset."""
    
    def __init__(self, base_path: Path = Path("./data")):
        self.base_path = base_path
    
    def get_paths(self, dataset: str) -> Dict[str, Path]:
        """Return all storage paths for a dataset."""
        ds_path = self.base_path / dataset
        return {
            "root": ds_path,
            "vector": ds_path / "vector",
            "graph": ds_path / "graph",
            "questions": ds_path / "questions.json",
            "articles_info": ds_path / "articles_info.json",
        }
    
    def ensure_dirs(self, dataset: str) -> None:
        """Create directory structure for a dataset."""
        paths = self.get_paths(dataset)
        paths["root"].mkdir(parents=True, exist_ok=True)
    
    def clear_dataset(self, dataset: str, chunks_only: bool = False) -> None:
        """
        Clear data for a dataset.

        Two modes:
          chunks_only=True:   Delete ONLY chunks_export.json (Phase 1 output).
                              Leaves vector/graph/extraction_results.json
                              intact. Use when re-running Phase 1 while the
                              KuzuDB graph may be locked by another process
                              (Windows holds Kuzu file locks until the OS
                              releases them).

          chunks_only=False:  Full reset. Rescues every .json file in the
                              tree (chunks_export, questions, articles_info,
                              extraction_results — all expensive to
                              regenerate, especially the Colab output),
                              then deletes the rest. PermissionError on
                              individual files (typically Kuzu .lock files
                              still held by the OS) is logged as a warning
                              and skipped — partial cleanup is acceptable
                              because re-ingestion will MERGE-overwrite.
        """
        paths = self.get_paths(dataset)
        root = paths["root"]
        if not root.exists():
            return

        # ── Mode A: chunks-only -> remove just chunks_export.json ─────────
        if chunks_only:
            chunks_file = root / "chunks_export.json"
            if chunks_file.exists():
                try:
                    chunks_file.unlink()
                    logger.info(f"Cleared: {chunks_file}")
                except PermissionError as exc:
                    logger.warning(f"Could not remove {chunks_file}: {exc}")
            return

        # ── Mode B: full reset with .json rescue + lock-tolerant rmtree ──
        rescued: dict[Path, bytes] = {}
        for json_file in root.rglob("*.json"):
            try:
                rescued[json_file.relative_to(root)] = json_file.read_bytes()
                logger.info(f"  Protected before --clear: {json_file}")
            except OSError as exc:
                logger.warning(f"  Could not read {json_file}: {exc}")

        # Best-effort rmtree. PermissionError on individual files (typical
        # cause: a still-held Kuzu .lock file) is logged but does not abort.
        def _on_error(func, path, exc_info):
            logger.warning(f"  Could not remove {path}: {exc_info[1]}")

        try:
            shutil.rmtree(root, onerror=_on_error)
            logger.info(f"Cleared (best-effort): {root}")
        except Exception as exc:
            logger.warning(f"Partial clear of {root}: {exc}")

        # Restore the rescued JSON files
        for relpath, data in rescued.items():
            target = root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            logger.info(f"  Restored: {target}")
    
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
        return paths["vector"].exists() and paths["questions"].exists()
    
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
    apply_coreference: bool = True,
) -> List[Document]:
    """
    Convert articles to LangChain Documents using SpacySentenceChunker.

    Pipeline:
      1. (optional) Coreference resolution per-article — replaces pronouns
         with their antecedent noun phrases so GLiNER can later capture the
         underlying named entity (Phase 2) and the graph captures the right
         cooccurrence pairs (Phase 3).
      2. SpacySentenceChunker (3-sentence sliding window) — primary path.
      3. Simple sentence-grouping fallback if SpaCy is unavailable.

    Coreference is silently skipped when `coreferee` or `en_core_web_md/lg`
    is not installed — the pipeline keeps working with reduced graph density.
    Pass apply_coreference=False to disable explicitly (e.g. for ablation).
    """

    # Lazy import — keeps benchmark_datasets.py functional when the data
    # layer is unavailable for non-ingestion subcommands.
    coref_resolver = None
    if apply_coreference:
        try:
            from src.data_layer.coreference import resolve_coreferences, is_available
            if is_available():
                coref_resolver = resolve_coreferences
                logger.info("Coreference resolution: ENABLED")
            else:
                logger.info("Coreference resolution: SKIPPED (coreferee or md/lg model missing)")
        except ImportError:
            logger.info("Coreference resolution: SKIPPED (module not importable)")

    # Method 1: SpacySentenceChunker (primary)
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
                # Apply coreference resolution if available — the chunker
                # then sees pronoun-resolved text and produces chunks where
                # GLiNER can re-identify the named entity behind every "He".
                article_text = article.text
                if coref_resolver is not None:
                    article_text = coref_resolver(article_text)
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
    
    # Method 2: Fallback (always available)
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
                f"| remaining ~{remaining/3600:.1f}h"
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
    """Compute exact match (EM) following the official HotpotQA metric.

    Rules (in order):
    1. Exact string match after normalisation.
    2. For multi-word gold answers (≥2 tokens): gold must appear as a
       contiguous word-boundary-anchored substring of the prediction
       (handles "the Eiffel Tower" inside "Eiffel Tower, Paris").
    3. For yes/no: the gold token must appear as a standalone word in
       the first 5 tokens of the prediction — NOT as a substring of a
       longer word (prevents "no" matching inside "I don't know").

    Deliberately NOT used: bare `gold in pred` substring check, which
    causes "no" to match "I don't know." as a false positive.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if not gold_norm:
        return False

    # Rule 1: exact match
    if pred_norm == gold_norm:
        return True

    # Rule 2: multi-word gold as contiguous phrase (word-boundary anchored)
    gold_tokens = gold_norm.split()
    if len(gold_tokens) >= 2:
        import re as _re
        pattern = r'\b' + _re.escape(gold_norm) + r'\b'
        if _re.search(pattern, pred_norm):
            return True

    # Rule 3: yes/no — standalone word match only
    if gold_norm in ("yes", "no"):
        pred_words = pred_norm.split()[:5]
        if gold_norm in pred_words:
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
    
    Implementation v4.1.0:
    - Uses: src.pipeline.agent_pipeline.AgentPipeline
    - Uses: create_full_pipeline factory
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
    pipeline_config["paths"]["vector"] = str(paths["vector"])
    pipeline_config["paths"]["graph_db"] = str(paths["graph"])

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
        vector_db_path=paths["vector"],
        graph_db_path=paths["graph"],
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

    # Wrap HybridStore in HybridRetriever (Navigator needs .retrieve()).
    # NOTE: vector_weight / graph_weight were removed from RetrievalConfig in the
    # 2026-05-06 cleanup audit (never read by production code — weighted-fusion
    # ablation is done via `mode`). The weights are still used above to pick the
    # RetrievalMode; they are not forwarded to the config.
    retrieval_config = RetrievalConfig(
        mode=retrieval_mode,
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
# EVALUATION RUNNER — pipeline vs. LLM failure separation
# ============================================================================
#
# The benchmark must answer two independent questions:
#   1. Did the pipeline (S_P + S_N) retrieve the correct supporting facts?
#      Measured by supporting-fact F1 against HotpotQA gold paragraphs.
#   2. Did the LLM (S_V) produce the correct answer GIVEN those facts?
#      Measured by EM/F1 restricted to questions where retrieval succeeded.
#
# These can fail independently. A timeout on the SLM is a model failure, not
# a pipeline failure. A missing gold paragraph is a pipeline failure, no
# matter how good the LLM is. The thesis argument depends on distinguishing
# the two.

# Module-level text→title index. Populated by the retriever monkey-patch
# below; consulted by _retrieved_titles_for_chunks() to look up the source
# title of a Navigator-filtered chunk. Keyed by the first 200 chars of chunk
# text (case-folded) so we tolerate downstream truncation.
_TEXT_TO_TITLE: Dict[str, str] = {}


def _text_key(text: str) -> str:
    return (text or "").strip()[:200].lower()


def _norm_title(title: str) -> str:
    """Lowercase, strip a leading 'hotpotqa_'/'2wiki_'/etc. dataset prefix,
    collapse whitespace. Mirrors the diagnose_verbose.py logic so SF
    comparison uses the same key space."""
    t = (title or "").strip().lower()
    if "_" in t:
        prefix, _, rest = t.partition("_")
        if prefix and " " not in prefix and rest:
            t = rest
    return " ".join(t.split())


def _gold_titles_from_supporting_facts(supporting_facts: List) -> List[str]:
    """HotpotQA/2WikiMHQA supporting_facts → set of normalised gold titles.

    supporting_facts is a list of (title, sent_id) tuples (see
    load_hotpotqa). We only need the unique titles."""
    seen, out = set(), []
    for entry in supporting_facts or []:
        title = entry[0] if isinstance(entry, (list, tuple)) and entry else str(entry)
        norm = _norm_title(title)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _install_retriever_title_capture(pipeline) -> Optional[callable]:
    """Monkey-patch HybridRetriever.retrieve() on the live pipeline so every
    raw RetrievalResult registers its (text→source_doc) mapping in
    _TEXT_TO_TITLE.

    Returns the original retrieve() so the caller can restore it after the
    benchmark run. None if no retriever is reachable from the pipeline.

    This is the inline alternative to threading source_doc through the
    Navigator/Verifier — the navigator strips chunks to text-only, so we
    rebuild the title lookup at the retriever boundary."""
    # Locate the HybridRetriever on the live pipeline. AgentPipeline exposes
    # it as `hybrid_retriever` (the canonical attribute used in
    # src/pipeline/agent_pipeline.py); older paths used `retriever`. The
    # Navigator stores it as `retriever` after set_retriever() is called.
    # Without this fallback chain matching ALL possible attribute names,
    # the patch silently no-ops and SF-F1 = 0.0 across every question —
    # which is exactly what was observed before this fix.
    retriever = None
    for attr in ("hybrid_retriever", "retriever", "_retriever"):
        candidate = getattr(pipeline, attr, None)
        if candidate is not None and hasattr(candidate, "retrieve"):
            retriever = candidate
            break
    if retriever is None:
        nav = getattr(pipeline, "navigator", None) or getattr(pipeline, "_navigator", None)
        if nav is not None:
            retriever = getattr(nav, "retriever", None) or getattr(nav, "_retriever", None)
    if retriever is None or not hasattr(retriever, "retrieve"):
        logger.warning(
            "Could not locate HybridRetriever on pipeline (tried "
            "hybrid_retriever / retriever / _retriever / navigator.retriever) "
            "— SF metrics will be 0. Pipeline attrs available: %s",
            [a for a in dir(pipeline) if not a.startswith('_')][:20],
        )
        return None
    logger.info("SF-title capture installed on %s.%s",
                type(pipeline).__name__,
                "hybrid_retriever" if retriever is getattr(pipeline, "hybrid_retriever", None)
                else "retriever")

    original = retriever.retrieve

    def _wrapped(*args, **kwargs):
        ret = original(*args, **kwargs)
        # HybridRetriever.retrieve() returns Tuple[List[RetrievalResult],
        # RetrievalMetrics]. Previous version of this patch iterated the
        # tuple directly and produced (RetrievalResult, RetrievalMetrics)
        # pairs — the metrics object has no .text / .source_doc, so every
        # capture silently fell through and SF-F1 was 0.0 across all
        # questions. The unpacking below is the actual fix.
        if isinstance(ret, tuple) and len(ret) >= 1:
            results_iter = ret[0]
        else:
            results_iter = ret
        try:
            for r in (results_iter or []):
                txt = (r.text if hasattr(r, "text")
                       else r.get("text", "") if isinstance(r, dict)
                       else "")
                src = (r.source_doc if hasattr(r, "source_doc")
                       else r.get("source_doc", "") if isinstance(r, dict)
                       else "")
                if txt and src:
                    _TEXT_TO_TITLE.setdefault(_text_key(txt), src)
        except Exception as exc:
            logger.debug("title-capture failed (non-fatal): %s", exc)
        return ret

    retriever.retrieve = _wrapped
    return original


def _retrieved_titles_for_chunks(chunks: List[str]) -> List[str]:
    """Map filtered-context chunks back to their source-doc titles using
    the text→title index built by the retriever monkey-patch."""
    out: List[str] = []
    seen: set = set()
    for c in chunks or []:
        title = _TEXT_TO_TITLE.get(_text_key(c)) or _TEXT_TO_TITLE.get(_text_key(c[:200]))
        if not title:
            continue
        norm = _norm_title(title)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _compute_sf_metrics(retrieved: List[str], gold: List[str]) -> Tuple[float, float, float, bool]:
    """Supporting-fact precision, recall, F1, and 'all_gold_retrieved' flag.

    HotpotQA-style: titles only, set-based (sent_id ignored — at chunk
    granularity we cannot resolve sentence-level supporting facts)."""
    gold_set = set(gold or [])
    retrieved_set = set(retrieved or [])
    if not gold_set:
        return 0.0, 0.0, 0.0, False
    if not retrieved_set:
        return 0.0, 0.0, 0.0, False
    tp = len(gold_set & retrieved_set)
    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    all_gold = gold_set.issubset(retrieved_set)
    return precision, recall, f1, all_gold


def _classify_llm_error(answer: str) -> Tuple[bool, str]:
    """Detect Verifier error sentinels.

    Verifier emits answers prefixed with '[Error:' on LLM-side failures
    (timeout, API error, no valid answer). Anything else is a real model
    response, even if substantively wrong."""
    if not answer or not answer.startswith("[Error:"):
        return False, ""
    low = answer.lower()
    if "timeout" in low:
        return True, "timeout"
    if "connect" in low or "ollama" in low:
        return True, "connection"
    if "api returned" in low:
        return True, "api"
    if "no valid answer" in low:
        return True, "no_answer"
    return True, "other"


def _extract_planner_diagnostics(planner_result: Dict[str, Any]) -> Tuple[str, int, int]:
    """(query_type, hop_count, n_entities) from PlannerResult.to_dict()."""
    if not isinstance(planner_result, dict):
        return "", 0, 0
    qt = planner_result.get("query_type", "") or ""
    hops = planner_result.get("hop_sequence", []) or []
    entities = planner_result.get("entities", []) or []
    return qt, len(hops), len(entities)


def _extract_verifier_diagnostics(verifier_result: Dict[str, Any]) -> Tuple[int, bool, str]:
    """(iterations, all_verified, confidence) from Verifier output."""
    if not isinstance(verifier_result, dict):
        return 0, False, ""
    iters = int(verifier_result.get("iterations", 0) or 0)
    allv = bool(verifier_result.get("all_verified", False))
    conf = str(verifier_result.get("confidence", "") or "")
    return iters, allv, conf


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
    jsonl_out: Optional[Path] = None,
    retrieval_only: bool = False,
) -> ConfigResult:
    """Evaluate dataset with given configuration.

    Args:
        jsonl_out: If given, append one JSON line per question to this file
            with the full EvalResult — for thesis analysis.
        retrieval_only: If True, skip the LLM (verifier) entirely and only
            measure pipeline retrieval quality. EM/F1 are forced to 0 in
            this mode; SF-F1/recall are the meaningful metrics.
    """

    results: List[EvalResult] = []

    # Reset text→title cache for this run and install retriever hook.
    _TEXT_TO_TITLE.clear()
    original_retrieve = _install_retriever_title_capture(pipeline)

    # Optional: disable verifier for retrieval-only evaluation.
    saved_enable_verifier = None
    if retrieval_only and hasattr(pipeline, "enable_verifier"):
        saved_enable_verifier = pipeline.enable_verifier
        pipeline.enable_verifier = False

    desc = f"Evaluating {dataset} [{config_name}]" + (" [retrieval-only]" if retrieval_only else "")

    try:
        for q in tqdm(questions, desc=desc, unit="q"):
            try:
                start = time.time()
                result = pipeline.process(q.question)
                elapsed = (time.time() - start) * 1000

                # ── Answer-level metrics (LLM correctness) ───────────────
                if retrieval_only:
                    em, f1 = False, 0.0
                    predicted = ""
                else:
                    em = compute_exact_match(result.answer, q.answer)
                    f1 = compute_f1(result.answer, q.answer)
                    predicted = result.answer

                # ── Retrieval-level metrics (pipeline correctness) ───────
                filtered_chunks: List[str] = []
                if hasattr(result, 'navigator_result'):
                    nav = result.navigator_result
                    if isinstance(nav, dict):
                        filtered_chunks = nav.get('filtered_context', []) or []
                retrieval_count = len(filtered_chunks)

                gold_titles = _gold_titles_from_supporting_facts(q.supporting_facts)
                retrieved_titles = _retrieved_titles_for_chunks(filtered_chunks)
                sf_p, sf_r, sf_f1, all_gold = _compute_sf_metrics(retrieved_titles, gold_titles)

                # ── Failure-mode separation ──────────────────────────────
                llm_err, llm_err_type = (False, "") if retrieval_only else _classify_llm_error(predicted)
                # Pipeline OK + LLM failed = retrieval was complete but model errored.
                pipeline_ok_llm_failed = bool(all_gold and llm_err)

                # ── Planner / Verifier diagnostics ───────────────────────
                p_qtype, hop_count, n_ents = _extract_planner_diagnostics(
                    getattr(result, "planner_result", {}) or {}
                )
                v_iters, v_verified, v_conf = _extract_verifier_diagnostics(
                    getattr(result, "verifier_result", {}) or {}
                )

                eval_result = EvalResult(
                    question_id=q.id,
                    question=q.question,
                    gold_answer=q.answer,
                    predicted_answer=predicted,
                    exact_match=em,
                    f1_score=f1,
                    retrieval_count=retrieval_count,
                    time_ms=elapsed,
                    dataset=q.dataset,
                    question_type=q.question_type,
                    gold_titles=gold_titles,
                    retrieved_titles=retrieved_titles,
                    retrieval_recall=sf_r,
                    retrieval_precision=sf_p,
                    sf_f1=sf_f1,
                    all_gold_retrieved=all_gold,
                    llm_error=llm_err,
                    llm_error_type=llm_err_type,
                    pipeline_succeeded_llm_failed=pipeline_ok_llm_failed,
                    planner_query_type=p_qtype,
                    hop_count=hop_count,
                    n_entities=n_ents,
                    verifier_iterations=v_iters,
                    all_verified=v_verified,
                    confidence=v_conf,
                )
                results.append(eval_result)

                # Per-question JSONL (one line per question) for thesis analysis.
                if jsonl_out is not None:
                    try:
                        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
                        with open(jsonl_out, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(asdict(eval_result), ensure_ascii=False) + "\n")
                    except Exception as exc:
                        logger.warning("JSONL write failed: %s", exc)

            except Exception as e:
                logger.warning(f"    Error on Q{q.id}: {str(e)[:80]}")
    finally:
        # Restore patched retriever and verifier flag so the pipeline isn't
        # permanently mutated (matters for ablation: same pipeline runs
        # multiple configurations).
        if original_retrieve is not None:
            for attr in ("hybrid_retriever", "retriever", "_retriever"):
                cand = getattr(pipeline, attr, None)
                if cand is not None and hasattr(cand, "retrieve"):
                    cand.retrieve = original_retrieve
                    break
            else:
                nav = getattr(pipeline, "navigator", None) or getattr(pipeline, "_navigator", None)
                if nav is not None:
                    inner = getattr(nav, "retriever", None) or getattr(nav, "_retriever", None)
                    if inner is not None:
                        inner.retrieve = original_retrieve
        if saved_enable_verifier is not None:
            pipeline.enable_verifier = saved_enable_verifier

    if not results:
        return None

    # ── Aggregate metrics ────────────────────────────────────────────────
    n = len(results)
    em_rate = sum(1 for r in results if r.exact_match) / n
    avg_f1 = sum(r.f1_score for r in results) / n
    avg_time = sum(r.time_ms for r in results) / n
    coverage = sum(1 for r in results if r.retrieval_count > 0) / n

    # Retrieval-level aggregates
    avg_sf_f1 = sum(r.sf_f1 for r in results) / n
    n_all_gold = sum(1 for r in results if r.all_gold_retrieved)
    sf_recall_rate = n_all_gold / n
    retrieval_only_em = (
        sum(1 for r in results if r.all_gold_retrieved and r.exact_match) / n_all_gold
        if n_all_gold > 0 else 0.0
    )
    llm_error_rate = sum(1 for r in results if r.llm_error) / n

    # Failure decomposition (mutually exclusive buckets, sum to 1.0):
    #   pipeline_failed       : not all_gold_retrieved
    #   pipeline_ok_llm_failed : all_gold AND llm error (timeout/etc.)
    #   pipeline_ok_llm_wrong  : all_gold AND no llm error AND not EM
    #   pipeline_ok_llm_ok     : all_gold AND EM
    pipeline_failed = sum(1 for r in results if not r.all_gold_retrieved) / n
    ok_llm_failed = sum(1 for r in results if r.all_gold_retrieved and r.llm_error) / n
    ok_llm_wrong = sum(
        1 for r in results
        if r.all_gold_retrieved and not r.llm_error and not r.exact_match
    ) / n
    ok_llm_ok = sum(
        1 for r in results if r.all_gold_retrieved and r.exact_match
    ) / n

    # By question type
    by_type: Dict[str, Dict] = {}
    for qtype in set(r.question_type for r in results):
        type_results = [r for r in results if r.question_type == qtype]
        tn = len(type_results)
        by_type[qtype] = {
            "count": tn,
            "exact_match": sum(1 for r in type_results if r.exact_match) / tn,
            "f1": sum(r.f1_score for r in type_results) / tn,
            "sf_f1": sum(r.sf_f1 for r in type_results) / tn,
            "sf_recall_rate": sum(1 for r in type_results if r.all_gold_retrieved) / tn,
            "llm_error_rate": sum(1 for r in type_results if r.llm_error) / tn,
        }

    return ConfigResult(
        dataset=dataset,
        config_name=config_name,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        n_questions=n,
        exact_match=em_rate,
        f1_score=avg_f1,
        avg_time_ms=avg_time,
        coverage=coverage,
        by_type=by_type,
        avg_sf_f1=avg_sf_f1,
        sf_recall_rate=sf_recall_rate,
        retrieval_only_em=retrieval_only_em,
        llm_error_rate=llm_error_rate,
        pipeline_failed_rate=pipeline_failed,
        pipeline_ok_llm_failed_rate=ok_llm_failed,
        pipeline_ok_llm_wrong_rate=ok_llm_wrong,
        pipeline_ok_llm_ok_rate=ok_llm_ok,
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
            # When --chunks-only is set we only want to wipe Phase-1 output;
            # the existing graph (often locked by KuzuDB on Windows) and the
            # Colab-produced extraction_results.json must NOT be touched.
            store_manager.clear_dataset(
                dataset,
                chunks_only=getattr(args, "chunks_only", False),
            )

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
                apply_coreference=not getattr(args, "no_coreference", False),
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
                logger.info(f"  [chunks-only] Exported: {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
                logger.info(f"  [chunks-only] Next step: upload file to Google Colab.")
                continue

            paths = store_manager.get_paths(dataset)
            run_ingestion(
                documents,
                paths["vector"],
                paths["graph"],
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

        retrieval_only = getattr(args, "retrieval_only", False)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_dir = Path("./evaluation_results")
        jsonl_path = jsonl_dir / f"{dataset}_{model_name.replace(':','-')}_{ts}.jsonl"
        # Truncate any prior partial file with the same name.
        if jsonl_path.exists():
            jsonl_path.unlink()

        result = evaluate_dataset(
            dataset, questions, pipeline,
            config_name, args.vector_weight, args.graph_weight,
            jsonl_out=jsonl_path,
            retrieval_only=retrieval_only,
        )

        # Print results
        logger.info(f"\n{'─'*70}")
        logger.info("RESULTS" + (" [retrieval-only]" if retrieval_only else ""))
        logger.info(f"{'─'*70}")
        logger.info(f"  Exact Match:           {result.exact_match:.2%}")
        logger.info(f"  F1 Score:              {result.f1_score:.3f}")
        logger.info(f"  Coverage:              {result.coverage:.2%}")
        logger.info(f"  Avg Time:              {result.avg_time_ms:.0f}ms")
        logger.info("")
        logger.info("  Pipeline (S_P + S_N) — retrieval quality:")
        logger.info(f"    Supporting-fact F1:  {result.avg_sf_f1:.3f}")
        logger.info(f"    All gold retrieved:  {result.sf_recall_rate:.2%}")
        if not retrieval_only:
            logger.info("")
            logger.info("  Failure decomposition (sum to 100%):")
            logger.info(f"    Pipeline failed:           {result.pipeline_failed_rate:.2%}  (gold not fully retrieved)")
            logger.info(f"    Pipeline ok, LLM failed:   {result.pipeline_ok_llm_failed_rate:.2%}  (timeout/api error)")
            logger.info(f"    Pipeline ok, LLM wrong:    {result.pipeline_ok_llm_wrong_rate:.2%}")
            logger.info(f"    Pipeline ok, LLM ok (EM):  {result.pipeline_ok_llm_ok_rate:.2%}")
            logger.info("")
            logger.info(f"  LLM error rate:        {result.llm_error_rate:.2%}")
            logger.info(f"  EM | all-gold-retrieved: {result.retrieval_only_em:.2%}  "
                        f"(model accuracy when retrieval succeeds)")

        if result.by_type:
            logger.info(f"\n  By Question Type:")
            for qtype, stats in result.by_type.items():
                logger.info(
                    f"    {qtype}: EM={stats['exact_match']:.2%} F1={stats['f1']:.3f} "
                    f"SF-F1={stats.get('sf_f1', 0.0):.3f} "
                    f"SF-Recall={stats.get('sf_recall_rate', 0.0):.2%} "
                    f"LLM-err={stats.get('llm_error_rate', 0.0):.2%}"
                )

        logger.info("")
        logger.info(f"  Per-question results: {jsonl_path}")
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

        retrieval_only = getattr(args, "retrieval_only", False)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_dir = Path("./evaluation_results")

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

                jsonl_path = jsonl_dir / (
                    f"{dataset}_{model_name.replace(':','-')}_{run_name}_{run_ts}.jsonl"
                )
                if jsonl_path.exists():
                    jsonl_path.unlink()

                result = evaluate_dataset(
                    dataset, questions, pipeline,
                    run_name, vector_weight, graph_weight,
                    jsonl_out=jsonl_path,
                    retrieval_only=retrieval_only,
                )

                if result:
                    dataset_results.append(result)
                    if run_name not in used_run_names:
                        used_run_names.append(run_name)
                    logger.info(
                        f"    EM: {result.exact_match:.2%}, F1: {result.f1_score:.3f}, "
                        f"SF-F1: {result.avg_sf_f1:.3f}, "
                        f"SF-Recall: {result.sf_recall_rate:.2%}, "
                        f"LLM-err: {result.llm_error_rate:.2%}, "
                        f"Latency: {result.avg_time_ms:.0f}ms"
                    )

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

                    jsonl_path = jsonl_dir / (
                        f"{dataset}_{model_name.replace(':','-')}_{run_name}_{run_ts}.jsonl"
                    )
                    if jsonl_path.exists():
                        jsonl_path.unlink()

                    result = evaluate_dataset(
                        dataset, questions, pipeline,
                        run_name, 0.5, 0.5,
                        jsonl_out=jsonl_path,
                        retrieval_only=retrieval_only,
                    )

                    if result:
                        dataset_results.append(result)
                        if run_name not in used_run_names:
                            used_run_names.append(run_name)
                        logger.info(
                            f"    EM: {result.exact_match:.2%}, F1: {result.f1_score:.3f}, "
                            f"SF-F1: {result.avg_sf_f1:.3f}, "
                            f"SF-Recall: {result.sf_recall_rate:.2%}, "
                            f"LLM-err: {result.llm_error_rate:.2%}, "
                            f"Latency: {result.avg_time_ms:.0f}ms"
                        )

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

    Baseline = first entry (vector_only or full).
    Positive = improvement per second of additional latency.
    None = latency difference too small for reliable measurement.
    """
    if not ds_results:
        return {}
    baseline = ds_results[0]
    mur: Dict[str, Optional[float]] = {baseline.config_name: None}
    for r in ds_results[1:]:
        delta_f1 = r.f1_score - baseline.f1_score
        delta_s = (r.avg_time_ms - baseline.avg_time_ms) / 1000.0
        if abs(delta_s) < 0.001:          # < 1ms difference → not measurable
            mur[r.config_name] = None
        elif delta_s > 0:
            mur[r.config_name] = delta_f1 / delta_s
        else:                              # Faster than baseline
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

        # SF-F1 row (pipeline retrieval quality)
        row = f"{'  (SF-F1)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.avg_sf_f1:>{col_w}.3f}" if r else f"{'':>{col_w}}"
        print(row)

        # SF-Recall (all-gold-retrieved rate)
        row = f"{'  (SF-Recall)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.sf_recall_rate:>{col_w}.1%}" if r else f"{'':>{col_w}}"
        print(row)

        # EM | all-gold-retrieved  (model accuracy when retrieval is correct)
        row = f"{'  (EM|retr.ok)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.retrieval_only_em:>{col_w}.1%}" if r else f"{'':>{col_w}}"
        print(row)

        # LLM error rate
        row = f"{'  (LLM-err)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.llm_error_rate:>{col_w}.1%}" if r else f"{'':>{col_w}}"
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
    print("Legend:")
    print("  EM/F1        = final answer correctness vs. gold (LLM output).")
    print("  SF-F1        = supporting-fact F1: did the pipeline retrieve the right paragraphs?")
    print("  SF-Recall    = % of questions where ALL gold supporting paragraphs were retrieved.")
    print("  EM|retr.ok   = EM among questions where retrieval succeeded (model accuracy")
    print("                 conditioned on correct retrieval — isolates LLM capability).")
    print("  LLM-err      = % of questions where the LLM returned a [Error:...] sentinel")
    print("                 (timeout/connection/api). Distinguishes pipeline failure from model failure.")
    print("  MUR          = ΔF1 / ΔLatency(s) vs. baseline (first config). Higher = better trade-off.")
    print()

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
                          help="Only build chunks and export as JSON, no ingestion")
    ingest_p.add_argument("--no-coreference", action="store_true",
                          help="Disable per-article coreference resolution. "
                               "Default: ON (requires coreferee + en_core_web_md/lg, "
                               "silently skipped if not installed)")
    
    # EVALUATE
    eval_p = subparsers.add_parser("evaluate", help="Evaluate single dataset")
    eval_p.add_argument("--dataset", "-d", type=str, required=True)
    eval_p.add_argument("--samples", "-n", type=int, default=100)
    eval_p.add_argument("--vector-weight", type=float, default=0.7)
    eval_p.add_argument("--graph-weight", type=float, default=0.3)
    eval_p.add_argument("--model", "-m", type=str, default=None,
                        help="Model name (e.g. phi3, llama3.2:3b). Default: from settings.yaml")
    eval_p.add_argument("--no-planner", action="store_true",
                        help="Skip S_P (ablation: no planner)")
    eval_p.add_argument("--no-verifier", action="store_true",
                        help="Skip S_V (ablation: no verifier)")
    eval_p.add_argument("--iterations", type=int, default=1,
                        help="Number of verifier iterations (1/2/3). Default: 1")
    eval_p.add_argument("--retrieval-only", action="store_true",
                        help="Skip the LLM entirely; only measure pipeline retrieval (SF-F1). "
                             "EM/F1 are forced to 0 in this mode. Useful for isolating "
                             "Planner/Navigator quality without LLM latency cost.")

    # ABLATION
    ablation_p = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_p.add_argument("--dataset", "-d", type=str, default="all")
    ablation_p.add_argument("--samples", "-n", type=int, default=100)
    ablation_p.add_argument("--model", "-m", type=str, default=None,
                            help="Model name (e.g. phi3, llama3.2:3b). Default: from settings.yaml")
    ablation_p.add_argument("--component-ablation", action="store_true",
                            help="Also run planner/verifier/iterations component ablation")
    ablation_p.add_argument("--retrieval-only", action="store_true",
                            help="Skip the LLM entirely across all ablation configs. Only "
                                 "supporting-fact metrics (SF-F1, SF-Recall) are meaningful. "
                                 "Much faster — useful for tuning retrieval-side config without "
                                 "paying LLM latency on every config.")
    
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