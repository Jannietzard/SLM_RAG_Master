"""
ingest_hotpotqa.py - Ingest HotpotQA Dataset for Benchmark Evaluation

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

This script:
1. Downloads HotpotQA from HuggingFace
2. Extracts Wikipedia articles (context)
3. Ingests them into your Vector Store + Knowledge Graph
4. Saves test questions for evaluation

Usage:
    pip install datasets  # First time only
    python ingest_hotpotqa.py
    python ingest_hotpotqa.py --samples 500  # Limit samples
    python ingest_hotpotqa.py --full          # Full dataset
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import yaml

# ============================================================================
# SETUP
# ============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_config(config_path: Path = Path("./config/settings.yaml")) -> Dict:
    """Load configuration."""
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TestQuestion:
    """A test question with gold answer."""
    id: str
    question: str
    answer: str
    question_type: str  # 'bridge' or 'comparison' for HotpotQA
    supporting_facts: List[Tuple[str, int]]  # [(article_title, sentence_idx), ...]
    level: str  # 'easy', 'medium', 'hard'


@dataclass 
class WikiArticle:
    """A Wikipedia article for ingestion."""
    title: str
    text: str
    sentences: List[str]
    source: str = "hotpotqa_wikipedia"


# ============================================================================
# HOTPOTQA LOADING
# ============================================================================

def load_hotpotqa(
    split: str = "validation",
    n_samples: int = None,
    logger: logging.Logger = None
) -> Tuple[List[WikiArticle], List[TestQuestion]]:
    """
    Load HotpotQA dataset from HuggingFace.
    
    Args:
        split: 'train', 'validation', or 'test'
        n_samples: Limit number of samples (None = all)
        logger: Logger instance
        
    Returns:
        Tuple of (articles, questions)
    """
    logger = logger or logging.getLogger(__name__)
    
    try:
        from Banchmark_datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed! Run: pip install datasets")
        sys.exit(1)
    
    logger.info(f"Loading HotpotQA ({split})...")
    logger.info("This may take a few minutes on first download...")
    
    # Load dataset
    # 'fullwiki' setting includes distractor paragraphs (harder, more realistic)
    # 'distractor' setting only includes relevant + distractor paragraphs
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    
    logger.info(f"Loaded {len(ds)} samples")
    
    # Extract articles and questions
    articles_dict = {}  # title -> WikiArticle (deduplicated)
    questions = []
    
    for idx, item in enumerate(ds):
        if idx % 100 == 0:
            logger.info(f"Processing {idx}/{len(ds)}...")
        
        # Extract question
        q = TestQuestion(
            id=item["id"],
            question=item["question"],
            answer=item["answer"],
            question_type=item["type"],
            supporting_facts=list(zip(
                item["supporting_facts"]["title"],
                item["supporting_facts"]["sent_id"]
            )),
            level=item["level"],
        )
        questions.append(q)
        
        # Extract Wikipedia articles from context
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        
        for title, sentences in zip(titles, sentences_list):
            if title not in articles_dict:
                # Join sentences into full text
                full_text = " ".join(sentences)
                
                articles_dict[title] = WikiArticle(
                    title=title,
                    text=full_text,
                    sentences=sentences,
                )
    
    articles = list(articles_dict.values())
    
    logger.info(f"Extracted {len(articles)} unique Wikipedia articles")
    logger.info(f"Extracted {len(questions)} test questions")
    
    return articles, questions


# ============================================================================
# DOCUMENT CREATION
# ============================================================================

def create_langchain_documents(
    articles: List[WikiArticle],
    chunk_sentences: int = 3,  # Sentences per chunk
    logger: logging.Logger = None
) -> List:
    """
    Convert WikiArticles to LangChain Documents with chunking.
    
    Args:
        articles: List of WikiArticle objects
        chunk_sentences: Number of sentences per chunk
        logger: Logger instance
        
    Returns:
        List of LangChain Document objects
    """
    from langchain.schema import Document
    
    logger = logger or logging.getLogger(__name__)
    
    documents = []
    chunk_id = 0
    
    for article in articles:
        sentences = article.sentences
        
        # Create chunks of N sentences
        for i in range(0, len(sentences), chunk_sentences):
            chunk_sents = sentences[i:i + chunk_sentences]
            chunk_text = " ".join(chunk_sents)
            
            # Skip very short chunks
            if len(chunk_text.strip()) < 50:
                continue
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": chunk_id,
                    "source_file": f"wikipedia_{article.title}",
                    "article_title": article.title,
                    "sentence_start": i,
                    "sentence_end": i + len(chunk_sents),
                    "source": "hotpotqa_wikipedia",
                }
            )
            documents.append(doc)
            chunk_id += 1
    
    logger.info(f"Created {len(documents)} document chunks")
    
    return documents


# ============================================================================
# INGESTION
# ============================================================================

def run_ingestion(
    documents: List,
    config: Dict,
    logger: logging.Logger
) -> None:
    """
    Ingest documents into Vector Store + Knowledge Graph.
    
    Uses your existing storage infrastructure.
    """
    logger.info("="*70)
    logger.info("STARTING INGESTION")
    logger.info("="*70)
    
    # Import your modules
    try:
        from src.data_layer.embeddings import BatchedOllamaEmbeddings
        from src.data_layer.storage import HybridStore, StorageConfig
    except ImportError as e:
        logger.error(f"Cannot import modules: {e}")
        sys.exit(1)
    
    # Initialize embeddings
    logger.info("Initializing embeddings...")
    embedding_config = config.get("embeddings", {})
    perf_config = config.get("performance", {})
    
    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=perf_config.get("batch_size", 32),
        cache_path=Path("./cache/embeddings.db"),
        device=perf_config.get("device", "cpu"),
    )
    
    # Initialize storage
    logger.info("Initializing storage...")
    vector_config = config.get("vector_store", {})
    
    # Create directories
    vector_db_path = Path(config.get("paths", {}).get("vector_db", "./data/vector_db"))
    graph_db_path = Path(config.get("paths", {}).get("graph_db", "./data/knowledge_graph"))
    
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    graph_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    storage_config = StorageConfig(
        vector_db_path=vector_db_path,
        graph_db_path=graph_db_path,
        embedding_dim=embedding_config.get("embedding_dim", 768),
        similarity_threshold=vector_config.get("similarity_threshold", 0.3),
        normalize_embeddings=vector_config.get("normalize_embeddings", True),
        distance_metric=vector_config.get("distance_metric", "cosine"),
    )
    
    hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)
    
    # Ingest documents
    logger.info(f"Ingesting {len(documents)} documents...")
    start_time = time.time()
    
    # Process in batches to show progress
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        hybrid_store.add_documents(batch)
        logger.info(f"  Ingested {min(i + batch_size, len(documents))}/{len(documents)}")
    
    # Save
    hybrid_store.save()
    
    elapsed = time.time() - start_time
    logger.info(f"Ingestion complete in {elapsed:.1f}s")
    
    # Print stats
    logger.info("="*70)
    logger.info("INGESTION SUMMARY")
    logger.info("="*70)
    logger.info(f"  Documents: {len(documents)}")
    logger.info(f"  Vector Store: {vector_db_path}")
    logger.info(f"  Knowledge Graph: {graph_db_path}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("="*70)


# ============================================================================
# SAVE TEST QUESTIONS
# ============================================================================

def save_test_questions(
    questions: List[TestQuestion],
    output_path: Path = Path("./data/hotpotqa_test_questions.json"),
    logger: logging.Logger = None
) -> None:
    """Save test questions for later evaluation."""
    logger = logger or logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dicts
    questions_dict = [asdict(q) for q in questions]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(questions)} test questions to {output_path}")
    
    # Also save a summary
    summary = {
        "total_questions": len(questions),
        "by_type": {},
        "by_level": {},
    }
    
    for q in questions:
        summary["by_type"][q.question_type] = summary["by_type"].get(q.question_type, 0) + 1
        summary["by_level"][q.level] = summary["by_level"].get(q.level, 0) + 1
    
    summary_path = output_path.parent / "hotpotqa_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Question types: {summary['by_type']}")
    logger.info(f"Question levels: {summary['by_level']}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest HotpotQA for benchmark evaluation"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=500,
        help="Number of samples to load (default: 500)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load full dataset (overrides --samples)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to use (default: validation)"
    )
    parser.add_argument(
        "--chunk-sentences",
        type=int,
        default=3,
        help="Sentences per chunk (default: 3)"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Only download and save questions (skip vector store)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("INFO")
    
    print("\n" + "="*70)
    print("HOTPOTQA INGESTION FOR BENCHMARK EVALUATION")
    print("="*70)
    print(f"Samples: {'FULL' if args.full else args.samples}")
    print(f"Split: {args.split}")
    print(f"Chunk size: {args.chunk_sentences} sentences")
    print("="*70 + "\n")
    
    # Load config
    config = load_config()
    
    # Determine sample count
    n_samples = None if args.full else args.samples
    
    # Load HotpotQA
    articles, questions = load_hotpotqa(
        split=args.split,
        n_samples=n_samples,
        logger=logger
    )
    
    # Save test questions (always)
    save_test_questions(questions, logger=logger)
    
    if args.skip_ingestion:
        logger.info("Skipping ingestion (--skip-ingestion flag)")
        return
    
    # Create document chunks
    documents = create_langchain_documents(
        articles,
        chunk_sentences=args.chunk_sentences,
        logger=logger
    )
    
    # Run ingestion
    run_ingestion(documents, config, logger)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Run evaluation: python evaluate_hotpotqa.py")
    print(f"  2. Or test manually: python main_agentic.py --interactive")
    print(f"\nTest questions saved to: ./data/hotpotqa_test_questions.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()