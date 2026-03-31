"""
═══════════════════════════════════════════════════════════════════════════════
LOCAL IMPORT INGESTION - Phase 3 der entkoppelten Ingestion
═══════════════════════════════════════════════════════════════════════════════

Dieses Skript importiert die Ergebnisse der Colab-Extraktion in die lokalen
Stores (LanceDB + KuzuDB).

Input:
    - chunks_export.json      (Phase 1: Chunks aus benchmark_datasets.py)
    - extraction_results.json (Phase 2: Entities + Relations aus Colab)

Output:
    - data/<dataset>/vector_db/       (LanceDB)
    - data/<dataset>/knowledge_graph/  (KuzuDB)

Usage:
    python local_import_ingestion.py \\
        --chunks data/hotpotqa/chunks_export.json \\
        --extractions extraction_results.json \\
        --dataset hotpotqa

    # Mit bestehendem config
    python local_import_ingestion.py \\
        --chunks data/hotpotqa/chunks_export.json \\
        --extractions extraction_results.json \\
        --dataset hotpotqa \\
        --config config/settings.yaml

    # Nur Graph-Import (Vector Store existiert bereits)
    python local_import_ingestion.py \\
        --chunks data/hotpotqa/chunks_export.json \\
        --extractions extraction_results.json \\
        --dataset hotpotqa \\
        --graph-only

═══════════════════════════════════════════════════════════════════════════════

Version: 1.0.0 - Decoupled Ingestion Phase 3
Author: Edge-RAG Research Project
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import yaml

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Quiet sub-modules
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ============================================================================
# IMPORTS MIT FALLBACK
# ============================================================================

try:
    from langchain.schema import Document
except ImportError:
    @dataclass
    class Document:
        page_content: str
        metadata: dict

try:
    from src.data_layer.storage import HybridStore, StorageConfig, KuzuGraphStore
    from src.data_layer.embeddings import BatchedOllamaEmbeddings
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.error(
        "Storage-Module nicht gefunden! "
        "Stelle sicher, dass du im Projekt-Root bist und "
        "src/data_layer/storage.py existiert."
    )

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable
        def __iter__(self):
            return iter(self._iterable) if self._iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kwargs):
            pass


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Lade Config aus YAML oder verwende Defaults."""
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # Versuche Standard-Pfad
    default_path = Path("./config/settings.yaml")
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    logger.warning("Keine Config gefunden — verwende Defaults")
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
        "performance": {
            "batch_size": 32,
            "device": "cpu",
        },
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunks(chunks_path: Path) -> List[Dict]:
    """Lade chunks_export.json."""
    logger.info(f"Lade Chunks: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"  {len(chunks)} Chunks geladen")
    return chunks


def load_extractions(extractions_path: Path) -> Dict:
    """Lade extraction_results.json."""
    logger.info(f"Lade Extraktionsergebnisse: {extractions_path}")
    with open(extractions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    results = data.get("results", [])

    logger.info(f"  {meta.get('total_chunks', '?')} Chunks verarbeitet")
    logger.info(f"  {meta.get('total_entities', '?')} Entities extrahiert")
    logger.info(f"  {meta.get('unique_entities', '?')} unique Entities")
    logger.info(f"  {meta.get('total_relations', '?')} Relationen extrahiert")
    logger.info(f"  Device: {meta.get('device', '?')}")
    logger.info(f"  NER: {meta.get('ner_time_seconds', '?')}s, RE: {meta.get('re_time_seconds', '?')}s")

    return data


def chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    """Konvertiere chunks_export.json-Format zu LangChain Documents."""
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        documents.append(doc)
    return documents


# ============================================================================
# PHASE 3a: VECTOR STORE INGESTION (Embeddings via Ollama)
# ============================================================================

def ingest_vector_store(
    documents: List[Document],
    vector_path: Path,
    config: Dict,
    dataset_name: str,
) -> None:
    """
    Ingestiere Chunks in LanceDB Vector Store.

    Verwendet BatchedOllamaEmbeddings für Embedding-Generierung.
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage-Module nicht verfügbar!")
        return

    logger.info(f"\n{'─'*70}")
    logger.info(f"PHASE 3a: VECTOR STORE INGESTION ({len(documents)} Chunks)")
    logger.info(f"{'─'*70}")

    embedding_config = config.get("embeddings", {})
    perf_config = config.get("performance", {})

    # Embedding-Cache
    cache_path = Path(f"./cache/{dataset_name}_embeddings.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=perf_config.get("batch_size", 32),
        cache_path=cache_path,
        device=perf_config.get("device", "cpu"),
    )

    # StorageConfig OHNE Entity Extraction (wir machen das separat)
    vector_config = config.get("vector_store", {})
    storage_config = StorageConfig(
        vector_db_path=vector_path,
        graph_db_path=vector_path.parent / "knowledge_graph",  # wird hier nicht genutzt
        embedding_dim=embedding_config.get("embedding_dim", 768),
        similarity_threshold=vector_config.get("similarity_threshold", 0.3),
        normalize_embeddings=vector_config.get("normalize_embeddings", True),
        distance_metric=vector_config.get("distance_metric", "cosine"),
        enable_entity_extraction=False,  # DEAKTIVIERT — kommt aus Colab
    )

    # Nur VectorStoreAdapter verwenden
    from src.data_layer.storage import VectorStoreAdapter
    vector_store = VectorStoreAdapter(
        db_path=str(vector_path),
        embedding_dim=storage_config.embedding_dim,
        distance_metric=storage_config.distance_metric,
    )

    # Batch-Ingestion
    start_time = time.time()
    batch_size = 100

    for i in tqdm(
        range(0, len(documents), batch_size),
        desc="Vector Store",
        unit="batch",
    ):
        batch = documents[i : i + batch_size]
        vector_store.add_documents_with_embeddings(batch, embeddings)

    elapsed = time.time() - start_time
    logger.info(f"  ✓ Vector Store: {len(documents)} Chunks in {elapsed:.1f}s")
    logger.info(f"    Pfad: {vector_path}")


# ============================================================================
# PHASE 3b: KNOWLEDGE GRAPH INGESTION (KuzuDB)
# ============================================================================

def ingest_knowledge_graph(
    documents: List[Document],
    extraction_results: List[Dict],
    graph_path: Path,
    dataset_name: str,
) -> Dict[str, int]:
    """
    Importiere Entities und Relationen in KuzuDB Knowledge Graph.

    Steps:
        1. DocumentChunk-Nodes erstellen
        2. SourceDocument-Nodes erstellen
        3. FROM_SOURCE + NEXT_CHUNK Relationen
        4. Entity-Nodes aus Extraktionsergebnissen
        5. MENTIONS Relationen (Chunk → Entity)
        6. RELATED_TO Relationen (Entity → Entity)
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage-Module nicht verfügbar!")
        return {}

    logger.info(f"\n{'─'*70}")
    logger.info(f"PHASE 3b: KNOWLEDGE GRAPH INGESTION")
    logger.info(f"{'─'*70}")

    # KuzuDB Graph Store initialisieren
    graph_store = KuzuGraphStore(str(graph_path))

    stats = {
        "document_chunks": 0,
        "source_documents": 0,
        "from_source": 0,
        "next_chunk": 0,
        "entities": 0,
        "unique_entities": 0,
        "mentions": 0,
        "relations": 0,
    }

    # Index: chunk_id → extraction result
    extraction_by_chunk = {}
    for result in extraction_results:
        extraction_by_chunk[str(result["chunk_id"])] = result

    # ── Step 1-3: Document Structure ─────────────────────────────────────

    logger.info("  Schritt 1-3: Dokumentstruktur aufbauen...")
    seen_sources = set()
    prev_chunk_id = None

    for doc in tqdm(documents, desc="Graph Nodes", unit="doc"):
        chunk_id = str(doc.metadata.get("chunk_id", "unknown"))
        source_file = doc.metadata.get("source_file", "unknown")

        # DocumentChunk Node
        try:
            graph_store.add_document_chunk(
                chunk_id=chunk_id,
                text=doc.page_content[:500],
                page_number=doc.metadata.get("page_number", 0),
                chunk_index=doc.metadata.get("chunk_index", 0),
                source_file=source_file,
            )
            stats["document_chunks"] += 1
        except Exception as e:
            logger.debug(f"    Chunk {chunk_id}: {e}")

        # SourceDocument Node (nur einmal pro source)
        if source_file not in seen_sources:
            try:
                graph_store.add_source_document(
                    doc_id=source_file,
                    filename=source_file,
                    total_pages=int(doc.metadata.get("total_pages", 0)),
                )
                seen_sources.add(source_file)
                stats["source_documents"] += 1
            except Exception as e:
                logger.debug(f"    Source {source_file}: {e}")

        # FROM_SOURCE Relation
        try:
            graph_store.add_from_source_relation(chunk_id, source_file)
            stats["from_source"] += 1
        except Exception as e:
            logger.debug(f"    FROM_SOURCE: {e}")

        # NEXT_CHUNK Relation (nur innerhalb desselben Source-Dokuments)
        if prev_chunk_id is not None:
            prev_doc = documents[stats["document_chunks"] - 2] if stats["document_chunks"] > 1 else None
            if prev_doc and prev_doc.metadata.get("source_file") == source_file:
                try:
                    graph_store.add_next_chunk_relation(prev_chunk_id, chunk_id)
                    stats["next_chunk"] += 1
                except Exception:
                    pass

        prev_chunk_id = chunk_id

    logger.info(f"    ✓ {stats['document_chunks']} Chunks, {stats['source_documents']} Sources")

    # ── Step 4-6: Entities & Relations ───────────────────────────────────

    logger.info("  Schritt 4-6: Entities und Relationen importieren...")

    seen_entities = set()
    entity_name_to_id = {}

    for result in tqdm(extraction_results, desc="Entities & Relations", unit="chunk"):
        chunk_id = str(result["chunk_id"])

        # Step 4: Entity Nodes
        for ent in result.get("entities", []):
            entity_id = ent["entity_id"]
            entity_name = ent["name"]
            entity_type = ent.get("entity_type") or ent.get("type", "UNKNOWN")
            confidence = ent.get("confidence", 0.5)

            entity_name_to_id[entity_name.lower()] = entity_id

            if entity_id not in seen_entities:
                try:
                    graph_store.add_entity(
                        entity_id=entity_id,
                        name=entity_name,
                        entity_type=entity_type,
                        confidence=confidence,
                    )
                    seen_entities.add(entity_id)
                    stats["unique_entities"] += 1
                except Exception as e:
                    logger.debug(f"    Entity {entity_id}: {e}")

            stats["entities"] += 1

            # Step 5: MENTIONS Relation (Chunk → Entity)
            try:
                graph_store.add_mentions_relation(
                    chunk_id=chunk_id,
                    entity_id=entity_id,
                )
                stats["mentions"] += 1
            except Exception as e:
                logger.debug(f"    MENTIONS {chunk_id}→{entity_id}: {e}")

        # Step 6: RELATED_TO Relations (Entity → Entity)
        for rel in result.get("relations", []):
            subject = rel.get("subject_entity") or rel.get("subject", "")
            obj = rel.get("object_entity") or rel.get("object", "")
            rel_type = rel.get("relation_type") or rel.get("relation", "related_to")

            subject_id = entity_name_to_id.get(subject.lower())
            object_id = entity_name_to_id.get(obj.lower())

            # Wenn Entity-IDs nicht direkt gefunden: versuche Substring-Match
            if not subject_id:
                for name, eid in entity_name_to_id.items():
                    if subject.lower() in name or name in subject.lower():
                        subject_id = eid
                        break
            if not object_id:
                for name, eid in entity_name_to_id.items():
                    if obj.lower() in name or name in obj.lower():
                        object_id = eid
                        break

            if subject_id and object_id:
                try:
                    graph_store.add_related_to_relation(
                        entity1_id=subject_id,
                        entity2_id=object_id,
                        relation_type=rel_type,
                    )
                    stats["relations"] += 1
                except Exception as e:
                    logger.debug(f"    RELATED_TO: {e}")

    logger.info(f"    ✓ {stats['unique_entities']} Entities, "
                f"{stats['mentions']} Mentions, {stats['relations']} Relations")

    return stats


# ============================================================================
# FULL IMPORT PIPELINE
# ============================================================================

def run_full_import(
    chunks_path: Path,
    extractions_path: Path,
    dataset_name: str,
    config: Dict,
    graph_only: bool = False,
    clear: bool = False,
) -> None:
    """
    Führe den kompletten Import durch.

    Args:
        chunks_path: Pfad zu chunks_export.json
        extractions_path: Pfad zu extraction_results.json
        dataset_name: Name des Datasets (z.B. "hotpotqa")
        config: Konfiguration
        graph_only: Nur Graph importieren, Vector Store überspringen
        clear: Bestehende Daten löschen
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage-Module nicht verfügbar — Abbruch")
        sys.exit(1)

    total_start = time.time()

    print()
    print("═" * 70)
    print("DECOUPLED INGESTION — PHASE 3: LOKALER IMPORT")
    print("═" * 70)
    print(f"  Dataset:      {dataset_name}")
    print(f"  Chunks:       {chunks_path}")
    print(f"  Extractions:  {extractions_path}")
    print(f"  Graph-Only:   {graph_only}")
    print("═" * 70)

    # Pfade
    base_path = Path("./data") / dataset_name
    vector_path = base_path / "vector_db"
    graph_path = base_path / "knowledge_graph"

    # Clear wenn gewünscht
    if clear:
        import shutil
        if base_path.exists():
            shutil.rmtree(base_path)
            logger.info(f"  Gelöscht: {base_path}")

    base_path.mkdir(parents=True, exist_ok=True)

    # Daten laden
    chunks = load_chunks(chunks_path)
    extraction_data = load_extractions(extractions_path)
    extraction_results = extraction_data.get("results", [])

    # Validierung: Chunk-Anzahl muss übereinstimmen
    if len(chunks) != len(extraction_results):
        logger.warning(
            f"  ⚠ Chunk-Anzahl stimmt nicht überein! "
            f"Chunks: {len(chunks)}, Extractions: {len(extraction_results)}"
        )
        logger.warning("  Fahre trotzdem fort (verwende Schnittmenge nach chunk_id)")

    documents = chunks_to_documents(chunks)

    # Phase 3a: Vector Store
    if not graph_only:
        try:
            ingest_vector_store(documents, vector_path, config, dataset_name)
        except Exception as e:
            logger.error(f"Vector Store Ingestion fehlgeschlagen: {e}")
            logger.error("Versuche --graph-only wenn Vector Store separat erstellt werden soll")
            raise
    else:
        logger.info("  Vector Store übersprungen (--graph-only)")

    # Phase 3b: Knowledge Graph
    try:
        stats = ingest_knowledge_graph(
            documents, extraction_results, graph_path, dataset_name
        )
    except Exception as e:
        logger.error(f"Knowledge Graph Ingestion fehlgeschlagen: {e}")
        raise

    # Zusammenfassung
    total_elapsed = time.time() - total_start

    print()
    print("═" * 70)
    print("IMPORT ABGESCHLOSSEN")
    print("═" * 70)
    print(f"  Gesamtzeit:       {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Vector Store:     {vector_path}")
    print(f"  Knowledge Graph:  {graph_path}")
    print()
    print("  Graph-Statistiken:")
    for key, val in stats.items():
        print(f"    {key:<20}: {val:>8,}")
    print()
    print("═" * 70)
    print("  Nächste Schritte:")
    print(f"    python benchmark_datasets.py evaluate --dataset {dataset_name} --samples 100")
    print(f"    python benchmark_datasets.py ablation --dataset {dataset_name} --samples 100")
    print("═" * 70)

    # Extraction-Metadaten speichern
    meta_path = base_path / "extraction_metadata.json"
    meta = extraction_data.get("metadata", {})
    meta["import_time_seconds"] = round(total_elapsed, 1)
    meta["graph_stats"] = stats
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"  Metadaten gespeichert: {meta_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decoupled Ingestion Phase 3: Lokaler Import",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard-Import (Vector Store + Knowledge Graph)
  python local_import_ingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions extraction_results.json \\
      --dataset hotpotqa

  # Nur Knowledge Graph (Vector Store existiert schon)
  python local_import_ingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions extraction_results.json \\
      --dataset hotpotqa \\
      --graph-only

  # Mit YAML Config
  python local_import_ingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions extraction_results.json \\
      --dataset hotpotqa \\
      --config config/settings.yaml
        """,
    )

    parser.add_argument(
        "--chunks", "-c",
        type=Path,
        required=True,
        help="Pfad zu chunks_export.json (Phase 1 Output)",
    )
    parser.add_argument(
        "--extractions", "-e",
        type=Path,
        required=True,
        help="Pfad zu extraction_results.json (Phase 2 / Colab Output)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Dataset-Name (z.B. hotpotqa, 2wikimultihop)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Pfad zu settings.yaml (optional)",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Nur Knowledge Graph importieren, Vector Store überspringen",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Bestehende Daten vor Import löschen",
    )

    args = parser.parse_args()

    # Validierung
    if not args.chunks.exists():
        logger.error(f"Chunks-Datei nicht gefunden: {args.chunks}")
        sys.exit(1)
    if not args.extractions.exists():
        logger.error(f"Extraktions-Datei nicht gefunden: {args.extractions}")
        sys.exit(1)

    config = load_config(args.config)

    run_full_import(
        chunks_path=args.chunks,
        extractions_path=args.extractions,
        dataset_name=args.dataset,
        config=config,
        graph_only=args.graph_only,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()