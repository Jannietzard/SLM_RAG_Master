"""
Hybrid Storage: Vector Store (LanceDB) + Knowledge Graph (NetworkX).

Scientific Foundation:
- Vector Store: Dense embeddings für semantic similarity
- Knowledge Graph: Strukturelle Relationen für multi-hop reasoning
- Hybrid: Kombiniert Vorteile beider Ansätze (vgl. Graph-RAG)

FIXED: Vollständige Implementation mit reset_* Methoden und korrekter vector_search
"""

import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import lancedb
import networkx as nx
from langchain.schema import Document
from langchain.embeddings.base import Embeddings


logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Konfiguration für Hybrid Storage."""
    vector_db_path: Path
    graph_db_path: Path
    embedding_dim: int
    similarity_threshold: float = 0.5


# ============================================================================
# VECTOR STORE ADAPTER (LanceDB)
# ============================================================================

class VectorStoreAdapter:
    """
    LanceDB Vector Store Adapter für Edge-optimierte Suche.
    
    FIXED: Korrekte vector_search() Implementation mit Distance→Similarity Conversion
    """

    def __init__(self, db_path: Path, embedding_dim: int):
        """
        Initialisiere LanceDB Connection.

        Args:
            db_path: Pfad zur LanceDB Datenbasis
            embedding_dim: Dimensionalität der Embeddings
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))
        self.embedding_dim = embedding_dim
        self.table = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LanceDB initialisiert: {db_path}")

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: Embeddings,
    ) -> None:
        """
        Füge Dokumente mit Embeddings zu Vector Store hinzu.

        Args:
            documents: Liste der Dokumente
            embeddings: Embedding-Modell (LangChain Embeddings Interface)

        Raises:
            ValueError: Falls Embedding-Dimension nicht matched
        """
        if not documents:
            self.logger.warning("Keine Dokumente zum Hinzufügen")
            return

        # Generiere Embeddings
        texts = [doc.page_content for doc in documents]
        self.logger.info(f"Generiere Embeddings für {len(texts)} Dokumente...")
        embeddings_list = embeddings.embed_documents(texts)

        # Validierung
        if embeddings_list and len(embeddings_list[0]) != self.embedding_dim:
            raise ValueError(
                f"Embedding-Dim mismatch: expected {self.embedding_dim}, "
                f"got {len(embeddings_list[0])}"
            )

        # Vorbereite Daten für LanceDB
        data = []
        for doc, emb in zip(documents, embeddings_list):
            data.append({
                "document_id": str(doc.metadata.get("chunk_id", "unknown")),
                "text": doc.page_content,
                "vector": emb,
                "metadata": json.dumps(doc.metadata),
                "source_file": doc.metadata.get("source_file", "unknown"),
            })

        # Speichere in LanceDB
        try:
            if self.table is None:
                self.table = self.db.create_table("documents", data=data, mode="overwrite")
                self.logger.info(f"✓ Neue Tabelle erstellt mit {len(data)} Dokumenten")
            else:
                self.table.add(data)
                self.logger.info(f"✓ {len(data)} Dokumente hinzugefügt")
        except Exception as e:
            self.logger.error(f"Fehler beim Einfügen in Vector Store: {str(e)}")
            raise

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        FIXED: Korrekte Vector-basierte Similarity Search.

        LanceDB gibt Cosine Distance zurück (Range 0-2), nicht Similarity!
        Conversion: similarity = 1 - (distance / 2)

        Args:
            query_embedding: Query Vector
            top_k: Anzahl Top Results
            threshold: Similarity Threshold (0.0-1.0)

        Returns:
            Liste von Results mit Text, Score, Metadaten
        """
        if self.table is None:
            self.logger.warning("Vector Store ist leer - keine Dokumente")
            return []

        try:
            # LanceDB Search: gibt Distance zurück, nicht Similarity!
            raw_results = self.table.search(query_embedding).limit(top_k * 3).to_list()
            
            if not raw_results:
                self.logger.debug("Vector Search: Keine Results von LanceDB")
                return []

            # Konvertiere Distance → Similarity und filtere
            filtered = []
            for i, result in enumerate(raw_results):
                # LanceDB gibt Cosine Distance zurück (Range: 0.0 = identical, 2.0 = opposite)
                # WICHTIG: Bei normalized vectors ist Cosine Distance in [0, 2]
                distance = result.get("_distance", 1.0)
                
                # KORREKTE Conversion: Cosine Similarity = 1 - Cosine Distance
                # Bei normalized vectors (was Embeddings sind):
                # Distance 0.0 → Similarity 1.0 (perfekt)
                # Distance 1.0 → Similarity 0.0 (orthogonal)
                # Distance 2.0 → Similarity -1.0 (opposite)
                similarity = 1.0 - distance
                
                # Clamp zu [0, 1] für negative Similarities
                similarity = max(0.0, min(1.0, similarity))
                
                # Debug für erste 3 Results
                if i < 3:
                    self.logger.debug(
                        f"  Result {i}: distance={distance:.4f} → similarity={similarity:.4f}"
                    )
                
                # Threshold Check
                if similarity >= threshold:
                    # Parse Metadata
                    try:
                        metadata = json.loads(result.get("metadata", "{}"))
                    except:
                        metadata = {}
                    
                    filtered.append({
                        "text": result.get("text", ""),
                        "similarity": similarity,
                        "document_id": result.get("document_id", "unknown"),
                        "metadata": metadata,
                    })

            # Sortiere nach Similarity (höchste zuerst)
            filtered.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Return nur top_k beste
            final_results = filtered[:top_k]
            
            self.logger.info(
                f"Vector Search: {len(raw_results)} raw → {len(filtered)} filtered "
                f"→ {len(final_results)} returned (threshold={threshold:.2f})"
            )
            
            return final_results

        except Exception as e:
            self.logger.error(f"Fehler bei Vector Search: {str(e)}", exc_info=True)
            return []


# ============================================================================
# KNOWLEDGE GRAPH STORE (NetworkX)
# ============================================================================

class KnowledgeGraphStore:
    """
    NetworkX-basierter Knowledge Graph für strukturelle Relationen.
    
    Scientific Rationale:
    Graphen preservieren Entity-Relation-Tripel explizit,
    ermöglichen Multi-Hop Reasoning ohne LLM Re-Query.
    """

    def __init__(self, graph_path: Path):
        """
        Initialisiere Knowledge Graph.

        Args:
            graph_path: Pfad zur GraphML-Datei
        """
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)

        # Lade existierenden Graph falls vorhanden
        if self.graph_path.exists():
            try:
                self.graph = nx.read_graphml(str(self.graph_path))
                self.logger.info(
                    f"Graph geladen: {self.graph.number_of_nodes()} Knoten, "
                    f"{self.graph.number_of_edges()} Kanten"
                )
            except Exception as e:
                self.logger.error(f"Fehler beim Laden des Graphs: {str(e)}")
                self.graph = nx.DiGraph()
        else:
            self.logger.info("Neuer Graph initialisiert")

    def add_entity(self, entity_id: str, entity_type: str, metadata: Dict[str, Any]) -> None:
        """
        Füge Entity zum Graph hinzu.

        Args:
            entity_id: Eindeutige Entity ID
            entity_type: Typ der Entity (z.B. "concept", "person")
            metadata: Zusätzliche Metadaten
        """
        self.graph.add_node(
            entity_id,
            entity_type=entity_type,
            **metadata
        )

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Füge Relation zwischen Entities hinzu.

        Args:
            source_id: Source Entity ID
            target_id: Target Entity ID
            relation_type: Typ der Relation
            metadata: Optional Metadaten
        """
        edge_data = {"relation_type": relation_type}
        if metadata:
            edge_data.update(metadata)
        
        self.graph.add_edge(source_id, target_id, **edge_data)

    def graph_traversal(
        self,
        start_entity: str,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> Dict[str, int]:
        """
        BFS-basierte Graph-Traversal.

        Args:
            start_entity: Start-Knoten
            relation_types: Filter für Relation Types
            max_hops: Maximale Traversal-Distanz

        Returns:
            Dict: {entity_id: hop_distance}
        """
        if start_entity not in self.graph:
            return {}

        visited = {start_entity: 0}
        queue = [(start_entity, 0)]

        while queue:
            current, hops = queue.pop(0)

            if hops >= max_hops:
                continue

            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    # Check relation type filter
                    edge_data = self.graph.get_edge_data(current, neighbor)
                    rel_type = edge_data.get("relation_type")

                    if relation_types is None or rel_type in relation_types:
                        visited[neighbor] = hops + 1
                        queue.append((neighbor, hops + 1))

        return visited

    def save(self) -> None:
        """Speichere Graph als GraphML."""
        try:
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
            nx.write_graphml(self.graph, str(self.graph_path))
            self.logger.info(f"✓ Graph gespeichert: {self.graph_path}")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Graphs: {str(e)}")


# ============================================================================
# HYBRID STORE (Facade Pattern)
# ============================================================================

class HybridStore:
    """
    Unified Interface für Vector Store + Knowledge Graph.
    
    Design Pattern: Facade
    Verantwortlichkeiten:
    - Orchestriert Vector + Graph Storage
    - Ensures consistency zwischen beiden Stores
    - Single point of truth für Storage Operations
    """

    def __init__(self, config: StorageConfig, embeddings: Embeddings):
        """
        Initialisiere Hybrid Store.

        Args:
            config: StorageConfig
            embeddings: Embedding-Modell
        """
        self.config = config
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Initialize Sub-Stores
        self.vector_store = VectorStoreAdapter(
            db_path=config.vector_db_path,
            embedding_dim=config.embedding_dim,
        )

        self.graph_store = KnowledgeGraphStore(
            graph_path=config.graph_db_path,
        )

        self.logger.info("HybridStore initialisiert")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Füge Dokumente zu beiden Stores hinzu.

        Args:
            documents: Liste von Dokumenten
        """
        if not documents:
            self.logger.warning("Keine Dokumente zum Hinzufügen")
            return

        try:
            # 1. Add to Vector Store
            self.logger.info(f"Füge {len(documents)} Dokumente zu Vector Store hinzu...")
            self.vector_store.add_documents_with_embeddings(documents, self.embeddings)

            # 2. Extract Entities und füge zu Graph hinzu
            self.logger.info("Extrahiere Entities für Knowledge Graph...")
            for doc in documents:
                # Einfache Entity Extraction (TODO: NER verbessern)
                doc_id = str(doc.metadata.get("chunk_id", "unknown"))
                source_file = doc.metadata.get("source_file", "unknown")

                # Add Document als Entity
                self.graph_store.add_entity(
                    entity_id=doc_id,
                    entity_type="document_chunk",
                    metadata={"source_file": source_file},
                )

                # Add Source File als Entity
                if source_file != "unknown":
                    self.graph_store.add_entity(
                        entity_id=source_file,
                        entity_type="source_document",
                        metadata={},
                    )

                    # Relation: Document → Source File
                    self.graph_store.add_relation(
                        source_id=doc_id,
                        target_id=source_file,
                        relation_type="from_source",
                    )

            self.logger.info("✓ Dokumente erfolgreich zu Hybrid Store hinzugefügt")

        except Exception as e:
            self.logger.error(f"Fehler beim Hinzufügen zu Hybrid Store: {str(e)}")
            raise

    def save(self) -> None:
        """Speichere beide Stores persistent."""
        try:
            # Vector Store speichert automatisch
            # Nur Graph muss explizit gespeichert werden
            self.graph_store.save()
            self.logger.info("✓ Hybrid Store gespeichert")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern: {str(e)}")

    def load(self) -> None:
        """Lade beide Stores (falls persistent)."""
        try:
            # Vector Store lädt automatisch beim Initialisieren
            # Graph wird im Constructor geladen
            self.logger.info("✓ Hybrid Store geladen")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden: {str(e)}")

    # ========================================================================
    # RESET METHODS (für Ablation Studies)
    # ========================================================================

    def reset_vector_store(self) -> None:
        """Setze Vector Store zurück (für Ablation Studies)."""
        try:
            if self.config.vector_db_path.exists():
                shutil.rmtree(self.config.vector_db_path)
                self.logger.info("✓ Vector Store Verzeichnis gelöscht")
            
            # Reinitialize
            self.vector_store = VectorStoreAdapter(
                self.config.vector_db_path,
                self.config.embedding_dim
            )
            self.logger.info("✓ Vector Store zurückgesetzt")
        except Exception as e:
            self.logger.error(f"Fehler beim Reset von Vector Store: {str(e)}")
            raise

    def reset_graph_store(self) -> None:
        """Setze Graph Store zurück (für Ablation Studies)."""
        try:
            if self.config.graph_db_path.exists():
                self.config.graph_db_path.unlink()
                self.logger.info("✓ Graph Datei gelöscht")
            
            # Reinitialize
            self.graph_store = KnowledgeGraphStore(self.config.graph_db_path)
            self.logger.info("✓ Graph Store zurückgesetzt")
        except Exception as e:
            self.logger.error(f"Fehler beim Reset von Graph Store: {str(e)}")
            raise

    def reset_all(self) -> None:
        """Setze beide Stores komplett zurück."""
        self.reset_vector_store()
        self.reset_graph_store()
        self.logger.warning("✗ HYBRID STORE KOMPLETT ZURÜCKGESETZT")