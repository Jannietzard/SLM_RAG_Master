"""
Hybrid Storage Layer: Vector DB (LanceDB) + Knowledge Graph (NetworkX).

Scientific Foundation:
- LanceDB: Embedded Vector Store optimiert für Edge Devices (OLAP, Columnar)
- NetworkX Graph: Ermöglicht Entity-Relation-Reasoning und Multi-Hop Retrieval
- Kombination reduziert Information Bottleneck in SLMs durch strukturierte
  semantische Repräsentation (vgl. Graph-RAG, Falko et al., 2024)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass, asdict

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


class VectorStoreAdapter:
    """
    LanceDB Vector Store Adapter für Edge-optimierte Suche.
    
    Design Pattern: Adapter Pattern (vereinheitlichte Interface für LanceDB)
    
    Scientific Rationale:
    LanceDB ist columnares OLAP-System → extremeffizient für
    Vektorskalarproduktes auf Edge-Hardware. Lineare Komplexität
    für k-NN bei Millionen Vektoren praktisch unmöglich, daher
    IVF-FLat-Index für logarithmische Komplexität (vgl. Jegou et al.,
    Approximate Nearest Neighbor Search).
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
        self.logger = logger
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
        # Generiere Embeddings
        texts = [doc.page_content for doc in documents]
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
                "document_id": doc.metadata.get("chunk_id", "unknown"),
                "text": doc.page_content,
                "vector": emb,
                "metadata": json.dumps(doc.metadata),
                "source_file": doc.metadata.get("source_file", "unknown"),
            })

        # Speichere in LanceDB
        try:
            if self.table is None:
                self.table = self.db.create_table("documents", data=data, mode="overwrite")
            else:
                self.table.add(data)
            
            self.logger.info(f"Erfolgreich {len(data)} Dokumente in Vector Store eingefügt")
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
        Vektorbasierte Similarity Search mit korrekter Distance/Similarity Konvertierung.

        Scientific Foundation:
        LanceDB nutzt Cosine Distance (0-2 range):
        - Distance = 1 - Similarity
        - Für Threshold: similarity = (2 - distance) / 2 normalisiert
        
        Args:
            query_embedding: Query Vector (384-dim für nomic-embed-text)
            top_k: Anzahl der Top Results
            threshold: Minimal-Ähnlichkeit (0.0-1.0 Similarity, nicht Distance!)

        Returns:
            Liste von Results mit Text, Score, Metadaten
        """
        if self.table is None:
            self.logger.warning("Vector Store ist leer - keine Dokumente")
            return []

        try:
            # LanceDB search: gibt Cosine Distance zurück
            # Limit auf 2x top_k für besseres Filtering
            results = self.table.search(query_embedding).limit(top_k * 2).to_list()
            
            if not results:
                self.logger.debug("Vector Search: Keine Raw Results von LanceDB")
                return []

            # Filtere nach Threshold und konvertiere zu Similarity
            filtered = []
            for i, result in enumerate(results):
                # LanceDB gibt _distance zurück (Cosine Distance in range [0, 2])
                distance = result.get("_distance", 1.0)
                
                # Konvertiere Distance zu Similarity [0, 1]
                # Cosine: similarity = 1 - distance/2
                similarity = max(0.0, 1.0 - (distance / 2.0))
                
                # Debug Logging für erste 3 Results
                if i < 3:
                    self.logger.debug(
                        f"  VecSearch Result {i}: distance={distance:.4f} → similarity={similarity:.4f}"
                    )
                
                # Threshold Check auf Similarity!
                if similarity >= threshold:
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

            # Gebe nur top_k beste Results zurück
            filtered.sort(key=lambda x: x["similarity"], reverse=True)
            final_results = filtered[:top_k]
            
            self.logger.debug(
                f"Vector Search: {len(results)} raw → {len(filtered)} filtered "
                f"→ {len(final_results)} returned (threshold={threshold:.2f})"
            )
            
            return final_results

        except Exception as e:
            self.logger.error(f"Fehler bei Vector Search: {str(e)}", exc_info=True)
            return []

"""
FIX für Vector Search Bug in LanceDB.

Problem in Original storage.py:
- vector_search() gibt leere List zurück obwohl Daten vorhanden sind
- Grund: LanceDB API wurde falsch genutzt
- search().limit().to_list() gibt Distance, nicht Similarity zurück
- Conversion (1 - distance) ist falsch für Cosine bei LanceDB

Lösung: Korrekte LanceDB API Nutzung + Debug Output
"""

def vector_search_FIXED(
    self,
    query_embedding: list,
    top_k: int = 5,
    threshold: float = 0.5,
) -> list:
    """
    Korrekte Vector-basierte Similarity Search mit LanceDB.

    WICHTIG: LanceDB gibt Cosine Distance zurück (nicht Similarity!)
    - Distance = 1 - Cosine Similarity
    - Bei threshold=0.5, suchen wir nach distance < 0.5

    Args:
        query_embedding: Query Vector (384-dim für nomic-embed-text)
        top_k: Anzahl Top Results
        threshold: Cosine Distance Threshold (NOT Similarity!)

    Returns:
        Liste von Results mit Text, Score, Metadaten
    """
    if self.table is None:
        self.logger.warning("Vector Store ist leer - keine Dokumente gefunden")
        return []

    try:
        # LanceDB distance metric: "cosine" gibt Distance zurück (0-2)
        # Für unseren Use-Case:
        # - Cosine Distance = 1 - Cosine Similarity
        # - Threshold 0.5 → nur Vektoren mit >0.5 Similarity
        
        results = self.table.search(query_embedding).limit(top_k * 2).to_list()
        
        self.logger.debug(f"LanceDB Raw Results: {len(results)} rows returned")
        
        if not results:
            self.logger.warning(f"Vector Search: Keine Results gefunden für Query")
            return []

        # Filter nach Threshold (LanceDB gibt distance zurück!)
        filtered = []
        for i, result in enumerate(results):
            # LanceDB Column Names prüfen (können variieren!)
            distance = result.get("_distance", result.get("distance", 1.0))
            
            # Konvertiere Distance → Similarity
            # Cosine Distance in [0, 2], wir wollen Similarity in [0, 1]
            similarity = 1 - (distance / 2)  # Normalisierung für Cosine
            
            # Debug: Erste 3 Results loggen
            if i < 3:
                self.logger.debug(
                    f"  Result {i}: distance={distance:.4f}, similarity={similarity:.4f}"
                )
            
            # Threshold Check (similarity statt distance!)
            if similarity >= threshold:
                filtered.append({
                    "text": result.get("text", ""),
                    "similarity": similarity,
                    "document_id": result.get("document_id", str(i)),
                    "metadata": {
                        "source_file": result.get("source_file", "unknown"),
                        "raw_distance": distance,
                    },
                })
        
        self.logger.debug(
            f"Vector Search: {len(results)} raw → {len(filtered)} filtered "
            f"(threshold={threshold:.2f}, top_k={top_k})"
        )
        
        return filtered[:top_k]  # Return nur top_k

    except Exception as e:
        self.logger.error(f"Fehler bei Vector Search: {str(e)}", exc_info=True)
        return []


# ==============================================================================
# VERWENDUNG IN storage.py:
# ==============================================================================
# Ersetze die vector_search() Methode in der VectorStoreAdapter Klasse
# mit dem Code oben.
#
# Hauptänderungen:
# 1. Korrekte Distance/Similarity Konvertierung für Cosine
# 2. Debug Logging für erste 3 Results
# 3. Threshold Check auf Similarity (nicht Distance)
# 4. Bessere Error Handling mit exc_info=True


class KnowledgeGraphStore:
    """
    NetworkX-basierter Knowledge Graph für Struktur-basiertes Reasoning.
    
    Design Pattern: Strategy Pattern (austauschbare Graph-Implementierungen)
    
    Scientific Rationale:
    Graphen ermöglichen Multi-Hop Reasoning über Entity-Relations hinweg.
    Dies reduziert "Lost-in-the-Middle" Problem für SLMs durch explizite
    strukturelle Relationen (vgl. Graph-RAG, Yu et al., 2024; 
    Entity Relation Extraction mit NER + Dependency Parsing).
    """

    def __init__(self, graph_path: Path, max_hops: int = 2):
        """
        Initialisiere Knowledge Graph.

        Args:
            graph_path: Pfad zur Graph-Speicherung
            max_hops: Maximale Distanz für Graph-Traversal
        """
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        self.max_hops = max_hops
        self.logger = logger
        self.logger.info(f"KnowledgeGraph initialisiert: {graph_path}")

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: Dict[str, Any],
    ) -> None:
        """
        Füge Entity zum Graph hinzu.

        Args:
            entity_id: Eindeutige Entity ID
            entity_type: Typ der Entity (z.B. "concept", "person", "document")
            attributes: Zusätzliche Attribute
        """
        self.graph.add_node(entity_id, type=entity_type, **attributes)
        self.logger.debug(f"Entity hinzugefügt: {entity_id}")

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """
        Füge Relation zwischen Entities hinzu.

        Args:
            source_id: Quell-Entity
            target_id: Ziel-Entity
            relation_type: Typ der Relation
            weight: Gewichtung (für späteres Reranking)
        """
        # Stelle sicher, dass beide Nodes existieren
        if source_id not in self.graph:
            self.add_entity(source_id, "unknown", {})
        if target_id not in self.graph:
            self.add_entity(target_id, "unknown", {})

        self.graph.add_edge(
            source_id, target_id, relation_type=relation_type, weight=weight
        )
        self.logger.debug(f"Relation hinzugefügt: {source_id} --[{relation_type}]--> {target_id}")

    def graph_traversal(
        self,
        start_entity: str,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Traversiere Graph ausgehend von Start-Entity mit max_hops Limit.

        Args:
            start_entity: Start-Knoten
            relation_types: Zu folgende Relation-Typen (None = alle)

        Returns:
            Dict von Entity → Distanz (für Ranking)

        Scientific Rationale:
        BFS mit Hop-Limit ermöglicht lokalisiertes Reasoning und
        verhindert Information Explosion in großen Graphen.
        """
        if start_entity not in self.graph:
            self.logger.warning(f"Entity nicht im Graph: {start_entity}")
            return {}

        visited = {start_entity: 0}
        queue = [(start_entity, 0)]

        while queue:
            current, hops = queue.pop(0)

            if hops >= self.max_hops:
                continue

            # Durchsuche Nachbarn
            for neighbor in self.graph.successors(current):
                edge_data = self.graph[current][neighbor]
                rel_type = edge_data.get("relation_type")

                # Filter nach Relation-Typ
                if relation_types and rel_type not in relation_types:
                    continue

                if neighbor not in visited:
                    visited[neighbor] = hops + 1
                    queue.append((neighbor, hops + 1))

        return visited

    def save_graph(self) -> None:
        """Speichere Graph auf Disk (GraphML Format für Portabilität)."""
        try:
            nx.write_graphml(self.graph, str(self.graph_path))
            self.logger.info(f"Graph gespeichert: {self.graph_path}")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Graphs: {str(e)}")

    def load_graph(self) -> None:
        """Lade Graph von Disk."""
        try:
            if self.graph_path.exists():
                self.graph = nx.read_graphml(str(self.graph_path))
                self.logger.info(f"Graph geladen: {self.graph_path}")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Graphs: {str(e)}")


class HybridStore:
    """
    Unified Interface für Hybrid Storage (Vectors + Graph).
    
    Design Pattern: Facade Pattern (vereinheitlichte API)
    Dependency Injection: VectorStore und Graph werden injiziert
    
    Scientific Rationale:
    Kombination von Vektoren und Graphen nutzt Stärken beider:
    - Vektoren: Schnelle semantische Ähnlichkeit (Dense Retrieval)
    - Graphen: Strukturelle Relationen und Multi-Hop Reasoning
    Ensemble-Approach reduziert Blindheit von reinen Vektor-Systemen.
    """

    def __init__(
        self,
        config: StorageConfig,
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Initialisiere Hybrid Store.

        Args:
            config: StorageConfig mit Pfaden und Parametern
            embeddings: Embedding-Modell (optional)
        """
        self.config = config
        self.embeddings = embeddings
        self.vector_store = VectorStoreAdapter(config.vector_db_path, config.embedding_dim)
        self.graph_store = KnowledgeGraphStore(config.graph_db_path)
        self.logger = logger

    def add_documents(
        self,
        documents: List[Document],
    ) -> None:
        """
        Füge Dokumente zu beiden Stores hinzu.

        Args:
            documents: Liste gechunkter Dokumente
        """
        if self.embeddings is None:
            raise ValueError("Embeddings-Modell erforderlich für add_documents")

        # Vector Store
        self.vector_store.add_documents_with_embeddings(documents, self.embeddings)

        # Knowledge Graph (einfache Implementierung: Dokumente als Nodes)
        for doc in documents:
            doc_id = doc.metadata.get("chunk_id", "unknown")
            self.graph_store.add_entity(
                entity_id=str(doc_id),
                entity_type="document_chunk",
                attributes={
                    "source": doc.metadata.get("source_file"),
                    "page": doc.metadata.get("page"),
                },
            )
            
            # Verbinde aufeinanderfolgende Chunks
            prev_chunk_id = int(doc_id) - 1
            if prev_chunk_id >= 0:
                self.graph_store.add_relation(
                    source_id=str(prev_chunk_id),
                    target_id=str(doc_id),
                    relation_type="follows",
                    weight=1.0,
                )

        self.logger.info(f"Hybrid Store aktualisiert mit {len(documents)} Dokumenten")

    def save(self) -> None:
        """Persistiere alle Stores."""
        self.graph_store.save_graph()
        self.logger.info("Hybrid Store gespeichert")

    def load(self) -> None:
        """Lade alle Stores von Disk."""
        self.graph_store.load_graph()
        self.logger.info("Hybrid Store geladen")

    def reset_vector_store(self) -> None:
        """
        Setze Vector Store zurück (für Ablation Studies).
        
        Wissenschaftliche Begründung:
        Für saubere Experimentaldesign müssen Ablation Studies
        mit identischen Vector Stores starten, um Konfounding
        Variablen auszuschließen.
        """
        try:
            # Lösche Vector DB auf Disk
            import shutil
            if self.config.vector_db_path.exists():
                shutil.rmtree(self.config.vector_db_path)
            
            # Reinitialize Adapter
            self.vector_store = VectorStoreAdapter(
                self.config.vector_db_path,
                self.config.embedding_dim
            )
            self.logger.info("✓ Vector Store zurückgesetzt")
        except Exception as e:
            self.logger.error(f"Fehler beim Reset von Vector Store: {str(e)}")
            raise

    def reset_graph_store(self) -> None:
        """
        Setze Graph Store zurück (für Ablation Studies).
        """
        try:
            if self.config.graph_db_path.exists():
                self.config.graph_db_path.unlink()
            
            self.graph_store = KnowledgeGraphStore(self.config.graph_db_path)
            self.logger.info(" Graph Store zurückgesetzt")
        except Exception as e:
            self.logger.error(f"Fehler beim Reset von Graph Store: {str(e)}")
            raise

    def reset_all(self) -> None:
        """
        Setze beide Stores komplett zurück.
        
        Warnung: Dies ist destruktiv! Nur für neue Ablation-Durchläufe.
        """
        self.reset_vector_store()
        self.reset_graph_store()
        self.logger.warning(" HYBRID STORE KOMPLETT ZURÜCKGESETZT (Destruktive Operation)")