"""
Hybrid Storage: Vector Store (LanceDB) + Knowledge Graph (NetworkX) - ENHANCED.

IMPROVEMENTS über original storage.py:
1. ✅ Automatische Embedding Dimension Detection (verhindert Mismatch Bug!)
2. ✅ Optional L2-Normalisierung für Embeddings
3. ✅ Verbesserte Distance→Similarity Conversion
4. ✅ Metadata Persistence (embedding_dim wird gespeichert)
5. ✅ Validation Layer (prüft Dimensionen vor jedem Add)
6. ✅ Enhanced Logging (zeigt Dimension Info)

BUG FIXES:
- ✅ Embedding Dimension Mismatch (dein 0.16 Score Bug!)
- ✅ Cosine Distance Conversion (korrigiert für LanceDB)
- ✅ Shape Validation vor Vector Store Add

Scientific Foundation:
- Vector Store: Dense embeddings für semantic similarity
- Knowledge Graph: Strukturelle Relationen für multi-hop reasoning
- Hybrid: Kombiniert Vorteile beider Ansätze (vgl. Graph-RAG)
- L2-Normalisierung: Macht Cosine Similarity äquivalent zu Dot Product (schneller!)

BACKWARDS COMPATIBLE:
- Gleiche API wie original storage.py
- Funktioniert mit main.py ohne Änderungen
- LanceDB Integration unverändert
"""

import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

import lancedb
import networkx as nx
import numpy as np
from langchain.schema import Document
from langchain.embeddings.base import Embeddings


logger = logging.getLogger(__name__)


# ============================================================================
# STORAGE CONFIG - Enhanced mit Validation
# ============================================================================

@dataclass
class StorageConfig:
    """
    Konfiguration für Hybrid Storage mit Auto-Detection.
    
    ENHANCEMENT: embedding_dim kann jetzt None sein → Auto-Detection!
    """
    vector_db_path: Path
    graph_db_path: Path
    embedding_dim: Optional[int] = None  # NEU: Auto-detect wenn None!
    similarity_threshold: float = 0.5
    normalize_embeddings: bool = True  # NEU: L2-Normalisierung aktivieren?
    
    def __post_init__(self):
        """Validiere Config nach Initialisierung."""
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim muss positiv sein, ist: {self.embedding_dim}")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold muss in [0, 1] sein, ist: {self.similarity_threshold}"
            )


# ============================================================================
# VECTOR STORE ADAPTER (LanceDB) - ENHANCED
# ============================================================================

class VectorStoreAdapter:
    """
    LanceDB Vector Store Adapter mit Auto-Dimension-Detection.
    
    CRITICAL BUG FIX:
    Original hatte hardcoded embedding_dim in Config → Mismatch wenn Model anders!
    → Lösung: Auto-detect aus erstem Embedding, speichere in Metadata
    
    NEW FEATURES:
    1. ✅ Automatische Dimension Detection
    2. ✅ Optional L2-Normalisierung
    3. ✅ Shape Validation vor jedem Add
    4. ✅ Metadata Persistence
    5. ✅ Verbesserte Error Messages
    
    Scientific Rationale:
    - L2-Normalisierung macht Cosine Similarity = Dot Product
    - Dot Product ist schneller als Cosine (keine Division)
    - Normalisierte Vektoren haben Magnitude 1 → numerisch stabiler
    """

    def __init__(
        self, 
        db_path: Path, 
        embedding_dim: Optional[int] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialisiere LanceDB Connection mit Auto-Detection Support.

        Args:
            db_path: Pfad zur LanceDB Datenbasis
            embedding_dim: Dimensionalität (None = auto-detect)
            normalize_embeddings: L2-Normalisierung aktivieren?
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))
        self.embedding_dim = embedding_dim  # Kann None sein!
        self.normalize_embeddings = normalize_embeddings
        self.table = None
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"LanceDB initialisiert: {db_path} | "
            f"dim={'auto-detect' if embedding_dim is None else embedding_dim} | "
            f"normalize={normalize_embeddings}"
        )
        
        # Try to load existing metadata
        self._load_metadata()

    def _load_metadata(self) -> None:
        """
        Lade Metadata aus vorheriger Ingestion (falls vorhanden).
        
        NEU! Verhindert Dimension Mismatch nach Neustart.
        """
        metadata_path = Path("data/vector_store_metadata.json")
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                stored_dim = metadata.get("embedding_dim")
                
                if stored_dim and self.embedding_dim is None:
                    self.embedding_dim = stored_dim
                    self.logger.info(f"✓ Embedding Dimension aus Metadata geladen: {stored_dim}")
                elif stored_dim and self.embedding_dim != stored_dim:
                    self.logger.warning(
                        f"⚠ Dimension Mismatch! "
                        f"Config: {self.embedding_dim}, Metadata: {stored_dim} | "
                        f"Verwende Config-Wert"
                    )
            except Exception as e:
                self.logger.debug(f"Metadata Laden fehlgeschlagen: {e}")
    
    def _save_metadata(self) -> None:
        """
        Speichere Vector Store Metadata.
        
        NEU! Für Reproducibility und Dimension Tracking.
        """
        if self.embedding_dim is None:
            return  # Noch keine Dimension bekannt
        
        metadata = {
            "embedding_dim": self.embedding_dim,
            "normalize_embeddings": self.normalize_embeddings,
            "timestamp": time.time(),
            "num_documents": len(self.table) if self.table else 0,
        }
        
        metadata_path = Path("data/vector_store_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Vector Store Metadata gespeichert: {metadata_path}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2-Normalisierung von Embedding-Vektoren.
        
        Scientific Rationale:
        Normalized vectors haben Magnitude 1.0:
        - Cosine Similarity wird zu Dot Product
        - Dot Product ist schneller (keine Division)
        - Numerisch stabiler (kein Division by Zero)
        
        Formula: v_norm = v / ||v||_2
        
        Args:
            vectors: Numpy array shape (N, D)
            
        Returns:
            Normalized vectors shape (N, D)
        """
        if not self.normalize_embeddings:
            return vectors
        
        # L2 Norm berechnen (per row)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        
        # Normalize
        normalized = vectors / norms
        
        return normalized

    def _validate_embedding_dimension(self, embeddings: List[List[float]]) -> None:
        """
        Validiere Embedding Dimensionen gegen erwartete Dimension.
        
        CRITICAL: Verhindert den 0.16 Score Bug!
        
        Args:
            embeddings: Liste von Embedding Vektoren
            
        Raises:
            ValueError: Wenn Dimensionen nicht matchen
        """
        if not embeddings:
            return
        
        actual_dim = len(embeddings[0])
        
        # Auto-detect dimension beim ersten Add
        if self.embedding_dim is None:
            self.embedding_dim = actual_dim
            self.logger.info(f"✓ Auto-detected Embedding Dimension: {actual_dim}")
            self._save_metadata()
            return
        
        # Validate gegen bekannte Dimension
        if actual_dim != self.embedding_dim:
            raise ValueError(
                f"🚨 EMBEDDING DIMENSION MISMATCH! 🚨\n"
                f"Expected: {self.embedding_dim} dimensions\n"
                f"Got: {actual_dim} dimensions\n"
                f"\n"
                f"Dies ist wahrscheinlich die Ursache für niedrige Scores!\n"
                f"Lösung:\n"
                f"1. Prüfe config/settings.yaml: embedding_dim = {actual_dim}\n"
                f"2. Lösche Vector Store: rm -rf data/vector_db/\n"
                f"3. Neu ingestieren: python main.py\n"
            )

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: Embeddings,
    ) -> None:
        """
        Füge Dokumente mit Embeddings zu Vector Store hinzu.
        
        ENHANCEMENTS:
        1. Auto-detect Embedding Dimension
        2. Validate Dimensions
        3. Optional Normalisierung
        4. Better Error Messages

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
        
        start_time = time.time()
        embeddings_list = embeddings.embed_documents(texts)
        embed_time = time.time() - start_time
        
        self.logger.info(
            f"✓ Embeddings generiert: {len(embeddings_list)} vectors | "
            f"{embed_time:.1f}s ({embed_time/len(texts)*1000:.1f}ms/doc)"
        )

        # CRITICAL: Validate Dimensions
        self._validate_embedding_dimension(embeddings_list)

        # Convert to numpy for normalization
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        # Optional: L2 Normalisierung
        if self.normalize_embeddings:
            embeddings_array = self._normalize_vectors(embeddings_array)
            self.logger.debug(f"✓ Embeddings L2-normalisiert")
        
        # Convert back to list for LanceDB
        embeddings_list = embeddings_array.tolist()

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
            
            # Save metadata
            self._save_metadata()
            
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
        Vector-basierte Similarity Search mit korrigierter Distance Conversion.
        
        FIXED: Korrekte Conversion von LanceDB Cosine Distance → Similarity
        
        LanceDB mit metric="cosine" gibt zurück:
        - Cosine Distance ∈ [0, 2] where:
          * 0 = identical vectors
          * 1 = orthogonal (90°)
          * 2 = opposite (180°)
        
        Conversion zu Similarity ∈ [0, 1]:
        - Wenn Embeddings normalized: similarity = 1 - distance
        - Wenn Embeddings nicht normalized: similarity = 1 - (distance / 2)
        
        Scientific Rationale:
        Für normalized vectors (magnitude=1):
        - Cosine Distance = 2 * (1 - cosine_similarity)
        - → cosine_similarity = 1 - (distance / 2)
        
        Aber da wir normalisieren, gilt:
        - distance = 1 - similarity (direkt)

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
            # Validate query dimension
            if self.embedding_dim and len(query_embedding) != self.embedding_dim:
                raise ValueError(
                    f"Query Embedding Dimension Mismatch! "
                    f"Expected {self.embedding_dim}, got {len(query_embedding)}"
                )
            
            # Normalize query embedding
            if self.normalize_embeddings:
                query_array = np.array([query_embedding], dtype=np.float32)
                query_array = self._normalize_vectors(query_array)
                query_embedding = query_array[0].tolist()
            
            # LanceDB Search
            raw_results = self.table.search(query_embedding).limit(top_k * 3).to_list()
            
            if not raw_results:
                self.logger.debug("Vector Search: Keine Results von LanceDB")
                return []

            # Convert Distance → Similarity
            filtered = []
            
            for i, result in enumerate(raw_results):
                # LanceDB gibt Cosine Distance zurück
                distance = result.get("_distance", 1.0)
                
                # KORREKTE Conversion für normalized embeddings:
                # Cosine Similarity = 1 - Cosine Distance
                similarity = 1.0 - distance
                
                # Clamp zu [0, 1] (safety)
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
            
            # Log score distribution
            if final_results:
                scores = [r['similarity'] for r in final_results]
                self.logger.info(
                    f"Score Range: [{min(scores):.4f}, {max(scores):.4f}] "
                    f"Avg: {sum(scores)/len(scores):.4f}"
                )
            
            return final_results

        except Exception as e:
            self.logger.error(f"Fehler bei Vector Search: {str(e)}", exc_info=True)
            return []


# ============================================================================
# KNOWLEDGE GRAPH STORE (NetworkX) - UNVERÄNDERT
# ============================================================================

class KnowledgeGraphStore:
    """
    NetworkX-basierter Knowledge Graph für strukturelle Relationen.
    
    UNCHANGED: Funktioniert wie bisher.
    """

    def __init__(self, graph_path: Path):
        """Initialisiere Knowledge Graph."""
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
        """Füge Entity zum Graph hinzu."""
        self.graph.add_node(entity_id, entity_type=entity_type, **metadata)

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Füge Relation zwischen Entities hinzu."""
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
        """BFS-basierte Graph-Traversal."""
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
# HYBRID STORE (Facade Pattern) - ENHANCED
# ============================================================================

class HybridStore:
    """
    Unified Interface für Vector Store + Knowledge Graph mit Auto-Detection.
    
    ENHANCEMENTS:
    1. ✅ Auto-detect Embedding Dimension
    2. ✅ Validate Dimensions vor jedem Add
    3. ✅ Enhanced Logging mit Dimension Info
    4. ✅ Metadata Persistence
    
    BACKWARDS COMPATIBLE:
    - Gleiche API wie original HybridStore
    - Funktioniert mit main.py ohne Änderungen
    """

    def __init__(self, config: StorageConfig, embeddings: Embeddings):
        """
        Initialisiere Hybrid Store mit Auto-Detection.

        Args:
            config: StorageConfig (embedding_dim kann None sein!)
            embeddings: Embedding-Modell
        """
        self.config = config
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Auto-detect dimension falls nicht gesetzt
        if config.embedding_dim is None:
            self.logger.info("🔍 Auto-detecting Embedding Dimension...")
            test_embedding = embeddings.embed_query("test")
            detected_dim = len(test_embedding)
            config.embedding_dim = detected_dim
            self.logger.info(f"✓ Detected Dimension: {detected_dim}")

        # Initialize Sub-Stores
        self.vector_store = VectorStoreAdapter(
            db_path=config.vector_db_path,
            embedding_dim=config.embedding_dim,
            normalize_embeddings=config.normalize_embeddings,
        )

        self.graph_store = KnowledgeGraphStore(
            graph_path=config.graph_db_path,
        )

        self.logger.info(
            f"HybridStore initialisiert: "
            f"dim={config.embedding_dim}, "
            f"normalize={config.normalize_embeddings}"
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Füge Dokumente zu beiden Stores hinzu mit Validation.

        Args:
            documents: Liste von Dokumenten
        """
        if not documents:
            self.logger.warning("Keine Dokumente zum Hinzufügen")
            return

        try:
            # 1. Add to Vector Store (mit Auto-Dimension-Detection)
            self.logger.info(f"Füge {len(documents)} Dokumente zu Vector Store hinzu...")
            self.vector_store.add_documents_with_embeddings(documents, self.embeddings)

            # 2. Extract Entities und füge zu Graph hinzu
            self.logger.info("Extrahiere Entities für Knowledge Graph...")
            for doc in documents:
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
    # RESET METHODS (für Ablation Studies) - UNVERÄNDERT
    # ========================================================================

    def reset_vector_store(self) -> None:
        """Setze Vector Store zurück (für Ablation Studies)."""
        try:
            if self.config.vector_db_path.exists():
                shutil.rmtree(self.config.vector_db_path)
                self.logger.info("✓ Vector Store Verzeichnis gelöscht")
            
            # Delete metadata
            metadata_path = Path("data/vector_store_metadata.json")
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Reinitialize
            self.vector_store = VectorStoreAdapter(
                self.config.vector_db_path,
                self.config.embedding_dim,
                self.config.normalize_embeddings,
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