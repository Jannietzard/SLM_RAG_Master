"""
CRITICAL FIX for src/storage.py - Vector Search Bug

Problem: vector_search() Methode ist nicht korrekt implementiert oder eingerückt
Lösung: Korrekte Implementation mit LanceDB Distance/Similarity Conversion
"""

# ============================================================================
# ERSETZE die VectorStoreAdapter Klasse in src/storage.py mit dieser Version:
# ============================================================================

class VectorStoreAdapter:
    """
    LanceDB Vector Store Adapter für Edge-optimierte Suche.
    
    FIX: Korrekte vector_search() Implementation mit Distance→Similarity Conversion
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
            # Wir holen mehr Results für besseres Filtering
            raw_results = self.table.search(query_embedding).limit(top_k * 3).to_list()
            
            if not raw_results:
                self.logger.debug("Vector Search: Keine Results von LanceDB")
                return []

            # Konvertiere Distance → Similarity und filtere
            filtered = []
            for i, result in enumerate(raw_results):
                # LanceDB Column: _distance (Cosine Distance in [0, 2])
                distance = result.get("_distance", 1.0)
                
                # Conversion: Cosine Distance → Similarity [0, 1]
                # distance=0 → perfect match → similarity=1.0
                # distance=2 → opposite → similarity=0.0
                similarity = max(0.0, 1.0 - (distance / 2.0))
                
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
# VERWENDUNG:
# ============================================================================
# 1. Öffne src/storage.py
# 2. Finde die VectorStoreAdapter Klasse (ca. Zeile 40-200)
# 3. Ersetze die GESAMTE Klasse mit dem Code oben
# 4. Stelle sicher, dass die Einrückung korrekt ist (class-Level!)
# 5. Entferne die alte vector_search_FIXED() Funktion (ist redundant)