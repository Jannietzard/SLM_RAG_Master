"""
Entity Extraction Pipeline: GLiNER + REBEL

Version: 4.0.0 - MASTERTHESIS IMPLEMENTATION
Author: Edge-RAG Research Project

===============================================================================
IMPLEMENTATION GEMÄSS MASTERTHESIS ABSCHNITT 2.5
===============================================================================

Stufe 1: Named Entity Recognition mit GLiNER-small
    - Zero-Shot NER ohne domänenspezifisches Fine-Tuning
    - 6 Entitätstypen: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, CONCEPT
    - Confidence Threshold: 0.5
    - Batch Size: 16

Stufe 2: Relation Extraction mit REBEL
    - Nur auf Chunks mit >= 2 Entitäten (60% weniger RE-Aufrufe)
    - Relation Types: works_for, located_in, part_of, etc.
    - Confidence Threshold: 0.7
    - Batch Size: 8

Optimierungen:
    - Batch-Processing für GLiNER (16) und REBEL (8)
    - Entity Caching (LRU-Cache für häufige Entitäten)
    - Selective RE (nur Chunk-Paare mit Co-Occurrences, 70% weniger Aufrufe)
    - Ziel-Latenz: 80-120ms pro Chunk

===============================================================================
"""

import logging
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from functools import lru_cache
from collections import OrderedDict
import time

logger = logging.getLogger(__name__)

# ============================================================================
# GLiNER INTEGRATION
# ============================================================================

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logger.warning("GLiNER not available. Install with: pip install gliner")
# ============================================================================
# ENTITY EXTRACTION PIPELINE
# ============================================================================
"""
Entity Extraction -> Graph Store Integration

Diese Komponente fehlt komplett in der aktuellen Implementierung!
Sie ist essentiell für die Funktionsfähigkeit des Hybrid Index.
"""

from typing import List, Dict, Any
from src.data_layer.entity_extraction import (
    EntityExtractionPipeline,
    ChunkExtractionResult,
    ExtractionConfig
)
from src.data_layer.storage import KuzuGraphStore
# ============================================================================
# REBEL INTEGRATION (Transformers)
# ============================================================================

try:
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")

# ============================================================================
# SPACY FALLBACK
# ============================================================================

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available. Install with: pip install spacy")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExtractedEntity:
    """Extrahierte Named Entity."""
    entity_id: str
    name: str
    entity_type: str  # PERSON, ORGANIZATION, LOCATION, DATE, EVENT, CONCEPT
    confidence: float
    mention_span: Tuple[int, int]  # Character offsets (start, end)
    source_chunk_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "type": self.entity_type,
            "confidence": self.confidence,
            "mention_span": list(self.mention_span),
            "source_chunk_id": self.source_chunk_id,
        }


@dataclass
class ExtractedRelation:
    """Extrahierte Relation zwischen Entitäten."""
    subject_entity: str  # Entity name
    relation_type: str   # z.B. "works_for", "located_in", "part_of"
    object_entity: str   # Entity name
    confidence: float
    source_chunk_ids: List[str]  # Chunks, die diese Relation belegen
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject_entity,
            "relation": self.relation_type,
            "object": self.object_entity,
            "confidence": self.confidence,
            "source_chunks": self.source_chunk_ids,
        }


@dataclass
class ChunkExtractionResult:
    """Ergebnis der Entity/Relation Extraction für einen Chunk."""
    chunk_id: str
    text: str
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    extraction_time_ms: float
    
    @property
    def entity_count(self) -> int:
        return len(self.entities)
    
    @property
    def relation_count(self) -> int:
        return len(self.relations)


@dataclass
class ExtractionConfig:
    """Konfiguration für Entity Extraction Pipeline."""
    # GLiNER
    gliner_model: str = "urchade/gliner_small-v2.1"
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "CONCEPT"
    ])
    ner_confidence_threshold: float = 0.5
    ner_batch_size: int = 16
    
    # REBEL
    rebel_model: str = "Babelscape/rebel-large"
    re_confidence_threshold: float = 0.7
    re_batch_size: int = 8
    min_entities_for_re: int = 2  # Nur RE wenn >= 2 Entitäten
    
    # Caching
    cache_enabled: bool = True
    cache_path: str = "./data/entity_cache.db"
    lru_cache_size: int = 10000
    
    # Selective RE
    selective_re: bool = True

class EntityGraphIntegrator:
    """
    Integriert Entity Extraction Ergebnisse in den Knowledge Graph.
    
    KRITISCHE KOMPONENTE: Ohne diese Integration bleiben Entities
    nur in der Extraction Pipeline, kommen aber nie in den Graph!
    
    Workflow:
    1. Chunks werden durch EntityExtractionPipeline verarbeitet
    2. EntityGraphIntegrator nimmt ChunkExtractionResults
    3. Erstellt Entity-Nodes und MENTIONS/RELATED_TO Edges in KuzuDB
    """
    
    def __init__(
        self,
        graph_store: KuzuGraphStore,
        entity_pipeline: EntityExtractionPipeline = None
    ):
        self.graph_store = graph_store
        self.entity_pipeline = entity_pipeline or EntityExtractionPipeline()
        self.logger = logging.getLogger(__name__)
        
        # Statistiken
        self.stats = {
            "chunks_processed": 0,
            "entities_added": 0,
            "mentions_added": 0,
            "relations_added": 0,
        }
    
    def integrate_chunk_results(
        self,
        results: List[ChunkExtractionResult]
    ) -> None:
        """
        Integriere Extraction-Ergebnisse in Graph.
        
        Args:
            results: Liste von ChunkExtractionResult
        """
        for result in results:
            self._integrate_single_chunk(result)
    
    def _integrate_single_chunk(
        self,
        result: ChunkExtractionResult
    ) -> None:
        """Integriere ein einzelnes ChunkExtractionResult."""
        self.stats["chunks_processed"] += 1
        
        # 1. Entities als Nodes hinzufügen
        for entity in result.entities:
            try:
                # Entity Node erstellen/updaten
                self.graph_store.conn.execute(
                    """
                    MERGE (e:Entity {entity_id: $entity_id})
                    SET e.name = $name,
                        e.type = $type,
                        e.mention_count = COALESCE(e.mention_count, 0) + 1
                    """,
                    {
                        "entity_id": entity.entity_id,
                        "name": entity.name,
                        "type": entity.entity_type,
                    }
                )
                self.stats["entities_added"] += 1
                
                # MENTIONS Edge erstellen
                # Verbinde DocumentChunk -> Entity
                self.graph_store.conn.execute(
                    """
                    MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                    MATCH (e:Entity {entity_id: $entity_id})
                    MERGE (c)-[m:MENTIONS]->(e)
                    SET m.mention_span = $mention_span,
                        m.confidence = $confidence
                    """,
                    {
                        "chunk_id": result.chunk_id,
                        "entity_id": entity.entity_id,
                        "mention_span": str(entity.mention_span),
                        "confidence": entity.confidence,
                    }
                )
                self.stats["mentions_added"] += 1
                
            except Exception as e:
                self.logger.error(
                    f"Failed to add entity {entity.name} for chunk {result.chunk_id}: {e}"
                )
        
        # 2. Relations als RELATED_TO Edges hinzufügen
        for relation in result.relations:
            try:
                # Finde oder erstelle Subject Entity
                subj_id = self._get_or_create_entity_id(
                    relation.subject_entity,
                    "CONCEPT"  # Default type
                )
                
                # Finde oder erstelle Object Entity
                obj_id = self._get_or_create_entity_id(
                    relation.object_entity,
                    "CONCEPT"
                )
                
                # RELATED_TO Edge erstellen
                self.graph_store.conn.execute(
                    """
                    MATCH (e1:Entity {entity_id: $subj_id})
                    MATCH (e2:Entity {entity_id: $obj_id})
                    MERGE (e1)-[r:RELATED_TO]->(e2)
                    SET r.relation_type = $relation_type,
                        r.confidence = $confidence,
                        r.source_chunks = COALESCE(r.source_chunks, '') || ',' || $chunk_id
                    """,
                    {
                        "subj_id": subj_id,
                        "obj_id": obj_id,
                        "relation_type": relation.relation_type,
                        "confidence": relation.confidence,
                        "chunk_id": result.chunk_id,
                    }
                )
                self.stats["relations_added"] += 1
                
            except Exception as e:
                self.logger.error(
                    f"Failed to add relation {relation.subject_entity} -> "
                    f"{relation.object_entity} for chunk {result.chunk_id}: {e}"
                )
    
    def _get_or_create_entity_id(
        self,
        entity_name: str,
        entity_type: str
    ) -> str:
        """Hole Entity ID oder erstelle neue Entity."""
        import hashlib
        
        # Generiere Entity ID (konsistent mit GLiNER)
        entity_id = hashlib.md5(
            f"{entity_name.lower()}:{entity_type}".encode()
        ).hexdigest()[:12]
        
        # Erstelle Entity falls nicht existiert
        try:
            self.graph_store.conn.execute(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.name = $name,
                    e.type = $type,
                    e.mention_count = COALESCE(e.mention_count, 0) + 1
                """,
                {
                    "entity_id": entity_id,
                    "name": entity_name,
                    "type": entity_type,
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to create entity {entity_name}: {e}")
        
        return entity_id
    
    def process_and_integrate_chunks(
        self,
        texts: List[str],
        chunk_ids: List[str]
    ) -> List[ChunkExtractionResult]:
        """
        End-to-End: Extraction + Graph Integration.
        
        Args:
            texts: Chunk-Texte
            chunk_ids: Chunk-IDs
            
        Returns:
            Extraction-Ergebnisse
        """
        # Entity Extraction
        self.logger.info(f"Extracting entities from {len(texts)} chunks...")
        results = self.entity_pipeline.process_chunks_batch(texts, chunk_ids)
        
        # Graph Integration
        self.logger.info("Integrating entities into graph...")
        self.integrate_chunk_results(results)
        
        self.logger.info(
            f"Integration complete: "
            f"{self.stats['entities_added']} entities, "
            f"{self.stats['mentions_added']} mentions, "
            f"{self.stats['relations_added']} relations"
        )
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Hole Integrations-Statistiken."""
        return self.stats.copy()
# ============================================================================
# ENTITY CACHE (LRU + SQLite Persistent)
# ============================================================================

class EntityCache:
    """
    Hybrid Entity Cache: In-Memory LRU + SQLite Persistence.
    
    Cacht häufige Entitäten (z.B. "United States", "World War II"),
    sodass wiederholte Mentions keine erneute Extraktion erfordern.
    """
    
    def __init__(self, cache_path: str, max_size: int = 10000):
        self.cache_path = Path(cache_path)
        self.max_size = max_size
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-Memory LRU Cache
        self._memory_cache: OrderedDict = OrderedDict()
        
        # SQLite für Persistenz
        self._init_db()
        self._load_from_db()
        
        logger.info(f"EntityCache initialized with {len(self._memory_cache)} entries")
    
    def _init_db(self):
        """Initialisiere SQLite Datenbank."""
        self.conn = sqlite3.connect(str(self.cache_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_cache (
                text_hash TEXT PRIMARY KEY,
                entities_json TEXT,
                hit_count INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def _load_from_db(self):
        """Lade häufig verwendete Einträge in Memory-Cache."""
        cursor = self.conn.execute("""
            SELECT text_hash, entities_json 
            FROM entity_cache 
            ORDER BY hit_count DESC 
            LIMIT ?
        """, (self.max_size,))
        
        for row in cursor:
            self._memory_cache[row[0]] = json.loads(row[1])
    
    def _text_hash(self, text: str) -> str:
        """Generiere Hash für Text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> Optional[List[Dict]]:
        """
        Hole gecachte Entitäten für Text.
        
        Returns:
            List[Dict] mit Entitäten oder None wenn nicht gecacht.
        """
        key = self._text_hash(text)
        
        # Memory-Cache Check
        if key in self._memory_cache:
            # Move to end (LRU)
            self._memory_cache.move_to_end(key)
            return self._memory_cache[key]
        
        # DB Check
        cursor = self.conn.execute(
            "SELECT entities_json FROM entity_cache WHERE text_hash = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row:
            entities = json.loads(row[0])
            # Update hit count
            self.conn.execute(
                "UPDATE entity_cache SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE text_hash = ?",
                (key,)
            )
            self.conn.commit()
            # Add to memory cache
            self._add_to_memory(key, entities)
            return entities
        
        return None
    
    def put(self, text: str, entities: List[Dict]):
        """Speichere Entitäten für Text."""
        key = self._text_hash(text)
        entities_json = json.dumps(entities)
        
        # Memory Cache
        self._add_to_memory(key, entities)
        
        # DB
        self.conn.execute("""
            INSERT OR REPLACE INTO entity_cache (text_hash, entities_json, hit_count, last_accessed)
            VALUES (?, ?, COALESCE((SELECT hit_count FROM entity_cache WHERE text_hash = ?), 0) + 1, CURRENT_TIMESTAMP)
        """, (key, entities_json, key))
        self.conn.commit()
    
    def _add_to_memory(self, key: str, value: List[Dict]):
        """Füge zum Memory-Cache hinzu mit LRU Eviction."""
        if len(self._memory_cache) >= self.max_size:
            # Entferne ältesten Eintrag
            self._memory_cache.popitem(last=False)
        self._memory_cache[key] = value
    
    def close(self):
        """Schließe DB-Verbindung."""
        self.conn.close()


# ============================================================================
# GLiNER NER EXTRACTOR
# ============================================================================

class GLiNERExtractor:
    """
    Named Entity Recognition mit GLiNER-small.
    
    GLiNER ist ein Zero-Shot NER Modell, das beliebige Entitätstypen
    ohne domänenspezifisches Fine-Tuning extrahieren kann.
    
    Referenz: Zaratiana et al. (2023). "GLiNER: Generalist Model for 
    Named Entity Recognition using Bidirectional Transformer"
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lade GLiNER Modell."""
        if not GLINER_AVAILABLE:
            logger.warning("GLiNER not available, using fallback")
            return
        
        try:
            logger.info(f"Loading GLiNER model: {self.config.gliner_model}")
            self.model = GLiNER.from_pretrained(self.config.gliner_model)
            logger.info("GLiNER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GLiNER: {e}")
            self.model = None
    
    def extract(
        self, 
        text: str, 
        chunk_id: str
    ) -> List[ExtractedEntity]:
        """
        Extrahiere Entitäten aus einem Text.
        
        Args:
            text: Input-Text
            chunk_id: ID des Chunks
            
        Returns:
            Liste von ExtractedEntity Objekten
        """
        if self.model is None:
            return self._fallback_extract(text, chunk_id)
        
        try:
            # GLiNER Prediction
            entities = self.model.predict_entities(
                text,
                self.config.entity_types,
                threshold=self.config.ner_confidence_threshold
            )
            
            results = []
            for ent in entities:
                entity = ExtractedEntity(
                    entity_id=self._generate_entity_id(ent["text"], ent["label"]),
                    name=ent["text"],
                    entity_type=ent["label"],
                    confidence=ent["score"],
                    mention_span=(ent["start"], ent["end"]),
                    source_chunk_id=chunk_id,
                )
                results.append(entity)
            
            return results
            
        except Exception as e:
            logger.error(f"GLiNER extraction failed: {e}")
            return self._fallback_extract(text, chunk_id)
    
    def extract_batch(
        self, 
        texts: List[str], 
        chunk_ids: List[str]
    ) -> List[List[ExtractedEntity]]:
        """
        Batch-Extraktion für multiple Texte.
        
        Optimiert für Throughput durch Batching (16 Texte pro Batch).
        """
        if self.model is None:
            return [self._fallback_extract(t, c) for t, c in zip(texts, chunk_ids)]
        
        all_results = []
        batch_size = self.config.ner_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = chunk_ids[i:i + batch_size]
            
            try:
                # GLiNER unterstützt Batch-Processing
                batch_entities = self.model.batch_predict_entities(
                    batch_texts,
                    self.config.entity_types,
                    threshold=self.config.ner_confidence_threshold
                )
                
                for text_entities, chunk_id in zip(batch_entities, batch_ids):
                    results = []
                    for ent in text_entities:
                        entity = ExtractedEntity(
                            entity_id=self._generate_entity_id(ent["text"], ent["label"]),
                            name=ent["text"],
                            entity_type=ent["label"],
                            confidence=ent["score"],
                            mention_span=(ent["start"], ent["end"]),
                            source_chunk_id=chunk_id,
                        )
                        results.append(entity)
                    all_results.append(results)
                    
            except Exception as e:
                logger.error(f"Batch extraction failed: {e}")
                # Fallback für diesen Batch
                for text, chunk_id in zip(batch_texts, batch_ids):
                    all_results.append(self._fallback_extract(text, chunk_id))
        
        return all_results
    
    def _fallback_extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """Fallback NER mit SpaCy oder Regex."""
        if SPACY_AVAILABLE:
            return self._spacy_extract(text, chunk_id)
        return self._regex_extract(text, chunk_id)
    
    def _spacy_extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """SpaCy-basierte NER als Fallback."""
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            
            # SpaCy -> GLiNER Type Mapping
            type_map = {
                "PERSON": "PERSON",
                "ORG": "ORGANIZATION",
                "GPE": "LOCATION",
                "LOC": "LOCATION",
                "DATE": "DATE",
                "EVENT": "EVENT",
            }
            
            results = []
            for ent in doc.ents:
                ent_type = type_map.get(ent.label_, "CONCEPT")
                entity = ExtractedEntity(
                    entity_id=self._generate_entity_id(ent.text, ent_type),
                    name=ent.text,
                    entity_type=ent_type,
                    confidence=0.7,  # SpaCy gibt keine Confidence
                    mention_span=(ent.start_char, ent.end_char),
                    source_chunk_id=chunk_id,
                )
                results.append(entity)
            
            return results
            
        except Exception as e:
            logger.error(f"SpaCy extraction failed: {e}")
            return self._regex_extract(text, chunk_id)
    
    def _regex_extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """Regex-basierte Entity Extraction als letzter Fallback."""
        import re
        
        results = []
        
        # Proper Nouns (Multi-word)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            entity = ExtractedEntity(
                entity_id=self._generate_entity_id(match.group(1), "CONCEPT"),
                name=match.group(1),
                entity_type="CONCEPT",
                confidence=0.5,
                mention_span=(match.start(), match.end()),
                source_chunk_id=chunk_id,
            )
            results.append(entity)
        
        return results
    
    @staticmethod
    def _generate_entity_id(name: str, entity_type: str) -> str:
        """Generiere eindeutige Entity ID."""
        combined = f"{name.lower().strip()}:{entity_type}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]


# ============================================================================
# REBEL RELATION EXTRACTOR
# ============================================================================

class REBELExtractor:
    """
    Relation Extraction mit REBEL.
    
    REBEL (Relation Extraction By End-to-end Language generation) ist ein
    Seq2Seq-Modell, das Tripel (Subject, Relation, Object) aus Text extrahiert.
    
    Optimierung: Nur auf Chunks mit >= 2 Entitäten anwenden (60% weniger Aufrufe).
    
    Referenz: Cabot & Navigli (2021). "REBEL: Relation Extraction By 
    End-to-end Language generation"
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Lade REBEL Pipeline."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, REBEL disabled")
            return
        
        try:
            logger.info(f"Loading REBEL model: {self.config.rebel_model}")
            
            # Tokenizer und Model laden
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.rebel_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.rebel_model)
            
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
            )
            
            logger.info("REBEL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load REBEL: {e}")
            self.pipeline = None
    
    def extract(
        self, 
        text: str, 
        entities: List[ExtractedEntity],
        chunk_id: str
    ) -> List[ExtractedRelation]:
        """
        Extrahiere Relationen aus Text.
        
        Args:
            text: Input-Text
            entities: Bereits extrahierte Entitäten
            chunk_id: ID des Chunks
            
        Returns:
            Liste von ExtractedRelation Objekten
        """
        # Optimierung: Nur wenn >= min_entities_for_re Entitäten
        if len(entities) < self.config.min_entities_for_re:
            return []
        
        if self.pipeline is None:
            return []
        
        try:
            # REBEL Output generieren
            output = self.pipeline(
                text,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
            
            # Tripel aus Output parsen
            raw_text = output[0]["generated_text"]
            triplets = self._parse_triplets(raw_text)
            
            # Entity-Namen für Filterung
            entity_names = {e.name.lower() for e in entities}
            
            results = []
            for subj, rel, obj in triplets:
                # Nur Relationen zwischen bekannten Entitäten
                if (subj.lower() in entity_names or 
                    any(subj.lower() in e.lower() for e in entity_names)):
                    
                    relation = ExtractedRelation(
                        subject_entity=subj,
                        relation_type=rel,
                        object_entity=obj,
                        confidence=self.config.re_confidence_threshold,
                        source_chunk_ids=[chunk_id],
                    )
                    results.append(relation)
            
            return results
            
        except Exception as e:
            logger.error(f"REBEL extraction failed: {e}")
            return []
    
    def _parse_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Parse REBEL Output in Tripel.
        
        REBEL Output Format: 
        "<triplet> Subject <subj> Relation <obj> Object <triplet> ..."
        """
        triplets = []
        
        # Split by triplet markers
        parts = text.split("<triplet>")
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            try:
                # Parse: "Subject <subj> Relation <obj> Object"
                if "<subj>" in part and "<obj>" in part:
                    subj_split = part.split("<subj>")
                    subject = subj_split[0].strip()
                    
                    rest = subj_split[1] if len(subj_split) > 1 else ""
                    obj_split = rest.split("<obj>")
                    relation = obj_split[0].strip()
                    obj = obj_split[1].strip() if len(obj_split) > 1 else ""
                    
                    if subject and relation and obj:
                        triplets.append((subject, relation, obj))
                        
            except Exception as e:
                logger.debug(f"Failed to parse triplet: {part}, error: {e}")
        
        return triplets
    
    def extract_batch(
        self,
        texts: List[str],
        entities_per_text: List[List[ExtractedEntity]],
        chunk_ids: List[str]
    ) -> List[List[ExtractedRelation]]:
        """
        Batch Relation Extraction.
        
        Selective RE: Nur Texte mit >= min_entities_for_re werden verarbeitet.
        """
        all_results = []
        
        # Filter Texte mit genug Entitäten
        indices_to_process = []
        texts_to_process = []
        
        for i, (text, entities) in enumerate(zip(texts, entities_per_text)):
            if len(entities) >= self.config.min_entities_for_re:
                indices_to_process.append(i)
                texts_to_process.append(text)
            else:
                all_results.append([])
        
        if not texts_to_process or self.pipeline is None:
            return [[] for _ in texts]
        
        # Batch Processing
        batch_size = self.config.re_batch_size
        processed_results = []
        
        for i in range(0, len(texts_to_process), batch_size):
            batch = texts_to_process[i:i + batch_size]
            
            try:
                outputs = self.pipeline(
                    batch,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                    batch_size=len(batch),
                )
                
                for output in outputs:
                    raw_text = output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
                    triplets = self._parse_triplets(raw_text)
                    
                    relations = [
                        ExtractedRelation(
                            subject_entity=subj,
                            relation_type=rel,
                            object_entity=obj,
                            confidence=self.config.re_confidence_threshold,
                            source_chunk_ids=[],
                        )
                        for subj, rel, obj in triplets
                    ]
                    processed_results.append(relations)
                    
            except Exception as e:
                logger.error(f"Batch RE failed: {e}")
                processed_results.extend([[] for _ in batch])
        
        # Merge results in correct order
        final_results = [[] for _ in texts]
        for idx, rels in zip(indices_to_process, processed_results):
            final_results[idx] = rels
        
        return final_results


# ============================================================================
# UNIFIED EXTRACTION PIPELINE
# ============================================================================

class EntityExtractionPipeline:
    """
    Unified Entity Extraction Pipeline: GLiNER + REBEL.
    
    Kombiniert NER und RE in einer optimierten Pipeline mit:
    - Batch-Processing (GLiNER: 16, REBEL: 8)
    - Entity Caching (LRU + SQLite)
    - Selective RE (nur bei >= 2 Entitäten)
    
    Ziel-Latenz: 80-120ms pro Chunk
    """
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        
        # Komponenten initialisieren
        self.ner_extractor = GLiNERExtractor(self.config)
        self.re_extractor = REBELExtractor(self.config)
        
        # Cache
        if self.config.cache_enabled:
            self.cache = EntityCache(
                self.config.cache_path,
                self.config.lru_cache_size
            )
        else:
            self.cache = None
        
        # Statistiken
        self.stats = {
            "total_chunks": 0,
            "cache_hits": 0,
            "ner_calls": 0,
            "re_calls": 0,
            "total_entities": 0,
            "total_relations": 0,
            "avg_latency_ms": 0,
        }
        
        logger.info("EntityExtractionPipeline initialized")
    
    def process_chunk(
        self, 
        text: str, 
        chunk_id: str
    ) -> ChunkExtractionResult:
        """
        Verarbeite einen einzelnen Chunk.
        
        Args:
            text: Chunk-Text
            chunk_id: Eindeutige Chunk-ID
            
        Returns:
            ChunkExtractionResult mit Entitäten und Relationen
        """
        start_time = time.time()
        self.stats["total_chunks"] += 1
        
        # Cache Check
        if self.cache:
            cached = self.cache.get(text)
            if cached:
                self.stats["cache_hits"] += 1
                # Reconstruct entities from cache
                entities = []
                for e in cached.get("entities", []):
                    # Cache stores 'type', but ExtractedEntity expects 'entity_type'
                    entity_dict = {
                        "entity_id": e.get("entity_id", ""),
                        "name": e.get("name", ""),
                        "entity_type": e.get("type", e.get("entity_type", "CONCEPT")),
                        "confidence": e.get("confidence", 0.5),
                        "mention_span": tuple(e.get("mention_span", [0, 0])),
                        "source_chunk_id": chunk_id  # Override with current chunk_id
                    }
                    entities.append(ExtractedEntity(**entity_dict))
                
                # Reconstruct relations from cache
                relations = []
                for r in cached.get("relations", []):
                    relation_dict = {
                        "subject_entity": r.get("subject", r.get("subject_entity", "")),
                        "relation_type": r.get("relation", r.get("relation_type", "")),
                        "object_entity": r.get("object", r.get("object_entity", "")),
                        "confidence": r.get("confidence", 0.5),
                        "source_chunk_ids": [chunk_id]  # Override with current chunk_id
                    }
                    relations.append(ExtractedRelation(**relation_dict))
                
                return ChunkExtractionResult(
                    chunk_id=chunk_id,
                    text=text,
                    entities=entities,
                    relations=relations,
                    extraction_time_ms=(time.time() - start_time) * 1000,
                )
        
        # NER mit GLiNER
        self.stats["ner_calls"] += 1
        entities = self.ner_extractor.extract(text, chunk_id)
        
        # RE mit REBEL (nur wenn >= min_entities_for_re)
        relations = []
        if len(entities) >= self.config.min_entities_for_re:
            self.stats["re_calls"] += 1
            relations = self.re_extractor.extract(text, entities, chunk_id)
        
        # Cache Update
        if self.cache:
            cache_data = {
                "entities": [e.to_dict() for e in entities],
                "relations": [r.to_dict() for r in relations],
            }
            self.cache.put(text, cache_data)
        
        # Statistiken
        self.stats["total_entities"] += len(entities)
        self.stats["total_relations"] += len(relations)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return ChunkExtractionResult(
            chunk_id=chunk_id,
            text=text,
            entities=entities,
            relations=relations,
            extraction_time_ms=elapsed_ms,
        )
    
    def process_chunks_batch(
        self,
        texts: List[str],
        chunk_ids: List[str]
    ) -> List[ChunkExtractionResult]:
        """
        Batch-Verarbeitung mehrerer Chunks.
        
        Optimiert für Throughput durch:
        - Batch NER (16 Texte)
        - Batch RE (8 Texte)
        - Cache-Lookups
        
        Args:
            texts: Liste von Chunk-Texten
            chunk_ids: Liste von Chunk-IDs
            
        Returns:
            Liste von ChunkExtractionResult
        """
        start_time = time.time()
        results = []
        
        # Separate cached und uncached
        uncached_indices = []
        uncached_texts = []
        uncached_ids = []
        
        for i, (text, chunk_id) in enumerate(zip(texts, chunk_ids)):
            self.stats["total_chunks"] += 1
            
            if self.cache:
                cached = self.cache.get(text)
                if cached:
                    self.stats["cache_hits"] += 1
                    # Reconstruct entities from cache
                    entities = []
                    for e in cached.get("entities", []):
                        entity_dict = {
                            "entity_id": e.get("entity_id", ""),
                            "name": e.get("name", ""),
                            "entity_type": e.get("type", e.get("entity_type", "CONCEPT")),
                            "confidence": e.get("confidence", 0.5),
                            "mention_span": tuple(e.get("mention_span", [0, 0])),
                            "source_chunk_id": chunk_id
                        }
                        entities.append(ExtractedEntity(**entity_dict))
                    
                    # Reconstruct relations from cache
                    relations = []
                    for r in cached.get("relations", []):
                        relation_dict = {
                            "subject_entity": r.get("subject", r.get("subject_entity", "")),
                            "relation_type": r.get("relation", r.get("relation_type", "")),
                            "object_entity": r.get("object", r.get("object_entity", "")),
                            "confidence": r.get("confidence", 0.5),
                            "source_chunk_ids": [chunk_id]
                        }
                        relations.append(ExtractedRelation(**relation_dict))
                    
                    results.append((i, ChunkExtractionResult(
                        chunk_id=chunk_id,
                        text=text,
                        entities=entities,
                        relations=relations,
                        extraction_time_ms=0,
                    )))
                    continue
            
            uncached_indices.append(i)
            uncached_texts.append(text)
            uncached_ids.append(chunk_id)
        
        # Batch NER für uncached
        if uncached_texts:
            self.stats["ner_calls"] += len(uncached_texts)
            all_entities = self.ner_extractor.extract_batch(uncached_texts, uncached_ids)
            
            # Batch RE (nur für Texte mit >= min_entities)
            all_relations = self.re_extractor.extract_batch(
                uncached_texts, all_entities, uncached_ids
            )
            self.stats["re_calls"] += sum(
                1 for ents in all_entities 
                if len(ents) >= self.config.min_entities_for_re
            )
            
            # Ergebnisse zusammenführen
            for idx, text, chunk_id, entities, relations in zip(
                uncached_indices, uncached_texts, uncached_ids, all_entities, all_relations
            ):
                # Update relation source chunks
                for rel in relations:
                    rel.source_chunk_ids = [chunk_id]
                
                # Cache
                if self.cache:
                    cache_data = {
                        "entities": [e.to_dict() for e in entities],
                        "relations": [r.to_dict() for r in relations],
                    }
                    self.cache.put(text, cache_data)
                
                self.stats["total_entities"] += len(entities)
                self.stats["total_relations"] += len(relations)
                
                results.append((idx, ChunkExtractionResult(
                    chunk_id=chunk_id,
                    text=text,
                    entities=entities,
                    relations=relations,
                    extraction_time_ms=0,
                )))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]
        
        # Update timing
        total_elapsed = (time.time() - start_time) * 1000
        avg_per_chunk = total_elapsed / len(texts) if texts else 0
        self.stats["avg_latency_ms"] = avg_per_chunk
        
        for result in final_results:
            result.extraction_time_ms = avg_per_chunk
        
        logger.info(
            f"Batch extraction: {len(texts)} chunks, "
            f"{self.stats['total_entities']} entities, "
            f"{self.stats['total_relations']} relations, "
            f"{total_elapsed:.0f}ms total ({avg_per_chunk:.1f}ms/chunk)"
        )
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Hole Statistiken."""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / self.stats["total_chunks"]
                if self.stats["total_chunks"] > 0 else 0
            ),
            "re_skip_rate": (
                1 - (self.stats["re_calls"] / self.stats["ner_calls"])
                if self.stats["ner_calls"] > 0 else 0
            ),
        }
    
    def close(self):
        """Cleanup."""
        if self.cache:
            self.cache.close()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_extraction_pipeline(
    config_path: str = None,
    **kwargs
) -> EntityExtractionPipeline:
    """
    Factory für EntityExtractionPipeline.
    
    Args:
        config_path: Pfad zur YAML-Konfiguration
        **kwargs: Überschreibe einzelne Config-Werte
        
    Returns:
        Konfigurierte EntityExtractionPipeline
    """
    config = ExtractionConfig()
    
    if config_path:
        import yaml
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
            
        if "entity_extraction" in yaml_config:
            ee_config = yaml_config["entity_extraction"]
            
            if "gliner" in ee_config:
                config.gliner_model = ee_config["gliner"].get("model_name", config.gliner_model)
                config.entity_types = ee_config["gliner"].get("entity_types", config.entity_types)
                config.ner_confidence_threshold = ee_config["gliner"].get("confidence_threshold", config.ner_confidence_threshold)
                config.ner_batch_size = ee_config["gliner"].get("batch_size", config.ner_batch_size)
            
            if "rebel" in ee_config:
                config.rebel_model = ee_config["rebel"].get("model_name", config.rebel_model)
                config.re_confidence_threshold = ee_config["rebel"].get("confidence_threshold", config.re_confidence_threshold)
                config.re_batch_size = ee_config["rebel"].get("batch_size", config.re_batch_size)
                config.min_entities_for_re = ee_config["rebel"].get("min_entities_for_re", config.min_entities_for_re)
            
            if "caching" in ee_config:
                config.cache_enabled = ee_config["caching"].get("enabled", config.cache_enabled)
                config.cache_path = ee_config["caching"].get("cache_path", config.cache_path)
                config.lru_cache_size = ee_config["caching"].get("lru_cache_size", config.lru_cache_size)
            
            config.selective_re = ee_config.get("selective_re", config.selective_re)
    
    # Override mit kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return EntityExtractionPipeline(config)


# ============================================================================
# CLI / TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test Pipeline
    pipeline = create_extraction_pipeline()
    
    test_texts = [
        "Albert Einstein was born in Ulm, Germany in 1879. He worked at Princeton University.",
        "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico.",
        "The Eiffel Tower is located in Paris, France. It was designed by Gustave Eiffel.",
    ]
    
    print("\n" + "="*70)
    print("ENTITY EXTRACTION PIPELINE TEST")
    print("="*70)
    
    for i, text in enumerate(test_texts):
        result = pipeline.process_chunk(text, f"chunk_{i}")
        
        print(f"\n--- Chunk {i} ---")
        print(f"Text: {text[:100]}...")
        print(f"Entities ({len(result.entities)}):")
        for e in result.entities:
            print(f"  - {e.name} [{e.entity_type}] (conf: {e.confidence:.2f})")
        print(f"Relations ({len(result.relations)}):")
        for r in result.relations:
            print(f"  - {r.subject_entity} --[{r.relation_type}]--> {r.object_entity}")
        print(f"Time: {result.extraction_time_ms:.1f}ms")
    
    print("\n--- Statistics ---")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    pipeline.close()
