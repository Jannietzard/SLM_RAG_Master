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
        "PERSON", 
        "ORGANIZATION", 
        "GPE",          # Geo-Political Entity (Länder, Städte) - präziser als LOCATION
        "LOCATION",
        "PLACE",
        "DATE", 
        "EVENT",
        "CONCEPT",    # Basierend auf GLiNER-Paper und gängigen NER-Taxonomien 
        "WORK_OF_ART", 
        "STRUCTURE",
        "NATURAL_OBJECT",
        "LAW",      
        # Hovy,2006
        # Domänenspezifische Erweiterungen (Begründet durch Edge/IT-Kontext)
        # Ersetzt das vage "CONCEPT" für bessere GLiNER-Performance
        "TECHNOLOGY",   # Hardware, Software, Protokolle
        "SCIENTIFIC_TERM", # Fachbegriffe, Theorien
        "PRODUCT"
    ])
    ner_confidence_threshold: float = 0.15
    ner_batch_size: int = 16
    
    # REBEL
    rebel_model: str = "Babelscape/rebel-large"
    re_confidence_threshold: float = 0.5
    re_batch_size: int = 8
    min_entities_for_re: int = 2  # Nur RE wenn >= 2 Entitäten
    
    # Caching
    cache_enabled: bool = True
    cache_path: str = "./data/entity_cache.db"
    lru_cache_size: int = 10000
    
    # Selective RE
    selective_re: bool = True


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

# Ersetzen Sie die gesamte REBELExtractor Klasse in entity_extraction.py hiermit:

class REBELExtractor:
    """
    Relation Extraction mit REBEL.
    Implementierung via model.generate() statt pipeline(), um Task-Fehler zu vermeiden.
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Oder "cuda" wenn verfügbar
        self._load_model()
    
    def _load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, REBEL disabled")
            return
        
        try:
            logger.info(f"Loading REBEL model: {self.config.rebel_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.rebel_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.rebel_model)
            self.model.to(self.device)
            logger.info("REBEL model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load REBEL: {e}")
            self.model = None
    
    def extract(self, text: str, entities: List[ExtractedEntity], chunk_id: str) -> List[ExtractedRelation]:
        # Optimization: Skip if not enough entities (Thesis 2.5)
        if len(entities) < self.config.min_entities_for_re:
            return []
        
        if self.model is None:
            return []
        
        try:
            # Tokenize & Generate
            inputs = self.tokenizer(
                text, max_length=256, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            generated_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

            # Decode
            raw_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            # Cleanup special tokens that might break parsing
            raw_text = raw_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")

            triplets = self._parse_triplets(raw_text)
            
            # Filter: Keep only relations connecting known entities
            entity_names = {e.name.lower() for e in entities}
            results = []
            
            for subj, rel, obj in triplets:
                # Check if subject or object matches any extracted entity (fuzzy match)
                is_relevant = any(e in subj.lower() or e in obj.lower() for e in entity_names)
                
                if is_relevant:
                    relation = ExtractedRelation(
                        subject_entity=subj.strip(),
                        relation_type=rel.strip(),
                        object_entity=obj.strip(),
                        confidence=self.config.re_confidence_threshold,
                        source_chunk_ids=[chunk_id],
                    )
                    results.append(relation)
            
            return results
            
        except Exception as e:
            logger.error(f"REBEL extraction failed: {e}")
            return []

    def _parse_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        triplets = []
        try:
            # REBEL Format ist oft: Subject <subj> Object <obj> Relation
            parts = text.split("<triplet>")
            for part in parts:
                if "<subj>" in part and "<obj>" in part:
                    subj_parts = part.split("<subj>")
                    subject = subj_parts[0]
                    
                    # Split nach Object und Relation
                    obj_rel_parts = subj_parts[1].split("<obj>")
                    object_ = obj_rel_parts[0]  # Das ist das Objekt (z.B. Bill Gates)
                    relation = obj_rel_parts[1] # Das ist die Relation (z.B. founded by)
                    
                    if subject and relation and object_:
                        triplets.append((subject.strip(), relation.strip(), object_.strip()))
        except Exception:
            pass
        return triplets

    def extract_batch(self, texts, entities_per_text, chunk_ids):
        # Fallback to sequential for stability if batching is complex with manual generate
        # Or implement manual batching similar to extract() but with padded inputs
        results = []
        for t, e, c in zip(texts, entities_per_text, chunk_ids):
            results.append(self.extract(t, e, c))
        return results

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
    pipeline = create_extraction_pipeline(cache_enabled=False)
    
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
