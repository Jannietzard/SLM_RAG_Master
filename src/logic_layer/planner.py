"""
===============================================================================
S_P: Regelbasierter Query Planner
===============================================================================

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"
Artefakt B: Agent-Based Query Processing

===============================================================================
ÜBERBLICK
===============================================================================

Der Planner (S_P) ist die erste Stufe des Agentic RAG Systems und fungiert als
deterministischer Router für Query-Analyse und Retrieval-Plan-Generierung.

Kernfunktionen gemäß Masterarbeit:
1. QUERY-KLASSIFIKATION (Heuristisch)
   - Nutzt SpaCy's Rule-Based Matcher statt ML-Modellen
   - Klassifiziert: Single-Hop, Multi-Hop, Comparison, Temporal Reasoning
   - Minimiert Latenz durch leichtgewichtige linguistische Heuristiken

2. ENTITY & BRIDGE DETECTION
   - Extraktion via SpaCy NER (Confidence > 0.7)
   - Dependency Parsing für syntaktische Abhängigkeiten
   - Bridge Entities für Multi-Hop Graph-Traversierung

3. PLAN-GENERIERUNG
   - Strukturierter JSON-Retrieval-Plan
   - Definiert: Strategie, Hop-Sequenz, Constraints
   - Strategien: Vector-Only, Graph-Only, Hybrid

===============================================================================
ARCHITEKTUR
===============================================================================

    User Query
        │
        ▼
    ┌─────────────────────────────────────────────────────┐
    │                    S_P (PLANNER)                     │
    │                                                      │
    │   ┌──────────────┐    ┌──────────────┐              │
    │   │  SpaCy NLP   │───▶│   Query      │              │
    │   │  Processing  │    │   Classifier │              │
    │   └──────────────┘    └──────────────┘              │
    │          │                    │                      │
    │          ▼                    ▼                      │
    │   ┌──────────────┐    ┌──────────────┐              │
    │   │   Entity &   │    │   Bridge     │              │
    │   │   NER Extract│    │   Detection  │              │
    │   └──────────────┘    └──────────────┘              │
    │          │                    │                      │
    │          └────────┬───────────┘                      │
    │                   ▼                                  │
    │          ┌──────────────┐                            │
    │          │  Retrieval   │                            │
    │          │  Plan Gen    │                            │
    │          └──────────────┘                            │
    │                   │                                  │
    └───────────────────┼──────────────────────────────────┘
                        ▼
                Retrieval Plan (JSON)
                   an Navigator (S_N)

===============================================================================
ACADEMIC REFERENCES
===============================================================================

- Ghallab, M., Nau, D., & Traverso, P. (2004). 
  "Automated Planning: Theory and Practice." Morgan Kaufmann.
- AIPlan4EU Consortium (2023). "Unified Planning Framework."
- International Planning Competition (IPC)

===============================================================================
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


# =============================================================================
# SPACY INTEGRATION
# =============================================================================
# SpaCy wird für Entity Extraction und Dependency Parsing verwendet.
# Falls nicht verfügbar, nutzen wir Regex-Fallbacks.

try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.tokens import Doc, Span
    
    # Versuche das englische Modell zu laden
    # Für Edge-Deployment empfohlen: en_core_web_sm (klein, schnell)
    try:
        NLP = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("SpaCy 'en_core_web_sm' Modell geladen für Query-Analyse")
    except OSError:
        # Fallback: Versuche das deutsche Modell
        try:
            NLP = spacy.load("de_core_news_sm")
            SPACY_AVAILABLE = True
            logger.info("SpaCy 'de_core_news_sm' Modell geladen für Query-Analyse")
        except OSError:
            NLP = None
            SPACY_AVAILABLE = False
            logger.warning(
                "Kein SpaCy-Modell verfügbar. Installiere mit:\n"
                "  python -m spacy download en_core_web_sm"
            )
except ImportError:
    NLP = None
    SPACY_AVAILABLE = False
    Matcher = None
    logger.warning(
        "SpaCy nicht installiert. Installiere mit:\n"
        "  pip install spacy\n"
        "  python -m spacy download en_core_web_sm"
    )


# =============================================================================
# DATENSTRUKTUREN
# =============================================================================

class QueryType(Enum):
    """
    Klassifikation von Query-Typen basierend auf Komplexität.
    
    Gemäß Masterarbeit Abschnitt 3.2:
    - SINGLE_HOP: Einfache Faktenfrage, ein Retrieval-Schritt
    - MULTI_HOP: Sequentielle Abhängigkeiten, Bridge Entities nötig
    - COMPARISON: Paralleler Retrieval + Vergleichslogik
    - TEMPORAL: Zeitliche Reasoning-Komponente
    - AGGREGATE: Kombination mehrerer Ergebnisse
    """
    SINGLE_HOP = "single_hop"       # z.B. "Was ist die Hauptstadt von Frankreich?"
    MULTI_HOP = "multi_hop"         # z.B. "Wer ist der Regisseur des Films mit Tom Hanks?"
    COMPARISON = "comparison"       # z.B. "Ist Berlin älter als München?"
    TEMPORAL = "temporal"           # z.B. "Was passierte nach dem 2. Weltkrieg?"
    AGGREGATE = "aggregate"         # z.B. "Liste alle Filme von 2020 auf"
    INTERSECTION = "intersection"   # z.B. "Was haben A und B gemeinsam?"


class RetrievalStrategy(Enum):
    """
    Retrieval-Strategien basierend auf Query-Komplexität.
    
    Gemäß Masterarbeit Abschnitt 3.2:
    - VECTOR_ONLY: Für einfache Single-Hop Queries
    - GRAPH_ONLY: Wenn explizite Relationen benötigt werden
    - HYBRID: Für komplexe Multi-Hop und Comparison Queries
    """
    VECTOR_ONLY = "vector_only"     # Schnell, für einfache Queries
    GRAPH_ONLY = "graph_only"       # Für Relation-basierte Queries
    HYBRID = "hybrid"               # Kombination für komplexe Queries


@dataclass
class EntityInfo:
    """
    Informationen über eine extrahierte Entity.
    
    Attributes:
        text: Der Entity-Text
        label: NER-Label (PERSON, ORG, GPE, etc.)
        confidence: Konfidenz der Extraktion (0.0-1.0)
        start_char: Start-Position im Original-Text
        end_char: End-Position im Original-Text
        is_bridge: True wenn dies eine Bridge Entity für Multi-Hop ist
    """
    text: str
    label: str = "UNKNOWN"
    confidence: float = 1.0
    start_char: int = 0
    end_char: int = 0
    is_bridge: bool = False


@dataclass
class HopStep:
    """
    Ein Schritt in der Multi-Hop Reasoning-Kette.
    
    Attributes:
        step_id: Eindeutige Schritt-ID
        sub_query: Die Sub-Query für diesen Schritt
        target_entities: Ziel-Entities für diesen Schritt
        depends_on: IDs der Schritte, von denen dieser abhängt
        is_bridge: True wenn dieser Schritt eine Bridge-Query ist
    """
    step_id: int
    sub_query: str
    target_entities: List[str] = field(default_factory=list)
    depends_on: List[int] = field(default_factory=list)
    is_bridge: bool = False


@dataclass
class RetrievalPlan:
    """
    Strukturierter Retrieval-Plan für den Navigator (S_N).
    
    Dies ist das Hauptausgabeformat des Planners, das vom Navigator
    zur Ausführung des Hybrid Retrieval verwendet wird.
    
    Attributes:
        original_query: Die ursprüngliche User-Query
        query_type: Klassifizierter Query-Typ
        strategy: Gewählte Retrieval-Strategie
        entities: Liste extrahierter Entities mit Metadaten
        hop_sequence: Geordnete Liste von Hop-Schritten
        sub_queries: Flache Liste aller Sub-Queries für Retrieval
        constraints: Zusätzliche Constraints (temporal, comparison, etc.)
        estimated_hops: Geschätzte Anzahl der Retrieval-Hops
        confidence: Konfidenz der Query-Klassifikation
        metadata: Zusätzliche Metadaten für Debugging
    """
    original_query: str
    query_type: QueryType
    strategy: RetrievalStrategy
    entities: List[EntityInfo] = field(default_factory=list)
    hop_sequence: List[HopStep] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_hops: int = 1
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für JSON-Serialisierung."""
        return {
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "strategy": self.strategy.value,
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "confidence": e.confidence,
                    "is_bridge": e.is_bridge
                }
                for e in self.entities
            ],
            "hop_sequence": [
                {
                    "step_id": h.step_id,
                    "sub_query": h.sub_query,
                    "target_entities": h.target_entities,
                    "depends_on": h.depends_on,
                    "is_bridge": h.is_bridge
                }
                for h in self.hop_sequence
            ],
            "sub_queries": self.sub_queries,
            "constraints": self.constraints,
            "estimated_hops": self.estimated_hops,
            "confidence": self.confidence,
        }
    
    def to_json(self) -> str:
        """Serialisiere zu JSON-String."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class PlannerConfig:
    """
    Konfiguration für den Query Planner.
    
    Attributes:
        min_entity_confidence: Minimale NER-Konfidenz für Entity-Extraktion
        max_entities: Maximale Anzahl zu extrahierender Entities
        enable_bridge_detection: Bridge Entity Detection aktivieren
        enable_temporal_parsing: Temporale Constraints parsen
        default_strategy: Standard-Strategie wenn keine Klassifikation möglich
    """
    min_entity_confidence: float = 0.7  # Gemäß Masterarbeit: Confidence > 0.7
    max_entities: int = 10
    enable_bridge_detection: bool = True
    enable_temporal_parsing: bool = True
    default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID


# =============================================================================
# QUERY CLASSIFIER
# =============================================================================

class QueryClassifier:
    """
    Regelbasierter Query Classifier mit SpaCy Matcher.
    
    Gemäß Masterarbeit Abschnitt 3.2:
    "Anstelle eines ML-Modells kommt SpaCy's Rule-Based Matcher zum Einsatz.
    Dieser identifiziert durch lexikalische Muster (Pattern Matching) effizient
    Query-Typen wie Comparison, Temporal oder Multi-Hop."
    
    Der Classifier nutzt:
    1. Lexikalische Pattern für Query-Typ-Erkennung
    2. Syntaktische Strukturen für Komplexitäts-Bestimmung
    3. Entity-Dichte für Multi-Hop Identifikation
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # LEXIKALISCHE PATTERN FÜR QUERY-KLASSIFIKATION
    # ─────────────────────────────────────────────────────────────────────────
    
    # Comparison-Indikatoren: Vergleichswörter und -strukturen
    COMPARISON_PATTERNS = [
        r"\b(older|younger|taller|shorter|bigger|smaller|larger|higher|lower)\s+than\b",
        r"\b(more|less|fewer)\s+\w+\s+than\b",
        r"\b(compare|comparison|versus|vs\.?|vs)\b",
        r"\bdifference\s+between\b",
        r"\bwhich\s+(is|was|are|were)\s+\w*(er|est)\b",
        r"\b(better|worse|best|worst)\b.*\bor\b",
        r"\bor\b.*\?(which|what)\s+(is|was)\s+\w*(er|est)",
    ]
    
    # Temporal-Indikatoren: Zeitbezüge und temporale Strukturen
    TEMPORAL_PATTERNS = [
        r"\b(before|after|during|since|until|when|while)\b",
        r"\b(year|month|day|century|decade|era)\s+\d+",
        r"\b\d{4}\b",  # Jahreszahlen
        r"\b(first|last|latest|earliest|recent|previous|next)\b",
        r"\b(history|historical|timeline|chronolog)\w*\b",
        r"\b(began|started|ended|founded|established)\b",
    ]
    
    # Multi-Hop Indikatoren: Verschachtelte Strukturen
    MULTI_HOP_PATTERNS = [
        r"\bof\s+the\s+\w+\s+(that|which|who)\b",
        r"\bwhere\s+.+\s+(was|is|were|are)\b",
        r"\b\w+\s+of\s+the\s+\w+\s+of\b",
        r"'s\s+\w+'s",  # Possessiv-Ketten: "John's sister's husband"
        r"\b(who|what)\s+\w+\s+(the|a)\s+\w+\s+(that|which)\b",
    ]
    
    # Intersection-Indikatoren: Gemeinsame Eigenschaften
    INTERSECTION_PATTERNS = [
        r"\bboth\s+.+\s+and\b",
        r"\bin\s+common\b",
        r"\b(also|too)\b.*\band\b",
        r"\bshared\s+(by|between)\b",
    ]
    
    # Aggregation-Indikatoren: Listen und Zusammenfassungen
    AGGREGATE_PATTERNS = [
        r"\b(list|enumerate|all|every|count|how\s+many)\b",
        r"\b(summarize|summary|overview)\b",
        r"\bwhat\s+(are|were)\s+the\b",
    ]
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialisiere den Query Classifier.
        
        Args:
            config: Planner-Konfiguration
        """
        self.config = config or PlannerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Kompiliere Regex-Pattern für Performance
        self._compiled_patterns = {
            QueryType.COMPARISON: [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS],
            QueryType.TEMPORAL: [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS],
            QueryType.MULTI_HOP: [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS],
            QueryType.INTERSECTION: [re.compile(p, re.IGNORECASE) for p in self.INTERSECTION_PATTERNS],
            QueryType.AGGREGATE: [re.compile(p, re.IGNORECASE) for p in self.AGGREGATE_PATTERNS],
        }
        
        # SpaCy Matcher für komplexere Pattern
        self._setup_spacy_matcher()
        
        self.logger.info("QueryClassifier initialisiert")
    
    def _setup_spacy_matcher(self):
        """
        Initialisiere SpaCy Matcher mit linguistischen Pattern.
        
        Der Matcher erkennt syntaktische Strukturen, die auf
        Query-Komplexität hindeuten.
        """
        if not SPACY_AVAILABLE or NLP is None:
            self.matcher = None
            return
        
        self.matcher = Matcher(NLP.vocab)
        
        # Pattern für Multi-Hop: "of the X that/which Y"
        # Beispiel: "the director of the film that won"
        multi_hop_pattern = [
            {"LOWER": "of"},
            {"LOWER": "the"},
            {"POS": {"IN": ["NOUN", "PROPN"]}},
            {"LOWER": {"IN": ["that", "which", "who"]}}
        ]
        self.matcher.add("MULTI_HOP", [multi_hop_pattern])
        
        # Pattern für Comparison: "X than Y"
        comparison_pattern = [
            {"TAG": {"IN": ["JJR", "RBR"]}},  # Komparativ
            {"LOWER": "than"}
        ]
        self.matcher.add("COMPARISON", [comparison_pattern])
        
        self.logger.debug("SpaCy Matcher konfiguriert")
    
    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Klassifiziere eine Query und bestimme den Query-Typ.
        
        Algorithmus:
        1. Prüfe Pattern-Matches für jeden Query-Typ
        2. Zähle Matches und berechne Konfidenz
        3. Bei mehreren Matches: Wähle spezifischsten Typ
        4. Fallback: SINGLE_HOP
        
        Args:
            query: Die zu klassifizierende Query
            
        Returns:
            Tuple von (QueryType, confidence)
        """
        query = query.strip()
        scores = {qt: 0.0 for qt in QueryType}
        
        # ─────────────────────────────────────────────────────────────────────
        # PHASE 1: Regex Pattern Matching
        # ─────────────────────────────────────────────────────────────────────
        
        for query_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[query_type] += 1.0
        
        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2: SpaCy Matcher (wenn verfügbar)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.matcher and NLP:
            doc = NLP(query)
            matches = self.matcher(doc)
            
            for match_id, start, end in matches:
                rule_name = NLP.vocab.strings[match_id]
                if rule_name == "MULTI_HOP":
                    scores[QueryType.MULTI_HOP] += 1.5  # Höheres Gewicht für syntaktische Matches
                elif rule_name == "COMPARISON":
                    scores[QueryType.COMPARISON] += 1.5
        
        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3: Entity-Dichte Analyse (Multi-Hop Heuristik)
        # ─────────────────────────────────────────────────────────────────────
        
        # Hohe Entity-Dichte deutet auf Multi-Hop hin
        if SPACY_AVAILABLE and NLP:
            doc = NLP(query)
            entity_count = len([ent for ent in doc.ents])
            noun_count = len([token for token in doc if token.pos_ in ["NOUN", "PROPN"]])
            
            # Mehr als 2 Entities oder 4+ Nomen → wahrscheinlich Multi-Hop
            if entity_count > 2 or noun_count > 4:
                scores[QueryType.MULTI_HOP] += 0.5
        
        # ─────────────────────────────────────────────────────────────────────
        # PHASE 4: Bestimme finalen Query-Typ
        # ─────────────────────────────────────────────────────────────────────
        
        # Finde den Typ mit dem höchsten Score
        max_score = max(scores.values())
        
        if max_score == 0:
            # Kein Pattern gematcht → Single-Hop
            return QueryType.SINGLE_HOP, 0.8
        
        # Wähle den Query-Typ mit dem höchsten Score
        # Bei Gleichstand: Priorisiere spezifischere Typen
        priority = [
            QueryType.COMPARISON,
            QueryType.TEMPORAL,
            QueryType.MULTI_HOP,
            QueryType.INTERSECTION,
            QueryType.AGGREGATE,
        ]
        
        for qt in priority:
            if scores[qt] == max_score:
                # Berechne Konfidenz basierend auf Score
                confidence = min(0.95, 0.6 + (max_score * 0.15))
                return qt, confidence
        
        # Fallback
        return QueryType.SINGLE_HOP, 0.7


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class EntityExtractor:
    """
    Entity Extractor mit SpaCy NER und Bridge Detection.
    
    Gemäß Masterarbeit Abschnitt 3.2:
    "Die Extraktion relevanter Entitäten erfolgt via SpaCy NER (Confidence > 0.7).
    Für komplexe Fragen nutzt das System Dependency Parsing, um syntaktische
    Abhängigkeiten aufzulösen. Dies ermöglicht die Identifikation von Bridge
    Entities."
    """
    
    # Entity-Typen, die für RAG relevant sind
    RELEVANT_ENTITY_TYPES = {
        "PERSON",      # Personen
        "ORG",         # Organisationen
        "GPE",         # Geo-politische Entities (Länder, Städte)
        "LOC",         # Orte
        "PRODUCT",     # Produkte
        "EVENT",       # Events
        "WORK_OF_ART", # Kunstwerke, Filme, Bücher
        "LAW",         # Gesetze
        "DATE",        # Datumsangaben
        "TIME",        # Zeitangaben
        "MONEY",       # Geldbeträge
        "QUANTITY",    # Mengen
        "NORP",        # Nationalitäten, religiöse/politische Gruppen
    }
    
    # Regex-Fallback Pattern für Entity-Extraktion
    ENTITY_PATTERNS = [
        (r'"([^"]+)"', "QUOTED"),                           # Zitierte Strings
        (r"'([^']+)'", "QUOTED"),                           # Einfach zitiert
        (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", "PROPN"), # Multi-Word Proper Nouns
        (r"\b([A-Z][a-z]{2,})\b", "PROPN"),                 # Single Proper Nouns
        (r"\b(\d{4})\b", "DATE"),                           # Jahreszahlen
    ]
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialisiere den Entity Extractor.
        
        Args:
            config: Planner-Konfiguration
        """
        self.config = config or PlannerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Kompiliere Regex-Pattern
        self._compiled_patterns = [
            (re.compile(pattern), label)
            for pattern, label in self.ENTITY_PATTERNS
        ]
        
        self.logger.info("EntityExtractor initialisiert")
    
    def extract(self, query: str) -> List[EntityInfo]:
        """
        Extrahiere Entities aus einer Query.
        
        Verwendet SpaCy NER wenn verfügbar, sonst Regex-Fallback.
        
        Args:
            query: Die Query, aus der Entities extrahiert werden
            
        Returns:
            Liste von EntityInfo-Objekten
        """
        entities = []
        seen_texts = set()  # Für Deduplizierung
        
        # ─────────────────────────────────────────────────────────────────────
        # METHODE 1: SpaCy NER (wenn verfügbar)
        # ─────────────────────────────────────────────────────────────────────
        
        if SPACY_AVAILABLE and NLP:
            doc = NLP(query)
            
            for ent in doc.ents:
                # Filtere nach relevanten Entity-Typen
                if ent.label_ in self.RELEVANT_ENTITY_TYPES:
                    # SpaCy liefert keine Konfidenz, daher schätzen wir basierend auf Label
                    confidence = self._estimate_confidence(ent)
                    
                    if confidence >= self.config.min_entity_confidence:
                        entity_text = ent.text.strip()
                        
                        if entity_text.lower() not in seen_texts:
                            entities.append(EntityInfo(
                                text=entity_text,
                                label=ent.label_,
                                confidence=confidence,
                                start_char=ent.start_char,
                                end_char=ent.end_char,
                                is_bridge=False
                            ))
                            seen_texts.add(entity_text.lower())
        
        # ─────────────────────────────────────────────────────────────────────
        # METHODE 2: Regex-Fallback (ergänzend oder als Hauptmethode)
        # ─────────────────────────────────────────────────────────────────────
        
        for pattern, label in self._compiled_patterns:
            for match in pattern.finditer(query):
                text = match.group(1) if match.lastindex else match.group(0)
                text = text.strip()
                
                # Filtere zu kurze oder bereits gesehene Entities
                if len(text) > 2 and text.lower() not in seen_texts:
                    # Filtere common words
                    if not self._is_stopword(text):
                        entities.append(EntityInfo(
                            text=text,
                            label=label,
                            confidence=0.75,  # Niedrigere Konfidenz für Regex
                            start_char=match.start(),
                            end_char=match.end(),
                            is_bridge=False
                        ))
                        seen_texts.add(text.lower())
        
        # Sortiere nach Position im Text
        entities.sort(key=lambda e: e.start_char)
        
        # Limitiere auf max_entities
        return entities[:self.config.max_entities]
    
    def detect_bridge_entities(
        self, 
        query: str, 
        entities: List[EntityInfo]
    ) -> List[EntityInfo]:
        """
        Identifiziere Bridge Entities für Multi-Hop Reasoning.
        
        Bridge Entities sind Entities, die als Zwischenschritte
        für die Graph-Traversierung benötigt werden.
        
        Gemäß Masterarbeit:
        "Dies ermöglicht die Identifikation von Bridge Entities
        (z.B. Subjekt-Objekt-Beziehungen in verschachtelten Sätzen),
        die als notwendige Zwischenschritte (Hops) für die
        Graph-Traversierung dienen."
        
        Args:
            query: Die Original-Query
            entities: Bereits extrahierte Entities
            
        Returns:
            Entities mit is_bridge Flag aktualisiert
        """
        if not self.config.enable_bridge_detection:
            return entities
        
        if not SPACY_AVAILABLE or NLP is None or len(entities) < 2:
            return entities
        
        doc = NLP(query)
        
        # Analysiere Dependency-Struktur für Bridge Detection
        # Bridge Entities sind typischerweise:
        # 1. Objekte von Präpositionen (pobj)
        # 2. Possessive Modifier (poss)
        # 3. Relative Clause Heads
        
        bridge_candidates = set()
        
        for token in doc:
            # Präpositionale Objekte in verschachtelten Strukturen
            if token.dep_ == "pobj" and token.head.dep_ == "prep":
                # Prüfe ob übergeordnetes Element auch ein Nomen ist
                if token.head.head.pos_ in ["NOUN", "PROPN"]:
                    bridge_candidates.add(token.text.lower())
            
            # Possessive in Ketten (z.B. "John's sister's husband")
            if token.dep_ == "poss":
                bridge_candidates.add(token.text.lower())
            
            # Relative Clause Subjects
            if token.dep_ == "relcl":
                for child in token.children:
                    if child.dep_ == "nsubj":
                        bridge_candidates.add(child.text.lower())
        
        # Markiere Entities als Bridge wenn sie in bridge_candidates sind
        # und nicht die Haupt-Entity sind (erste oder letzte)
        for i, entity in enumerate(entities):
            if entity.text.lower() in bridge_candidates:
                # Erste und letzte Entity sind typischerweise keine Bridges
                if 0 < i < len(entities) - 1:
                    entity.is_bridge = True
        
        return entities
    
    def _estimate_confidence(self, ent) -> float:
        """
        Schätze Konfidenz für SpaCy Entity.
        
        SpaCy liefert keine native Konfidenz, daher schätzen wir
        basierend auf Entity-Typ und Kontext.
        """
        # Basis-Konfidenz nach Entity-Typ
        type_confidence = {
            "PERSON": 0.9,
            "ORG": 0.85,
            "GPE": 0.9,
            "LOC": 0.85,
            "DATE": 0.95,
            "EVENT": 0.8,
            "WORK_OF_ART": 0.75,
        }
        
        base = type_confidence.get(ent.label_, 0.7)
        
        # Bonus für längere Entities (weniger ambig)
        length_bonus = min(0.1, len(ent.text.split()) * 0.03)
        
        return min(1.0, base + length_bonus)
    
    def _is_stopword(self, text: str) -> bool:
        """Prüfe ob Text ein Stopword ist."""
        stopwords = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'however', 'therefore', 'furthermore', 'moreover',
            'although', 'because', 'since', 'while', 'when',
            'what', 'which', 'who', 'whom', 'whose', 'where',
            'how', 'why', 'if', 'then', 'else', 'but', 'and', 'or',
        }
        return text.lower() in stopwords


# =============================================================================
# RETRIEVAL PLAN GENERATOR
# =============================================================================

class PlanGenerator:
    """
    Generator für strukturierte Retrieval-Pläne.
    
    Erzeugt basierend auf Query-Klassifikation und Entities
    einen detaillierten Plan für den Navigator (S_N).
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialisiere den Plan Generator.
        
        Args:
            config: Planner-Konfiguration
        """
        self.config = config or PlannerConfig()
        self.logger = logging.getLogger(__name__)
    
    def generate(
        self,
        query: str,
        query_type: QueryType,
        confidence: float,
        entities: List[EntityInfo]
    ) -> RetrievalPlan:
        """
        Generiere einen Retrieval-Plan.
        
        Args:
            query: Original-Query
            query_type: Klassifizierter Query-Typ
            confidence: Klassifikations-Konfidenz
            entities: Extrahierte Entities
            
        Returns:
            Vollständiger RetrievalPlan
        """
        # ─────────────────────────────────────────────────────────────────────
        # SCHRITT 1: Bestimme Retrieval-Strategie
        # ─────────────────────────────────────────────────────────────────────
        
        strategy = self._determine_strategy(query_type, entities)
        
        # ─────────────────────────────────────────────────────────────────────
        # SCHRITT 2: Generiere Hop-Sequenz und Sub-Queries
        # ─────────────────────────────────────────────────────────────────────
        
        hop_sequence, sub_queries = self._generate_hops(
            query, query_type, entities
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # SCHRITT 3: Extrahiere Constraints
        # ─────────────────────────────────────────────────────────────────────
        
        constraints = self._extract_constraints(query, query_type, entities)
        
        # ─────────────────────────────────────────────────────────────────────
        # SCHRITT 4: Erstelle Plan
        # ─────────────────────────────────────────────────────────────────────
        
        plan = RetrievalPlan(
            original_query=query,
            query_type=query_type,
            strategy=strategy,
            entities=entities,
            hop_sequence=hop_sequence,
            sub_queries=sub_queries,
            constraints=constraints,
            estimated_hops=len(hop_sequence),
            confidence=confidence,
            metadata={
                "entity_count": len(entities),
                "bridge_count": sum(1 for e in entities if e.is_bridge),
                "spacy_available": SPACY_AVAILABLE,
            }
        )
        
        self.logger.info(
            f"Plan generiert: type={query_type.value}, "
            f"strategy={strategy.value}, "
            f"hops={len(hop_sequence)}, "
            f"sub_queries={len(sub_queries)}"
        )
        
        return plan
    
    def _determine_strategy(
        self, 
        query_type: QueryType, 
        entities: List[EntityInfo]
    ) -> RetrievalStrategy:
        """
        Bestimme die optimale Retrieval-Strategie.
        
        Gemäß Masterarbeit Abschnitt 3.2:
        - VECTOR_ONLY: Für einfache Single-Hop Queries
        - GRAPH_ONLY: Wenn explizite Relationen benötigt werden
        - HYBRID: Für komplexe Multi-Hop und Comparison Queries
        """
        # Simple Queries → Vector-Only (schneller)
        if query_type == QueryType.SINGLE_HOP and len(entities) <= 1:
            return RetrievalStrategy.VECTOR_ONLY
        
        # Multi-Hop mit Bridge Entities → Hybrid erforderlich
        if query_type == QueryType.MULTI_HOP:
            has_bridges = any(e.is_bridge for e in entities)
            if has_bridges:
                return RetrievalStrategy.HYBRID
            return RetrievalStrategy.HYBRID
        
        # Comparison und Intersection → Hybrid
        if query_type in [QueryType.COMPARISON, QueryType.INTERSECTION]:
            return RetrievalStrategy.HYBRID
        
        # Temporal und Aggregate → Graph kann helfen
        if query_type in [QueryType.TEMPORAL, QueryType.AGGREGATE]:
            return RetrievalStrategy.HYBRID
        
        # Default
        return self.config.default_strategy
    
    def _generate_hops(
        self,
        query: str,
        query_type: QueryType,
        entities: List[EntityInfo]
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Generiere Hop-Sequenz und Sub-Queries.
        
        Erstellt eine geordnete Sequenz von Retrieval-Schritten
        mit Abhängigkeiten für Multi-Hop Reasoning.
        """
        hop_sequence = []
        sub_queries = []
        
        if query_type == QueryType.SINGLE_HOP:
            # Ein Schritt, keine Abhängigkeiten
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False
            ))
            sub_queries = [query]
        
        elif query_type == QueryType.MULTI_HOP:
            # Sequentielle Schritte mit Abhängigkeiten
            hop_sequence, sub_queries = self._decompose_multi_hop(query, entities)
        
        elif query_type == QueryType.COMPARISON:
            # Parallele Schritte für Entities, dann Vergleich
            hop_sequence, sub_queries = self._decompose_comparison(query, entities)
        
        elif query_type == QueryType.INTERSECTION:
            # Ähnlich wie Comparison
            hop_sequence, sub_queries = self._decompose_intersection(query, entities)
        
        elif query_type == QueryType.TEMPORAL:
            # Temporal-spezifische Dekomposition
            hop_sequence, sub_queries = self._decompose_temporal(query, entities)
        
        elif query_type == QueryType.AGGREGATE:
            # Aggregate-Dekomposition
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False
            ))
            sub_queries = [query]
        
        return hop_sequence, sub_queries
    
    def _decompose_multi_hop(
        self, 
        query: str, 
        entities: List[EntityInfo]
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Dekomponiere Multi-Hop Query in Schritte.
        
        Strategie:
        1. Identifiziere abhängige Teile der Query
        2. Erstelle Schritte für Bridge Entities zuerst
        3. Finale Query verwendet Ergebnisse der Bridges
        """
        hop_sequence = []
        sub_queries = []
        
        # Versuche Query an "that/which/who/where" zu splitten
        split_patterns = [
            r"\s+(that|which|who)\s+",
            r"\s+of\s+the\s+",
        ]
        
        parts = [query]
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                split_result = re.split(pattern, part, maxsplit=1, flags=re.IGNORECASE)
                new_parts.extend(split_result)
            parts = [p.strip() for p in new_parts if p.strip() and len(p.strip()) > 5]
        
        if len(parts) > 1:
            # Erstelle abhängige Schritte
            for i, part in enumerate(reversed(parts)):
                # Abhängigkeits-Liste: Alle vorherigen Schritte
                depends = list(range(i)) if i > 0 else []
                
                # Finde Entities für diesen Teil
                part_entities = [
                    e.text for e in entities 
                    if e.text.lower() in part.lower()
                ]
                
                sub_query = self._form_sub_query(part, i == len(parts) - 1)
                
                hop_sequence.append(HopStep(
                    step_id=i,
                    sub_query=sub_query,
                    target_entities=part_entities,
                    depends_on=depends,
                    is_bridge=(i < len(parts) - 1)
                ))
                sub_queries.append(sub_query)
        else:
            # Konnte nicht splitten → Single Hop
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False
            ))
            sub_queries = [query]
        
        return hop_sequence, sub_queries
    
    def _decompose_comparison(
        self, 
        query: str, 
        entities: List[EntityInfo]
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Dekomponiere Comparison Query.
        
        Strategie:
        1. Parallele Retrieval für jede Entity
        2. Finaler Vergleichsschritt
        """
        hop_sequence = []
        sub_queries = []
        
        # Schritt für jede Entity (können parallel laufen)
        for i, entity in enumerate(entities[:2]):  # Max 2 für Comparison
            sub_query = f"What is {entity.text}?"
            hop_sequence.append(HopStep(
                step_id=i,
                sub_query=sub_query,
                target_entities=[entity.text],
                depends_on=[],  # Keine Abhängigkeiten → parallel
                is_bridge=True
            ))
            sub_queries.append(sub_query)
        
        # Finaler Vergleichsschritt
        if len(entities) >= 2:
            hop_sequence.append(HopStep(
                step_id=len(entities[:2]),
                sub_query=query,
                target_entities=[e.text for e in entities[:2]],
                depends_on=list(range(len(entities[:2]))),  # Abhängig von allen vorherigen
                is_bridge=False
            ))
            sub_queries.append(query)
        
        return hop_sequence, sub_queries
    
    def _decompose_intersection(
        self, 
        query: str, 
        entities: List[EntityInfo]
    ) -> Tuple[List[HopStep], List[str]]:
        """Dekomponiere Intersection Query (ähnlich wie Comparison)."""
        return self._decompose_comparison(query, entities)
    
    def _decompose_temporal(
        self, 
        query: str, 
        entities: List[EntityInfo]
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Dekomponiere Temporal Query.
        
        Bei temporalen Queries wird die Zeit-Komponente
        als Constraint behandelt, nicht als separater Hop.
        """
        hop_sequence = []
        sub_queries = []
        
        # Temporal Queries sind oft Single-Hop mit Constraint
        hop_sequence.append(HopStep(
            step_id=0,
            sub_query=query,
            target_entities=[e.text for e in entities],
            depends_on=[],
            is_bridge=False
        ))
        sub_queries = [query]
        
        return hop_sequence, sub_queries
    
    def _form_sub_query(self, part: str, is_final: bool) -> str:
        """Forme einen Teil zu einer vollständigen Sub-Query."""
        part = part.strip()
        
        # Entferne führende Konjunktionen
        part = re.sub(r"^(and|or|but|that|which|who|where)\s+", "", part, flags=re.IGNORECASE)
        
        # Stelle sicher, dass es eine Frage ist
        if not part.endswith("?"):
            # Füge Fragewort hinzu wenn nötig
            if not re.match(r"^(what|who|where|when|why|how|which|is|are|was|were|did|does|do)\b", part, re.IGNORECASE):
                part = f"What is {part}?"
            else:
                part = f"{part}?"
        
        return part
    
    def _extract_constraints(
        self,
        query: str,
        query_type: QueryType,
        entities: List[EntityInfo]
    ) -> Dict[str, Any]:
        """
        Extrahiere Constraints aus der Query.
        
        Constraints sind zusätzliche Bedingungen wie:
        - Temporal: Zeiträume, Datumsangaben
        - Comparison: Vergleichsrichtung (größer/kleiner)
        - Filter: Einschränkungen auf bestimmte Eigenschaften
        """
        constraints = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # TEMPORAL CONSTRAINTS
        # ─────────────────────────────────────────────────────────────────────
        
        if query_type == QueryType.TEMPORAL or self.config.enable_temporal_parsing:
            # Extrahiere Jahreszahlen
            years = re.findall(r"\b(1\d{3}|20\d{2})\b", query)
            if years:
                constraints["years"] = years
            
            # Extrahiere relative Zeitbegriffe
            temporal_terms = re.findall(
                r"\b(before|after|during|since|until|recent|latest|first|last)\b",
                query,
                re.IGNORECASE
            )
            if temporal_terms:
                constraints["temporal_relation"] = temporal_terms[0].lower()
        
        # ─────────────────────────────────────────────────────────────────────
        # COMPARISON CONSTRAINTS
        # ─────────────────────────────────────────────────────────────────────
        
        if query_type == QueryType.COMPARISON:
            # Extrahiere Vergleichsrichtung
            if re.search(r"\b(older|bigger|larger|more|higher|greater)\b", query, re.IGNORECASE):
                constraints["comparison_direction"] = "greater"
            elif re.search(r"\b(younger|smaller|less|lower|fewer)\b", query, re.IGNORECASE):
                constraints["comparison_direction"] = "less"
            
            # Extrahiere Vergleichsattribut
            attr_match = re.search(
                r"\b(older|younger|taller|shorter|bigger|smaller|richer|poorer)\b",
                query,
                re.IGNORECASE
            )
            if attr_match:
                constraints["comparison_attribute"] = attr_match.group(1).lower()
        
        return constraints


# =============================================================================
# MAIN PLANNER CLASS
# =============================================================================

class Planner:
    """
    S_P: Regelbasierter Query Planner.
    
    Hauptklasse, die Query-Klassifikation, Entity-Extraktion und
    Plan-Generierung orchestriert.
    
    Verwendung:
        planner = Planner()
        plan = planner.plan("Who directed the movie with Tom Hanks?")
        sub_queries = planner.decompose_query("Is Berlin older than Munich?")
    """
    
    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        # Legacy-Parameter für API-Kompatibilität
        model_name: str = None,
        base_url: str = None,
        **kwargs
    ):
        """
        Initialisiere den Planner.
        
        Args:
            config: PlannerConfig (optional)
            model_name: Ignoriert (API-Kompatibilität)
            base_url: Ignoriert (API-Kompatibilität)
        """
        self.config = config or PlannerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialisiere Komponenten
        self.classifier = QueryClassifier(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.plan_generator = PlanGenerator(self.config)
        
        self.logger.info(
            f"Planner initialisiert: SpaCy={'verfügbar' if SPACY_AVAILABLE else 'nicht verfügbar'}"
        )
    
    def plan(self, query: str) -> RetrievalPlan:
        """
        Generiere vollständigen Retrieval-Plan für eine Query.
        
        Dies ist die Hauptmethode, die den gesamten Planner-Workflow
        ausführt und einen strukturierten Plan zurückgibt.
        
        Args:
            query: Die User-Query
            
        Returns:
            RetrievalPlan mit Strategie, Entities, Hops und Constraints
        """
        start_time = time.perf_counter()
        
        query = query.strip()
        if not query:
            return self._empty_plan(query)
        
        # ─────────────────────────────────────────────────────────────────────
        # SCHRITT 1: Query-Klassifikation
        # ─────────────────────────────────────────────────────────────────────
        
        query_type, confidence = self.classifier.classify(query)
        
        self.logger.debug(f"Query klassifiziert: {query_type.value} (conf={confidence:.2f})")
        
        # ─────────────────────────────────────────────────────────────────────
        # SCHRITT 2: Entity-Extraktion
        # ─────────────────────────────────────────────────────────────────────
        
        entities = self.entity_extractor.extract(query)
        
        # Bridge Detection für Multi-Hop
        if query_type == QueryType.MULTI_HOP:
            entities = self.entity_extractor.detect_bridge_entities(query, entities)
        
        self.logger.debug(f"Entities extrahiert: {len(entities)}")
        
        # ─────────────────────────────────────────────────────────────────────
        # SCHRITT 3: Plan-Generierung
        # ─────────────────────────────────────────────────────────────────────
        
        plan = self.plan_generator.generate(
            query=query,
            query_type=query_type,
            confidence=confidence,
            entities=entities
        )
        
        # Timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        plan.metadata["planning_time_ms"] = elapsed_ms
        
        self.logger.info(
            f"Plan generiert in {elapsed_ms:.1f}ms: "
            f"type={query_type.value}, "
            f"entities={len(entities)}, "
            f"hops={plan.estimated_hops}"
        )
        
        return plan
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Dekomponiere Query in Sub-Queries.
        
        Vereinfachte Methode für Kompatibilität mit dem Agent.
        Gibt nur die Sub-Query-Liste zurück.
        
        Args:
            query: Die User-Query
            
        Returns:
            Liste von Sub-Queries für Retrieval
        """
        plan = self.plan(query)
        return plan.sub_queries
    
    def get_plan(self, query: str) -> RetrievalPlan:
        """Alias für plan() - API-Kompatibilität."""
        return self.plan(query)
    
    def get_query_type(self, query: str) -> str:
        """
        Bestimme Query-Typ als String.
        
        Args:
            query: Die User-Query
            
        Returns:
            Query-Typ als String (z.B. "multi_hop")
        """
        plan = self.plan(query)
        return plan.query_type.value
    
    def _empty_plan(self, query: str) -> RetrievalPlan:
        """Erstelle leeren Plan für ungültige Queries."""
        return RetrievalPlan(
            original_query=query,
            query_type=QueryType.SINGLE_HOP,
            strategy=RetrievalStrategy.VECTOR_ONLY,
            entities=[],
            hop_sequence=[],
            sub_queries=[query] if query else [],
            constraints={},
            estimated_hops=1,
            confidence=0.0,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_planner(
    model_name: str = None,  # Ignoriert - API-Kompatibilität
    base_url: str = None,    # Ignoriert - API-Kompatibilität
    **kwargs
) -> Planner:
    """
    Factory-Funktion für Planner.
    
    Erstellt einen konfigurierten Planner mit optionalen Parametern.
    
    Args:
        model_name: Ignoriert (für API-Kompatibilität mit LLM-basiertem Planner)
        base_url: Ignoriert (für API-Kompatibilität)
        **kwargs: Weitere PlannerConfig-Parameter
        
    Returns:
        Konfigurierte Planner-Instanz
    """
    config = PlannerConfig(**kwargs) if kwargs else None
    return Planner(config=config)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Test-Queries für verschiedene Typen
    test_queries = [
        # Single-Hop
        ("What is the capital of France?", QueryType.SINGLE_HOP),
        
        # Multi-Hop
        ("Who is the director of the film that stars Tom Hanks?", QueryType.MULTI_HOP),
        ("What is the capital of the country where Einstein was born?", QueryType.MULTI_HOP),
        
        # Comparison
        ("Is Berlin older than Munich?", QueryType.COMPARISON),
        ("Which is taller, the Eiffel Tower or Big Ben?", QueryType.COMPARISON),
        
        # Temporal
        ("What happened after World War 2?", QueryType.TEMPORAL),
        ("Who was president in 1990?", QueryType.TEMPORAL),
        
        # Intersection
        ("Which movies star both Brad Pitt and Leonardo DiCaprio?", QueryType.INTERSECTION),
    ]
    
    print("=" * 70)
    print("S_P: REGELBASIERTER QUERY PLANNER TEST")
    print(f"SpaCy verfügbar: {SPACY_AVAILABLE}")
    print("=" * 70)
    
    planner = Planner()
    
    total_time = 0
    correct = 0
    
    for query, expected_type in test_queries:
        plan = planner.plan(query)
        elapsed = plan.metadata.get("planning_time_ms", 0)
        total_time += elapsed
        
        is_correct = plan.query_type == expected_type
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        
        print(f"\n{status} Query: {query}")
        print(f"   Erwartet: {expected_type.value}, Erkannt: {plan.query_type.value}")
        print(f"   Strategie: {plan.strategy.value}")
        print(f"   Entities: {[e.text for e in plan.entities]}")
        print(f"   Sub-Queries: {plan.sub_queries}")
        print(f"   Hops: {plan.estimated_hops}")
        print(f"   Zeit: {elapsed:.1f}ms")
    
    print("\n" + "=" * 70)
    print(f"Genauigkeit: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.1f}%)")
    print(f"Durchschnittliche Planungszeit: {total_time/len(test_queries):.1f}ms")
    print("=" * 70)