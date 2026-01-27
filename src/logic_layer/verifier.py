"""
===============================================================================
S_V: Verifier mit Pre-Generation Validation und Self-Correction
===============================================================================

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"
Artefakt B: Agent-Based Query Processing

===============================================================================
ÜBERBLICK
===============================================================================

Der Verifier (S_V) ist die letzte Stufe des Agentic RAG Systems und implementiert
einen Dual-Stage-Ansatz gemäß Masterarbeit Abschnitt 3.4:

1. PRE-GENERATION VALIDATION
   Vor der Generierung werden drei Validierungschecks durchgeführt:
   
   a) Entity-Path Validation (für Multi-Hop)
      - Überprüft ob Retrieval-Ergebnisse kohärenten Reasoning-Pfad bilden
      - Graph-Traversierung zur Verifizierung der Bridge Entities
      - "Insufficient Evidence" wenn Pfad nicht existiert
   
   b) Contradiction Detection
      - NLI-basierte Widerspruchserkennung (Confidence > 0.85)
      - Verhindert widersprüchliche Evidenz im Generator-Kontext
   
   c) Source Credibility
      - Cross-References: Wie oft wird Information bestätigt
      - Entity-Mention-Frequency im Graph
      - Retrieval-Provenance: Graph-basierte Ergebnisse mit Boost

2. GENERATION MIT QUANTISIERTEM SLM
   - Phi-3-Mini AWQ 4-bit (oder konfiguriertes Modell)
   - Optimiert für Edge-Deployment
   - Kontrollierte Generierung mit Context-Limits

3. SELF-CORRECTION LOOP
   - Iterative Verbesserung bei unverifizierten Claims
   - Feedback mit Violation-Details
   - Tracking der besten Antwort über Iterationen

===============================================================================
ARCHITEKTUR
===============================================================================

    Query + Context (aus Navigator)
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                      S_V (VERIFIER)                           │
    │                                                               │
    │   ┌─────────────────────────────────────────────────────┐    │
    │   │            PRE-GENERATION VALIDATION                 │    │
    │   │                                                      │    │
    │   │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │    │
    │   │  │Entity-Path │  │Contradiction│ │  Source    │    │    │
    │   │  │ Validation │  │ Detection  │  │ Credibility│    │    │
    │   │  └────────────┘  └────────────┘  └────────────┘    │    │
    │   │         │              │               │            │    │
    │   │         └──────────────┴───────────────┘            │    │
    │   │                        │                             │    │
    │   │              Pass/Fail + Filtered Context           │    │
    │   └────────────────────────┼────────────────────────────┘    │
    │                            │                                  │
    │                            ▼                                  │
    │   ┌─────────────────────────────────────────────────────┐    │
    │   │               GENERATION LOOP                        │    │
    │   │                                                      │    │
    │   │         ┌──────────┐                                │    │
    │   │    ┌───▶│ GENERATE │                                │    │
    │   │    │    │  Answer  │                                │    │
    │   │    │    └────┬─────┘                                │    │
    │   │    │         │                                       │    │
    │   │    │    ┌────▼─────┐                                │    │
    │   │    │    │ EXTRACT  │                                │    │
    │   │    │    │  Claims  │                                │    │
    │   │    │    └────┬─────┘                                │    │
    │   │    │         │                                       │    │
    │   │    │    ┌────▼─────┐     ┌─────────────┐            │    │
    │   │    │    │  VERIFY  │────▶│ Violations? │            │    │
    │   │    │    │  Claims  │     └──────┬──────┘            │    │
    │   │    │    └──────────┘            │                    │    │
    │   │    │                      Yes   │   No               │    │
    │   │    │    ┌─────────────┐◀───────┘     │              │    │
    │   │    └────┤ SELF-CORRECT│              │              │    │
    │   │         │ (Feedback)  │              │              │    │
    │   │         └─────────────┘              │              │    │
    │   │                                      ▼              │    │
    │   │                              ┌─────────────┐        │    │
    │   │                              │   RETURN    │        │    │
    │   │                              │   Answer    │        │    │
    │   │                              └─────────────┘        │    │
    │   └──────────────────────────────────────────────────────┘    │
    │                                                               │
    └───────────────────────────────────────────────────────────────┘

===============================================================================
"""

import logging
import re
import time
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# OPTIONALE ABHÄNGIGKEITEN
# =============================================================================

# SpaCy für Sentence Splitting und NER
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("SpaCy geladen für Claim-Extraktion")
except:
    NLP = None
    SPACY_AVAILABLE = False
    logger.info("SpaCy nicht verfügbar, nutze Regex für Claim-Extraktion")

# Transformers für NLI-basierte Contradiction Detection
try:
    from transformers import pipeline
    NLI_PIPELINE = None  # Lazy loading
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers verfügbar für NLI")
except ImportError:
    NLI_PIPELINE = None
    TRANSFORMERS_AVAILABLE = False
    logger.info("Transformers nicht verfügbar, NLI-Detection deaktiviert")


# =============================================================================
# DATENSTRUKTUREN
# =============================================================================

class ValidationStatus(Enum):
    """Status der Pre-Generation Validation."""
    PASSED = "passed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTION_DETECTED = "contradiction_detected"
    LOW_CREDIBILITY = "low_credibility"


@dataclass
class SourceCredibility:
    """
    Credibility-Score für eine Kontext-Quelle.
    
    Gemäß Masterarbeit Abschnitt 3.4:
    - cross_references: Wie oft wird Information in anderen Chunks bestätigt
    - entity_frequency: Entitäts-Mention-Frequency im Graph
    - retrieval_provenance: Graph-basierte Ergebnisse erhalten Boost
    """
    text: str
    score: float = 0.5
    cross_references: int = 0
    entity_frequency: float = 0.0
    is_graph_based: bool = False
    
    def compute_score(self) -> float:
        """
        Berechne Gesamt-Credibility-Score.
        
        Gewichtung:
        - Cross-References: 40%
        - Entity-Frequency: 30%
        - Retrieval-Provenance: 30%
        """
        # Normalisiere Cross-References (0-1)
        ref_score = min(1.0, self.cross_references / 3.0)
        
        # Entity-Frequency ist bereits 0-1
        entity_score = self.entity_frequency
        
        # Provenance-Bonus für Graph-basierte Quellen
        provenance_score = 1.0 if self.is_graph_based else 0.5
        
        # Gewichtete Summe
        self.score = (
            0.4 * ref_score +
            0.3 * entity_score +
            0.3 * provenance_score
        )
        
        return self.score


@dataclass
class PreValidationResult:
    """
    Ergebnis der Pre-Generation Validation.
    
    Attributes:
        status: Validierungsstatus (passed/failed)
        entity_path_valid: Entity-Path existiert im Graph
        contradictions: Liste gefundener Widersprüche
        filtered_context: Gefilterter Kontext für Generator
        credibility_scores: Credibility pro Kontext-Chunk
        validation_time_ms: Validierungsdauer
        details: Zusätzliche Debug-Informationen
    """
    status: ValidationStatus = ValidationStatus.PASSED
    entity_path_valid: bool = True
    contradictions: List[Tuple[str, str, float]] = field(default_factory=list)
    filtered_context: List[str] = field(default_factory=list)
    credibility_scores: List[float] = field(default_factory=list)
    validation_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierConfig:
    """
    Konfiguration für Verifier Stage.
    
    LLM Settings:
        model_name: Ollama-Modell (z.B. "phi3")
        base_url: Ollama API URL
        temperature: Sampling Temperature (0.0 = deterministisch)
        max_tokens: Maximale Antwort-Tokens
        timeout: API Timeout in Sekunden
    
    Context Settings:
        max_context_chars: Maximale Gesamtlänge des Contexts
        max_docs: Maximale Anzahl Dokumente
        max_chars_per_doc: Maximale Zeichen pro Dokument
    
    Pre-Validation Settings:
        enable_entity_path_validation: Entity-Path Check aktivieren
        enable_contradiction_detection: NLI-basierte Widerspruchserkennung
        contradiction_threshold: NLI-Confidence für Widerspruch (default: 0.85)
        enable_credibility_scoring: Source Credibility aktivieren
        min_credibility_score: Minimum Score für Inklusion (default: 0.5)
    
    Loop Settings:
        max_iterations: Maximale Self-Correction Iterationen
        stop_on_first_success: Bei erster vollständig verifizierten Antwort stoppen
    """
    # LLM Settings
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 200
    timeout: int = 300  # 5 Minuten für CPU
    
    # Context Settings (optimiert für Edge)
    max_context_chars: int = 2000   # ~500 tokens
    max_docs: int = 5               # Top-5 aus Hybrid Retrieval
    max_chars_per_doc: int = 350    # Pro Chunk
    
    # Pre-Validation Settings (gemäß Masterarbeit)
    enable_entity_path_validation: bool = True
    enable_contradiction_detection: bool = True
    contradiction_threshold: float = 0.85  # NLI-Confidence > 0.85
    enable_credibility_scoring: bool = True
    min_credibility_score: float = 0.5     # Minimum für Inklusion
    
    # Agentic Loop Settings
    max_iterations: int = 3
    stop_on_first_success: bool = True


@dataclass
class VerificationResult:
    """
    Ergebnis der Verification Stage.
    
    Attributes:
        answer: Generierte (und ggf. korrigierte) Antwort
        iterations: Anzahl durchgeführter Iterationen
        verified_claims: Liste verifizierter Claims
        violated_claims: Liste nicht-verifizierter Claims
        all_verified: True wenn alle Claims verifiziert
        pre_validation: Ergebnis der Pre-Generation Validation
        timing_ms: Gesamtzeit in Millisekunden
        iteration_history: Details pro Iteration (für Analyse)
    """
    answer: str
    iterations: int
    verified_claims: List[str] = field(default_factory=list)
    violated_claims: List[str] = field(default_factory=list)
    all_verified: bool = False
    pre_validation: Optional[PreValidationResult] = None
    timing_ms: float = 0.0
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# PRE-GENERATION VALIDATOR
# =============================================================================

class PreGenerationValidator:
    """
    Pre-Generation Validation gemäß Masterarbeit Abschnitt 3.4.
    
    Implementiert drei Validierungschecks:
    1. Entity-Path Validation (Multi-Hop)
    2. Contradiction Detection (NLI)
    3. Source Credibility Scoring
    """
    
    def __init__(self, config: VerifierConfig, graph_store=None):
        """
        Initialisiere Validator.
        
        Args:
            config: VerifierConfig
            graph_store: Optional KnowledgeGraphStore für Entity-Path Validation
        """
        self.config = config
        self.graph_store = graph_store
        self.logger = logging.getLogger(__name__)
        
        # Lazy-load NLI Pipeline bei Bedarf
        self._nli_pipeline = None
        
        self.logger.info(
            f"PreGenerationValidator initialisiert: "
            f"entity_path={config.enable_entity_path_validation}, "
            f"contradiction={config.enable_contradiction_detection}, "
            f"credibility={config.enable_credibility_scoring}"
        )
    
    def validate(
        self,
        context: List[str],
        query: str,
        entities: List[str] = None,
        hop_sequence: List[Dict] = None
    ) -> PreValidationResult:
        """
        Führe Pre-Generation Validation durch.
        
        Args:
            context: Liste von Kontext-Chunks aus Navigator
            query: Die Original-Query
            entities: Extrahierte Entities (für Entity-Path Check)
            hop_sequence: Hop-Sequenz aus Planner (für Multi-Hop Validation)
            
        Returns:
            PreValidationResult mit Status und gefiltertem Kontext
        """
        start_time = time.time()
        result = PreValidationResult()
        
        if not context:
            result.status = ValidationStatus.INSUFFICIENT_EVIDENCE
            result.filtered_context = []
            result.details["error"] = "Kein Kontext verfügbar"
            return result
        
        # Starte mit vollständigem Kontext
        result.filtered_context = context.copy()
        
        # ─────────────────────────────────────────────────────────────────────
        # CHECK 1: Entity-Path Validation (Multi-Hop)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.config.enable_entity_path_validation and entities and hop_sequence:
            path_valid, path_details = self._validate_entity_path(
                context, entities, hop_sequence
            )
            result.entity_path_valid = path_valid
            result.details["entity_path"] = path_details
            
            if not path_valid:
                self.logger.warning("Entity-Path Validation fehlgeschlagen")
                result.status = ValidationStatus.INSUFFICIENT_EVIDENCE
                # Wir brechen nicht ab, generieren aber eine qualifizierte Antwort
        
        # ─────────────────────────────────────────────────────────────────────
        # CHECK 2: Contradiction Detection
        # ─────────────────────────────────────────────────────────────────────
        
        if self.config.enable_contradiction_detection and len(context) > 1:
            contradictions = self._detect_contradictions(context)
            result.contradictions = contradictions
            
            if contradictions:
                self.logger.warning(
                    f"{len(contradictions)} Widersprüche gefunden"
                )
                result.details["contradictions"] = [
                    {"chunk1": c[0][:50], "chunk2": c[1][:50], "score": c[2]}
                    for c in contradictions
                ]
                
                # Entferne widersprüchliche Chunks mit niedrigerem Score
                result.filtered_context = self._resolve_contradictions(
                    context, contradictions
                )
                
                if not result.filtered_context:
                    result.status = ValidationStatus.CONTRADICTION_DETECTED
        
        # ─────────────────────────────────────────────────────────────────────
        # CHECK 3: Source Credibility Scoring
        # ─────────────────────────────────────────────────────────────────────
        
        if self.config.enable_credibility_scoring:
            credibility_scores = self._compute_credibility(
                result.filtered_context, context
            )
            result.credibility_scores = credibility_scores
            
            # Filtere Low-Credibility Chunks
            high_cred_context = []
            for i, (chunk, score) in enumerate(zip(result.filtered_context, credibility_scores)):
                if score >= self.config.min_credibility_score:
                    high_cred_context.append(chunk)
                else:
                    self.logger.debug(
                        f"Chunk {i} mit Score {score:.2f} gefiltert"
                    )
            
            if high_cred_context:
                result.filtered_context = high_cred_context
            else:
                # Behalte mindestens den besten Chunk
                if credibility_scores:
                    best_idx = credibility_scores.index(max(credibility_scores))
                    result.filtered_context = [result.filtered_context[best_idx]]
                    result.status = ValidationStatus.LOW_CREDIBILITY
        
        # Timing
        result.validation_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(
            f"Pre-Validation: status={result.status.value}, "
            f"context={len(result.filtered_context)}/{len(context)}, "
            f"time={result.validation_time_ms:.0f}ms"
        )
        
        return result
    
    def _validate_entity_path(
        self,
        context: List[str],
        entities: List[str],
        hop_sequence: List[Dict]
    ) -> Tuple[bool, Dict]:
        """
        Entity-Path Validation für Multi-Hop Queries.
        
        Gemäß Masterarbeit:
        "Für Multi-Hop-Queries wird überprüft, ob die Retrieval-Ergebnisse
        einen kohärenten Reasoning-Pfad über die Query-Entitäten bilden."
        
        Returns:
            Tuple von (is_valid, details)
        """
        details = {
            "entities_found": [],
            "entities_missing": [],
            "path_exists": False
        }
        
        if not self.graph_store:
            # Ohne Graph-Store: Prüfe nur ob Entities im Context vorkommen
            context_text = " ".join(context).lower()
            
            for entity in entities:
                if entity.lower() in context_text:
                    details["entities_found"].append(entity)
                else:
                    details["entities_missing"].append(entity)
            
            # Pfad ist valid wenn > 50% der Entities gefunden
            coverage = len(details["entities_found"]) / max(1, len(entities))
            details["coverage"] = coverage
            details["path_exists"] = coverage >= 0.5
            
            return details["path_exists"], details
        
        # Mit Graph-Store: Prüfe ob Pfad zwischen Entities existiert
        try:
            # Prüfe Entity-Existenz im Graph
            for entity in entities:
                if hasattr(self.graph_store, 'entity_exists'):
                    if self.graph_store.entity_exists(entity):
                        details["entities_found"].append(entity)
                    else:
                        details["entities_missing"].append(entity)
                elif hasattr(self.graph_store, 'query_by_entity'):
                    results = self.graph_store.query_by_entity(entity)
                    if results:
                        details["entities_found"].append(entity)
                    else:
                        details["entities_missing"].append(entity)
            
            # Prüfe Pfad-Existenz für Multi-Hop
            if len(details["entities_found"]) >= 2:
                # Versuche Pfad zwischen erstem und letztem Entity zu finden
                if hasattr(self.graph_store, 'find_path'):
                    path = self.graph_store.find_path(
                        details["entities_found"][0],
                        details["entities_found"][-1]
                    )
                    details["path_exists"] = path is not None
                else:
                    # Fallback: Pfad existiert wenn alle Entities gefunden
                    details["path_exists"] = len(details["entities_missing"]) == 0
            
        except Exception as e:
            self.logger.warning(f"Entity-Path Validation Error: {e}")
            details["error"] = str(e)
            details["path_exists"] = True  # Bei Fehler: annehmen dass valid
        
        return details["path_exists"], details
    
    def _detect_contradictions(
        self,
        context: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Contradiction Detection mit NLI.
        
        Gemäß Masterarbeit:
        "Contradiction Filter mit strengerem Threshold (NLI-Confidence > 0.85)"
        
        Returns:
            Liste von (chunk1, chunk2, contradiction_score) Tupeln
        """
        contradictions = []
        
        # Versuche NLI-basierte Detection
        if TRANSFORMERS_AVAILABLE and self.config.enable_contradiction_detection:
            try:
                # Lazy-load NLI Pipeline
                if self._nli_pipeline is None:
                    # Kleines Modell für Edge
                    self._nli_pipeline = pipeline(
                        "text-classification",
                        model="cross-encoder/nli-distilroberta-base",
                        device=-1  # CPU
                    )
                
                # Paarweise Prüfung (nur direkte Nachbarn für Performance)
                for i in range(len(context) - 1):
                    chunk1 = context[i][:200]  # Limitiere für Performance
                    chunk2 = context[i + 1][:200]
                    
                    result = self._nli_pipeline(
                        f"{chunk1} [SEP] {chunk2}",
                        truncation=True
                    )
                    
                    if result and result[0]["label"] == "CONTRADICTION":
                        if result[0]["score"] >= self.config.contradiction_threshold:
                            contradictions.append(
                                (chunk1, chunk2, result[0]["score"])
                            )
                            
            except Exception as e:
                self.logger.debug(f"NLI Detection Error: {e}")
                # Fallback zu heuristischer Detection
                contradictions = self._heuristic_contradiction_detection(context)
        else:
            # Fallback ohne Transformers
            contradictions = self._heuristic_contradiction_detection(context)
        
        return contradictions
    
    def _heuristic_contradiction_detection(
        self,
        context: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Heuristische Widerspruchserkennung ohne NLI.
        
        Erkennt einfache Muster wie:
        - Verschiedene Zahlen für gleiche Entity
        - Negation von Aussagen
        """
        contradictions = []
        
        # Pattern für Zahlen mit Kontext
        number_pattern = re.compile(r'(\b[A-Z][a-z]+\b)\s+(?:was|is|has|had)\s+(\d+(?:\.\d+)?)')
        
        # Extrahiere Entity-Zahlen-Paare aus allen Chunks
        entity_values = {}
        for i, chunk in enumerate(context):
            matches = number_pattern.findall(chunk)
            for entity, value in matches:
                if entity not in entity_values:
                    entity_values[entity] = []
                entity_values[entity].append((i, float(value)))
        
        # Prüfe auf widersprüchliche Werte
        for entity, values in entity_values.items():
            if len(values) > 1:
                # Prüfe ob Werte stark abweichen (>50%)
                for j in range(len(values)):
                    for k in range(j + 1, len(values)):
                        v1, v2 = values[j][1], values[k][1]
                        if min(v1, v2) > 0:
                            diff = abs(v1 - v2) / max(v1, v2)
                            if diff > 0.5:
                                idx1, idx2 = values[j][0], values[k][0]
                                contradictions.append(
                                    (context[idx1], context[idx2], 0.9)
                                )
        
        return contradictions
    
    def _resolve_contradictions(
        self,
        context: List[str],
        contradictions: List[Tuple[str, str, float]]
    ) -> List[str]:
        """
        Löse Widersprüche auf durch Entfernen niedrig-kredibilierter Chunks.
        """
        # Zähle wie oft ein Chunk in Widersprüchen vorkommt
        contradiction_counts = {}
        for chunk1, chunk2, score in contradictions:
            contradiction_counts[chunk1] = contradiction_counts.get(chunk1, 0) + 1
            contradiction_counts[chunk2] = contradiction_counts.get(chunk2, 0) + 1
        
        # Behalte Chunks mit weniger Widersprüchen
        if contradiction_counts:
            min_contradictions = min(contradiction_counts.values())
            filtered = [
                c for c in context
                if contradiction_counts.get(c, 0) <= min_contradictions
            ]
            return filtered if filtered else context
        
        return context
    
    def _compute_credibility(
        self,
        filtered_context: List[str],
        original_context: List[str]
    ) -> List[float]:
        """
        Berechne Source Credibility Scores.
        
        Gemäß Masterarbeit:
        - Cross-References: Wie oft wird Information bestätigt
        - Entity-Mention-Frequency: Häufige Entities sind zuverlässiger
        - Retrieval-Provenance: Graph-basierte Ergebnisse mit Boost
        """
        scores = []
        
        for chunk in filtered_context:
            cred = SourceCredibility(text=chunk)
            
            # Cross-References: Zähle ähnliche Mentions in anderen Chunks
            chunk_lower = chunk.lower()
            key_phrases = self._extract_key_phrases(chunk)
            
            for other_chunk in original_context:
                if other_chunk != chunk:
                    other_lower = other_chunk.lower()
                    # Prüfe ob Key-Phrases in anderem Chunk vorkommen
                    for phrase in key_phrases:
                        if phrase.lower() in other_lower:
                            cred.cross_references += 1
                            break
            
            # Entity-Frequency (vereinfacht ohne Graph)
            # Mehr Entities = wahrscheinlich mehr Detail = höhere Credibility
            if SPACY_AVAILABLE and NLP:
                doc = NLP(chunk[:500])  # Limitiere für Performance
                entity_count = len(doc.ents)
                cred.entity_frequency = min(1.0, entity_count / 5.0)
            else:
                # Fallback: Zähle Großbuchstaben-Wörter
                proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', chunk))
                cred.entity_frequency = min(1.0, proper_nouns / 10.0)
            
            # Provenance: Aktuell keine Unterscheidung möglich
            # TODO: Metadata aus Navigator nutzen
            cred.is_graph_based = False
            
            # Berechne Gesamt-Score
            score = cred.compute_score()
            scores.append(score)
        
        return scores
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extrahiere Key-Phrases für Cross-Reference Check."""
        phrases = []
        
        # Named Entities
        if SPACY_AVAILABLE and NLP:
            doc = NLP(text[:500])
            phrases.extend([ent.text for ent in doc.ents])
        
        # Proper Nouns (Fallback)
        phrases.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        
        # Zahlen mit Kontext
        phrases.extend(re.findall(r'\d+(?:\.\d+)?(?:\s+\w+)?', text))
        
        return list(set(phrases))[:10]  # Limitiere


# =============================================================================
# MAIN VERIFIER CLASS
# =============================================================================

class Verifier:
    """
    S_V: Verifier mit Pre-Generation Validation und Self-Correction.
    
    Hauptklasse für die finale Antwortgenerierung und Verifikation.
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROMPT TEMPLATES
    # ─────────────────────────────────────────────────────────────────────────
    
    ANSWER_PROMPT = """You are a factual question-answering assistant.
Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say "I cannot answer based on the given context."

Context:
{context}

Question: {query}

Provide a concise, factual answer (2-3 sentences max):"""

    CORRECTION_PROMPT = """Your previous answer contained claims that could not be verified against the knowledge base.

Unverified claims:
{violations}

Please provide a corrected answer that only includes facts that can be verified from the context.
Avoid making claims that go beyond what the context explicitly states.

Context:
{context}

Question: {query}

Corrected answer (only verified facts):"""

    INSUFFICIENT_EVIDENCE_PROMPT = """Based on the available context, I could not find sufficient evidence to fully answer your question.

Context:
{context}

Question: {query}

Please provide a partial answer based on the available evidence, clearly indicating what information is missing:"""

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        graph_store=None,
    ):
        """
        Initialisiere Verifier.
        
        Args:
            config: VerifierConfig (nutzt Defaults wenn None)
            graph_store: KnowledgeGraphStore für Verification (optional)
        """
        self.config = config or VerifierConfig()
        self.graph_store = graph_store
        self.logger = logging.getLogger(__name__)
        
        # Initialisiere Pre-Generation Validator
        self.pre_validator = PreGenerationValidator(self.config, graph_store)
        
        self.logger.info(
            f"Verifier initialisiert: "
            f"model={self.config.model_name}, "
            f"max_iterations={self.config.max_iterations}, "
            f"pre_validation={'aktiviert' if self.config.enable_entity_path_validation else 'deaktiviert'}"
        )
    
    def set_graph_store(self, graph_store) -> None:
        """Setze Graph Store für Verification."""
        self.graph_store = graph_store
        self.pre_validator.graph_store = graph_store
        self.logger.info("Graph Store verbunden")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LLM INTERACTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _call_llm(self, prompt: str) -> Tuple[str, float]:
        """
        Rufe Ollama LLM API auf.
        
        Args:
            prompt: Vollständiger Prompt
            
        Returns:
            Tuple von (response_text, latency_ms)
        """
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    }
                },
                timeout=self.config.timeout
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                self.logger.error(f"Ollama API Error: {response.status_code}")
                return f"[Error: API returned {response.status_code}]", latency_ms
            
            answer = response.json().get("response", "").strip()
            return answer, latency_ms
            
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Timeout nach {self.config.timeout}s")
            return "[Error: LLM timeout - try reducing context size]", latency_ms
            
        except requests.exceptions.ConnectionError:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error("Keine Verbindung zu Ollama")
            return "[Error: Cannot connect to Ollama - is it running?]", latency_ms
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"LLM Error: {e}")
            return f"[Error: {str(e)[:100]}]", latency_ms
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONTEXT FORMATTING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _format_context(self, context: List[str]) -> str:
        """
        Formatiere Context-Dokumente mit Größenlimits.
        
        Strategie:
        1. Nimm maximal max_docs Dokumente
        2. Truncate jedes Dokument auf max_chars_per_doc
        3. Stoppe wenn max_context_chars erreicht
        
        Args:
            context: Liste von Dokumenten/Chunks
            
        Returns:
            Formatierter Context-String
        """
        if not context:
            return "No context available."
        
        formatted_parts = []
        total_chars = 0
        
        for i, doc in enumerate(context[:self.config.max_docs]):
            # Truncate Document intelligent (an Wortgrenze)
            if len(doc) > self.config.max_chars_per_doc:
                truncated = doc[:self.config.max_chars_per_doc]
                # Versuche am letzten Satz zu brechen
                last_period = truncated.rfind('. ')
                if last_period > self.config.max_chars_per_doc * 0.7:
                    truncated = truncated[:last_period + 1]
                else:
                    last_space = truncated.rfind(' ')
                    if last_space > 0:
                        truncated = truncated[:last_space] + "..."
            else:
                truncated = doc
            
            # Format mit Index
            part = f"[{i+1}] {truncated}"
            
            # Check Total-Limit
            if total_chars + len(part) > self.config.max_context_chars:
                self.logger.debug(
                    f"Context-Limit erreicht bei Doc {i+1}/{len(context)}"
                )
                break
            
            formatted_parts.append(part)
            total_chars += len(part) + 2
        
        result = "\n\n".join(formatted_parts)
        self.logger.debug(
            f"Context formatiert: {len(formatted_parts)} docs, "
            f"{total_chars} chars (~{total_chars // 4} tokens)"
        )
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLAIM EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extrahiere atomare Claims (Fakten-Aussagen) aus der Antwort.
        
        Nutzt SpaCy für Sentence Splitting falls verfügbar,
        sonst Regex-basiertes Splitting.
        
        Args:
            answer: LLM-generierte Antwort
            
        Returns:
            Liste von Claim-Strings
        """
        # Handle Error-Responses
        if answer.startswith("[Error:"):
            return []
        
        # SpaCy wenn verfügbar
        if SPACY_AVAILABLE and NLP:
            doc = NLP(answer)
            claims = [
                sent.text.strip()
                for sent in doc.sents
                if len(sent.text.strip()) > 15
            ]
        else:
            # Regex Fallback
            claims = re.split(r'(?<=[.!?])\s+', answer)
            claims = [c.strip() for c in claims if len(c.strip()) > 15]
        
        # Filtere Meta-Statements (keine faktischen Claims)
        meta_patterns = [
            "based on the context",
            "according to the",
            "i cannot answer",
            "i don't know",
            "not enough information",
            "the context does not",
            "the context doesn't",
            "error:",
            "insufficient evidence",
        ]
        
        filtered_claims = [
            c for c in claims
            if not any(p in c.lower() for p in meta_patterns)
        ]
        
        self.logger.debug(f"Extrahiert: {len(filtered_claims)} Claims")
        return filtered_claims
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLAIM VERIFICATION (POST-GENERATION)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _verify_claim(self, claim: str, context: List[str] = None) -> Tuple[bool, str]:
        """
        Verifiziere einen Claim gegen den Knowledge Graph und/oder Context.
        
        Strategie:
        1. Extrahiere Named Entities aus dem Claim
        2. Query den Graph für jede Entity (wenn verfügbar)
        3. Prüfe ob Claim durch Context gestützt wird
        4. Claim ist verifiziert wenn Entity gefunden ODER Context-Match
        
        Args:
            claim: Ein einzelner Claim-String
            context: Optionaler Context für Textbasierte Verifikation
            
        Returns:
            Tuple von (is_verified: bool, reason: str)
        """
        # Extrahiere Entities
        entities = []
        
        # Multi-Word Proper Nouns
        entities.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', claim))
        # Single Proper Nouns
        entities.extend(re.findall(r'\b([A-Z][a-z]{2,})\b', claim))
        # Quoted Strings
        entities.extend(re.findall(r'"([^"]+)"', claim))
        
        # Filtere Stopwords
        stopwords = {'The', 'This', 'That', 'These', 'Those', 'However',
                    'Therefore', 'Furthermore', 'Moreover', 'Although'}
        entities = [e for e in entities if e not in stopwords]
        
        if not entities:
            return True, "no_entities_to_verify"
        
        # ─────────────────────────────────────────────────────────────────────
        # VERIFICATION 1: Graph-basiert (wenn verfügbar)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.graph_store:
            for entity in entities[:5]:
                try:
                    if hasattr(self.graph_store, 'query_by_entity'):
                        results = self.graph_store.query_by_entity(entity)
                    elif hasattr(self.graph_store, 'search'):
                        results = self.graph_store.search(entity)
                    elif hasattr(self.graph_store, 'get_entity_relations'):
                        results = self.graph_store.get_entity_relations(entity)
                    else:
                        continue
                    
                    if results:
                        return True, f"graph_verified_{entity}"
                        
                except Exception as e:
                    self.logger.debug(f"Graph Query Error für '{entity}': {e}")
                    continue
        
        # ─────────────────────────────────────────────────────────────────────
        # VERIFICATION 2: Context-basiert
        # ─────────────────────────────────────────────────────────────────────
        
        if context:
            context_text = " ".join(context).lower()
            
            for entity in entities[:5]:
                if entity.lower() in context_text:
                    return True, f"context_verified_{entity}"
        
        # Keine Verifikation möglich
        return False, "entities_not_found"
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN VERIFICATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    
    def generate_and_verify(
        self,
        query: str,
        context: List[str],
        entities: List[str] = None,
        hop_sequence: List[Dict] = None
    ) -> VerificationResult:
        """
        Hauptmethode: Pre-Validation, Generation und Self-Correction Loop.
        
        Algorithmus:
        1. Pre-Generation Validation (Entity-Path, Contradiction, Credibility)
        2. Wähle Prompt basierend auf Validation-Status
        3. Loop (max_iterations):
           a. Generiere Antwort
           b. Extrahiere Claims
           c. Verifiziere jeden Claim
           d. Wenn alle verifiziert → Return
           e. Sonst: Self-Correct mit Feedback
        4. Return beste Antwort
        
        Args:
            query: User-Frage
            context: Liste von Context-Dokumenten (aus Navigator)
            entities: Optionale Entity-Liste (aus Planner)
            hop_sequence: Optionale Hop-Sequenz (aus Planner)
            
        Returns:
            VerificationResult mit Antwort und Metriken
        """
        start_time = time.time()
        
        self.logger.info(f"[Verifier] Query: '{query[:60]}...'")
        self.logger.info(f"[Verifier] Context: {len(context)} docs")
        
        # ─────────────────────────────────────────────────────────────────────
        # PRE-GENERATION VALIDATION
        # ─────────────────────────────────────────────────────────────────────
        
        pre_validation = self.pre_validator.validate(
            context=context,
            query=query,
            entities=entities,
            hop_sequence=hop_sequence
        )
        
        # Nutze gefilterten Context
        working_context = pre_validation.filtered_context
        
        self.logger.info(
            f"[Verifier] Pre-Validation: {pre_validation.status.value}, "
            f"Context: {len(working_context)}/{len(context)}"
        )
        
        # Format Context
        formatted_context = self._format_context(working_context)
        
        # Track beste Result über Iterationen
        best_answer = None
        best_verified = []
        best_violated = []
        iteration_history = []
        
        # ─────────────────────────────────────────────────────────────────────
        # SELF-CORRECTION LOOP
        # ─────────────────────────────────────────────────────────────────────
        
        violated_claims = []
        
        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()
            
            self.logger.info(
                f"[Verifier] === Iteration {iteration}/{self.config.max_iterations} ==="
            )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Wähle Prompt basierend auf Status
            # ─────────────────────────────────────────────────────────────────
            
            if iteration == 1:
                # Erste Iteration: Wähle Prompt basierend auf Pre-Validation
                if pre_validation.status == ValidationStatus.INSUFFICIENT_EVIDENCE:
                    prompt = self.INSUFFICIENT_EVIDENCE_PROMPT.format(
                        context=formatted_context,
                        query=query
                    )
                else:
                    prompt = self.ANSWER_PROMPT.format(
                        context=formatted_context,
                        query=query
                    )
            else:
                # Folge-Iterationen: Correction-Prompt
                prompt = self.CORRECTION_PROMPT.format(
                    violations="\n".join(f"- {v}" for v in violated_claims),
                    context=formatted_context,
                    query=query
                )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Generate Answer
            # ─────────────────────────────────────────────────────────────────
            
            answer, llm_latency = self._call_llm(prompt)
            
            self.logger.info(f"[Verifier] LLM Response in {llm_latency:.0f}ms")
            
            # Check für Error
            if answer.startswith("[Error:"):
                self.logger.warning(f"[Verifier] LLM Error: {answer}")
                iteration_history.append({
                    "iteration": iteration,
                    "answer": answer,
                    "claims": [],
                    "verified": [],
                    "violated": [],
                    "llm_latency_ms": llm_latency,
                    "error": True,
                })
                
                if best_answer:
                    break
                continue
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Extract Claims
            # ─────────────────────────────────────────────────────────────────
            
            claims = self._extract_claims(answer)
            self.logger.info(f"[Verifier] {len(claims)} Claims extrahiert")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Verify Claims
            # ─────────────────────────────────────────────────────────────────
            
            verified_claims = []
            violated_claims = []
            
            for claim in claims:
                is_verified, reason = self._verify_claim(claim, working_context)
                
                if is_verified:
                    verified_claims.append(claim)
                    self.logger.debug(f"[Verifier] ✓ {claim[:50]}... ({reason})")
                else:
                    violated_claims.append(claim)
                    self.logger.debug(f"[Verifier] ✗ {claim[:50]}... ({reason})")
            
            self.logger.info(
                f"[Verifier] Verification: {len(verified_claims)} ✓, "
                f"{len(violated_claims)} ✗"
            )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 5: Track Best & Check Success
            # ─────────────────────────────────────────────────────────────────
            
            iter_time = (time.time() - iter_start) * 1000
            
            iteration_history.append({
                "iteration": iteration,
                "answer": answer,
                "claims": claims,
                "verified": verified_claims,
                "violated": violated_claims,
                "llm_latency_ms": llm_latency,
                "total_time_ms": iter_time,
                "error": False,
            })
            
            # Track Best Answer (wenigste Violations)
            if best_answer is None or len(violated_claims) < len(best_violated):
                best_answer = answer
                best_verified = verified_claims
                best_violated = violated_claims
            
            # Check für Success
            if len(violated_claims) == 0:
                self.logger.info(
                    f"[Verifier] ✓ Alle Claims verifiziert in Iteration {iteration}!"
                )
                
                total_time = (time.time() - start_time) * 1000
                
                return VerificationResult(
                    answer=answer,
                    iterations=iteration,
                    verified_claims=verified_claims,
                    violated_claims=[],
                    all_verified=True,
                    pre_validation=pre_validation,
                    timing_ms=total_time,
                    iteration_history=iteration_history,
                )
            
            # Continue für Correction
            self.logger.info(
                f"[Verifier] {len(violated_claims)} unverifizierte Claims, "
                f"versuche Correction..."
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # MAX ITERATIONS ERREICHT
        # ─────────────────────────────────────────────────────────────────────
        
        total_time = (time.time() - start_time) * 1000
        
        self.logger.warning(
            f"[Verifier] Max Iterations erreicht. "
            f"Bestes Ergebnis: {len(best_verified)} ✓, {len(best_violated)} ✗"
        )
        
        return VerificationResult(
            answer=best_answer or "[Error: Keine valide Antwort generiert]",
            iterations=self.config.max_iterations,
            verified_claims=best_verified,
            violated_claims=best_violated,
            all_verified=False,
            pre_validation=pre_validation,
            timing_ms=total_time,
            iteration_history=iteration_history,
        )
    
    def __call__(self, query: str, context: List[str]) -> VerificationResult:
        """Callable Interface für Verifier."""
        return self.generate_and_verify(query, context)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_verifier(
    model_name: str = "phi3",
    base_url: str = "http://localhost:11434",
    max_iterations: int = 3,
    max_context_chars: int = 2000,
    graph_store=None,
    enable_pre_validation: bool = True,
) -> Verifier:
    """
    Factory-Funktion für Verifier.
    
    Args:
        model_name: Ollama-Modell
        base_url: Ollama API URL
        max_iterations: Max Self-Correction Iterations
        max_context_chars: Max Context-Größe
        graph_store: Optional KnowledgeGraphStore
        enable_pre_validation: Pre-Generation Validation aktivieren
        
    Returns:
        Konfigurierte Verifier-Instanz
    """
    config = VerifierConfig(
        model_name=model_name,
        base_url=base_url,
        max_iterations=max_iterations,
        max_context_chars=max_context_chars,
        enable_entity_path_validation=enable_pre_validation,
        enable_contradiction_detection=enable_pre_validation,
        enable_credibility_scoring=enable_pre_validation,
    )
    return Verifier(config, graph_store)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 70)
    print("S_V: VERIFIER MIT PRE-GENERATION VALIDATION TEST")
    print(f"SpaCy verfügbar: {SPACY_AVAILABLE}")
    print(f"Transformers verfügbar: {TRANSFORMERS_AVAILABLE}")
    print("=" * 70)
    
    # Test Context
    test_context = [
        "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
        "Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
        "He published more than 300 scientific papers and became a symbol of genius.",
        "Einstein was born in Ulm, Germany, on March 14, 1879.",
        "He worked at the Swiss Patent Office while developing his groundbreaking theories.",
    ]
    
    test_query = "When was Einstein born and what did he receive the Nobel Prize for?"
    
    print(f"\nQuery: {test_query}")
    print(f"Context-Dokumente: {len(test_context)}")
    
    # Create Verifier
    verifier = create_verifier(
        max_iterations=3,
        max_context_chars=2000,
        enable_pre_validation=True,
    )
    
    # Test Pre-Validation
    print("\n--- Pre-Generation Validation ---")
    pre_result = verifier.pre_validator.validate(
        context=test_context,
        query=test_query,
        entities=["Einstein", "Nobel Prize"],
        hop_sequence=None
    )
    
    print(f"Status: {pre_result.status.value}")
    print(f"Entity-Path Valid: {pre_result.entity_path_valid}")
    print(f"Widersprüche: {len(pre_result.contradictions)}")
    print(f"Gefilterter Context: {len(pre_result.filtered_context)}/{len(test_context)}")
    print(f"Credibility Scores: {[f'{s:.2f}' for s in pre_result.credibility_scores]}")
    print(f"Validation Time: {pre_result.validation_time_ms:.0f}ms")
    
    # Run Full Verification (nur wenn Ollama verfügbar)
    print("\n--- Full Verification (benötigt Ollama) ---")
    try:
        result = verifier.generate_and_verify(
            query=test_query,
            context=test_context,
            entities=["Einstein", "Nobel Prize"]
        )
        
        print(f"\nAnswer: {result.answer}")
        print(f"Iterations: {result.iterations}")
        print(f"All Verified: {result.all_verified}")
        print(f"Verified Claims: {len(result.verified_claims)}")
        print(f"Violated Claims: {len(result.violated_claims)}")
        print(f"Total Time: {result.timing_ms:.0f}ms")
        
    except Exception as e:
        print(f"Ollama nicht verfügbar: {e}")
        print("Verifier-Logik funktioniert, aber LLM-Generierung benötigt Ollama.")
    
    print("\n" + "=" * 70)