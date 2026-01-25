"""
Verifier Stage (Sv) - Answer Generation + Graph-Grounded Verification

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

===============================================================================
OVERVIEW
===============================================================================

Der Verifier ist die letzte Stufe des Agentic RAG Systems und implementiert:

1. ANSWER GENERATION
   - Generiert Antworten basierend auf Context aus dem HybridRetriever
   - Nutzt quantisiertes SLM (phi3) für Edge-Deployment

2. CLAIM EXTRACTION  
   - Extrahiert atomare Claims (Fakten-Aussagen) aus der Antwort
   - Nutzt spaCy für Sentence Splitting (falls verfügbar)

3. GRAPH-GROUNDED VERIFICATION
   - Verifiziert jeden Claim gegen den Knowledge Graph
   - Prüft ob Entitäten und Relationen im Graph existieren

4. SELF-CORRECTION LOOP
   - Bei Violations: Feedback an LLM mit nicht-verifizierten Claims
   - Iteriert bis alle Claims verifiziert ODER max_iterations erreicht
   - Tracked beste Antwort über alle Iterationen

===============================================================================
ARCHITECTURE (Agentic Loop)
===============================================================================

    ┌─────────────────────────────────────────────────────────────┐
    │                    VERIFIER STAGE                           │
    │                                                             │
    │   Query + Context                                           │
    │        │                                                    │
    │        ▼                                                    │
    │   ┌─────────────┐                                          │
    │   │   GENERATE  │◄────────────────────┐                    │
    │   │   Answer    │                     │                    │
    │   └─────────────┘                     │                    │
    │        │                              │                    │
    │        ▼                              │                    │
    │   ┌─────────────┐                     │                    │
    │   │   EXTRACT   │                     │                    │
    │   │   Claims    │                     │ Self-Correction    │
    │   └─────────────┘                     │ (with feedback)    │
    │        │                              │                    │
    │        ▼                              │                    │
    │   ┌─────────────┐      Violations     │                    │
    │   │   VERIFY    │─────────────────────┘                    │
    │   │   Claims    │                                          │
    │   └─────────────┘                                          │
    │        │                                                    │
    │        ▼ All Verified                                       │
    │   ┌─────────────┐                                          │
    │   │   RETURN    │                                          │
    │   │   Answer    │                                          │
    │   └─────────────┘                                          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

===============================================================================
"""

import logging
import re
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# OPTIONAL: spaCy für bessere Sentence Segmentation
# =============================================================================

try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("spaCy loaded for claim extraction")
except:
    NLP = None
    SPACY_AVAILABLE = False
    logger.info("spaCy not available, using regex for claim extraction")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VerifierConfig:
    """
    Konfiguration für Verifier Stage.
    
    Context Settings:
        max_context_chars: Maximale Gesamtlänge des Contexts
        max_docs: Maximale Anzahl Dokumente im Context
        max_chars_per_doc: Maximale Zeichen pro Dokument
    
    LLM Settings:
        model_name: Ollama Modell (z.B. "phi3")
        base_url: Ollama API URL
        temperature: Sampling Temperature (0.0 = deterministisch)
        max_tokens: Maximale Antwort-Tokens
        timeout: API Timeout in Sekunden
    
    Loop Settings:
        max_iterations: Maximale Self-Correction Iterationen
        stop_on_first_success: Bei erster vollständig verifizierten Antwort stoppen
    """
    # LLM Settings
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 200
    timeout: int = 300  # 5 Minuten - genug für CPU
    
    # Context Settings (für Edge angepasst)
    max_context_chars: int = 2000   # ~500 tokens
    max_docs: int = 5               # Top-5 aus Hybrid Retrieval
    max_chars_per_doc: int = 350    # Pro Chunk
    
    # Agentic Loop Settings
    max_iterations: int = 3         # Self-Correction Iterations
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
        timing_ms: Gesamtzeit in Millisekunden
        iteration_history: Details pro Iteration (für Analyse)
    """
    answer: str
    iterations: int
    verified_claims: List[str] = field(default_factory=list)
    violated_claims: List[str] = field(default_factory=list)
    all_verified: bool = False
    timing_ms: float = 0.0
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# VERIFIER CLASS
# =============================================================================

class Verifier:
    """
    Verifier Stage: Answer Generation + Graph-Grounded Verification.
    
    Implementiert den Self-Correction Loop:
    1. Generiere Answer basierend auf Context
    2. Extrahiere atomare Claims
    3. Verifiziere Claims gegen Knowledge Graph
    4. Bei Violations → Retry mit Feedback (bis max_iterations)
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

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def __init__(
        self, 
        config: Optional[VerifierConfig] = None,
        graph_store = None,
    ):
        """
        Initialize Verifier.
        
        Args:
            config: VerifierConfig instance (uses defaults if None)
            graph_store: KnowledgeGraphStore für Verification (optional)
        """
        self.config = config or VerifierConfig()
        self.graph_store = graph_store
        self.logger = logger
        
        self.logger.info(
            f"Verifier initialized: "
            f"model={self.config.model_name}, "
            f"max_iterations={self.config.max_iterations}, "
            f"max_context={self.config.max_context_chars} chars"
        )
    
    def set_graph_store(self, graph_store) -> None:
        """Setze Graph Store für Verification."""
        self.graph_store = graph_store
        self.logger.info("Graph store connected to Verifier")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LLM INTERACTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _call_llm(self, prompt: str) -> Tuple[str, float]:
        """
        Rufe Ollama LLM API auf.
        
        Args:
            prompt: Vollständiger Prompt
            
        Returns:
            Tuple of (response_text, latency_ms)
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
            self.logger.error(f"[VERIFIER] Timeout after {self.config.timeout}s")
            return "[Error: LLM timeout - try reducing context size]", latency_ms
            
        except requests.exceptions.ConnectionError:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error("[VERIFIER] Cannot connect to Ollama")
            return "[Error: Cannot connect to Ollama - is it running?]", latency_ms
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"[VERIFIER] Error: {e}")
            return f"[Error: {str(e)[:100]}]", latency_ms
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONTEXT FORMATTING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _format_context(self, context: List[str]) -> str:
        """
        Formatiere Context-Dokumente mit Größenlimits.
        
        Strategy:
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
            # Truncate document intelligently (at word boundary)
            if len(doc) > self.config.max_chars_per_doc:
                truncated = doc[:self.config.max_chars_per_doc]
                # Try to break at last complete sentence or word
                last_period = truncated.rfind('. ')
                if last_period > self.config.max_chars_per_doc * 0.7:
                    truncated = truncated[:last_period + 1]
                else:
                    last_space = truncated.rfind(' ')
                    if last_space > 0:
                        truncated = truncated[:last_space] + "..."
            else:
                truncated = doc
            
            # Format with index
            part = f"[{i+1}] {truncated}"
            
            # Check total limit
            if total_chars + len(part) > self.config.max_context_chars:
                self.logger.debug(
                    f"Context limit reached at doc {i+1}/{len(context)}, "
                    f"using {len(formatted_parts)} docs"
                )
                break
            
            formatted_parts.append(part)
            total_chars += len(part) + 2  # +2 for newlines
        
        result = "\n\n".join(formatted_parts)
        self.logger.debug(
            f"Context formatted: {len(formatted_parts)} docs, "
            f"{total_chars} chars (~{total_chars // 4} tokens)"
        )
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLAIM EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extrahiere atomare Claims (Fakten-Aussagen) aus der Antwort.
        
        Nutzt spaCy für Sentence Splitting falls verfügbar,
        sonst Regex-basiertes Splitting.
        
        Args:
            answer: LLM-generierte Antwort
            
        Returns:
            Liste von Claim-Strings
        """
        # Handle error responses
        if answer.startswith("[Error:"):
            return []
        
        # Use spaCy if available
        if SPACY_AVAILABLE and NLP:
            doc = NLP(answer)
            claims = [
                sent.text.strip() 
                for sent in doc.sents 
                if len(sent.text.strip()) > 15
            ]
        else:
            # Regex fallback
            claims = re.split(r'(?<=[.!?])\s+', answer)
            claims = [c.strip() for c in claims if len(c.strip()) > 15]
        
        # Filter meta-statements (these aren't factual claims)
        meta_patterns = [
            "based on the context",
            "according to the",
            "i cannot answer",
            "i don't know",
            "not enough information",
            "the context does not",
            "the context doesn't",
            "error:",
        ]
        
        filtered_claims = [
            c for c in claims 
            if not any(p in c.lower() for p in meta_patterns)
        ]
        
        self.logger.debug(f"Extracted {len(filtered_claims)} claims from answer")
        return filtered_claims
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLAIM VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _verify_claim(self, claim: str) -> Tuple[bool, str]:
        """
        Verifiziere einen Claim gegen den Knowledge Graph.
        
        Verification Strategy:
        1. Extrahiere Named Entities aus dem Claim
        2. Query den Graph für jede Entity
        3. Claim ist verifiziert wenn mindestens eine Entity gefunden
        
        Args:
            claim: Ein einzelner Claim-String
            
        Returns:
            Tuple of (is_verified: bool, reason: str)
        """
        # No graph store → assume verified (can't check)
        if self.graph_store is None:
            return True, "no_graph_store"
        
        try:
            # Extract potential entities (capitalized words, quoted strings)
            entities = []
            
            # Multi-word proper nouns
            entities.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', claim))
            # Single proper nouns
            entities.extend(re.findall(r'\b([A-Z][a-z]{2,})\b', claim))
            # Quoted strings
            entities.extend(re.findall(r'"([^"]+)"', claim))
            
            # Filter common words
            stopwords = {'The', 'This', 'That', 'These', 'Those', 'However', 
                        'Therefore', 'Furthermore', 'Moreover', 'Although'}
            entities = [e for e in entities if e not in stopwords]
            
            if not entities:
                # No entities to verify → assume OK
                return True, "no_entities_to_verify"
            
            # Query graph for each entity
            for entity in entities[:5]:  # Limit to first 5 entities
                try:
                    # Try different query methods based on graph_store interface
                    if hasattr(self.graph_store, 'query_by_entity'):
                        results = self.graph_store.query_by_entity(entity)
                    elif hasattr(self.graph_store, 'search'):
                        results = self.graph_store.search(entity)
                    elif hasattr(self.graph_store, 'get_entity_relations'):
                        results = self.graph_store.get_entity_relations(entity)
                    else:
                        # Unknown interface
                        return True, "graph_interface_unknown"
                    
                    if results:
                        return True, f"verified_via_{entity}"
                        
                except Exception as e:
                    self.logger.debug(f"Graph query error for '{entity}': {e}")
                    continue
            
            # No entity found in graph
            return False, "entities_not_in_graph"
            
        except Exception as e:
            self.logger.warning(f"Verification error: {e}")
            return True, f"verification_error_{str(e)[:20]}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN VERIFICATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    
    def generate_and_verify(
        self, 
        query: str, 
        context: List[str]
    ) -> VerificationResult:
        """
        Hauptmethode: Generiere und verifiziere Antwort mit Self-Correction Loop.
        
        Algorithm:
        1. Formatiere Context
        2. Loop (max_iterations):
           a. Generiere Antwort (mit Correction-Feedback ab Iteration 2)
           b. Extrahiere Claims
           c. Verifiziere jeden Claim gegen Graph
           d. Wenn alle verifiziert → Return
           e. Sonst: Tracke beste Antwort, continue
        3. Return beste Antwort
        
        Args:
            query: User-Frage
            context: Liste von Context-Dokumenten (aus HybridRetriever)
            
        Returns:
            VerificationResult mit Antwort und Metriken
        """
        start_time = time.time()
        
        self.logger.info(f"[Verifier] Processing: '{query[:60]}...'")
        self.logger.info(f"[Verifier] Context: {len(context)} documents")
        
        # Format context
        formatted_context = self._format_context(context)
        
        # Track best result across iterations
        best_answer = None
        best_verified = []
        best_violated = []
        iteration_history = []
        
        # ─────────────────────────────────────────────────────────────────────
        # SELF-CORRECTION LOOP
        # ─────────────────────────────────────────────────────────────────────
        
        violated_claims = []  # For correction feedback
        
        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()
            
            self.logger.info(
                f"[Verifier] === Iteration {iteration}/{self.config.max_iterations} ==="
            )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Generate Answer
            # ─────────────────────────────────────────────────────────────────
            
            if iteration == 1:
                # First iteration: standard prompt
                prompt = self.ANSWER_PROMPT.format(
                    context=formatted_context,
                    query=query
                )
            else:
                # Subsequent iterations: correction prompt with violations
                prompt = self.CORRECTION_PROMPT.format(
                    violations="\n".join(f"- {v}" for v in violated_claims),
                    context=formatted_context,
                    query=query
                )
            
            answer, llm_latency = self._call_llm(prompt)
            
            self.logger.info(f"[Verifier] LLM response in {llm_latency:.0f}ms")
            self.logger.debug(f"[Verifier] Answer: {answer[:100]}...")
            
            # Check for error
            if answer.startswith("[Error:"):
                self.logger.warning(f"[Verifier] LLM Error: {answer}")
                
                # Record iteration
                iteration_history.append({
                    "iteration": iteration,
                    "answer": answer,
                    "claims": [],
                    "verified": [],
                    "violated": [],
                    "llm_latency_ms": llm_latency,
                    "error": True,
                })
                
                # If we have a previous good answer, use it
                if best_answer:
                    break
                    
                # Otherwise continue trying
                continue
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Extract Claims
            # ─────────────────────────────────────────────────────────────────
            
            claims = self._extract_claims(answer)
            self.logger.info(f"[Verifier] Extracted {len(claims)} claims")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Verify Claims
            # ─────────────────────────────────────────────────────────────────
            
            verified_claims = []
            violated_claims = []
            
            for claim in claims:
                is_verified, reason = self._verify_claim(claim)
                
                if is_verified:
                    verified_claims.append(claim)
                    self.logger.debug(f"[Verifier] ✓ Verified: {claim[:50]}... ({reason})")
                else:
                    violated_claims.append(claim)
                    self.logger.debug(f"[Verifier] ✗ Violated: {claim[:50]}... ({reason})")
            
            self.logger.info(
                f"[Verifier] Verification: {len(verified_claims)} verified, "
                f"{len(violated_claims)} violated"
            )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Track Best & Check Success
            # ─────────────────────────────────────────────────────────────────
            
            iter_time = (time.time() - iter_start) * 1000
            
            # Record iteration history
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
            
            # Track best answer (fewest violations)
            if best_answer is None or len(violated_claims) < len(best_violated):
                best_answer = answer
                best_verified = verified_claims
                best_violated = violated_claims
            
            # Check for success
            if len(violated_claims) == 0:
                self.logger.info(f"[Verifier] ✓ All claims verified in iteration {iteration}!")
                
                total_time = (time.time() - start_time) * 1000
                
                return VerificationResult(
                    answer=answer,
                    iterations=iteration,
                    verified_claims=verified_claims,
                    violated_claims=[],
                    all_verified=True,
                    timing_ms=total_time,
                    iteration_history=iteration_history,
                )
            
            # Continue to next iteration for correction
            self.logger.info(
                f"[Verifier] {len(violated_claims)} unverified claims, "
                f"attempting correction..."
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # MAX ITERATIONS REACHED
        # ─────────────────────────────────────────────────────────────────────
        
        total_time = (time.time() - start_time) * 1000
        
        self.logger.warning(
            f"[Verifier] Max iterations reached. "
            f"Best result: {len(best_verified)} verified, {len(best_violated)} violated"
        )
        
        return VerificationResult(
            answer=best_answer or "[Error: No valid answer generated]",
            iterations=self.config.max_iterations,
            verified_claims=best_verified,
            violated_claims=best_violated,
            all_verified=False,
            timing_ms=total_time,
            iteration_history=iteration_history,
        )
    
    def __call__(self, query: str, context: List[str]) -> VerificationResult:
        """Callable interface for Verifier."""
        return self.generate_and_verify(query, context)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_verifier(
    model_name: str = "phi3",
    base_url: str = "http://localhost:11434",
    max_iterations: int = 3,
    max_context_chars: int = 2000,
    graph_store = None,
) -> Verifier:
    """
    Factory function für Verifier.
    
    Args:
        model_name: Ollama Modell
        base_url: Ollama API URL
        max_iterations: Max Self-Correction Iterations
        max_context_chars: Max Context-Größe
        graph_store: Optional KnowledgeGraphStore
        
    Returns:
        Configured Verifier instance
    """
    config = VerifierConfig(
        model_name=model_name,
        base_url=base_url,
        max_iterations=max_iterations,
        max_context_chars=max_context_chars,
    )
    return Verifier(config, graph_store)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 70)
    print("VERIFIER STAGE TEST")
    print("=" * 70)
    
    # Test context
    test_context = [
        "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
        "Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
        "He published more than 300 scientific papers and became a symbol of genius.",
        "Einstein was born in Ulm, Germany, on March 14, 1879.",
        "He worked at the Swiss Patent Office while developing his groundbreaking theories.",
    ]
    
    test_query = "When was Einstein born and what did he receive the Nobel Prize for?"
    
    print(f"\nQuery: {test_query}")
    print(f"Context documents: {len(test_context)}")
    
    # Create verifier
    verifier = create_verifier(
        max_iterations=3,
        max_context_chars=2000,
    )
    
    # Test context formatting
    print("\n--- Context Formatting ---")
    formatted = verifier._format_context(test_context)
    print(f"Formatted length: {len(formatted)} chars")
    
    # Run verification
    print("\n--- Running Verification Loop ---")
    result = verifier.generate_and_verify(test_query, test_context)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"All Verified: {result.all_verified}")
    print(f"Verified Claims: {len(result.verified_claims)}")
    print(f"Violated Claims: {len(result.violated_claims)}")
    print(f"Total Time: {result.timing_ms:.0f}ms")
    
    print("\n--- Iteration History ---")
    for hist in result.iteration_history:
        print(f"  Iteration {hist['iteration']}: "
              f"{len(hist['verified'])} verified, "
              f"{len(hist['violated'])} violated, "
              f"{hist['llm_latency_ms']:.0f}ms")
    
    print("=" * 70)