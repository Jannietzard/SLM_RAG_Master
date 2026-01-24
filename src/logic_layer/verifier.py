"""
Verifier Stage (Sv) - Answer Generation + Graph-Grounded Verification.

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

Funktion:
- Generiert Antworten basierend auf Context (RAG)
- Extrahiert atomare Claims aus Antworten
- Verifiziert Claims gegen Knowledge Graph (dein storage.py)
- Implementiert Self-Correction Loop (max 3 Iterationen)

Arbeitet mit deinen bestehenden Modulen:
- storage.py → KnowledgeGraphStore für Verification
"""

import logging
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import requests

logger = logging.getLogger(__name__)

# Optional: spaCy für bessere Claim Extraction
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    NLP = None
    SPACY_AVAILABLE = False


@dataclass
class VerifierConfig:
    """Konfiguration für Verifier Stage."""
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 300
    max_iterations: int = 3


@dataclass
class VerificationResult:
    """Ergebnis der Verifikation."""
    answer: str
    iterations: int
    verified_claims: List[str] = field(default_factory=list)
    violated_claims: List[str] = field(default_factory=list)
    all_verified: bool = False


class Verifier:
    """
    Verifier Stage: Answer Generation + Graph-Grounded Verification.
    
    Implementiert Self-Correction Loop:
    1. Generiere Answer
    2. Extrahiere Claims
    3. Verifiziere gegen Graph
    4. Bei Violations → Retry mit Feedback
    """
    
    ANSWER_PROMPT = """You are a factual question-answering assistant. Answer based ONLY on the provided context.

Context:
{context}

{additional_instructions}

Question: {query}

Answer:"""

    CORRECTION_PROMPT = """The following claims could not be verified:
{violations}

Please correct your answer to only include verifiable facts.

Question: {query}
Context: {context}

Corrected answer:"""

    def __init__(
        self, 
        config: Optional[VerifierConfig] = None,
        graph_store=None,  # KnowledgeGraphStore aus deiner storage.py
    ):
        self.config = config or VerifierConfig()
        self.graph_store = graph_store
        self.logger = logger
        self.logger.info(f"Verifier initialisiert: model={self.config.model_name}")
    
    def set_graph_store(self, graph_store) -> None:
        """Setze Graph Store (KnowledgeGraphStore aus storage.py)."""
        self.graph_store = graph_store
    
    def _call_llm(self, prompt: str) -> str:
        """Rufe Ollama LLM API auf."""
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
            timeout=60
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API Error: {response.status_code}")
        return response.json().get("response", "").strip()
    
    def _format_context(self, context: List[str]) -> str:
        """Formatiere Context-Dokumente."""
        if not context:
            return "No context available."
        return "\n\n".join([f"[{i+1}] {doc[:1000]}" for i, doc in enumerate(context)])
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extrahiere atomare Claims (Sätze) aus Answer."""
        if SPACY_AVAILABLE and NLP:
            doc = NLP(answer)
            claims = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        else:
            claims = re.split(r'[.!?]+', answer)
            claims = [c.strip() for c in claims if len(c.strip()) > 10]
        
        # Filter Meta-Aussagen
        meta_patterns = ["based on", "according to", "i don't know", "not enough"]
        return [c for c in claims if not any(p in c.lower() for p in meta_patterns)]
    
    def _verify_claim(self, claim: str) -> Tuple[bool, str]:
        """
        Verifiziere Claim gegen Graph.
        
        Nutzt deinen KnowledgeGraphStore aus storage.py.
        """
        if self.graph_store is None:
            return True, "no_graph_store"  # Graceful degradation
        
        if not SPACY_AVAILABLE:
            return True, "no_spacy"
        
        # Extrahiere Entities aus Claim
        doc = NLP(claim)
        entities = [ent.text for ent in doc.ents 
                   if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT']]
        
        if len(entities) < 2:
            # Einzelne Entity: prüfe Existenz
            if len(entities) == 1:
                entity = entities[0].lower()
                for node in self.graph_store.graph.nodes():
                    if entity in str(node).lower():
                        return True, "entity_exists"
                return False, f"entity_not_found: {entities[0]}"
            return True, "no_entities"
        
        # Zwei+ Entities: prüfe ob Pfad existiert
        e1, e2 = entities[0], entities[1]
        
        # Finde Nodes im Graph
        node1 = node2 = None
        for node in self.graph_store.graph.nodes():
            node_str = str(node).lower()
            if e1.lower() in node_str:
                node1 = node
            if e2.lower() in node_str:
                node2 = node
        
        if node1 and node2:
            # Nutze graph_traversal aus deiner storage.py
            try:
                neighbors = self.graph_store.graph_traversal(node1, max_hops=2)
                if node2 in neighbors:
                    return True, "path_exists"
            except:
                pass
        
        return False, f"no_path: {e1} -> {e2}"
    
    def generate_and_verify(
        self, 
        query: str, 
        context: List[str],
    ) -> VerificationResult:
        """
        Generate Answer + Verify mit Self-Correction Loop.
        
        Args:
            query: User Query
            context: Context Dokumente (aus deinem HybridRetriever)
            
        Returns:
            VerificationResult
        """
        self.logger.info(f"[Verifier] Processing: '{query[:50]}...'")
        
        iteration = 0
        best_answer = None
        best_violations = None
        formatted_context = self._format_context(context)
        
        while iteration < self.config.max_iterations:
            iteration += 1
            self.logger.info(f"[Verifier] Iteration {iteration}/{self.config.max_iterations}")
            
            # Generate Answer
            if iteration == 1:
                prompt = self.ANSWER_PROMPT.format(
                    context=formatted_context,
                    query=query,
                    additional_instructions=""
                )
            else:
                prompt = self.CORRECTION_PROMPT.format(
                    violations="\n".join(f"- {v}" for v in violated_claims),
                    query=query,
                    context=formatted_context
                )
            
            answer = self._call_llm(prompt)
            
            # Extract & Verify Claims
            claims = self._extract_claims(answer)
            verified_claims = []
            violated_claims = []
            
            for claim in claims:
                is_ok, reason = self._verify_claim(claim)
                if is_ok:
                    verified_claims.append(claim)
                else:
                    violated_claims.append(claim)
                    self.logger.debug(f"[Verifier] Violation: {claim[:40]}... ({reason})")
            
            # Track best
            if best_answer is None or len(violated_claims) < len(best_violations or []):
                best_answer = answer
                best_violations = violated_claims
                best_verified = verified_claims
            
            # Check success
            if len(violated_claims) == 0:
                self.logger.info(f"[Verifier] ✓ All claims verified")
                return VerificationResult(
                    answer=answer,
                    iterations=iteration,
                    verified_claims=verified_claims,
                    violated_claims=[],
                    all_verified=True,
                )
        
        self.logger.warning(f"[Verifier] Max iterations, {len(best_violations)} unverified")
        return VerificationResult(
            answer=best_answer,
            iterations=iteration,
            verified_claims=best_verified,
            violated_claims=best_violations,
            all_verified=False,
        )
    
    def __call__(self, query: str, context: List[str]) -> VerificationResult:
        return self.generate_and_verify(query, context)


def create_verifier(
    model_name: str = "phi3",
    base_url: str = "http://localhost:11434",
    max_iterations: int = 3,
    graph_store=None,
) -> Verifier:
    """Factory für Verifier."""
    config = VerifierConfig(
        model_name=model_name,
        base_url=base_url,
        max_iterations=max_iterations,
    )
    return Verifier(config, graph_store)