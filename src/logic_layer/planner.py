"""
Planner Stage (Sp) - Query Decomposition für Agentic RAG Controller.

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

Funktion:
- Zerlegt komplexe Multi-Hop-Queries in einfachere Sub-Queries
- Nutzt Few-shot Prompting für konsistente Decomposition

Arbeitet mit deinen bestehenden Modulen:
- retrieval.py → Navigator nutzt deinen HybridRetriever
- storage.py → Graph Store für Verification
"""

import logging
import re
from typing import List, Optional
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class PlannerConfig:
    """Konfiguration für Planner Stage."""
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 300
    max_sub_queries: int = 3


class Planner:
    """
    Planner Stage: Query Decomposition via Few-shot Prompting.
    """
    
    FEW_SHOT_EXAMPLES = """
Example 1:
Question: "What university did the founder of Microsoft attend?"
Sub-questions:
1. Who founded Microsoft?
2. What university did that person attend?

Example 2:
Question: "What is the capital of the country where the Eiffel Tower is located?"
Sub-questions:
1. In which country is the Eiffel Tower located?
2. What is the capital of that country?

Example 3:
Question: "Who is the CEO of the company that created the iPhone?"
Sub-questions:
1. Which company created the iPhone?
2. Who is the CEO of that company?
"""

    DECOMPOSITION_PROMPT = """You are a query decomposition assistant. Break down complex questions into simpler sub-questions.

Rules:
- Use 1-3 sub-questions (not more)
- If the question is already simple, return it as a single sub-question
- Number each sub-question (1., 2., etc.)

{examples}

Now decompose this question:
Question: "{query}"

Sub-questions:"""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.logger = logger
        self._test_connection()
        self.logger.info(f"Planner initialisiert: model={self.config.model_name}")
    
    def _test_connection(self) -> None:
        """Teste Ollama API Verbindung."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"HTTP {response.status_code}")
        except Exception as e:
            self.logger.error(f"Ollama Connection fehlgeschlagen: {e}")
            raise
    
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
        return response.json().get("response", "")
    
    def _is_simple_query(self, query: str) -> bool:
        """Prüfe ob Query einfach ist."""
        multi_hop_indicators = [
            "who founded", "who created", "the founder of", "the creator of",
            "the capital of the country where", "the CEO of the company that",
        ]
        query_lower = query.lower()
        for indicator in multi_hop_indicators:
            if indicator in query_lower:
                return False
        return len(query.split()) <= 8
    
    def _parse_sub_queries(self, llm_response: str) -> List[str]:
        """Parse Sub-Queries aus LLM Response."""
        sub_queries = []
        for line in llm_response.strip().split('\n'):
            line = line.strip()
            match = re.match(r'^[\d]+[.):\s]+(.+)$', line)
            if match:
                query = match.group(1).strip()
                if query and len(query) > 5:
                    sub_queries.append(query)
        return sub_queries[:self.config.max_sub_queries]
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Zerlege Query in Sub-Queries.
        
        Args:
            query: User Query
            
        Returns:
            Liste von Sub-Queries
        """
        self.logger.info(f"[Planner] Decomposing: '{query[:50]}...'")
        
        if self._is_simple_query(query):
            return [query]
        
        prompt = self.DECOMPOSITION_PROMPT.format(
            examples=self.FEW_SHOT_EXAMPLES, query=query
        )
        
        try:
            response = self._call_llm(prompt)
            sub_queries = self._parse_sub_queries(response)
            if not sub_queries:
                return [query]
            self.logger.info(f"[Planner] {len(sub_queries)} Sub-Queries generiert")
            return sub_queries
        except Exception as e:
            self.logger.error(f"[Planner] Fehler: {e}")
            return [query]
    
    def __call__(self, query: str) -> List[str]:
        return self.decompose_query(query)


def create_planner(
    model_name: str = "phi3",
    base_url: str = "http://localhost:11434",
) -> Planner:
    """Factory für Planner."""
    return Planner(PlannerConfig(model_name=model_name, base_url=base_url))