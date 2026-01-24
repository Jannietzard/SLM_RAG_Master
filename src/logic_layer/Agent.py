"""
Agentic Controller - LangGraph DAG Orchestration.

Masterthesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"

Funktion:
- Orchestriert Planner → Navigator → Verifier Pipeline
- Implementiert als Directed Acyclic Graph (DAG)
- State Management über TypedDict

Arbeitet mit deinen bestehenden Modulen:
- retrieval.py → HybridRetriever für Navigator Stage
- storage.py → KnowledgeGraphStore für Verifier

WICHTIG: Nutzt deinen bestehenden HybridRetriever, ersetzt nichts!
"""

import logging
import time
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass

# LangGraph (optional)
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph nicht installiert: pip install langgraph")

# Lokale Imports
from src.logic_layer.planner import Planner, create_planner
from src.logic_layer.verifier import Verifier, create_verifier, VerificationResult

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State für den Agentic Controller."""
    query: str
    sub_queries: List[str]
    context: List[str]
    answer: str
    iterations: int
    verified_claims: List[str]
    violated_claims: List[str]
    all_verified: bool
    total_time_ms: float
    errors: List[str]


@dataclass
class ControllerConfig:
    """Konfiguration für Agentic Controller."""
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_verification_iterations: int = 3


class AgenticController:
    """
    Agentic Controller: Planner → Navigator → Verifier Pipeline.
    
    Navigator nutzt deinen bestehenden HybridRetriever aus retrieval.py!
    
    Architektur:
    
    START
      │
      ▼
    [Planner] ─── Query Decomposition
      │
      ▼
    [Navigator] ─── Dein HybridRetriever (retrieval.py)
      │
      ▼
    [Verifier] ─── Answer + Self-Correction
      │
      ▼
     END
    """
    
    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        planner: Optional[Planner] = None,
        verifier: Optional[Verifier] = None,
    ):
        self.config = config or ControllerConfig()
        self.logger = logger
        
        # Initialize Components
        self.planner = planner or create_planner(
            model_name=self.config.model_name,
            base_url=self.config.base_url,
        )
        
        self.verifier = verifier or create_verifier(
            model_name=self.config.model_name,
            base_url=self.config.base_url,
            max_iterations=self.config.max_verification_iterations,
        )
        
        # Retriever wird später gesetzt (dein HybridRetriever)
        self.retriever = None
        self.documents = {}  # doc_id → text mapping
        
        # Build Workflow
        if LANGGRAPH_AVAILABLE:
            self.app = self._build_workflow()
            self.logger.info("AgenticController mit LangGraph initialisiert")
        else:
            self.app = None
            self.logger.info("AgenticController mit Simple Pipeline initialisiert")
    
    def set_retriever(self, retriever, documents: Dict[str, str] = None) -> None:
        """
        Setze deinen HybridRetriever aus retrieval.py.
        
        Args:
            retriever: HybridRetriever Instance
            documents: Optional dict doc_id → text
        """
        self.retriever = retriever
        if documents:
            self.documents = documents
        self.logger.info("HybridRetriever verbunden")
    
    def set_graph_store(self, graph_store) -> None:
        """
        Setze KnowledgeGraphStore aus storage.py für Verification.
        
        Args:
            graph_store: KnowledgeGraphStore Instance
        """
        self.verifier.set_graph_store(graph_store)
        self.logger.info("GraphStore für Verification verbunden")
    
    def _build_workflow(self):
        """Build LangGraph Workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("navigator", self._navigator_node)
        workflow.add_node("verifier", self._verifier_node)
        
        workflow.add_edge("planner", "navigator")
        workflow.add_edge("navigator", "verifier")
        workflow.add_edge("verifier", END)
        
        workflow.set_entry_point("planner")
        
        return workflow.compile()
    
    def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Planner Node: Query Decomposition."""
        self.logger.info("\n" + "="*50)
        self.logger.info("[PLANNER] Query Decomposition")
        self.logger.info("="*50)
        
        try:
            sub_queries = self.planner.decompose_query(state["query"])
            for i, sq in enumerate(sub_queries, 1):
                self.logger.info(f"  {i}. {sq}")
            return {"sub_queries": sub_queries}
        except Exception as e:
            self.logger.error(f"[PLANNER] Error: {e}")
            return {"sub_queries": [state["query"]], "errors": [str(e)]}
    
    def _navigator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Navigator Node: Hybrid Retrieval.
        
        Nutzt deinen HybridRetriever aus retrieval.py!
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("[NAVIGATOR] Hybrid Retrieval")
        self.logger.info("="*50)
        
        if self.retriever is None:
            self.logger.warning("Kein Retriever gesetzt!")
            return {"context": []}
        
        try:
            all_contexts = []
            
            for sub_query in state["sub_queries"]:
                # Nutze deinen HybridRetriever.retrieve()
                results = self.retriever.retrieve(sub_query)
                
                for result in results[:5]:  # Top 5 pro sub-query
                    # result ist ein RetrievalResult aus deiner retrieval.py
                    all_contexts.append(result.text)
            
            # Deduplizieren
            unique_contexts = list(dict.fromkeys(all_contexts))[:10]
            
            self.logger.info(f"[NAVIGATOR] {len(unique_contexts)} unique Context-Docs")
            return {"context": unique_contexts}
            
        except Exception as e:
            self.logger.error(f"[NAVIGATOR] Error: {e}")
            return {"context": [], "errors": state.get("errors", []) + [str(e)]}
    
    def _verifier_node(self, state: AgentState) -> Dict[str, Any]:
        """Verifier Node: Answer Generation + Verification."""
        self.logger.info("\n" + "="*50)
        self.logger.info("[VERIFIER] Answer Generation + Verification")
        self.logger.info("="*50)
        
        try:
            result = self.verifier.generate_and_verify(
                state["query"],
                state["context"]
            )
            
            self.logger.info(f"[VERIFIER] {result.iterations} Iteration(s)")
            self.logger.info(f"[VERIFIER] All verified: {result.all_verified}")
            
            return {
                "answer": result.answer,
                "iterations": result.iterations,
                "verified_claims": result.verified_claims,
                "violated_claims": result.violated_claims,
                "all_verified": result.all_verified,
            }
        except Exception as e:
            self.logger.error(f"[VERIFIER] Error: {e}")
            return {
                "answer": f"Error: {e}",
                "iterations": 0,
                "all_verified": False,
                "errors": state.get("errors", []) + [str(e)]
            }
    
    def _run_simple_pipeline(self, query: str) -> AgentState:
        """Fallback: Simple Pipeline ohne LangGraph."""
        state: AgentState = {
            "query": query,
            "sub_queries": [],
            "context": [],
            "answer": "",
            "iterations": 0,
            "verified_claims": [],
            "violated_claims": [],
            "all_verified": False,
            "total_time_ms": 0,
            "errors": [],
        }
        
        # Planner
        update = self._planner_node(state)
        state.update(update)
        
        # Navigator
        update = self._navigator_node(state)
        state.update(update)
        
        # Verifier
        update = self._verifier_node(state)
        state.update(update)
        
        return state
    
    def run(self, query: str) -> AgentState:
        """
        Führe Agentic Pipeline aus.
        
        Args:
            query: User Query
            
        Returns:
            AgentState mit Answer und Metadata
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("AGENTIC CONTROLLER - Pipeline Start")
        self.logger.info("="*70)
        self.logger.info(f"Query: {query}")
        
        start_time = time.time()
        
        initial_state: AgentState = {
            "query": query,
            "sub_queries": [],
            "context": [],
            "answer": "",
            "iterations": 0,
            "verified_claims": [],
            "violated_claims": [],
            "all_verified": False,
            "total_time_ms": 0,
            "errors": [],
        }
        
        if self.app is not None:
            final_state = self.app.invoke(initial_state)
        else:
            final_state = self._run_simple_pipeline(query)
        
        total_time = (time.time() - start_time) * 1000
        final_state["total_time_ms"] = total_time
        
        self.logger.info("\n" + "="*70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info(f"Time: {total_time:.0f}ms | Context: {len(final_state['context'])} | "
                        f"Iterations: {final_state['iterations']}")
        self.logger.info("="*70)
        
        return final_state
    
    def __call__(self, query: str) -> str:
        """Shortcut: Return nur Answer."""
        return self.run(query)["answer"]


def create_controller(
    model_name: str = "phi3",
    base_url: str = "http://localhost:11434",
    max_iterations: int = 3,
) -> AgenticController:
    """Factory für AgenticController."""
    config = ControllerConfig(
        model_name=model_name,
        base_url=base_url,
        max_verification_iterations=max_iterations,
    )
    return AgenticController(config)