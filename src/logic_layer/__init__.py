"""
Logic Layer - Agent-Based Query Processing

Implementiert Artefakt B der Masterarbeit:
"Enhancing Reasoning Fidelity in Quantized SLMs on Edge Devices"

Drei-Agenten-Architektur:
    S_P (Planner)   → Query-Klassifikation, Entity-Extraktion, Plan-Generierung
    S_N (Navigator) → Hybrid Retrieval, RRF-Fusion, Pre-Generative Filtering
    S_V (Verifier)  → Pre-Validation, Answer Generation, Self-Correction Loop

Usage:
    from src.logic_layer import AgenticController, create_controller
    
    controller = create_controller()
    result = controller.process("What is the capital of France?")
"""

# =============================================================================
# PLANNER (S_P) - Query Analysis & Planning
# =============================================================================
from .planner import (
    # Main class
    Planner,
    
    # Factory function
    create_planner,
    
    # Enums
    QueryType,
    RetrievalStrategy,
    
    # Data classes
    EntityInfo,
    HopStep,
    RetrievalPlan,
    
    # Component classes (für erweiterte Nutzung)
    QueryClassifier,
    EntityExtractor,
    PlanGenerator,
)

# =============================================================================
# VERIFIER (S_V) - Validation & Generation
# =============================================================================
from .verifier import (
    # Main class
    Verifier,
    
    # Factory function
    create_verifier,
    
    # Config
    VerifierConfig,
    
    # Enums
    ValidationStatus,
    
    # Data classes
    SourceCredibility,
    PreValidationResult,
    VerificationResult,
    
    # Component classes
    PreGenerationValidator,
)

# =============================================================================
# AGENT (S_N + Controller) - Navigation & Orchestration
# =============================================================================
from .agent import (
    # Main classes
    Navigator,
    AgenticController,
    
    # Factory function
    create_controller,
    
    # Data classes
    NavigatorResult,
    ControllerConfig,
    
    # State (für LangGraph Integration)
    AgentState,
)

# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    # Planner
    'Planner',
    'create_planner',
    'QueryType',
    'RetrievalStrategy',
    'EntityInfo',
    'HopStep',
    'RetrievalPlan',
    'QueryClassifier',
    'EntityExtractor',
    'PlanGenerator',
    
    # Verifier
    'Verifier',
    'create_verifier',
    'VerifierConfig',
    'ValidationStatus',
    'SourceCredibility',
    'PreValidationResult',
    'VerificationResult',
    'PreGenerationValidator',
    
    # Agent
    'Navigator',
    'AgenticController',
    'create_controller',
    'NavigatorResult',
    'ControllerConfig',
    'AgentState',
]

__version__ = '2.0.0'
__author__ = 'Jan Nietzard'