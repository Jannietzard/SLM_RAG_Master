"""
Logic Layer — Agent-Based Query Processing

Implements Artifact B of the master's thesis:
"Enhancing Reasoning Fidelity in Quantized SLMs on Edge Devices"

Three-agent architecture:
    S_P (Planner)   — Query classification, entity extraction, plan generation
    S_N (Navigator) — Hybrid retrieval, RRF fusion, pre-generative filtering
    S_V (Verifier)  — Pre-validation, answer generation, self-correction loop

Usage:
    from src.logic_layer import AgenticController, create_controller

    controller = create_controller()
    result = controller.run("What is the capital of France?")
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
    
    # Component classes (for advanced use)
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
# Navigator (S_N) - Retrieval & Pre-Generative Filtering
# =============================================================================
from .navigator import (
    # Main class
    Navigator,

    # Data classes
    NavigatorResult,
    ControllerConfig,
)

# =============================================================================
# Controller (S_P → S_N → S_V) - Pipeline Orchestration
# =============================================================================
from .controller import (
    # Main class
    AgenticController,

    # Factory function
    create_controller,

    # State (for LangGraph integration)
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