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
# PLANNER (S_P) — Query Analysis & Planning
# =============================================================================
from .planner import (
    Planner,
    create_planner,
    QueryType,
    RetrievalStrategy,
    EntityInfo,
    HopStep,
    RetrievalPlan,
    # QueryClassifier, EntityExtractor, PlanGenerator are internal
    # sub-components of Planner — not part of the public API.
)

# =============================================================================
# NAVIGATOR (S_N) — Retrieval & Pre-Generative Filtering
# =============================================================================
from .navigator import (
    Navigator,
    NavigatorResult,
)

# ControllerConfig is defined in _config.py; import from there to keep the
# public API stable regardless of which production file is refactored.
from ._config import ControllerConfig

# =============================================================================
# VERIFIER (S_V) — Validation & Generation
# =============================================================================
from .verifier import (
    Verifier,
    create_verifier,
    VerifierConfig,
    ValidationStatus,
    ConfidenceLevel,
    SourceCredibility,
    PreValidationResult,
    VerificationResult,
    PreGenerationValidator,
)

# =============================================================================
# CONTROLLER (S_P → S_N → S_V) — Pipeline Orchestration
# =============================================================================
from .controller import (
    AgenticController,
    create_controller,
    AgentState,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Planner
    "Planner",
    "create_planner",
    "QueryType",
    "RetrievalStrategy",
    "EntityInfo",
    "HopStep",
    "RetrievalPlan",
    # Navigator
    "Navigator",
    "NavigatorResult",
    "ControllerConfig",
    # Verifier
    "Verifier",
    "create_verifier",
    "VerifierConfig",
    "ValidationStatus",
    "ConfidenceLevel",
    "SourceCredibility",
    "PreValidationResult",
    "VerificationResult",
    "PreGenerationValidator",
    # Controller
    "AgenticController",
    "create_controller",
    "AgentState",
]

__version__ = "4.0.0"
__author__ = "Edge-RAG Research Project"