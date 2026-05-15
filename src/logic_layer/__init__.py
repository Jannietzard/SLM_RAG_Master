"""
Logic Layer — Agent-Based Query Processing

Implements Artifact B of the master's thesis:
"Enhancing Reasoning Fidelity in Quantized SLMs on Edge Devices"

Three-agent architecture:
    S_P (Planner)   — Query classification, entity extraction, plan generation
    S_N (Navigator) — Hybrid retrieval, RRF fusion, pre-generative filtering
    S_V (Verifier)  — Pre-validation, answer generation, self-correction loop

Usage:
    from src.pipeline import AgentPipeline, create_full_pipeline

    pipeline = create_full_pipeline()
    result = pipeline.process("What is the capital of France?")

Note (B7, 2026-05-15):
    AgenticController is now a static-helper container for bridge-entity
    extraction (see src/logic_layer/controller.py). The orchestrator
    methods (run, __call__, _build_workflow, _run_simple_pipeline) and the
    AgentState TypedDict were removed in the B7 cleanup. The production
    entry point is AgentPipeline.process().
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
# CONTROLLER — static helpers only (B7 cleanup, 2026-05-15)
# =============================================================================
# The full orchestrator (run/__call__/_run_simple_pipeline/LangGraph) was
# removed in B7. AgenticController remains exported as a static-helper
# container (bridge-entity extraction + hop-query rewriting) consumed by
# AgentPipeline._iterative_navigate. create_controller and AgentState were
# removed — use src.pipeline.AgentPipeline / create_full_pipeline instead.
from .controller import AgenticController

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
    # Controller (static-helper container after B7)
    "AgenticController",
]

__version__ = "4.0.0"
__author__ = "Edge-RAG Research Project"