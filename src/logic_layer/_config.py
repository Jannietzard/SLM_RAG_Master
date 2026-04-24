"""
Shared configuration dataclass for the logic layer.

Internal module — not part of the public API.  Holds ControllerConfig so
that both navigator.py and controller.py can import it without creating a
circular dependency (controller.py → navigator.py → controller.py).

External consumers should import ControllerConfig via the package surface:
    from src.logic_layer import ControllerConfig
or via the controller module:
    from src.logic_layer.controller import ControllerConfig
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ControllerConfig:
    """
    Configuration for the AgenticController pipeline.

    All numeric defaults match the thesis evaluation settings documented in
    config/settings.yaml and serve only as emergency fallbacks.  Use
    ``ControllerConfig.from_yaml()`` or ``create_controller()`` for
    production use so live settings.yaml values are honoured.

    LLM Settings:
        model_name: Ollama model for S_V (e.g. "qwen2:1.5b").
        base_url: Ollama API endpoint.
        temperature: Sampling temperature (0.0 = fully deterministic).

    Pipeline Settings:
        max_verification_iterations: Maximum self-correction rounds.
            Default 2 = thesis configuration (1 initial + 1 correction).
            Reference: Madaan et al. (2023). "Self-Refine." NeurIPS 2023.

    Navigator Settings (pre-generative filtering, thesis section 3.3):
        relevance_threshold_factor: Dynamic relevance threshold multiplier.
        redundancy_threshold: Jaccard deduplication threshold.
        max_context_chunks: Maximum chunks passed to S_V after filtering.
        rrf_k: RRF smoothing constant (Cormack et al., 2009. SIGIR).
        top_k_per_subquery: Retrieval results kept per sub-query.
        max_chars_per_doc: Per-chunk truncation limit for S_V prompt.
        corroboration_source_weight: RRF boost per additional unique source.
        corroboration_query_weight: RRF boost per additional sub-query hit.
        contradiction_overlap_threshold: Word-overlap threshold for numeric
            contradiction detection.
        contradiction_ratio_threshold: Minimum numeric ratio to flag conflict.
        contradiction_min_value: Minimum numeric value to consider for ratio.
    """

    # LLM Settings — emergency fallbacks; live values read from settings.yaml
    model_name: str = "qwen2:1.5b"           # settings.yaml: llm.model_name
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    # Pipeline Settings
    max_verification_iterations: int = 2      # settings.yaml: agent.max_verification_iterations

    # Navigator Settings
    relevance_threshold_factor: float = 0.85  # settings.yaml: navigator.relevance_threshold_factor
    redundancy_threshold: float = 0.8         # settings.yaml: navigator.redundancy_threshold
    max_context_chunks: int = 10              # settings.yaml: navigator.max_context_chunks
    rrf_k: int = 60                           # settings.yaml: navigator.rrf_k
    top_k_per_subquery: int = 10              # settings.yaml: navigator.top_k_per_subquery
    max_chars_per_doc: int = 500              # settings.yaml: llm.max_chars_per_doc
    corroboration_source_weight: float = 0.1  # settings.yaml: navigator.corroboration_source_weight
    corroboration_query_weight: float = 0.05  # settings.yaml: navigator.corroboration_query_weight
    contradiction_overlap_threshold: float = 0.3   # settings.yaml: navigator.contradiction_overlap_threshold
    contradiction_ratio_threshold: float = 2.0     # settings.yaml: navigator.contradiction_ratio_threshold
    contradiction_min_value: float = 10.0          # settings.yaml: navigator.contradiction_min_value

    def __post_init__(self) -> None:
        import warnings as _warnings
        if self.temperature != 0.0:
            _warnings.warn(
                "ControllerConfig: temperature=%g — use 0.0 for deterministic "
                "thesis evaluation. Set llm.temperature in config/settings.yaml." % self.temperature,
                stacklevel=2,
            )

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "ControllerConfig":
        """
        Build a ControllerConfig from a settings.yaml dict.

        Reads the ``navigator``, ``llm``, and ``agent`` blocks. All defaults
        match the thesis evaluation settings documented in settings.yaml.
        Follows the same pattern as PlannerConfig.from_yaml().

        Args:
            config: Full settings.yaml dict (or the relevant sub-dict).

        Returns:
            ControllerConfig populated from the provided settings dict.
        """
        nav = config.get("navigator", {})
        llm = config.get("llm", {})
        agent = config.get("agent", {})
        return cls(
            model_name=llm.get("model_name", "qwen2:1.5b"),
            base_url=llm.get("base_url", "http://localhost:11434"),
            temperature=llm.get("temperature", 0.0),
            max_verification_iterations=agent.get("max_verification_iterations", 2),
            relevance_threshold_factor=nav.get("relevance_threshold_factor", 0.85),
            redundancy_threshold=nav.get("redundancy_threshold", 0.8),
            max_context_chunks=nav.get("max_context_chunks", 10),
            rrf_k=nav.get("rrf_k", 60),
            top_k_per_subquery=nav.get("top_k_per_subquery", 10),
            max_chars_per_doc=llm.get("max_chars_per_doc", 500),
            corroboration_source_weight=nav.get("corroboration_source_weight", 0.1),
            corroboration_query_weight=nav.get("corroboration_query_weight", 0.05),
            contradiction_overlap_threshold=nav.get("contradiction_overlap_threshold", 0.3),
            contradiction_ratio_threshold=nav.get("contradiction_ratio_threshold", 2.0),
            contradiction_min_value=nav.get("contradiction_min_value", 10.0),
        )
