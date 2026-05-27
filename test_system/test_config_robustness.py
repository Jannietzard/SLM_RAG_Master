"""
Robustness tests for ControllerConfig.from_yaml().

Validates that the config loader degrades gracefully when settings.yaml
keys are missing, partially present, or contain edge-case values.

Classified CRITICAL in the 2026-04-25 test health audit: all prior config
tests were happy-path only — no negative or boundary coverage existed.

Run:
    pytest test_system/test_config_robustness.py -v

Review History:
    Created:        2026-04-25
    Review Result:  Addresses 🔴 DEFEKT rating from test health audit
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logic_layer._config import ControllerConfig


# =============================================================================
# TestControllerConfigFromYaml
# =============================================================================

class TestControllerConfigFromYaml:
    """Boundary and negative tests for ControllerConfig.from_yaml()."""

    def test_empty_dict_returns_all_defaults(self) -> None:
        """from_yaml({}) must return a config with every field at its default value."""
        cfg = ControllerConfig.from_yaml({})
        assert cfg.model_name == "qwen2:1.5b"
        assert cfg.temperature == pytest.approx(0.0)
        assert cfg.rrf_k == 60
        assert cfg.max_verification_iterations == 2
        assert cfg.relevance_threshold_factor == pytest.approx(0.85)
        assert cfg.redundancy_threshold == pytest.approx(0.8)
        assert cfg.max_context_chunks == 10
        assert cfg.top_k_per_subquery == 10

    def test_missing_navigator_block_uses_nav_defaults(self) -> None:
        """When 'navigator' block is absent, all navigator fields fall back to defaults."""
        cfg = ControllerConfig.from_yaml({"llm": {"model_name": "phi3"}})
        assert cfg.rrf_k == 60
        assert cfg.max_context_chunks == 10
        assert cfg.top_k_per_subquery == 10
        assert cfg.relevance_threshold_factor == pytest.approx(0.85)

    def test_missing_llm_block_uses_llm_defaults(self) -> None:
        """When 'llm' block is absent, all LLM fields fall back to defaults."""
        cfg = ControllerConfig.from_yaml({"navigator": {"rrf_k": 30}})
        assert cfg.model_name == "qwen2:1.5b"
        assert cfg.temperature == pytest.approx(0.0)
        assert cfg.max_chars_per_doc == 500

    def test_missing_agent_block_uses_agent_defaults(self) -> None:
        """When 'agent' block is absent, max_verification_iterations falls back to 2."""
        cfg = ControllerConfig.from_yaml({"llm": {}, "navigator": {}})
        assert cfg.max_verification_iterations == 2

    def test_partial_navigator_honours_supplied_values(self) -> None:
        """Partially-specified navigator: supplied values honoured, unspecified at defaults."""
        cfg = ControllerConfig.from_yaml({
            "navigator": {
                "rrf_k": 30,
                "max_context_chunks": 5,
            }
        })
        assert cfg.rrf_k == 30
        assert cfg.max_context_chunks == 5
        # Unspecified fields must still have their defaults.
        assert cfg.relevance_threshold_factor == pytest.approx(0.85)
        assert cfg.redundancy_threshold == pytest.approx(0.8)

    def test_all_blocks_provided_uses_all_values(self) -> None:
        """When all three settings blocks are present, every supplied value is applied."""
        cfg = ControllerConfig.from_yaml({
            "llm": {
                "model_name": "phi3",
                "temperature": 0.0,
                "max_chars_per_doc": 300,
            },
            "agent": {"max_verification_iterations": 3},
            "navigator": {
                "rrf_k": 45,
                "relevance_threshold_factor": 0.9,
                "redundancy_threshold": 0.7,
                "max_context_chunks": 8,
            },
        })
        assert cfg.model_name == "phi3"
        assert cfg.max_verification_iterations == 3
        assert cfg.rrf_k == 45
        assert cfg.relevance_threshold_factor == pytest.approx(0.9)
        assert cfg.redundancy_threshold == pytest.approx(0.7)
        assert cfg.max_context_chunks == 8
        assert cfg.max_chars_per_doc == 300

    def test_unknown_keys_in_blocks_are_silently_ignored(self) -> None:
        """Extra keys in a settings block must not raise an error."""
        cfg = ControllerConfig.from_yaml({
            "navigator": {
                "rrf_k": 60,
                "nonexistent_future_key": "some_value",
            }
        })
        assert cfg.rrf_k == 60


# =============================================================================
# TestControllerConfigDefaults
# =============================================================================

class TestControllerConfigDefaults:
    """Unit tests for ControllerConfig direct-construction defaults."""

    def test_default_model_is_thesis_model(self) -> None:
        """ControllerConfig() default model must be the thesis evaluation model."""
        cfg = ControllerConfig()
        assert cfg.model_name == "qwen2:1.5b"

    def test_nonzero_temperature_emits_user_warning(self) -> None:
        """ControllerConfig(temperature=0.5) must emit UserWarning about non-determinism."""
        with pytest.warns(UserWarning, match="temperature"):
            ControllerConfig(temperature=0.5)

    def test_zero_temperature_does_not_warn(self) -> None:
        """ControllerConfig(temperature=0.0) must not emit any warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = ControllerConfig(temperature=0.0)
        assert cfg.temperature == pytest.approx(0.0)


# =============================================================================
# TestConfigFileLoading (Action #8)
# =============================================================================

class TestConfigFileLoading:
    """File-level config loading: graceful degradation on bad/missing files (F1+F2).

    _load_settings() is designed to never raise — it returns {} and logs on
    errors so the pipeline can always fall back to dataclass defaults.  These
    tests document and enforce that contract.
    """

    def test_malformed_yaml_returns_empty_dict_not_raises(self) -> None:
        """yaml.YAMLError during parse must return {} without propagating the exception."""
        import yaml
        from src.logic_layer._settings_loader import _load_settings
        from unittest.mock import patch
        with patch("yaml.safe_load", side_effect=yaml.YAMLError("simulated bad yaml")):
            result = _load_settings()
        assert isinstance(result, dict), "_load_settings() must always return a dict"
        assert result == {}, (
            "_load_settings() must return {} on YAML parse error; "
            f"got: {result!r}"
        )

    def test_missing_settings_file_returns_empty_dict_not_raises(self) -> None:
        """When settings.yaml does not exist, _load_settings() must return {}."""
        from src.logic_layer._settings_loader import _load_settings
        from unittest.mock import patch
        from pathlib import Path
        with patch.object(Path, "exists", return_value=False):
            result = _load_settings()
        assert isinstance(result, dict), "_load_settings() must always return a dict"
        assert result == {}, (
            "_load_settings() must return {} when settings.yaml is missing; "
            f"got: {result!r}"
        )
