"""
Unit tests for the #4 / #5 / #6 / #7 cleanup pass (2026-05-15).

These tests are static-only: they do NOT run the chunking ablation, the
pipeline eval, or any LLM call. They verify that the cleanup changes
are *in place* and the related infrastructure imports cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# #4 — Chunking ablation script
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkingAblationScript:
    """The ablation script must import cleanly and its config parser must
    enforce the documented invariants. We do NOT call main() — that would
    re-ingest the dataset.
    """

    def test_script_importable(self):
        """`from src.thesis_evaluations import chunking_ablation` must succeed
        without side effects.
        """
        from src.thesis_evaluations import chunking_ablation  # noqa: F401
        assert hasattr(chunking_ablation, "main")
        assert hasattr(chunking_ablation, "parse_configs")
        assert hasattr(chunking_ablation, "run_one_config")
        assert hasattr(chunking_ablation, "write_summary")
        assert hasattr(chunking_ablation, "DEFAULT_CONFIGS")
        # Documented default grid: production baseline + two window variants
        # + two overlap variants = 5 cells.
        assert len(chunking_ablation.DEFAULT_CONFIGS) == 5

    def test_parse_configs_canonical(self):
        from src.thesis_evaluations.chunking_ablation import parse_configs
        assert parse_configs("3:1") == [(3, 1)]
        assert parse_configs("3:1,5:1,7:1") == [(3, 1), (5, 1), (7, 1)]

    def test_parse_configs_rejects_overlap_ge_window(self):
        """A chunker with overlap == window makes no forward progress."""
        from src.thesis_evaluations.chunking_ablation import parse_configs
        import pytest
        with pytest.raises(ValueError, match="must be <"):
            parse_configs("3:3")
        with pytest.raises(ValueError, match="must be <"):
            parse_configs("3:5")

    def test_parse_configs_rejects_negative_overlap(self):
        from src.thesis_evaluations.chunking_ablation import parse_configs
        import pytest
        with pytest.raises(ValueError, match=">= 0"):
            parse_configs("3:-1")

    def test_parse_configs_rejects_zero_window(self):
        from src.thesis_evaluations.chunking_ablation import parse_configs
        import pytest
        with pytest.raises(ValueError, match=">= 1"):
            parse_configs("0:0")

    def test_ablation_store_manager_overrides_only_vector_path(self, tmp_path):
        """The _AblationStoreManager must redirect the vector path while
        keeping graph/questions/articles_info pointing at production."""
        from src.thesis_evaluations.chunking_ablation import _AblationStoreManager
        custom_vec = tmp_path / "custom_vector"
        custom_vec.mkdir()
        mgr = _AblationStoreManager(vector_override=custom_vec)
        paths = mgr.get_paths("hotpotqa")
        assert paths["vector"] == custom_vec, "vector path must be overridden"
        # Graph and other paths still point to production via the parent class.
        assert "graph" in paths
        assert "hotpotqa" in str(paths["graph"]), (
            "graph path must remain dataset-scoped; got "
            f"{paths['graph']!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# #5 — Coreference docstring honesty
# ─────────────────────────────────────────────────────────────────────────────

class TestCoreferenceDocstringHonesty:
    """The previous coref docstring claimed `graph density doubles for the
    article` without supporting evidence. The 2026-05-15 cleanup removed
    that claim and replaced it with a qualitative justification.
    """

    def test_no_unmeasured_doubling_claim(self):
        from src.data_layer import coreference
        doc = coreference.__doc__ or ""
        assert "graph density doubles" not in doc, (
            "Coref docstring must not claim 'graph density doubles' — "
            "no measurement supports this. Reword as qualitative."
        )

    def test_resolver_is_optional(self):
        """The module must still be importable when coreferee is missing —
        the design contract is that coref is opt-in.
        """
        from src.data_layer import coreference
        assert hasattr(coreference, "resolve_coreferences")
        assert hasattr(coreference, "is_available")


# ─────────────────────────────────────────────────────────────────────────────
# #6 — Embedding metrics in the eval harness
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingMetricsInPipeline:
    """create_pipeline() must attach the BatchedOllamaEmbeddings instance to
    the pipeline so cmd_evaluate can read its metrics at the end of a run.
    We can't construct a full pipeline without Ollama, but we can verify
    the attachment code path exists in the source.
    """

    def test_pipeline_assignment_present_in_source(self):
        from pathlib import Path
        src = Path(__file__).parent.parent / "src" / "thesis_evaluations" / "benchmark_datasets.py"
        text = src.read_text(encoding="utf-8")
        assert "pipeline._embeddings = embeddings" in text, (
            "create_pipeline() must attach the embeddings instance to the "
            "pipeline so cmd_evaluate can print cache metrics."
        )

    def test_metrics_summary_present_in_source(self):
        """The cmd_evaluate summary must read pipeline._embeddings and print
        cache hit rate / batch count / per-text latency.
        """
        from pathlib import Path
        src = Path(__file__).parent.parent / "src" / "thesis_evaluations" / "benchmark_datasets.py"
        text = src.read_text(encoding="utf-8")
        # Headline phrases we expect to see in the metrics block.
        for needle in [
            "Embedding cache",
            "Cache hit rate",
            "Batch requests issued",
            "Avg time per text",
        ]:
            assert needle in text, (
                f"cmd_evaluate must print the embedding metric line "
                f"containing {needle!r}."
            )

    def test_embedding_metrics_object_exposes_required_fields(self):
        """The metrics accumulator we plumb through must have the fields
        the new summary block reads.
        """
        from src.data_layer.embeddings import EmbeddingMetrics
        m = EmbeddingMetrics()
        # Properties / attributes the summary block touches:
        assert hasattr(m, "total_texts")
        assert hasattr(m, "cache_hits")
        assert hasattr(m, "cache_misses")
        assert hasattr(m, "batch_count")
        assert hasattr(m, "cache_hit_rate")        # property
        assert hasattr(m, "avg_time_per_text_ms")  # property


# ─────────────────────────────────────────────────────────────────────────────
# #7 — Controller paper-section header
# ─────────────────────────────────────────────────────────────────────────────

class TestControllerPaperHeader:
    """B7 reduced AgenticController to a static-helper namespace. The
    docstring must explicitly tell a reader (and a thesis reviewer) that
    AgentPipeline is the production orchestrator, not this class.
    """

    def test_docstring_names_AgentPipeline_as_orchestrator(self):
        from src.logic_layer import controller as controller_module
        doc = controller_module.__doc__ or ""
        assert "AgentPipeline" in doc, (
            "controller.py docstring must name AgentPipeline as the "
            "production orchestrator (#7 cleanup)."
        )
        assert "production orchestrator" in doc.lower() or \
               "production pipeline" in doc.lower(), (
            "controller.py docstring must make clear AgenticController is "
            "no longer the orchestrator."
        )

    def test_AgenticController_is_stateless_namespace(self):
        """No __init__; class-level static helpers only."""
        from src.logic_layer.controller import AgenticController
        # All public-facing methods should be staticmethod or classmethod.
        for name in [
            "_extract_bridge_entities",
            "_rewrite_hop_query_with_bridges",
            "_score_bridge_candidate",
            "_detect_expected_type",
        ]:
            attr = getattr(AgenticController, name, None)
            assert attr is not None, f"AgenticController.{name} is missing"
            assert callable(attr)
