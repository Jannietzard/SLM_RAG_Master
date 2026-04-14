"""
Tests for src/data_layer/embeddings.py

Covers:
  - EmbeddingCache: put/get, batch lookup, duplicate handling, access_count
    consistency, miss → None, model-scope isolation, clear, get_stats
  - EmbeddingMetrics: cache_hit_rate, avg_time_per_text_ms, reset
  - BatchedOllamaEmbeddings: embed_documents (cache miss/hit, order preservation,
    total_texts counter), embed_query (total_texts regression guard, cache hit),
    clear_cache, get_metrics, context manager protocol
  - create_embeddings: factory parameter mapping from settings dict

All Ollama API calls are mocked; no network connection is required.

Run from project root:
    pytest test_system/test_embeddings.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_response(embeddings_list):
    """Return a mock requests.Response yielding the given embeddings list."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embeddings": embeddings_list}
    return resp


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path (isolated per test)."""
    return tmp_path / "test_embeddings.db"


@pytest.fixture
def embedder(tmp_db):
    """
    BatchedOllamaEmbeddings instance with a mocked Ollama connection.

    The _test_connection call during __init__ is satisfied by the mock
    returning a single 768-dim vector.
    """
    with patch("requests.post", return_value=_mock_response([[0.0] * 768])):
        from src.data_layer.embeddings import BatchedOllamaEmbeddings
        return BatchedOllamaEmbeddings(
            model_name="test-model",
            batch_size=2,
            cache_path=tmp_db,
        )


# ── EmbeddingCache ─────────────────────────────────────────────────────────────

class TestEmbeddingCache:

    def test_put_and_get_hit(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        cache.put("hello world", [0.1, 0.2, 0.3], "test-model")
        result = cache.get("hello world", "test-model")
        assert result == [0.1, 0.2, 0.3]
        cache.close()

    def test_get_miss_returns_none(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        assert cache.get("unknown text", "test-model") is None
        cache.close()

    def test_model_scope_isolation(self, tmp_db):
        """Entry stored for model-A must not be returned for model-B."""
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        cache.put("text", [1.0], "model-A")
        assert cache.get("text", "model-B") is None
        cache.close()

    def test_hash_is_deterministic(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        h1 = cache._hash_text("deterministic input")
        h2 = cache._hash_text("deterministic input")
        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)
        cache.close()

    def test_different_texts_produce_different_hashes(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        assert cache._hash_text("text A") != cache._hash_text("text B")
        cache.close()

    def test_access_count_increments_on_get(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        cache.put("x", [0.5], "m")
        cache.get("x", "m")
        cache.get("x", "m")
        stats = cache.get_stats()
        # Initial access_count=1 (from put) + 2 get hits = 3
        assert stats["total_accesses"] >= 3
        cache.close()

    def test_clear_removes_all_entries(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        cache.put("a", [1.0], "m")
        cache.put("b", [2.0], "m")
        cache.clear()
        assert cache.get("a", "m") is None
        assert cache.get_stats()["total_entries"] == 0
        cache.close()

    def test_get_stats_returns_expected_keys(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        stats = cache.get_stats()
        for key in ("total_entries", "total_accesses", "size_bytes", "size_mb"):
            assert key in stats
        cache.close()

    def test_get_batch_returns_hits_only(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        cache.put("alpha", [1.0], "m")
        cache.put("beta", [2.0], "m")
        result = cache.get_batch(["alpha", "beta", "gamma"], "m")
        assert 0 in result and result[0] == [1.0]
        assert 1 in result and result[1] == [2.0]
        assert 2 not in result, "Cache miss must not appear in result"
        cache.close()

    def test_get_batch_handles_duplicate_texts(self, tmp_db):
        """Duplicate input texts must each receive an index in the result."""
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        cache.put("dup", [9.9], "m")
        result = cache.get_batch(["dup", "dup"], "m")
        assert 0 in result and 1 in result
        assert result[0] == result[1] == [9.9]
        cache.close()

    def test_get_batch_updates_access_count(self, tmp_db):
        """get_batch must increment access_count (consistency with get())."""
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        cache.put("trackme", [1.0], "m")
        before = cache.get_stats()["total_accesses"]
        cache.get_batch(["trackme"], "m")
        after = cache.get_stats()["total_accesses"]
        assert after > before
        cache.close()

    def test_get_batch_empty_input(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        assert cache.get_batch([], "m") == {}
        cache.close()

    def test_db_path_alias(self, tmp_db):
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(tmp_db)
        assert cache.db_path == cache.cache_path
        cache.close()


# ── EmbeddingMetrics ──────────────────────────────────────────────────────────

class TestEmbeddingMetrics:

    def test_cache_hit_rate_zero_when_empty(self):
        from src.data_layer.embeddings import EmbeddingMetrics

        assert EmbeddingMetrics().cache_hit_rate == 0.0

    def test_cache_hit_rate_calculation(self):
        from src.data_layer.embeddings import EmbeddingMetrics

        m = EmbeddingMetrics(cache_hits=3, cache_misses=1)
        assert m.cache_hit_rate == 75.0

    def test_cache_hit_rate_all_hits(self):
        from src.data_layer.embeddings import EmbeddingMetrics

        m = EmbeddingMetrics(cache_hits=10, cache_misses=0)
        assert m.cache_hit_rate == 100.0

    def test_avg_time_zero_when_no_texts(self):
        from src.data_layer.embeddings import EmbeddingMetrics

        assert EmbeddingMetrics().avg_time_per_text_ms == 0.0

    def test_avg_time_calculation(self):
        from src.data_layer.embeddings import EmbeddingMetrics

        m = EmbeddingMetrics(total_texts=4, total_time_ms=200.0)
        assert m.avg_time_per_text_ms == 50.0

    def test_reset_clears_all_fields(self):
        from src.data_layer.embeddings import EmbeddingMetrics

        m = EmbeddingMetrics(
            total_texts=10, cache_hits=5, cache_misses=5,
            batch_count=2, total_time_ms=100.0,
        )
        m.reset()
        assert m.total_texts == 0
        assert m.cache_hits == 0
        assert m.cache_misses == 0
        assert m.batch_count == 0
        assert m.total_time_ms == 0.0


# ── BatchedOllamaEmbeddings ───────────────────────────────────────────────────

class TestBatchedOllamaEmbeddings:

    def test_embed_documents_returns_correct_count(self, embedder):
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.1] * 768, [0.2] * 768],
        ):
            result = embedder.embed_documents(["A", "B"])
        assert len(result) == 2

    def test_embed_documents_preserves_order(self, embedder):
        """Output order must match input order regardless of batch boundaries."""
        vecs = [[float(i) / 10] * 768 for i in range(4)]
        with patch.object(
            embedder, "_embed_batch",
            side_effect=[vecs[:2], vecs[2:]],
        ):
            result = embedder.embed_documents(["a", "b", "c", "d"])
        for i, vec in enumerate(result):
            assert vec[0] == pytest.approx(i / 10)

    def test_embed_documents_cache_hit_on_second_call(self, embedder):
        """Second call with same texts must not trigger any API call."""
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.5] * 768, [0.6] * 768],
        ):
            embedder.embed_documents(["X", "Y"])

        with patch.object(
            embedder, "_embed_batch",
            side_effect=AssertionError("API must not be called on cache hit"),
        ):
            embedder.embed_documents(["X", "Y"])

        assert embedder.metrics.cache_hits == 2

    def test_embed_documents_increments_total_texts(self, embedder):
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.1] * 768],
        ):
            embedder.embed_documents(["one text"])
        assert embedder.metrics.total_texts == 1

    def test_embed_documents_counts_misses(self, embedder):
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.1] * 768, [0.2] * 768],
        ):
            embedder.embed_documents(["new-a", "new-b"])
        assert embedder.metrics.cache_misses == 2

    def test_embed_documents_empty_input(self, embedder):
        assert embedder.embed_documents([]) == []
        assert embedder.metrics.total_texts == 0

    def test_embed_query_increments_total_texts(self, embedder):
        """embed_query must increment total_texts (regression guard for the
        bug where only embed_documents updated this counter)."""
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.3] * 768],
        ):
            embedder.embed_query("query text")
        assert embedder.metrics.total_texts == 1

    def test_embed_query_cache_hit_does_not_call_api(self, embedder):
        """After a first embed_query call, a second identical call must be
        served from cache without any API request."""
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.7] * 768],
        ):
            embedder.embed_query("cached query")

        with patch.object(
            embedder, "_embed_batch",
            side_effect=AssertionError("API must not be called on cache hit"),
        ):
            embedder.embed_query("cached query")

        assert embedder.metrics.cache_hits >= 1

    def test_embed_query_total_texts_increments_on_cache_hit(self, embedder):
        """total_texts must be incremented even when the result comes from cache."""
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.7] * 768],
        ):
            embedder.embed_query("repeat me")
            embedder.embed_query("repeat me")
        assert embedder.metrics.total_texts == 2

    def test_clear_cache_resets_metrics(self, embedder):
        embedder.metrics.total_texts = 42
        embedder.metrics.cache_hits = 10
        embedder.clear_cache()
        assert embedder.metrics.total_texts == 0
        assert embedder.metrics.cache_hits == 0

    def test_get_metrics_returns_all_keys(self, embedder):
        m = embedder.get_metrics()
        for key in (
            "total_texts", "cache_hits", "cache_misses",
            "cache_hit_rate", "batch_count", "total_time_ms",
            "avg_time_per_text_ms",
        ):
            assert key in m, f"Missing key: {key}"

    def test_embedding_dim_detected_at_init(self, embedder):
        """_embedding_dim must be set after the connection test."""
        assert embedder.embedding_dim == 768

    def test_context_manager_closes_cache(self, tmp_db):
        """with-statement must close the SQLite connection cleanly."""
        with patch("requests.post", return_value=_mock_response([[0.0] * 768])):
            from src.data_layer.embeddings import BatchedOllamaEmbeddings

            with BatchedOllamaEmbeddings(
                model_name="m", batch_size=2, cache_path=tmp_db,
            ) as emb:
                pass
        assert emb.cache.conn is None, "Cache connection must be closed after __exit__"


# ── create_embeddings factory ─────────────────────────────────────────────────

class TestCreateEmbeddings:

    def test_factory_reads_settings(self, tmp_db):
        """All settings.yaml entries must be forwarded to the embedder."""
        cfg = {
            "embeddings": {
                "model_name": "custom-model",
                "base_url": "http://localhost:11434",
                "cache_path": str(tmp_db),
            },
            "performance": {"batch_size": 16, "device": "cpu"},
            "llm": {"timeout": 30},
        }
        with patch(
            "requests.post",
            return_value=_mock_response([[0.0] * 768]),
        ):
            from src.data_layer.embeddings import create_embeddings

            emb = create_embeddings(cfg)

        assert emb.model_name == "custom-model"
        assert emb.batch_size == 16
        assert emb.timeout == 30

    def test_factory_none_uses_defaults(self, tmp_db):
        """create_embeddings(None) must use class-level defaults without error."""
        with patch(
            "requests.post",
            return_value=_mock_response([[0.0] * 768]),
        ):
            from src.data_layer.embeddings import create_embeddings, BatchedOllamaEmbeddings

            # Redirect default cache path to temp dir to avoid touching ./cache/
            _orig = BatchedOllamaEmbeddings.__init__

            def _patched(self, *args, **kwargs):
                kwargs["cache_path"] = tmp_db
                _orig(self, *args, **kwargs)

            with patch.object(BatchedOllamaEmbeddings, "__init__", _patched):
                emb = create_embeddings(None)

        # batch_size default must match settings.yaml performance.batch_size = 64
        assert emb.batch_size == 64
