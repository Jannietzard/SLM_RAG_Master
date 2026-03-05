"""
pytest Tests für src/pipeline

Abgedeckte Klassen:
    - PipelineResult
    - AgentPipeline
    - BatchProcessor
    - IngestionConfig
    - IngestionMetrics
    - DocumentLoader
    - MockEmbeddingGenerator / MockEntityExtractor
    - IngestionPipeline
    - create_pipeline / create_ingestion_pipeline (Factory Functions)

Alle LLM-Aufrufe werden gemockt (kein laufendes Ollama nötig).
"""

import json
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest


# ============================================================================
# HELPERS
# ============================================================================

def _make_mock_verifier_result():
    """Erstelle VerificationResult-ähnliches Mock-Objekt."""
    from src.logic_layer.verifier import VerificationResult, ConfidenceLevel
    result = VerificationResult(
        answer="Test answer.",
        iterations=1,
        verified_claims=["claim_a"],
        violated_claims=[],
        all_verified=True,
        timing_ms=10.0,
    )
    return result


# ============================================================================
# TestPipelineResult
# ============================================================================

class TestPipelineResult:
    """Tests für PipelineResult Dataclass."""

    def _make_result(self, **kwargs):
        from src.pipeline.agent_pipeline import PipelineResult
        defaults = dict(
            answer="Test answer",
            confidence="high",
            query="test query",
            planner_result={"query_type": "single_hop"},
            navigator_result={"filtered_context": []},
            verifier_result={"answer": "Test answer", "confidence": "high"},
            planner_time_ms=5.0,
            navigator_time_ms=20.0,
            verifier_time_ms=100.0,
            total_time_ms=125.0,
        )
        defaults.update(kwargs)
        return PipelineResult(**defaults)

    def test_to_dict_has_required_keys(self):
        """to_dict() enthält answer, confidence, query, stages, timing, optimization."""
        result = self._make_result()
        d = result.to_dict()
        assert "answer" in d
        assert "confidence" in d
        assert "query" in d
        assert "stages" in d
        assert "timing" in d
        assert "optimization" in d

    def test_to_dict_stages_structure(self):
        """stages-Dict enthält planner, navigator, verifier."""
        result = self._make_result()
        stages = result.to_dict()["stages"]
        assert "planner" in stages
        assert "navigator" in stages
        assert "verifier" in stages

    def test_to_dict_timing_keys(self):
        """timing-Dict enthält alle vier Timing-Werte."""
        result = self._make_result()
        timing = result.to_dict()["timing"]
        assert "planner_ms" in timing
        assert "navigator_ms" in timing
        assert "verifier_ms" in timing
        assert "total_ms" in timing

    def test_to_json_is_valid_json(self):
        """to_json() erzeugt valides JSON."""
        result = self._make_result()
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["answer"] == "Test answer"

    def test_optimization_flags_default_false(self):
        """early_exit_used und cached_result sind standardmäßig False."""
        result = self._make_result()
        assert result.early_exit_used is False
        assert result.cached_result is False

    def test_optimization_flags_in_to_dict(self):
        """Optimization-Flags erscheinen in to_dict()."""
        result = self._make_result(early_exit_used=True, cached_result=True)
        opt = result.to_dict()["optimization"]
        assert opt["early_exit"] is True
        assert opt["cached"] is True


# ============================================================================
# TestAgentPipeline
# ============================================================================

class TestAgentPipeline:
    """Tests für AgentPipeline Orchestrator."""

    @pytest.fixture
    def mock_agents(self):
        """Erstelle Mock-Agenten für schnelle Tests."""
        from src.logic_layer.planner import Planner
        from src.logic_layer.navigator import Navigator, ControllerConfig
        from src.logic_layer.verifier import Verifier, VerifierConfig

        planner = Planner()
        navigator = Navigator(ControllerConfig())
        verifier = Verifier(config=VerifierConfig())
        return planner, navigator, verifier

    @pytest.fixture
    def pipeline(self, mock_agents):
        """Pipeline mit Mock-Agenten, Caching deaktiviert."""
        from src.pipeline.agent_pipeline import AgentPipeline
        planner, navigator, verifier = mock_agents
        return AgentPipeline(
            planner=planner,
            navigator=navigator,
            verifier=verifier,
            enable_caching=False,
        )

    def test_initialization_stores_agents(self, mock_agents):
        """Pipeline speichert alle drei Agenten."""
        from src.pipeline.agent_pipeline import AgentPipeline
        planner, navigator, verifier = mock_agents
        pipeline = AgentPipeline(planner=planner, navigator=navigator, verifier=verifier)
        assert pipeline.planner is planner
        assert pipeline.navigator is navigator
        assert pipeline.verifier is verifier

    def test_initialization_defaults(self):
        """Pipeline kann ohne Agenten initialisiert werden."""
        from src.pipeline.agent_pipeline import AgentPipeline
        pipeline = AgentPipeline()
        assert pipeline.planner is None
        assert pipeline.navigator is None
        assert pipeline.verifier is None
        assert pipeline.enable_early_exit is True
        assert pipeline.enable_caching is True

    def test_get_stats_initial_values(self, pipeline):
        """Statistiken sind initial alle null."""
        stats = pipeline.get_stats()
        assert stats["total_queries"] == 0
        assert stats["cache_hits"] == 0
        assert stats["early_exits"] == 0

    def test_get_stats_has_required_keys(self, pipeline):
        """Statistiken enthalten alle erwarteten Keys."""
        stats = pipeline.get_stats()
        for key in ("total_queries", "cache_hits", "early_exits",
                    "avg_latency_ms", "cache_size", "cache_hit_rate", "early_exit_rate"):
            assert key in stats, f"Fehlender Key: {key}"

    def test_clear_cache(self):
        """clear_cache() leert den internen Cache."""
        from src.pipeline.agent_pipeline import AgentPipeline, PipelineResult
        pipeline = AgentPipeline()
        # Manuell eintragen
        pipeline._cache["key"] = Mock()
        assert len(pipeline._cache) == 1
        pipeline.clear_cache()
        assert len(pipeline._cache) == 0

    def test_process_increments_total_queries(self, pipeline):
        """Jede process()-Anfrage erhöht total_queries."""
        mock_result = _make_mock_verifier_result()
        with patch.object(pipeline.verifier, '_call_llm', return_value=("Answer.", 0.05)):
            pipeline.process("What is machine learning?")
        assert pipeline.get_stats()["total_queries"] == 1

    def test_process_returns_pipeline_result(self, pipeline):
        """process() gibt PipelineResult zurück."""
        from src.pipeline.agent_pipeline import PipelineResult
        with patch.object(pipeline.verifier, '_call_llm', return_value=("ML answer.", 0.05)):
            result = pipeline.process("What is machine learning?")
        assert isinstance(result, PipelineResult)
        assert isinstance(result.answer, str)
        assert result.answer != ""

    def test_process_result_has_timing(self, pipeline):
        """PipelineResult enthält positive Timing-Werte."""
        with patch.object(pipeline.verifier, '_call_llm', return_value=("Answer.", 0.05)):
            result = pipeline.process("What is AI?")
        assert result.total_time_ms > 0
        assert result.planner_time_ms >= 0
        assert result.navigator_time_ms >= 0
        assert result.verifier_time_ms >= 0

    def test_cache_hit_on_repeated_query(self, mock_agents):
        """Zweite identische Anfrage trifft den Cache."""
        from src.pipeline.agent_pipeline import AgentPipeline
        planner, navigator, verifier = mock_agents
        pipeline = AgentPipeline(
            planner=planner, navigator=navigator, verifier=verifier,
            enable_caching=True,
        )
        with patch.object(pipeline.verifier, '_call_llm', return_value=("Answer.", 0.05)):
            pipeline.process("What is Python?")
            result2 = pipeline.process("What is Python?")
        assert result2.cached_result is True
        assert pipeline.get_stats()["cache_hits"] == 1

    def test_caching_disabled_no_cache_hits(self, pipeline):
        """Mit enable_caching=False gibt es keine Cache-Hits."""
        with patch.object(pipeline.verifier, '_call_llm', return_value=("Answer.", 0.05)):
            pipeline.process("What is Python?")
            pipeline.process("What is Python?")
        assert pipeline.get_stats()["cache_hits"] == 0

    def test_verifier_result_contains_confidence(self, pipeline):
        """verifier_result-Dict enthält 'confidence' Key (Bug Fix #3)."""
        with patch.object(pipeline.verifier, '_call_llm', return_value=("Answer.", 0.05)):
            result = pipeline.process("Who invented the telephone?")
        assert "confidence" in result.verifier_result


# ============================================================================
# TestBatchProcessor
# ============================================================================

class TestBatchProcessor:
    """Tests für BatchProcessor."""

    @pytest.fixture
    def processor(self):
        """BatchProcessor mit einfacher Mock-Pipeline."""
        from src.pipeline.agent_pipeline import AgentPipeline, BatchProcessor, PipelineResult
        mock_pipeline = Mock(spec=AgentPipeline)
        mock_pipeline.process.return_value = PipelineResult(
            answer="test answer",
            confidence="high",
            query="q",
            planner_result={},
            navigator_result={},
            verifier_result={},
            planner_time_ms=1.0,
            navigator_time_ms=2.0,
            verifier_time_ms=3.0,
            total_time_ms=6.0,
        )
        mock_pipeline.get_stats.return_value = {"total_queries": 0}
        return BatchProcessor(mock_pipeline)

    def test_process_batch_returns_list(self, processor):
        """process_batch() gibt Liste zurück."""
        results = processor.process_batch(["Q1", "Q2", "Q3"])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_process_batch_simplified_keys(self, processor):
        """Simplified-Modus enthält query, answer, confidence, latency_ms."""
        results = processor.process_batch(["Q1"])
        r = results[0]
        assert "query" in r
        assert "answer" in r
        assert "confidence" in r
        assert "latency_ms" in r

    def test_process_batch_return_details(self, processor):
        """return_details=True gibt vollständiges to_dict() zurück."""
        results = processor.process_batch(["Q1"], return_details=True)
        r = results[0]
        assert "stages" in r or "answer" in r  # to_dict()-Format

    def test_process_batch_handles_exception(self, processor):
        """Fehler in process() werden pro Query abgefangen, nicht als Crash."""
        processor.pipeline.process.side_effect = [
            Exception("simulated error"),
            processor.pipeline.process.return_value,
        ]
        processor.pipeline.process.side_effect = None
        processor.pipeline.process.side_effect = Exception("err")
        results = processor.process_batch(["Bad query"])
        assert results[0]["error"] == "err"

    def test_exact_match_true(self):
        """_exact_match() ist case-insensitiv und strip()-sicher."""
        from src.pipeline.agent_pipeline import BatchProcessor
        assert BatchProcessor._exact_match("Paris", "paris") is True
        assert BatchProcessor._exact_match("  Paris  ", "Paris") is True

    def test_exact_match_false(self):
        """_exact_match() False bei unterschiedlichen Antworten."""
        from src.pipeline.agent_pipeline import BatchProcessor
        assert BatchProcessor._exact_match("Paris", "Berlin") is False

    def test_evaluate_returns_metrics(self, processor):
        """evaluate() gibt Dict mit accuracy und total_queries zurück."""
        processor.pipeline.process.side_effect = None
        from src.pipeline.agent_pipeline import PipelineResult
        processor.pipeline.process.return_value = PipelineResult(
            answer="test answer",
            confidence="high",
            query="q",
            planner_result={},
            navigator_result={},
            verifier_result={},
            planner_time_ms=1.0,
            navigator_time_ms=2.0,
            verifier_time_ms=3.0,
            total_time_ms=6.0,
        )
        metrics = processor.evaluate(["Q1"], ["test answer"])
        assert "accuracy" in metrics
        assert "total_queries" in metrics
        assert metrics["accuracy"] == 1.0


# ============================================================================
# TestIngestionConfig
# ============================================================================

class TestIngestionConfig:
    """Tests für IngestionConfig Dataclass."""

    def test_defaults(self):
        """IngestionConfig hat sinnvolle Standardwerte."""
        from src.pipeline.ingestion_pipeline import IngestionConfig
        config = IngestionConfig()
        assert config.sentences_per_chunk == 3
        assert config.sentence_overlap == 1
        assert config.embedding_dim == 768
        assert config.gliner_batch_size == 16
        assert config.rebel_batch_size == 8
        assert config.enable_caching is True

    def test_from_yaml_uses_provided_values(self):
        """from_yaml() übernimmt Werte aus Dict."""
        from src.pipeline.ingestion_pipeline import IngestionConfig
        yaml_cfg = {
            "chunking": {"sentence_chunking": {"sentences_per_chunk": 5}},
            "embedding": {"dimension": 384},
        }
        config = IngestionConfig.from_yaml(yaml_cfg)
        assert config.sentences_per_chunk == 5
        assert config.embedding_dim == 384

    def test_from_yaml_uses_defaults_for_missing_keys(self):
        """from_yaml() fällt auf Defaults zurück wenn Keys fehlen."""
        from src.pipeline.ingestion_pipeline import IngestionConfig
        config = IngestionConfig.from_yaml({})
        assert config.sentences_per_chunk == 3
        assert config.embedding_dim == 768


# ============================================================================
# TestIngestionMetrics
# ============================================================================

class TestIngestionMetrics:
    """Tests für IngestionMetrics Dataclass."""

    def test_defaults_zero(self):
        """Alle Zähler starten bei 0."""
        from src.pipeline.ingestion_pipeline import IngestionMetrics
        m = IngestionMetrics()
        assert m.documents_processed == 0
        assert m.chunks_created == 0
        assert m.entities_extracted == 0
        assert m.relations_extracted == 0

    def test_to_dict_structure(self):
        """to_dict() enthält counts, timing_ms, performance."""
        from src.pipeline.ingestion_pipeline import IngestionMetrics
        m = IngestionMetrics(documents_processed=2, chunks_created=10)
        d = m.to_dict()
        assert d["counts"]["documents"] == 2
        assert d["counts"]["chunks"] == 10
        assert "timing_ms" in d
        assert "performance" in d


# ============================================================================
# TestDocumentLoader
# ============================================================================

class TestDocumentLoader:
    """Tests für DocumentLoader."""

    @pytest.fixture
    def loader(self):
        from src.pipeline.ingestion_pipeline import DocumentLoader
        return DocumentLoader()

    def test_load_text_file(self, loader, tmp_path):
        """Lädt Plain-Text-Datei und gibt Dokument zurück."""
        f = tmp_path / "doc.txt"
        f.write_text("Hello world. This is a test.", encoding="utf-8")
        docs = list(loader.load(str(f)))
        assert len(docs) == 1
        assert docs[0]["text"] == "Hello world. This is a test."
        assert "source" in docs[0]["metadata"]

    def test_load_json_file_list(self, loader, tmp_path):
        """Lädt JSON-Array und gibt ein Dokument pro Item zurück."""
        data = [{"text": "First doc"}, {"text": "Second doc"}]
        f = tmp_path / "docs.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        docs = list(loader.load(str(f)))
        assert len(docs) == 2
        assert docs[0]["text"] == "First doc"

    def test_load_json_file_single_dict(self, loader, tmp_path):
        """Lädt JSON-Dict als einzelnes Dokument."""
        f = tmp_path / "doc.json"
        f.write_text(json.dumps({"text": "Single doc"}), encoding="utf-8")
        docs = list(loader.load(str(f)))
        assert len(docs) == 1

    def test_load_jsonl_file(self, loader, tmp_path):
        """Lädt JSONL-Datei (eine JSON-Zeile pro Dokument)."""
        lines = [json.dumps({"text": "Line 1"}), json.dumps({"text": "Line 2"})]
        f = tmp_path / "docs.jsonl"
        f.write_text("\n".join(lines), encoding="utf-8")
        docs = list(loader.load(str(f)))
        assert len(docs) == 2

    def test_load_hotpotqa_format(self, loader, tmp_path):
        """Parst HotpotQA-Format korrekt (context als Liste von Tupeln)."""
        item = {
            "_id": "abc123",
            "question": "Where was Einstein born?",
            "answer": "Ulm",
            "context": [
                ["Einstein", ["He was born in Ulm.", "He won the Nobel Prize."]],
                ["Nobel", ["The Nobel Prize was founded in 1895."]],
            ]
        }
        f = tmp_path / "hotpot.jsonl"
        f.write_text(json.dumps(item), encoding="utf-8")
        docs = list(loader.load(str(f)))
        assert len(docs) == 1
        assert "Einstein" in docs[0]["text"]
        assert docs[0]["id"] == "abc123"

    def test_load_missing_file_raises(self, loader):
        """Nicht existierender Pfad → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list(loader.load("/nonexistent/path/file.txt"))

    def test_load_directory(self, loader, tmp_path):
        """Verzeichnis mit mehreren .txt-Dateien wird rekursiv geladen."""
        (tmp_path / "a.txt").write_text("Doc A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Doc B", encoding="utf-8")
        docs = list(loader.load(str(tmp_path)))
        assert len(docs) == 2

    def test_generate_id_is_deterministic(self, loader):
        """Gleicher Source-String → gleiche ID."""
        from src.pipeline.ingestion_pipeline import DocumentLoader
        id1 = DocumentLoader._generate_id("test_source")
        id2 = DocumentLoader._generate_id("test_source")
        assert id1 == id2

    def test_generate_id_different_sources(self):
        """Verschiedene Sources → verschiedene IDs."""
        from src.pipeline.ingestion_pipeline import DocumentLoader
        assert DocumentLoader._generate_id("a") != DocumentLoader._generate_id("b")


# ============================================================================
# TestMockComponents
# ============================================================================

class TestMockComponents:
    """Tests für Mock-Komponenten (MockEmbeddingGenerator, MockEntityExtractor)."""

    def test_mock_embedding_shape(self):
        """MockEmbeddingGenerator gibt korrektes Shape zurück."""
        from src.pipeline.ingestion_pipeline import MockEmbeddingGenerator
        gen = MockEmbeddingGenerator(embedding_dim=768)
        embeddings = gen.embed(["text1", "text2", "text3"])
        assert embeddings.shape == (3, 768)

    def test_mock_embedding_l2_normalized(self):
        """Mock-Embeddings sind L2-normalisiert."""
        from src.pipeline.ingestion_pipeline import MockEmbeddingGenerator
        gen = MockEmbeddingGenerator(embedding_dim=128)
        embeddings = gen.embed(["hello world"])
        norms = np.linalg.norm(embeddings, axis=1)
        assert abs(norms[0] - 1.0) < 1e-6

    def test_mock_embedding_empty_input(self):
        """Leere Input-Liste → leeres Array."""
        from src.pipeline.ingestion_pipeline import MockEmbeddingGenerator
        gen = MockEmbeddingGenerator()
        result = gen.embed([])
        assert len(result) == 0

    def test_mock_entity_extractor_returns_lists(self):
        """MockEntityExtractor gibt zwei Listen zurück."""
        from src.pipeline.ingestion_pipeline import MockEntityExtractor
        extractor = MockEntityExtractor()
        chunks = [{"chunk_id": "c1", "text": "Apple was founded by Steve Jobs."}]
        entities, relations = extractor.process_chunks_batch(chunks)
        assert isinstance(entities, list)
        assert isinstance(relations, list)

    def test_mock_entity_extractor_extracts_capitalized(self):
        """MockEntityExtractor findet großgeschriebene Wörter als Entities."""
        from src.pipeline.ingestion_pipeline import MockEntityExtractor
        extractor = MockEntityExtractor()
        chunks = [{"chunk_id": "c1", "text": "Albert Einstein worked at Princeton University."}]
        entities, _ = extractor.process_chunks_batch(chunks)
        names = [e.name for e in entities]
        assert any("Einstein" in n or "Albert" in n or "Princeton" in n for n in names)


# ============================================================================
# TestIngestionPipeline
# ============================================================================

class TestIngestionPipeline:
    """Tests für IngestionPipeline (mit Mock-Komponenten)."""

    @pytest.fixture
    def pipeline(self):
        """Pipeline mit use_mocks=True — kein GPU/Modell nötig."""
        from src.pipeline.ingestion_pipeline import IngestionPipeline
        return IngestionPipeline(use_mocks=True)

    def test_initialization_with_mocks(self, pipeline):
        """Initialisierung mit use_mocks=True schlägt nicht fehl."""
        from src.pipeline.ingestion_pipeline import IngestionPipeline
        assert isinstance(pipeline, IngestionPipeline)

    def test_ingest_text_file(self, pipeline, tmp_path):
        """Ingestiert eine .txt-Datei und gibt IngestionMetrics zurück."""
        from src.pipeline.ingestion_pipeline import IngestionMetrics
        f = tmp_path / "test.txt"
        f.write_text(
            "Albert Einstein was born in 1879. He developed the theory of relativity. "
            "Einstein received the Nobel Prize in 1921.",
            encoding="utf-8"
        )
        metrics = pipeline.ingest(str(f))
        assert isinstance(metrics, IngestionMetrics)
        assert metrics.documents_processed == 1
        assert metrics.chunks_created >= 1

    def test_ingest_returns_metrics_with_timing(self, pipeline, tmp_path):
        """Metriken enthalten positive Gesamtzeit."""
        f = tmp_path / "test.txt"
        f.write_text("Hello world. This is test content.", encoding="utf-8")
        metrics = pipeline.ingest(str(f))
        assert metrics.total_time_ms > 0

    def test_ingest_resets_metrics_between_calls(self, pipeline, tmp_path):
        """Metriken werden bei jedem ingest()-Aufruf zurückgesetzt, nicht akkumuliert."""
        f = tmp_path / "test.txt"
        f.write_text(
            "The quick brown fox jumps over the lazy dog. "
            "Scientists discovered a new species in the Amazon rainforest. "
            "The experiment yielded unexpected results in the laboratory.",
            encoding="utf-8"
        )
        pipeline.ingest(str(f))
        pipeline.ingest(str(f))
        # documents_processed sollte 1 sein (nur letzte Ausführung, nicht 2)
        assert pipeline.get_metrics().documents_processed == 1

    def test_get_metrics_returns_ingestion_metrics(self, pipeline):
        """get_metrics() gibt IngestionMetrics zurück."""
        from src.pipeline.ingestion_pipeline import IngestionMetrics
        assert isinstance(pipeline.get_metrics(), IngestionMetrics)

    def test_chunk_document_fallback_no_crash(self):
        """Fallback-Chunker (chunker=None) läuft ohne Fehler."""
        from src.pipeline.ingestion_pipeline import IngestionPipeline, IngestionConfig
        pipeline = IngestionPipeline(
            config=IngestionConfig(sentences_per_chunk=3, sentence_overlap=1),
            use_mocks=True,
        )
        pipeline.chunker = None  # erzwinge Fallback
        chunks = pipeline._chunk_document(
            "Paris is the capital. France is in Europe. The Eiffel Tower is famous.",
            doc_id="doc1",
            metadata={}
        )
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_chunk_document_fallback_step_zero_no_crash(self):
        """Fallback-Chunker crasht nicht wenn sentences_per_chunk == sentence_overlap."""
        from src.pipeline.ingestion_pipeline import (
            IngestionPipeline, IngestionConfig, MockEntityExtractor, MockEmbeddingGenerator
        )
        config = IngestionConfig(sentences_per_chunk=2, sentence_overlap=2)
        # Chunker explizit als Mock übergeben, damit _init_chunker() nicht
        # SpacySentenceChunker mit invalider Config aufruft (overlap >= per_chunk)
        pipeline = IngestionPipeline(
            config=config,
            chunker=Mock(),
            entity_extractor=MockEntityExtractor(),
            embedding_generator=MockEmbeddingGenerator(config.embedding_dim),
            hybrid_store=Mock(),
        )
        pipeline.chunker = None  # Fallback-Pfad erzwingen
        chunks = pipeline._chunk_document(
            "Sentence one. Sentence two. Sentence three.",
            doc_id="doc1",
            metadata={}
        )
        assert isinstance(chunks, list)  # kein ValueError


# ============================================================================
# TestFactoryFunctions
# ============================================================================

class TestFactoryFunctions:
    """Tests für create_pipeline() und create_ingestion_pipeline()."""

    def test_create_ingestion_pipeline_default(self):
        """create_ingestion_pipeline() ohne Args gibt IngestionPipeline zurück."""
        from src.pipeline.ingestion_pipeline import create_ingestion_pipeline, IngestionPipeline
        pipeline = create_ingestion_pipeline(use_mocks=True)
        assert isinstance(pipeline, IngestionPipeline)

    def test_create_ingestion_pipeline_with_config(self):
        """create_ingestion_pipeline() mit YAML-Config übernimmt Werte."""
        from src.pipeline.ingestion_pipeline import create_ingestion_pipeline
        cfg = {"chunking": {"sentence_chunking": {"sentences_per_chunk": 5}}}
        pipeline = create_ingestion_pipeline(config=cfg, use_mocks=True)
        assert pipeline.config.sentences_per_chunk == 5

    def test_create_pipeline_returns_agent_pipeline(self):
        """create_pipeline() gibt AgentPipeline zurück."""
        from src.pipeline.agent_pipeline import create_pipeline, AgentPipeline
        pipeline = create_pipeline()
        assert isinstance(pipeline, AgentPipeline)
        assert pipeline.planner is not None
        assert pipeline.navigator is not None
        assert pipeline.verifier is not None

    def test_pipeline_imports_from_init(self):
        """__init__.py exportiert alle öffentlichen Klassen korrekt."""
        from src.pipeline import (
            AgentPipeline, PipelineResult, BatchProcessor,
            create_pipeline, create_full_pipeline,
            IngestionPipeline, IngestionConfig, IngestionMetrics,
            DocumentLoader, EmbeddingGenerator, create_ingestion_pipeline,
        )
        assert AgentPipeline is not None
        assert IngestionPipeline is not None
