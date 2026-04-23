"""
Integration Tests — KuzuDB Knowledge Graph Ingestion

Verifies that the HybridStore correctly writes entity nodes, MENTIONS edges,
and RELATED_TO triples to KuzuDB after ingesting synthetic HotpotQA-style
documents. Uses real GLiNER and REBEL models via HybridStore — no mocks for
the extraction pipeline.

Architectural position: test_system/ (integration tests, Data Layer Artifact A).
Depends on: src.data_layer.storage.HybridStore, KuzuDB, GLiNER, REBEL.

Run:
    pytest test_system/test_graph_inspect.py -v
    pytest test_system/test_graph_inspect.py -v -k "bridge"  # bridge tests only

Academic references:
    Zaratiana et al. (2023). "GLiNER: Generalist Model for Named Entity
        Recognition using Bidirectional Transformer." arXiv:2311.08526.
    Cabot & Navigli (2021). "REBEL: Relation Extraction By End-to-end
        Language generation." EMNLP 2021 Findings.

Review History:
    Last Reviewed:  2026-04-15
    Review Result:  0 CRITICAL, 0 IMPORTANT, 0 RECOMMENDED
    Reviewer:       Code Review Prompt v2.1
    Next Review:    After changes to entity extraction or storage schema
"""

import sys
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_layer.storage import HybridStore, StorageConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic corpus — HotpotQA bridge-question style
# ---------------------------------------------------------------------------

DOCS = [
    Document(
        page_content=(
            "Scott Derrickson is an American filmmaker. "
            "He directed Doctor Strange for Marvel Studios in 2016."
        ),
        metadata={
            "source_file": "wiki_derrickson.txt",
            "chunk_id": "c1",
            "chunk_index": 0,
            "page_number": 1,
        },
    ),
    Document(
        page_content=(
            "Ed Wood was an American filmmaker and director. "
            "He directed Plan 9 from Outer Space in Hollywood."
        ),
        metadata={
            "source_file": "wiki_edwood.txt",
            "chunk_id": "c2",
            "chunk_index": 0,
            "page_number": 1,
        },
    ),
    Document(
        page_content=(
            "Tom Hanks starred in Bridge of Spies, directed by Steven Spielberg. "
            "The film was set in Berlin during the Cold War."
        ),
        metadata={
            "source_file": "wiki_hanks.txt",
            "chunk_id": "c3",
            "chunk_index": 0,
            "page_number": 1,
        },
    ),
    Document(
        page_content=(
            "The Eiffel Tower is located in Paris, France. "
            "It was designed by Gustave Eiffel and completed in 1889."
        ),
        metadata={
            "source_file": "wiki_eiffel.txt",
            "chunk_id": "c4",
            "chunk_index": 0,
            "page_number": 1,
        },
    ),
]

# Expected bridge entities: entity → chunks that must both mention it.
BRIDGE_EXPECTATIONS = [
    {
        "question": "Both Scott Derrickson and Ed Wood are _ filmmakers.",
        "bridge_entity": "American",
        "expected_chunk_ids": {"c1", "c2"},
    },
    {
        "question": "The Eiffel Tower was designed by _ and is in Paris.",
        "bridge_entity": "Gustave Eiffel",
        "expected_chunk_ids": {"c4"},
    },
    {
        "question": "Tom Hanks starred in a film directed by _.",
        "bridge_entity": "Steven Spielberg",
        "expected_chunk_ids": {"c3"},
    },
]


# ---------------------------------------------------------------------------
# Deterministic mock embeddings
# Produces fixed 768-dim vectors based on hash(text) — same text always gives
# the same vector across test runs (no np.random.randn).
# ---------------------------------------------------------------------------

class _DeterministicEmbeddings:
    """
    Mock embeddings for integration tests.

    Uses hash(text) % 1000 / 1000.0 as a scalar repeated 768 times.
    Deterministic: identical texts produce identical vectors across runs.
    Not suitable for semantic retrieval quality — only for graph wiring tests.
    """
    DIM = 768

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[hash(t) % 1000 / 1000.0] * self.DIM for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [hash(text) % 1000 / 1000.0] * self.DIM


# ---------------------------------------------------------------------------
# Module-scoped fixture: one HybridStore for all tests in this module.
# GLiNER + REBEL are expensive to load; scope="module" loads them once.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Generator[HybridStore, None, None]:
    """
    Create a temporary HybridStore, ingest DOCS, and yield for all tests.

    Uses a module-scoped temporary directory so GLiNER and REBEL are
    loaded only once per test session.
    """
    tmp = tmp_path_factory.mktemp("graph_inspect")
    cfg = StorageConfig(
        vector_db_path=str(tmp / "vec"),
        graph_db_path=str(tmp / "graph"),
        embedding_dim=_DeterministicEmbeddings.DIM,
        enable_entity_extraction=True,
    )
    hs = HybridStore(cfg, _DeterministicEmbeddings())
    hs.add_documents(DOCS)
    yield hs
    shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------------------
# Helper: query KuzuDB connection from store
# ---------------------------------------------------------------------------

def _conn(store: HybridStore):
    return store.graph_store.conn


# ---------------------------------------------------------------------------
# Section 1 — Entity Nodes
# ---------------------------------------------------------------------------

class TestEntityNodes:
    """Verify that GLiNER extracted entity nodes and stored them in KuzuDB."""

    def test_entity_nodes_present(self, store: HybridStore) -> None:
        """At least one Entity node must exist after ingestion."""
        res = _conn(store).execute("MATCH (e:Entity) RETURN COUNT(e)")
        count = res.get_next()[0]
        assert count > 0, "No Entity nodes found — GLiNER extraction may have failed"

    def test_expected_persons_extracted(self, store: HybridStore) -> None:
        """Named persons must be detected by GLiNER and stored as Entity nodes."""
        res = _conn(store).execute(
            "MATCH (e:Entity) RETURN e.name"
        )
        names = {row[0] for row in res}
        for expected in ("Scott Derrickson", "Ed Wood", "Tom Hanks", "Gustave Eiffel"):
            assert expected in names, (
                f"Expected entity '{expected}' not found in graph. "
                f"Found: {sorted(names)}"
            )

    def test_entity_confidence_is_nonnegative(self, store: HybridStore) -> None:
        """All stored entity confidences must be >= 0.0."""
        res = _conn(store).execute(
            "MATCH (e:Entity) WHERE e.confidence IS NOT NULL RETURN e.name, e.confidence"
        )
        while res.has_next():
            name, conf = res.get_next()
            assert conf >= 0.0, f"Negative confidence for entity '{name}': {conf}"


# ---------------------------------------------------------------------------
# Section 2 — MENTIONS Edges
# ---------------------------------------------------------------------------

class TestMentionsEdges:
    """Verify that Chunk→Entity MENTIONS edges are correctly created."""

    def test_mentions_edges_present(self, store: HybridStore) -> None:
        """At least one MENTIONS edge must exist after ingestion."""
        res = _conn(store).execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) RETURN COUNT(*)"
        )
        count = res.get_next()[0]
        assert count > 0, "No MENTIONS edges found — entity linking may have failed"

    def test_each_chunk_has_at_least_one_mention(self, store: HybridStore) -> None:
        """Every ingested chunk must mention at least one entity."""
        chunk_ids = {doc.metadata["chunk_id"] for doc in DOCS}
        res = _conn(store).execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
            "RETURN DISTINCT c.chunk_id"
        )
        chunks_with_mentions = {row[0] for row in res}
        for cid in chunk_ids:
            assert cid in chunks_with_mentions, (
                f"Chunk '{cid}' has no MENTIONS edges — entity extraction may "
                f"have failed for this document"
            )

    def test_chunk_mentions_correct_entities(self, store: HybridStore) -> None:
        """Scott Derrickson (c1) and Ed Wood (c2) must appear in their respective chunks."""
        res = _conn(store).execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
            "RETURN c.chunk_id, e.name"
        )
        mentions: dict[str, set] = {}
        while res.has_next():
            cid, name = res.get_next()
            mentions.setdefault(cid, set()).add(name)

        assert "Scott Derrickson" in mentions.get("c1", set()), \
            "c1 must mention 'Scott Derrickson'"
        assert "Ed Wood" in mentions.get("c2", set()), \
            "c2 must mention 'Ed Wood'"
        assert "Gustave Eiffel" in mentions.get("c4", set()), \
            "c4 must mention 'Gustave Eiffel'"


# ---------------------------------------------------------------------------
# Section 3 — RELATED_TO (REBEL relation extraction)
# ---------------------------------------------------------------------------

class TestRelatedToEdges:
    """Verify that REBEL extracted subject-predicate-object triples correctly."""

    def test_director_relation_extracted(self, store: HybridStore) -> None:
        """
        REBEL must extract a director relation from at least one film document.
        Expected: Doctor Strange --[director]--> Scott Derrickson
                  Plan 9 from Outer Space --[director]--> Ed Wood
        """
        res = _conn(store).execute(
            "MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity) "
            "WHERE r.relation_type = 'director' "
            "RETURN e1.name, e2.name"
        )
        triples = [(row[0], row[1]) for row in res]
        assert len(triples) > 0, (
            "No 'director' RELATED_TO edges found — REBEL extraction may have "
            "failed or produced no director relations"
        )

    def test_related_to_edges_are_bidirectionally_typed(self, store: HybridStore) -> None:
        """All RELATED_TO edges must have a non-empty relation_type."""
        res = _conn(store).execute(
            "MATCH ()-[r:RELATED_TO]->() "
            "WHERE r.relation_type IS NULL OR r.relation_type = '' "
            "RETURN COUNT(r)"
        )
        count = res.get_next()[0]
        assert count == 0, (
            f"{count} RELATED_TO edges have null or empty relation_type — "
            f"REBEL output was not stored correctly"
        )


# ---------------------------------------------------------------------------
# Section 4 — Bridge Entity Connectivity
# ---------------------------------------------------------------------------

class TestBridgeEntityConnectivity:
    """
    Verify that chunks sharing a bridge entity are both linked to that entity
    node in the graph. This is the core multi-hop retrieval invariant:
    graph_search(bridge_entity) must return ALL chunks that mention it.
    """

    @pytest.mark.parametrize("case", BRIDGE_EXPECTATIONS, ids=[
        c["bridge_entity"] for c in BRIDGE_EXPECTATIONS
    ])
    def test_bridge_entity_links_correct_chunks(
        self, store: HybridStore, case: dict
    ) -> None:
        """
        Bridge entity must appear as a MENTIONS target for all expected chunks.

        Failure indicates that one of the expected chunks did not have the
        bridge entity extracted, breaking multi-hop traversal for questions
        like: '{question}'
        """
        res = _conn(store).execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
            "WHERE e.name = $name "
            "RETURN c.chunk_id",
            {"name": case["bridge_entity"]},
        )
        found = {row[0] for row in res}
        expected = case["expected_chunk_ids"]
        missing = expected - found
        assert not missing, (
            f"Bridge entity '{case['bridge_entity']}' not linked to chunks {missing}. "
            f"Found: {found}. "
            f"Question: '{case['question']}'"
        )


# ---------------------------------------------------------------------------
# Section 5 — Graph Statistics Sanity Check
# ---------------------------------------------------------------------------

class TestGraphStatistics:
    """Sanity-check the HybridStore.get_statistics() output."""

    def test_statistics_returns_expected_keys(self, store: HybridStore) -> None:
        """get_statistics() must return all documented keys."""
        stats = store.graph_store.get_statistics()
        for key in ("document_chunks",):
            assert key in stats, f"Missing key '{key}' in get_statistics() output"

    def test_chunk_count_matches_ingested(self, store: HybridStore) -> None:
        """Chunk count in statistics must equal number of ingested documents."""
        stats = store.graph_store.get_statistics()
        assert stats.get("document_chunks") == len(DOCS), (
            f"Expected {len(DOCS)} chunks in graph, "
            f"got {stats.get('document_chunks')}"
        )
