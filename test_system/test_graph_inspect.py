"""
Graph Inspektion: Was landet wirklich im KuzuDB nach Ingestion?
Testet Entity-Nodes, MENTIONS- und RELATED_TO-Kanten direkt in der DB.
"""
import sys
from pathlib import Path
# Projektverzeichnis zu sys.path hinzufügen (damit src.* Imports funktionieren)
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, warnings, logging, tempfile, shutil
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import numpy as np
from langchain.schema import Document
from src.data_layer.storage import HybridStore, StorageConfig

# ── Beispieltexte (typisch HotpotQA Bridge-Fragen) ────────────────────────
DOCS = [
    Document(
        page_content=(
            "Scott Derrickson is an American filmmaker. "
            "He directed Doctor Strange for Marvel Studios in 2016."
        ),
        metadata={"source_file": "wiki_derrickson.txt", "chunk_id": "c1", "chunk_index": 0, "page_number": 1},
    ),
    Document(
        page_content=(
            "Ed Wood was an American filmmaker and director. "
            "He directed Plan 9 from Outer Space in Hollywood."
        ),
        metadata={"source_file": "wiki_edwood.txt", "chunk_id": "c2", "chunk_index": 0, "page_number": 1},
    ),
    Document(
        page_content=(
            "Tom Hanks starred in Bridge of Spies, directed by Steven Spielberg. "
            "The film was set in Berlin during the Cold War."
        ),
        metadata={"source_file": "wiki_hanks.txt", "chunk_id": "c3", "chunk_index": 0, "page_number": 1},
    ),
    Document(
        page_content=(
            "The Eiffel Tower is located in Paris, France. "
            "It was designed by Gustave Eiffel and completed in 1889."
        ),
        metadata={"source_file": "wiki_eiffel.txt", "chunk_id": "c4", "chunk_index": 0, "page_number": 1},
    ),
]

BRIDGE_QUERIES = [
    {
        "question": "Both Scott Derrickson and Ed Wood are _ filmmakers.",
        "bridge_entity": "American",
        "chunk_ids": ["c1", "c2"],
    },
    {
        "question": "The Eiffel Tower was designed by _ and is in Paris.",
        "bridge_entity": "Gustave Eiffel",
        "chunk_ids": ["c4"],
    },
    {
        "question": "Tom Hanks starred in a film directed by _.",
        "bridge_entity": "Steven Spielberg",
        "chunk_ids": ["c3"],
    },
]


class MockEmbeddings:
    """Dummy-Embeddings — nur für den Graphen relevant, nicht für Qualität."""
    def embed_documents(self, texts):
        return [np.random.randn(768).tolist() for _ in texts]
    def embed_query(self, text):
        return np.random.randn(768).tolist()


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


tmp = tempfile.mkdtemp(prefix="graph_inspect_")
try:
    cfg = StorageConfig(
        vector_db_path=f"{tmp}/vec",
        graph_db_path=f"{tmp}/graph",
        embedding_dim=768,
        enable_entity_extraction=True,
    )

    print("Lade Modelle und ingestiere Texte...")
    store = HybridStore(cfg, MockEmbeddings())
    store.add_documents(DOCS)
    print("Ingestion abgeschlossen.")

    conn = store.graph_store.conn

    section("1. ENTITY NODES im Graphen (mit Confidence)")
    res = conn.execute("MATCH (e:Entity) RETURN e.name, e.type, e.confidence ORDER BY e.type, e.name")
    entities_in_graph = []
    while res.has_next():
        row = res.get_next()
        conf = row[2] if row[2] is not None else 0.0
        entities_in_graph.append((row[0], row[1]))
        print(f"  {row[1]:15s}  {row[0]:30s}  conf={conf:.2f}")
    if not entities_in_graph:
        print("  (keine Entity-Nodes gefunden!)")

    section("2. MENTIONS  (Chunk -> Entity)")
    res = conn.execute(
        "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
        "RETURN c.chunk_id, e.name, e.type ORDER BY c.chunk_id"
    )
    mentions = []
    while res.has_next():
        row = res.get_next()
        mentions.append((row[0], row[1], row[2]))
        print(f"  [{row[0]}]  mentions  '{row[1]}'  ({row[2]})")
    if not mentions:
        print("  (keine MENTIONS-Kanten gefunden!)")

    section("3. RELATED_TO  (Entity -> Entity, via REBEL)")
    res = conn.execute(
        "MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity) "
        "RETURN e1.name, r.relation_type, e2.name"
    )
    relations = []
    while res.has_next():
        row = res.get_next()
        relations.append((row[0], row[1], row[2]))
        print(f"  '{row[0]}'  --[{row[1]}]-->  '{row[2]}'")
    if not relations:
        print("  (keine RELATED_TO-Kanten gefunden!)")

    section("4a. MULTI-HOP TEST: graph_search() mit 2-Hop Traversal")
    multi_hop_queries = [
        ("Doctor Strange", ["c1"]),
        ("Tom Hanks", ["c3"]),
        ("Gustave Eiffel", ["c4"]),
    ]
    for query_entity, expected_direct in multi_hop_queries:
        results = store.graph_search(entities=[query_entity], max_hops=2, top_k=10)
        direct = [r for r in results if r["hops"] == 0]
        via_hop = [r for r in results if r["hops"] == 2]
        print(f"\n  Query: '{query_entity}'")
        print(f"    1-Hop direkt:  {[r['chunk_id'] for r in direct]}")
        if via_hop:
            for r in via_hop:
                print(f"    2-Hop via '{r['bridge_entity']}' [{r['relation_type']}]: chunk={r['chunk_id']}")
        else:
            print(f"    2-Hop: (keine)")

    section("4b. BRIDGE-TEST: Chunks ueber gemeinsame Entity verbunden?")
    for bq in BRIDGE_QUERIES:
        entity_name = bq["bridge_entity"]
        expected_chunks = set(bq["chunk_ids"])
        res = conn.execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
            "WHERE e.name = $name "
            "RETURN c.chunk_id",
            {"name": entity_name},
        )
        found_chunks = set()
        while res.has_next():
            found_chunks.add(res.get_next()[0])
        ok = expected_chunks.issubset(found_chunks)
        status = "OK " if ok else "---"
        print(f"\n  [{status}] '{bq['question']}'")
        print(f"         Bridge-Entity: '{entity_name}'")
        print(f"         Erwartet in:   {sorted(expected_chunks)}")
        print(f"         Gefunden in:   {sorted(found_chunks)}")
        if not ok:
            missing = expected_chunks - found_chunks
            print(f"         FEHLT in:      {sorted(missing)}")

    section("5. ZUSAMMENFASSUNG")
    stats = store.graph_store.get_statistics()
    print(f"  DocumentChunks:  {stats.get('document_chunks', 0)}")
    print(f"  Entity-Nodes:    {len(entities_in_graph)}")
    print(f"  MENTIONS-Kanten: {len(mentions)}")
    print(f"  RELATED_TO:      {len(relations)}")
    bridge_ok = sum(
        1 for bq in BRIDGE_QUERIES
        if set(bq["chunk_ids"]).issubset(
            {r[0] for r in mentions if r[1] == bq["bridge_entity"]}
        )
    )
    print(f"  Bridge-Tests:    {bridge_ok}/{len(BRIDGE_QUERIES)} bestanden")

finally:
    shutil.rmtree(tmp, ignore_errors=True)
