import sys
from pathlib import Path
# Projektverzeichnis zu sys.path hinzufügen (für konsistente Umgebung)
sys.path.insert(0, str(Path(__file__).parent.parent))

import kuzu

PROJECT_ROOT = Path(__file__).parent.parent
db = kuzu.Database(str(PROJECT_ROOT / "data/hotpotqa/graph"))
conn = kuzu.Connection(db)

# Knotenanzahl
for table in ["DocumentChunk", "SourceDocument", "Entity"]:
    r = conn.execute(f"MATCH (n:{table}) RETURN COUNT(n)")
    print(f"{table}: {r.get_next()[0]}")

# Kantenanzahl
for rel in ["FROM_SOURCE", "NEXT_CHUNK", "MENTIONS", "RELATED_TO"]:
    r = conn.execute(f"MATCH ()-[r:{rel}]->() RETURN COUNT(r)")
    print(f"{rel}: {r.get_next()[0]}")

# Beispiel: Entities eines Chunks
print("\nBeispiel MENTIONS:")
r = conn.execute("""
    MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
    RETURN c.chunk_id, e.name, e.type
    LIMIT 10
""")
while r.has_next():
    print(r.get_next())
