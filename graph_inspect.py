import kuzu

db = kuzu.Database("./data/hotpotqa/knowledge_graph")
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
