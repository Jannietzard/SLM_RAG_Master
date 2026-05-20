import kuzu
db = kuzu.Database("data/hotpotqa/graph/graph_KuzuDB")
conn = kuzu.Connection(db)

print("=== Does the 'hotpotqa_Ed Wood' source even exist? ===")
r = conn.execute(
    "MATCH (c:DocumentChunk) "
    "WHERE c.source_file = 'hotpotqa_Ed Wood' "
    "RETURN c.chunk_id, c.source_file"
)
chunks_found = []
while r.has_next():
    row = r.get_next()
    print(row)
    chunks_found.append(row[0])

if not chunks_found:
    print("  (no chunks found for hotpotqa_Ed Wood — checking similar source_files)")
    r = conn.execute(
        "MATCH (c:DocumentChunk) "
        "WHERE toLower(c.source_file) CONTAINS 'ed wood' "
        "RETURN DISTINCT c.source_file"
    )
    while r.has_next():
        print("  available:", r.get_next())

print()
print("=== What entities ARE extracted from those chunks? ===")
for cid in chunks_found:
    print(f"\n--- chunk {cid} entities ---")
    r = conn.execute(
        f"MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
        f"WHERE c.chunk_id = '{cid}' "
        f"RETURN e.name, e.type "
        f"LIMIT 30"
    )
    n = 0
    while r.has_next():
        print(" ", r.get_next())
        n += 1
    if n == 0:
        print("  (NO entities mentioned from this chunk — extraction failed on it)")

print()
print("=== Raw chunk text (first 600 chars) ===")
for cid in chunks_found[:2]:
    r = conn.execute(
        f"MATCH (c:DocumentChunk) "
        f"WHERE c.chunk_id = '{cid}' "
        f"RETURN c.text"
    )
    if r.has_next():
        text = r.get_next()[0] or ""
        print(f"\n--- chunk {cid} ---")
        print(text[:600])
