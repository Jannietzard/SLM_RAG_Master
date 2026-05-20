import kuzu
db = kuzu.Database("data/hotpotqa/graph/graph_KuzuDB")
conn = kuzu.Connection(db)

print("=== Schema (tables actually present) ===")
r = conn.execute("CALL show_tables() RETURN *")
while r.has_next():
    print(r.get_next())

print()
print("=== Entities matching 'ed wood' ===")
r = conn.execute(
    "MATCH (e:Entity) "
    "WHERE toLower(e.name) CONTAINS 'ed wood' "
    "RETURN e.name, e.type "
    "LIMIT 20"
)
count = 0
while r.has_next():
    print(r.get_next())
    count += 1
if count == 0:
    print("  (no entities matched)")

print()
print("=== Chunks mentioning any 'ed wood' entity (with source_doc) ===")
r = conn.execute(
    "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
    "WHERE toLower(e.name) CONTAINS 'ed wood' "
    "RETURN e.name, e.type, c.source_file, c.chunk_id "
    "LIMIT 30"
)
count = 0
while r.has_next():
    print(r.get_next())
    count += 1
if count == 0:
    print("  (no chunks matched)")

print()
print("=== For comparison: chunks mentioning Scott Derrickson ===")
r = conn.execute(
    "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
    "WHERE toLower(e.name) CONTAINS 'scott derrickson' "
    "RETURN e.name, e.type, c.source_file "
    "LIMIT 10"
)
count = 0
while r.has_next():
    print(r.get_next())
    count += 1
if count == 0:
    print("  (no chunks matched)")
