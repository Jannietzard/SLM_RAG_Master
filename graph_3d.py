import kuzu
from pyvis.network import Network

db = kuzu.Database("./data/hotpotqa/knowledge_graph")
conn = kuzu.Connection(db)

net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
net.barnes_hut()

# Chunks -> Entities (MENTIONS)
r = conn.execute("""
    MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
    RETURN c.chunk_id, e.name, e.type
    LIMIT 200
""")
while r.has_next():
    row = r.get_next()
    chunk_id, ent_name, ent_type = row
    net.add_node(chunk_id, label=chunk_id[-8:], color="#4477ff", size=10)
    net.add_node(ent_name, label=ent_name, color="#ff7744", size=15)
    net.add_edge(chunk_id, ent_name, title="MENTIONS")

# Entities -> Entities (RELATED_TO)
r = conn.execute("""
    MATCH (e1:Entity)-[rel:RELATED_TO]->(e2:Entity)
    RETURN e1.name, rel.relation_type, e2.name
    LIMIT 100
""")
while r.has_next():
    e1, rel_type, e2 = r.get_next()
    net.add_node(e1, color="#ff7744")
    net.add_node(e2, color="#ff7744")
    net.add_edge(e1, e2, title=rel_type, color="#aaffaa")

net.show("graph_preview.html", notebook=False)
print("-> graph_preview.html im Browser oeffnen")
