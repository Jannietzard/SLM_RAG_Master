"""
graph_3d.py — Knowledge-Graph-Visualisierung für Masterarbeit

Erzeugt eine PNG-Grafik die für die Thesis geeignet ist:
  - Hub-Entitäten (Pronomen, generische Terme) werden gefiltert
  - Nur Entitäten mit tatsächlichen RELATED_TO-Kanten werden gezeigt
  - Farb-Codierung nach Entity-Typ (Person, Org, Location, etc.)
  - Knotengröße proportional zur Anzahl Verbindungen
  - Komplex genug um die Graphstruktur zu zeigen, nicht überfüllt

Ausgabe:
  test_system/graph_preview.png  — PNG für Thesis
  test_system/graph_preview.html — Interaktive Version zum Erkunden

Usage:
  python test_system/graph_3d.py
  python test_system/graph_3d.py --top 100      # mehr Knoten zeigen
  python test_system/graph_3d.py --no-html      # nur PNG
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import kuzu
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = Path(__file__).parent

# ─── Argumente ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--top",     type=int, default=80,
                    help="Maximale Anzahl Entitäten im Graphen (default: 80)")
parser.add_argument("--no-html", action="store_true",
                    help="Kein interaktives HTML erzeugen")
args = parser.parse_args()

# ─── Datenbankverbindung ──────────────────────────────────────────────────────
db   = kuzu.Database(str(PROJECT_ROOT / "data/hotpotqa/knowledge_graph"))
conn = kuzu.Connection(db)

# ─── Hub-Filter: Pronomen und generische Terme ausschließen ──────────────────
HUB_WORDS = {
    "i", "he", "she", "it", "they", "we", "you",
    "his", "her", "their", "him", "them", "who", "that",
    "this", "these", "those", "its", "one",
    "american", "united states", "us", "uk", "british",
    "country", "film", "movie", "people", "world", "city",
    "government", "man", "woman", "year", "time", "place",
    "football", "song", "album", "book", "company", "university",
    "music", "television", "television series", "series", "new",
}

def is_hub(name: str) -> bool:
    if not name or len(name.strip()) <= 2:
        return True
    return name.strip().lower() in HUB_WORDS

# ─── 1. RELATED_TO-Kanten laden (nur spezifische Named Entities) ─────────────
print("Lade RELATED_TO-Kanten ...")
r = conn.execute("""
    MATCH (e1:Entity)-[rel:RELATED_TO]->(e2:Entity)
    WHERE e1.type IN ['PERSON', 'ORGANIZATION', 'WORK_OF_ART', 'GPE', 'EVENT', 'LOC', 'FAC']
      AND e2.type IN ['PERSON', 'ORGANIZATION', 'WORK_OF_ART', 'GPE', 'EVENT', 'LOC', 'FAC']
    RETURN e1.name, e1.type, rel.relation_type, e2.name, e2.type
    LIMIT 2000
""")

raw_edges = []
while r.has_next():
    row = r.get_next()
    e1_name, e1_type, rel_type, e2_name, e2_type = row
    if not is_hub(e1_name) and not is_hub(e2_name):
        raw_edges.append((e1_name, e1_type, rel_type, e2_name, e2_type))

print(f"  Nach Hub-Filter: {len(raw_edges)} Kanten übrig")

# ─── 2. Top-N Entitäten nach Verbindungsgrad auswählen ───────────────────────
degree = Counter()
for e1, _, _, e2, _ in raw_edges:
    degree[e1] += 1
    degree[e2] += 1

top_entities = {e for e, _ in degree.most_common(args.top)}
print(f"  Top-{args.top} Entitäten ausgewählt (min. Grad: "
      f"{min(degree[e] for e in top_entities)})")

# Nur Kanten zwischen Top-Entitäten
edges = [(e1, t1, rt, e2, t2)
         for e1, t1, rt, e2, t2 in raw_edges
         if e1 in top_entities and e2 in top_entities]
print(f"  Kanten zwischen Top-{args.top}: {len(edges)}")

# ─── 3. NetworkX Graph aufbauen ──────────────────────────────────────────────
G = nx.DiGraph()
entity_types = {}

for e1, t1, rt, e2, t2 in edges:
    entity_types[e1] = t1
    entity_types[e2] = t2
    G.add_edge(e1, e2, relation=rt)

# Isolated nodes (aus top_entities die keine Kanten zu anderen top haben)
# nicht hinzufügen — sie würden das Bild unlesbar machen
print(f"  Graph: {G.number_of_nodes()} Knoten, {G.number_of_edges()} Kanten")

# ─── 4. Layout berechnen ─────────────────────────────────────────────────────
print("Berechne Layout (spring_layout) ...")
# k = idealer Knotenabstand, iterations = Qualität
pos = nx.spring_layout(G, k=2.0, iterations=80, seed=42)

# ─── 5. Farben und Größen ────────────────────────────────────────────────────
TYPE_COLORS = {
    "PERSON":       "#e74c3c",   # Rot
    "ORGANIZATION": "#3498db",   # Blau
    "GPE":          "#2ecc71",   # Grün (geopolitische Entität)
    "WORK_OF_ART":  "#f39c12",   # Orange
    "EVENT":        "#9b59b6",   # Lila
    "LOC":          "#1abc9c",   # Türkis
    "FAC":          "#e67e22",   # Dunkelorange
}
DEFAULT_COLOR = "#7f8c8d"

node_colors = [TYPE_COLORS.get(entity_types.get(n, ""), DEFAULT_COLOR) for n in G.nodes()]
# Knotengröße: Basisgröße + proportional zur Degree-Zentralität
node_sizes  = [max(150, 100 + 80 * G.degree(n)) for n in G.nodes()]

# ─── 6. Matplotlib-Abbildung ─────────────────────────────────────────────────
print("Zeichne Grafik ...")
fig, ax = plt.subplots(figsize=(18, 13), facecolor="#16213e")
ax.set_facecolor("#16213e")

# Kanten (Pfeile)
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edge_color="#5a6694",
    arrows=True,
    arrowsize=8,
    alpha=0.45,
    width=0.7,
    connectionstyle="arc3,rad=0.08",
    min_source_margin=12,
    min_target_margin=12,
)

# Knoten
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.92,
    linewidths=0.5,
    edgecolors="#ffffff40",
)

# Labels: nur für Knoten mit Grad >= 3 (zu viele Labels = Chaos)
min_label_degree = 3
labels = {n: n for n in G.nodes() if G.degree(n) >= min_label_degree}
nx.draw_networkx_labels(
    G, pos, labels=labels, ax=ax,
    font_color="white",
    font_size=6.5,
    font_weight="bold",
)

# ─── 7. Legende ──────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color="#e74c3c", label="Person"),
    mpatches.Patch(color="#3498db", label="Organization"),
    mpatches.Patch(color="#2ecc71", label="Location (GPE)"),
    mpatches.Patch(color="#f39c12", label="Work of Art"),
    mpatches.Patch(color="#9b59b6", label="Event"),
    mpatches.Patch(color="#1abc9c", label="Location (LOC)"),
    mpatches.Patch(color="#7f8c8d", label="Other"),
]
legend = ax.legend(
    handles=legend_patches,
    loc="upper left",
    facecolor="#1a1a3e",
    edgecolor="#5a6694",
    labelcolor="white",
    fontsize=9,
    framealpha=0.85,
    title="Entity Type",
    title_fontsize=9,
)
legend.get_title().set_color("white")

# ─── 8. Titel ────────────────────────────────────────────────────────────────
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
ax.set_title(
    f"Knowledge Graph — HotpotQA Corpus\n"
    f"{n_nodes} Named Entities · {n_edges} RELATED_TO Edges  "
    f"(Top-{args.top} by degree, hub entities filtered)",
    color="white",
    fontsize=11,
    pad=15,
)

# Stats-Box unten rechts
stats_text = (
    f"Total graph: 36,996 entities · 14,445 relations · 66,585 mentions\n"
    f"Extraction: GLiNER (urchade/gliner_small-v2.1) + REBEL"
)
ax.text(
    0.99, 0.01, stats_text,
    transform=ax.transAxes,
    fontsize=7.5,
    color="#aaaacc",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a3e", alpha=0.7, edgecolor="#5a6694"),
)

ax.axis("off")
plt.tight_layout(pad=1.5)

# ─── 9. PNG speichern ────────────────────────────────────────────────────────
output_png = OUTPUT_DIR / "graph_preview.png"
plt.savefig(str(output_png), dpi=200, bbox_inches="tight", facecolor="#16213e")
print(f"\nPNG gespeichert: {output_png}")
print(f"  Empfehlung fuer Thesis: dpi=300 fuer Druckqualitaet")

plt.show()

# ─── 10. Interaktives HTML (optional) ────────────────────────────────────────
if not args.no_html:
    try:
        from pyvis.network import Network

        net = Network(height="900px", width="100%", bgcolor="#16213e",
                      font_color="white", directed=True)
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

        # Knoten
        for node in G.nodes():
            etype = entity_types.get(node, "")
            color = TYPE_COLORS.get(etype, DEFAULT_COLOR)
            size  = 12 + 3 * G.degree(node)
            net.add_node(node, label=node, color=color, size=size,
                         title=f"{node} ({etype})")

        # Kanten
        for e1, e2, data in G.edges(data=True):
            net.add_edge(e1, e2,
                         title=data.get("relation", ""),
                         color="#5a6694", width=1.2)

        output_html = OUTPUT_DIR / "graph_preview.html"
        net.show(str(output_html), notebook=False)
        print(f"HTML gespeichert: {output_html}")
    except ImportError:
        print("  (pyvis nicht installiert - nur PNG erzeugt)")
