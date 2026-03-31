"""
Qualitätsanalyse der GLiNER Entity Extraction.
Zeigt: Was wird extrahiert, was fehlt, was ist Rauschen?
"""
import sys
from pathlib import Path
# Projektverzeichnis zu sys.path hinzufügen (damit src.* Imports funktionieren)
sys.path.insert(0, str(Path(__file__).parent.parent))

import time, warnings, logging, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from src.data_layer.entity_extraction import EntityExtractionPipeline, ExtractionConfig

# ── Aktueller Config (wie er jetzt ist) ───────────────────────────────────────
config_current = ExtractionConfig(
    ner_confidence_threshold=0.5,
    re_confidence_threshold=0.7,
    selective_re=True,
    entity_types=[
        "PERSON", "ORGANIZATION", "GPE", "LOCATION", "PLACE",
        "DATE", "EVENT", "CONCEPT", "WORK_OF_ART", "STRUCTURE",
        "NATURAL_OBJECT", "LAW", "TECHNOLOGY", "SCIENTIFIC_TERM", "PRODUCT"
    ],
)

# ── Verbesserter Config ───────────────────────────────────────────────────────
config_improved = ExtractionConfig(
    ner_confidence_threshold=0.35,   # niedriger: Paris/France werden erkannt
    re_confidence_threshold=0.6,
    selective_re=True,
    entity_types=[                   # nur nützliche Typen für HotpotQA
        "PERSON", "ORGANIZATION", "GPE", "LOCATION",
        "WORK_OF_ART", "EVENT", "PRODUCT"
    ],
)

# ── Test-Chunks (typisch für HotpotQA) ────────────────────────────────────────
chunks = [
    # Bridge of Spies Beispiel aus deiner Thesis-Folie
    "Tom Hanks starred as James Donovan, a lawyer, in Bridge of Spies (2015), "
    "a historical thriller directed by Steven Spielberg. The film is set in Berlin "
    "and New York during the Cold War.",

    # Nationalitäts-Vergleich (wie in HotpotQA Comparison Queries)
    "Scott Derrickson is an American filmmaker. He directed Sinister (2012) and "
    "Doctor Strange (2016) for Marvel Studios.",

    "Ed Wood was an American filmmaker, actor and director. Born in Poughkeepsie, "
    "New York, he moved to Hollywood and directed Plan 9 from Outer Space (1957).",
]
ids = ['tom_hanks_chunk', 'derrickson_chunk', 'edwood_chunk']

def run_test(label, config):
    pipe = EntityExtractionPipeline(config)
    t0 = time.time()
    results = pipe.process_chunks_batch(chunks, ids)
    ms = (time.time()-t0)*1000
    print(f"\n{'='*70}")
    print(f"  {label}  ({ms:.0f}ms total)")
    print(f"{'='*70}")
    for r in results:
        print(f"\n[{r.chunk_id}]")
        if r.entities:
            for e in r.entities:
                marker = "+" if e.entity_type in ("PERSON","ORGANIZATION","GPE","LOCATION","WORK_OF_ART","EVENT") else "~"
                print(f"  {marker} {repr(e.name):35s} {e.entity_type:15s} {e.confidence:.2f}")
        else:
            print("  (keine Entities)")
        if r.relations:
            print("  RELATIONS:")
            for rel in r.relations:
                print(f"    ({repr(rel.subject_entity)}) --[{rel.relation_type}]--> ({repr(rel.object_entity)})")


print("Initialisiere Pipelines...")
run_test("AKTUELLER CONFIG (threshold=0.5, 15 Typen)", config_current)
run_test("VERBESSERTER CONFIG (threshold=0.35, 7 Typen)", config_improved)
