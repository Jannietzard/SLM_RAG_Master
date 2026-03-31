"""
Teste welche Entity-Type-Namen GLiNER am besten funktionieren.
GLiNER ist zero-shot: der Typname beeinflusst direkt die Erkennung.
"""
import sys
from pathlib import Path
# Projektverzeichnis zu sys.path hinzufügen (für konsistente Umgebung)
sys.path.insert(0, str(Path(__file__).parent.parent))

import time, warnings, logging, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from gliner import GLiNER

print("Lade GLiNER...")
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
print("Geladen.\n")

text = ("Tom Hanks starred as James Donovan in Bridge of Spies (2015), "
        "a historical thriller directed by Steven Spielberg. "
        "The film is set in Berlin and New York during the Cold War.")

text2 = ("Scott Derrickson is an American filmmaker. "
         "He directed Sinister (2012) and Doctor Strange (2016) for Marvel Studios.")

# ── Test 1: Uppercase types (aktuell) ─────────────────────────────────────
types_current = ["PERSON", "ORGANIZATION", "GPE", "LOCATION", "WORK_OF_ART", "EVENT"]
# ── Test 2: Lowercase natural language (besser für zero-shot) ──────────────
types_natural = ["person", "organization", "city", "country", "film", "movie", "event"]
# ── Test 3: Mix ────────────────────────────────────────────────────────────
types_mix = ["person", "organization", "city", "country", "film", "movie", "event",
             "director", "actor"]

def show(label, entities):
    print(f"\n  [{label}]")
    if not entities:
        print("    (nichts)")
        return
    for e in entities:
        print(f"    {e['text']:30s}  {e['label']:15s}  {e['score']:.2f}")

for threshold in [0.15, 0.30]:
    print(f"\n{'='*65}")
    print(f"  THRESHOLD = {threshold}")
    print(f"{'='*65}")
    for txt_label, txt in [("Bridge of Spies", text), ("Derrickson", text2)]:
        print(f"\n  TEXT: {txt_label}")
        for type_label, types in [
            ("UPPERCASE", types_current),
            ("lowercase", types_natural),
            ("mix+roles", types_mix),
        ]:
            ents = model.predict_entities(txt, types, threshold=threshold)
            show(f"{type_label} @ {threshold}", ents)
