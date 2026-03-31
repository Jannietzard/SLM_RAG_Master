"""
NER Qualitätstest: 10 typische HotpotQA-Sätze.
Zeigt was extrahiert wird vs. was erwartet wird.
"""
import sys
from pathlib import Path
# Projektverzeichnis zu sys.path hinzufügen (damit src.* Imports funktionieren)
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, warnings, logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from src.data_layer.entity_extraction import EntityExtractionPipeline, ExtractionConfig

TESTS = [
    {
        "text": "Scott Derrickson is an American filmmaker. He directed Doctor Strange for Marvel Studios.",
        "expect": {"Scott Derrickson": "PERSON", "Doctor Strange": "WORK_OF_ART", "Marvel Studios": "ORGANIZATION"},
    },
    {
        "text": "Ed Wood was an American filmmaker. He directed Plan 9 from Outer Space in Hollywood.",
        "expect": {"Ed Wood": "PERSON", "Plan 9 from Outer Space": "WORK_OF_ART", "Hollywood": "GPE"},
    },
    {
        "text": "Tom Hanks starred in Bridge of Spies, directed by Steven Spielberg in 2015.",
        "expect": {"Tom Hanks": "PERSON", "Bridge of Spies": "WORK_OF_ART", "Steven Spielberg": "PERSON"},
    },
    {
        "text": "The Eiffel Tower is located in Paris, France. It was built by Gustave Eiffel.",
        "expect": {"Eiffel Tower": "WORK_OF_ART", "Paris": "GPE", "France": "GPE", "Gustave Eiffel": "PERSON"},
    },
    {
        "text": "Marie Curie was a Polish physicist. She won the Nobel Prize in Physics in 1903.",
        "expect": {"Marie Curie": "PERSON", "Nobel Prize in Physics": "ORGANIZATION"},
    },
    {
        "text": "Christopher Nolan directed Inception and The Dark Knight for Warner Bros.",
        "expect": {"Christopher Nolan": "PERSON", "Inception": "WORK_OF_ART", "The Dark Knight": "WORK_OF_ART", "Warner Bros.": "ORGANIZATION"},
    },
    {
        "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "expect": {"Apple Inc.": "ORGANIZATION", "Steve Jobs": "PERSON", "Cupertino": "GPE", "California": "GPE"},
    },
    {
        "text": "The Cold War was a geopolitical conflict between the United States and the Soviet Union.",
        "expect": {"Cold War": "EVENT", "United States": "GPE", "Soviet Union": "GPE"},
    },
    {
        "text": "Barack Obama was the 44th President of the United States, born in Hawaii.",
        "expect": {"Barack Obama": "PERSON", "United States": "GPE", "Hawaii": "GPE"},
    },
    {
        "text": "The Beatles released Abbey Road in 1969. The band was formed in Liverpool.",
        "expect": {"The Beatles": "ORGANIZATION", "Abbey Road": "WORK_OF_ART", "Liverpool": "GPE"},
    },
]

config = ExtractionConfig(cache_enabled=False)
pipe = EntityExtractionPipeline(config)

texts = [t["text"] for t in TESTS]
ids   = [f"s{i+1}" for i in range(len(TESTS))]

print("Extrahiere...\n")
results = pipe.process_chunks_batch(texts, ids)

total_expected = 0
total_found    = 0
total_correct  = 0

for i, (r, test) in enumerate(zip(results, TESTS)):
    extracted = {e.name: e.entity_type for e in r.entities}
    expected  = test["expect"]

    hits   = [k for k in expected if k in extracted and extracted[k] == expected[k]]
    missed = [k for k in expected if k not in extracted]
    wrong  = [f"{k}->{extracted[k]}" for k in expected if k in extracted and extracted[k] != expected[k]]
    noise  = [k for k in extracted if k not in expected]

    total_expected += len(expected)
    total_found    += len(extracted)
    total_correct  += len(hits)

    status = "OK " if len(hits) == len(expected) else "---"
    print(f"[{status}] S{i+1}: {test['text'][:65]}...")
    if hits:
        print(f"      + Korrekt:  {hits}")
    if missed:
        print(f"      - Fehlt:    {missed}")
    if wrong:
        print(f"      ! Falsch:   {wrong}")
    if noise:
        print(f"      ~ Rauschen: {noise}")
    print()

print("=" * 60)
print(f"  Erwartet:  {total_expected} Entities")
print(f"  Korrekt:   {total_correct}/{total_expected}  ({100*total_correct//total_expected}%)")
print(f"  Extrahiert insgesamt: {total_found}")
print("=" * 60)
