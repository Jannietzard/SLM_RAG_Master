"""
NER Qualitätstest: 20 typische HotpotQA-Sätze.
Zeigt was extrahiert wird vs. was erwartet wird.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os, warnings, logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from src.data_layer.entity_extraction import create_extraction_pipeline

_settings_path = PROJECT_ROOT / "config" / "settings.yaml"

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
        "expect": {"Eiffel Tower": "LOCATION", "Paris": "GPE", "France": "GPE", "Gustave Eiffel": "PERSON"},
    },
    {
        "text": "Marie Curie was a Polish physicist. She won the Nobel Prize in Physics in 1903.",
        "expect": {"Marie Curie": "PERSON", "Nobel Prize in Physics": "WORK_OF_ART"},
    },
    {
        "text": "Christopher Nolan directed Inception and The Dark Knight for Warner Bros.",
        "expect": {"Christopher Nolan": "PERSON", "Inception": "WORK_OF_ART", "The Dark Knight": "WORK_OF_ART", "Warner Bros": "ORGANIZATION"},
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
    # ── S11–S20: zweite Runde ────────────────────────────────────────────────
    {
        "text": "Quentin Tarantino wrote and directed Pulp Fiction, released by Miramax Films.",
        "expect": {"Quentin Tarantino": "PERSON", "Pulp Fiction": "WORK_OF_ART", "Miramax Films": "ORGANIZATION"},
    },
    {
        "text": "The World War II ended in 1945 with the surrender of Germany and Japan.",
        "expect": {"World War II": "EVENT", "Germany": "GPE", "Japan": "GPE"},
    },
    {
        "text": "J.K. Rowling wrote the Harry Potter series, published by Bloomsbury Publishing.",
        "expect": {"J.K. Rowling": "PERSON", "Harry Potter": "WORK_OF_ART", "Bloomsbury Publishing": "ORGANIZATION"},
    },
    {
        "text": "The Louvre is a historic monument in Paris that houses the Mona Lisa.",
        "expect": {"Louvre": "LOCATION", "Paris": "GPE"},
    },
    {
        "text": "Nikola Tesla was a Serbian-American inventor. He worked for Thomas Edison in New York.",
        "expect": {"Nikola Tesla": "PERSON", "Thomas Edison": "PERSON", "New York": "GPE"},
    },
    {
        "text": "Led Zeppelin released their debut album Led Zeppelin in 1969 on Atlantic Records.",
        "expect": {"Atlantic Records": "ORGANIZATION"},
    },
    {
        "text": "The French Revolution began in 1789 and led to the rise of Napoleon Bonaparte.",
        "expect": {"French Revolution": "EVENT", "Napoleon Bonaparte": "PERSON"},
    },
    {
        "text": "Stanley Kubrick directed 2001: A Space Odyssey, produced by Metro-Goldwyn-Mayer.",
        "expect": {"Stanley Kubrick": "PERSON", "2001: A Space Odyssey": "WORK_OF_ART", "Metro-Goldwyn-Mayer": "ORGANIZATION"},
    },
    {
        "text": "Amazon was founded by Jeff Bezos in Seattle, Washington in 1994.",
        "expect": {"Amazon": "ORGANIZATION", "Jeff Bezos": "PERSON", "Seattle, Washington": "GPE"},
    },
    {
        "text": "The Vietnam War was fought in Vietnam, Laos, and Cambodia from 1955 to 1975.",
        "expect": {"Vietnam War": "EVENT", "Vietnam": "GPE", "Laos": "GPE", "Cambodia": "GPE"},
    },
]

pipe = create_extraction_pipeline(config_path=_settings_path, cache_enabled=True)

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
