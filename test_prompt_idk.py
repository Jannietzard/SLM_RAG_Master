"""
test_prompt_idk.py — Test whether different local models can answer
a specific prompt that qwen2:1.5b answers with "I don't know."

Root cause under investigation:
  The model receives a context containing the answer (Roud index 821) but
  fails because "What Are Little Girls Made Of?" (Star Trek episode) must
  be recognised as INSPIRED BY "What Are Little Boys Made Of?" (nursery
  rhyme).  The paraphrase bridge Girls→Boys is in Chunk [5] but qwen2:1.5b
  does not follow it.

Usage:
    python -X utf8 test_prompt_idk.py
    python -X utf8 test_prompt_idk.py --models qwen2.5:3b qwen3:4b
    python -X utf8 test_prompt_idk.py --url http://localhost:11434

Models tested by default:
    qwen2:1.5b    (current production baseline)
    qwen2.5:3b    (recommended next step)
    qwen3:4b      (upper ablation bound)
    gemma3:4b     (alternative architecture, Gemini family)
"""

import argparse
import json
import time
import urllib.request
import urllib.error

# ── The exact prompt from the failing trace ────────────────────────────────
PROMPT = """  You are a factual QA assistant. Answer based ONLY on the context below.
  
  This is a multi-step question. Use the following reasoning chain to find the answer:
  Step 1: Netherlands wear during Oranjegekte or to celebrate the national holiday
  Koningsdag? → Orange-Nassau
  Step 2: What color clothing do people? → THIS IS THE ANSWER
  
  Rules:
  - Give the shortest possible answer: a name, place, date, number, or yes/no.
  - Do NOT explain or add sentences beyond the direct answer.
  - If the answer is a number or statistic (e.g. population, count, year): reply with
  just the number.
  - If the context does not contain the answer: reply with "I don't know."
  
  Context:
  [1] Oranjegekte (Orange craze) or Oranjekoorts (Orange fever) is a phenomenon in the
  Netherlands that occurs during major sporting events, especially international
  football championships, and during Koningsdag, an annual holiday celebrating the
  king's birthday. It manifests itself in the wearing of orange clothing such as
  T-shirts, caps and scarfs; lavish attention for sports and sports fans in the media;
  and the decoration of cars, rooms, houses, shops, and even entire streets in orange,
  the traditional color of the Dutch royal family, the House of Orange-Nassau.
  
  [2] Koningsdag (] ) or King's Day is a national holiday in the Kingdom of the
  Netherlands. Celebrated on 27 April (26 April if the 27th is a Sunday), the date marks
  the birth of King Willem-Alexander.
  
  [3] The casera outfits are normally dark blue, but can come in many colors. They wear
  caseras because that is what the people of the mountain wear and the holiday used to
  celebrate the peasants who sold their goods in town and came on Santo Tomas to pay
  rent to landlords in the city.
  
  [4] Queen's Day or "Koninginnedag" is the former name of a national holiday in the
  Netherlands, the Netherlands Antilles and Aruba from 1890 to 2013. It has since been
  renamed King's Day or "Koningsdag" to honor the birthday of King Willem-Alexander.
  
  [5] It was recognized as a public holiday in the southern districts of Belize in 1943,
  and declared a national holiday in 1977. The holiday celebrates the settlement of the
  Garifuna people in Belize after being exiled from the Grenadines by the British army.
  The major festivities for the holiday occur in the town of Dangriga, including
  parades, street music, and traditional dancing.
  
  Question: What color clothing do people of the Netherlands wear during Oranjegekte or
  to celebrate the national holiday Koningsdag?
  
  Answer (as short as possible):"""

# Testet: idx: 63,65,71,79

GOLD_ANSWER = "orange"

DEFAULT_MODELS = [
    # timed out 90sec
    "qwen2:1.5b",       # baseline
    "qwen2.5:3b",       # intra-family scale
    #"qwen3:4b",         # next-gen same vendor (uses /no_think)
    #"gemma3:4b",        # cross-architecture (Google)
    #"phi3.5:3.8b",      # Microsoft, strong on SQuAD
    "llama3.2:3b",      # Meta, 128k context
    #"mistral:7b",       # upper edge bound
    #"nomic-embed-text:latest",
    #"gemma4:latest",
    "phi3:latest"
]


def call_ollama(model: str, prompt: str, base_url: str) -> tuple[str, float]:
    """Call Ollama /api/generate and return (answer, latency_ms).

    qwen3:* models have Chain-of-Thought thinking enabled by default, which
    generates 500–2000 internal tokens before the answer and causes timeouts.
    Disable it via the /no_think suffix in the prompt (Ollama ≥0.6.2).
    """
    # Disable qwen3 thinking mode to prevent timeout on short-answer tasks
    effective_prompt = prompt
    if model.startswith("qwen3"):
        effective_prompt = prompt + " /no_think"

    payload = json.dumps({
        "model": model,
        "prompt": effective_prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            latency_ms = (time.time() - t0) * 1000
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip(), latency_ms
    except urllib.error.HTTPError as e:
        latency_ms = (time.time() - t0) * 1000
        if e.code == 404:
            return f"[MODEL NOT FOUND — run: ollama pull {model}]", latency_ms
        return f"[HTTP {e.code}: {e.reason}]", latency_ms
    except urllib.error.URLError as e:
        return f"[CONNECTION ERROR: {e.reason}]", (time.time() - t0) * 1000
    except Exception as e:
        return f"[ERROR: {e}]", (time.time() - t0) * 1000


def is_correct(answer: str, gold: str) -> bool:
    import re
    def norm(t):
        t = t.lower()
        t = re.sub(r'\b(a|an|the)\b', ' ', t)
        t = re.sub(r'[^\w\s]', '', t)
        return ' '.join(t.split())
    a, g = norm(answer), norm(gold)
    return a == g or (g and bool(re.search(r'\b' + re.escape(g) + r'\b', a)))


def main():
    parser = argparse.ArgumentParser(description="IDK failure regression test across models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Ollama model names to test")
    parser.add_argument("--url", default="http://localhost:11434",
                        help="Ollama base URL")
    args = parser.parse_args()

    print("=" * 72)
    print("  IDK FAILURE — CROSS-MODEL PROMPT TEST")
    print("=" * 72)
    print(f"\n  Gold answer: {GOLD_ANSWER}")
    print(f"  Root cause:  paraphrase bridge Girls → Boys not followed by small model")
    print(f"  Ollama URL:  {args.url}")
    print(f"  Models:      {', '.join(args.models)}\n")

    col_w = max(len(m) for m in args.models) + 2
    header = f"  {'Model':<{col_w}}  {'Answer':<30}  {'Correct':>7}  {'Latency':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for model in args.models:
        print(f"  {model:<{col_w}}  {'(querying...)':<30}", end="\r", flush=True)
        answer, latency_ms = call_ollama(model, PROMPT, args.url)
        correct = is_correct(answer, GOLD_ANSWER)
        answer_short = answer[:28] + ".." if len(answer) > 30 else answer
        marker = "OK" if correct else "XX"
        print(f"  {model:<{col_w}}  {answer_short:<30}  {marker:>7}  {latency_ms:>8.0f} ms")
        results.append({
            "model": model,
            "answer": answer,
            "correct": correct,
            "latency_ms": latency_ms,
        })

    print()
    correct_count = sum(1 for r in results if r["correct"])
    print(f"  {correct_count}/{len(results)} models answered correctly.\n")

    # Diagnosis summary — show actual answer, not a label
    print("  Diagnosis:")
    for r in results:
        actual = r["answer"][:80].replace("\n", " ")
        if r["correct"]:
            print(f"    OK  {r['model']}  →  \"{actual}\"")
        elif "[MODEL NOT FOUND" in r["answer"]:
            print(f"    --  {r['model']} — not installed (run: ollama pull {r['model']})")
        elif "[ERROR" in r["answer"]:
            print(f"    !!  {r['model']}  →  {actual}")
        else:
            print(f"    XX  {r['model']}  →  \"{actual}\"")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()
