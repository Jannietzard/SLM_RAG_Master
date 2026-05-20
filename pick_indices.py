import json
from pathlib import Path

# Load the latest JSONL
jsonl_files = sorted(Path("evaluation_results").glob("hotpotqa_*.jsonl"))
latest = jsonl_files[-1]
print(f"Using: {latest.name}")
print()

rows = []
with open(latest, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

# Load questions to get index mapping
with open("data/hotpotqa/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

# Build qid -> idx map
qid_to_idx = {q.get("_id") or q.get("id") or q.get("question_id"): i
              for i, q in enumerate(questions)}

def categorize(row):
    if row.get("llm_error"):
        return "llm_error"
    if not row.get("all_gold_retrieved"):
        return "retrieval_fail"
    if not row.get("exact_match"):
        return "reasoning_fail"
    return "success"

# Group by (question_type, category)
from collections import defaultdict
buckets = defaultdict(list)
for r in rows:
    qid = r.get("question_id")
    if qid not in qid_to_idx:
        continue
    cat = categorize(r)
    qtype = r.get("question_type", "?")
    buckets[(qtype, cat)].append((qid_to_idx[qid], r.get("question", "")[:70]))

for key in sorted(buckets):
    qtype, cat = key
    print(f"  {qtype:12s} {cat:18s} (n={len(buckets[key])})")
    for idx, q in buckets[key][:3]:
        print(f"      idx={idx:3d}  {q}")
    print()
