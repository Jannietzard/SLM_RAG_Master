import re
from pathlib import Path

src = Path("src/logic_layer/verifier.py").read_text(encoding="utf-8")
lines = src.splitlines()

# Find any string template containing "Step 1" / "Step 2" / "reasoning chain"
keywords = ["Step 1", "Step 2", "reasoning chain", "multi-step", "derive the final"]
hits = set()
for i, line in enumerate(lines, start=1):
    for kw in keywords:
        if kw in line:
            hits.add(i)

# Group into clusters and print with context
sorted_hits = sorted(hits)
clusters = []
current = []
for h in sorted_hits:
    if not current or h - current[-1] <= 15:
        current.append(h)
    else:
        clusters.append(current)
        current = [h]
if current:
    clusters.append(current)

for cluster in clusters:
    start = max(1, cluster[0] - 5)
    end = min(len(lines), cluster[-1] + 15)
    print()
    print("=" * 75)
    print(f"  Lines {start}-{end}")
    print("=" * 75)
    for i in range(start - 1, end):
        marker = "* " if (i + 1) in hits else "  "
        print(f"{marker}{i+1:5d}  {lines[i]}")
