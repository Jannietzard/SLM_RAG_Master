import re
from pathlib import Path

src = Path("src/logic_layer/planner.py").read_text(encoding="utf-8")
lines = src.splitlines()

# Search for the verb fragments that might be in templates
patterns_to_find = [
    r'\bhold\b',
    r'Who\s+\w+',
    r'f"Who',
    r'_find_relative_clause_bridge',
    r'_find_passive_agent_bridge',
    r'_find_implicit_bridge',
    r'_decompose_implicit_bridge',
]

hits_by_line = {}
for i, line in enumerate(lines, start=1):
    for pat in patterns_to_find:
        if re.search(pat, line):
            hits_by_line.setdefault(i, []).append(pat)

# Group nearby hits and print each cluster with surrounding context
clusters = []
current = []
for line_no in sorted(hits_by_line):
    if not current or line_no - current[-1] <= 15:
        current.append(line_no)
    else:
        clusters.append(current)
        current = [line_no]
if current:
    clusters.append(current)

for cluster in clusters:
    start = max(1, cluster[0] - 3)
    end = min(len(lines), cluster[-1] + 8)
    print()
    print("=" * 75)
    print(f"  Cluster: lines {start}-{end}  (hits at {cluster})")
    print("=" * 75)
    for i in range(start - 1, end):
        marker = "* " if (i + 1) in hits_by_line else "  "
        print(f"{marker}{i+1:5d}  {lines[i]}")
