import re
from pathlib import Path

src = Path("src/logic_layer/planner.py").read_text(encoding="utf-8")
lines = src.splitlines()

# _VAGUE_REFS regex and _form_sub_query function bodies
ranges = [
    ("_VAGUE_REFS and surrounding", 1430, 1470),
    ("_form_sub_query body", 2958, 2975),
]
for label, start, end in ranges:
    print()
    print("=" * 75)
    print(f"  {label}  lines {start}-{end}")
    print("=" * 75)
    for i in range(start - 1, min(end, len(lines))):
        print(f"{i+1:5d}  {lines[i]}")

# Also: search for any string that could produce "Who form that?" — that has to
# be a verb fragment combined with a generic pronoun "that"
print()
print("=" * 75)
print('  Lines containing literal "that" inside an f-string or substitution')
print("=" * 75)
for i, line in enumerate(lines, start=1):
    if re.search(r'f["\'].*?\bthat\b.*?["\']', line) or 'sub_query' in line and '"that"' in line:
        start = max(1, i - 2)
        end = min(len(lines), i + 2)
        for j in range(start - 1, end):
            print(f"{j+1:5d}  {lines[j]}")
        print()
