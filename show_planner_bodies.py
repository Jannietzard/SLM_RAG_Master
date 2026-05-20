import re
from pathlib import Path

src = Path("src/logic_layer/planner.py").read_text(encoding="utf-8")
lines = src.splitlines()

# Show specific line ranges
ranges = [
    ("_form_sub_query and helpers", 2958, 3035),
    ("multi-hop decomposition w/ _form_sub_query call site", 2440, 2500),
    ("Pattern J implicit bridge (line 2233+)", 2233, 2330),
    ("classifier-consistency fallback (line 2490+)", 2490, 2555),
]

for label, start, end in ranges:
    print()
    print("=" * 75)
    print(f"  {label}    lines {start}-{end}")
    print("=" * 75)
    for i in range(start - 1, min(end, len(lines))):
        print(f"{i+1:5d}  {lines[i]}")
