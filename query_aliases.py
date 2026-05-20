import kuzu
db = kuzu.Database("data/hotpotqa/graph/graph_KuzuDB")
conn = kuzu.Connection(db)

# Pull all PERSON entities once
print("Loading all PERSON entities...")
r = conn.execute("MATCH (e:Entity) WHERE e.type = 'PERSON' RETURN e.name")
names = []
while r.has_next():
    n = r.get_next()[0]
    if n:
        names.append(n)
print(f"  {len(names)} PERSON entities loaded")
print()

# Index by surname (last token, lowercased)
by_surname = {}
for n in names:
    toks = n.split()
    if len(toks) < 2:
        continue
    by_surname.setdefault(toks[-1].lower(), []).append((n, toks))

# Find pairs: long_name (>=3 tokens) <-> short_name (exactly 2 tokens)
# Constraints:
#   - same surname
#   - first-token initial of short matches long's first token (case-insensitive)
pairs = []
for surname, entries in by_surname.items():
    longs  = [(n, t) for n, t in entries if len(t) >= 3]
    shorts = [(n, t) for n, t in entries if len(t) == 2]
    if not longs or not shorts:
        continue
    for long_name, long_toks in longs:
        long_first = long_toks[0]
        for short_name, short_toks in shorts:
            short_first = short_toks[0]
            if short_first.lower() == long_first.lower():
                pairs.append((long_name, short_name))
            elif long_first.lower().startswith(short_first.lower()):
                pairs.append((long_name, short_name))
            elif short_first[0].lower() == long_first[0].lower() and len(short_first) <= 4:
                # initial-only match — looser, often noisy, kept for inspection
                pairs.append((long_name, short_name))

# Dedupe
seen = set()
unique = []
for p in pairs:
    key = (p[0].lower(), p[1].lower())
    if key not in seen:
        seen.add(key)
        unique.append(p)

print(f"Found {len(unique)} candidate alias pairs")
print()
print("First 30:")
for long_name, short_name in unique[:30]:
    print(f"  {long_name!r:50s}  <->  {short_name!r}")

# Quick distribution check
print()
print("Of those pairs, how many short-names actually appear in chunks?")
checked = 0
short_with_chunks = 0
for long_name, short_name in unique[:60]:
    rr = conn.execute(
        "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
        "WHERE e.name = $n "
        "RETURN COUNT(c)",
        {"n": short_name}
    )
    cnt = rr.get_next()[0] if rr.has_next() else 0
    if cnt > 0:
        short_with_chunks += 1
    checked += 1
print(f"  {short_with_chunks}/{checked} short-name entities have >=1 chunk")
