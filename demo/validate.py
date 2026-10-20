import json, collections, os, sys
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards.jsonl")
cards, threads, errs = {}, [], []
for i, line in enumerate(open(p, encoding="utf-8"), 1):
    o = json.loads(line)
    if o["type"] == "card":
        if threads: errs.append(f"card after thread line {i}")
        if o["key"] in cards: errs.append("dup key " + o["key"])
        cards[o["key"]] = o
        for k in ("at","period","quote","source","tone","what","feeling","why","not_this","gate","author"):
            if k not in o: errs.append(f"{o['key']} missing {k}")
        if not o["what"].startswith("我"): errs.append(o["key"]+" what not 我")
        if o["tone"] not in ("认真","玩笑","气话","撒娇"): errs.append(o["key"]+" tone")
    else: threads.append(o)
pairs = set()
for t in threads:
    for k in (t["a"], t["b"]):
        if k not in cards: errs.append("missing key " + k)
    pr = frozenset((t["a"], t["b"]))
    if pr in pairs: errs.append(f"dup pair {t['a']}-{t['b']}")
    pairs.add(pr)
    if not 0.4 <= t["strength"] <= 0.95: errs.append("strength " + str(t))
src = collections.Counter("relayed" if c["source"].startswith("relayed") else c["source"] for c in cards.values())
deg = collections.Counter(k for t in threads for k in (t["a"], t["b"]))
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries.txt"), encoding="utf-8"):
    q, s, path, _ = line.rstrip("\n").split("\t")
    if path != "-":
        ks = path.split(">")
        if ks[0] != s: errs.append("query start mismatch " + q)
        for x, y in zip(ks, ks[1:]):
            if frozenset((x, y)) not in pairs: errs.append(f"query path {x}>{y} has no thread ({q})")
print("cards", len(cards), "threads", len(threads))
print("kinds", dict(collections.Counter(t["kind"] for t in threads)))
print("gates", dict(collections.Counter(c["gate"] for c in cards.values())))
print("sources", dict(src), "feeling null", sum(c["feeling"] is None for c in cards.values()))
print("hubs", deg.most_common(5), "isolated", [k for k in cards if k not in deg])
print("ERRORS", errs or "none")
