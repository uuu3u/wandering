"""#82 AML-style local test, step 1: feed one LoCoMo conversation into a fresh wandering db.

Add = tagger model over windows of the conversation. Speaker A is written as 用户,
speaker B as 我, because that is the only shape the tagger knows. Resumable: each
window's tagger output is cached in aml/out/tag_<n>.json.

    py aml/ingest.py [--conv 0] [--win 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from wandering import db, tagger  # noqa: E402
from wandering.load import embed_cards  # noqa: E402

GATE_FROM_HINT = {"unsure": "tender"}


def session_windows(conv, win):
    a, b = conv["speaker_a"], conv["speaker_b"]
    who = {a: "用户", b: "我"}
    n = 1
    while f"session_{n}" in conv:
        when = conv[f"session_{n}_date_time"]
        at = datetime.strptime(when, "%I:%M %p on %d %B, %Y").strftime("%Y-%m-%d %H:%M")
        turns = conv[f"session_{n}"]
        for i in range(0, len(turns), win):
            lines = []
            for t in turns[i:i + win]:
                s = f"{who.get(t['speaker'], t['speaker'])}: {t['text']}"
                if t.get("blip_caption"):
                    s += f" [发了张图：{t['blip_caption']}]"
                lines.append(s)
            yield {"session": n, "at": at, "ids": [t["dia_id"] for t in turns[i:i + win]],
                   "text": f"[{at}]\n" + "\n".join(lines)}
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--win", type=int, default=8)
    a = ap.parse_args()

    data = json.loads((HERE / "data" / "locomo10.json").read_text(encoding="utf-8"))
    conv = data[a.conv]["conversation"]
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    wins = list(session_windows(conv, a.win))
    (out / "windows.json").write_text(json.dumps(wins, ensure_ascii=False, indent=1), encoding="utf-8")

    conn = db.connect(HERE / f"locomo{a.conv}.sqlite")
    recent: list[dict] = []
    for n, w in enumerate(wins):
        cache = out / f"tag_{n}.json"
        if cache.exists():
            res = json.loads(cache.read_text(encoding="utf-8"))
        else:
            res = tagger.tag(w["text"], recent[-20:])
            res["_window"] = n
            cache.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        if res.get("_loaded"):
            continue
        keymap, new_ids = {}, []
        for i, c in enumerate(res["cards"]):
            try:
                cid = db.add_card(conn, quote=c["quote"], what=c["what"], at=c.get("at") or w["at"],
                                  period=c.get("period"), source=c.get("source"), tone=c.get("tone"),
                                  why=c.get("why"), not_this=c.get("not_this"),
                                  gate=GATE_FROM_HINT.get(c.get("gate_hint"), c.get("gate_hint") or "normal"),
                                  author="tagger", charge=c.get("charge"),
                                  realm=c.get("realm") or "real", commit=False)
            except ValueError as e:
                print(f"win {n} card {i} rejected: {e}", file=sys.stderr)
                continue
            keymap[f"new{i}"] = cid
            new_ids.append(cid)
            recent.append({"id": str(cid), "at": w["at"], "quote": c["quote"], "what": c["what"]})
        for th in res["threads"]:
            ends = [keymap.get(str(x)) or (int(x) if str(x).isdigit() else None) for x in (th["a"], th["b"])]
            if None in ends or ends[0] == ends[1]:
                continue
            db.add_thread(conn, ends[0], ends[1], th["kind"], th.get("because", ""),
                          th.get("strength", 0.7), commit=False)
        conn.commit()
        embed_cards(conn, new_ids)
        res["_loaded"] = True
        cache.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"win {n}/{len(wins)}: +{len(new_ids)} cards, {len(res['threads'])} threads, flags={len(res['_flags'])}",
              flush=True)

    nc = conn.execute("select count(*) from cards").fetchone()[0]
    nt = conn.execute("select count(*) from threads").fetchone()[0]
    print(f"done: {nc} cards, {nt} threads")


if __name__ == "__main__":
    main()
