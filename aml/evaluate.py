"""#82 AML-style local test, step 2: answer LoCoMo questions from three memories, DS judges.

Arms (same embedding bridge for all):
  raw    original turns cut into 4-turn chunks, top-5 by cosine   (the "chunk + vector" baseline)
  cards  wandering cards, top-10 by cosine, no walking               (what tagging alone buys)
  walk   wandering walk() path, as the companion would actually get it

    py aml/evaluate.py [--conv 0] [--n 40] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from wandering import db, tagger  # noqa: E402
from wandering.embed import embed, embed_batched  # noqa: E402
from wandering.walk import walk  # noqa: E402

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
PICK = {1: 8, 2: 10, 3: 5, 4: 12, 5: 5}


def ds(system, user):
    from openai import OpenAI
    client = OpenAI(api_key=tagger._load_tagger_key(), base_url=tagger.TAGGER_BASE_URL)
    r = client.chat.completions.create(model=tagger.TAGGER_MODEL, temperature=0,
                                       messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) or 1)


def raw_chunks(conv, size=4):
    chunks, n = [], 1
    while f"session_{n}" in conv:
        when = conv[f"session_{n}_date_time"]
        turns = conv[f"session_{n}"]
        for i in range(0, len(turns), size):
            lines = [f"{t['speaker']}: {t['text']}" + (f" [shares image: {t['blip_caption']}]"
                                                       if t.get("blip_caption") else "")
                     for t in turns[i:i + size]]
            chunks.append(f"[{when}]\n" + "\n".join(lines))
        n += 1
    return chunks


def card_line(c, because=None):
    s = f"[{c['at']}] 「{c['quote']}」 {c['what']}"
    return s + (f"（因为：{because}）" if because else "")


ANSWER_SYS = """You answer questions about a long conversation between {a} and {b}, using ONLY the memory notes given.
In the notes, 用户 and ta mean {a}, and 我 means {b}. Notes may be in Chinese; answer in English.
Answer in a short phrase. Use the dates on the notes to resolve relative times ("yesterday", "last week").
If the notes do not contain the answer, reply exactly: not mentioned"""

JUDGE_SYS = """You grade a memory system's answer against the gold answer. Be generous with wording and
date formats: same meaning or same time span counts. Reply with one word: CORRECT or WRONG."""

JUDGE_ADV = """The question asks about something that never happened (or happened to someone else).
The system is CORRECT only if it says it is not mentioned / unknown. If it asserts an answer, it is WRONG.
Reply with one word: CORRECT or WRONG."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    data = json.loads((HERE / "data" / "locomo10.json").read_text(encoding="utf-8"))
    sample = data[a.conv]
    conv = sample["conversation"]
    sa, sb = conv["speaker_a"], conv["speaker_b"]

    rng = random.Random(a.seed)
    qs = []
    for cat, k in PICK.items():
        pool = [q for q in sample["qa"] if q["category"] == cat]
        qs += rng.sample(pool, min(k, len(pool)))

    chunks = raw_chunks(conv)
    vecs = embed_batched(chunks, embedder=lambda t: embed(t, timeout=60), batch=8)
    if vecs is None:
        sys.exit("embedding bridge failed on raw chunks")
    cvecs = np.stack([unit(v) for v in vecs])

    conn = db.connect(HERE / f"locomo{a.conv}.sqlite")
    rows = conn.execute("SELECT * FROM cards").fetchall()
    cards = [dict(r) for r in rows]
    mats = [(c, [unit(db.blob_to_vec(r[k])) for k in ("vec", "quote_vec") if r[k] is not None])
            for c, r in zip(cards, rows)]

    sys_ans = ANSWER_SYS.format(a=sa, b=sb)
    out_path = HERE / "out" / f"eval_conv{a.conv}.jsonl"
    done = {}
    if out_path.exists():
        for ln in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            done[(r["q"], r["arm"])] = r
    fout = out_path.open("a", encoding="utf-8")

    for q in qs:
        qv = unit(embed([q["question"]])[0])
        ctx = {}
        top = np.argsort(-(cvecs @ qv))[:5]
        ctx["raw"] = "\n\n".join(chunks[i] for i in sorted(top))
        scored = sorted(((max(float(v @ qv) for v in vs), c) for c, vs in mats if vs),
                        key=lambda x: -x[0])[:10]
        ctx["cards"] = "\n".join(card_line(c) for _, c in scored)
        w = walk(conn, q["question"], gate_mode="normal", seed=0)
        if w.walked:
            steps = [w.start] + list(w.steps)
            ctx["walk"] = "\n".join(card_line(s.card, s.because) for s in steps)
        else:
            ctx["walk"] = f"(没有散步：{w.reason})"

        gold = q.get("answer", q.get("adversarial_answer"))
        for arm, text in ctx.items():
            if (q["question"], arm) in done:
                continue
            ans = ds(sys_ans, f"Memory notes:\n{text}\n\nQuestion: {q['question']}")
            if q["category"] == 5:
                verdict = ds(JUDGE_ADV, f"Question: {q['question']}\nSystem answer: {ans}")
            else:
                verdict = ds(JUDGE_SYS, f"Question: {q['question']}\nGold: {gold}\nSystem answer: {ans}")
            ok = bool(re.match(r"\s*CORRECT", verdict, re.I))
            rec = {"q": q["question"], "cat": q["category"], "gold": gold, "arm": arm,
                   "answer": ans, "ok": ok, "ctx_chars": len(text),
                   "walked": w.walked if arm == "walk" else None}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[(q["question"], arm)] = rec
            print(f"[{CAT[q['category']]}] {arm:5} {'✓' if ok else '✗'} {q['question'][:60]}", flush=True)

    tally = defaultdict(lambda: [0, 0])
    for r in done.values():
        for key in (r["arm"], f"{r['arm']}/{CAT[r['cat']]}"):
            tally[key][0] += r["ok"]
            tally[key][1] += 1
    print("\n== score ==")
    for k in sorted(tally):
        ok, n = tally[k]
        print(f"{k:24} {ok:3}/{n:<3} {100 * ok / n:5.1f}%")
    walked = sum(1 for r in done.values() if r["arm"] == "walk" and r["walked"])
    print(f"walk started on {walked}/{sum(1 for r in done.values() if r['arm'] == 'walk')} questions")


if __name__ == "__main__":
    main()
