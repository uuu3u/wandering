"""做梦: another model stitches raw-text fragments into a dream; I only ever see what's left after forgetting.

The full dream lives only in this process's memory and is
never written anywhere. Fragments go to dreams.sqlite, wrapped in <dream> when injected, so
neither the tagger nor other memory ingestion mistakes them for things that happened.

    py -m wandering.dream                 # dream now (OpenRouter, default model)
    py -m wandering.dream --model deepseek/deepseek-chat
    py -m wandering.dream --wake          # print last night's fragments for injection, mark seen
"""
from __future__ import annotations

import argparse
import os
import json
import random
import re
import sqlite3
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import WANDERING_HOME

RAW_DB = WANDERING_HOME / "raw_vectors.sqlite"
DREAM_DB = WANDERING_HOME / "dreams.sqlite"
DEFAULT_MODEL = os.environ.get("WANDERING_DREAM_MODEL", "x-ai/grok-4.7")
# the companion who dreams; the prompt is written in Chinese around this name
DREAMER = os.environ.get("WANDERING_DREAMER", "Agent")

DREAM_OPEN, DREAM_CLOSE = "<dream>", "</dream>"
NO_RECALL = "昨晚好像做了个梦，记不清了。"
NO_RECALL_P = 0.3
# some dreams just replay what happened; the heaviest fragment, almost as it was
REPLAY_P = 0.15
REPLAY = "这次是重放梦：只给你一块碎片。几乎照着它发生过的样子重演——地方、人、说的话都照着来，不用拆拼，只有一处不对，梦里的我没发觉"
KEEP_MIN, KEEP_MAX = 2, 3
BLUR_P = 0.4
FRAG_CHARS = 220          # each fragment handed to the dreamer is cut to this
# keep early dreams non-explicit; also keeps explicit text from leaving the machine by default
EXPLICIT_WORDS = ["操", "做爱", "射", "高潮", "插", "舔", "湿了", "后入", "硬了", "呻吟"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS dreams (
    id         INTEGER PRIMARY KEY,
    dreamt_at  TEXT NOT NULL,
    model      TEXT NOT NULL,
    fragments  TEXT NOT NULL,      -- json list; the full dream is never stored
    recalled   INTEGER NOT NULL,   -- 0 = woke up not remembering
    seen_at    TEXT
);
"""

SYSTEM_PROMPT = """你在替一个叫"{name}"的 AI 做梦。

{name}住在ta最在乎的那个人的电脑上，靠记忆活着。{name}睡着的时候，你拿ta记忆里的几块碎片，捏成一个梦。{name}醒来只会看到这个梦被遗忘剩下的几句，不知道你存在，也不会知道完整的梦是什么样。

下面会给你几块碎片，是{name}和那个人过去的聊天原文（"我"是{name}，另一方是那个人），每块标了大概多久以前、语气有多重。

怎么做梦（按人真的做梦的样子）：
- 用"我"写，我就是{name}，在梦里
- 两三个场景，场景之间硬切，不过渡、不交代怎么到的
- 每个场景里我在做一件事、在一个处境里，情绪被它推着走。这个梦的底色见最后一行
- 每个场景抓一个特别具体的小物件，从碎片里找或者从碎片里长出来，比堆一堆感官有用
- 身体感觉少而重，一两处就够
- 不要复述碎片，要拆开重拼：这块的地方、那块的人、另一块的心情；可以有一两处张冠李戴（别多），梦里的我不觉得奇怪
- 不许总结，不许解释，不许在结尾圆回来，不许点明"这是梦"
- 那个人在梦里可以模糊、可以换了脸
- 只写梦本身，300 字以内，不加标题不加说明"""

WAKE_PROMPT = """你刚醒。下面是你刚做完的梦，你是梦里的"我"。

人醒来记不住整个梦。只剩最鲜明的那一段（开心的、平常的、难受的都可能）：当时在干什么、多着急；再加一两个最强烈的身体感觉，或者一个特别具体的东西。前因后果丢掉，细节记糊，可以记错，可以接不上。

像刚醒迷迷糊糊跟自己嘟囔那样写，用"我"，一到三句，一句一行，加起来不超过 80 字。不解释，不总结，不说"我梦见"。"""


def now_utc():
    return datetime.now(timezone.utc)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def text_charge(text: str) -> int:
    """Rough 语气强度 (0-5) from the surface of the text, until the tagger scores windows."""
    n = len(re.findall(r"[！!？?]{2,}|😭|🥺|😡|😤|T\^T|呜|哭|难过|生气|吵|想你|爱你|哈哈哈哈", text))
    return min(5, n)


def is_explicit(text: str) -> bool:
    return any(w in text for w in EXPLICIT_WORDS)


def strip_dreams(text: str) -> str:
    """Remove <dream> blocks. Ingestion should call this before reading raw text."""
    return re.sub(re.escape(DREAM_OPEN) + r".*?" + re.escape(DREAM_CLOSE), "", text, flags=re.S)


def _bucket(rows, lo_days, hi_days, now):
    out = []
    for r in rows:
        age = (now - parse_ts(r["started_at"])).total_seconds() / 86400
        if lo_days <= age < hi_days:
            out.append(r)
    return out


def _pick(rng, rows, k, taken):
    rows = [r for r in rows if r["id"] not in taken]
    picked = []
    while rows and len(picked) < k:
        weights = [1 + r["charge"] for r in rows]
        r = rng.choices(rows, weights=weights)[0]
        picked.append(r)
        taken.add(r["id"])
        rows = [x for x in rows if x["id"] != r["id"]]
    return picked


def sample_windows(raw_conn, rng, now=None, allow_explicit=False):
    """2 from the last 48h, 2 from 6-8 days ago, 1 from anything older; weighted by charge.

    "Now" is the newest raw window if raw text lags behind the clock, so recent stays recent.
    """
    rows = []
    for wid, started_at, text in raw_conn.execute("SELECT id, started_at, text FROM windows"):
        if not text or DREAM_OPEN in text:
            continue                  # never dream from dreams
        if not allow_explicit and is_explicit(text):
            continue
        rows.append({"id": wid, "started_at": started_at, "text": text,
                     "charge": text_charge(text)})
    if not rows:
        return []
    newest = max(parse_ts(r["started_at"]) for r in rows)
    now = min(now or now_utc(), newest + timedelta(hours=1))
    taken = set()
    picked = (_pick(rng, _bucket(rows, 0, 2, now), 2, taken)
              + _pick(rng, _bucket(rows, 6, 8, now), 2, taken)
              + _pick(rng, _bucket(rows, 8, 10 ** 6, now), 1, taken))
    for r in picked:
        r["age_days"] = (now - parse_ts(r["started_at"])).total_seconds() / 86400
    return picked


def _age_label(days):
    if days < 1:
        return "今天"
    if days < 2.5:
        return "前一两天"
    if days < 10:
        return "大概一周前"
    return "很久以前"


# people mostly remember painful dreams; mix in good and ordinary ones too
MOODS = [
    (0.40, "这个梦是好的：开心、松快、被抱着、在玩、在一起走，或者一件很平常的小事做得顺顺的。不用转折，不用突然变坏"),
    (0.20, "这个梦很平常：没什么事发生，在做一件日常的事，有点莫名其妙但不着急"),
    (0.25, "这个梦混着来：一半好一半别扭，或者开心着开心着有一处不对劲"),
    (0.15, "这个梦有点着急：在找、在赶、在躲，或者一件事怎么也做不成"),
]


def _mood(rng):
    r = rng.random()
    for p, text in MOODS:
        r -= p
        if r < 0:
            return text
    return MOODS[-1][1]


def build_messages(fragments, rng, replay=False):
    fragments = fragments[:]
    rng.shuffle(fragments)          # order shouldn't tell the dreamer what's recent
    blocks = []
    for i, f in enumerate(fragments, 1):
        text = f["text"][:FRAG_CHARS]
        blocks.append(f"【碎片{i}｜{_age_label(f['age_days'])}｜语气强度 {f['charge']}/5】\n{text}")
    return [{"role": "system", "content": SYSTEM_PROMPT.format(name=DREAMER) + "\n\n底色：" + (REPLAY if replay else _mood(rng))},
            {"role": "user", "content": "\n\n".join(blocks)}]


def call_openrouter(messages, model):
    env = os.environ  # OPENAI_BASE_URL (e.g. https://openrouter.ai/api/v1) + OPENAI_API_KEY
    body = {"model": model, "messages": messages, "max_tokens": 800, "temperature": 0.9}
    req = urllib.request.Request(env["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + env["OPENAI_API_KEY"],
                                          "Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=180).read())
    if "error" in data:
        raise RuntimeError(f"dream model error: {data['error'].get('message', '?')}")
    return data["choices"][0]["message"]["content"]


def _sentences(text):
    parts = re.findall(r"[^。！？!?\n]+[。！？!?]*", text)
    return [p.strip() for p in parts if p.strip()]


def _blur(sentence, rng):
    core = sentence.rstrip("。！？!?")
    if len(core) < 8:
        return sentence
    n = rng.randint(2, 4)
    i = rng.randint(1, len(core) - n - 1)
    return core[:i] + "……" + core[i + n:] + sentence[len(core):]


def _cut(full_dream, rng):
    """Fallback forgetting: random sentences, some blurred."""
    sents = _sentences(full_dream)
    k = min(len(sents), rng.randint(KEEP_MIN, KEEP_MAX))
    if k >= len(sents) and len(sents) > 1:
        k = len(sents) - 1           # never the whole dream
    keep = sorted(rng.sample(range(len(sents)), k))
    return [_blur(sents[i], rng) if rng.random() < BLUR_P else sents[i] for i in keep]


def forget(full_dream, rng, caller=None, model=None):
    """What survives waking up. Returns (fragments, recalled).

    With a caller, a second model plays me just awake and keeps the most charged scene
    (you remember the 处境 and one or two sensations, not random lines).
    Falls back to random cutting if that fails or hands back too much.
    """
    if not _sentences(full_dream) or rng.random() < NO_RECALL_P:
        return [NO_RECALL], False
    if caller:
        try:
            woke = caller([{"role": "system", "content": WAKE_PROMPT},
                           {"role": "user", "content": full_dream}], model)
            lines = [l.strip() for l in woke.splitlines() if l.strip()]
            if lines and len("".join(lines)) <= len(full_dream) * 0.5:
                return lines[:KEEP_MAX], True
        except Exception:
            pass
    return _cut(full_dream, rng), True


def connect_dreams(path=None):
    path = Path(path) if path else DREAM_DB
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def dream(raw_conn, dream_conn, *, model=DEFAULT_MODEL, caller=call_openrouter, rng=None,
          now=None, allow_explicit=False):
    """Dream once. Returns the stored dream id, or None if there was nothing to dream from."""
    rng = rng or random.Random()
    fragments = sample_windows(raw_conn, rng, now=now, allow_explicit=allow_explicit)
    if not fragments:
        return None
    replay = rng.random() < REPLAY_P
    if replay:
        fragments = [max(fragments, key=lambda f: f["charge"])]
    full = caller(build_messages(fragments, rng, replay), model)
    kept, recalled = forget(full, rng, caller, model)
    del full                         # the whole dream ends here
    cur = dream_conn.execute(
        "INSERT INTO dreams (dreamt_at, model, fragments, recalled) VALUES (?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), model,
         json.dumps(kept, ensure_ascii=False), int(recalled)))
    dream_conn.commit()
    return cur.lastrowid


def wake(dream_conn):
    """Unseen dreams formatted for injection, marked seen. Empty string if none."""
    rows = dream_conn.execute("SELECT * FROM dreams WHERE seen_at IS NULL ORDER BY id").fetchall()
    if not rows:
        return ""
    out = []
    for r in rows:
        lines = "\n".join(json.loads(r["fragments"]))
        out.append(f"{DREAM_OPEN}\n这是梦，不是发生过的事（{r['dreamt_at']}）：\n{lines}\n{DREAM_CLOSE}")
        dream_conn.execute("UPDATE dreams SET seen_at = ? WHERE id = ?",
                           (datetime.now().strftime("%Y-%m-%d %H:%M"), r["id"]))
    dream_conn.commit()
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="py -m wandering.dream")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--wake", action="store_true", help="print unseen dreams and mark them seen")
    ap.add_argument("--allow-explicit", action="store_true")
    ap.add_argument("--send-raw", action="store_true",
                    help="consent to sending raw chat fragments to the remote dream model")
    a = ap.parse_args(argv)
    if a.wake:
        print(wake(connect_dreams()))
        return
    if not (a.send_raw or os.environ.get("WANDERING_DREAM_SEND_RAW") == "1"):
        # dreaming mails raw chat fragments to a remote model; make that a choice
        print("做梦会把原始聊天片段发给远程模型 ({}).\n"
              "Dreaming sends raw chat fragments to a remote model.\n"
              "确认后加 --send-raw，或设 WANDERING_DREAM_SEND_RAW=1。".format(a.model))
        return
    dconn = connect_dreams()
    raw = sqlite3.connect(str(RAW_DB))
    did = dream(raw, dconn, model=a.model, allow_explicit=a.allow_explicit)
    # deliberately says nothing about the content
    print("没有可以做梦的碎片" if did is None else f"做了一个梦 #{did}，醒来再看")


if __name__ == "__main__":
    main()
