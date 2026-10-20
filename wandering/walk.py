"""散步: start from the card closest to a query, then wander along threads."""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np

from . import config
from .db import blob_to_vec, now_str
from .embed import embed as default_embed

GATE_MODES = ("normal", "affection", "intimate", "skip")


@dataclass
class Step:
    card: dict
    kind: str | None = None        # thread kind (None for the start)
    because: str | None = None
    thread_id: int | None = None
    sim: float | None = None       # cosine vs query (None in fallback / missing vec)
    drift: bool = False            # 走神想到的
    returned: bool = False         # reached after jumping back to the start


@dataclass
class WalkResult:
    query: str
    gate_mode: str
    walked: bool = False
    reason: str | None = None      # why no walk happened
    fallback: bool = False         # start found by FTS, off-topic checks skipped
    start: Step | None = None
    steps: list = field(default_factory=list)
    stop: str | None = None        # 'light' | 'max_steps' | 'no_threads' | 'offtopic' | 'drift'
    light: float = 1.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- helpers

def _card_dict(row) -> dict:
    d = {k: row[k] for k in row.keys() if k != "vec"}
    return d


def _cos(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_at(at):
    if not at:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(at, fmt).date()
        except ValueError:
            continue
    return None


def salience(card) -> float:
    s = 1.0
    if card["feeling"] and str(card["feeling"]).strip():
        s += 0.3
    if card["why"] and str(card["why"]).strip():
        s += 0.2
    return s


def recency(card, now: date) -> float:
    d = _parse_at(card["at"])
    if d is None:
        return 0.5
    days = max(0, (now - d).days)
    return 0.5 + 0.5 * math.exp(-days / config.RECENCY_DAYS)


def _allowed(card, gate_mode, as_start, sim=None) -> bool:
    g = card["gate"]
    # intimate = 亲昵（小名、晚安、抱抱），亲昵或亲密语境可翻；explicit 只在亲密语境
    if g == "intimate" and gate_mode not in ("affection", "intimate"):
        return False
    if g == "explicit" and gate_mode != "intimate":
        return False
    if g == "tender":
        if not as_start:
            return False
        return sim is not None and sim >= config.TENDER_START_MIN
    return True


def _softmax_pick(rng, items, scores):
    m = max(scores)
    weights = [math.exp((s - m) / config.TEMP) for s in scores]
    r = rng.random() * sum(weights)
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r < acc:
            return it
    return items[-1]


def _fts_start(conn, query, gate_mode):
    terms = [t for t in query.split() if t]
    if not terms:
        return None
    long_terms = [t for t in terms if len(t) >= 3]
    rows = []
    if long_terms:
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in long_terms)
        try:
            rows = conn.execute(
                """SELECT c.* FROM cards_fts f JOIN cards c ON c.id = f.rowid
                   WHERE cards_fts MATCH ? ORDER BY bm25(cards_fts) LIMIT 50""",
                (match,)).fetchall()
        except Exception:
            rows = []
    if not rows:
        where = " OR ".join(
            "(quote LIKE ? OR what LIKE ? OR IFNULL(feeling,'') LIKE ?)" for _ in terms)
        params = []
        for t in terms:
            params += [f"%{t}%"] * 3
        rows = conn.execute(f"SELECT * FROM cards WHERE {where} ORDER BY id", params).fetchall()
    for r in rows:
        # no similarity in fallback -> tender cards can't qualify as a start
        if _allowed(r, gate_mode, as_start=True, sim=None):
            return r
    return None


# ---------------------------------------------------------------- walk

def walk(conn, query, gate_mode="normal", seed=None, embedder=default_embed,
         now=None, dry_run=False) -> WalkResult:
    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode must be one of {GATE_MODES}")
    res = WalkResult(query=query, gate_mode=gate_mode)
    if gate_mode == "skip":
        res.reason = "gate=skip，不散步"
        return res
    if now is None:
        now = date.today()
    elif isinstance(now, datetime):
        now = now.date()
    elif isinstance(now, str):
        now = _parse_at(now) or date.today()
    rng = random.Random(seed)

    rows = conn.execute("SELECT * FROM cards").fetchall()
    by_id = {r["id"]: r for r in rows}
    if not rows:
        res.reason = "库里还没有卡片"
        return res

    qvec = None
    try:
        out = embedder([query]) if embedder else None
        if out:
            qvec = np.asarray(out[0], dtype=np.float32)
    except Exception:
        qvec = None

    sims: dict[int, float] = {}
    if qvec is not None:
        for r in rows:
            cs = [_cos(qvec, v) for v in (blob_to_vec(r["vec"]), blob_to_vec(r["quote_vec"]))
                  if v is not None]
            if cs:
                sims[r["id"]] = max(cs)

    def sim_of(cid):
        return sims.get(cid) if qvec is not None else None

    # ---- start
    if qvec is not None and sims:
        best_id, best = None, -2.0
        for cid, s in sims.items():
            if _allowed(by_id[cid], gate_mode, as_start=True, sim=s) and s > best:
                best_id, best = cid, s
        if best_id is None or best < config.START_MIN:
            res.reason = f"最像的一张只有 {best:.2f}，低于起步线 {config.START_MIN}"
            return res
        start = by_id[best_id]
    else:
        res.fallback = True
        start = _fts_start(conn, query, gate_mode)
        if start is None:
            res.reason = "嵌入不可用，关键词也没碰到卡片"
            return res

    res.walked = True
    res.start = Step(card=_card_dict(start), sim=sim_of(start["id"]))
    visited = {start["id"]}
    used_threads = []
    cur_id = start["id"]
    returned_done = False
    light = 1.0

    def candidates(from_id):
        out = []
        for t in conn.execute("SELECT * FROM threads WHERE a = ? OR b = ? ORDER BY id",
                              (from_id, from_id)):
            other = t["b"] if t["a"] == from_id else t["a"]
            if other in visited or other not in by_id:
                continue
            c = by_id[other]
            if not _allowed(c, gate_mode, as_start=False):
                continue
            score = float(t["strength"]) * salience(c) * recency(c, now)
            out.append((t, c, score))
        return out

    while True:
        if len(res.steps) >= config.MAX_STEPS:
            res.stop = "max_steps"
            break
        returned = False
        if not returned_done and res.steps and cur_id != start["id"]:
            if rng.random() < config.RETURN_P:
                back = candidates(start["id"])
                if back:
                    returned_done = True
                    returned = True
                    cur_id = start["id"]
        cands = candidates(cur_id)
        if not cands:
            res.stop = "no_threads"
            break
        t, c, _ = _softmax_pick(rng, cands, [x[2] for x in cands])
        s = sim_of(c["id"])
        step = Step(card=_card_dict(c), kind=t["kind"], because=t["because"],
                    thread_id=t["id"], sim=s, returned=returned)
        if not res.fallback and s is not None:
            if s < config.OFFTOPIC_STOP:
                res.stop = "offtopic"
                break
            if s < config.OFFTOPIC_DRIFT:
                step.drift = True
                res.steps.append(step)
                used_threads.append(t["id"])
                light *= config.DECAY
                res.stop = "drift"
                break
        res.steps.append(step)
        used_threads.append(t["id"])
        visited.add(c["id"])
        cur_id = c["id"]
        light *= config.DECAY
        if light < config.LIGHT_MIN:
            res.stop = "light"
            break

    res.light = light
    if used_threads and not dry_run:
        ts = now_str()
        for tid in used_threads:
            conn.execute("UPDATE threads SET walked = walked + 1, last_walked = ? WHERE id = ?",
                         (ts, tid))
        conn.commit()
    return res


# ---------------------------------------------------------------- format

KIND_ZH = {
    "same_time": "同一段时间",
    "same_person": "同一个人",
    "same_thing": "同一件事",
    "same_feel": "同一种感觉",
    "same_concept": "同一个念头",
}


def _because_text(step: Step) -> str:
    b = (step.because or "").strip()
    if b.startswith("因为"):
        b = b[2:].strip()
    return b or KIND_ZH.get(step.kind, step.kind or "")


def _dream_mark(card) -> str:
    # a card about a dream I told is never a fact; say so every time it surfaces
    return "【梦】" if card.get("realm") == "dream" else ""


def format_walk(res: WalkResult) -> str:
    if not res.walked:
        return f"没走：{res.reason}"
    lines = []
    s = res.start
    tail = f"(相似 {s.sim:.2f})" if s.sim is not None else "(关键词碰到的)"
    lines.append(f"起点 #{s.card['id']} {_dream_mark(s.card)}「{s.card['quote']}」 {tail}")
    for st in res.steps:
        if st.returned:
            lines.append(f"  ↩ 回到起点 #{s.card['id']}，换一条线")
        line = f"  → 因为{_because_text(st)}（{st.kind}）：#{st.card['id']} {_dream_mark(st.card)}「{st.card['quote']}」"
        if st.card.get("feeling"):
            line += f"  我当时：{st.card['feeling']}"
        if st.drift:
            line += "  [走神想到的]"
        lines.append(line)
    stop = {
        "light": f"停：亮度 {res.light:.2f}",
        "max_steps": f"停：走满{len(res.steps)}步",
        "no_threads": "停：没有线了",
        "offtopic": "停：跑题",
        "drift": "停：跑题",
    }.get(res.stop, f"停：{res.stop}")
    lines.append(stop)
    if res.fallback:
        lines.append("（嵌入桥没接上，用的关键词起步，没查跑题）")
    return "\n".join(lines)
