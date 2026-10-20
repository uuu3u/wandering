"""JSONL loader and re-embedding."""
from __future__ import annotations

import json
import sys

from . import db
from .embed import embed as default_embed, embed_batched

CARD_FIELDS = ("at", "period", "quote", "source", "tone", "what", "feeling", "why",
               "not_this", "gate", "author", "charge", "realm")


def load_jsonl(conn, path, embedder=default_embed, warn=None) -> dict:
    """Load cards/threads from JSONL. All-or-nothing. Returns stats incl. key->id map."""
    warn = warn or (lambda m: print(m, file=sys.stderr))
    keymap: dict[str, int] = {}
    pending_threads = []
    new_ids = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            for lineno, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw or raw.startswith("//"):
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"line {lineno}: bad JSON: {e}") from None
                typ = obj.get("type")
                if typ == "card":
                    kwargs = {k: obj[k] for k in CARD_FIELDS if k in obj and obj[k] is not None}
                    try:
                        cid = db.add_card(conn, sources=obj.get("sources"), commit=False, **kwargs)
                    except Exception as e:
                        raise ValueError(f"line {lineno}: card rejected: {e}") from None
                    key = obj.get("key")
                    if key is not None:
                        if str(key) in keymap:
                            raise ValueError(f"line {lineno}: duplicate key {key!r}")
                        keymap[str(key)] = cid
                    new_ids.append(cid)
                elif typ == "thread":
                    pending_threads.append((lineno, obj))
                else:
                    raise ValueError(f"line {lineno}: unknown type {typ!r}")

        def resolve(ref, lineno):
            if str(ref) in keymap:
                return keymap[str(ref)]
            if isinstance(ref, int) and db.get_card(conn, ref) is not None:
                return ref
            raise ValueError(f"line {lineno}: unknown card key {ref!r}")

        n_threads = 0
        for lineno, obj in pending_threads:
            try:
                db.add_thread(conn, resolve(obj["a"], lineno), resolve(obj["b"], lineno),
                              obj["kind"], obj.get("because", ""),
                              obj.get("strength", 0.7), commit=False)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"line {lineno}: thread rejected: {e}") from None
            n_threads += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    embedded = embed_cards(conn, new_ids, embedder)
    if new_ids and embedded == 0:
        warn("warning: embedding bridge unavailable; cards stored without vectors "
             "(run `py -m wandering reembed` later)")
    return {"cards": len(new_ids), "threads": n_threads, "embedded": embedded, "keys": keymap}


def embed_cards(conn, ids, embedder=default_embed) -> int:
    if not ids or embedder is None:
        return 0
    rows = [db.get_card(conn, i) for i in ids]
    texts = [db.card_text_for_embedding(r) for r in rows] + [r["quote"] for r in rows]
    vecs = embed_batched(texts, embedder)
    if vecs is None:
        return 0
    n = len(rows)
    for r, v, qv in zip(rows, vecs[:n], vecs[n:]):
        db.set_vec(conn, r["id"], v, qv, commit=False)
    conn.commit()
    return len(rows)


def reembed(conn, embedder=default_embed) -> tuple[int, int]:
    ids = [r[0] for r in conn.execute("SELECT id FROM cards WHERE vec IS NULL OR quote_vec IS NULL ORDER BY id")]
    return embed_cards(conn, ids, embedder), len(ids)
