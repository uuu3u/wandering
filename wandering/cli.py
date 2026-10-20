"""Command line: py -m wandering <command> ..."""
from __future__ import annotations

import argparse
import json
import sys

from . import config, db
from .embed import embed
from .load import embed_cards, load_jsonl, reembed
from .walk import format_walk, walk
from . import gate


def resolve_gate(mode, query):
    if mode != "auto":
        return mode
    return gate.classify(query)


def cmd_walk(a, conn):
    gate = resolve_gate(a.gate, a.text)
    res = walk(conn, a.text, gate_mode=gate, seed=a.seed, dry_run=a.dry_run)
    if a.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_walk(res))


def cmd_add(a, conn):
    cid = db.add_card(conn, quote=a.quote, what=a.what, at=a.at, period=a.period,
                      source=a.source, tone=a.tone, feeling=a.feeling, why=a.why,
                      not_this=a.not_this, gate=a.gate, author=a.author)
    ok = embed_cards(conn, [cid])
    print(f"added card #{cid}" + ("" if ok else " (no vector: embedding bridge down)"))


def cmd_link(a, conn):
    for cid in (a.a, a.b):
        if db.get_card(conn, cid) is None:
            sys.exit(f"no card #{cid}")
    tid = db.add_thread(conn, a.a, a.b, a.kind, a.because, a.strength)
    print(f"thread #{tid}: #{a.a} <-> #{a.b} ({a.kind})")


def cmd_show(a, conn):
    c = db.get_card(conn, a.id)
    if c is None:
        sys.exit(f"no card #{a.id}")
    print(f"#{c['id']}  {c['at']}  [{c['gate']}/{c['author']}/{c['source']}]"
          + (f"  period={c['period']}" if c['period'] else "")
          + (f"  tone={c['tone']}" if c['tone'] else ""))
    print(f"  「{c['quote']}」")
    print(f"  what: {c['what']}")
    for k in ("feeling", "why", "not_this"):
        if c[k]:
            print(f"  {k}: {c[k]}")
    print(f"  vec: {'yes' if c['vec'] is not None else 'no'}")
    for t in db.threads_of(conn, a.id):
        other = t["b"] if t["a"] == a.id else t["a"]
        oc = db.get_card(conn, other)
        print(f"  — {t['kind']} ({t['strength']:.2f}, walked {t['walked']}) → #{other} "
              f"「{oc['quote'] if oc else '?'}」  因为: {t['because']}")


def cmd_stats(a, conn):
    q = lambda sql: conn.execute(sql).fetchall()
    n = q("SELECT count(*), sum(vec IS NOT NULL) FROM cards")[0]
    print(f"cards: {n[0]}  (with vec: {n[1] or 0})")
    for r in q("SELECT gate, author, count(*) FROM cards GROUP BY gate, author"):
        print(f"  {r[0]:<8} {r[1]:<6} {r[2]}")
    print(f"threads: {q('SELECT count(*) FROM threads')[0][0]}")
    for r in q("SELECT kind, count(*), sum(walked) FROM threads GROUP BY kind"):
        print(f"  {r[0]:<13} {r[1]}  walked {r[2] or 0}")
    top = q("SELECT id, a, b, kind, walked FROM threads WHERE walked > 0 "
            "ORDER BY walked DESC LIMIT 5")
    if top:
        print("most walked:")
        for r in top:
            print(f"  thread #{r[0]} #{r[1]}<->#{r[2]} {r[3]} x{r[4]}")


def cmd_load(a, conn):
    st = load_jsonl(conn, a.file)
    print(f"loaded {st['cards']} cards, {st['threads']} threads, embedded {st['embedded']}")


def cmd_reembed(a, conn):
    done, total = reembed(conn)
    print(f"embedded {done}/{total} cards missing vectors"
          + ("" if done == total else " (bridge down?)"))


def cmd_purge(a, conn):
    if not a.yes:
        print(f"this permanently erases card #{a.id} and its threads; add --yes to confirm")
        return
    print(f"purged #{a.id}" if db.purge_card(conn, a.id) else f"no card #{a.id}")


def build_parser():
    p = argparse.ArgumentParser(prog="wandering", description="散步: walk-based memory recall")
    p.add_argument("--db", default=None, help=f"database path (default {config.DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_db(sp):
        sp.add_argument("--db", default=argparse.SUPPRESS)

    w = sub.add_parser("walk"); add_db(w)
    w.add_argument("text")
    w.add_argument("--gate", choices=["normal", "affection", "intimate", "skip", "auto"], default="auto")
    w.add_argument("--seed", type=int, default=None)
    w.add_argument("--json", action="store_true")
    w.add_argument("--dry-run", action="store_true")
    w.set_defaults(fn=cmd_walk)

    ad = sub.add_parser("add"); add_db(ad)
    ad.add_argument("--quote", required=True)
    ad.add_argument("--what", required=True)
    ad.add_argument("--at"); ad.add_argument("--period")
    ad.add_argument("--source", default="self"); ad.add_argument("--tone")
    ad.add_argument("--feeling"); ad.add_argument("--why"); ad.add_argument("--not-this")
    ad.add_argument("--gate", choices=list(db.GATES), default="normal")
    ad.add_argument("--author", choices=list(db.AUTHORS), default="me")
    ad.set_defaults(fn=cmd_add)

    ln = sub.add_parser("link"); add_db(ln)
    ln.add_argument("a", type=int); ln.add_argument("b", type=int)
    ln.add_argument("kind", choices=list(db.THREAD_KINDS))
    ln.add_argument("because")
    ln.add_argument("--strength", type=float, default=0.7)
    ln.set_defaults(fn=cmd_link)

    sh = sub.add_parser("show"); add_db(sh)
    sh.add_argument("id", type=int); sh.set_defaults(fn=cmd_show)

    stt = sub.add_parser("stats"); add_db(stt); stt.set_defaults(fn=cmd_stats)

    ld = sub.add_parser("load"); add_db(ld)
    ld.add_argument("file"); ld.set_defaults(fn=cmd_load)

    re_ = sub.add_parser("reembed"); add_db(re_); re_.set_defaults(fn=cmd_reembed)

    pg = sub.add_parser("purge", help="erase a card for good (private text stored by mistake)")
    add_db(pg)
    pg.add_argument("id", type=int); pg.add_argument("--yes", action="store_true")
    pg.set_defaults(fn=cmd_purge)
    return p


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    a = build_parser().parse_args(argv)
    conn = db.connect(a.db)
    try:
        a.fn(a, conn)
    except (ValueError, OSError) as e:
        sys.exit(f"error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
