"""wandering tests. Fixtures are small and fictional. Run: py -m unittest discover -s tests"""
import json
import math
import os
import sqlite3
import tempfile
import unittest
from datetime import date

import numpy as np

from wandering import config, db
from wandering.load import load_jsonl, reembed
from wandering.walk import format_walk, walk

NOW = date(2026, 10, 17)
Q = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def v(sim, axis=1):
    """Unit vector with cosine `sim` against Q."""
    out = [sim, 0.0, 0.0]
    out[axis] = math.sqrt(max(0.0, 1 - sim * sim))
    return np.array(out, dtype=np.float32)


def fake_embedder(texts):
    return [Q.copy() for _ in texts]


def dead_embedder(texts):
    return None


def mem():
    return db.connect(":memory:")


def card(conn, quote, sim=None, **kw):
    kw.setdefault("what", f"关于{quote}")
    kw.setdefault("at", "2026-10-10 12:00")
    return db.add_card(conn, quote=quote, vec=None if sim is None else v(sim), **kw)


class SchemaTests(unittest.TestCase):
    def test_gate_and_author_checks(self):
        c = mem()
        with self.assertRaises(ValueError):
            card(c, "x", gate="secret")
        with self.assertRaises(sqlite3.IntegrityError):
            c.execute("INSERT INTO cards(at,quote,what,gate,author,created_at) "
                      "VALUES('2026-01-01','q','w','weird','me','now')")
        with self.assertRaises(sqlite3.IntegrityError):
            c.execute("INSERT INTO cards(at,quote,what,gate,author,created_at) "
                      "VALUES('2026-01-01','q','w','normal','robot','now')")

    def test_source_check(self):
        c = mem()
        card(c, "a", source="relayed:甲")
        card(c, "b", source="external")
        with self.assertRaises(sqlite3.IntegrityError):
            card(c, "c", source="gossip")
        with self.assertRaises(sqlite3.IntegrityError):
            card(c, "d", source="relayed:")

    def test_thread_because_required_and_kind(self):
        c = mem()
        a, b = card(c, "a"), card(c, "b")
        with self.assertRaises(ValueError):
            db.add_thread(c, a, b, "same_time", "  ")
        with self.assertRaises(sqlite3.IntegrityError):
            c.execute("INSERT INTO threads(a,b,kind,because) VALUES(?,?,'same_time','')", (a, b))
        with self.assertRaises(ValueError):
            db.add_thread(c, a, b, "same_color", "都是蓝的")
        with self.assertRaises(sqlite3.IntegrityError):
            c.execute("INSERT INTO threads(a,b,kind,because) VALUES(?,?,'nope','x')", (a, b))
        t = db.add_thread(c, a, b, "same_time", "同一天")
        row = c.execute("SELECT * FROM threads WHERE id=?", (t,)).fetchone()
        self.assertEqual((row["strength"], row["walked"]), (0.7, 0))

    def test_card_sources_kind(self):
        c = mem()
        cid = db.add_card(c, quote="q", what="w", sources=[{"kind": "events", "ref_id": 5}])
        self.assertEqual(c.execute("SELECT count(*) FROM card_sources").fetchone()[0], 1)
        with self.assertRaises(sqlite3.IntegrityError):
            c.execute("INSERT INTO card_sources VALUES(?, 'chats', 1)", (cid,))

    def test_append_only(self):
        c = mem()
        cid = card(c, "原话")
        with self.assertRaises(sqlite3.IntegrityError):
            c.execute("UPDATE cards SET quote='改了' WHERE id=?", (cid,))
        with self.assertRaises(sqlite3.IntegrityError):
            c.execute("DELETE FROM cards WHERE id=?", (cid,))
        db.set_vec(c, cid, v(0.5))  # vec is the one allowed mutation
        self.assertIsNotNone(db.get_card(c, cid)["vec"])
        self.assertFalse(hasattr(db, "update_card"))

    def test_fts_trigram(self):
        c = mem()
        card(c, "下雨天一起吃火锅")
        rows = c.execute("SELECT rowid FROM cards_fts WHERE cards_fts MATCH '\"吃火锅\"'").fetchall()
        self.assertEqual(len(rows), 1)


class TaggerRuleTests(unittest.TestCase):
    def test_tagger_cannot_have_feeling(self):
        c = mem()
        with self.assertRaises(ValueError):
            db.add_card(c, quote="q", what="w", author="tagger", feeling="开心")
        with self.assertRaises(sqlite3.IntegrityError):
            c.execute("INSERT INTO cards(at,quote,what,feeling,author,created_at) "
                      "VALUES('2026-01-01','q','w','开心','tagger','now')")
        cid = db.add_card(c, quote="q", what="w", author="tagger", why="她提过")
        self.assertIsNone(db.get_card(c, cid)["feeling"])
        db.add_card(c, quote="q2", what="w", author="me", feeling="开心")


def chain(conn, sims, gates=None, start_sim=0.9):
    """start card + a linear chain of cards with given sims. returns ids."""
    gates = gates or ["normal"] * len(sims)
    ids = [card(conn, "起点", start_sim)]
    for i, (s, g) in enumerate(zip(sims, gates)):
        ids.append(card(conn, f"卡{i}", s, gate=g))
        db.add_thread(conn, ids[-2], ids[-1], "same_thing", f"因为线{i}")
    return ids


class WalkTests(unittest.TestCase):
    def test_skip_and_low_start(self):
        c = mem()
        chain(c, [0.3], start_sim=0.4)
        r = walk(c, "q", gate_mode="skip", embedder=fake_embedder)
        self.assertFalse(r.walked)
        r = walk(c, "q", embedder=fake_embedder, now=NOW)
        self.assertFalse(r.walked)
        self.assertIn("起步线", r.reason)

    def test_quote_vec_rescues_start(self):
        # quote+what diluted to 0.4, the quote alone is close: start uses the max
        c = mem()
        cid = card(c, "你不要离开我", sim=0.4)
        r = walk(c, "q", embedder=fake_embedder, now=NOW)
        self.assertFalse(r.walked)
        db.set_vec(c, cid, v(0.4), v(0.8))
        r = walk(c, "q", embedder=fake_embedder, now=NOW)
        self.assertTrue(r.walked)
        self.assertAlmostEqual(r.start.sim, 0.8, places=4)

    def test_determinism_with_seed(self):
        def build():
            c = mem()
            s = card(c, "起点", 0.9)
            for i in range(6):
                n = card(c, f"邻{i}", 0.7, feeling="想起来了" if i % 2 else None)
                db.add_thread(c, s, n, "same_feel", f"感觉{i}", strength=0.5 + 0.05 * i)
                m = card(c, f"远{i}", 0.6)
                db.add_thread(c, n, m, "same_person", f"人{i}")
            return c
        outs = []
        for _ in range(3):
            r = walk(build(), "q", seed=42, embedder=fake_embedder, now=NOW, dry_run=True)
            outs.append([st.card["id"] for st in r.steps])
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[1], outs[2])
        self.assertTrue(len(outs[0]) >= 1)
        varied = {tuple(st.card["id"] for st in walk(build(), "q", seed=s, embedder=fake_embedder,
                                                      now=NOW, dry_run=True).steps)
                  for s in range(30)}
        self.assertGreater(len(varied), 1)

    def test_max_steps_and_walked_counter(self):
        c = mem()
        ids = chain(c, [0.8] * 6)
        r = walk(c, "q", seed=1, embedder=fake_embedder, now=NOW)
        self.assertEqual(len(r.steps), config.MAX_STEPS)
        self.assertEqual(r.stop, "max_steps")
        self.assertEqual([s.card["id"] for s in r.steps], ids[1:5])
        walked = [row[0] for row in c.execute("SELECT walked FROM threads ORDER BY id")]
        self.assertEqual(walked, [1, 1, 1, 1, 0, 0])
        r2 = walk(c, "q", seed=1, embedder=fake_embedder, now=NOW, dry_run=True)
        walked2 = [row[0] for row in c.execute("SELECT walked FROM threads ORDER BY id")]
        self.assertEqual(walked, walked2)
        self.assertIn("停：走满4步", format_walk(r2))

    def test_light_stop(self):
        c = mem()
        chain(c, [0.8] * 6)
        orig = config.DECAY
        try:
            config.DECAY = 0.5   # 1 -> .5 -> .25 < .3
            r = walk(c, "q", seed=1, embedder=fake_embedder, now=NOW, dry_run=True)
        finally:
            config.DECAY = orig
        self.assertEqual((len(r.steps), r.stop), (2, "light"))
        self.assertIn("停：亮度 0.25", format_walk(r))

    def test_no_threads(self):
        c = mem()
        chain(c, [0.8])
        r = walk(c, "q", embedder=fake_embedder, now=NOW, dry_run=True)
        self.assertEqual((len(r.steps), r.stop), (1, "no_threads"))
        self.assertIn("停：没有线了", format_walk(r))

    def test_intimate_excluded_in_normal(self):
        c = mem()
        s = card(c, "起点", 0.7)
        hot = card(c, "亲密的", 0.99, gate="intimate")
        db.add_thread(c, s, hot, "same_person", "她")
        r = walk(c, "q", embedder=fake_embedder, now=NOW, dry_run=True)
        self.assertEqual(r.start.card["id"], s)       # intimate never a start in normal
        self.assertEqual(r.steps, [])
        r = walk(c, "q", gate_mode="intimate", embedder=fake_embedder, now=NOW, dry_run=True)
        self.assertEqual(r.start.card["id"], hot)
        self.assertEqual([x.card["id"] for x in r.steps], [s])

    def test_affection_allows_intimate_not_explicit(self):
        c = mem()
        s = card(c, "起点", 0.7)
        soft = card(c, "小名", 0.6, gate="intimate")
        hot = card(c, "露骨的", 0.99, gate="explicit")
        db.add_thread(c, s, soft, "same_person", "她")
        db.add_thread(c, s, hot, "same_person", "她")
        r = walk(c, "q", gate_mode="affection", embedder=fake_embedder, now=NOW, dry_run=True)
        ids = [r.start.card["id"]] + [x.card["id"] for x in r.steps]
        self.assertNotIn(hot, ids)
        self.assertIn(soft, ids)
        r = walk(c, "q", gate_mode="intimate", embedder=fake_embedder, now=NOW, dry_run=True)
        self.assertEqual(r.start.card["id"], hot)

    def test_tender_start_only(self):
        c = mem()
        s = card(c, "起点", 0.7)
        t = card(c, "柔软的", 0.95, gate="tender")
        n = card(c, "普通的", 0.8)
        db.add_thread(c, s, t, "same_feel", "一样难过", strength=5.0)
        db.add_thread(c, s, n, "same_thing", "同一件事", strength=0.1)
        for seed in range(20):
            r = walk(c, "q", seed=seed, embedder=fake_embedder, now=NOW, dry_run=True)
            # tender sim 0.95 >= TENDER_START_MIN -> it is the start
            self.assertEqual(r.start.card["id"], t)
        c2 = mem()
        s = card(c2, "起点", 0.7)
        t = card(c2, "柔软的", 0.55, gate="tender")   # below TENDER_START_MIN, above START_MIN
        db.add_thread(c2, s, t, "same_feel", "一样难过", strength=5.0)
        for seed in range(20):
            r = walk(c2, "q", seed=seed, embedder=fake_embedder, now=NOW, dry_run=True)
            self.assertEqual(r.start.card["id"], s)
            self.assertNotIn(t, [x.card["id"] for x in r.steps])
            self.assertEqual(r.stop, "no_threads")

    def test_offtopic_stop(self):
        c = mem()
        ids = chain(c, [0.8, 0.2, 0.8])
        r = walk(c, "q", embedder=fake_embedder, now=NOW, dry_run=True)
        self.assertEqual([x.card["id"] for x in r.steps], [ids[1]])
        self.assertEqual(r.stop, "offtopic")
        self.assertIn("停：跑题", format_walk(r))

    def test_offtopic_drift(self):
        c = mem()
        ids = chain(c, [0.8, 0.3, 0.8])
        r = walk(c, "q", embedder=fake_embedder, now=NOW, dry_run=True)
        self.assertEqual([x.card["id"] for x in r.steps], [ids[1], ids[2]])
        self.assertTrue(r.steps[-1].drift)
        self.assertEqual(r.stop, "drift")
        self.assertIn("[走神想到的]", format_walk(r))

    def test_fallback_when_embedder_fails(self):
        c = mem()
        s = card(c, "那天我们去看海")
        far = card(c, "完全无关的数学题")
        db.add_thread(c, s, far, "same_time", "同一天")
        r = walk(c, "看海", embedder=dead_embedder, now=NOW, dry_run=True)
        self.assertTrue(r.walked and r.fallback)
        self.assertEqual(r.start.card["id"], s)
        self.assertEqual([x.card["id"] for x in r.steps], [far])  # no off-topic check
        out = format_walk(r)
        self.assertIn("关键词", out)
        r = walk(c, "海", embedder=dead_embedder, now=NOW, dry_run=True)  # <3 chars -> LIKE
        self.assertEqual(r.start.card["id"], s)
        r = walk(c, "火星基地", embedder=dead_embedder, now=NOW, dry_run=True)
        self.assertFalse(r.walked)

    def test_return_to_start(self):
        c = mem()
        s = card(c, "起点", 0.9)
        a = card(c, "甲", 0.8); b = card(c, "乙", 0.8); a2 = card(c, "甲二", 0.8)
        db.add_thread(c, s, a, "same_time", "同一天", strength=3.0)
        db.add_thread(c, a, a2, "same_time", "接着那天")
        db.add_thread(c, s, b, "same_person", "因为都有甲", strength=0.1)
        orig = config.RETURN_P
        try:
            config.RETURN_P = 1.0
            r = walk(c, "q", seed=0, embedder=fake_embedder, now=NOW, dry_run=True)
        finally:
            config.RETURN_P = orig
        flags = [x.returned for x in r.steps]
        self.assertEqual(flags.count(True), 1)
        ret = r.steps[flags.index(True)]
        self.assertEqual(ret.card["id"], b)
        out = format_walk(r)
        self.assertIn("回到起点", out)
        self.assertIn("因为都有甲（same_person）", out)  # leading 因为 stripped
        self.assertNotIn("因为因为", out)

    def test_salience_recency_bias(self):
        c = mem()
        s = card(c, "起点", 0.9)
        rich = card(c, "有感觉的近事", 0.8, feeling="心里一热", why="她记得", at="2026-10-16 09:00")
        poor = card(c, "很久以前", 0.8, at="2024-01-01 09:00")
        db.add_thread(c, s, rich, "same_feel", "一样")
        db.add_thread(c, s, poor, "same_feel", "一样")
        first = [walk(c, "q", seed=k, embedder=fake_embedder, now=NOW, dry_run=True).steps[0].card["id"]
                 for k in range(200)]
        self.assertGreater(first.count(rich), first.count(poor))

    def test_format_feeling(self):
        c = mem()
        s = card(c, "起点", 0.9)
        n = card(c, "下一张", 0.8, feeling="有点想笑")
        db.add_thread(c, s, n, "same_time", "同一天")
        out = format_walk(walk(c, "q", embedder=fake_embedder, now=NOW, dry_run=True))
        self.assertTrue(out.startswith(f"起点 #{s} 「起点」 (相似 0.90)"))
        self.assertIn(f"  → 因为同一天（same_time）：#{n} 「下一张」  我当时：有点想笑", out)


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "cards.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, objs):
        with open(self.path, "w", encoding="utf-8") as f:
            for o in objs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

    def test_roundtrip(self):
        self.write([
            {"type": "card", "key": "c1", "at": "2026-09-01 20:00", "quote": "今晚月亮好圆",
             "what": "中秋看月亮", "feeling": "安静", "gate": "normal", "author": "me",
             "sources": [{"kind": "windows", "ref_id": 3}]},
            {"type": "card", "key": "c2", "at": "2026-09-01 21:00", "quote": "月饼太甜了",
             "what": "吃月饼", "author": "tagger", "source": "relayed:甲"},
            {"type": "thread", "a": "c1", "b": "c2", "kind": "same_time", "because": "同一个晚上",
             "strength": 0.9},
        ])
        conn = db.connect(os.path.join(self.tmp.name, "t.sqlite"))
        warnings = []
        st = load_jsonl(conn, self.path, embedder=fake_embedder, warn=warnings.append)
        self.assertEqual((st["cards"], st["threads"], st["embedded"]), (2, 1, 2))
        self.assertEqual(warnings, [])
        c1, c2 = st["keys"]["c1"], st["keys"]["c2"]
        self.assertEqual(db.get_card(conn, c2)["source"], "relayed:甲")
        t = db.threads_of(conn, c1)[0]
        self.assertEqual((t["a"], t["b"], t["strength"]), (c1, c2, 0.9))
        r = walk(conn, "月亮", embedder=fake_embedder, now=NOW, dry_run=True)
        self.assertTrue(r.walked)
        conn.close()

    def test_bridge_down_warns_then_reembed(self):
        self.write([{"type": "card", "key": "x", "quote": "q", "what": "w"}])
        conn = db.connect(":memory:")
        warnings = []
        st = load_jsonl(conn, self.path, embedder=dead_embedder, warn=warnings.append)
        self.assertEqual(st["embedded"], 0)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(reembed(conn, embedder=fake_embedder), (1, 1))
        self.assertEqual(reembed(conn, embedder=fake_embedder), (0, 0))

    def test_bad_file_rolls_back(self):
        self.write([
            {"type": "card", "key": "ok", "quote": "q", "what": "w"},
            {"type": "card", "key": "bad", "quote": "q", "what": "w", "author": "tagger",
             "feeling": "偷偷写感受"},
        ])
        conn = db.connect(":memory:")
        with self.assertRaises(ValueError):
            load_jsonl(conn, self.path, embedder=fake_embedder)
        self.assertEqual(conn.execute("SELECT count(*) FROM cards").fetchone()[0], 0)
        self.write([{"type": "card", "key": "a", "quote": "q", "what": "w"},
                    {"type": "thread", "a": "a", "b": "zzz", "kind": "same_time", "because": "x"}])
        with self.assertRaises(ValueError):
            load_jsonl(conn, self.path, embedder=fake_embedder)
        self.assertEqual(conn.execute("SELECT count(*) FROM cards").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()


class GateClassifyTest(unittest.TestCase):
    def test_classify(self):
        from wandering import gate
        self.assertEqual(gate.classify("嗯"), "skip")
        self.assertEqual(gate.classify("好！"), "skip")
        self.assertEqual(gate.classify("在吗？"), "normal")
        self.assertEqual(gate.classify("晚安灯泡"), "affection")
        self.assertEqual(gate.classify("今天天好蓝啊"), "normal")
        self.assertEqual(gate.classify("想你", affection_words=[]), "normal")


class PrivacyTests(unittest.TestCase):
    def test_gate_negation(self):
        from wandering import gate
        self.assertEqual(gate.classify("我不想你了"), "normal")
        self.assertEqual(gate.classify("别叫我宝宝"), "normal")
        self.assertEqual(gate.classify("不知道，想你"), "affection")
        self.assertEqual(gate.classify("想你，不想睡"), "affection")

    def test_purge_card(self):
        c = mem()
        a = card(c, "不该存的私事")
        b = card(c, "别的")
        db.add_thread(c, a, b, "same_thing", "连着")
        db.set_intensity(c, a, 3)
        self.assertTrue(db.purge_card(c, a))
        self.assertIsNone(db.get_card(c, a))
        self.assertEqual(c.execute("SELECT count(*) FROM threads").fetchone()[0], 0)
        self.assertEqual(c.execute("SELECT count(*) FROM intensity").fetchone()[0], 0)
        hits = c.execute("SELECT rowid FROM cards_fts WHERE cards_fts MATCH ?", ("该存的",)).fetchall()
        self.assertEqual(hits, [])
        with self.assertRaises(sqlite3.IntegrityError):  # append-only is back
            c.execute("DELETE FROM cards WHERE id = ?", (b,))
        self.assertFalse(db.purge_card(c, a))

    def test_purge_leaves_no_text_in_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.sqlite")
            c = db.connect(p)
            a = card(c, "SECRETMARKER123")
            c.commit()
            db.purge_card(c, a)
            c.close()
            with open(p, "rb") as f:
                self.assertNotIn(b"SECRETMARKER123", f.read())

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_db_file_private(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sub", "m.sqlite")
            db.connect(p).close()
            self.assertEqual(os.stat(p).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(os.path.dirname(p)).st_mode & 0o777, 0o700)

    def test_embed_url_policy(self):
        from wandering import embed
        self.assertTrue(embed.url_ok("http://127.0.0.1:8749/v1/embeddings"))
        self.assertTrue(embed.url_ok("http://localhost/v1/embeddings"))
        self.assertTrue(embed.url_ok("https://api.example.com/v1/embeddings"))
        self.assertFalse(embed.url_ok("http://api.example.com/v1/embeddings"))
        self.assertIsNone(embed.embed(["x"], url="http://api.example.com/v1/embeddings"))
