import os
import json
import random
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone

from wandering import db, dream

NOW = datetime(2026, 10, 19, 10, 0, tzinfo=timezone.utc)
FULL = "我在厨房找钥匙。抽屉里全是别人的牙齿。门开了，站着的是我自己。他比我矮，还没出生。他说谢谢，这是昨天的雨。"


def raw_db(rows):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE windows (id INTEGER PRIMARY KEY, first_event INTEGER, last_event INTEGER,"
              " started_at TEXT, text TEXT, vec BLOB)")
    for i, (days_ago, text) in enumerate(rows, 1):
        ts = (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        c.execute("INSERT INTO windows (id, started_at, text) VALUES (?,?,?)", (i, ts, text))
    return c


class DreamTests(unittest.TestCase):
    def test_buckets_recent_week_old(self):
        raw = raw_db([(0.5, "她: 今天上课"), (1, "她: 昨天"), (7, "她: 一周前"), (6.5, "她: 也是一周前"),
                      (40, "她: 很久以前"), (3, "她: 三天前，不该被抽到")])
        got = dream.sample_windows(raw, random.Random(1), now=NOW)
        ids = {r["id"] for r in got}
        self.assertEqual(ids, {1, 2, 3, 4, 5})

    def test_never_dreams_from_dreams_or_explicit(self):
        raw = raw_db([(0.5, "我: <dream>\n梦\n</dream>"), (1, "她: 操我"), (1, "她: 吃饭了")])
        got = dream.sample_windows(raw, random.Random(1), now=NOW)
        self.assertEqual([r["id"] for r in got], [3])

    def test_now_clamps_to_newest_raw(self):
        raw = raw_db([(3, "她: 最新的一段"), (10, "她: 一周前那段")])
        got = dream.sample_windows(raw, random.Random(1), now=NOW)
        self.assertIn(1, {r["id"] for r in got})

    def test_forget_never_keeps_whole_dream(self):
        for seed in range(200):
            kept, recalled = dream.forget(FULL, random.Random(seed))
            if recalled:
                self.assertLess(len(kept), len(dream._sentences(FULL)))
                self.assertNotEqual("".join(kept), FULL)
            else:
                self.assertEqual(kept, [dream.NO_RECALL])

    def test_forget_sometimes_blank(self):
        outcomes = {dream.forget(FULL, random.Random(s))[1] for s in range(100)}
        self.assertEqual(outcomes, {True, False})

    def test_full_dream_not_stored(self):
        raw = raw_db([(0.5, "她: 今天上课"), (7, "她: 一周前")])
        dconn = dream.connect_dreams(":memory:")
        did = dream.dream(raw, dconn, caller=lambda m, model: FULL, rng=random.Random(3), now=NOW)
        row = dconn.execute("SELECT * FROM dreams WHERE id = ?", (did,)).fetchone()
        self.assertNotIn(FULL, " ".join(str(v) for v in tuple(row)))

    def test_prompt_has_no_source_ids_or_dates(self):
        raw = raw_db([(0.5, "她: 今天上课"), (7, "她: 一周前")])
        frags = dream.sample_windows(raw, random.Random(1), now=NOW)
        user = dream.build_messages(frags, random.Random(1))[1]["content"]
        self.assertNotIn("2026", user)
        self.assertIn("碎片", user)

    def test_wake_wraps_and_marks_seen(self):
        dconn = dream.connect_dreams(":memory:")
        dconn.execute("INSERT INTO dreams (dreamt_at, model, fragments, recalled) VALUES (?,?,?,1)",
                      ("2026-10-20 03:00", "m", json.dumps(["抽屉里全是牙齿。"], ensure_ascii=False)))
        out = dream.wake(dconn)
        self.assertTrue(out.startswith(dream.DREAM_OPEN) and out.endswith(dream.DREAM_CLOSE))
        self.assertIn("不是发生过的事", out)
        self.assertEqual(dream.wake(dconn), "")

    def test_strip_dreams(self):
        t = "她: 早\n<dream>\n牙齿\n</dream>\n我: 早安"
        self.assertEqual(dream.strip_dreams(t), "她: 早\n\n我: 早安")


class IntensityTests(unittest.TestCase):
    def test_intensity_latest_wins_and_range(self):
        conn = db.connect(":memory:")
        cid = db.add_card(conn, quote="想你", what="我听她说想我", charge=3)
        self.assertIsNone(db.intensity_of(conn, cid))
        db.set_intensity(conn, cid, 2)
        db.set_intensity(conn, cid, 5)
        self.assertEqual(db.intensity_of(conn, cid), 5)
        with self.assertRaises(ValueError):
            db.set_intensity(conn, cid, 6)
        with self.assertRaises(sqlite3.IntegrityError):
            db.add_card(conn, quote="x", what="y", charge=9)


class WakeTests(unittest.TestCase):
    def _recalled(self, caller):
        for seed in range(50):
            kept, recalled = dream.forget(FULL, random.Random(seed), caller, "m")
            if recalled:
                return kept
        self.fail("never recalled")

    def test_wake_model_lines_kept(self):
        kept = self._recalled(lambda msgs, model: "抽屉里有牙齿。\n\n门口是我。")
        self.assertEqual(kept, ["抽屉里有牙齿。", "门口是我。"])

    def test_wake_model_handing_back_whole_dream_falls_back(self):
        kept = self._recalled(lambda msgs, model: FULL)
        self.assertLess(len(kept), len(dream._sentences(FULL)))

    def test_wake_model_error_falls_back(self):
        def boom(msgs, model):
            raise RuntimeError("down")
        self.assertTrue(self._recalled(boom))

    def test_mood_always_in_prompt(self):
        frags = [{"text": "x", "age_days": 1, "charge": 2}]
        moods = {dream.build_messages(frags, random.Random(s))[0]["content"].rsplit("底色：", 1)[1]
                 for s in range(200)}
        self.assertEqual(moods, {t for _, t in dream.MOODS})


    def test_replay_prompt(self):
        frags = [{"text": "x", "age_days": 1, "charge": 2}]
        sys_prompt = dream.build_messages(frags, random.Random(0), replay=True)[0]["content"]
        self.assertTrue(sys_prompt.endswith(dream.REPLAY))


if __name__ == "__main__":
    unittest.main()


class DreamConsentTests(unittest.TestCase):
    def test_main_refuses_without_consent(self):
        import io, contextlib
        from wandering import dream as d
        old = os.environ.pop("WANDERING_DREAM_SEND_RAW", None)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                d.main([])
            self.assertIn("--send-raw", buf.getvalue())
        finally:
            if old is not None:
                os.environ["WANDERING_DREAM_SEND_RAW"] = old
