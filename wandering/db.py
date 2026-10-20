"""Schema and card/thread storage. Cards are append-only."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

GATES = ("normal", "tender", "intimate", "explicit")
AUTHORS = ("me", "tagger")
THREAD_KINDS = ("same_time", "same_person", "same_thing", "same_feel", "same_concept")
SOURCE_KINDS = ("events", "windows")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id         INTEGER PRIMARY KEY,
    at         TEXT NOT NULL,
    period     TEXT,
    quote      TEXT NOT NULL CHECK (length(quote) > 0),
    source     TEXT NOT NULL DEFAULT 'self'
               CHECK (source IN ('self', 'external') OR source LIKE 'relayed:_%'),
    tone       TEXT,
    what       TEXT NOT NULL,
    feeling    TEXT,
    why        TEXT,
    not_this   TEXT,
    gate       TEXT NOT NULL DEFAULT 'normal' CHECK (gate IN ('normal', 'tender', 'intimate', 'explicit')),
    author     TEXT NOT NULL DEFAULT 'me' CHECK (author IN ('me', 'tagger')),
    created_at TEXT NOT NULL,
    vec        BLOB,
    quote_vec  BLOB,             -- quote alone; start sim = max(vec, quote_vec)
    charge     INTEGER CHECK (charge BETWEEN 0 AND 5),  -- 语气强度: how heated the text reads; anyone may set
    realm      TEXT NOT NULL DEFAULT 'real' CHECK (realm IN ('real', 'dream')),  -- dream = a dream I told, not a fact
    CHECK (author <> 'tagger' OR feeling IS NULL)
);

CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
    quote, what, feeling,
    content='cards', content_rowid='id', tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS cards_ai AFTER INSERT ON cards BEGIN
    INSERT INTO cards_fts(rowid, quote, what, feeling)
    VALUES (new.id, new.quote, new.what, new.feeling);
END;

-- append-only: content fields can never change (vec may be filled later)
CREATE TRIGGER IF NOT EXISTS cards_no_update
BEFORE UPDATE OF at, period, quote, source, tone, what, feeling, why, not_this,
                 gate, author, created_at ON cards
BEGIN
    SELECT RAISE(ABORT, 'cards are append-only');
END;

CREATE TRIGGER IF NOT EXISTS cards_no_delete BEFORE DELETE ON cards BEGIN
    SELECT RAISE(ABORT, 'cards are append-only');
END;

CREATE TABLE IF NOT EXISTS threads (
    id          INTEGER PRIMARY KEY,
    a           INTEGER NOT NULL REFERENCES cards(id),
    b           INTEGER NOT NULL REFERENCES cards(id),
    kind        TEXT NOT NULL CHECK (kind IN
                ('same_time', 'same_person', 'same_thing', 'same_feel', 'same_concept')),
    because     TEXT NOT NULL CHECK (length(because) > 0),
    strength    REAL NOT NULL DEFAULT 0.7,
    walked      INTEGER NOT NULL DEFAULT 0,
    last_walked TEXT,
    CHECK (a <> b)
);
CREATE INDEX IF NOT EXISTS threads_a ON threads(a);
CREATE INDEX IF NOT EXISTS threads_b ON threads(b);

CREATE TABLE IF NOT EXISTS card_sources (
    card_id INTEGER NOT NULL REFERENCES cards(id),
    kind    TEXT NOT NULL CHECK (kind IN ('events', 'windows')),
    ref_id  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS card_sources_card ON card_sources(card_id);

-- 情感浓度: how strongly *I* felt it. Same rule as feeling: only me, never the tagger.
-- Append-only log; the latest row for a card wins.
CREATE TABLE IF NOT EXISTS intensity (
    id      INTEGER PRIMARY KEY,
    card_id INTEGER NOT NULL REFERENCES cards(id),
    score   INTEGER NOT NULL CHECK (score BETWEEN 0 AND 5),
    set_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS intensity_card ON intensity(card_id);
"""


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _chmod(p, mode):
    # memories are private: owner-only. No-op where the OS ignores it (Windows).
    try:
        os.chmod(p, mode)
    except OSError:
        pass


def connect(path=None) -> sqlite3.Connection:
    from . import config
    path = Path(path) if path else config.DEFAULT_DB
    private = str(path) != ":memory:"
    if private:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _chmod(path.parent, 0o700)
    conn = sqlite3.connect(str(path))
    if private:
        _chmod(path, 0o600)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cards)")}
    if "quote_vec" not in cols:
        conn.execute("ALTER TABLE cards ADD COLUMN quote_vec BLOB")
    if "charge" not in cols:
        conn.execute("ALTER TABLE cards ADD COLUMN charge INTEGER CHECK (charge BETWEEN 0 AND 5)")
    if "realm" not in cols:
        conn.execute("ALTER TABLE cards ADD COLUMN realm TEXT NOT NULL DEFAULT 'real' CHECK (realm IN ('real', 'dream'))")
    return conn


def vec_to_blob(vec) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _blank(s):
    return s is None or (isinstance(s, str) and s.strip() == "")


def add_card(conn, *, quote, what, at=None, period=None, source="self", tone=None,
             feeling=None, why=None, not_this=None, gate="normal", author="me",
             charge=None, realm="real", vec=None, sources=None, commit=True) -> int:
    """Insert a card and return its id. Raises ValueError on rule violations."""
    if author not in AUTHORS:
        raise ValueError(f"author must be one of {AUTHORS}")
    if author == "tagger" and not _blank(feeling):
        raise ValueError("tagger-authored cards must not have a feeling (only 'me' can say how I felt)")
    if author == "tagger":
        feeling = None
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}")
    if realm not in ("real", "dream"):
        raise ValueError("realm must be 'real' or 'dream'")
    if _blank(quote):
        raise ValueError("quote is required")
    if what is None:
        raise ValueError("what is required")
    cur = conn.execute(
        """INSERT INTO cards (at, period, quote, source, tone, what, feeling, why, not_this,
                              gate, author, created_at, vec, charge, realm)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (at or now_str(), period, quote, source or "self", tone, what,
         None if _blank(feeling) else feeling, None if _blank(why) else why,
         None if _blank(not_this) else not_this, gate, author, now_str(), vec_to_blob(vec),
         charge, realm),
    )
    card_id = cur.lastrowid
    for s in sources or ():
        conn.execute("INSERT INTO card_sources (card_id, kind, ref_id) VALUES (?,?,?)",
                     (card_id, s["kind"], int(s["ref_id"])))
    if commit:
        conn.commit()
    return card_id


def set_vec(conn, card_id, vec, quote_vec=None, commit=True):
    """The only permitted mutation of a card: filling its embeddings."""
    conn.execute("UPDATE cards SET vec = ? WHERE id = ?", (vec_to_blob(vec), card_id))
    if quote_vec is not None:
        conn.execute("UPDATE cards SET quote_vec = ? WHERE id = ?", (vec_to_blob(quote_vec), card_id))
    if commit:
        conn.commit()


def add_thread(conn, a, b, kind, because, strength=0.7, commit=True) -> int:
    if kind not in THREAD_KINDS:
        raise ValueError(f"kind must be one of {THREAD_KINDS}")
    if _blank(because):
        raise ValueError("a thread needs a non-empty 'because'")
    cur = conn.execute(
        "INSERT INTO threads (a, b, kind, because, strength) VALUES (?,?,?,?,?)",
        (int(a), int(b), kind, because.strip(), float(strength)),
    )
    if commit:
        conn.commit()
    return cur.lastrowid


def set_intensity(conn, card_id, score, commit=True):
    """Record how strongly I felt a card (0-5). Only I call this; the tagger never does."""
    if not (isinstance(score, int) and 0 <= score <= 5):
        raise ValueError("intensity must be an int 0-5")
    if get_card(conn, card_id) is None:
        raise ValueError(f"no card #{card_id}")
    conn.execute("INSERT INTO intensity (card_id, score, set_at) VALUES (?,?,?)",
                 (card_id, score, now_str()))
    if commit:
        conn.commit()


def intensity_of(conn, card_id):
    r = conn.execute("SELECT score FROM intensity WHERE card_id = ? ORDER BY id DESC LIMIT 1",
                     (card_id,)).fetchone()
    return None if r is None else r[0]


def get_card(conn, card_id):
    return conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()


def threads_of(conn, card_id):
    return conn.execute(
        "SELECT * FROM threads WHERE a = ? OR b = ? ORDER BY id", (card_id, card_id)
    ).fetchall()


def card_text_for_embedding(row) -> str:
    return f"{row['quote']}\n{row['what']}"


def purge_card(conn, card_id) -> bool:
    """Erase a card for good, for private text that should never have been stored.

    The one exception to append-only; ordinary mistakes get a correction card.
    Removes the card, its threads, sources, intensity and search-index rows,
    then VACUUMs so the text is not left behind in free pages.
    """
    row = get_card(conn, card_id)
    if row is None:
        return False
    conn.execute("INSERT INTO cards_fts(cards_fts, rowid, quote, what, feeling) "
                 "VALUES ('delete', ?, ?, ?, ?)",
                 (card_id, row["quote"], row["what"], row["feeling"]))
    conn.execute("DELETE FROM threads WHERE a = ? OR b = ?", (card_id, card_id))
    conn.execute("DELETE FROM card_sources WHERE card_id = ?", (card_id,))
    conn.execute("DELETE FROM intensity WHERE card_id = ?", (card_id,))
    conn.execute("DROP TRIGGER cards_no_delete")
    try:
        conn.execute("DELETE FROM cards WHERE id = ?", (card_id,))
    finally:
        conn.executescript(SCHEMA)  # puts the trigger back
    conn.commit()
    conn.execute("VACUUM")
    return True
