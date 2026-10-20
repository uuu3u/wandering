"""wandering.tagger -- turn a passage into memory cards via an assistant model.

Iron rule: the tagger never writes `feeling`. Any `feeling` key coming back
from a model is dropped and logged.

Usage:
    py -m wandering.tagger FILE [--backend api|none] [--existing cards.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

log = logging.getLogger("wandering.tagger")

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "tagger_prompt.md"
TAGGER_BASE_URL = os.environ.get("WANDERING_TAGGER_BASE_URL", "https://api.deepseek.com")
TAGGER_MODEL = os.environ.get("WANDERING_TAGGER_MODEL", "deepseek-v4-pro")

CARD_FIELDS = ("at", "period", "quote", "source", "tone", "what", "why", "not_this")
ALLOWED_KINDS = {"same_time", "same_person", "same_thing", "same_feel", "same_concept"}
ALLOWED_TONES = {"认真", "玩笑", "气话", "撒娇", "拿不准"}
OPTIONAL_CARD_FIELDS = ("gate_hint", "opaque", "charge", "realm")  # tolerated if present, never required
ALLOWED_GATE_HINTS = {"normal", "tender", "intimate", "explicit", "unsure"}
SOURCE_RE = re.compile(r"^(self|external|relayed:.+)$")
WHAT_VERBS = ("我听ta说", "我听ta转述", "我听她说", "我听她转述", "我听他说", "我听他转述", "我看到", "我猜", "我答应了", "我记错了", "我梦见")


# ---------------------------------------------------------------- prompt / key

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _load_tagger_key() -> str:
    """Tagger API key from the environment. Never logged."""
    key = os.environ.get("WANDERING_TAGGER_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("set WANDERING_TAGGER_API_KEY (or DEEPSEEK_API_KEY)")
    return key


def build_user_message(text: str, existing_cards: list[dict]) -> str:
    parts = ["## 要标注的文字\n", text.strip(), "\n"]
    if existing_cards:
        slim = [{k: c.get(k) for k in ("id", "at", "quote", "what", "why") if k in c}
                for c in existing_cards]
        parts += ["\n## 已有卡片（用它们的 id 连线）\n",
                  json.dumps(slim, ensure_ascii=False, indent=1)]
    else:
        parts.append("\n## 已有卡片\n（无）")
    return "".join(parts)


# ---------------------------------------------------------------- backends

def _call_api(text: str, existing_cards: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_load_tagger_key(), base_url=TAGGER_BASE_URL)
    resp = client.chat.completions.create(
        model=TAGGER_MODEL,
        messages=[
            {"role": "system", "content": load_prompt()},
            {"role": "user", "content": build_user_message(text, existing_cards)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


def _call_none(text: str, existing_cards: list[dict]) -> str:
    """Offline stub: one card quoting the first line, so validation can run."""
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    quote = first.split("：", 1)[-1].split(":", 1)[-1].strip() or None
    return json.dumps({
        "cards": [{
            "at": None, "period": None, "quote": quote, "source": "self",
            "tone": "认真", "what": "我听ta说了一句话（none 后端占位）。",
            "why": "测试脚手架时", "not_this": None,
        }],
        "threads": [],
    }, ensure_ascii=False)


BACKENDS = {"api": _call_api, "none": _call_none}


# ---------------------------------------------------------------- parse / validate

def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def validate(result: dict, text: str, existing_ids: set[str] | None = None) -> dict:
    """Clean a parsed tagger result in place-ish and attach `_flags`.

    - drops any `feeling` key (warns)
    - flags quotes that don't appear verbatim in the text
    - drops threads with bad kind / empty because / unknown endpoints
    """
    existing_ids = existing_ids or set()
    flags: list[str] = []
    cards = result.get("cards") or []
    if not isinstance(cards, list):
        flags.append("cards is not a list; replaced with []")
        cards = []

    clean_cards = []
    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            flags.append(f"new{i}: not an object, dropped")
            continue
        if "feeling" in card:
            log.warning("new%d: tagger emitted 'feeling' -- dropped (iron rule)", i)
            flags.append(f"new{i}: feeling key dropped")
            card.pop("feeling")
        extra = set(card) - set(CARD_FIELDS) - set(OPTIONAL_CARD_FIELDS)
        for k in extra:
            flags.append(f"new{i}: unknown field '{k}' dropped")
            card.pop(k)
        for k in CARD_FIELDS:
            card.setdefault(k, None)

        if "gate_hint" in card and card["gate_hint"] not in ALLOWED_GATE_HINTS:
            flags.append(f"new{i}: bad gate_hint {card['gate_hint']!r}, set to 'unsure'")
            card["gate_hint"] = "unsure"

        if "realm" in card and card["realm"] not in ("real", "dream"):
            flags.append(f"new{i}: bad realm {card['realm']!r}, set to 'dream' (safer)")
            card["realm"] = "dream"
        if "charge" in card and not (isinstance(card["charge"], int) and 0 <= card["charge"] <= 5):
            flags.append(f"new{i}: bad charge {card['charge']!r}, dropped")
            card.pop("charge")

        q = card["quote"]
        if q:
            if q in text:
                pass
            elif _norm_ws(q) in _norm_ws(text):
                flags.append(f"new{i}: quote matches only after whitespace normalisation")
            else:
                flags.append(f"new{i}: QUOTE NOT VERBATIM in input")
                card["_quote_not_verbatim"] = True
        src = card["source"]
        if not (isinstance(src, str) and SOURCE_RE.match(src)):
            flags.append(f"new{i}: bad source {src!r}")
        if card["tone"] not in ALLOWED_TONES:
            flags.append(f"new{i}: bad tone {card['tone']!r}")
        what = card["what"] or ""
        if not what.startswith("我"):
            flags.append(f"new{i}: what does not start with 我")
        elif not what.startswith(WHAT_VERBS):
            flags.append(f"new{i}: what lacks a listed source verb (soft)")
        if card.get("realm") == "dream" and "梦" not in what:
            flags.append(f"new{i}: realm=dream but what doesn't say it's a dream")
        clean_cards.append(card)

    new_ids = {f"new{i}" for i in range(len(clean_cards))}
    valid_ids = new_ids | set(existing_ids)
    clean_threads = []
    for j, th in enumerate(result.get("threads") or []):
        if not isinstance(th, dict):
            flags.append(f"thread{j}: not an object, dropped")
            continue
        th.pop("feeling", None)
        if th.get("kind") not in ALLOWED_KINDS:
            flags.append(f"thread{j}: bad kind {th.get('kind')!r}, dropped")
            continue
        if not str(th.get("because") or "").strip():
            flags.append(f"thread{j}: empty because, dropped")
            continue
        a, b = str(th.get("a")), str(th.get("b"))
        if a not in valid_ids or b not in valid_ids:
            flags.append(f"thread{j}: unknown endpoint {a!r}->{b!r}, dropped")
            continue
        if a == b:
            flags.append(f"thread{j}: self-loop, dropped")
            continue
        clean_threads.append({"a": a, "b": b, "kind": th["kind"],
                              "because": th["because"].strip()})

    return {"cards": clean_cards, "threads": clean_threads, "_flags": flags}


# ---------------------------------------------------------------- public API

def tag(text: str, existing_cards: list[dict] | None = None,
        backend: str = "api", return_raw: bool = False) -> dict:
    existing_cards = existing_cards or []
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    raw = BACKENDS[backend](text, existing_cards)
    try:
        parsed = _extract_json(raw)
    except json.JSONDecodeError as e:
        log.error("tagger output is not JSON: %s", e)
        return {"cards": [], "threads": [], "_flags": [f"unparseable output: {e}"],
                **({"_raw": raw} if return_raw else {})}
    ids = {str(c["id"]) for c in existing_cards if isinstance(c, dict) and "id" in c}
    out = validate(parsed, text, ids)
    if return_raw:
        out["_raw"] = raw
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="py -m wandering.tagger")
    ap.add_argument("file")
    ap.add_argument("--backend", default="api", choices=sorted(BACKENDS))
    ap.add_argument("--existing", help="JSON file with a list of existing cards (need 'id')")
    ap.add_argument("--raw", action="store_true", help="include raw model output as _raw")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    text = Path(args.file).read_text(encoding="utf-8")
    existing = json.loads(Path(args.existing).read_text(encoding="utf-8")) if args.existing else []
    result = tag(text, existing, backend=args.backend, return_raw=args.raw)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
