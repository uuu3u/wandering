"""什么时候走：把用户这句话分成 skip / intimate / affection / normal。

- skip: 系统文本（交给可选的外部 intent_gate）或"嗯""好"这种短句
- intimate: 露骨语域（外部 intent_gate 判的），可翻 intimate + explicit 卡
- affection: 亲昵语域（晚安、小名、抱抱），可翻 intimate 卡
- normal: 都不翻
"""
import re
import sys

from . import config

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _intent_gate(query):
    if not config.INTENT_GATE_SRC:
        return "normal"
    try:
        src = str(config.INTENT_GATE_SRC)
        if src not in sys.path:
            sys.path.insert(0, src)
        import intent_gate  # type: ignore
        g = intent_gate.classify(query)
        return g if g in ("skip", "intimate", "normal") else "normal"
    except Exception:
        return "normal"


def classify(query, affection_words=None):
    q = (query or "").strip()
    core = _PUNCT_RE.sub("", q)
    if (len(core) <= 1 or core.lower() in config.TRIVIAL) and "?" not in q and "？" not in q:
        return "skip"
    g = _intent_gate(q)
    if g in ("skip", "intimate"):
        return g
    words = config.AFFECTION_WORDS if affection_words is None else affection_words
    if any(_affectionate(q, w) for w in words):
        return "affection"
    return "normal"


# "我不想你了" / "别叫我宝宝": a negation just before the word cancels it
_NEGATIONS = ("不", "别", "没", "甭", "勿")
_NEG_WINDOW = 3


def _affectionate(q, word):
    start = q.find(word)
    while start != -1:
        before = q[max(0, start - _NEG_WINDOW):start]
        if not any(n in before for n in _NEGATIONS):
            return True
        start = q.find(word, start + 1)
    return False
