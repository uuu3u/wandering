"""All wandering tunables live here."""
import os
from pathlib import Path

# walk start / stop thresholds
START_MIN = 0.45          # best query-card cosine must reach this to start a walk
TENDER_START_MIN = 0.6    # tender cards may start a walk only at this similarity
OFFTOPIC_STOP = 0.25      # new card below this vs query -> stop before adding
OFFTOPIC_DRIFT = 0.35     # below this -> add as drift ("走神想到的"), then stop

# walk dynamics
DECAY = 0.75              # light *= DECAY per step
LIGHT_MIN = 0.3           # stop once light falls below this
MAX_STEPS = 4
RETURN_P = 0.15           # once per walk: chance to jump back to start
TEMP = 0.5                # softmax temperature over step scores
RECENCY_DAYS = 60         # recency = 0.5 + 0.5*exp(-days/RECENCY_DAYS)

# embedding bridge (never load a model in-process)
# any OpenAI-compatible /v1/embeddings endpoint; tools/embed_bridge.py serves bge-small-zh here
EMBED_URL = os.environ.get("WANDERING_EMBED_URL", "http://127.0.0.1:8749/v1/embeddings")
EMBED_TIMEOUT = float(os.environ.get("WANDERING_EMBED_TIMEOUT", "2"))
EMBED_BATCH = 32

WANDERING_HOME = Path(os.environ.get("WANDERING_HOME", Path.home() / ".wandering"))
DEFAULT_DB = WANDERING_HOME / "wandering.sqlite"

# optional external intent gate: a directory holding intent_gate.py with classify(q) -> skip|intimate|normal
INTENT_GATE_SRC = os.environ.get("WANDERING_INTENT_GATE")
# 亲昵语域：命中任一词 → affection（可翻 intimate 卡，不翻 explicit）。按数据集补小名
AFFECTION_WORDS = ["晚安", "早安", "抱抱", "抱一下", "抱紧", "亲亲", "贴贴", "想你", "宝宝", "老公", "老婆", "摸摸头", "蹭蹭", "爱你"]
# 去掉标点表情后落在这张表里（或只剩一个字）→ skip。"想你"两个字也有分量，所以不按长度一刀切
TRIVIAL = {"嗯", "嗯嗯", "好", "好的", "好滴", "哦", "噢", "喔", "行", "收到", "ok", "okk", "哈哈", "哈哈哈", "嘿嘿", "对", "是", "嗯呢"}
