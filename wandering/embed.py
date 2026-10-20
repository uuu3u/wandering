"""Embeddings via the local HTTP bridge only. Never loads a model in-process."""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

import numpy as np

from . import config


_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def url_ok(url):
    """Card text goes out in the request body: plain http only to this machine."""
    u = urllib.parse.urlsplit(url)
    return u.scheme == "https" or (u.scheme == "http" and u.hostname in _LOCAL_HOSTS)


def embed(texts, url=None, timeout=None):
    """Return list[np.float32 array] (one per text) or None on any failure."""
    texts = list(texts)
    if not texts:
        return []
    url = url or config.EMBED_URL
    if not url_ok(url):
        print(f"wandering: refusing to send card text over plain http to {url}; use https",
              file=sys.stderr)
        return None
    try:
        body = json.dumps({"input": texts}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout or config.EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data["data"]
        if len(items) != len(texts):
            return None
        # honour "index" if present, otherwise trust order
        if all("index" in it for it in items):
            items = sorted(items, key=lambda it: it["index"])
        return [np.asarray(it["embedding"], dtype=np.float32) for it in items]
    except Exception:
        return None


def embed_batched(texts, embedder=embed, batch=None):
    """Embed in batches; returns None if any batch fails."""
    batch = batch or config.EMBED_BATCH
    texts = list(texts)
    out = []
    for i in range(0, len(texts), batch):
        part = embedder(texts[i:i + batch])
        if part is None:
            return None
        out.extend(part)
    return out
