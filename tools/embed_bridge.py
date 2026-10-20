"""Minimal embedding HTTP bridge for BAAI/bge-small-zh-v1.5.
Keeps model resident in memory. OpenAI-compatible /v1/embeddings endpoint.
Start: python fastembed_bridge.py  (binds 127.0.0.1:8749). pip install fastembed
"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
PORT = 8749
_model = None


def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL_NAME)
    return _model


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        texts = body.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        model = get_model()
        vecs = list(model.embed(texts))
        data = [{"object": "embedding", "index": i, "embedding": v.tolist()}
                for i, v in enumerate(vecs)]
        resp = json.dumps({"data": data, "model": MODEL_NAME}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Loading {MODEL_NAME}...", flush=True)
    get_model()
    print(f"Bridge ready on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
