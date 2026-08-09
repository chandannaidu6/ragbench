"""
embeddings/ollama_embedder.py

Ollama implementation of BaseEmbedder. Talks to a local (or remote) Ollama
server over its native HTTP API - no API key, no cloud calls, genuinely
free. Pull a free embedding model first, e.g. `ollama pull nomic-embed-text`
or `ollama pull mxbai-embed-large`.

Uses only the standard library (urllib) so there is no extra dependency to
install beyond running Ollama itself.

count_tokens is a length-based approximation (~4 chars/token): Ollama does
not expose a per-model tokenizer over its HTTP API. Since local models are
never present in metrics/cost.PRICING, the count only affects the "tokens
used" report - billed cost is always $0.0 for this provider either way.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

import numpy as np

from ragbench.embeddings.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    provider = "ollama"

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: Optional[str] = None, batch_size: int = 64):
        super().__init__(model)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL")
                         or "http://localhost:11434").rstrip("/")
        self.batch_size = batch_size

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            # covers urllib.error.URLError (connection refused, DNS, ...) AND
            # http.client.RemoteDisconnected (server accepted the connection
            # but closed it before sending a response - e.g. Ollama crashed
            # or ran out of memory mid-request) - both are OSError subclasses,
            # but RemoteDisconnected specifically bypasses urlopen's usual
            # URLError wrapping, so narrowing to just URLError missed it
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url} - is `ollama serve` "
                f"running and is model '{self.model}' pulled? Also check it "
                f"didn't crash/OOM mid-request if this happened partway "
                f"through a large batch. ({exc})"
            ) from exc

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        clean_texts = [str(t).strip() or " " for t in texts]
        all_embeddings = []
        for i in range(0, len(clean_texts), self.batch_size):
            batch = clean_texts[i:i + self.batch_size]
            result = self._post("/api/embed", {"model": self.model, "input": batch})
            all_embeddings.extend(result["embeddings"])
        return np.array(all_embeddings, dtype=np.float32)

    def count_tokens(self, texts: List[str]) -> int:
        return sum(max(1, len(str(t)) // 4) for t in texts)
