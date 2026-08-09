"""
llm/ollama_client.py

Ollama implementation of BaseLLMClient. Talks to a local (or remote) Ollama
server over its native HTTP API - no API key required, genuinely free. Pull
a model first, e.g. `ollama pull llama3.2`.

Uses only the standard library (urllib) so there is no extra dependency to
install beyond running Ollama itself.

Ollama's /api/chat response reports exact prompt/completion token counts
(`prompt_eval_count` / `eval_count`), so usage tracking here is as precise
as any metered cloud provider's - it's just that metrics/cost.py has no
price entry for local model names, so the billed cost comes out to $0.0.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Dict, Any

from ragbench.llm.base import BaseLLMClient


try:
    import mlflow
    _trace = mlflow.trace
except Exception:
    def _trace(fn):
        return fn


PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    """Load a prompt template lazily and cache it. Relative to the package,
    so it works no matter where the user runs from."""
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


class OllamaClient(BaseLLMClient):
    provider = "ollama"

    def __init__(self, model: str = "llama3.2", base_url: Optional[str] = None):
        super().__init__(model)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL")
                         or "http://localhost:11434").rstrip("/")

    def build_prompt(self, query: Optional[str] = None, context: Optional[str] = None,
                     chunk: Optional[Dict[str, Any]] = None,
                     mode: Literal["plain", "cot", "short", "long", "question"] = "plain") -> str:
        if mode == "plain":
            return _load_template("v1_plain.txt").format(context=context, query=query)
        elif mode == "cot":
            return _load_template("v2_cot.txt").format(context=context, query=query)
        elif mode == "short":
            return _load_template("hyde_short.txt").format(query=query)
        elif mode == "long":
            return _load_template("hyde_long.txt").format(query=query)
        else:  # question
            return _load_template("benchmark.txt").format(
                chunk_id=chunk["chunk_id"], chunk_text=chunk["text"]
            )

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
            # covers urllib.error.URLError AND http.client.RemoteDisconnected
            # (server closed the connection mid-response, e.g. crash/OOM) -
            # see ollama_embedder.py's _post() for why URLError alone misses it
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url} - is `ollama serve` "
                f"running and is model '{self.model}' pulled? Also check it "
                f"didn't crash/OOM mid-request. ({exc})"
            ) from exc

    def _call(self, prompt: str) -> str:
        response = self._post("/api/chat", {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        })
        # Ollama reports exact token counts on every response -> exact usage
        self._set_usage(
            response.get("prompt_eval_count", 0),
            response.get("eval_count", 0),
        )
        return response["message"]["content"]

    @_trace
    def generate_context(self, query: str, mode: Literal["short", "long"] = "short") -> str:
        return self._call(self.build_prompt(query=query, mode=mode))

    @_trace
    def chat(self, query: str, context: str, mode: Literal["plain", "cot"] = "plain") -> str:
        return self._call(self.build_prompt(query=query, context=context, mode=mode))

    @_trace
    def generate_prompt(self, chunk: Dict[str, Any], mode: str = "question") -> str:
        return self._call(self.build_prompt(chunk=chunk, mode=mode))
