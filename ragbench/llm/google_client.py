"""
llm/google_client.py

Google Gemini implementation of BaseLLMClient, via the unified `google-genai`
SDK (`pip install google-genai`; `from google import genai`).

Packaging-safe:
  - prompts load lazily via the same templates OpenAIClient/AnthropicClient
    use, relative to THIS file (not the CWD) - so an installed user isn't
    required to run from a specific directory.
  - reads the standard GOOGLE_API_KEY / GEMINI_API_KEY env vars.
  - captures response.usage_metadata into self.last_usage after each call,
    so token counting is exact.
  - mlflow tracing is optional: if mlflow isn't installed, tracing is a no-op
    rather than an import-time crash.
"""
from __future__ import annotations

import os
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


class GoogleClient(BaseLLMClient):
    provider = "google"

    def __init__(self, model: str = "gemini-3.1-flash-lite", api_key: Optional[str] = None):
        super().__init__(model)
        from google import genai                      # lazy import
        self._client = genai.Client(
            api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )

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

    def _call(self, prompt: str) -> str:
        response = self._client.models.generate_content(model=self.model, contents=prompt)
        # capture usage from the provider's own response -> exact counts
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self._set_usage(
                getattr(usage, "prompt_token_count", 0) or 0,
                getattr(usage, "candidates_token_count", 0) or 0,
            )
        return response.text

    @_trace
    def generate_context(self, query: str, mode: Literal["short", "long"] = "short") -> str:
        return self._call(self.build_prompt(query=query, mode=mode))

    @_trace
    def chat(self, query: str, context: str, mode: Literal["plain", "cot"] = "plain") -> str:
        return self._call(self.build_prompt(query=query, context=context, mode=mode))

    @_trace
    def generate_prompt(self, chunk: Dict[str, Any], mode: str = "question") -> str:
        return self._call(self.build_prompt(chunk=chunk, mode=mode))
