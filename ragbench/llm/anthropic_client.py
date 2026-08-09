"""
llm/anthropic_client.py

Anthropic implementation of BaseLLMClient.

Packaging-safe:
  - prompts load lazily via the same templates OpenAIClient uses, relative to
    THIS file (not the CWD) - so an installed user isn't required to run from
    a specific directory.
  - reads the standard ANTHROPIC_API_KEY env var.
  - captures response.usage into self.last_usage after each call, so token
    counting is exact.
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


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5", api_key: Optional[str] = None,
                 max_tokens: int = 4096):
        super().__init__(model)
        from anthropic import Anthropic               # lazy import
        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._max_tokens = max_tokens

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
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # capture usage from the provider's own response -> exact counts
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._set_usage(
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )
        return next(block.text for block in response.content if block.type == "text")

    @_trace
    def generate_context(self, query: str, mode: Literal["short", "long"] = "short") -> str:
        return self._call(self.build_prompt(query=query, mode=mode))

    @_trace
    def chat(self, query: str, context: str, mode: Literal["plain", "cot"] = "plain") -> str:
        return self._call(self.build_prompt(query=query, context=context, mode=mode))

    @_trace
    def generate_prompt(self, chunk: Dict[str, Any], mode: str = "question") -> str:
        return self._call(self.build_prompt(chunk=chunk, mode=mode))
