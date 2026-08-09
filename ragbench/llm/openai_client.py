"""
llm/openai_client.py

OpenAI implementation of BaseLLMClient.

Packaging-safe:
  - prompts load lazily, relative to THIS file (not the CWD), and ship as
    package data - so an installed user isn't required to run from a specific
    directory.
  - reads the standard OPENAI_API_KEY env var.
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


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        super().__init__(model)
        from openai import OpenAI                    # lazy import
        # base_url lets this class target any OpenAI-compatible free/local
        # endpoint (Ollama's OpenAI-compat API, LM Studio, vLLM, OpenRouter,
        # Groq, ...) instead of api.openai.com. Most self-hosted servers
        # don't check the key at all, but the SDK still requires a string.
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if resolved_api_key is None and resolved_base_url is not None:
            resolved_api_key = "not-needed"
        self._client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

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
        response = self._client.responses.create(model=self.model, input=prompt)
        # capture usage from the provider's own response -> exact counts
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._set_usage(
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )
        return response.output[0].content[0].text

    @_trace
    def generate_context(self, query: str, mode: Literal["short", "long"] = "short") -> str:
        return self._call(self.build_prompt(query=query, mode=mode))

    @_trace
    def chat(self, query: str, context: str, mode: Literal["plain", "cot"] = "plain") -> str:
        return self._call(self.build_prompt(query=query, context=context, mode=mode))

    @_trace
    def generate_prompt(self, chunk: Dict[str, Any], mode: str = "question") -> str:
        return self._call(self.build_prompt(chunk=chunk, mode=mode))
