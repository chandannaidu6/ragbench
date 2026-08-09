"""
llm/base.py

The contract every LLM client must satisfy, regardless of provider (OpenAI,
Anthropic, Google, ...). The wrapper (TrackedLLMClient) and the retrievers
depend on THIS interface, never on a concrete provider - that's what makes
the package provider-agnostic.

Two things every client promises:
  1. generate_context / chat          - the actual generation methods
  2. self.last_usage after each call  - {"prompt": int, "completion": int}
     read from that provider's OWN response, so token counting is exact for
     any provider (only the client knows how its provider reports usage).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLMClient(ABC):
    provider: str = "base"

    def __init__(self, model: str):
        self.model = model
        # every subclass sets this from its provider response after each call
        self.last_usage: Optional[Dict[str, int]] = None

    @abstractmethod
    def generate_context(self, query: str, mode: str = "short") -> str:
        """Generate a hypothetical document (used by HyDE)."""
        ...

    @abstractmethod
    def chat(self, query: str, context: str, mode: str = "plain") -> str:
        """Answer a query given retrieved context (used by RAG)."""
        ...

    def complete(self, prompt: str) -> str:
        """Send a fully-formed prompt as-is, no template wrapping. Used by
        callers that build their own prompt text and just need raw text
        back. Every subclass already implements `_call(prompt)` for this
        exact purpose internally - this just exposes it."""
        return self._call(prompt)

    def generate_prompt(self, chunk: Dict[str, Any], mode: str = "question") -> str:
        """Generate a benchmark question from a chunk (used by synthetic
        benchmark). Optional: only clients that support question generation
        need override this."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_prompt."
        )

    def _set_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Subclasses call this after each API response to expose usage."""
        self.last_usage = {"prompt": int(prompt_tokens),
                           "completion": int(completion_tokens)}
