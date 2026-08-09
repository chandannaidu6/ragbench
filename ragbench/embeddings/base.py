"""
embeddings/base.py

The contract every embedder must satisfy, regardless of provider (OpenAI,
HuggingFace, Cohere, ...). Retrievers and the TrackedEmbedder wrapper depend
on THIS interface, never on a concrete provider.

Two promises:
  1. embed_text / embed_texts / embed_sentences  - the actual embedding calls
  2. count_tokens(texts) -> int                   - each embedder knows how ITS
     provider tokenizes, so token counting is correct per provider instead of
     the wrapper guessing with an OpenAI-only tokenizer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    provider: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        ...

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.empty((0, 0), dtype=np.float32)
        return self.embed_texts(sentences)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    @abstractmethod
    def count_tokens(self, texts: List[str]) -> int:
        """Return the total input-token count for these texts, using THIS
        provider's tokenizer. Exact per provider - the wrapper reads this
        instead of assuming OpenAI's tokenizer for everyone."""
        ...
