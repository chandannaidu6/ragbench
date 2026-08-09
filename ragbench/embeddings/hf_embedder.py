"""
embeddings/hf_embedder.py

HuggingFace / sentence-transformers implementation of BaseEmbedder. Runs
fully locally - no API key, no network calls after the model weights are
first downloaded and cached - so it is genuinely free. This is the default
choice for a no-cost embedding provider; `sentence-transformers/all-MiniLM-L6-v2`
is small, fast on CPU, and the most widely used free embedding model for RAG.

count_tokens uses the model's own AutoTokenizer, so usage numbers are exact
per model rather than assuming an OpenAI-shaped tokenizer.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ragbench.embeddings.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    provider = "huggingface"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None, batch_size: int = 32):
        super().__init__(model)
        from sentence_transformers import SentenceTransformer   # lazy import
        self._model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        self._tokenizer = None                       # AutoTokenizer, built on demand

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        clean_texts = [str(t).strip() or " " for t in texts]
        embeddings = self._model.encode(
            clean_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def count_tokens(self, texts: List[str]) -> int:
        if self._tokenizer is None:
            from transformers import AutoTokenizer    # lazy import
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        return sum(len(self._tokenizer.encode(t)) for t in texts)
