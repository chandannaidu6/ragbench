"""
embeddings/openai_embedder.py

OpenAI implementation of BaseEmbedder. count_tokens uses tiktoken, which is
correct for OpenAI embedding models.
"""
from __future__ import annotations

import os
import time
from typing import List

import numpy as np

from ragbench.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    provider = "openai"

    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: str | None = None, base_url: str | None = None,
                 batch_size: int = 100):
        super().__init__(model)
        from openai import OpenAI                    # lazy import
        # base_url lets this class target any OpenAI-compatible free/local
        # embedding endpoint instead of api.openai.com.
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if resolved_api_key is None and resolved_base_url is not None:
            resolved_api_key = "not-needed"
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self.batch_size = batch_size
        self._enc = None                             # tiktoken encoder, built on demand

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        from openai import RateLimitError            # lazy import

        clean_texts = []
        for t in texts:
            if t is None:
                text = " "
            elif isinstance(t, list):
                text = " ".join(str(x) for x in t).strip()
            else:
                text = str(t).strip()
            if not text:
                text = " "
            clean_texts.append(text)

        all_embeddings = []
        for i in range(0, len(clean_texts), self.batch_size):
            batch = clean_texts[i:i + self.batch_size]
            while True:
                try:
                    resp = self.client.embeddings.create(model=self.model, input=batch)
                    all_embeddings.extend([d.embedding for d in resp.data])
                    break
                except RateLimitError:
                    time.sleep(1.0)
                    continue
        return np.array(all_embeddings, dtype=np.float32)

    def count_tokens(self, texts: List[str]) -> int:
        import tiktoken                              # lazy import
        if self._enc is None:
            try:
                self._enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(self._enc.encode(t)) for t in texts)
