"""
rerank/reranker.py

Cross-encoder reranker. sentence-transformers (which pulls in PyTorch, a very
large dependency) is imported LAZILY and treated as an optional extra, so a
user who never reranks doesn't download hundreds of MB of torch, and
`import ragbench` never fails on a missing reranker dependency.

    pip install ragbench[rerank]
"""
from __future__ import annotations

from typing import List, Dict, Any


class Reranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        try:
            from sentence_transformers import CrossEncoder      # lazy import
        except ImportError as e:
            raise ImportError(
                "Reranking requires sentence-transformers (and PyTorch). "
                "Install it with:  pip install ragbench[rerank]"
            ) from e
        self.model = CrossEncoder(model)

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = []
        normalized_candidates = []
        for c in candidates:
            text = c.get("text", "")
            if isinstance(text, list):
                text = " ".join(str(x) for x in text)
            elif text is None:
                text = ""
            else:
                text = str(text)
            item = dict(c)
            item["text"] = text
            normalized_candidates.append(item)
            pairs.append((str(query), text))

        scores = self.model.predict(pairs)

        reranked = []
        for candidate, score in zip(normalized_candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
