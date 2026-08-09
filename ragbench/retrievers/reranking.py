"""
retrievers/reranking.py

Wraps any BaseRetriever with a cross-encoder reranking pass, so reranking is
a config flag (RunConfig.use_reranker) rather than a property baked into a
specific retriever. Over-fetches `candidate_k` results from the wrapped
retriever, then reranks down to `top_k` with rerank.reranker.Reranker - this
composes with bm25/dense/hybrid/hyde without any of them knowing
reranking exists, the same way TrackedEmbedder/TrackedLLMClient wrap a raw
embedder/LLM client without the retrievers knowing usage is being tracked.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ragbench.retrievers.base import BaseRetriever
from ragbench.rerank.reranker import Reranker


class RerankingRetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, reranker: Optional[Reranker] = None,
                 model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
                 candidate_k: Optional[int] = None, candidate_multiplier: int = 4):
        self.base_retriever = base_retriever
        self.reranker = reranker or Reranker(model=model)
        self.candidate_k = candidate_k
        self.candidate_multiplier = candidate_multiplier
        base_name = getattr(base_retriever, "retriever_name", type(base_retriever).__name__)
        self.retriever_name = f"{base_name}+rerank"

    def build_index(self) -> None:
        self.base_retriever.build_index()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # over-fetch so the cross-encoder has real candidates to re-rank,
        # then cut back down to what the caller actually asked for
        fetch_k = self.candidate_k or max(top_k * self.candidate_multiplier, top_k)
        candidates = self.base_retriever.retrieve(query=query, top_k=fetch_k)
        return self.reranker.rerank(query, candidates, top_k=top_k)
