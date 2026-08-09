from typing import List

from ragbench.retrievers.bm25 import BM25
from ragbench.retrievers.dense import DenseRetriever
from ragbench.retrievers.hybrid import HybridRetriever
from ragbench.retrievers.hyde import HydeRetriever


class RegistryRetriever:
    """A factory, not itself a retriever - deliberately does NOT inherit
    from BaseRetriever (which would require implementing an abstract
    retrieve() this class has no business having)."""

    RETRIEVER_REGISTRY = {
        "bm25": BM25,
        "dense": DenseRetriever,
        "hybrid": HybridRetriever,
        "hyde": HydeRetriever,
    }

    def __init__(self, name: str):
        self.name = name.lower()

    def get_retriever(self, **kwargs):
        if self.name not in self.RETRIEVER_REGISTRY:
            raise ValueError(
                f"Unknown retriever '{self.name}'. "
                f"Available: {', '.join(self.available_retrievers())}"
            )
        retriever_cls = self.RETRIEVER_REGISTRY[self.name]
        return retriever_cls(**kwargs)

    @classmethod
    def available_retrievers(cls) -> List[str]:
        return list(cls.RETRIEVER_REGISTRY.keys())
