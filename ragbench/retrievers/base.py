from abc import ABC, abstractmethod
from typing import List
from ragbench.schema import RetrievedChunk


class BaseRetriever(ABC):
    retriever_name: str = "base"

    def build_index(self) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        ...