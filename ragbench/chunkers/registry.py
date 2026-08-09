from ragbench.chunkers.fixed_size import FixedSize
from ragbench.chunkers.recursive import RecursiveChunker
from ragbench.chunkers.semantic_chunking import SemanticChunker
from ragbench.chunkers.sentence_chunking import SentenceChunker
from ragbench.chunkers.slide_chunking import SlideChunker
from pathlib import Path
from typing import Optional,List

class Registry:
    CHUNKER_REGISTRY = {
        "fixed_size":FixedSize,
        "recursive":RecursiveChunker,
        "semantic":SemanticChunker,
        "sentence":SentenceChunker,
        "slide":SlideChunker
    }
    def __init__(self,pdf_path:str|Path,name:str,embedder:Optional[object] = None,**params):
        self.pdf_path = pdf_path
        self.name = name
        self.embedder = embedder
        self.params = params

    def get_chunker(self):
        if self.name not in self.CHUNKER_REGISTRY:
            raise ValueError(
                f"Unknown chunker '{self.name}'. "
                f"Available: {', '.join(self.available_chunkers())}"
            )
        chunker_cls = self.CHUNKER_REGISTRY[self.name]
        if self.name == "semantic":
            return chunker_cls(self.pdf_path,self.embedder)

        return chunker_cls(self.pdf_path,**self.params)

    @classmethod
    def available_chunkers(cls)->List[str]:
        return list(cls.CHUNKER_REGISTRY.keys())
