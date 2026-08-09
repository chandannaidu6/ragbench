import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List,Dict,Any,Tuple
from ragbench.schema import Chunk
from abc import ABC,abstractmethod

class BaseChunker(ABC):
    chunker_name:str = "base"
    source_unit:str = "units"

    def __init__(self,pdf_path):
        self.pdf_path = pdf_path

    @staticmethod
    def clean_text(text:str) -> str:
        text = text.replace("\r\n","\n").replace("\r","\n")
        text = re.sub(r"[ \t]+"," ",text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}","\n\n",text)
        return text.strip()

    def _build_chunk(self,chunk_id: int, items: List[Dict[str, Any]], position: str = "middle") -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            text=" ".join(item["text"] for item in items),
            doc_name=items[0]["doc_name"],
            pages=sorted({item["page_number"] for item in items}),
            chunker_name=self.chunker_name,
            source_unit=self.source_unit,
            start_idx=items[0]["global_idx"],
            end_idx=items[-1]["global_idx"],
            word_count=sum(len(item["text"].split()) for item in items),
            position=position,
            metadata={
                "unit_count": len(items),
                "source_units": items,
            }
        )

    def _finalize(self,chunks:List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        

        for idx,chunk in enumerate(chunks):
            chunk.chunk_id = idx
            if n==1:
                chunk.position = "full"
            elif idx == 0:
                chunk.position = "start"
            elif idx == n-1:
                chunk.position = "end"
            else:
                chunk.position = "middle"
        return chunks

    @abstractmethod
    def build_chunks(self)->List[Chunk]:
        ...



        

        

        

    
    