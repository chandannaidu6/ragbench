import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List,Dict,Any,Tuple
from ragbench.chunkers.base import BaseChunker

class SentenceChunker(BaseChunker):
    def __init__(self,pdf_path:str|Path,sentence_per_chunk:int= 20):
        self.pdf_path = pdf_path
        self.sentence_per_chunk = sentence_per_chunk

    @staticmethod
    def clean_text(text:str) -> str:
        text = text.replace("\r\n","\n").replace("\r","\n")
        text = re.sub(r"[ \t]+"," ",text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}","\n\n",text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r'(?<=[.?!])\s+', text) if p.strip()]

    def extract(self)->List[dict[str,Any]]:
        doc_name  = self.pdf_path.split("/")[-1]
        sentence_units:List[Dict[str,Any]] = []
        global_idx = 0
        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                raw_text = page.get_text()
                cleaned = self.clean_text(raw_text)
                if not cleaned:
                    continue
                units = self.split_sentences(cleaned)
                for unit in units:
                    sentence_units.append({
                        "text":unit,
                        "doc_name":doc_name,
                        "page_number":page.number+1,
                        "global_idx":global_idx+1
                    })
                    global_idx+=1

        return sentence_units


    @staticmethod
    def _build_chunk(chunk_id:int,items:List[Dict[str,Any]],position:str="middle")->Dict[str,Any]:
        return {
            "chunk_id":chunk_id,
            "text":" ".join(item["text"] for item in items),
            "doc_name":items[0]["doc_name"],
            "pages":sorted({item["page_number"] for item in items}),
            "chunker_name":"sentence_chunker",
            "source_unit": "sentence_split",
            "start_idx": items[0]["global_idx"],
            "end_idx": items[-1]["global_idx"],
            "unit_count": len(items),
            "word_count": sum(len(item["text"].split()) for item in items),
            "position": position,
            "source_units": items,

    }

    def build_chunks(self)->List[Dict[str,Any]]:
        units = self.extract()
        if not units:
            return []

        chunks:List[Dict[str,Any]] = []
        for start in range(0,len(units),self.sentence_per_chunk):
            group = units[start:start+self.sentence_per_chunk]
            chunks.append(self._build_chunk(chunk_id=len(chunks),items=group))
        for idx, chunk in enumerate(chunks):
            if len(chunks) == 1:
                chunk["position"] = "full"
            elif idx == 0:
                chunk["position"] = "start"
            elif idx == len(chunks) - 1:
                chunk["position"] = "end"
            else:
                chunk["position"] = "middle"
            chunk["chunk_id"] = idx

        return chunks



    