import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List,Dict,Any,Tuple
from ragbench.chunkers.base import BaseChunker

class RecursiveChunker(BaseChunker):
    def __init__(self,pdf_path:str|Path,max_size:int = 120):
        self.pdf_path = pdf_path
        self.max_size = max_size

    @staticmethod
    def clean_text(text:str) -> str:
        text = text.replace("\r\n","\n").replace("\r","\n")
        text = re.sub(r"[ \t]+"," ",text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}","\n\n",text)
        return text.strip()


    def split_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r'(?:\r?\n){2,}', text) if p.strip()]

    def split_lines(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r'(?:\r?\n)', text) if p.strip()]

    def split_sentences(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r'(?<=[.?!])\s+', text) if p.strip()]

    def split_spaces(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r' +', text) if p.strip()] 

    

    def recursive_split(self,text:str,separators)->List[str]:
        if len(text.split())<=self.max_size:
            return [text]
        if not separators:
            words = text.split()
            return [
                " ".join(words[i:i+self.max_size]) for i in range(0,len(words),self.max_size)
            ]
        splitter = separators[0]
        rest = separators[1:]
        pieces = splitter(text)

        if len(pieces)<=1:
            return self.recursive_split(text,rest)

        results :List[str] = []
        buffer : List[str] = []

        for piece in pieces:
            candidate = buffer + [piece]
            if len(" ".join(candidate).split()) <= self.max_size:
                buffer = candidate

            else:
                if buffer:
                    results.append(" ".join(buffer))
                    buffer = []

                if len(piece.split()) > self.max_size:
                    results.extend(self.recursive_split(piece,rest))
                else:
                    buffer = [piece]

        if buffer:
            results.append(" ".join(buffer))


        return results

    def extract(self)->List[dict[str,Any]]:
        doc_name  = self.pdf_path.split("/")[-1]
        recur:List[Dict[str,Any]] = []
        global_idx = 0
        separators = [self.split_paragraphs,self.split_lines,self.split_sentences,self.split_spaces]
        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                raw_text = page.get_text()
                cleaned = self.clean_text(raw_text)
                if not cleaned:
                    continue
                units = self.recursive_split(cleaned,separators)
                for unit in units:
                    recur.append({
                        "text":unit,
                        "doc_name":doc_name,
                        "page_number":page.number+1,
                        "global_idx":global_idx+1
                    })
                    global_idx+=1

        return recur

    @staticmethod
    def _build_chunk(chunk_id:int,items:List[Dict[str,Any]],position:str="middle")->Dict[str,Any]:
        return {
            "chunk_id":chunk_id,
            "text":" ".join(item["text"] for item in items),
            "doc_name":items[0]["doc_name"],
            "pages":sorted({item["page_number"] for item in items}),
            "chunker_name":"recursive",
            "source_unit": "recursive_split",
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

        chunks:List[Dict[str,Any]] = [
            self._build_chunk(chunk_id=idx,items=[unit]) for idx,unit in enumerate(units)
        ]
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










        
