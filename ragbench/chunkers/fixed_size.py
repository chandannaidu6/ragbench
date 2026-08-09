import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ragbench.chunkers.base import BaseChunker

class FixedSize(BaseChunker):
    def __init__(self, pdf_path: str, chunk_size: int = 200, overlap: int = 30):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_pages(self) -> List[Dict[str, Any]]:
        doc_name = self.pdf_path.split('/')[-1]
        pages_data = []
        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                cleaned_text = self.clean_text(page.get_text())
                if not cleaned_text:
                    continue
                pages_data.append({
                    "doc_name": doc_name,
                    "page_number": page.number + 1,
                    "text": cleaned_text,
                })
        return pages_data

    @staticmethod
    def words_with_metadata(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        words = []
        global_idx = 0
        for page in pages_data:
            for match in re.finditer(r"\S+", page["text"]):
                words.append({
                    "text": match.group(),          
                    "page_number": page["page_number"],
                    "doc_name": page["doc_name"],
                    "global_idx": global_idx,
                })
                global_idx += 1
        return words

    @staticmethod
    def _build_chunk(chunk_id: int, items: List[Dict[str, Any]], position: str = "middle") -> Dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "text": " ".join(item["text"] for item in items),
            "doc_name": items[0]["doc_name"],
            "pages": sorted({item["page_number"] for item in items}),
            "chunker_name": "fixed_size",
            "source_unit": "words",
            "start_idx": items[0]["global_idx"],
            "end_idx": items[-1]["global_idx"],
            "word_count": len(items),
            "position": position,
            "source_words": items,
        }

    @staticmethod
    def chunk_words(word_units: List[Dict[str, Any]], chunk_size: int = 200, overlap: int = 30) -> List[Dict[str, Any]]:
        if not word_units:
            return []
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero")
        if overlap < 0:
            raise ValueError("Overlap size must be >= zero")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk size")

        groups: List[List[Dict[str, Any]]] = []
        total = len(word_units)
        stride = chunk_size - overlap
        start_idx = 0
        while start_idx < total:
            end_idx = min(start_idx + chunk_size, total)
            groups.append(word_units[start_idx:end_idx])
            if end_idx == total:
                break
            start_idx += stride

        chunks = [FixedSize._build_chunk(chunk_id=i, items=g) for i, g in enumerate(groups)]

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

    def build_chunks(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[Dict[str, Any]]:
        pages_data = self.extract_pages()
        word_units = self.words_with_metadata(pages_data)
        return self.chunk_words(
            word_units,
            chunk_size=chunk_size if chunk_size is not None else self.chunk_size,
            overlap=overlap if overlap is not None else self.overlap,
        )