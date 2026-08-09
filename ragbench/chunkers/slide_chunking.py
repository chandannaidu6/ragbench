import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ragbench.chunkers.base import BaseChunker


class SlideChunker(BaseChunker):
    def __init__(self, pdf_path: str | Path, chunk_size: int = 5, overlap: int = 2):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size (else the window never advances)")
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

    @staticmethod
    def split_sentences(text: str, min_length: int = 20) -> List[str]:
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= min_length]

    def extract_text(self) -> List[Dict[str, Any]]:
        doc_name = self.pdf_path.split('/')[-1]
        pages_data: List[Dict[str, Any]] = []
        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                cleaned = self.clean_text(page.get_text())
                if not cleaned:
                    continue
                pages_data.append({
                    "text": cleaned,
                    "number": page.number,
                    "title": doc_name,
                })
        return pages_data

    def metadata_sentences(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sentences: List[Dict[str, Any]] = []
        global_id = 0
        for page in data:
            for sen_text in self.split_sentences(page['text']):
                sentences.append({
                    'text': sen_text,
                    'doc_name': page['title'],
                    'page_number': page['number'] + 1,   # 1-indexed, like the others
                    'global_idx': global_id,
                })
                global_id += 1
        return sentences

    @staticmethod
    def _build_chunk(chunk_id: int, items: List[Dict[str, Any]], position: str = "middle") -> Dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "text": " ".join(item["text"] for item in items),
            "doc_name": items[0]["doc_name"],
            "pages": sorted({item["page_number"] for item in items}),
            "chunker_name": "slide",
            "source_unit": "sentences",
            "start_idx": items[0]["global_idx"],
            "end_idx": items[-1]["global_idx"],
            "sentence_count": len(items),
            "word_count": sum(len(item["text"].split()) for item in items),
            "position": position,
            "source_sentences": items,
        }

    def build_chunks(self, min_sentences: int = 2) -> List[Dict[str, Any]]:
        sentences = self.metadata_sentences(self.extract_text())
        if not sentences:
            return []

        groups: List[List[Dict[str, Any]]] = []
        step = self.chunk_size - self.overlap
        start = 0
        total = len(sentences)
        while start < total:
            end = min(start + self.chunk_size, total)
            groups.append(sentences[start:end])
            if end == total:
                break
            start += step

        if len(groups) >= 2 and len(groups[-1]) < min_sentences:
            tail = groups.pop()
            seen = {s["global_idx"] for s in groups[-1]}
            groups[-1].extend(s for s in tail if s["global_idx"] not in seen)

        chunks = [self._build_chunk(chunk_id=i, items=g) for i, g in enumerate(groups)]

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