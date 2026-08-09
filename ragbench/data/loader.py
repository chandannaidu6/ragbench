"""
data/loader.py

Loads a corpus: either a pre-chunked JSON/JSONL file (parsed directly into
Chunk objects) or a raw PDF/directory of PDFs (returned as a source_path for
a chunker to process). Exposes LoadResult + a load() function, matching what
data/__init__.py and evaluation/runner.py expect.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragbench.schema import Chunk


class LoadResult:
    FIELD_ALIASES = {
        "content": "text", "body": "text", "passage": "text",
        "id": "chunk_id", "document": "doc_name", "doc_id": "doc_name",
    }
    DROP_FIELDS = {"score", "rerank_score"}

    def __init__(self,path:str|Path):
        self.path = Path(path)
        self.chunks:Optional[List[Chunk]] = None
        self.needs_chunking:bool = True
        self.source_path:Optional[Path] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"File does not exist: {self.path}")

        if self.path.is_dir() or self.path.suffix.lower() not in (".json",".jsonl"):
            # raw PDF (or a directory of PDFs) - a chunker will process this
            self.needs_chunking = True
            self.source_path = self.path
            return

        records = self._read_records()
        if not records:
            raise ValueError(f"File is empty or has no records: {self.path}")
        if self._looks_like_chunks(records[0]):
            self.chunks = self._normalize(records)
            self.needs_chunking = False
            self.source_path = self.path
        else:
            raise NotImplementedError(
                "Looks like raw documents, not chunks. Provide a PDF/txt corpus "
                "for chunking, or a pre-chunked JSON/JSONL file."
            )

    def _read_records(self)->List[Dict[str,Any]]:
        text = self.path.read_text(encoding="utf-8")
        if self.path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        return data if isinstance(data,list) else [data]

    def _looks_like_chunks(self,record:Dict[str,Any]) -> bool:
        canon = {self.FIELD_ALIASES.get(k,k) for k in record}
        return "text" in canon and ("chunk_id" in canon or "doc_name" in canon)

    def _normalize(self,records:List[Dict[str,Any]])->List[Chunk]:
        chunk_fields = set(Chunk.model_fields.keys())
        chunks:List[Chunk] = []

        for idx,raw in enumerate(records):
            mapped:Dict[str,Any] = {}
            metadata:Dict[str,Any] = {}
            for key,value in raw.items():
                if key in self.DROP_FIELDS:
                    continue

                canonical = self.FIELD_ALIASES.get(key,key)
                if canonical in  chunk_fields:
                    mapped[canonical] = value
                else:
                    metadata[canonical] = value

            text = mapped.get("text")
            if not text or not str(text).strip():
                continue
            if mapped.get("chunk_id") is None:
                mapped["chunk_id"] = idx
            if not mapped.get("doc_name"):
                mapped["doc_name"] = str(mapped["chunk_id"])
            mapped["metadata"] = {**(metadata.get("metadata") or {}), **metadata}
            try:
                chunks.append(Chunk(**mapped))
            except Exception:
                continue
        return chunks


def load(path: str | Path) -> LoadResult:
    return LoadResult(path)
