import json
import re
from typing import List, Dict, Any, Optional

from ragbench.schema import Query
from ragbench.llm.base import BaseLLMClient

class SyntheticGenerator:
    def __init__(self,llm_client: BaseLLMClient | None,chunks:List[Dict[str,Any]],min_chunk_chars:int=50,sample_size:int = 40):
        self.chunks = chunks
        self.min_chunk_chars = min_chunk_chars
        self.sample_size = sample_size
        self.llm_client = llm_client

    def is_big_chunk(self,chunk:Dict[str,Any]) -> bool:
        return len(chunk.get("text","")) >= self.min_chunk_chars
    

    def is_small_corpus(self) -> None:
        total_chars = sum(len(c.get("text","")) for c in self.chunks)
        if total_chars < 500:
            raise ValueError("Corpus is too short to generate a synthetic benchmark.")


    def pick_chunks(self) -> List[Dict[str,Any]]:
        eligible = [c for c in self.chunks if self.is_big_chunk(c)]
        if not eligible:
            raise ValueError("No chunk is long enough to generate a question from.")

        n = len(eligible)
        k = n//2
        take = min(20,n)
        picked: List[Dict[str, Any]] = []
        seen_ids = set()
        def add(candidates):
            for c in candidates:
                key = (c.get("doc_name"),c.get("chunk_id"))
                if key not in seen_ids:
                    seen_ids.add(key)
                    picked.append(c)

        add(eligible[:take])
        add(eligible[k:k+take])
        add(eligible[-take:])
        return picked[:self.sample_size]


    
    def generate_benchmark_file(self)->List[Query]:
        self.is_small_corpus()
        chunks = self.pick_chunks()
        queries: List[Query] = []
        for i,chunk in enumerate(chunks):
            raw = self.llm_client.generate_prompt(chunk = chunk)
            parsed = self._parse_response(raw)
            if parsed is None or "text" not in parsed:
                continue
            queries.append(Query(
                query_id = f"synthetic-{i}",
                text = parsed["text"],
                gold_doc_ids = [str(chunk["doc_name"])]
            ))
        return queries

    @staticmethod
    def _parse_response(raw:str) -> Optional[Dict[str,Any]]:
        if not raw:
            return None

        raw = raw.strip()
        try:
            return json.loads(raw)

        except Exception:
            pass
        match = re.search(r"\{.*\}",raw,re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

        return None

            

