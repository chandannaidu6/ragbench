from __future__ import annotations
from typing import List,Dict,Any,Tuple,Optional
from ragbench.retrievers.bm25 import BM25
from ragbench.retrievers.dense import DenseRetriever
from ragbench.retrievers.base import BaseRetriever

class HybridRetriever(BaseRetriever):
    retriever_name = "hybrid"

    def __init__(self,bm25_chunks:List[Dict[str,Any]],dense_retriever:DenseRetriever,rrf_k:int = 60,bm25_weight:float = 1.0,dense_weight:float=1.0):
        self.bm25_chunks = bm25_chunks
        self._bm25 = BM25(chunks=bm25_chunks)
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def build_index(self) -> None:
        # dense_retriever's own index is expected to already be built by the
        # caller (see evaluation/runner.py) - rebuilding it here would mean
        # re-embedding every chunk a second time.
        self._bm25.build_index()

    def _bm25_search(self,query:str,top_k:int = 5)->List[Dict[str,Any]]:
        return self._bm25.retrieve(query,top_k=top_k)

    def _dense_search(self,query:str,top_k:int = 5)->List[Dict[str,Any]]:
        return self.dense_retriever.retrieve(query,top_k)


    @staticmethod
    def _doc_key(doc:Dict[str,Any])->str:
        return f"{doc.get('doc_name', 'Unknown')}::{doc.get('chunk_id', 'na')}"

    def _rrf_merge(self,rankings:List[Tuple[str,float,List[Dict[str,Any]]]])->List[Dict[str,Any]]:
        fused:Dict[str,Dict[str,Any]] = {}
        for source_name,weight,results in rankings:
            for rank,item in enumerate(results,start=1):
                key = self._doc_key(item)
                if key not in fused:
                    fused[key] = {
                        "doc_name":item.get("doc_name","Unknown"),
                        "chunk_id":item.get("chunk_id"),
                        "page_number":item.get("page_number"),
                        "score":0.0,
                        "text":item.get("text",""),
                        "source_scores": {},
                        "sources":set()
                    }
                fused[key]["score"] += weight * (1.0/(self.rrf_k+rank))
                fused[key]["source_scores"][source_name] = item.get("score")
                fused[key]["sources"].add(source_name)

                if not fused[key]["text"] and item.get("text"):
                    fused[key]["text"] = item.get("text")
        merged = list(fused.values())
        for item in merged:
            item["sources"] = sorted(item["sources"])

        merged.sort(key=lambda x:x["score"],reverse=True)
        return merged

    def retrieve(self,query:str,top_k:int = 5,candidate_k:Optional[int] = None)->List[Dict[str,Any]]:
        candidate_k = candidate_k or max(top_k*15,10)
        bm25_results = self._bm25_search(query,candidate_k)
        dense_results = self._dense_search(query,candidate_k)
        merged = self._rrf_merge([
            ("bm25",self.bm25_weight,bm25_results),
            ("dense",self.dense_weight,dense_results)
        ])
        return merged[:top_k]








