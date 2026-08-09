"""
retrievers/bm25.py

Standalone BM25 retriever. Builds its own vocabulary and IDF statistics
directly from chunk texts at construction time (tokenize -> vocabulary ->
tf/idf), so it needs nothing but `chunks` to work - no external
preprocessing step or vocabulary builder required.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

from ragbench.retrievers.base import BaseRetriever

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(str(text).lower())


class BM25(BaseRetriever):
    retriever_name = "bm25"

    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.vocabulary: Dict[str, int] = {}
        self.stats: Dict[str, Any] = {}

    def build_index(self) -> None:
        tokenized = [tokenize(c.get("text", "")) for c in self.chunks]

        vocabulary: Dict[str, int] = {}
        for tokens in tokenized:
            for word in tokens:
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)

        self.vocabulary = vocabulary
        self.stats = self._compute_stats(tokenized, vocabulary)

    @staticmethod
    def _compute_stats(tokenized: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        N = len(tokenized)
        V = len(vocabulary)
        tf_matrix = np.zeros((N, V), dtype=np.float32)
        df_array = np.zeros(V, dtype=np.float32)
        doc_len = np.zeros(N, dtype=np.float32)

        for chunk_id, tokens in enumerate(tokenized):
            doc_len[chunk_id] = len(tokens)
            seen_in_chunk = set()
            for word in tokens:
                word_id = vocabulary[word]
                tf_matrix[chunk_id][word_id] += 1
                if word_id not in seen_in_chunk:
                    df_array[word_id] += 1
                    seen_in_chunk.add(word_id)

        avg_dl = float(np.mean(doc_len)) if N else 0.0
        idf_array = np.log(((N - df_array + 0.5) / (df_array + 0.5)) + 1)
        return {
            "N": N,
            "tf_matrix": tf_matrix,
            "df_array": df_array,
            "doc_len": doc_len,
            "avgdl": avg_dl,
            "idf_array": idf_array,
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.vocabulary:
            self.build_index()

        query_word_ids = [self.vocabulary[w] for w in tokenize(query) if w in self.vocabulary]
        if not query_word_ids:
            return []

        N = self.stats["N"]
        tf_matrix = self.stats["tf_matrix"]
        doc_len = self.stats["doc_len"]
        avgdl = self.stats["avgdl"] or 1.0
        idf_array = self.stats["idf_array"]

        scores = np.zeros(N, dtype=np.float32)
        for word_id in query_word_ids:
            idf = idf_array[word_id]
            tf = tf_matrix[:, word_id]
            candidates = tf > 0
            len_norm = 1.0 - self.b + self.b * (doc_len[candidates] / avgdl)
            numerator = tf[candidates] * (self.k1 + 1.0)
            denominator = tf[candidates] + self.k1 * len_norm
            word_score = idf * (numerator / denominator)
            scores[candidates] += word_score

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "score": float(scores[idx]),
                "text": chunk.get("text", ""),
                "doc_name": chunk.get("doc_name", "Unknown"),
                "pages": chunk.get("pages", []),
                "chunk_id": chunk.get("chunk_id", int(idx)),
            })
        return results
