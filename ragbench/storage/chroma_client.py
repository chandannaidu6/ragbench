"""
storage/chroma_client.py

Vector store backed by ChromaDB.

Packaging-safe:
  - Defaults to PersistentClient (embedded, SQLite-backed, NO server to run).
    This is the key fix: HttpClient required users to start a Chroma server
    first, which breaks a plain `pip install`. Embedded mode just works.
  - "memory" mode for pure in-RAM (nothing written to disk).
  - "http" mode still available for users who WANT a remote shared server.
  - chromadb is imported lazily, so `import ragbench` never fails just because
    Chroma isn't installed (only dense-family retrievers need it).

"memory" mode holds every embedding in the Python process's RAM with no disk
backing at all - unlike "persistent", it can't page anything out and the
whole corpus has to be re-embedded from scratch on every run. upsert() warns
once (see memory_mode_warn_at) if that corpus is large enough to risk
exhausting RAM.
"""
from __future__ import annotations

import os
import warnings
from typing import List, Dict, Any, Optional, Literal


class ChromaVector:
    def __init__(self, collection_name: str = "rag_chunks",
                 mode: Literal["persistent", "memory", "http"] = "persistent",
                 persist_directory: str = "./chroma_store",
                 host: Optional[str] = None, port: Optional[int] = None,
                 memory_mode_warn_at: Optional[int] = 20_000):
        try:
            import chromadb                          # lazy import
        except ImportError as e:
            raise ImportError(
                "Dense/HyDE/hybrid retrieval requires chromadb. "
                "Install it with:  pip install ragbench[chroma]"
            ) from e

        self.collection_name = collection_name
        self.mode = mode
        # only meaningful for mode="memory" - None disables the warning
        self._memory_mode_warn_at = memory_mode_warn_at
        self._total_docs = 0
        self._warned_large_corpus = False

        if mode == "http":
            # opt-in: connect to a running Chroma SERVER (advanced/shared setup)
            resolved_host = host or os.getenv("CHROMA_HOST", "localhost")
            env_port = os.getenv("CHROMA_PORT")
            resolved_port = port if port is not None else (int(env_port) if env_port else 8000)
            self.client = chromadb.HttpClient(host=resolved_host, port=resolved_port)
        elif mode == "memory":
            # pure in-RAM: nothing persisted, re-embeds every run
            self.client = chromadb.Client()
        else:  # persistent (DEFAULT): embedded SQLite file, no server needed
            self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids: List[str], documents: List[str],
               embeddings: List[List[float]],
               metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        # let chromadb handle "no metadata" itself - current chromadb
        # versions reject a list of empty {} dicts as invalid metadata
        self._total_docs += len(ids)
        if (self.mode == "memory" and self._memory_mode_warn_at is not None
                and not self._warned_large_corpus
                and self._total_docs >= self._memory_mode_warn_at):
            self._warned_large_corpus = True
            warnings.warn(
                f"ChromaVector(mode='memory') now holds {self._total_docs:,} chunks "
                f"entirely in RAM, with nothing written to disk - a corpus this "
                f"large risks exhausting memory, and everything has to be "
                f"re-embedded from scratch on every run. Consider "
                f"mode='persistent' (the default) instead. Raise this threshold "
                f"with memory_mode_warn_at=N, or pass memory_mode_warn_at=None "
                f"to silence it.",
                ResourceWarning,
                stacklevel=2,
            )

        self.collection.upsert(
            ids=ids, documents=documents,
            embeddings=embeddings, metadatas=metadatas,
        )

    def query(self, query_embeddings: List[float], top_k: int = 5) -> Dict[str, Any]:
        results = self.collection.query(
            query_embeddings=[query_embeddings], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
        }

    def reset_collection(self) -> None:
        self._total_docs = 0
        self._warned_large_corpus = False
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
