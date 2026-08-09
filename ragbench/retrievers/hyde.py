from typing import Any, Dict, List, Optional

from ragbench.embeddings.openai_embedder import OpenAIEmbedder
from ragbench.storage.chroma_client import ChromaVector
from ragbench.llm.openai_client import OpenAIClient
from ragbench.retrievers.base import BaseRetriever

class HydeRetriever(BaseRetriever):
    retriever_name = "hyde"

    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None,
                 llm_client: OpenAIClient | None = None,
                 embedder: OpenAIEmbedder | None = None,
                 vector_store: ChromaVector | None = None,
                 mode: str = "short", batch_size: int = 512):
        self.chunks = chunks or []
        self.llm_client = llm_client
        self.embedder = embedder or OpenAIEmbedder()
        self.vector_store = vector_store or ChromaVector(collection_name="rag_chunks_hyde")
        self.mode = mode
        self.batch_size = batch_size

    def build_index(self) -> None:
        if not self.chunks:
            return
        # start from a clean collection - otherwise a later run in a matrix
        # sweep can retrieve stale vectors left over from a previous chunker
        self.vector_store.reset_collection()

        texts = [c["text"] for c in self.chunks]
        embeddings = self.embedder.embed_texts(texts)
        ids, metadatas = [], []
        for id_, chunk in enumerate(self.chunks):
            ids.append(str(id_))
            metadatas.append({
                "doc_name": chunk.get("doc_name", "Unknown"),
                "page_number": chunk.get("page_number"),
                "chunk_id": chunk.get("chunk_id", id_),
            })
        for i in range(0, len(ids), self.batch_size):
            batch_embeddings = embeddings[i:i + self.batch_size]
            self.vector_store.upsert(
                ids=ids[i:i + self.batch_size],
                documents=texts[i:i + self.batch_size],
                embeddings=batch_embeddings.tolist() if hasattr(batch_embeddings, "tolist") else batch_embeddings,
                metadatas=metadatas[i:i + self.batch_size],
            )

    def generate_hypothetical_document(self,query:str,mode:str = "short")->str:
        return self.llm_client.generate_context(query=query,mode=mode)

    def retrieve(self,query:str,top_k:int = 5,mode:Optional[str] = None) -> List[Dict[str, Any]]:
        hypothetical_doc = self.generate_hypothetical_document(query,mode or self.mode)
        q_vec = self.embedder.embed_text(hypothetical_doc)

        if hasattr(q_vec,"tolist"):
            q_vec = q_vec.tolist()

        results = self.vector_store.query(query_embeddings=q_vec,top_k=top_k)
        output = []
        docs = results["documents"]
        metas = results["metadatas"]
        dists = results["distances"]

        for doc,meta,dist in zip(docs,metas,dists):
            meta = meta or {}
            output.append({
                "score": float(1-dist),
                "text":doc,
                "doc_name":meta.get("doc_name","Unknown"),
                "page_number":meta.get("page_number"),
                "chunk_id":meta.get("chunk_id"),

            })
        return output
