from typing import List,Dict,Any
from ragbench.embeddings.openai_embedder import OpenAIEmbedder
from ragbench.storage.chroma_client import ChromaVector
from ragbench.retrievers.base import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(self,chunks:List[Dict[str,Any]], embedder:OpenAIEmbedder | None = None,vector_store:ChromaVector | None = None, batch_size:int = 512):
        self.chunks = chunks
        self.embedder = embedder or OpenAIEmbedder()
        self.vector_store = vector_store or ChromaVector(collection_name="rag_chunks")
        self.batch_size = batch_size
    
    def build_index(self) -> None:
        # start from a clean collection - otherwise a later run in a matrix
        # sweep can retrieve stale vectors left over from a previous chunker
        self.vector_store.reset_collection()
        texts = [c['text'] for c in self.chunks]
        embeddings = self.embedder.embed_texts(texts)
        ids,metadatas = [],[]
        batch_size = self.batch_size
        for id,chunk in enumerate(self.chunks):
            ids.append(str(id))
            metadatas.append(
                {
                    "doc_name":chunk.get("doc_name","Unknown"),
                    "page_number":chunk.get("page_number"),
                    "chunk_id":id
                }
            )
        for i in range(0,len(ids),batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = texts[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            self.vector_store.upsert(
                ids = batch_ids,
                documents=batch_docs,
                embeddings = batch_embeddings.tolist() if hasattr(batch_embeddings,"tolist") else batch_embeddings,
                metadatas = batch_metas
            )
           


    def retrieve(self,query:str,top_k:int = 5,) -> List[Dict[str,Any]]:

        q_vec = self.embedder.embed_text(query)
        if hasattr(q_vec,"tolist"):
            q_vec = q_vec.tolist()

        results = self.vector_store.query(query_embeddings=q_vec,top_k=top_k)

        output: List[Dict[str,Any]] = []
        docs = results["documents"]
        metas = results["metadatas"]
        dists = results["distances"]
        for doc,meta,dist in zip(
            docs,
            metas,
            dists
        ):
            meta = meta or {}
            output.append(
                {
                    "score":float(1-dist),
                    "text":doc,
                    "doc_name":meta.get("doc_name","Unknown"),
                    "page_number":meta.get("page_number"),
                    "chunk_id":meta.get("chunk_id")
                }
            )
        return output

        



