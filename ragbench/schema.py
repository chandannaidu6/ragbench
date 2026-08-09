from pydantic import BaseModel,Field,field_validator
from typing import List, Optional,Literal,Dict,Any

class Chunk(BaseModel):
    chunk_id:int
    doc_name:str
    text:str
    pages:Optional[List[int]] = Field(default_factory=list)
    title:Optional[str] = None
    metadata:Optional[Dict[str,Any]] = Field(default_factory=dict)
    word_count:Optional[int] = None
    position:Optional[str] = None
    start_idx:Optional[int] = None
    end_idx:Optional[int] = None
    chunker_name:Optional[str] = None
    source_unit:Optional[str] = None

class Query(BaseModel):
    query_id:str
    text:str
    gold_doc_ids:List[str]
    gold_relevance:Optional[Dict[str,int]] = Field(default_factory=dict)
    metadata:Optional[Dict[str,Any]] = Field(default_factory=dict)

    @field_validator("gold_doc_ids")
    @classmethod
    def gold_must_not_be_empty(cls,v:List[str])->List[str]:
        if not v:
            raise ValueError(
                "gold_doc_ids is empty - the qrels/corpus join likely failed. "
                "A benchmark query must have at least one relevant doc id."
            )
        return v

class RetrievedChunk(BaseModel):
    doc_name:str
    chunk_id: Optional[int] =None
    score: float
    text:str
    pages:List[int] = Field(default_factory= list)
    rerank_score: Optional[float] = None

class Usage(BaseModel):
    prompt_tokens:int = 0
    completion_tokens:int = 0
    embedding_tokens:int = 0
    total_cost:float = 0.0

class RetrievalResult(BaseModel):
    query_id:str
    retriever:str
    retrieved:List[RetrievedChunk]
    latency_ms:Optional[float] = None
    usage:Optional[Usage] = None

class EvaluationMetrics(BaseModel):
    hit_at_k: int
    hit_rate_at_k: bool
    precision_at_k: float
    recall_at_k: float
    mrr: float = 0.0
    ndcg_at_k: Optional[float] = None

 


