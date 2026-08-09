from ragbench.schema import Usage
from typing import List, Callable, Optional
import numpy as np

class UsageTracker:
    def __init__(self):
        self.used = {"prompt":0,"completion":0,"embedding":0,"total":0.0}

    def add_llm(self,prompt_tokens:int,completion_tokens:int):
        self.used["prompt"] += prompt_tokens
        self.used["completion"] += completion_tokens

    def add_embedding(self,embedding_tokens):
        self.used["embedding"] += embedding_tokens

    def reset(self):
        self.used["prompt"] = 0
        self.used["completion"] = 0
        self.used["embedding"] = 0

    def snapshot(self)->Usage:
        return Usage(
            prompt_tokens = self.used["prompt"],
            completion_tokens = self.used["completion"],
            embedding_tokens  = self.used["embedding"]
        )

class TrackedEmbedder:
    def __init__(self,embedder,tracker,count_fn:Optional[Callable[[List[str]],int]] = None):
        self._embedder = embedder
        self._tracker = tracker
        # every BaseEmbedder implements count_tokens() itself, using its own
        # provider's tokenizer (or a reasonable approximation) - so this
        # works for any provider without needing a caller-supplied count_fn
        self._count = count_fn or self._embedder.count_tokens

    def embed_texts(self,texts:List[str]) -> np.ndarray:
        result = self._embedder.embed_texts(texts)
        self._tracker.add_embedding(self._count(texts))
        return result

    def embed_sentences(self,sentences:List[str]) -> np.ndarray:
        result = self._embedder.embed_texts(sentences)
        self._tracker.add_embedding(self._count(sentences))
        return result

    def embed_text(self,text:str)->np.ndarray:
        # embed_texts() takes a LIST of strings, not a bare string - a bare
        # string is itself iterable, so passing it directly here would embed
        # one vector PER CHARACTER instead of one vector for the whole text
        result = self._embedder.embed_texts([text])[0]
        self._tracker.add_embedding(self._count([text]))
        return result


class TrackedLLMClient:
    def __init__(self,client,tracker,count_fn:Optional[Callable[[str,str],tuple]] = None):
        self._client = client
        self._tracker = tracker
        self._count_fn = count_fn

    def _record(self,prompt_text:str,output_text:str):
        last = getattr(self._client,"last_usage",None)
        if last is not None:
            self._tracker.add_llm(int(last.get("prompt",0)),int(last.get("completion",0)))
            return

        if self._count_fn is not None:
            p,c = self._count_fn(prompt_text,output_text)
            self._tracker.add_llm(int(p),int(c))
            return

        raise ValueError(
            "LLM client does not expose `last_usage` and no count function was "
            "provided; cannot record token usage. Make the client report "
            "its usage, or pass a count function."
        )

    def generate_context(self,query:str,mode:str = "short")->str:
        result = self._client.generate_context(query=query,mode=mode)
        self._record(self._safe_prompt(query,None,mode),result)
        return result

    def chat(self,query:str,context:str,mode:str="plain")->str:
        result = self._client.chat(query=query,context=context,mode=mode)
        self._record(self._safe_prompt(query,context,mode),result)
        return result

    def complete(self,prompt:str)->str:
        result = self._client.complete(prompt)
        self._record(prompt,result)
        return result

    def _safe_prompt(self,query,context,mode):
        if hasattr(self._client,"build_prompt"):
            try:
                return self._client.build_prompt(query,context,mode)
            except Exception:
                return query

        return query

    




    

        