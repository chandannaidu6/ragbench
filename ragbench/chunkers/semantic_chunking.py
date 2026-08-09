import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List,Dict,Any,Tuple
from ragbench.embeddings.openai_embedder import OpenAIEmbedder
from ragbench.chunkers.base import BaseChunker

class SemanticChunker(BaseChunker):
    def __init__(self,pdf_path:str|Path,embedder:OpenAIEmbedder | None):
        self.pdf_path = pdf_path
        self.embedder = embedder or OpenAIEmbedder()

    @staticmethod
    def clean_text(text:str) -> str:
        text = text.replace("\r\n","\n").replace("\r","\n")
        text = re.sub(r"[ \t]+"," ",text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}","\n\n",text)
        return text.strip()


    @staticmethod
    def split_sentences(text:str,min_length=20) -> List[str]:
        sentences = re.split(r'(?<=[.?!])\s+',text)
        return [s.strip() for s in sentences if len(s.strip())>=min_length] 

    def extract_sentence_units(self,min_length:int = 20) -> List[Dict[str,Any]]:
        doc_name = self.pdf_path.split("/")[-1]
        sentence_units:List[Dict[str,Any]] = []
        global_idx = 0
        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                raw_text = page.get_text()
                cleaned = self.clean_text(raw_text)

                if not cleaned:
                    continue

                sentences = self.split_sentences(cleaned,min_length=min_length)
                for sent in sentences:
                    sentence_units.append(
                        {
                            "text":sent,
                            "doc_name":doc_name,
                            "page_number":page.number+1,
                            "global_idx":global_idx+1
                        }
                    )
                    global_idx += 1

        return sentence_units

    @staticmethod
    def cosine_similarity(a:np.ndarray,b:np.ndarray)->float:
        denom = (np.linalg.norm(a)*np.linalg.norm(b))
        if denom == 0:
            return 0.0

        return float(np.dot(a,b)/denom)

    def compute_adjacent_similarities(self,sentences:List[Dict[str,Any]])->np.ndarray:
        if len(sentences)<2:
            return np.array([],dtype=np.float32)

        texts = [s['text'] for s in sentences]
        embeddings = self.embedder.embed_sentences(texts)
        sims = []
        for i in range(len(embeddings)-1):
            sim = self.cosine_similarity(embeddings[i],embeddings[i+1])
            sims.append(sim)

        return np.array(sims,dtype=np.float32)

    @staticmethod
    def _build_chunk(chunk_id:int,items:List[Dict[str,Any]],boundary_reason:str|None=None,position:str="middle")->Dict[str,Any]:
        return {
            "chunk_id":chunk_id,
            "text":" ".join(item["text"] for item in items),
            "doc_name":items[0]["doc_name"],
            "pages":sorted({item["page_number"] for item in items}),
            "chunker_name":"semantic",
            "source_unit": "sentences",
            "start_idx": items[0]["global_idx"],
            "end_idx": items[-1]["global_idx"],
            "sentence_count": len(items),
            "word_count": sum(len(item["text"].split()) for item in items),
            "position": position,
            "boundary_reason": boundary_reason,
            "source_sentences": items,    
        }

    @staticmethod
    def _merge_small_chunks(chunks:List[Dict[str,Any]],min_sentences:int,)->List[Dict[str,Any]]:
        if not chunks:
            return chunks

        merged:List[Dict[str,Any]] = []
        for chunk in chunks:
            if (merged and chunk["sentence_count"]<min_sentences):
                prev = merged[-1]
                combined_sentences = prev["source_sentences"]+chunk["source_sentences"]
                merged[-1] = {
                    **prev,
                    "text":" ".join(s["text"] for s in combined_sentences),
                    "pages":sorted({s["page_number"] for s in combined_sentences}),
                    "end_idx":combined_sentences[-1]["global_idx"],
                    "sentence_count":len(combined_sentences),
                    "word_count":sum(len(s["text"].split()) for s in combined_sentences),
                    "source_sentences":combined_sentences,
                    "boundary_reason":"merged_small_chunk"
                }
            else:
                merged.append(chunk)

        if len(merged)>=2 and merged[0]["sentence_count"] < min_sentences:
            first = merged.pop(0)
            second = merged.pop(0)
            combined_sentences = first["source_sentences"] + second["source_sentences"]
            merged.insert(0,{
                **second,
                "text":" ".join(s["text"] for s in combined_sentences),
                "pages":sorted({s["page_number"] for s in combined_sentences}),
                "start_idx":combined_sentences[0]["global_idx"],
                "end_idx":combined_sentences[-1]["global_idx"],
                "sentence_count":len(combined_sentences),
                "word_count":sum(len(s["text"].split()) for s in combined_sentences),
                "source_sentences":combined_sentences,
                "boundary_reason":"merged_small_chunk",            
            }
            )
        return merged

    def build_chunks(self,threshold:float = 0.55,min_sentences:int = 2,max_sentences:int = 6, min_length:int = 20)->List[Dict[str,Any]]:
        if min_sentences<=0:
            raise ValueError("min sentences must be > 0")
        if max_sentences<min_sentences:
            raise ValueError("max sentences must be >= min sentences")

        sentences  = self.extract_sentence_units(min_length=min_length)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                self._build_chunk(
                    chunk_id = 0,
                    items = sentences,
                    boundary_reason="single sentence",
                    position="full"
                )
            ]
        similarities = self.compute_adjacent_similarities(sentences)
        chunks:List[Dict[str,Any]] = []
        current_chunk = [sentences[0]]
        chunk_id = 0

        for i in range(1,len(sentences)):
            sim_prev = similarities[i-1]
            current_size = len(current_chunk)
            should_split_similarity = sim_prev<threshold and current_size>=min_sentences
            should_split_max = current_size>=max_sentences

            if should_split_similarity or should_split_max:
                boundary_reason = "similarity_drop" if should_split_similarity else "max_sentences"
                chunks.append(self._build_chunk(
                    chunk_id=chunk_id,
                    items=current_chunk,
                    boundary_reason=boundary_reason
                ))
                chunk_id+=1
                current_chunk = [sentences[i]]

            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(self._build_chunk(
                chunk_id=chunk_id,
                items=current_chunk,
                boundary_reason="document_end"
            ))

        chunks = self._merge_small_chunks(chunks,min_sentences=min_sentences)

        for idx,chunk in enumerate(chunks):
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






            

            



            


            




        







