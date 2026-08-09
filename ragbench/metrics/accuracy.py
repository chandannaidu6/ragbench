import math

class Evaluation:
    def __init__(self,retrieved_ids,source_ids,k,total = None):
        deduped = self._dedup_keep_first([str(x) for x in list(retrieved_ids)[:k]])
        self.source_ids =  {str(x) for x in source_ids}
        self.k = k
        self.ranked = deduped
        self.top_k = deduped[:k]
        self.total = total

    @staticmethod
    def _dedup_keep_first(ids):
        seen = set()
        out = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def _relevant_in_topk(self):
        return [d for d in self.top_k if d in self.source_ids]


    def hit_at_k(self)->int:
        return len(self._relevant_in_topk())
    
    def hit_rate_k(self)->bool:
        return self.hit_at_k()>0
    
    def precision_at_k(self)->float:
        if self.k == 0:
            return 0.0
        
        return self.hit_at_k()/self.k
    

    def recall_at_k(self)->float:
        if not self.source_ids:
            return 0.0
        return self.hit_at_k()/len(self.source_ids)

    def mrr_at_k(self)->float:
        for rank,doc in enumerate(self.top_k,start=1):
            if doc in self.source_ids:
                return 1.0/rank
        return 0.0

    def ndcg_at_k(self,gold_relevance = None)->float:
        rel = {str(d):1 for d in self.source_ids} if gold_relevance is None \
            else {str(d):v for d,v in gold_relevance.items()}
        dcg = 0.0
        for i,doc in enumerate(self.top_k,start=1):
            dcg+= rel.get(doc,0)/math.log2(i+1)

        ideal_gains = sorted(rel.values(), reverse=True)[:self.k]
        idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_gains, start=1))

        return dcg / idcg if idcg > 0 else 0.0



    


