import time
import math
from typing import List, Dict, Optional

class Timer:
    def __init__(self):
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.ms: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.ms = (self.end - self.start)*1000
        return False

def _percentile(sorted_values:List[float],pct:float)->float:
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = math.ceil((pct/100)*len(sorted_values))
    rank = max(1,min(rank,len(sorted_values)))
    return sorted_values[rank-1]

def summarize(latencies_ms:List[float])->Dict[str,float]:
    if not latencies_ms:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0,
                "p99": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}

    s = sorted(latencies_ms)
    n = len(s)
    total = sum(s)
    return {
        "count": n,
        "mean": total / n,
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
        "min": s[0],
        "max": s[-1],
        "total": total,
    }



    
