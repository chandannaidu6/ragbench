from typing import Any, Dict, Optional

PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI GPT - flagship
    "gpt-5.6-sol": {"input": 5.00, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-pro": {"input": 30.00, "output": 180.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    # OpenAI GPT-4 series
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
    "gpt-4-0613": {"input": 30.00, "output": 60.00},
    # OpenAI reasoning models
    "o1": {"input": 15.00, "output": 60.00},
    "o1-pro": {"input": 150.00, "output": 600.00},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-pro": {"input": 20.00, "output": 80.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # OpenAI GPT-3.5 series
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    # Anthropic Claude
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "claude-mythos-5": {"input": 10.00, "output": 50.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
    # claude-sonnet-5 is $2.00/$10.00 as an introductory rate through
    # 2026-08-31; this is the standard rate that applies afterward.
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
    # Google Gemini (text tier, prompts <=200k tokens - longer prompts on
    # gemini-3.1-pro-preview/gemini-2.5-pro bill at a higher rate that this
    # flat table doesn't model)
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.1-pro": {"input": 2.00, "output": 12.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash": {"input": 0.50, "output": 3.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    # Embedding models (embedding rate only)
    "text-embedding-3-small": {"embedding": 0.02},
    "text-embedding-3-large": {"embedding": 0.13},
    "text-embedding-ada-002": {"embedding": 0.10},
    "gemini-embedding-001": {"embedding": 0.15},
    "gemini-embedding-2": {"embedding": 0.20},
}
 
PER_TOKENS = 1_000_000
 
def _normalize_rates(rates: Dict[str, float]) -> Dict[str, float]:
    return {
        "input": float(rates.get("input", 0.0)),
        "output": float(rates.get("output", 0.0)),
        "embedding": float(rates.get("embedding", 0.0)),
    }

def get_price(model:str,override:Optional[Dict[str,Dict[str,float]]] = None):
    key = model.strip().lower()
    if override:
        ov = {k.lower():v for k,v in override.items()}
        if key in ov:
            return _normalize_rates(ov[key])
    if key in PRICING:
        return _normalize_rates(PRICING[key])

    return None

def _get(usage:Any,field:str):
    if isinstance(usage,dict):
        return usage.get(field,0)
    return int(getattr(usage,field,0) or 0)

def estimate_cost(usage: Any, model: str,overrides: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    prompt = _get(usage, "prompt_tokens")
    completion = _get(usage, "completion_tokens")
    embedding = _get(usage, "embedding_tokens")
 
    rates = get_price(model, overrides)
    if rates is None:
        import warnings
        warnings.warn(f"Unknown model '{model}': cost not tracked (returning 0.0).")
        return {"total": 0.0, "input_cost": 0.0, "output_cost": 0.0,
                "embedding_cost": 0.0, "tracked": False, "model": model}
 
    input_cost = prompt * rates["input"] / PER_TOKENS
    output_cost = completion * rates["output"] / PER_TOKENS
    embedding_cost = embedding * rates["embedding"] / PER_TOKENS
    return {
        "total": input_cost + output_cost + embedding_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "embedding_cost": embedding_cost,
        "tracked": True,
        "model": model,
    }

def estimate_cost_multi(usage_by_model: Dict[str, Any],
                        overrides: Optional[Dict[str, Dict[str, float]]] = None
                        ) -> Dict[str, Any]:
    total = 0.0
    per_model = {}
    any_untracked = False
    for model, usage in usage_by_model.items():
        c = estimate_cost(usage, model, overrides)
        per_model[model] = c
        total += c["total"]
        if not c["tracked"]:
            any_untracked = True
    return {"total": total, "per_model": per_model, "all_tracked": not any_untracked}

def aggregate_costs(per_query_costs: list, total_tokens: int, num_queries: int
                    ) -> Dict[str, float]:
    total_cost = float(sum(per_query_costs))
    return {
        "total_cost": total_cost,
        "cost_per_query": total_cost / num_queries if num_queries else 0.0,
        "cost_per_1k_tokens": (total_cost / (total_tokens / 1000)) if total_tokens else 0.0,
        "total_tokens": total_tokens,
        "num_queries": num_queries,
    }




