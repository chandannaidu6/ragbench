"""
llm/factory.py

The package - not the user - picks the right client class for a provider.
Add a new frontier model by adding one branch here; nothing else changes,
because everything downstream depends only on BaseLLMClient.
"""
from __future__ import annotations

from typing import Optional

from ragbench.llm.base import BaseLLMClient

SUPPORTED_LLM_PROVIDERS = {"openai", "anthropic", "google", "ollama", "cohere"}


def create_llm_client(provider: str, model: Optional[str] = None,
                      api_key: Optional[str] = None) -> BaseLLMClient:
    """Build a provider-specific LLM client. Lazy-imports the concrete client
    so a missing provider SDK doesn't break the whole package on import.
    `model=None` is valid - retrievers that don't need an LLM (bm25/dense/
    hybrid) leave RunConfig.llm_model unset, but callers that need *some*
    client (e.g. synthetic benchmark generation) still call this - omitting
    `model` from kwargs here lets each client fall back to its own default
    instead of being forced to `model=None`."""
    provider = provider.lower()
    kwargs = {"api_key": api_key} if model is None else {"model": model, "api_key": api_key}

    if provider == "openai":
        from ragbench.llm.openai_client import OpenAIClient
        return OpenAIClient(**kwargs)

    if provider == "anthropic":
        from ragbench.llm.anthropic_client import AnthropicClient
        return AnthropicClient(**kwargs)

    if provider == "google":
        from ragbench.llm.google_client import GoogleClient
        return GoogleClient(**kwargs)

    if provider == "ollama":
        # Local, free: talks to a local Ollama server, no API key required.
        from ragbench.llm.ollama_client import OllamaClient
        return OllamaClient(model=model) if model else OllamaClient()

    if provider == "cohere":
        raise NotImplementedError(
            "Cohere LLM client not implemented yet. Add llm/cohere_client.py "
            "implementing BaseLLMClient, then wire it here."
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}"
    )
