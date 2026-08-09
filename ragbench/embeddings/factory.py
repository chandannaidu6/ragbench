"""
embeddings/factory.py

The package - not the user - picks the embedder for a provider and gives it a
correct tokenizer. Because each BaseEmbedder implements count_tokens itself,
the TrackedEmbedder wrapper never needs a provider-specific count_fn from the
caller - it just calls embedder.count_tokens().
"""
from __future__ import annotations

from typing import Optional

from ragbench.embeddings.base import BaseEmbedder

SUPPORTED_EMBEDDING_PROVIDERS = {"openai", "huggingface", "ollama", "cohere", "voyage"}


def create_embedder(provider: str, model: str,
                    api_key: Optional[str] = None) -> BaseEmbedder:
    """Build a provider-specific embedder. Lazy-imports the concrete class so
    a missing provider SDK doesn't break the whole package on import."""
    provider = provider.lower()

    if provider == "openai":
        from ragbench.embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(model=model, api_key=api_key)

    if provider == "huggingface":
        # Local, free: sentence-transformers, no API key required.
        from ragbench.embeddings.hf_embedder import HuggingFaceEmbedder
        return HuggingFaceEmbedder(model=model)

    if provider == "ollama":
        # Local, free: talks to a local Ollama server, no API key required.
        from ragbench.embeddings.ollama_embedder import OllamaEmbedder
        return OllamaEmbedder(model=model)

    if provider == "cohere":
        raise NotImplementedError(
            "Cohere embedder not implemented yet. Add embeddings/cohere_embedder.py "
            "implementing BaseEmbedder, then wire it here."
        )

    if provider == "voyage":
        raise NotImplementedError(
            "Voyage embedder not implemented yet. Add embeddings/voyage_embedder.py "
            "implementing BaseEmbedder, then wire it here."
        )

    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))}"
    )
