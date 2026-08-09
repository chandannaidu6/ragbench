"""
ragbench/api.py

The Python-first entry point for using ragbench as a library instead of the
CLI: build a config from keyword arguments, run it, get RunResult(s) back
directly - no JSON config file, no terminal.

By default, API keys aren't passed as arguments at all - every LLM/embedding
client already reads its provider's standard env var (OPENAI_API_KEY,
ANTHROPIC_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY) via os.getenv. Put them
in a .env file next to your script; `import ragbench` auto-loads it (see
ragbench/__init__.py), so os.getenv sees them with zero extra setup - this
is what makes `ragbench.run(...)` usable without ever touching a terminal.

For callers that manage keys themselves instead (multiple accounts, a
secrets manager, etc.), run()/compare() also accept an explicit `api_key`
override - see below.
"""
from __future__ import annotations

from typing import Any, List, Optional

from ragbench.config import RunConfig, MatrixConfig
from ragbench.evaluation.runner import RunResult, run_matrix, run_single, run_single_isolated


def _apply_api_key(config_kwargs: dict, api_key: Optional[str]) -> dict:
    if api_key is not None:
        config_kwargs.setdefault("llm_api_key", api_key)
        config_kwargs.setdefault("embedding_api_key", api_key)
    return config_kwargs


def run(*, isolated: bool = False, api_key: Optional[str] = None,
       **config_kwargs: Any) -> RunResult:

    config = RunConfig(**_apply_api_key(config_kwargs, api_key))
    return run_single_isolated(config) if isolated else run_single(config)


def compare(*, api_key: Optional[str] = None, **config_kwargs: Any) -> List[RunResult]:

    matrix = MatrixConfig(**_apply_api_key(config_kwargs, api_key))
    return run_matrix(matrix)
