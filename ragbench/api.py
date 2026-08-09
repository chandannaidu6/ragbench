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
    """api_key is a convenience: it sets BOTH llm_api_key and
    embedding_api_key, covering the common case of one provider (or one key
    shared across providers). If the caller needs different keys per
    provider (e.g. embedding via OpenAI, generation via Anthropic in the
    same run), pass llm_api_key=/embedding_api_key= directly instead -
    those, if present, always win over the api_key shorthand."""
    if api_key is not None:
        config_kwargs.setdefault("llm_api_key", api_key)
        config_kwargs.setdefault("embedding_api_key", api_key)
    return config_kwargs


def run(*, isolated: bool = False, api_key: Optional[str] = None,
       **config_kwargs: Any) -> RunResult:
    """Run one chunker x retriever combination and return its RunResult.

    Example (keys from a .env file, the default):
        import ragbench

        result = ragbench.run(
            corpus_path="chunks.json",
            benchmark_queries_path="questions.json",
            benchmark_qrels_path="qrels.json",
            chunker_name="recursive",
            retriever_name="hyde",
            llm_provider="openai",
            llm_model="gpt-4.1-mini",
        )
        print(result.accuracy, result.cost)

    Example (explicit key instead of .env/environment):
        result = ragbench.run(
            corpus_path="chunks.json",
            chunker_name="recursive",
            retriever_name="hyde",
            llm_provider="anthropic",
            llm_model="claude-haiku-4-5",
            api_key="sk-ant-...",
        )

    config_kwargs accepts every RunConfig field (see ragbench.config.RunConfig)
    - corpus_path, chunker_name, retriever_name, embedding_provider/model,
    llm_provider/model, top_k, use_reranker, etc. This also includes
    llm_api_key/embedding_api_key directly, if you need different keys for
    the LLM vs. the embedder in the same run (api_key sets both to the same
    value; pass those two instead for the mixed-provider case).

    isolated=False (default) runs in-process: lower latency, and exceptions
    raise normally instead of being flattened into a RunResult.failed. Pass
    isolated=True to run in a subprocess instead, so a native-library crash
    (chromadb/torch/onnxruntime) can't take down your process - this is what
    the CLI always does, since a matrix sweep shouldn't lose every result to
    one bad combo, but a single interactive call is easy to just re-run.
    """
    config = RunConfig(**_apply_api_key(config_kwargs, api_key))
    return run_single_isolated(config) if isolated else run_single(config)


def compare(*, api_key: Optional[str] = None, **config_kwargs: Any) -> List[RunResult]:
    """Run every chunker x retriever combination in the matrix and return a
    RunResult per combo.

    Example:
        import ragbench

        results = ragbench.compare(
            corpus_path="chunks.json",
            benchmark_queries_path="questions.json",
            benchmark_qrels_path="qrels.json",
            chunker_names=["recursive"],
            retriever_names=["bm25", "dense", "hyde"],
        )
        for r in results:
            print(r.chunker_name, r.retriever_name, r.accuracy.get("recall_at_k"))

    config_kwargs accepts every MatrixConfig field (see
    ragbench.config.MatrixConfig) - the plural chunker_names/retriever_names
    plus the same corpus/model/reranker/api_key fields as run().

    Each combination runs in its own subprocess (same crash-isolation as the
    CLI's `ragbench compare`), so one bad combo just shows up as a failed
    result instead of losing the rest of the sweep.
    """
    matrix = MatrixConfig(**_apply_api_key(config_kwargs, api_key))
    return run_matrix(matrix)
