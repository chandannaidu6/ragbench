"""
ragbench - benchmark chunking and retrieval strategies on your own data.

Quick start (no terminal/CLI required):
    import ragbench

    # reads OPENAI_API_KEY etc. from a .env file next to your script,
    # auto-loaded below - no manual env var setup needed
    result = ragbench.run(
        corpus_path="corpus.jsonl",
        benchmark_queries_path="queries.json",
        benchmark_qrels_path="qrels.json",
        chunker_name="recursive",
        retriever_name="bm25",
    )
    print(result.accuracy, result.cost)

    results = ragbench.compare(
        corpus_path="corpus.jsonl",
        benchmark_queries_path="queries.json",
        benchmark_qrels_path="qrels.json",
        chunker_names=["recursive", "fixed_size"],
        retriever_names=["bm25", "dense", "hyde"],
    )
"""
try:
    # searches the current + parent directories for a .env file and loads it
    # into os.environ - so every client's os.getenv("OPENAI_API_KEY") (etc.)
    # picks it up automatically, whether the caller used the CLI or imported
    # ragbench directly as a library. Never overrides a var already set in
    # the real environment.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ragbench.schema import Chunk, Query, RetrievedChunk, Usage, RetrievalResult, EvaluationMetrics
from ragbench.config import RunConfig, MatrixConfig
from ragbench.api import run, compare

__version__ = "0.1.1"

__all__ = [
    "Chunk", "Query", "RetrievedChunk", "Usage", "RetrievalResult", "EvaluationMetrics",
    "RunConfig", "MatrixConfig",
    "run", "compare",
    "__version__",
]
