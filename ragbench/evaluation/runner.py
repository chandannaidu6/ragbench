from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ragbench.schema import Query, Chunk, RetrievalResult, RetrievedChunk, Usage
from ragbench.config import RunConfig, MatrixConfig
from ragbench.data import loader
from ragbench.data.benchmark import Benchmark
from ragbench.evaluation.synthetic_benchmark import SyntheticGenerator
from ragbench.chunkers.registry import Registry as ChunkerRegistry
from ragbench.retrievers.registry import RegistryRetriever
from ragbench.metrics.accuracy import Evaluation
from ragbench.metrics.latency import Timer, summarize
from ragbench.metrics.cost import estimate_cost, aggregate_costs
from ragbench.evaluation.usage import UsageTracker, TrackedEmbedder, TrackedLLMClient


class RunResult(BaseModel):
    chunker_name: str
    retriever_name: str
    reranked: bool = False
    benchmark_type: str
    num_queries: int
    num_chunks: int
    index_build_ms: float
    accuracy: Dict[str, float]          
    latency: Dict[str, float]           
    cost: Dict[str, float]              
    errors: List[str] = []              
    failed: bool = False                
    error_message: Optional[str] = None


def load_and_chunk(config: RunConfig) -> List[Dict[str, Any]]:
    result = loader.load(config.corpus_path)

    if not result.needs_chunking:
        return [c.model_dump() for c in result.chunks]

    # source_path may be a single PDF or a directory of PDFs (loader.py
    # detects both) - every concrete chunker only knows how to open ONE
    # file (pymupdf.open(self.pdf_path)), so a directory is expanded here,
    # once, into one chunker instance per file, rather than teaching all 5
    # chunkers about directories individually
    pdf_paths = _resolve_pdf_paths(result.source_path)

    all_chunks: List[Dict[str, Any]] = []
    for pdf_path in pdf_paths:
        chunker_registry = ChunkerRegistry(
            pdf_path=str(pdf_path),
            name=config.chunker_name,
            embedder=None,                # semantic chunker builds its own default embedder
            **config.chunker_params,
        )
        chunker = chunker_registry.get_chunker()
        chunks = chunker.build_chunks()
        # normalize to dicts regardless of whether the chunker returns Chunk objects or dicts
        all_chunks.extend(c.model_dump() if hasattr(c, "model_dump") else c for c in chunks)

    # each per-file chunker restarts chunk_id at 0 - re-sequence globally so
    # ids stay unique across files (chromadb upserts key off chunk_id)
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = idx

    return all_chunks


def _resolve_pdf_paths(source_path: Path) -> List[Path]:
    if source_path.is_dir():
        pdf_paths = sorted(source_path.glob("*.pdf"))
        if not pdf_paths:
            raise ValueError(f"No PDF files found in directory: {source_path}")
        return pdf_paths
    return [source_path]


# ---------------------------------------------------------------------------
# Stage 3: queries (real benchmark, or synthetic if none provided)
# ---------------------------------------------------------------------------

def get_queries(config: RunConfig, chunks: List[Dict[str, Any]],
                llm_client=None) -> (List[Query], str):
    has_real_benchmark = bool(config.benchmark_queries_path and config.benchmark_qrels_path)

    if has_real_benchmark:
        bench = Benchmark(
            qrel_path=config.benchmark_qrels_path,
            question_path=config.benchmark_queries_path,
            chunks=[Chunk(**c) for c in chunks],
        )
        return bench.load(), "real"

    if llm_client is None:
        raise ValueError(
            "No benchmark files provided and no llm_client available to "
            "generate a synthetic benchmark."
        )
    gen = SyntheticGenerator(llm_client=llm_client, chunks=chunks)
    return gen.generate_benchmark_file(), "synthetic"


# ---------------------------------------------------------------------------
# Stage 4: build the retriever (+ its dependencies)
# ---------------------------------------------------------------------------

def build_retriever(config: RunConfig, chunks: List[Dict[str, Any]],
                    tracker: UsageTracker):
    """The wiring step: some retrievers need other retrievers/clients built
    first. Kept explicit and inline rather than hidden in the registry, so
    the dependency order is visible and easy to extend."""

    retriever_registry = RegistryRetriever(name=config.retriever_name)

    if config.retriever_name == "bm25":
        retriever = retriever_registry.get_retriever(chunks=chunks)

    elif config.retriever_name == "dense":
        embedder = _build_tracked_embedder(config, tracker)
        retriever = retriever_registry.get_retriever(chunks=chunks, embedder=embedder)

    elif config.retriever_name == "hybrid":
        embedder = _build_tracked_embedder(config, tracker)
        dense = RegistryRetriever(name="dense").get_retriever(chunks=chunks, embedder=embedder)
        dense.build_index()
        retriever = retriever_registry.get_retriever(
            bm25_chunks=chunks, dense_retriever=dense,
        )

    elif config.retriever_name == "hyde":
        embedder = _build_tracked_embedder(config, tracker)
        llm_client = _build_tracked_llm(config, tracker)
        retriever = retriever_registry.get_retriever(
            chunks=chunks, llm_client=llm_client, embedder=embedder,
        )

    else:
        raise ValueError(f"No wiring defined for retriever '{config.retriever_name}'")

    if config.use_reranker:
        from ragbench.retrievers.reranking import RerankingRetriever
        retriever = RerankingRetriever(
            retriever,
            model=config.reranker_model,
            candidate_k=config.rerank_candidate_k,
        )

    return retriever


def _build_tracked_embedder(config: RunConfig, tracker: UsageTracker):
    from ragbench.embeddings.factory import create_embedder
    raw = create_embedder(provider=config.embedding_provider, model=config.embedding_model,
                          api_key=config.embedding_api_key)
    return TrackedEmbedder(raw, tracker)


def _build_tracked_llm(config: RunConfig, tracker: UsageTracker):
    from ragbench.llm.factory import create_llm_client
    raw = create_llm_client(provider=config.llm_provider, model=config.llm_model,
                            api_key=config.llm_api_key)
    return TrackedLLMClient(raw, tracker)


# ---------------------------------------------------------------------------
# Stages 5-7: index, query loop, aggregate
# ---------------------------------------------------------------------------

def run_single(config: RunConfig, llm_client_for_synthetic=None) -> RunResult:
    try:
        # 1-2: load + chunk
        chunks = load_and_chunk(config)

        # 3: queries
        has_real_benchmark = bool(config.benchmark_queries_path and config.benchmark_qrels_path)
        if llm_client_for_synthetic is None and not has_real_benchmark:
            # the isolated-subprocess CLI path never has a live client to
            # pass in (RunConfig crosses the process boundary as JSON, not a
            # Python object) - build one here from the config itself so the
            # synthetic-benchmark path is actually reachable from the CLI
            from ragbench.llm.factory import create_llm_client
            llm_client_for_synthetic = create_llm_client(
                provider=config.llm_provider, model=config.llm_model,
                api_key=config.llm_api_key,
            )
        queries, benchmark_type = get_queries(config, chunks, llm_client_for_synthetic)

        # 4: build retriever (+ dependencies), with usage tracking wired in
        tracker = UsageTracker()
        retriever = build_retriever(config, chunks, tracker)

        # 5: build index (timed separately - one-time setup cost)
        with Timer() as index_timer:
            retriever.build_index()

        # 6: per-query loop
        per_query_metrics: List[Dict[str, float]] = []
        per_query_latencies: List[float] = []
        per_query_costs: List[float] = []
        total_tokens = 0
        errors: List[str] = []

        cost_model = config.llm_model if config.retriever_name == "hyde" \
            else config.embedding_model

        for query in queries:
            try:
                tracker.reset()
                with Timer() as t:
                    raw_results = retriever.retrieve(query.text, top_k=config.top_k)

                usage_snapshot = tracker.snapshot()
                doc_ids = [r.get("doc_name") if isinstance(r, dict) else r.doc_name
                          for r in raw_results]

                ev = Evaluation(retrieved_ids=doc_ids, source_ids=query.gold_doc_ids,
                                k=config.top_k)
                per_query_metrics.append({
                    "hit_at_k": ev.hit_at_k(),
                    "hit_rate_at_k": float(ev.hit_rate_k()),
                    "precision_at_k": ev.precision_at_k(),
                    "recall_at_k": ev.recall_at_k(),
                    "mrr": ev.mrr_at_k(),
                    "ndcg_at_k": ev.ndcg_at_k(query.gold_relevance or None),
                })
                per_query_latencies.append(t.ms)

                cost_result = estimate_cost(usage_snapshot, cost_model or "unknown")
                per_query_costs.append(cost_result["total"])
                total_tokens += (usage_snapshot.prompt_tokens
                                + usage_snapshot.completion_tokens
                                + usage_snapshot.embedding_tokens)

            except Exception as e:
                errors.append(f"{query.query_id}: {e}")
                continue

        # 7: aggregate
        accuracy = _mean_metrics(per_query_metrics)
        latency_summary = summarize(per_query_latencies)
        cost_summary = aggregate_costs(per_query_costs, total_tokens, len(per_query_metrics))

        return RunResult(
            chunker_name=config.chunker_name,
            retriever_name=config.retriever_name + ("+rerank" if config.use_reranker else ""),
            reranked=config.use_reranker,
            benchmark_type=benchmark_type,
            num_queries=len(queries),
            num_chunks=len(chunks),
            index_build_ms=index_timer.ms,
            accuracy=accuracy,
            latency=latency_summary,
            cost=cost_summary,
            errors=errors,
        )

    except Exception as e:
        # a whole-run failure (bad config, load failure, etc.) does not
        # take down the rest of a matrix - it's recorded and skipped
        return RunResult(
            chunker_name=config.chunker_name,
            retriever_name=config.retriever_name + ("+rerank" if config.use_reranker else ""),
            reranked=config.use_reranker,
            benchmark_type="n/a",
            num_queries=0,
            num_chunks=0,
            index_build_ms=0.0,
            accuracy={}, latency={}, cost={},
            failed=True,
            error_message=str(e),
        )


def run_single_isolated(config: RunConfig, timeout: Optional[float] = None) -> RunResult:
    """Run one config in its own subprocess, so a hard crash in a retriever's
    native dependencies (a segfault in chromadb/torch/onnxruntime, for
    example - the OS kills the whole process before Python's try/except ever
    runs) only takes down that subprocess. The caller always gets a
    RunResult back - a crash just becomes an ordinary failed=True result -
    instead of losing every result in a matrix sweep to one bad combo."""
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ragbench.evaluation._isolated_worker"],
            input=config.model_dump_json(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _crash_result(config, f"Timed out after {timeout}s.")

    if proc.returncode == 0 and proc.stdout.strip():
        try:
            return RunResult.model_validate_json(proc.stdout)
        except Exception:
            pass  # fall through to the crash-result path below

    stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:]) if proc.stderr else ""
    return _crash_result(
        config,
        f"Subprocess exited with code {proc.returncode} - likely a native-library "
        f"crash (e.g. a segfault in chromadb/torch/onnxruntime) that Python's "
        f"try/except cannot catch, since the OS kills the process before any "
        f"exception can be raised.\n{stderr_tail}",
    )


def _crash_result(config: RunConfig, message: str) -> RunResult:
    return RunResult(
        chunker_name=config.chunker_name,
        retriever_name=config.retriever_name + ("+rerank" if config.use_reranker else ""),
        reranked=config.use_reranker,
        benchmark_type="n/a",
        num_queries=0, num_chunks=0, index_build_ms=0.0,
        accuracy={}, latency={}, cost={},
        failed=True,
        error_message=message,
    )


def run_matrix(matrix_config: MatrixConfig, llm_client_for_synthetic=None) -> List[RunResult]:
    results: List[RunResult] = []
    for run_config in matrix_config.expand():
        if llm_client_for_synthetic is None:
            # no live object to hand to a subprocess - safe to isolate
            results.append(run_single_isolated(run_config))
        else:
            results.append(run_single(run_config, llm_client_for_synthetic))
    return results


def _mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(m[k] for m in per_query) / len(per_query) for k in keys}