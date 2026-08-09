"""
evaluation/_isolated_worker.py

Runs exactly one RunConfig, in its own process. Invoked as
`python -m ragbench.evaluation._isolated_worker` with the RunConfig JSON on
stdin; writes the RunResult JSON to stdout.

This exists purely so that a hard crash in a retriever's native dependencies
(a segfault in chromadb/torch/onnxruntime's compiled bindings, for example -
these cannot be caught by Python's try/except, since the OS kills the whole
process before any Python exception machinery runs) only takes down this one
subprocess instead of an entire matrix sweep. See
evaluation.runner.run_single_isolated, which spawns this and turns a crashed
subprocess into an ordinary failed RunResult instead of losing every result
in the run.
"""
from __future__ import annotations

import contextlib
import io
import sys

from ragbench.config import RunConfig
from ragbench.evaluation.runner import run_single


def main() -> None:
    config_json = sys.stdin.read()
    config = RunConfig.model_validate_json(config_json)

    # run_single (and things it calls, like Benchmark.load()'s status
    # message) may print to stdout - redirect that away so this process's
    # real stdout carries ONLY the RunResult JSON the parent expects to parse
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_single(config)

    sys.stdout.write(result.model_dump_json())


if __name__ == "__main__":
    main()
