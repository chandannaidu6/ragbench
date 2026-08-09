"""
ragbench/cli.py

Command-line entry point. Ties config -> runner -> results together.

Usage:
    ragbench run --config run.json
    ragbench compare --config matrix.json [--save-json out.json] [--save-csv out.csv]
    ragbench detail --config run.json --chunker recursive --retriever bm25
    ragbench list-chunkers
    ragbench list-retrievers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragbench.config import RunConfig, MatrixConfig, VALID_CHUNKERS, VALID_RETRIEVERS
from ragbench.evaluation.runner import run_single_isolated, run_matrix
from ragbench.reporting.results import (
    print_comparison_table, print_detail, save_json, save_csv,
)


def _load_config_file(path: str) -> dict:
    """Load a JSON (or YAML, if pyyaml is installed) config file into a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load .yaml config files. "
                "Install it with: pip install ragbench[yaml]"
            )
        return yaml.safe_load(text)
    return json.loads(text)


def cmd_run(args: argparse.Namespace) -> int:
    data = _load_config_file(args.config)
    config = RunConfig(**data)
    result = run_single_isolated(config)
    print_comparison_table([result])
    if args.save_json:
        save_json([result], args.save_json)
        print(f"Saved JSON -> {args.save_json}")
    if args.save_csv:
        save_csv([result], args.save_csv)
        print(f"Saved CSV -> {args.save_csv}")
    return 1 if result.failed else 0


def cmd_compare(args: argparse.Namespace) -> int:
    data = _load_config_file(args.config)
    matrix = MatrixConfig(**data)
    results = run_matrix(matrix)
    print_comparison_table(results, sort_by=args.sort_by)
    if args.save_json:
        save_json(results, args.save_json)
        print(f"Saved JSON -> {args.save_json}")
    if args.save_csv:
        save_csv(results, args.save_csv)
        print(f"Saved CSV -> {args.save_csv}")
    return 1 if any(r.failed for r in results) else 0


def cmd_detail(args: argparse.Namespace) -> int:
    data = _load_config_file(args.config)
    data["chunker_name"] = args.chunker
    data["retriever_name"] = args.retriever
    config = RunConfig(**data)
    result = run_single_isolated(config)
    print_detail(result)
    return 1 if result.failed else 0


def cmd_list_chunkers(args: argparse.Namespace) -> int:
    for name in sorted(VALID_CHUNKERS):
        print(name)
    return 0


def cmd_list_retrievers(args: argparse.Namespace) -> int:
    for name in sorted(VALID_RETRIEVERS):
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbench",
        description="Benchmark chunking and retrieval strategies on your own data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a single chunker x retriever config")
    p_run.add_argument("--config", required=True, help="Path to a RunConfig JSON/YAML file")
    p_run.add_argument("--save-json", help="Also write full results to this JSON path")
    p_run.add_argument("--save-csv", help="Also write the summary table to this CSV path")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Run a matrix of chunkers x retrievers")
    p_cmp.add_argument("--config", required=True, help="Path to a MatrixConfig JSON/YAML file")
    p_cmp.add_argument("--sort-by", default="recall_at_k",
                       choices=["recall_at_k", "mrr", "p95_ms", "cost_usd"])
    p_cmp.add_argument("--save-json", help="Also write full results to this JSON path")
    p_cmp.add_argument("--save-csv", help="Also write the summary table to this CSV path")
    p_cmp.set_defaults(func=cmd_compare)

    p_detail = sub.add_parser("detail", help="Full metric breakdown for one combo")
    p_detail.add_argument("--config", required=True, help="Path to a base RunConfig JSON/YAML file")
    p_detail.add_argument("--chunker", required=True, choices=sorted(VALID_CHUNKERS))
    p_detail.add_argument("--retriever", required=True, choices=sorted(VALID_RETRIEVERS))
    p_detail.set_defaults(func=cmd_detail)

    p_lc = sub.add_parser("list-chunkers", help="List available chunker names")
    p_lc.set_defaults(func=cmd_list_chunkers)

    p_lr = sub.add_parser("list-retrievers", help="List available retriever names")
    p_lr.set_defaults(func=cmd_list_retrievers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
