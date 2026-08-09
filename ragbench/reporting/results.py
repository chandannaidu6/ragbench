import json
import csv
from typing import List, Dict, Any, Optional
 
from rich.console import Console
from rich.table import Table
from rich.text import Text
 
from ragbench.evaluation.runner import RunResult

def _build_row(result:RunResult)->Dict[str,Any]:
    if result.failed:
        return {
            "chunker":result.chunker_name,
            "retriever": result.retriever_name,
            "status": "FAILED",
            "recall_at_k": None,
            "mrr": None,
            "p95_ms": None,
            "cost_usd": None,
            "benchmark_type": result.benchmark_type,
            "error": result.error_message,
        }
    if result.num_queries == 0 or not result.accuracy:
        status = "NO DATA"
    else:
        status = "ok"

    cost = result.cost.get("total_cost") if result.cost else None

    cost_tracked = result.cost.get("total_tokens",0) > 0 or (result.retriever_name.split("+")[0] == "bm25")

    return {
        "chunker": result.chunker_name,
        "retriever": result.retriever_name,
        "status": status,
        "recall_at_k": result.accuracy.get("recall_at_k") if status == "ok" else None,
        "mrr": result.accuracy.get("mrr") if status == "ok" else None,
        "p95_ms": result.latency.get("p95") if status == "ok" else None,
        "cost_usd": cost if (status == "ok" and cost_tracked) else (None if status != "ok" else "N/A" if cost is None else cost),
        "benchmark_type": result.benchmark_type,
        "error": None,
    }


def _find_winners(rows:List[Dict[str,Any]])->Dict[str,Any]:
    winners = {}
    higher_is_better = ["recall_at_k", "mrr"]
    lower_is_better = ["p95_ms", "cost_usd"]

    def key(r):
        return (r["chunker"],r["retriever"])

    for col in higher_is_better:
        candidates = [r for r in rows if isinstance(r.get(col),(int,float))]
        if candidates:
            best = max(candidates,key=lambda r: r[col])
            winners[col] = key(best)

    for col in lower_is_better:
        candidates = [r for r in rows if isinstance(r.get(col), (int, float))]
        if candidates:
            best = min(candidates, key=lambda r: r[col])
            winners[col] = key(best)
 
    return winners

def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    if value == "N/A":
        return "N/A"
    if kind == "score":
        return f"{value:.3f}"
    if kind == "ms":
        return f"{value:,.1f}"
    if kind == "usd":
        return f"${value:.4f}"
    return str(value)

def print_comparison_table(result:List[RunResult],sort_by:str = "recall_at_k",console:Optional[Console] = None)->None:
    console = console or Console()
    rows = [_build_row(r) for r in result]
    def sort_key(r):
        if r["status"] == "ok":
            val = r.get(sort_by)
            return (0,-(val if isinstance(val,(int,float)) else 0))
        elif r["status"] == "NO DATA":
            return (1,0)
        else:
            return (2,0)
    rows.sort(key=sort_key)
    winners = _find_winners(rows)
    table = Table(title="Chunker x Retriever Comparison", show_lines=False)
    table.add_column("Chunker", style="bold")
    table.add_column("Retriever", style="bold")
    table.add_column("Status")
    table.add_column("Recall@k", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Benchmark")
    
    def cell(row,col,kind):
        val = row.get(col)
        text = _fmt(val,kind)
        is_winner = winners.get(col) == (row["chunker"],row["retriever"])
        return Text(text,style="bold green" if is_winner else None)
    for row in rows:
        if row["status"] == "FAILED":
            table.add_row(
                row["chunker"], row["retriever"],
                Text("FAILED", style="bold red"),
                "-", "-", "-", "-", row["benchmark_type"],
            )
            continue
        if row["status"] == "NO DATA":
            table.add_row(
                row["chunker"], row["retriever"],
                Text("NO DATA", style="yellow"),
                "-", "-", "-", "-", row["benchmark_type"],
            )
            continue
 
        table.add_row(
            row["chunker"], row["retriever"], "ok",
            cell(row, "recall_at_k", "score"),
            cell(row, "mrr", "score"),
            cell(row, "p95_ms", "ms"),
            cell(row, "cost_usd", "usd"),
            row["benchmark_type"],
        )
 
    console.print(table)
    summary_parts = []
    for col, label in [("recall_at_k", "Best accuracy"), ("mrr", "Best ranking"),
                       ("p95_ms", "Fastest"), ("cost_usd", "Cheapest")]:
        key = winners.get(col)
        if key:
            summary_parts.append(f"{label}: [bold]{key[0]} + {key[1]}[/bold]")
    if summary_parts:
        console.print(" | ".join(summary_parts))
 
    failed_count = sum(1 for r in rows if r["status"] == "FAILED")
    if failed_count:
        console.print(f"[bold red]{failed_count} run(s) failed[/bold red] - see errors below:")
        for r in rows:
            if r["status"] == "FAILED":
                console.print(f"  [red]{r['chunker']} x {r['retriever']}[/red]: {r['error']}")



def print_detail(result: RunResult, console: Optional[Console] = None) -> None:
    console = console or Console()
    console.print(f"[bold]{result.chunker_name} x {result.retriever_name}[/bold] "
                  f"({result.benchmark_type} benchmark, {result.num_queries} queries)")
 
    if result.failed:
        console.print(f"[red]FAILED: {result.error_message}[/red]")
        return
 
    acc = Table(title="Accuracy", show_header=True)
    acc.add_column("Metric"); acc.add_column("Value", justify="right")
    for k, v in result.accuracy.items():
        acc.add_row(k, f"{v:.3f}")
    console.print(acc)
 
    lat = Table(title="Latency (ms)", show_header=True)
    lat.add_column("Stat"); lat.add_column("Value", justify="right")
    for k, v in result.latency.items():
        lat.add_row(k, f"{v:,.1f}")
    console.print(lat)
 
    cost = Table(title="Cost (USD)", show_header=True)
    cost.add_column("Field"); cost.add_column("Value", justify="right")
    for k, v in result.cost.items():
        cost.add_row(k, f"{v:.5f}" if "cost" in k else f"{v:,.0f}")
    console.print(cost)
 
    if result.errors:
        console.print(f"[yellow]{len(result.errors)} query error(s):[/yellow]")
        for e in result.errors[:10]:
            console.print(f"  {e}")


def save_json(results: List[RunResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, ensure_ascii=False)


def save_csv(results: List[RunResult], path: str) -> None:
    rows = [_build_row(r) for r in results]
    fieldnames = ["chunker", "retriever", "status", "recall_at_k", "mrr",
                 "p95_ms", "cost_usd", "benchmark_type", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)






        


