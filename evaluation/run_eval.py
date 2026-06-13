"""
AGORA evaluation script — added for the paper's evaluation section.

Runs the 8 evaluation queries through the full AGORA pipeline and saves
structured results to eval_results.json. Each record captures:
  - question, type, domain
  - planner subqueries
  - datasets retrieved and selected
  - per-dataset routing decision (rag / technical)
  - final synthesised answer
  - pipeline latency and estimated LLM cost

Usage (from agora/src/backend with venv active and backend + Weaviate running):
    python -m evaluation.run_eval [--k 5] [--out evaluation/eval_results.json]

The script imports the orchestrator directly (no HTTP round-trip) so it captures
internal pipeline state that is not exposed through the streaming API.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as: python -m evaluation.run_eval (from agora/src/backend)
# or: python evaluation/run_eval.py (from agora/evaluation)
_repo_root = Path(__file__).resolve().parent.parent  # agora/
_backend = _repo_root / "src" / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.agents.orchestrator import AgentOrchestrator  # noqa: E402

# Import query set from sibling file
_eval_dir = Path(__file__).resolve().parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))
from queries import QUERIES  # noqa: E402


def run_one(orchestrator: AgentOrchestrator, entry: dict, k: int) -> dict:
    qid = entry["id"]
    question = entry["question"]
    print(f"\n{'='*68}")
    print(f"[{qid}] {question}")
    print(f"  type={entry['type']}  domain={entry['domain']}")

    t0 = time.time()
    try:
        result = orchestrator.run(question, k=k)
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        print(f"  ERROR after {elapsed}s: {exc}")
        return {**entry, "error": str(exc), "elapsed_s": elapsed}
    elapsed = round(time.time() - t0, 1)

    # ---- plan ----
    plan = result.plan
    subqueries = [sq.question for sq in (plan.subqueries if plan else [])]

    # ---- dataset selection + routing ----
    selected = result.selected_datasets or []
    routing_counts = {}
    dataset_decisions = []
    for ds in selected:
        mode = getattr(ds, "execution_mode", "unknown")
        routing_counts[mode] = routing_counts.get(mode, 0) + 1
        dataset_decisions.append({
            "dataset_id": getattr(ds, "dataset_id", None),
            "title": getattr(ds, "title", None),
            "execution_mode": mode,
        })

    # ---- usage / cost ----
    usage_raw = getattr(result, "usage", None) or {}
    total_tokens = 0
    cost_usd = None
    for v in (usage_raw.values() if hasattr(usage_raw, "values") else []):
        if hasattr(v, "total_tokens"):
            total_tokens += v.total_tokens
        if hasattr(v, "cost_usd") and v.cost_usd is not None:
            cost_usd = (cost_usd or 0) + v.cost_usd

    record = {
        "id": qid,
        "question": question,
        "type": entry["type"],
        "domain": entry["domain"],
        "elapsed_s": elapsed,
        "subqueries": subqueries,
        "n_subqueries": len(subqueries),
        "datasets_retrieved": len(result.hits or []),
        "datasets_selected": len(selected),
        "routing_counts": routing_counts,
        "dataset_decisions": dataset_decisions,
        "answer": result.answer,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }

    print(f"  elapsed={elapsed}s  subqueries={len(subqueries)}  "
          f"retrieved={record['datasets_retrieved']}  selected={len(selected)}  "
          f"routing={routing_counts}")
    snippet = (result.answer or "")[:200].replace("\n", " ")
    print(f"  answer: {snippet}...")
    return record


def summarise(results: list[dict]) -> None:
    print(f"\n\n{'='*68}")
    print("EVALUATION SUMMARY")
    print(f"{'='*68}")
    header = f"{'ID':<4} {'Domain':<14} {'Type':<12} {'Sub':>4} {'Sel':>4} {'RAG':>4} {'Tech':>5} {'Time':>6}s"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['id']:<4} {r['domain']:<14} {r['type']:<12}  ERR")
            continue
        rag = r["routing_counts"].get("rag", 0)
        tech = r["routing_counts"].get("technical", 0)
        print(f"{r['id']:<4} {r['domain']:<14} {r['type']:<12} "
              f"{r['n_subqueries']:>4} {r['datasets_selected']:>4} "
              f"{rag:>4} {tech:>5} {r['elapsed_s']:>6.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AGORA evaluation runner")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of datasets to retrieve per subquery (default 5)")
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).parent / "eval_results.json"),
                        help="Output JSON file path")
    parser.add_argument("--ids", nargs="*",
                        help="Run only specific query IDs, e.g. --ids Q1 Q3")
    args = parser.parse_args()

    queries = QUERIES
    if args.ids:
        queries = [q for q in queries if q["id"] in args.ids]

    out_path = Path(args.out)
    orchestrator = AgentOrchestrator()
    results = []

    for entry in queries:
        record = run_one(orchestrator, entry, k=args.k)
        results.append(record)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  -> saved to {out_path}")

    summarise(results)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
