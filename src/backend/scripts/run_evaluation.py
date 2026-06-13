"""
Batch evaluation script for AGORA paper.
Runs a set of questions through the full pipeline, captures structured results,
and saves them to eval_results.json for paper writing.
"""
import json
import sys
import time
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.orchestrator import AgentOrchestrator

QUESTIONS = [
    # (id, question, type, domain)
    ("Q1", "Quels jeux de données sont disponibles sur la qualité de l'air en France ?",
     "discovery", "environment"),
    ("Q2", "Quelles données sur les accidents de la route en France sont disponibles ?",
     "discovery", "transport"),
    ("Q3", "Combien de stations de vélos en libre-service existe-t-il à Paris ?",
     "analytical", "transport"),
    ("Q4", "Quels sont les jeux de données sur le budget des communes françaises ?",
     "discovery", "finance"),
    ("Q5", "Quelles sont les communes françaises de plus de 100 000 habitants ?",
     "analytical", "demographics"),
    ("Q6", "Quels jeux de données concernent l'éducation nationale en France ?",
     "discovery", "education"),
    ("Q7", "Quels sont les jeux de données sur les énergies renouvelables en France ?",
     "discovery", "energy"),
    ("Q8", "Combien d'établissements de santé y a-t-il en Île-de-France ?",
     "analytical", "health"),
]

def run_query(qid, question, qtype, domain, k=5):
    print(f"\n{'='*70}")
    print(f"[{qid}] {question}")
    print(f"  type={qtype}  domain={domain}")
    print(f"{'='*70}")

    orchestrator = AgentOrchestrator()
    t0 = time.time()
    try:
        result = orchestrator.run(question, k=k)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    elapsed = round(time.time() - t0, 1)

    # Extract routing decisions
    selected = result.selected_datasets or []
    routing = {}
    for ds in selected:
        mode = getattr(ds, "execution_mode", "unknown")
        routing[mode] = routing.get(mode, 0) + 1

    # Extract plan
    plan = result.plan
    subqueries = [sq.question for sq in (plan.subqueries if plan else [])]

    # Extract cost and usage
    usage = getattr(result, "usage", {}) or {}
    cost = None
    for v in (usage.values() if hasattr(usage, "values") else []):
        if isinstance(v, dict) and "cost_usd" in v:
            cost = v["cost_usd"]
            break
    # Try flat cost field
    if cost is None:
        cost = getattr(result, "cost_usd", None)

    record = {
        "id": qid,
        "question": question,
        "type": qtype,
        "domain": domain,
        "elapsed_s": elapsed,
        "subqueries": subqueries,
        "datasets_retrieved": len(result.hits or []),
        "datasets_selected": len(selected),
        "routing": routing,
        "answer": result.answer,
        "cost_usd": cost,
    }

    print(f"  elapsed: {elapsed}s  selected: {len(selected)}  routing: {routing}")
    print(f"  subqueries: {len(subqueries)}")
    print(f"\n  ANSWER:\n{result.answer[:800]}")

    return record


def main():
    results = []
    out_path = Path(__file__).parent / "eval_results.json"

    for qid, question, qtype, domain in QUESTIONS:
        record = run_query(qid, question, qtype, domain)
        if record:
            results.append(record)
            # Save after each query so we don't lose results if something crashes
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            print(f"\n  Saved to {out_path}")

    print(f"\n\n{'='*70}")
    print(f"EVALUATION COMPLETE: {len(results)}/{len(QUESTIONS)} queries")
    print(f"Results: {out_path}")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n{'ID':<4} {'Domain':<14} {'Type':<12} {'Sel':>4} {'RAG':>4} {'Tech':>5} {'Time':>6}  Answer snippet")
    print("-" * 80)
    for r in results:
        rag = r["routing"].get("rag", 0)
        tech = r["routing"].get("technical", 0)
        ans = (r["answer"] or "")[:50].replace("\n", " ")
        print(f"{r['id']:<4} {r['domain']:<14} {r['type']:<12} {r['datasets_selected']:>4} {rag:>4} {tech:>5} {r['elapsed_s']:>5.1f}s  {ans}")


if __name__ == "__main__":
    main()
