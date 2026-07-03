"""
Evaluation script for AGORA with LLaMA-3.3-70B-Instruct backend.

Runs the same 8 benchmark questions as run_evaluation.py but swaps the DSPy
LM to the LLaMA endpoint before starting, so results are directly comparable
to the GPT-5-mini baseline in eval_results.json.

Output: agora/evaluation/eval_results_llama.json
"""
import json
import sys
import time
import os
from pathlib import Path

# Add project root to path before any app imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Override DSPy config with LLaMA before the orchestrator (or any agent) is
# imported — configure_dspy() is called lazily on first agent use, so we must
# call configure_dspy_for_model() first to ensure our setting wins.
from app.settings import settings
from app.services.dspy_setup import configure_dspy_for_model

configure_dspy_for_model(
    model_key=settings.azure_llama_deployment,
    api_base=settings.azure_llama_endpoint,
    api_key=settings.azure_openai_api_key,
)

# Now safe to import orchestrator (it will use the already-configured LM).
from app.agents.orchestrator import AgentOrchestrator

QUESTIONS = [
    ("Q1", "Quels jeux de données sont disponibles sur la qualité de l'air en France ?",
     "discovery", "environment"),
    ("Q2", "Quelles données sur les accidents de la route en France sont disponibles sur data.gouv.fr ?",
     "discovery", "transport"),
    ("Q3", "Combien de stations de vélos en libre-service existe-t-il à Paris ?",
     "analytical", "transport"),
    ("Q4", "Quels sont les jeux de données disponibles sur le budget des communes françaises ?",
     "discovery", "finance"),
    ("Q5", "Quelles sont les communes françaises ayant une population supérieure à 100 000 habitants ?",
     "analytical", "demographics"),
    ("Q6", "Quels jeux de données concernent les établissements scolaires en France ?",
     "discovery", "education"),
    ("Q7", "Quels sont les jeux de données sur la production d'énergies renouvelables en France ?",
     "discovery", "energy"),
    ("Q8", "Combien de bornes de recharge pour véhicules électriques sont référencées en France ?",
     "analytical", "energy"),
]

OUT_PATH = Path(__file__).parent.parent.parent.parent / "evaluation" / "eval_results_llama.json"


def run_query(qid, question, qtype, domain, k=3):
    print(f"\n{'='*70}")
    print(f"[{qid}] {question}")
    print(f"  type={qtype}  domain={domain}  model={settings.azure_llama_deployment}")
    print(f"{'='*70}")

    orchestrator = AgentOrchestrator()
    t0 = time.time()
    try:
        result = orchestrator.run(question, k=k)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    elapsed = round(time.time() - t0, 1)

    selected = result.selected_datasets or []
    routing_counts: dict[str, int] = {}
    dataset_decisions = []
    for ds in selected:
        mode = ds.get("execution_mode", "unknown") if isinstance(ds, dict) else getattr(ds, "execution_mode", "unknown")
        routing_counts[mode] = routing_counts.get(mode, 0) + 1
        if isinstance(ds, dict):
            ds_id = ds.get("dataset_id")
            reasoning = ds.get("reasoning")
        else:
            ds_id = getattr(ds, "dataset_id", None)
            reasoning = getattr(ds, "reasoning", None)
        dataset_decisions.append({
            "dataset_id": ds_id,
            "execution_mode": mode,
            "reasoning": reasoning,
        })

    plan = result.plan
    subqueries = [sq.question for sq in (plan.subqueries if plan else [])]

    usage = getattr(result, "usage", {}) or {}
    cost_usd = getattr(result, "cost_usd", None)
    if cost_usd is None:
        for v in (usage.values() if hasattr(usage, "values") else []):
            if isinstance(v, dict) and "cost_usd" in v:
                cost_usd = v["cost_usd"]
                break

    record = {
        "id": qid,
        "question": question,
        "type": qtype,
        "domain": domain,
        "model": settings.azure_llama_deployment,
        "elapsed_s": elapsed,
        "subqueries": subqueries,
        "n_subqueries": len(subqueries),
        "datasets_retrieved": len(result.hits or []),
        "datasets_selected": len(selected),
        "routing_counts": routing_counts,
        "dataset_decisions": dataset_decisions,
        "answer": result.answer,
        "cost_usd": cost_usd,
    }

    rag = routing_counts.get("rag", 0)
    tech = routing_counts.get("technical", 0)
    print(f"  elapsed: {elapsed}s  selected: {len(selected)}  rag/tech: {rag}/{tech}")
    print(f"  subqueries: {len(subqueries)}")
    print(f"\n  ANSWER (first 600 chars):\n{(result.answer or '')[:600]}")
    return record


def main():
    print(f"LLaMA evaluation — model: {settings.azure_llama_deployment}")
    print(f"Endpoint: {settings.azure_llama_endpoint}")
    print(f"Output: {OUT_PATH}\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for qid, question, qtype, domain in QUESTIONS:
        record = run_query(qid, question, qtype, domain)
        if record:
            results.append(record)
            OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            print(f"  Saved ({len(results)}/{len(QUESTIONS)}) -> {OUT_PATH}")

    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(results)}/{len(QUESTIONS)} queries")
    print(f"{'='*70}")
    print(f"\n{'ID':<4} {'Domain':<14} {'Type':<12} {'Sel':>4} {'RAG':>4} {'Tech':>5} {'Time':>6}  Answer snippet")
    print("-" * 80)
    for r in results:
        rag = r["routing_counts"].get("rag", 0)
        tech = r["routing_counts"].get("technical", 0)
        ans = (r["answer"] or "")[:50].replace("\n", " ")
        print(f"{r['id']:<4} {r['domain']:<14} {r['type']:<12} {r['datasets_selected']:>4} {rag:>4} {tech:>5} {r['elapsed_s']:>5.1f}s  {ans}")


if __name__ == "__main__":
    main()
