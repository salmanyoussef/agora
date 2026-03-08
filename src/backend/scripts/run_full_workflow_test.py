"""
Run the exact full agent workflow (planner → retrieval → dataset selector → general/technical → synthesis)
from a single user question. Ideal for end-to-end testing without starting the API.

RAG vs technical per dataset is controlled by USE_ONLY_GENERAL_AGENT in app.agents.orchestrator.

Usage (from src/backend with venv active):
  python -m scripts.run_full_workflow_test "Quels jeux de données sur la qualité de l'air à Paris ?"
  python -m scripts.run_full_workflow_test "Transports en France" --k 5
  LOG_LEVEL=DEBUG python -m scripts.run_full_workflow_test "Your question"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.orchestrator import AgentOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full agent workflow (planner → selector → RAG/technical → synthesis) from one question"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Quels jeux de données ouverts existent sur les transports en France ?",
        help="User question (default: example French open data question)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of datasets to retrieve per subquery (default 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full response as JSON at the end",
    )
    args = parser.parse_args()

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "Running full workflow: question=%r k=%d",
        args.question[:80] + ("..." if len(args.question) > 80 else ""),
        args.k,
    )

    orchestrator = AgentOrchestrator()
    result = orchestrator.run(args.question, k=args.k)

    # Pretty-print the result (mimics what the UI would show)
    print("\n" + "=" * 70)
    print("PLAN")
    print("=" * 70)
    print("Intent:", result.plan.intent)
    for i, sq in enumerate(result.plan.subqueries, 1):
        print(f"  {i}. {sq.question}")
        print(f"     Purpose: {sq.purpose}")
    print()

    print("=" * 70)
    print("PROGRESS (datasets checked)")
    print("=" * 70)
    for msg in result.user_messages:
        print(" ", msg)
    print()

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    print("=" * 70)

    if args.json:
        print("\nFull response (JSON):")
        # Pydantic model_dump(); evidence blocks are ExecutionResult with mode, subquery, evidence
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
