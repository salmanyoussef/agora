"""
Run only the General (RAG) agent for a given subquery and dataset(s).

Use this to test download + extraction + RAG without running planner, selector,
or synthesis.

Usage (from src/backend with venv active):
  python -m scripts.run_general_agent_test "What does this dataset contain?" --dataset-id <data.gouv dataset id>
  python -m scripts.run_general_agent_test "Résumé des données" --dataset-id abc-123 --dataset-id def-456
  LOG_LEVEL=DEBUG python -m scripts.run_general_agent_test "..." --dataset-id xyz

Example dataset IDs (French open data): use any ID from https://www.data.gouv.fr/fr/datasets/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.general import GeneralAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only the General (RAG) agent with a subquery and dataset ID(s)"
    )
    parser.add_argument(
        "subquery",
        help="Question/subquery to answer from the dataset(s) (e.g. 'What does this dataset contain?')",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        required=True,
        help="data.gouv.fr dataset ID (repeat for multiple). Example: 5e6e3a0c8b4c4a0018b4567",
    )
    parser.add_argument(
        "--reasoning",
        default="",
        help="Optional: why this dataset was chosen (passed as focus to the RAG prompt)",
    )
    args = parser.parse_args()

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Running General agent only: subquery=%r dataset_ids=%s", args.subquery, args.dataset_ids)

    # Build minimal hit dicts (general agent only needs dataset_id / id; title is for logs)
    datasets = [
        {"dataset_id": did, "id": did, "title": f"Dataset {did}"}
        for did in args.dataset_ids
    ]

    agent = GeneralAgent()
    result = agent.run(
        subquery=args.subquery,
        datasets=datasets,
        dataset_reasoning=args.reasoning,
    )

    print("\n" + "=" * 60)
    print("GENERAL AGENT RESULT")
    print("=" * 60)
    print("mode:", result.mode)
    print("subquery:", result.subquery)
    print("\nevidence (RAG answer):")
    print("-" * 40)
    print(result.evidence)
    print("=" * 60)

    if os.environ.get("DUMP_JSON"):
        print("\nFull result (JSON):")
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
