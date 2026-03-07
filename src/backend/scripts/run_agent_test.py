"""
Run the agent pipeline from the CLI.

Usage:
  # From src/backend with venv active:
  LOG_LEVEL=INFO python -m scripts.run_agent_test "Your question here"
  python -m scripts.run_agent_test "Quels jeux de données sur les transports ?" --k 3
  python -m scripts.run_agent_test "Question..." --inspect-history 20
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure app is on path when run as python -m scripts.run_agent_test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.orchestrator import AgentOrchestrator
from app.services.dspy_setup import inspect_dspy_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AGORA agent pipeline")
    parser.add_argument(
        "question",
        nargs="?",
        default="Quels jeux de données ouverts existent sur les transports en France ?",
        help="User question to run through the pipeline",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of datasets to retrieve per subquery (default 3)",
    )
    parser.add_argument(
        "--inspect-history",
        type=int,
        default=0,
        help="Print last N DSPy calls via dspy.inspect_history (0 disables)",
    )
    args = parser.parse_args()

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Running agent test: question=%r k=%d", args.question, args.k)

    orchestrator = AgentOrchestrator()
    result = orchestrator.run(args.question, k=args.k)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result.model_dump() if hasattr(result, "model_dump") else result)
    print("=" * 60)

    if args.inspect_history > 0:
        print("\n" + "=" * 60)
        print(f"DSPY HISTORY (last {args.inspect_history})")
        print("=" * 60)
        inspect_dspy_history(n=args.inspect_history)


if __name__ == "__main__":
    main()
