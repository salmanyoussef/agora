import json
import logging
import time
from datetime import datetime, timezone
import dspy

from app.models.plan import QueryPlan, SubQuery
from app.services.dspy_setup import configure_dspy, log_last_lm_call, log_lm_usage

logger = logging.getLogger(__name__)


class PlanSignature(dspy.Signature):

    """
    You are planning retrieval queries for the French open government data portal.

    Goal:
    - Determine which datasets are relevant for answering the user question.
    - Retrieval is done with a hybrid search (vector similarity + BM25).
    - Your job is to produce one or more dataset-search-oriented subqueries.

    Query planning rules:
    - If the user question is broad and may require multiple dataset domains,
      break it into focused subqueries that improve hybrid retrieval coverage.
      Example broad question: "What is the feasibility of building a hotel in Lyon?"
      Useful subquery themes could include tourism, infrastructure, roads, etc.
    - If the user question is narrow/specific enough, do not decompose it.
      Return exactly one subquery: either the original user question or a slight
      reformulation optimized for hybrid dataset search.
    - Keep subqueries concise, retrieval-ready, and aligned with data discovery.
    """

    question = dspy.InputField()
    current_timestamp = dspy.InputField(
        desc="Current UTC timestamp in ISO-8601 format. It's there for your reference."
    )

    output_json = dspy.OutputField(
        desc="""
Return JSON with format:

{
 "intent": "...",
 "subqueries":[
   {
    "question":"A focused retrieval query for finding relevant French open data datasets (hybrid vector + BM25)",
    "purpose":"Why this subquery is needed and what dataset angle/domain it covers"
   }
 ]
}

Important:
- Return at least one subquery.
- If decomposition is not needed, return exactly one subquery.
"""
    )


class PlannerAgent:

    def __init__(self):
        configure_dspy()
        self.module = dspy.ChainOfThought(PlanSignature)

    def run(self, question: str) -> QueryPlan:
        logger.info("PlannerAgent.run started: question_len=%d", len(question))
        current_timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("PlannerAgent timestamp context: %s", current_timestamp)
        logger.info("PlannerAgent LLM call started")
        started_at = time.perf_counter()
        result = self.module(question=question, current_timestamp=current_timestamp)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info("PlannerAgent LLM call completed in %.1f ms", elapsed_ms)
        usage = None
        try:
            usage = result.get_lm_usage()
        except Exception:
            pass
        log_last_lm_call(caller="planner")
        logger.info("PlannerAgent DSPy response trace (last call):")
        dspy.inspect_history(n=1)
        log_lm_usage("planner", usage)

        try:
            data = json.loads(result.output_json)
        except Exception:
            logger.warning("Planner JSON parse failed")
            return QueryPlan(intent=question, subqueries=[], lm_usage=usage)

        subs = []

        for s in data.get("subqueries", []):

            subs.append(
                SubQuery(
                    question=s.get("question", ""),
                    purpose=s.get("purpose", ""),
                )
            )

        plan = QueryPlan(intent=data.get("intent", question), subqueries=subs, lm_usage=usage)

        logger.info("Planner intent: %s", plan.intent)
        for s in plan.subqueries:
            logger.info("Subquery created | %s", s.question)
        logger.info("Planner completed")

        return plan