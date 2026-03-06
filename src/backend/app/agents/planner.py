import json
import logging
import dspy

from app.models.plan import QueryPlan, SubQuery
from app.services.dspy_setup import configure_dspy

logger = logging.getLogger(__name__)


class PlanSignature(dspy.Signature):

    """
    Break a user question into dataset retrieval subqueries.
    Decide if each subquery requires RAG or numeric calculations.
    """

    question = dspy.InputField()

    output_json = dspy.OutputField(
        desc="""
Return JSON with format:

{
 "intent": "...",
 "subqueries":[
   {
    "question":"...",
    "purpose":"...",
    "execution_mode":"rag | technical"
   }
 ]
}
"""
    )


class PlannerAgent:

    def __init__(self):
        configure_dspy()
        self.module = dspy.ChainOfThought(PlanSignature)

    def run(self, question: str) -> QueryPlan:

        result = self.module(question=question)

        try:
            data = json.loads(result.output_json)
        except Exception:
            logger.warning("Planner JSON parse failed")
            return QueryPlan(intent=question, subqueries=[])

        subs = []

        for s in data.get("subqueries", []):

            subs.append(
                SubQuery(
                    question=s.get("question", ""),
                    purpose=s.get("purpose", ""),
                    execution_mode=s.get("execution_mode", "unknown"),
                )
            )

        return QueryPlan(intent=data.get("intent", question), subqueries=subs)