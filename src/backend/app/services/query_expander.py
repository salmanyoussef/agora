from __future__ import annotations

from typing import List

from app.models.plan import QueryPlan


def expand_queries(plan: QueryPlan, original_question: str) -> List[str]:
    queries: List[str] = []

    original_question = original_question.strip()
    if original_question:
        queries.append(original_question)

    for sq in plan.subqueries:
        q = sq.question.strip()
        if q and q not in queries:
            queries.append(q)

    return queries