from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel

from app.models.plan import QueryPlan
from app.models.execution_result import ExecutionResult


class AgentResponse(BaseModel):

    answer: str
    plan: QueryPlan
    evidence: List[ExecutionResult]
    hits: List[Dict[str, Any]]