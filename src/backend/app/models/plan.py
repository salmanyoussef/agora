from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field

ExecutionMode = Literal["rag", "technical", "unknown"]


class SubQuery(BaseModel):
    question: str
    purpose: str
    execution_mode: ExecutionMode = "unknown"


class QueryPlan(BaseModel):
    intent: str
    subqueries: List[SubQuery] = Field(default_factory=list)