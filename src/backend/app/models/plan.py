from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    question: str
    purpose: str


class QueryPlan(BaseModel):
    intent: str
    subqueries: List[SubQuery] = Field(default_factory=list)
    lm_usage: Dict[str, Any] | None = None  # DSPy get_lm_usage() for pipeline cost tracking