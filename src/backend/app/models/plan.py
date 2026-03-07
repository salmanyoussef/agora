from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    question: str
    purpose: str


class QueryPlan(BaseModel):
    intent: str
    subqueries: List[SubQuery] = Field(default_factory=list)