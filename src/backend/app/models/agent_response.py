from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel, Field

from app.models.plan import QueryPlan
from app.models.execution_result import ExecutionResult


class AgentResponse(BaseModel):

    question: str = ""
    answer: str
    plan: QueryPlan
    evidence: List[ExecutionResult]
    hits: List[Dict[str, Any]]
    selected_datasets: List[Dict[str, Any]] = Field(default_factory=list)
    user_messages: List[str] = Field(default_factory=list)
    lm_usage_grand_total: Dict[str, Any] | None = None  # Merged chat token usage for entire pipeline (cost estimation)
    embed_usage_grand_total: Dict[str, int] | None = None  # Merged embedding usage (prompt_tokens, total_tokens) for pipeline
    pipeline_cost_usd: str | None = None  # LLM + embeddings total (decimal string, unrounded)