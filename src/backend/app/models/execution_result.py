from __future__ import annotations

from typing import Any, Dict, Literal
from pydantic import BaseModel


class ExecutionResult(BaseModel):

    mode: Literal["rag", "technical"]
    subquery: str
    evidence: str
    lm_usage: Dict[str, Any] | None = None  # DSPy get_lm_usage() for pipeline cost tracking
    embed_usage: Dict[str, int] | None = None  # Embedding API usage (prompt_tokens, total_tokens) for pipeline cost
    # Technical / RLM only: rows loaded into the REPL `records` variable
    repl_rows: int | None = None
    resource_total_rows: int | None = None  # estimated rows in file, if known
    resource_title: str | None = None
    max_repl_rows: int | None = None  # cap (e.g. 50_000) for UI