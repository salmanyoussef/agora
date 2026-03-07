from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


class ExecutionResult(BaseModel):

    mode: Literal["rag", "technical"]
    subquery: str
    evidence: str