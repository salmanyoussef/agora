from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel


class ResourceSelection(BaseModel):
    """One resource chosen for technical / RLM analysis on a dataset."""

    resource_id: str
    reasoning: str = ""
    lm_usage: Dict[str, Any] | None = None
