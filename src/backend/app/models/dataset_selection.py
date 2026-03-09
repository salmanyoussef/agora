from __future__ import annotations

from typing import Any, Dict, List, Literal
from pydantic import BaseModel

ExecutionMode = Literal["rag", "technical"]


class SelectedDataset(BaseModel):
    """A selected dataset with its execution mode and per-dataset reasoning."""

    dataset_id: str
    execution_mode: ExecutionMode = "rag"
    reasoning: str = ""


class DatasetSelection(BaseModel):

    selected_datasets: List[SelectedDataset]
    lm_usage: Dict[str, Any] | None = None  # DSPy get_lm_usage() for pipeline cost tracking