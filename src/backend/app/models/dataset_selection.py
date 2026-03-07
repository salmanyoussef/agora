from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel

ExecutionMode = Literal["rag", "technical"]


class SelectedDataset(BaseModel):
    """A selected dataset with its execution mode and per-dataset reasoning."""

    dataset_id: str
    execution_mode: ExecutionMode = "rag"
    reasoning: str = ""


class DatasetSelection(BaseModel):

    selected_datasets: List[SelectedDataset]