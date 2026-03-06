from __future__ import annotations

from typing import List
from pydantic import BaseModel


class DatasetSelection(BaseModel):

    selected_dataset_ids: List[str]
    reasoning: str