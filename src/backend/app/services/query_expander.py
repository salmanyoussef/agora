from __future__ import annotations

from typing import List

def expand_queries(original_question: str) -> List[str]:
    """Return only the current subquery as retrieval query."""
    original_question = original_question.strip()
    return [original_question] if original_question else []