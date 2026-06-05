"""SSE payloads for per-dataset evidence shown before synthesis."""

from __future__ import annotations

from app.models.execution_result import ExecutionResult


def build_evidence_sse_payload(
    result: ExecutionResult,
    *,
    dataset_title: str,
    dataset_url: str = "",
    dataset_organization: str = "",
) -> dict:
    """Build event dict for one agent evidence block (RAG or technical)."""
    payload: dict = {
        "event": "evidence",
        "mode": result.mode,
        "subquery": result.subquery,
        "dataset_title": dataset_title,
        "dataset_url": dataset_url or "",
        "dataset_organization": dataset_organization or "",
        "evidence": result.evidence or "",
    }
    if result.resource_title:
        payload["resource_title"] = result.resource_title
    if result.repl_rows is not None:
        payload["repl_rows"] = result.repl_rows
        payload["resource_total_rows"] = result.resource_total_rows
        payload["max_repl_rows"] = result.max_repl_rows
    return payload
