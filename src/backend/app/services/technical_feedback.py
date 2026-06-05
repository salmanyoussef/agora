"""User-facing messages and SSE payloads for technical / RLM row counts."""

from __future__ import annotations

from app.models.execution_result import ExecutionResult


def _fmt(n: int) -> str:
    return f"{n:,}"


def format_rows_loaded_progress(
    *,
    resource_title: str,
    repl_rows: int,
    resource_total_rows: int | None,
    max_repl_rows: int,
) -> str:
    """Exact row counts after parse, before RLM (stream progress)."""
    res = resource_title.strip() or "resource"
    if repl_rows == 0:
        return (
            f"Parsed «{res}»: 0 tabular rows loaded into REPL "
            f"(non-tabular or empty; analysis may use extracted text only)."
        )
    msg = f"Parsed «{res}»: {_fmt(repl_rows)} row"
    msg += "s" if repl_rows != 1 else ""
    msg += " loaded into REPL for analysis"
    if resource_total_rows is not None:
        msg += f" (~{_fmt(resource_total_rows)} rows in file"
        if resource_total_rows > repl_rows:
            msg += f"; showing {_fmt(repl_rows)} due to {_fmt(max_repl_rows)} cap"
        msg += ")"
    elif repl_rows >= max_repl_rows:
        msg += (
            f" (at {_fmt(max_repl_rows)}-row cap; "
            "total rows in file could not be estimated)"
        )
    msg += "."
    return msg


def format_technical_repl_message(
    *,
    dataset_title: str,
    resource_title: str,
    repl_rows: int,
    resource_total_rows: int | None,
    max_repl_rows: int,
) -> str:
    """Human-readable progress line for stream UI."""
    ds = dataset_title.strip() or "Dataset"
    res = resource_title.strip() or "resource"

    if repl_rows == 0:
        return (
            f"Technical REPL · «{ds}» · «{res}»: no tabular rows loaded "
            f"(analysis may use extracted text only)."
        )

    exploring = f"Technical REPL · «{ds}» · «{res}»: exploring {_fmt(repl_rows)} row"
    exploring += "s" if repl_rows != 1 else ""
    exploring += " in Python"

    if resource_total_rows is not None:
        exploring += f" (resource has ~{_fmt(resource_total_rows)} rows total"
        if resource_total_rows > repl_rows:
            exploring += f"; loaded {_fmt(repl_rows)} up to {_fmt(max_repl_rows)} cap"
        exploring += ")"
    elif repl_rows >= max_repl_rows:
        exploring += f" (at {_fmt(max_repl_rows)}-row analysis cap; total size in file not estimated)"

    exploring += "."
    return exploring


def is_rows_truncated(
    repl_rows: int,
    resource_total_rows: int | None,
    max_repl_rows: int,
) -> bool:
    if resource_total_rows is not None and resource_total_rows > repl_rows:
        return True
    if resource_total_rows is None and repl_rows >= max_repl_rows:
        return True
    return False


def build_technical_repl_sse_payload(
    result: ExecutionResult,
    dataset_title: str,
) -> dict | None:
    if result.mode != "technical":
        return None
    max_rows = result.max_repl_rows or 50_000
    resource_title = result.resource_title or "resource"
    repl_rows = result.repl_rows if result.repl_rows is not None else 0
    message = format_technical_repl_message(
        dataset_title=dataset_title,
        resource_title=resource_title,
        repl_rows=repl_rows,
        resource_total_rows=result.resource_total_rows,
        max_repl_rows=max_rows,
    )
    return {
        "event": "technical_repl",
        "message": message,
        "dataset_title": dataset_title,
        "resource_title": resource_title,
        "repl_rows": repl_rows,
        "resource_total_rows": result.resource_total_rows,
        "max_repl_rows": max_rows,
        "truncated": is_rows_truncated(
            repl_rows,
            result.resource_total_rows,
            max_rows,
        ),
    }
