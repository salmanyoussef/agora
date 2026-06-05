"""
Resource Selector Agent: after a dataset is chosen for technical analysis, pick the
single resource best suited for REPL exploration (tabular CSV/JSON/XLSX preferred).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import dspy

from app.clients.data_gouv import extract_resource_urls
from app.models.resource_selection import ResourceSelection
from app.services.dspy_setup import configure_dspy, log_last_lm_call, log_lm_usage

logger = logging.getLogger(__name__)

MAX_SELECTOR_ATTEMPTS = 3
MAX_RESOURCE_DESCRIPTION_CHARS = 400

# Prefer these for computation in the technical REPL
_TABULAR_FORMAT_HINTS = frozenset(
    {"csv", "tsv", "xlsx", "xls", "json", "jsonl", "ndjson", "geojson"}
)


def _truncate(s: str, max_chars: int) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _format_bytes(size: Any) -> str:
    if size is None:
        return "unknown"
    try:
        n = int(size)
    except (TypeError, ValueError):
        return "unknown"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def build_resource_summaries(url_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Metadata-only summaries for the selector (no download)."""
    summaries: List[Dict[str, Any]] = []
    for idx, item in enumerate(url_items):
        res = item.get("resource") or {}
        rid = res.get("id")
        fmt = (res.get("format") or "").strip().lower()
        mime = (res.get("mime") or "").strip().lower()
        summaries.append(
            {
                "index": idx,
                "resource_id": str(rid) if rid is not None else None,
                "title": _truncate(res.get("title") or "", 200),
                "format": fmt,
                "mime": mime.split(";")[0] if mime else "",
                "size": _format_bytes(res.get("size")),
                "url": _truncate(item.get("url") or "", 180),
                "description": _truncate(res.get("description") or "", MAX_RESOURCE_DESCRIPTION_CHARS),
            }
        )
    return summaries


def find_url_item_by_resource_id(
    url_items: List[Dict[str, Any]],
    resource_id: str,
) -> Optional[Dict[str, Any]]:
    target = str(resource_id).strip()
    for item in url_items:
        res = item.get("resource") or {}
        if res.get("id") is not None and str(res.get("id")) == target:
            return item
    return None


def _heuristic_pick(url_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prefer first tabular-looking resource; else first item with a URL."""
    for item in url_items:
        res = item.get("resource") or {}
        fmt = (res.get("format") or "").strip().lower()
        mime = (res.get("mime") or "").strip().lower()
        if fmt in _TABULAR_FORMAT_HINTS:
            return item
        if any(h in mime for h in ("csv", "json", "spreadsheet", "excel")):
            return item
    for item in url_items:
        if item.get("url"):
            return item
    return url_items[0] if url_items else None


class ResourceSelectorSignature(dspy.Signature):
    """
    A dataset has been selected for technical (computation) analysis. Choose exactly
    ONE resource from the list to pass to a Python REPL agent.

    Prefer resources that are machine-readable and tabular:
    - Best: CSV, TSV, XLSX, JSON array of records, JSONL
    - Avoid when alternatives exist: PDF, HTML pages, ZIP archives without clear CSV inside,
      API documentation links, images, shapefiles without tabular export

    The REPL agent will load up to 50,000 rows from your choice and run Python (pandas-friendly
    list of dicts). Pick the resource most likely to answer the subquery with computations
    (filter, aggregate, sort, time series).
    """

    user_question = dspy.InputField()
    subquery = dspy.InputField()
    dataset_title = dspy.InputField(desc="Title of the parent dataset")
    dataset_description = dspy.InputField(desc="Short description of the parent dataset")
    dataset_selector_reasoning = dspy.InputField(
        desc="Why the dataset selector chose this dataset for technical mode"
    )
    resources = dspy.InputField(
        desc="JSON list of resource metadata (resource_id, title, format, mime, size, url, description)"
    )

    output_json = dspy.OutputField(
        desc="""
Return JSON:

{
  "resource_id": "<id from resources list>",
  "reasoning": "Why this resource is best for REPL/tabular analysis"
}

- resource_id: must match one of the resource_id values provided (or null if list empty).
- reasoning: short explanation focused on format and relevance to the subquery.
"""
    )


class ResourceSelectorAgent:
    def __init__(self) -> None:
        configure_dspy()
        self.module = dspy.ChainOfThought(ResourceSelectorSignature)

    def _parse_output(self, raw: str) -> tuple[dict, bool]:
        try:
            parsed = json.loads(raw or "{}")
        except Exception as e:
            logger.warning("ResourceSelectorAgent JSON parse failed: %s", e)
            return {}, False
        if not isinstance(parsed, dict):
            return {}, False
        rid = parsed.get("resource_id")
        if rid is None:
            return parsed, False
        return parsed, True

    def run(
        self,
        question: str,
        subquery: str,
        dataset: Dict[str, Any],
        dataset_selector_reasoning: str = "",
    ) -> ResourceSelection:
        url_items = extract_resource_urls(dataset)
        summaries = build_resource_summaries(url_items)

        if not url_items:
            logger.warning("ResourceSelectorAgent: dataset has no resources")
            return ResourceSelection(resource_id="", reasoning="No resources on dataset.")

        title = (dataset.get("title") or dataset.get("name") or "Unknown").strip()
        desc = _truncate(dataset.get("description") or "", 600)

        logger.info(
            "ResourceSelectorAgent.run: dataset=%r resources_count=%d",
            title[:80],
            len(url_items),
        )

        if len(url_items) == 1:
            only = url_items[0]
            rid = str((only.get("resource") or {}).get("id") or "")
            logger.info("ResourceSelectorAgent: single resource, skipping LLM (id=%s)", rid)
            return ResourceSelection(
                resource_id=rid,
                reasoning="Only one resource available on this dataset.",
            )

        resources_json = json.dumps(summaries, ensure_ascii=False)
        parsed: dict = {}
        usage = None
        last_result = None

        for attempt in range(1, MAX_SELECTOR_ATTEMPTS + 1):
            logger.info(
                "ResourceSelectorAgent LLM call started (attempt %d/%d)",
                attempt,
                MAX_SELECTOR_ATTEMPTS,
            )
            started_at = time.perf_counter()
            try:
                last_result = self.module(
                    user_question=question,
                    subquery=subquery,
                    dataset_title=title,
                    dataset_description=desc,
                    dataset_selector_reasoning=dataset_selector_reasoning or "(none)",
                    resources=resources_json,
                )
            except Exception as e:
                logger.warning("ResourceSelectorAgent LLM call failed: %s", e)
                if attempt == MAX_SELECTOR_ATTEMPTS:
                    break
                continue
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info("ResourceSelectorAgent LLM call completed in %.1f ms", elapsed_ms)
            try:
                usage = last_result.get_lm_usage()
            except Exception:
                pass
            log_last_lm_call(caller="resource_selector")
            log_lm_usage("resource_selector", usage)

            parsed, valid = self._parse_output(last_result.output_json or "{}")
            if valid:
                break
            if attempt < MAX_SELECTOR_ATTEMPTS:
                logger.warning(
                    "ResourceSelectorAgent invalid output, retrying (%d/%d)",
                    attempt,
                    MAX_SELECTOR_ATTEMPTS,
                )

        resource_id = str(parsed.get("resource_id") or "").strip()
        reasoning = (parsed.get("reasoning") or "").strip()
        item = find_url_item_by_resource_id(url_items, resource_id) if resource_id else None
        if item is None:
            item = _heuristic_pick(url_items)
            fallback_rid = str((item.get("resource") or {}).get("id") or "") if item else ""
            logger.warning(
                "ResourceSelectorAgent: LLM id %r not found; heuristic pick id=%s",
                resource_id,
                fallback_rid,
            )
            resource_id = fallback_rid
            if not reasoning:
                reasoning = "Heuristic fallback: first tabular-looking resource."

        res_title = ((item or {}).get("resource") or {}).get("title") or "?"
        logger.info(
            "ResourceSelector selected resource_id=%s title=%s | %s",
            resource_id,
            res_title,
            reasoning[:120],
        )

        selection = ResourceSelection(resource_id=resource_id, reasoning=reasoning)
        if usage is not None:
            selection = selection.model_copy(update={"lm_usage": usage})
        return selection
