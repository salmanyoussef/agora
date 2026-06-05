"""
Technical Agent: structured-data pipeline for computation-oriented subqueries.

Flow: resource selector (one resource) → download → parse up to 50k rows → RLM explores
`records` in a Python REPL with `resource_context` metadata.

Uses DSPy's RLM (Recursive Language Model). Falls back to dspy.Predict if REPL unavailable.
"""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any, Callable, Dict, Iterator, List, Union

import dspy

from app.agents.resource_selector import (
    ResourceSelectorAgent,
    find_url_item_by_resource_id,
)
from app.clients.data_gouv import DataGouvDatasetsClient, extract_resource_urls
from app.models.execution_result import ExecutionResult
from app.models.resource_selection import ResourceSelection
from app.services.dspy_setup import (
    configure_dspy,
    log_last_lm_call,
    log_lm_usage,
    merge_lm_usage,
)
from app.services.technical_feedback import format_rows_loaded_progress
from app.services.structured_data import (
    ParsedData,
    build_resource_context,
    detect_structure,
    parse_into_records,
)
from app.services.text_extraction import download_file, extract_text_from_file

logger = logging.getLogger(__name__)

# One resource per technical run (chosen by ResourceSelectorAgent)
MAX_ROWS_REPL = 50_000


def _resource_display_title(resource: Dict[str, Any] | None, url: str) -> str:
    if resource:
        title = (resource.get("title") or "").strip()
        if title:
            return title
    path_part = (url or "").split("?")[0].rstrip("/").split("/")[-1]
    return path_part[:120] if path_part else "resource"


def _format_resource_size_hint(resource: Dict[str, Any] | None) -> str:
    if not resource:
        return ""
    size = resource.get("size")
    if size is None:
        return ""
    try:
        n = int(size)
    except (TypeError, ValueError):
        return ""
    if n < 1024:
        return f" ({n} B)"
    if n < 1024 * 1024:
        return f" ({n / 1024:.1f} KB)"
    return f" ({n / (1024 * 1024):.1f} MB)"


def _technical_meta(
    resource: Dict[str, Any] | None,
    url: str,
    parsed: ParsedData | None,
    records: List[Dict[str, Any]],
) -> dict:
    repl_rows = len(records)
    total = parsed.total_rows_in_file if parsed else None
    return {
        "repl_rows": repl_rows,
        "resource_total_rows": total,
        "resource_title": _resource_display_title(resource, url),
        "max_repl_rows": MAX_ROWS_REPL,
    }


MAX_CHARS_UNSTRUCTURED = 50_000

# DSPy RLM knobs
RLM_MAX_ITERATIONS = 15
RLM_MAX_LLM_CALLS = 30
RLM_VERBOSE = True


def _explore_with_rlm(
    question: str,
    records: List[Dict[str, Any]],
    resource_context: str,
    focus: str = "",
) -> tuple[str, dict | None]:
    configure_dspy()
    focus_text = (focus or "(No specific focus provided.)").strip()
    logger.info(
        "TechnicalAgent RLM call: question_len=%d records=%d context_len=%d focus_len=%d",
        len(question),
        len(records),
        len(resource_context),
        len(focus_text),
    )
    started_at = time.perf_counter()
    use_rlm = True
    pred = None

    try:
        rlm = dspy.RLM(
            ExploreTechnicalContext,
            max_iterations=RLM_MAX_ITERATIONS,
            max_llm_calls=RLM_MAX_LLM_CALLS,
            verbose=RLM_VERBOSE,
        )
        pred = rlm(
            records=records,
            resource_context=resource_context,
            question=question,
            focus=focus_text,
        )
    except Exception as e:
        use_rlm = False
        logger.warning(
            "TechnicalAgent: RLM REPL unavailable (%s), falling back to Predict. "
            "For full RLM, run: uv run python -m app.scripts.setup_repl",
            e,
        )
        pred = dspy.Predict(ExploreTechnicalContext)(
            records=records,
            resource_context=resource_context,
            question=question,
            focus=focus_text,
        )

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "TechnicalAgent %s call completed in %.1f ms",
        "RLM" if use_rlm else "Predict (fallback)",
        elapsed_ms,
    )
    usage = None
    try:
        usage = pred.get_lm_usage()
    except Exception as e:
        logger.debug("Technical get_lm_usage not available: %s", e)
    log_last_lm_call(caller="technical_rlm" if use_rlm else "technical_predict_fallback")
    n_history = min(RLM_MAX_ITERATIONS + 2, 25) if use_rlm else 1
    logger.info(
        "TechnicalAgent DSPy response trace (last %d call(s)):",
        n_history,
    )
    dspy.inspect_history(n=n_history)
    log_lm_usage("technical_rlm" if use_rlm else "technical_predict_fallback", usage)
    return pred.answer or "No answer could be produced from the technical context.", usage


class ExploreTechnicalContext(dspy.Signature):
    """
    Technical step over French open government data. You receive:

    - `records`: list of dicts (up to 50,000 rows) loaded from ONE dataset resource.
      Explore with Python in the REPL: len(records), filtering, aggregates, sorting.
      Prefer plain Python; try pandas only if available.
    - `resource_context`: metadata (dataset title, resource format/URL, column schema).

    Answer the subquery using only this data. Your output becomes evidence for synthesis.
    Same language as the question (typically French).
    """

    records = dspy.InputField(
        desc="List of row dicts for REPL (up to 50k). Primary data to analyze."
    )
    resource_context = dspy.InputField(
        desc="Dataset and resource metadata, schema summary, selector reasoning."
    )
    question = dspy.InputField(desc="The subquery to answer.")
    focus = dspy.InputField(
        desc="Why this dataset/resource was chosen for this subquery."
    )
    answer = dspy.OutputField(
        desc="Structured evidence block from analyzing `records` (same language as question)."
    )


class TechnicalAgent:
    def __init__(self) -> None:
        self.resource_selector = ResourceSelectorAgent()

    def run(
        self,
        subquery: str,
        hits: list[dict],
        dataset_reasoning: str = "",
        user_question: str = "",
        resource_selection: ResourceSelection | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Run technical analysis; optional on_progress for user-facing status lines."""
        result: ExecutionResult | None = None
        for event in self.iter_run(
            subquery,
            hits,
            dataset_reasoning=dataset_reasoning,
            user_question=user_question,
            resource_selection=resource_selection,
        ):
            if isinstance(event, str):
                if on_progress:
                    on_progress(event)
            else:
                result = event
        assert result is not None
        return result

    def iter_run(
        self,
        subquery: str,
        hits: list[dict],
        dataset_reasoning: str = "",
        user_question: str = "",
        resource_selection: ResourceSelection | None = None,
    ) -> Iterator[Union[str, ExecutionResult]]:
        """
        Yields progress messages (str) then a final ExecutionResult.
        Used by the streaming orchestrator for live download / parse feedback.
        """
        logger.info(
            "TechnicalAgent.run: subquery=%r hits_count=%d reasoning_len=%d",
            subquery[:200] + ("..." if len(subquery) > 200 else ""),
            len(hits),
            len(dataset_reasoning),
        )

        if not hits:
            yield ExecutionResult(
                mode="technical",
                subquery=subquery,
                evidence="Technical analysis requested but no dataset hit was provided.",
            )
            return

        hit = hits[0]
        dataset_id = hit.get("dataset_id") or hit.get("id")
        if not dataset_id:
            yield ExecutionResult(
                mode="technical",
                subquery=subquery,
                evidence="Technical analysis requested but dataset id is missing.",
            )
            return

        client = DataGouvDatasetsClient()
        usage_parts: List[dict | None] = []

        try:
            ds = client.get_dataset(dataset_id)
        except Exception as e:
            logger.warning("TechnicalAgent: failed to fetch dataset %s: %s", dataset_id, e)
            yield ExecutionResult(
                mode="technical",
                subquery=subquery,
                evidence=f"Technical analysis failed: could not fetch dataset {dataset_id}.",
            )
            return

        dataset_title = (ds.get("title") or ds.get("name") or "dataset").strip()
        url_items = extract_resource_urls(ds)

        question_for_selector = (user_question or subquery).strip()
        if resource_selection is None:
            if len(url_items) > 1:
                yield f"Selecting one resource for «{dataset_title}»…"
            resource_selection = self.resource_selector.run(
                question_for_selector,
                subquery,
                ds,
                dataset_selector_reasoning=dataset_reasoning,
            )
        if resource_selection.lm_usage:
            usage_parts.append(resource_selection.lm_usage)

        if not url_items:
            yield ExecutionResult(
                mode="technical",
                subquery=subquery,
                evidence="Technical analysis requested but dataset has no resources.",
                lm_usage=merge_lm_usage(usage_parts) or None,
            )
            return

        item = find_url_item_by_resource_id(url_items, resource_selection.resource_id)
        if item is None:
            item = url_items[0]
            logger.warning(
                "TechnicalAgent: resource_id %r not found; using first resource",
                resource_selection.resource_id,
            )

        url = item.get("url")
        resource = item.get("resource")
        if not url:
            yield ExecutionResult(
                mode="technical",
                subquery=subquery,
                evidence="Technical analysis failed: selected resource has no URL.",
                lm_usage=merge_lm_usage(usage_parts) or None,
            )
            return

        if resource is not None and resource.get("size") is None and resource.get("id"):
            try:
                resource = client.get_resource(dataset_id, str(resource["id"]))
            except Exception as e:
                logger.debug("TechnicalAgent: get_resource failed: %s", e)

        res_title = _resource_display_title(resource, url)
        yield f"Selected resource «{res_title}» for technical analysis."

        size_hint = _format_resource_size_hint(resource)
        yield f"Downloading «{res_title}»{size_hint}…"

        records: List[Dict[str, Any]] = []
        parsed: ParsedData | None = None
        unstructured_text: str | None = None

        with tempfile.TemporaryDirectory(prefix="agora_technical_") as out_dir:
            try:
                path = download_file(url, out_dir, resource=resource)
            except Exception as e:
                logger.warning("TechnicalAgent: download failed %s: %s", url[:80], e)
                yield ExecutionResult(
                    mode="technical",
                    subquery=subquery,
                    evidence=f"Technical analysis failed: could not download resource ({e!s}).",
                    lm_usage=merge_lm_usage(usage_parts) or None,
                    repl_rows=0,
                    resource_title=res_title,
                    max_repl_rows=MAX_ROWS_REPL,
                )
                return

            yield f"Download complete. Parsing «{res_title}»…"

            if detect_structure(path, resource) == "unsuitable":
                yield f"Extracting text from «{res_title}» (non-tabular file)…"
                try:
                    text = extract_text_from_file(path, max_rows=200, resource=resource)
                    if isinstance(text, list):
                        unstructured_text = "\n\n".join(
                            (t or "").strip() for t in text if (t or "").strip()
                        )
                    else:
                        unstructured_text = (text or "").strip()
                    if unstructured_text and len(unstructured_text) > MAX_CHARS_UNSTRUCTURED:
                        unstructured_text = (
                            unstructured_text[:MAX_CHARS_UNSTRUCTURED]
                            + "\n\n[Text truncated.]"
                        )
                except Exception as e:
                    logger.warning("TechnicalAgent: text extraction failed: %s", e)
            else:
                parsed = parse_into_records(
                    path,
                    resource=resource,
                    max_rows=MAX_ROWS_REPL,
                    resource_id=str(resource_selection.resource_id or dataset_id),
                    metadata="",
                )
                if parsed and parsed.records:
                    records = parsed.records
                    logger.info(
                        "TechnicalAgent: loaded %d rows, %d columns for REPL",
                        parsed.row_count,
                        len(parsed.columns),
                    )

            total_in_file = parsed.total_rows_in_file if parsed else None
            yield format_rows_loaded_progress(
                resource_title=res_title,
                repl_rows=len(records),
                resource_total_rows=total_in_file,
                max_repl_rows=MAX_ROWS_REPL,
            )

        resource_context = build_resource_context(
            dataset=ds,
            resource=resource,
            url=url,
            parsed=parsed,
            unstructured_text=unstructured_text,
            resource_selector_reasoning=resource_selection.reasoning,
            dataset_selector_reasoning=dataset_reasoning,
            max_rows_loaded=MAX_ROWS_REPL,
        )

        focus = dataset_reasoning
        if resource_selection.reasoning:
            focus = f"{focus}\nResource: {resource_selection.reasoning}".strip()

        if not records and not unstructured_text:
            evidence = (
                "Technical analysis could not load tabular records or extracted text "
                "from the selected resource. The resource may be an archive, API link, "
                "or unsupported format."
            )
            meta = _technical_meta(resource, url, parsed, records)
            yield ExecutionResult(
                mode="technical",
                subquery=subquery,
                evidence=evidence,
                lm_usage=merge_lm_usage(usage_parts) or None,
                **meta,
            )
            return

        yield "Exploring data in Python REPL…"

        evidence, tech_usage = _explore_with_rlm(
            subquery,
            records,
            resource_context,
            focus=focus,
        )
        if tech_usage:
            usage_parts.append(tech_usage)

        meta = _technical_meta(resource, url, parsed, records)
        yield ExecutionResult(
            mode="technical",
            subquery=subquery,
            evidence=evidence,
            lm_usage=merge_lm_usage(usage_parts) or None,
            **meta,
        )
