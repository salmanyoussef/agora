"""
Technical Agent: structured-data pipeline for computation-oriented subqueries.

Flow: download → detect structure → parse into dataframe/records → build technical
context → RLM explores → final structured answer.

Unlike the General Agent (download → extract text → chunk → semantic retrieval → answer),
we do not chunk or embed; we normalize data into machine-usable records and let the
model reason over schema + preview.
"""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any, Dict, List

import dspy

from app.clients.data_gouv import DataGouvDatasetsClient, extract_resource_urls
from app.models.execution_result import ExecutionResult
from app.services.dspy_setup import configure_dspy, log_last_lm_call
from app.services.structured_data import (
    ParsedData,
    build_technical_context,
    detect_structure,
    parse_into_records,
)
from app.services.text_extraction import download_file, extract_text_from_file

logger = logging.getLogger(__name__)

# Limits for technical pipeline
MAX_DATASETS = 3
MAX_RESOURCES_PER_DATASET = 5
MAX_ROWS_PER_RESOURCE = 10_000
TECHNICAL_PREVIEW_ROWS = 50
MAX_CONTEXT_CHARS = 80_000  # cap context to avoid token overflow
# For unsuitable resources we use the same text extraction as the General Agent
MAX_ROWS_UNSTRUCTURED = 200  # same as GeneralAgent's MAX_ROWS_PER_RESOURCE for text
MAX_CHARS_UNSTRUCTURED_PER_BLOCK = 15_000  # cap each extracted text block


def _resource_metadata_str(
    dataset: Dict[str, Any],
    resource: Dict[str, Any] | None,
    url: str,
) -> str:
    """Short metadata line for the LLM: dataset title, org, format, url."""
    title = (dataset.get("title") or dataset.get("name") or "Unknown").strip()
    org_raw = dataset.get("organization")
    if isinstance(org_raw, dict):
        org = (org_raw.get("name") or org_raw.get("title") or "").strip()
    else:
        org = (org_raw or "").strip() if isinstance(org_raw, str) else ""
    parts = [f"Dataset: {title}"]
    if org:
        parts.append(f"Org: {org}")
    if resource:
        fmt = (resource.get("format") or "").strip()
        if fmt:
            parts.append(f"Format: {fmt}")
    parts.append(f"URL: {url[:100]}{'…' if len(url) > 100 else ''}")
    return " | ".join(parts)


class ExploreTechnicalContext(dspy.Signature):
    """
    You are the technical step in a multi-step pipeline over French open government data.

    Pipeline role: The user asked a question that requires computation or structured
    analysis. A planner split it into subqueries; a selector chose dataset(s) suited
    for technical analysis. Resources were downloaded and parsed into structured
    form (tables/records). You receive the schema and a preview of the data. Your job
    is to explore this technical context and answer the subquery: describe what the
    data contains, what computations or aggregations would answer the question, and
    give a clear, structured evidence block. Use only the provided data; no external
    knowledge. Answer in the same language as the question (typically French).

    What your output is used for: Your answer becomes one evidence block that is
    combined with evidence from other datasets and passed to a synthesis step. Write
    a self-contained, factual summary that the synthesis can cite—include concrete
    numbers or findings from the preview when possible, and say when the data is
    insufficient to fully answer the question.
    """

    technical_context = dspy.InputField(
        desc="Structured data: schema (columns, types) and preview rows for tabular/JSON resources. May also include 'Unstructured resources': extracted text from PDFs, DOCX, etc., using the same extraction as the general (RAG) agent—use both to answer."
    )
    question = dspy.InputField(desc="The subquery to answer using the technical context.")
    focus = dspy.InputField(
        desc="Why this dataset was chosen for this subquery (selector reasoning)."
    )
    answer = dspy.OutputField(
        desc="A clear, structured evidence block based only on the technical context (same language as the question). Include what the data shows and what computations would help. This will be merged with other evidence for the final user answer."
    )


def _explore_with_rlm(
    question: str,
    technical_context: str,
    focus: str = "",
) -> str:
    configure_dspy()
    focus_text = (focus or "(No specific focus provided.)").strip()
    logger.info(
        "TechnicalAgent RLM call: question_len=%d context_len=%d focus_len=%d",
        len(question),
        len(technical_context),
        len(focus_text),
    )
    started_at = time.perf_counter()
    pred = dspy.Predict(ExploreTechnicalContext)(
        technical_context=technical_context,
        question=question,
        focus=focus_text,
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info("TechnicalAgent RLM call completed in %.1f ms", elapsed_ms)
    try:
        usage = pred.get_lm_usage()
        if usage is not None:
            logger.debug("Technical Prediction.get_lm_usage(): %s", usage)
    except Exception as e:
        logger.debug("Technical get_lm_usage not available: %s", e)
    log_last_lm_call(caller="technical_explore")
    logger.info("TechnicalAgent DSPy response trace (last call):")
    dspy.inspect_history(n=1)
    return pred.answer or "No answer could be produced from the technical context."


class TechnicalAgent:
    def run(
        self,
        subquery: str,
        hits: list[dict],
        dataset_reasoning: str = "",
    ) -> ExecutionResult:
        """
        Run technical analysis: download resources → detect structure → parse into
        records → build technical context → RLM explores → return structured evidence.
        """
        logger.info(
            "TechnicalAgent.run: subquery=%r hits_count=%d reasoning_len=%d",
            subquery[:200] + ("..." if len(subquery) > 200 else ""),
            len(hits),
            len(dataset_reasoning),
        )

        client = DataGouvDatasetsClient()
        parsed_list: List[ParsedData] = []
        unstructured_blocks: List[Dict[str, str]] = []

        with tempfile.TemporaryDirectory(prefix="agora_technical_") as out_dir:
            for hit in hits[:MAX_DATASETS]:
                dataset_id = hit.get("dataset_id") or hit.get("id")
                if not dataset_id:
                    logger.debug("TechnicalAgent: skipping hit with no dataset_id: %s", hit)
                    continue
                try:
                    ds = client.get_dataset(dataset_id)
                    url_items = extract_resource_urls(ds)
                    logger.info(
                        "TechnicalAgent: dataset_id=%s resources_count=%d",
                        dataset_id,
                        len(url_items),
                    )
                except Exception as e:
                    logger.warning("TechnicalAgent: failed to fetch dataset %s: %s", dataset_id, e)
                    continue

                for res_idx, item in enumerate(url_items[:MAX_RESOURCES_PER_DATASET]):
                    url = item.get("url")
                    resource = item.get("resource")
                    if not url:
                        continue
                    resource_id = f"{dataset_id}_{res_idx}"
                    metadata_str = _resource_metadata_str(ds, resource, url)
                    try:
                        path = download_file(url, out_dir, resource=resource)
                    except Exception as e:
                        logger.warning("TechnicalAgent: download failed %s: %s", url[:80], e)
                        continue

                    if detect_structure(path, resource) == "unsuitable":
                        # Use same text extraction as General Agent so RLM can use this resource
                        try:
                            text = extract_text_from_file(
                                path,
                                max_rows=MAX_ROWS_UNSTRUCTURED,
                                resource=resource,
                            )
                            if isinstance(text, list):
                                for doc in text:
                                    content = (doc or "").strip()
                                    if content:
                                        if len(content) > MAX_CHARS_UNSTRUCTURED_PER_BLOCK:
                                            content = content[:MAX_CHARS_UNSTRUCTURED_PER_BLOCK] + "\n\n[Text truncated.]"
                                        unstructured_blocks.append({"metadata": metadata_str, "content": content})
                            else:
                                content = (text or "").strip()
                                if content:
                                    if len(content) > MAX_CHARS_UNSTRUCTURED_PER_BLOCK:
                                        content = content[:MAX_CHARS_UNSTRUCTURED_PER_BLOCK] + "\n\n[Text truncated.]"
                                    unstructured_blocks.append({"metadata": metadata_str, "content": content})
                            logger.info(
                                "TechnicalAgent: unsuitable resource %s -> extracted text (fallback)",
                                resource_id,
                            )
                        except Exception as e:
                            logger.warning(
                                "TechnicalAgent: text extraction failed for unsuitable %s: %s",
                                resource_id,
                                e,
                            )
                        continue

                    parsed = parse_into_records(
                        path,
                        resource=resource,
                        max_rows=MAX_ROWS_PER_RESOURCE,
                        resource_id=resource_id,
                        metadata=metadata_str,
                    )
                    if parsed and parsed.records:
                        parsed_list.append(parsed)
                        logger.info(
                            "TechnicalAgent: parsed resource %s -> %d rows, %d columns",
                            resource_id,
                            parsed.row_count,
                            len(parsed.columns),
                        )
                    elif parsed:
                        logger.debug("TechnicalAgent: parsed %s but no records", resource_id)

        if not parsed_list and not unstructured_blocks:
            evidence = (
                "Technical analysis requested but no structured data was found: "
                "resources were not tabular/record-based (e.g. CSV, JSON, XLSX, GeoJSON) or parsing failed."
            )
            logger.info("TechnicalAgent: no structured data, returning fallback evidence")
            return ExecutionResult(
                mode="technical",
                subquery=subquery,
                evidence=evidence,
            )

        technical_context = build_technical_context(
            parsed_list,
            preview_rows=TECHNICAL_PREVIEW_ROWS,
            unstructured_blocks=unstructured_blocks if unstructured_blocks else None,
        )
        if len(technical_context) > MAX_CONTEXT_CHARS:
            technical_context = (
                technical_context[:MAX_CONTEXT_CHARS]
                + "\n\n[Technical context truncated for length.]"
            )

        evidence = _explore_with_rlm(
            subquery,
            technical_context,
            focus=dataset_reasoning,
        )
        logger.info("TechnicalAgent: evidence len=%d", len(evidence))

        return ExecutionResult(
            mode="technical",
            subquery=subquery,
            evidence=evidence,
        )
