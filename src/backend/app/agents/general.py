from __future__ import annotations

import logging
import tempfile
import time
from typing import Any, Dict, List

import dspy

from app.clients.data_gouv import DataGouvDatasetsClient, extract_resource_urls
from app.models.execution_result import ExecutionResult
from app.services.dspy_setup import configure_dspy, log_last_lm_call
from app.services.text_extraction import download_file, extract_text_from_file

logger = logging.getLogger(__name__)

# Context size knobs for RAG (general) agent — reduce to shrink context passed to the LM
MAX_DATASETS_FOR_RAG = 3
MAX_RESOURCES_PER_DATASET = 5
MAX_ROWS_PER_RESOURCE = 200
MAX_CONTEXT_CHARS = 0  # 0 = no truncation; set to e.g. 8000 to cap total context length


class AnswerFromContext(dspy.Signature):
    """Answer the user's question using only the provided context from open data resources."""

    context = dspy.InputField(desc="Relevant excerpts from dataset resources (CSV, JSON, text, etc.).")
    question = dspy.InputField(desc="The user's question to answer.")
    focus = dspy.InputField(
        desc="Why this dataset was chosen."
    )
    answer = dspy.OutputField(
        desc="A concise, summarized answer. If the context does not support an answer, say so."
    )


def _rag_answer(question: str, context: str, focus: str = "") -> str:
    configure_dspy()
    focus_text = focus.strip() if focus else "(No specific focus provided.)"
    logger.info("GeneralAgent RAG call started: question_len=%d context_len=%d focus_len=%d", len(question), len(context), len(focus_text))
    started_at = time.perf_counter()
    pred = dspy.Predict(AnswerFromContext)(context=context, question=question, focus=focus_text)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info("GeneralAgent RAG call completed in %.1f ms", elapsed_ms)
    try:
        usage = pred.get_lm_usage()
        if usage is not None:
            logger.debug("RAG Prediction.get_lm_usage(): %s", usage)
    except Exception as e:
        logger.debug("RAG get_lm_usage not available: %s", e)
    log_last_lm_call(caller="general_rag")
    logger.info("GeneralAgent DSPy response trace (last call):")
    dspy.inspect_history(n=1)
    return pred.answer or "No answer could be produced from the given context."


class GeneralAgent:
    def run(
        self,
        subquery: str,
        datasets: List[Dict[str, Any]],
        dataset_reasoning: str = "",
    ) -> ExecutionResult:
        logger.info(
            "GeneralAgent.run: subquery=%r datasets_count=%d reasoning_len=%d",
            subquery[:200] + ("..." if len(subquery) > 200 else ""),
            len(datasets),
            len(dataset_reasoning),
        )
        for d in datasets[:MAX_DATASETS_FOR_RAG]:
            logger.info(
                "GeneralAgent candidate dataset | id=%s | title=%s | tags=%s",
                d.get("dataset_id") or d.get("id"),
                d.get("title"),
                d.get("tags"),
            )
        client = DataGouvDatasetsClient()
        evidence_blocks: List[str] = []
        dataset_ids_to_try = [d.get("dataset_id") or d.get("id") for d in datasets[:MAX_DATASETS_FOR_RAG]]
        logger.debug("GeneralAgent: dataset_ids_to_try=%s", dataset_ids_to_try)

        with tempfile.TemporaryDirectory(prefix="agora_general_") as out_dir:
            logger.debug("GeneralAgent: download out_dir=%s", out_dir)
            for dataset in datasets[:MAX_DATASETS_FOR_RAG]:
                dataset_id = dataset.get("dataset_id") or dataset.get("id")
                if not dataset_id:
                    logger.debug("GeneralAgent: skipping dataset with no id: %s", dataset)
                    continue
                try:
                    ds = client.get_dataset(dataset_id)
                    url_items = extract_resource_urls(ds)
                    logger.info(
                        "GeneralAgent: dataset_id=%s resources_count=%d",
                        dataset_id,
                        len(url_items),
                    )
                except Exception as e:
                    logger.warning("Failed to fetch dataset %s: %s", dataset_id, e)
                    evidence_blocks.append(
                        f"[Dataset {dataset_id} fetch failed: {e}]"
                    )
                    continue

                for item in url_items[:MAX_RESOURCES_PER_DATASET]:
                    url = item.get("url")
                    resource = item.get("resource")
                    if not url:
                        continue
                    try:
                        path = download_file(url, out_dir)
                        logger.debug(
                            "GeneralAgent: downloaded dataset_id=%s url=%s -> path=%s",
                            dataset_id,
                            url[:80] + ("..." if len(url) > 80 else ""),
                            path,
                        )
                        text = extract_text_from_file(
                            path, max_rows=MAX_ROWS_PER_RESOURCE, resource=resource
                        )
                        if isinstance(text, list):
                            evidence_blocks.extend(text)
                            logger.debug(
                                "GeneralAgent: extracted list of %d docs from %s",
                                len(text),
                                path,
                            )
                        else:
                            evidence_blocks.append(
                                f"=== {path} ===\nSOURCE: {url}\n\n{text}"
                            )
                            logger.debug(
                                "GeneralAgent: extracted single doc len=%d from %s",
                                len(text),
                                path,
                            )
                    except Exception as e:
                        logger.warning("Download/extract failed for %s: %s", url, e)
                        evidence_blocks.append(
                            f"=== FILE FAILED ===\nURL: {url}\nERROR: {e}"
                        )

        context = "\n\n".join(evidence_blocks)
        if MAX_CONTEXT_CHARS and len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated for length.]"
        context_len = len(context)
        if not context.strip():
            context = "(No content could be extracted from the selected datasets.)"
        logger.info(
            "GeneralAgent: evidence_blocks=%d context_len=%d",
            len(evidence_blocks),
            context_len,
        )

        rag_answer = _rag_answer(subquery, context, focus=dataset_reasoning)
        logger.info(
            "GeneralAgent: RAG answer len=%d",
            len(rag_answer),
        )

        return ExecutionResult(
            mode="rag",
            subquery=subquery,
            evidence=rag_answer,
        )
