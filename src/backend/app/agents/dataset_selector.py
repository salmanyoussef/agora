import json
import logging
import time

import dspy

from app.models.dataset_selection import DatasetSelection, SelectedDataset
from app.services.dspy_setup import configure_dspy, log_last_lm_call

logger = logging.getLogger(__name__)

MAX_SELECTOR_ATTEMPTS = 3
# Truncate dataset description so the selector gets the gist without a wall of text
MAX_DESCRIPTION_CHARS = 500


def _truncate_description(desc: str, max_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    """Return first max_chars of description, stripping leading markdown and newlines."""
    if not desc or not isinstance(desc, str):
        return ""
    s = desc.strip()
    # Drop common leading markdown (e.g. **bold**)
    if s.startswith("**"):
        s = s[2:].strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


class DatasetSelectorSignature(dspy.Signature):

    """
    Select which of the given datasets are relevant to answer the user question for
    this subquery. You may keep the list of selected datasets Empty if none of the
    candidates would add real value (e.g. they are off-topic or too generic).

    For each selected dataset, assign execution_mode:

    - "rag": use ONLY when a random sample of the dataset resource is enough to
      provide context for answering the question. RAG means we will add a random
      sample of the dataset contents as context; choose rag only when that sample
      is sufficient (e.g. for general description, overview, or synthesis).

    - "technical": use when we need to search for something specific in the dataset
      resource (e.g. a particular record, value, or pattern) OR compute something
      over the data (e.g. largest value, mean, smallest value, count, sum, filter).
      If the question requires finding or calculating over the full data, choose
      technical.
    """

    user_question = dspy.InputField()
    subquery = dspy.InputField()
    datasets = dspy.InputField(
        desc="List of datasets with title, description, and organization"
    )

    output_json = dspy.OutputField(
        desc="""
Return JSON with format:

{
 "selected_datasets": [
   { "dataset_id": "<id>", "execution_mode": "rag", "reasoning": "Why this dataset and this mode." },
   { "dataset_id": "<id>", "execution_mode": "technical", "reasoning": "Why this dataset and this mode." }
 ]
}

- selected_datasets: list of objects (may be empty). Each object must have dataset_id, execution_mode, and reasoning.
- execution_mode:
  - "rag": only when a RANDOM SAMPLE of the dataset resource is enough context for the question.
  - "technical": when we need to SEARCH for something specific in the dataset OR COMPUTE values (e.g. largest, mean, smallest, count, sum).
- reasoning: short explanation for why this dataset and this execution_mode.
"""
    )


class DatasetSelectorAgent:

    def __init__(self):

        configure_dspy()
        self.module = dspy.ChainOfThought(DatasetSelectorSignature)

    def _parse_and_validate(self, raw_output: str) -> tuple[dict, bool]:
        """Parse JSON and validate structure. Returns (parsed, is_valid)."""
        try:
            parsed = json.loads(raw_output)
        except Exception as e:
            logger.warning("DatasetSelectorAgent JSON parse failed: %s", e)
            return {"selected_datasets": []}, False
        if not isinstance(parsed, dict):
            logger.warning("DatasetSelectorAgent response is not a JSON object: %s", type(parsed).__name__)
            return {"selected_datasets": []}, False
        raw_list = parsed.get("selected_datasets")
        if not isinstance(raw_list, list):
            logger.warning(
                "DatasetSelectorAgent selected_datasets missing or not a list (got %s)",
                type(raw_list).__name__ if raw_list is not None else "None",
            )
            return {"selected_datasets": []}, False
        return parsed, True

    def _build_selection(self, parsed: dict) -> DatasetSelection:
        """Build DatasetSelection from validated parsed dict."""
        raw_list = parsed.get("selected_datasets") or []
        selected_datasets = []
        for item in raw_list:
            if isinstance(item, dict):
                ds_id = item.get("dataset_id")
                mode = item.get("execution_mode", "rag")
                if mode not in ("rag", "technical"):
                    mode = "rag"
                reasoning = (item.get("reasoning") or "").strip()
                if ds_id is not None:
                    selected_datasets.append(
                        SelectedDataset(
                            dataset_id=str(ds_id),
                            execution_mode=mode,
                            reasoning=reasoning,
                        )
                    )
            elif isinstance(item, str):
                selected_datasets.append(
                    SelectedDataset(dataset_id=str(item), execution_mode="rag", reasoning="")
                )
        return DatasetSelection(selected_datasets=selected_datasets)

    def run(self, question: str, subquery: str, datasets: list):

        dataset_summary = []
        for d in datasets:
            raw_desc = d.get("description") or ""
            dataset_summary.append(
                {
                    "dataset_id": d.get("dataset_id"),
                    "title": d.get("title"),
                    "description": _truncate_description(raw_desc),
                    "organization": d.get("organization") or "",
                }
            )

        logger.info(
            "DatasetSelectorAgent.run started: question_len=%d subquery_len=%d datasets_count=%d",
            len(question),
            len(subquery),
            len(datasets),
        )
        datasets_json = json.dumps(dataset_summary)
        parsed = {"selected_datasets": []}
        last_result = None

        for attempt in range(1, MAX_SELECTOR_ATTEMPTS + 1):
            logger.info("DatasetSelectorAgent LLM call started (attempt %d/%d)", attempt, MAX_SELECTOR_ATTEMPTS)
            started_at = time.perf_counter()
            try:
                last_result = self.module(
                    user_question=question,
                    subquery=subquery,
                    datasets=datasets_json,
                )
            except Exception as e:
                logger.warning("DatasetSelectorAgent LLM call failed: %s", e)
                if attempt == MAX_SELECTOR_ATTEMPTS:
                    break
                continue
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info("DatasetSelectorAgent LLM call completed in %.1f ms", elapsed_ms)
            log_last_lm_call(caller="dataset_selector")
            logger.info("DatasetSelectorAgent DSPy response trace (last call):")
            dspy.inspect_history(n=1)

            parsed, valid = self._parse_and_validate(last_result.output_json or "{}")
            if valid:
                break
            if attempt < MAX_SELECTOR_ATTEMPTS:
                logger.warning("DatasetSelectorAgent invalid format, retrying (%d/%d)", attempt, MAX_SELECTOR_ATTEMPTS)

        selection_model = self._build_selection(parsed)

        selected_id_set = {s.dataset_id for s in selection_model.selected_datasets}
        summary_selected = [
            d for d in dataset_summary if str(d.get("dataset_id")) in selected_id_set
        ]
        missing_ids = [
            ds_id for ds_id in selected_id_set
            if str(ds_id) not in {str(d.get("dataset_id")) for d in summary_selected}
        ]

        logger.info(
            "DatasetSelector selected %d/%d datasets (per-dataset execution_mode)",
            len(selection_model.selected_datasets),
            len(datasets),
        )
        for sel in selection_model.selected_datasets:
            d = next(
                (x for x in dataset_summary if str(x.get("dataset_id")) == str(sel.dataset_id)),
                None,
            )
            title = d.get("title") if d else "?"
            reason_preview = (sel.reasoning or "")[:120] + ("..." if len(sel.reasoning or "") > 120 else "")
            logger.info(
                "Selected dataset | id=%s | title=%s | execution_mode=%s | reasoning=%s",
                sel.dataset_id,
                title,
                sel.execution_mode,
                reason_preview,
            )
        if missing_ids:
            logger.warning(
                "DatasetSelector returned ids not present in candidate hits: %s",
                missing_ids,
            )

        return selection_model