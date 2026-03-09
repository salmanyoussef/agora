import logging
from typing import List

from app.services.dspy_setup import (
    estimate_and_log_embedding_cost,
    estimate_and_log_pipeline_cost,
    format_embed_usage,
    format_lm_usage,
    merge_embed_usage,
    merge_lm_usage,
)
from app.agents.planner import PlannerAgent
from app.agents.general import GeneralAgent
from app.agents.technical import TechnicalAgent
from app.agents.synthesis import SynthesisAgent, _build_synthesis_context
from app.agents.dataset_selector import DatasetSelectorAgent

from app.pipelines.retrieval import search_datasets

from app.models.agent_response import AgentResponse
from app.models.dataset_selection import SelectedDataset

logger = logging.getLogger(__name__)

# Set to False to use the dataset selector's choice (general vs technical agent) per dataset.
# When True, every dataset is run with the general (RAG) agent only.
USE_ONLY_GENERAL_AGENT = False

# Public dataset page on data.gouv.fr (used when hit has no url)
DATA_GOUV_DATASET_PAGE_BASE = "https://www.data.gouv.fr/fr/datasets"


def _dataset_ref_from_hit(hit: dict) -> dict:
    """Build a reference dict (title, organization, url) for a dataset hit."""
    dataset_id = hit.get("dataset_id") or hit.get("id") or ""
    url = (hit.get("url") or "").strip()
    if not url and dataset_id:
        url = f"{DATA_GOUV_DATASET_PAGE_BASE}/{dataset_id}/"
    return {
        "title": hit.get("title") or "Unknown dataset",
        "organization": hit.get("organization") or "Unknown organization",
        "url": url,
    }


def _stream_run(
    orchestrator: "AgentOrchestrator",
    question: str,
    k: int,
    use_only_general_agent: bool | None = None,
):
    """Generator that runs the orchestrator and yields SSE-style events in real time.
    use_only_general_agent: if True, RAG only; if False, RAG + technical; if None, use USE_ONLY_GENERAL_AGENT.
    """
    use_rag_only = use_only_general_agent if use_only_general_agent is not None else USE_ONLY_GENERAL_AGENT
    yield {"event": "status", "message": "Planning your question…"}
    plan = orchestrator.planner.run(question)
    yield {"event": "plan", "plan": plan.model_dump()}
    usage_records_stream: List[dict] = [plan.lm_usage] if getattr(plan, "lm_usage", None) else []
    embed_usage_records_stream: List[dict | None] = []

    evidence_blocks = []
    all_hits = []
    user_messages = []
    used_dataset_refs: list[dict] = []
    seen_dataset_ids: set[str] = set()

    for sub in plan.subqueries:
        yield {"event": "status", "message": f"Searching datasets for: {sub.question}"}
        hits = []
        query = sub.question.strip()
        if query:
            hits, search_embed_usage = search_datasets(query, k=k)
            embed_usage_records_stream.append(search_embed_usage)
        unique_hits = {}
        for h in hits:
            dataset_id = h.get("dataset_id")
            if dataset_id:
                unique_hits[dataset_id] = h
        hits = list(unique_hits.values())
        all_hits.extend(hits)

        try:
            selection = orchestrator.selector.run(question, sub.question, hits)
            if getattr(selection, "lm_usage", None):
                usage_records_stream.append(selection.lm_usage)
            if use_rag_only and selection.selected_datasets:
                selection = type(selection)(
                    selected_datasets=[
                        SelectedDataset(
                            dataset_id=s.dataset_id,
                            execution_mode="rag",
                            reasoning=s.reasoning or "",
                        )
                        for s in selection.selected_datasets
                    ]
                )
            hits_by_id = {h.get("dataset_id"): h for h in hits if h.get("dataset_id")}

            if selection.selected_datasets:
                for sel in selection.selected_datasets:
                    hit = hits_by_id.get(sel.dataset_id)
                    if not hit:
                        continue
                    ref = _dataset_ref_from_hit(hit)
                    title, org, url = ref["title"], ref["organization"], ref.get("url") or ""
                    mode_label = "RAG" if sel.execution_mode == "rag" else "technical"
                    if sel.reasoning and sel.reasoning.strip():
                        msg = (
                            f"Checking dataset: «{title}» by {org} ({mode_label}) - "
                            f"{sel.reasoning.strip()}"
                        )
                    else:
                        msg = (
                            f"Checking dataset: «{title}» by {org} ({mode_label}) "
                            f"for relevant information."
                        )
                    if url:
                        msg = f"{msg} | {url}"
                    user_messages.append(msg)
                    yield {"event": "user_message", "message": msg}
                    did = hit.get("dataset_id")
                    if did and did not in seen_dataset_ids:
                        seen_dataset_ids.add(did)
                        used_dataset_refs.append(ref)
                    if use_rag_only:
                        result = orchestrator.general_agent.run(
                            sub.question, [hit], dataset_reasoning=sel.reasoning or ""
                        )
                    elif sel.execution_mode == "rag":
                        result = orchestrator.general_agent.run(
                            sub.question, [hit], dataset_reasoning=sel.reasoning or ""
                        )
                    else:
                        result = orchestrator.technical_agent.run(
                            sub.question, [hit], dataset_reasoning=sel.reasoning or ""
                        )
                    evidence_blocks.append(result)
                    if getattr(result, "lm_usage", None):
                        usage_records_stream.append(result.lm_usage)
                    if getattr(result, "embed_usage", None):
                        embed_usage_records_stream.append(result.embed_usage)

        except Exception as e:
            logger.warning("Dataset selector failed, using raw hits. Error: %s", e)
            for hit in hits:
                ref = _dataset_ref_from_hit(hit)
                title, org, url = ref["title"], ref["organization"], ref.get("url") or ""
                msg = (
                    f"Checking dataset: «{title}» by {org} (RAG) "
                    f"for relevant information."
                )
                if url:
                    msg = f"{msg} | {url}"
                user_messages.append(msg)
                yield {"event": "user_message", "message": msg}
                did = hit.get("dataset_id")
                if did and did not in seen_dataset_ids:
                    seen_dataset_ids.add(did)
                    used_dataset_refs.append(ref)
                result = orchestrator.general_agent.run(
                    sub.question, [hit], dataset_reasoning=""
                )
                evidence_blocks.append(result)
                if getattr(result, "lm_usage", None):
                    usage_records_stream.append(result.lm_usage)
                if getattr(result, "embed_usage", None):
                    embed_usage_records_stream.append(result.embed_usage)

    yield {"event": "status", "message": "Synthesizing answer…"}
    evidence_text = "\n\n".join(e.evidence for e in evidence_blocks)
    subquery_lines = [f"{s.question} — {s.purpose}" for s in plan.subqueries]
    synthesis_ctx = _build_synthesis_context(
        plan.intent, subquery_lines, dataset_refs=used_dataset_refs
    )
    answer, synthesis_usage = orchestrator.synthesis.run(question, evidence_text, context=synthesis_ctx)
    if synthesis_usage is not None:
        usage_records_stream.append(synthesis_usage)
    grand_total_stream = merge_lm_usage(usage_records_stream)
    logger.info("LLM usage [PIPELINE GRAND TOTAL] (stream): %s", format_lm_usage(grand_total_stream))
    estimate_and_log_pipeline_cost(grand_total_stream)

    embed_grand_total_stream = merge_embed_usage(embed_usage_records_stream)
    if embed_grand_total_stream.get("total_tokens"):
        logger.info("Embedding usage [PIPELINE GRAND TOTAL] (stream): %s", format_embed_usage(embed_grand_total_stream))
        estimate_and_log_embedding_cost(embed_grand_total_stream)

    response = AgentResponse(
        answer=answer,
        plan=plan,
        evidence=evidence_blocks,
        hits=all_hits,
        user_messages=user_messages,
        lm_usage_grand_total=grand_total_stream if grand_total_stream else None,
        embed_usage_grand_total=embed_grand_total_stream if embed_grand_total_stream.get("total_tokens") else None,
    )
    yield {"event": "done", "response": response.model_dump()}


class AgentOrchestrator:

    def __init__(self):

        self.planner = PlannerAgent()
        self.general_agent = GeneralAgent()
        self.technical_agent = TechnicalAgent()

        self.selector = DatasetSelectorAgent()

        self.synthesis = SynthesisAgent()

    def run(self, question: str, k: int = 5):
        """
        Run the full pipeline: planner → retrieval → dataset selector → general/technical per dataset → synthesis.
        Use USE_ONLY_GENERAL_AGENT in this module to force RAG-only mode.
        """
        logger.info(
            "AgentOrchestrator starting for question: %s (USE_ONLY_GENERAL_AGENT=%s)",
            question,
            USE_ONLY_GENERAL_AGENT,
        )

        plan = self.planner.run(question)
        usage_records: List[dict] = [plan.lm_usage] if getattr(plan, "lm_usage", None) else []
        embed_usage_records: List[dict | None] = []

        evidence_blocks = []
        all_hits = []
        user_messages = []
        used_dataset_refs: list[dict] = []
        seen_dataset_ids: set[str] = set()

        for sub in plan.subqueries:

            logger.info("Processing subquery: %s", sub.question)

            hits = []

            # Retrieve datasets using the current planned subquery only
            query = sub.question.strip()
            if query:
                logger.info("Retrieval query: %s", query)
                hits, search_embed_usage = search_datasets(query, k=k)
                embed_usage_records.append(search_embed_usage)
            else:
                hits = []

            # Deduplicate hits by dataset_id
            unique_hits = {}
            for h in hits:
                dataset_id = h.get("dataset_id")
                if dataset_id:
                    unique_hits[dataset_id] = h

            hits = list(unique_hits.values())

            logger.info("Retrieved %d unique datasets", len(hits))

            all_hits.extend(hits)

            # --- Dataset selection agent (per-dataset execution mode) ---
            try:

                selection = self.selector.run(
                    question,
                    sub.question,
                    hits
                )
                if getattr(selection, "lm_usage", None):
                    usage_records.append(selection.lm_usage)
                if USE_ONLY_GENERAL_AGENT and selection.selected_datasets:
                    selection = type(selection)(
                        selected_datasets=[
                            SelectedDataset(
                                dataset_id=s.dataset_id,
                                execution_mode="rag",
                                reasoning=s.reasoning or "",
                            )
                            for s in selection.selected_datasets
                        ]
                    )
                hits_by_id = {h.get("dataset_id"): h for h in hits if h.get("dataset_id")}

                if selection.selected_datasets:
                    logger.info(
                        "Dataset selector: %d selected (calling RAG/technical per dataset)",
                        len(selection.selected_datasets),
                    )
                    for sel in selection.selected_datasets:
                        hit = hits_by_id.get(sel.dataset_id)
                        if not hit:
                            logger.warning("Selected dataset id not in hits: %s", sel.dataset_id)
                            continue
                        ref = _dataset_ref_from_hit(hit)
                        title, org, url = ref["title"], ref["organization"], ref.get("url") or ""
                        mode_label = "RAG" if sel.execution_mode == "rag" else "technical"
                        if sel.reasoning and sel.reasoning.strip():
                            msg = (
                                f"Checking dataset: «{title}» by {org} ({mode_label}) - "
                                f"{sel.reasoning.strip()}"
                            )
                        else:
                            msg = (
                                f"Checking dataset: «{title}» by {org} ({mode_label}) "
                                f"for relevant information."
                            )
                        if url:
                            msg = f"{msg} | {url}"
                        user_messages.append(msg)
                        logger.info("User message: %s", msg)
                        did = hit.get("dataset_id")
                        if did and did not in seen_dataset_ids:
                            seen_dataset_ids.add(did)
                            used_dataset_refs.append(ref)
                        if USE_ONLY_GENERAL_AGENT:
                            result = self.general_agent.run(
                                sub.question,
                                [hit],
                                dataset_reasoning=sel.reasoning or "",
                            )
                        elif sel.execution_mode == "rag":
                            result = self.general_agent.run(
                                sub.question,
                                [hit],
                                dataset_reasoning=sel.reasoning or "",
                            )
                        else:
                            result = self.technical_agent.run(
                                sub.question,
                                [hit],
                                dataset_reasoning=sel.reasoning or "",
                            )
                        evidence_blocks.append(result)
                        if getattr(result, "lm_usage", None):
                            usage_records.append(result.lm_usage)
                        if getattr(result, "embed_usage", None):
                            embed_usage_records.append(result.embed_usage)

            except Exception as e:

                logger.warning(
                    "Dataset selector failed, using raw hits. Error: %s",
                    e
                )
                for hit in hits:
                    ref = _dataset_ref_from_hit(hit)
                    title, org, url = ref["title"], ref["organization"], ref.get("url") or ""
                    msg = (
                        f"Checking dataset: «{title}» by {org} (RAG) "
                        f"for relevant information."
                    )
                    if url:
                        msg = f"{msg} | {url}"
                    user_messages.append(msg)
                    logger.info("User message: %s", msg)
                    did = hit.get("dataset_id")
                    if did and did not in seen_dataset_ids:
                        seen_dataset_ids.add(did)
                        used_dataset_refs.append(ref)
                    result = self.general_agent.run(
                        sub.question,
                        [hit],
                        dataset_reasoning="",
                    )
                    evidence_blocks.append(result)
                    if getattr(result, "lm_usage", None):
                        usage_records.append(result.lm_usage)
                    if getattr(result, "embed_usage", None):
                        embed_usage_records.append(result.embed_usage)

        # Combine evidence
        evidence_text = "\n\n".join(
            e.evidence for e in evidence_blocks
        )
        subquery_lines = [f"{s.question} — {s.purpose}" for s in plan.subqueries]
        synthesis_ctx = _build_synthesis_context(
            plan.intent, subquery_lines, dataset_refs=used_dataset_refs
        )

        logger.info("Running synthesis agent")

        answer, synthesis_usage = self.synthesis.run(question, evidence_text, context=synthesis_ctx)
        if synthesis_usage is not None:
            usage_records.append(synthesis_usage)

        grand_total = merge_lm_usage(usage_records)
        logger.info("LLM usage [PIPELINE GRAND TOTAL]: %s", format_lm_usage(grand_total))
        estimate_and_log_pipeline_cost(grand_total)

        embed_grand_total = merge_embed_usage(embed_usage_records)
        if embed_grand_total.get("total_tokens"):
            logger.info("Embedding usage [PIPELINE GRAND TOTAL]: %s", format_embed_usage(embed_grand_total))
            estimate_and_log_embedding_cost(embed_grand_total)

        return AgentResponse(
            answer=answer,
            plan=plan,
            evidence=evidence_blocks,
            hits=all_hits,
            user_messages=user_messages,
            lm_usage_grand_total=grand_total if grand_total else None,
            embed_usage_grand_total=embed_grand_total if embed_grand_total.get("total_tokens") else None,
        )