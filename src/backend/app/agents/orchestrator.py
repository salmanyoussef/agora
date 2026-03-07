import logging

from app.agents.planner import PlannerAgent
from app.agents.general import GeneralAgent
from app.agents.technical import TechnicalAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.dataset_selector import DatasetSelectorAgent

from app.pipelines.retrieval import search_datasets

from app.models.agent_response import AgentResponse

logger = logging.getLogger(__name__)


def _stream_run(orchestrator: "AgentOrchestrator", question: str, k: int):
    """Generator that runs the orchestrator and yields SSE-style events in real time."""
    yield {"event": "status", "message": "Planning your question…"}
    plan = orchestrator.planner.run(question)
    yield {"event": "plan", "plan": plan.model_dump()}

    evidence_blocks = []
    all_hits = []
    user_messages = []

    for sub in plan.subqueries:
        yield {"event": "status", "message": f"Searching datasets for: {sub.question}"}
        hits = []
        query = sub.question.strip()
        if query:
            hits = search_datasets(query, k=k)
        unique_hits = {}
        for h in hits:
            dataset_id = h.get("dataset_id")
            if dataset_id:
                unique_hits[dataset_id] = h
        hits = list(unique_hits.values())
        all_hits.extend(hits)

        try:
            selection = orchestrator.selector.run(question, sub.question, hits)
            hits_by_id = {h.get("dataset_id"): h for h in hits if h.get("dataset_id")}

            if selection.selected_datasets:
                for sel in selection.selected_datasets:
                    hit = hits_by_id.get(sel.dataset_id)
                    if not hit:
                        continue
                    title = hit.get("title") or "Unknown dataset"
                    org = hit.get("organization") or "Unknown organization"
                    mode_label = "RAG" if sel.execution_mode == "rag" else "technical"
                    if sel.reasoning and sel.reasoning.strip():
                        msg = (
                            f"Checking dataset: «{title}» by {org} ({mode_label}). "
                            f"{sel.reasoning.strip()}"
                        )
                    else:
                        msg = (
                            f"Checking dataset: «{title}» by {org} ({mode_label}) "
                            f"for relevant information."
                        )
                    user_messages.append(msg)
                    yield {"event": "user_message", "message": msg}
                    if sel.execution_mode == "rag":
                        result = orchestrator.general_agent.run(
                            sub.question, [hit], dataset_reasoning=sel.reasoning or ""
                        )
                        evidence_blocks.append(result)
                    else:
                        result = orchestrator.technical_agent.run(
                            sub.question, [hit], dataset_reasoning=sel.reasoning or ""
                        )
                        evidence_blocks.append(result)

        except Exception as e:
            logger.warning("Dataset selector failed, using raw hits. Error: %s", e)
            for hit in hits:
                title = hit.get("title") or "Unknown dataset"
                org = hit.get("organization") or "Unknown organization"
                msg = (
                    f"Checking dataset: «{title}» by {org} (RAG) "
                    f"for relevant information."
                )
                user_messages.append(msg)
                yield {"event": "user_message", "message": msg}
                result = orchestrator.general_agent.run(
                    sub.question, [hit], dataset_reasoning=""
                )
                evidence_blocks.append(result)

    yield {"event": "status", "message": "Synthesizing answer…"}
    evidence_text = "\n\n".join(e.evidence for e in evidence_blocks)
    answer = orchestrator.synthesis.run(question, evidence_text)

    response = AgentResponse(
        answer=answer,
        plan=plan,
        evidence=evidence_blocks,
        hits=all_hits,
        user_messages=user_messages,
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

        logger.info("AgentOrchestrator starting for question: %s", question)

        plan = self.planner.run(question)

        evidence_blocks = []
        all_hits = []
        user_messages = []

        for sub in plan.subqueries:

            logger.info("Processing subquery: %s", sub.question)

            hits = []

            # Retrieve datasets using the current planned subquery only
            query = sub.question.strip()
            if query:
                logger.info("Retrieval query: %s", query)
                hits = search_datasets(query, k=k)

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
                        title = hit.get("title") or "Unknown dataset"
                        org = hit.get("organization") or "Unknown organization"
                        mode_label = "RAG" if sel.execution_mode == "rag" else "technical"
                        if sel.reasoning and sel.reasoning.strip():
                            msg = (
                                f"Checking dataset: «{title}» by {org} ({mode_label}) "
                                f"to find relevant information about: {sel.reasoning.strip()}"
                            )
                        else:
                            msg = (
                                f"Checking dataset: «{title}» by {org} ({mode_label}) "
                                f"for relevant information."
                            )
                        user_messages.append(msg)
                        logger.info("User message: %s", msg)
                        if sel.execution_mode == "rag":
                            result = self.general_agent.run(
                                sub.question,
                                [hit],
                                dataset_reasoning=sel.reasoning or "",
                            )
                            evidence_blocks.append(result)
                        else:
                            result = self.technical_agent.run(
                                sub.question,
                                [hit],
                                dataset_reasoning=sel.reasoning or "",
                            )
                            evidence_blocks.append(result)

            except Exception as e:

                logger.warning(
                    "Dataset selector failed, using raw hits. Error: %s",
                    e
                )
                for hit in hits:
                    title = hit.get("title") or "Unknown dataset"
                    org = hit.get("organization") or "Unknown organization"
                    msg = (
                        f"Checking dataset: «{title}» by {org} (RAG) "
                        f"for relevant information."
                    )
                    user_messages.append(msg)
                    logger.info("User message: %s", msg)
                    result = self.general_agent.run(
                        sub.question,
                        [hit],
                        dataset_reasoning="",
                    )
                    evidence_blocks.append(result)

        # Combine evidence
        evidence_text = "\n\n".join(
            e.evidence for e in evidence_blocks
        )

        logger.info("Running synthesis agent")

        answer = self.synthesis.run(question, evidence_text)

        return AgentResponse(
            answer=answer,
            plan=plan,
            evidence=evidence_blocks,
            hits=all_hits,
            user_messages=user_messages,
        )