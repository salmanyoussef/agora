import logging

from app.agents.planner import PlannerAgent
from app.agents.router import Router
from app.agents.general import GeneralAgent
from app.agents.technical import TechnicalAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.dataset_selector import DatasetSelectorAgent

from app.services.query_expander import expand_queries
from app.pipelines.retrieval import search_datasets

from app.models.agent_response import AgentResponse

logger = logging.getLogger(__name__)


class AgentOrchestrator:

    def __init__(self):

        self.planner = PlannerAgent()
        self.router = Router()

        self.general_agent = GeneralAgent()
        self.technical_agent = TechnicalAgent()

        self.selector = DatasetSelectorAgent()

        self.synthesis = SynthesisAgent()

    def run(self, question: str, k: int = 5):

        logger.info("AgentOrchestrator starting for question: %s", question)

        plan = self.planner.run(question)

        evidence_blocks = []
        all_hits = []

        for sub in plan.subqueries:

            logger.info("Processing subquery: %s", sub.question)

            queries = expand_queries(plan, sub.question)

            hits = []

            # Retrieve datasets
            for q in queries:

                logger.info("Retrieval query: %s", q)

                r = search_datasets(q, k=k)

                hits.extend(r)

            # Deduplicate hits by dataset_id
            unique_hits = {}
            for h in hits:
                dataset_id = h.get("dataset_id")
                if dataset_id:
                    unique_hits[dataset_id] = h

            hits = list(unique_hits.values())

            logger.info("Retrieved %d unique datasets", len(hits))

            all_hits.extend(hits)

            # --- Dataset selection agent ---
            try:

                selection = self.selector.run(
                    question,
                    sub.question,
                    hits
                )

                selected_hits = [
                    h for h in hits
                    if h.get("dataset_id") in selection.selected_dataset_ids
                ]

                logger.info(
                    "Dataset selector chose %d datasets",
                    len(selected_hits)
                )

            except Exception as e:

                logger.warning(
                    "Dataset selector failed, using raw hits. Error: %s",
                    e
                )

                selected_hits = hits

            if not selected_hits:
                logger.warning(
                    "No datasets selected for subquery: %s",
                    sub.question
                )

            # --- Routing ---
            mode = self.router.route(sub)

            logger.info("Execution mode: %s", mode)

            if mode == "technical":

                result = self.technical_agent.run(
                    sub.question,
                    selected_hits
                )

            else:

                result = self.general_agent.run(
                    sub.question,
                    selected_hits
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
            hits=all_hits
        )