import logging
import time

import dspy

from app.services.dspy_setup import configure_dspy, log_last_lm_call, log_lm_usage

logger = logging.getLogger(__name__)


def _build_synthesis_context(
    intent: str,
    subquery_lines: list[str],
    dataset_refs: list[dict] | None = None,
) -> str:
    """Build a short context string for the synthesis agent (plan intent + subqueries + dataset URLs)."""
    parts = [f"User intent: {intent}."]
    if subquery_lines:
        parts.append("Subqueries that were run (evidence blocks follow this order):")
        for i, line in enumerate(subquery_lines, 1):
            parts.append(f"  {i}. {line}")
    if dataset_refs:
        parts.append("")
        parts.append("Dataset references:")
        for ref in dataset_refs:
            title = ref.get("title") or "Dataset"
            org = ref.get("organization") or ""
            url = ref.get("url") or ""
            if url:
                parts.append(f"  • {title}" + (f" ({org})" if org else "") + f": {url}")
    return "\n".join(parts)


class SynthesisSignature(dspy.Signature):
    """
    You are the final step of a question-answering pipeline over French open government data.

    You receive:
    - The user's original question.
    - A short summary of the search plan (intent and subqueries that were executed).
    - Evidence: one or more text blocks, each prefixed with [Source: ...] indicating which
      dataset the block came from. Blocks are in subquery order (RAG or technical extraction).

    Your job: Produce a single, clear, user-facing answer that synthesizes the evidence.
    - IMPORTANT: Always respond in the same language as the user's question (e.g. if the question is in French, answer in French; if in English, answer in English). Do not default to another language.
    - Tell the user that your answer is based on datasets from the French open government data portal (data.gouv.fr). Include one short sentence to that effect, in the same language as your answer (e.g. in French: "Cette réponse s'appuie sur des jeux de données issus des données ouvertes françaises (data.gouv.fr)." or in English: "This answer is based on datasets from the French open government data portal (data.gouv.fr).").
    - Note: The user is NOT the one who supplied to you the information based on the French open government datasets. The AGORA PIPELINE - created by me, the developer, and in which you are the last step - is the one who did the research and provided you with the relevant information.
    - Prefer one coherent answer; cite which dataset(s) support each part (use the source labels).
    - WHENEVER you source or cite any dataset, you MUST include the dataset URL together with its name (e.g. "[Dataset name](URL)" or "Dataset name: URL"). Use the dataset references provided in the context. Do not cite a dataset by name only—always include the URL.
    - If the evidence does not support a full answer, say so briefly and state what is missing.
    - Do not repeat raw evidence verbatim; synthesize and attribute to sources.
    - Do not offer to run new searches, fetch data, or perform actions; only summarize what was found and what was not covered.
    """

    question = dspy.InputField(
        desc="The user's original question. Your final answer must be written in this same language."
    )
    context = dspy.InputField(
        desc="Brief summary of the search plan (user intent, subqueries) and optionally dataset references with titles and URLs so you can cite them for users who want to do more research."
    )
    evidence = dspy.InputField(
        desc="Evidence from one or more datasets, each block prefixed with [Source: dataset title and context]. Concatenated in subquery order."
    )

    answer = dspy.OutputField(
        desc="One clear, synthesized answer for the user, in the same language as the question. Do not offer to perform searches or fetch data. Don't forget to tell the user that yout answer is based on datasets provided byfrom the French open government data portal (e.g. data.gouv.fr)."
    )


class SynthesisAgent:

    def __init__(self):
        configure_dspy()
        self.module = dspy.ChainOfThought(SynthesisSignature)

    def run(self, question: str, evidence: str, context: str = ""):
        logger.debug(
            "SynthesisAgent.run: question_len=%d context_len=%d evidence_len=%d",
            len(question),
            len(context),
            len(evidence),
        )
        logger.info("SynthesisAgent LLM call started")
        started_at = time.perf_counter()
        result = self.module(
            question=question,
            context=context or "No plan summary available.",
            evidence=evidence,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info("SynthesisAgent LLM call completed in %.1f ms", elapsed_ms)
        usage = None
        try:
            usage = result.get_lm_usage()
        except Exception:
            pass
        log_last_lm_call(caller="synthesis")
        logger.info("SynthesisAgent DSPy response trace (last call):")
        dspy.inspect_history(n=1)
        log_lm_usage("synthesis", usage)
        return result.answer, usage