import logging

import dspy

from app.services.dspy_setup import configure_dspy, log_last_lm_call

logger = logging.getLogger(__name__)


def _build_synthesis_context(intent: str, subquery_lines: list[str]) -> str:
    """Build a short context string for the synthesis agent (plan intent + subqueries)."""
    parts = [f"User intent: {intent}."]
    if subquery_lines:
        parts.append("Subqueries that were run (evidence blocks follow this order):")
        for i, line in enumerate(subquery_lines, 1):
            parts.append(f"  {i}. {line}")
    return "\n".join(parts)


class SynthesisSignature(dspy.Signature):
    """
    You are the final step of a question-answering pipeline over French open government data.

    You receive:
    - The user's original question.
    - A short summary of the search plan (intent and subqueries that were executed).
    - Evidence: one or more text blocks produced by earlier agents. Each block comes from
      a specific dataset (retrieved via hybrid search and then processed with RAG or
      technical extraction). Blocks are concatenated in the same order as the subqueries.

    Your job: Produce a single, clear, user-facing answer that synthesizes the evidence.
    - Answer in the same language as the user question (typically French).
    - Prefer one coherent answer; cite or summarize which data/sources support it.
    - If the evidence does not support a full answer, say so and state what is missing.
    - Do not repeat raw evidence verbatim; synthesize and attribute.
    """

    question = dspy.InputField(desc="The user's original question.")
    context = dspy.InputField(
        desc="Brief summary of the search plan: user intent and the subqueries that were run, so you know how the evidence is structured."
    )
    evidence = dspy.InputField(
        desc="Concatenated evidence from one or more datasets (RAG or technical extraction), in subquery order."
    )

    answer = dspy.OutputField(
        desc="One clear, synthesized answer for the user, in the same language as the question."
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
        result = self.module(
            question=question,
            context=context or "No plan summary available.",
            evidence=evidence,
        )
        log_last_lm_call(caller="synthesis")
        return result.answer