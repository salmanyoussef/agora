import logging

import dspy

from app.services.dspy_setup import configure_dspy, log_last_lm_call

logger = logging.getLogger(__name__)


class SynthesisSignature(dspy.Signature):

    question = dspy.InputField()
    evidence = dspy.InputField()

    answer = dspy.OutputField()


class SynthesisAgent:

    def __init__(self):
        configure_dspy()
        self.module = dspy.ChainOfThought(SynthesisSignature)

    def run(self, question: str, evidence: str):
        logger.debug(
            "SynthesisAgent.run: question_len=%d evidence_len=%d",
            len(question),
            len(evidence),
        )
        result = self.module(
            question=question,
            evidence=evidence,
        )
        log_last_lm_call(caller="synthesis")
        return result.answer