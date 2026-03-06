import dspy

from app.services.dspy_setup import configure_dspy


class SynthesisSignature(dspy.Signature):

    question = dspy.InputField()
    evidence = dspy.InputField()

    answer = dspy.OutputField()


class SynthesisAgent:

    def __init__(self):
        configure_dspy()
        self.module = dspy.ChainOfThought(SynthesisSignature)

    def run(self, question: str, evidence: str):

        result = self.module(
            question=question,
            evidence=evidence
        )

        return result.answer