import json
import dspy

from app.services.dspy_setup import configure_dspy
from app.models.dataset_selection import DatasetSelection


class DatasetSelectorSignature(dspy.Signature):

    user_question = dspy.InputField()
    subquery = dspy.InputField()
    datasets = dspy.InputField(
        desc="List of datasets with title and description"
    )

    output_json = dspy.OutputField(
        desc="""
Return JSON:

{
 "selected_dataset_ids":[...],
 "reasoning":"..."
}
"""
    )


class DatasetSelectorAgent:

    def __init__(self):

        configure_dspy()
        self.module = dspy.ChainOfThought(DatasetSelectorSignature)

    def run(self, question: str, subquery: str, datasets: list):

        dataset_summary = []

        for d in datasets:

            dataset_summary.append(
                {
                    "dataset_id": d.get("dataset_id"),
                    "title": d.get("title"),
                    "description": d.get("description"),
                    "tags": d.get("tags"),
                }
            )

        result = self.module(
            user_question=question,
            subquery=subquery,
            datasets=json.dumps(dataset_summary),
        )

        try:
            parsed = json.loads(result.output_json)
        except Exception:
            parsed = {"selected_dataset_ids": [], "reasoning": ""}

        return DatasetSelection(**parsed)