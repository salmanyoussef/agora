from app.models.execution_result import ExecutionResult


class GeneralAgent:

    def run(self, subquery: str, datasets: list):

        evidence_blocks = []

        for dataset in datasets[:3]:

            dataset_id = dataset["dataset_id"]

            dataset_details = fetch_dataset_details(dataset_id)

            resources = dataset_details["resources"]

            for r in resources[:2]:

                path = download_file(r["url"])

                text = extract_text_from_file(path)

                evidence_blocks.append(text)

        context = "\n\n".join(evidence_blocks)

        rag_answer = rag_llm(subquery, context)

        return ExecutionResult(
            mode="rag",
            subquery=subquery,
            evidence=rag_answer
        )