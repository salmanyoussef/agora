from app.models.execution_result import ExecutionResult


class TechnicalAgent:

    def run(
        self,
        subquery: str,
        hits: list[dict],
        dataset_reasoning: str = "",
    ) -> ExecutionResult:
        """Run technical analysis on the given dataset(s). dataset_reasoning is why
        the selector chose this dataset and what to search/compute (for future use)."""
        evidence = (
            "Technical analysis requested but computation engine "
            "not implemented yet."
        )

        return ExecutionResult(
            mode="technical",
            subquery=subquery,
            evidence=evidence
        )

#TO add later:
# dataset_downloader
# file_loader
# calculation_engine