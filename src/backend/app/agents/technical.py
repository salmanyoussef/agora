from app.models.execution_result import ExecutionResult


class TechnicalAgent:

    def run(self, subquery: str, hits: list[dict]) -> ExecutionResult:

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