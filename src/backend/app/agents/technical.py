from app.models.execution_result import ExecutionResult


# TODO: Add DSPy logging like in GeneralAgent: configure_dspy(), log_last_lm_call(caller="technical_..."),
# timing (time.perf_counter()), pred.get_lm_usage(), and dspy.inspect_history(n=1) after any LM call.
# See app.agents.general (e.g. _answer_from_context and the surrounding logger.info/debug calls).


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