from app.models.plan import SubQuery


class Router:

    def route(self, subquery: SubQuery) -> str:

        if subquery.execution_mode == "technical":
            return "technical"

        return "general"