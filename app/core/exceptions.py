class AgenticException(Exception):
    """Base for failures in agent logic (as opposed to infrastructure)."""

    error_code = "AGENT_ERROR"
    status_code = 422

    def __init__(
        self,
        message: str = "The agent failed to complete the request.",
        *,
        details: dict | list | str | None = None,
    ):
        self.message = message
        self.details = details
        super().__init__(message)


class MaxRecursionError(AgenticException):
    """The agent graph exceeded its recursion / step limit."""

    error_code = "MAX_RECURSION_REACHED"

    def __init__(self, details: dict | list | str | None = None):
        super().__init__(
            "Agent entered a loop or exceeded its reasoning-step budget.",
            details=details,
        )
