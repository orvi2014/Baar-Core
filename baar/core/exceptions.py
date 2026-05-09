"""Domain exceptions for BAAR orchestration."""


class BudgetExhausted(RuntimeError):
    """
    Raised by the kill-switch when remaining budget is too low for any call.
    Distinct from BudgetExceeded (per-call) — this means the session budget is
    permanently exhausted. Callers can catch this specifically without
    string-matching RuntimeError.
    """

    def __init__(self, message: str, remaining: float = 0.0):
        super().__init__(message)
        self.remaining = remaining


class TaskRejected(Exception):
    """
    Raised when value_fn estimates task value below the pre-flight cost estimate.
    The rejection is still recorded on the routing log (tier REJECT, $0 spend).
    """

    def __init__(
        self,
        message: str,
        *,
        estimated_value: float,
        estimated_cost_usd: float,
        task: str = "",
    ):
        super().__init__(message)
        self.estimated_value = estimated_value
        self.estimated_cost_usd = estimated_cost_usd
        self.task = task
