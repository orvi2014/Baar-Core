"""Domain exceptions for BAAR orchestration."""


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
